# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""route() — weighted replica ranking for the KVCAware router.

The Balancer delegates each request to ``route(strategies, prompt_ids,
provider, replicas)`` and maps ``ranking[0]`` back to a server handle.
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol, runtime_checkable

from ..logging import get_router_logger

logger = get_router_logger("routing")


@runtime_checkable
class RoutingStrategy(Protocol):
    """Routing scoring strategy.

    Each strategy scores a batch of replicas independently and returns a list
    of the same length and order. ``route()`` weighted-sums the outputs.
    """

    def score(
        self,
        prompt_ids: list[int] | None,
        store: Any,
        replicas: list[Any],
        request_id: str | None = None,
        gpu_hash_strs: list[str] | None = None,
    ) -> list[float]:
        """Score each replica. Larger is better; negatives are allowed.

        ``request_id`` enables sticky-session short-circuit: a strategy reads
        the bound replica from ``store.get_sticky_binding(request_id)`` and may
        return a pre-built score list that places it first when it is not
        overloaded (see ``KVCacheAwareStrategy``). Strategies that ignore
        stickiness accept ``request_id`` and proceed with their own scoring.

        ``gpu_hash_strs`` is the caller-computed prefix-hash chain (shared
        across all replicas, computed once in ``route()`` from
        ``prompt_ids`` + ``request_id``). Strategies that need it
        (``prefix-load-aware``, ``capacity-token-aware``) consume it directly
        instead of re-resolving; ``None`` means the caller did not pre-resolve
        and the strategy may compute it itself.
        """
        ...


def _rank_key(score: float) -> float:
    """Sort key treating non-finite scores (NaN/inf) as worst."""
    return score if math.isfinite(score) else float("-inf")


def route(
    strategies: list[tuple[Any, float]],
    prompt_ids: list[int] | None,
    store: Any,
    replicas: list[Any],
    request_id: str | None = None,
    gpu_hash_strs: list[str] | None = None,
) -> list[str]:
    """Return replica ids ranked best-first.

    Falls back to a random shuffle of replica ids if any strategy raises or
    returns a wrong-length score list — routing remains available even when
    metrics are temporarily unavailable.

    Args:
        strategies: ``[(strategy, weight), ...]`` — weighted strategies.
        prompt_ids: prompt token ids (content-aware routing; may be ``None``).
        store: ``DataStore`` for metric + sticky-session queries.
        replicas: ``[ReplicaInfo, ...]`` — candidate replicas.
        request_id: session id for sticky-session routing (may be ``None``).
        gpu_hash_strs: caller-computed prefix-hash chain for ``prompt_ids``
            (shared across replicas). Computed once by the caller
            (``acquire_server``) and threaded through here so strategies and the
            post-route dispatch-recorder share one resolution. ``None`` →
            resolve inside the strategy (backward-compat).

    Returns:
        Replica ids sorted by total score, best first. Falls back to random
        order on scoring failure.

    Raises:
        RuntimeError: ``replicas`` is empty.
    """
    n = len(replicas)
    if n == 0:
        raise RuntimeError("no available replicas")

    final = [0.0] * n
    for strategy, weight in strategies:
        name = type(strategy).__name__
        try:
            scores = strategy.score(
                prompt_ids,
                store,
                replicas,
                request_id,
                gpu_hash_strs,
            )
            if len(scores) != n:
                raise ValueError(f"{name}.score() returned {len(scores)} scores, expected {n}")
        except Exception as exc:  # noqa: BLE001
            ids = [r.replica_id for r in replicas]
            random.shuffle(ids)
            logger.warning(
                f"route(): {name} failed ({type(exc).__name__}: {exc}), falling back to random order",
            )
            return ids
        for idx in range(n):
            final[idx] += weight * scores[idx]

    # Rank best-first. Among replicas tied at the top score, pick one at random
    # (instead of the stable sort's always-first bias) so a true cold start or
    # a same-prompt rollout wave spreads across the pool rather than collapsing
    # onto pool[0]. Lower ranks keep their stable order — only the winner set
    # is randomized.
    ranking = sorted(range(n), key=lambda idx: _rank_key(final[idx]), reverse=True)
    top_score = _rank_key(final[ranking[0]])
    top_ties = [idx for idx in ranking if _rank_key(final[idx]) == top_score]
    if len(top_ties) > 1:
        chosen = random.choice(top_ties)
        # Move the random pick to the head; keep the rest in stable order.
        ranking = [chosen] + [idx for idx in ranking if idx != chosen]
    scores_str = ", ".join(f"{replicas[idx].replica_id}={final[idx]:.4f}" for idx in ranking)
    logger.info(f"route(): replicas={n} ranking=[{scores_str}]")
    return [replicas[idx].replica_id for idx in ranking]
