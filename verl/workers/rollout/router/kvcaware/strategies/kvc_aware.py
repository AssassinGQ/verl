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

"""KVCache-aware runtime strategy."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from ..config.strategy import KVCAwareStrategyConfig
from ..insight import emitter
from ..logging import get_router_logger
from ..types import Layer, MetricKey, OverloadMode, SlowCut
from ..utils.prefix_cache import resolve_prefix_hashes
from .registry import StrategyRegistry

if TYPE_CHECKING:
    from ..store import DataStore
    from .base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

STICKY_TOP_SCORE = 1e9

DEFAULT_LOAD_WEIGHTS: tuple[float, float, float, float] = (0.5, 0.0, 0.0, 0.5)


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[Layer, float],
        collector_names: list[str],
        weight: float,
        memory_overload_filter: bool = True,
        do_shortcut: bool = True,
        slow_cut: SlowCut | str = SlowCut.PREFIX_LOAD_AWARE,
        load_weights: tuple[float, float, float, float] = DEFAULT_LOAD_WEIGHTS,
        overload_mode: OverloadMode | str = OverloadMode.KV_LOAD,
        first_bind_window: int = 1,
        first_bind_weighted: bool = True,
        sticky_overload_threshold: float | None = None,
        capacity_reserve_threshold: float | None = None,
        skew_window: int = 60,
        skew_delta: int = 2,
        skew_factor: float = 2.0,
        tie_epsilon: float = 0.01,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_layers = {Layer.GPU, Layer.CPU, Layer.SSD}
        if set(layer_weights.keys()) != _valid_layers:
            raise StrategyError(f"layer_weights keys must be {_valid_layers}, got {set(layer_weights.keys())}")
        for layer_key, layer_weight in layer_weights.items():
            if layer_weight < 0:
                raise StrategyError(f"layer_weights[{layer_key}] must be >= 0, got {layer_weight}")
        weights_sum = sum(layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise StrategyError(f"layer_weights values must sum to 1.0, got {weights_sum}")
        if not isinstance(memory_overload_filter, bool):
            raise StrategyError(f"memory_overload_filter must be a bool, got {memory_overload_filter!r}")
        if not isinstance(do_shortcut, bool):
            raise StrategyError(f"do_shortcut must be a bool, got {do_shortcut!r}")
        try:
            slow_cut = SlowCut(slow_cut)
        except ValueError as exc:
            raise StrategyError(f"slow_cut must be one of {[m.value for m in SlowCut]}, got {slow_cut!r}") from exc
        try:
            overload_mode = OverloadMode(overload_mode)
        except ValueError as exc:
            raise StrategyError(
                f"overload_mode must be one of {[m.value for m in OverloadMode]}, got {overload_mode!r}"
            ) from exc
        if len(load_weights) != 4 or any(w < 0 for w in load_weights):
            raise StrategyError(f"load_weights must be 4 non-negative values, got {load_weights}")
        if abs(sum(load_weights) - 1.0) > 1e-6:
            raise StrategyError(f"load_weights must sum to 1.0, got {sum(load_weights)}")
        if not isinstance(first_bind_window, int) or first_bind_window < 0:
            raise StrategyError(f"first_bind_window must be a non-negative int, got {first_bind_window!r}")
        if not isinstance(first_bind_weighted, bool):
            raise StrategyError(f"first_bind_weighted must be a bool, got {first_bind_weighted!r}")
        for name, value in (
            ("sticky_overload_threshold", sticky_overload_threshold),
            ("capacity_reserve_threshold", capacity_reserve_threshold),
        ):
            if value is not None and not 0 < value < 1:
                raise StrategyError(f"{name} must be in (0, 1) or None, got {value}")
        if not isinstance(skew_window, int) or skew_window < 1:
            raise StrategyError(f"skew_window must be a positive int, got {skew_window!r}")
        if not isinstance(skew_delta, int) or skew_delta < 0:
            raise StrategyError(f"skew_delta must be a non-negative int, got {skew_delta!r}")
        if not isinstance(skew_factor, int | float) or skew_factor <= 1.0:
            raise StrategyError(f"skew_factor must be > 1.0, got {skew_factor!r}")
        if not isinstance(tie_epsilon, int | float) or tie_epsilon < 0:
            raise StrategyError(f"tie_epsilon must be a non-negative number, got {tie_epsilon!r}")

        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        self.memory_overload_filter = memory_overload_filter
        self.do_shortcut = do_shortcut
        self.slow_cut = slow_cut
        self.load_weights = tuple(load_weights)
        self.overload_mode = overload_mode
        self.first_bind_window = first_bind_window
        self.first_bind_weighted = first_bind_weighted
        # Resolved thresholds: explicit override or the legacy shared knob.
        self.sticky_overload_threshold = (
            float(sticky_overload_threshold) if sticky_overload_threshold is not None else float(load_threshold)
        )
        self.capacity_reserve_threshold = (
            float(capacity_reserve_threshold) if capacity_reserve_threshold is not None else float(load_threshold)
        )
        self.skew_window = skew_window
        self.skew_delta = skew_delta
        self.skew_factor = float(skew_factor)
        self.tie_epsilon = float(tie_epsilon)
        # SKEW streak counter per replica (see _is_skew_overloaded) — the only
        # mutable state in the strategy; the router actor is single-threaded so
        # no lock is needed.
        self._skew_streak: dict[str, int] = {}
        self._max_num_seqs: int | None = None
        self._max_num_batched_tokens: int | None = None
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, load_weights={self.load_weights}, "
            f"memory_overload_filter={self.memory_overload_filter}, do_shortcut={self.do_shortcut}, "
            f"slow_cut={self.slow_cut.value}, overload_mode={self.overload_mode.value}, "
            f"first_bind_window={self.first_bind_window}, first_bind_weighted={self.first_bind_weighted}, "
            f"sticky_overload_threshold={self.sticky_overload_threshold:.2f}, "
            f"capacity_reserve_threshold={self.capacity_reserve_threshold:.2f}, "
            f"skew=(window={self.skew_window}, delta={self.skew_delta}, factor={self.skew_factor}), "
            f"tie_epsilon={self.tie_epsilon}"
        )

    def set_capacity(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """Inject ``--max-num-seqs`` from the server handle's rollout config."""
        if not isinstance(max_num_seqs, int) or max_num_seqs <= 0:
            raise StrategyError(f"max_num_seqs must be a positive int, got {max_num_seqs}")
        if not isinstance(max_num_batched_tokens, int) or max_num_batched_tokens <= 0:
            raise StrategyError(f"max_num_batched_tokens must be a positive int, got {max_num_batched_tokens}")
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        logger.info(
            f"KVCacheAwareStrategy capacity set: max_num_seqs={max_num_seqs}"
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> KVCacheAwareStrategy:
        """Construct from config. ``max_num_seqs`` is injected by the Balancer
        via ``set_capacity`` after fetching from the server handle."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
            memory_overload_filter=cfg.memory_overload_filter,
            do_shortcut=cfg.do_shortcut,
            slow_cut=cfg.slow_cut,
            overload_mode=cfg.overload_mode,
            first_bind_window=cfg.first_bind_window,
            first_bind_weighted=cfg.first_bind_weighted,
            sticky_overload_threshold=cfg.sticky_overload_threshold,
            capacity_reserve_threshold=cfg.capacity_reserve_threshold,
            skew_window=cfg.skew_window,
            skew_delta=cfg.skew_delta,
            skew_factor=cfg.skew_factor,
            tie_epsilon=cfg.tie_epsilon,
        )

    def _near_top_pick(self, cap: float, rows: list[dict], pool: list[int], tag: str) -> int:
        """Pick the capacity-branch winner: near-top set + uniform random (P5).

        ``remaining`` is a continuous float, so exact-equality ties essentially
        never occur (exp2: 75 across 40,292 routes, all during the cold-start
        15s) — a strict argmax is deterministic in practice and its feedback
        (argmax → fills → lower remaining → argmax elsewhere) concentrates
        traffic. The top set is widened to every ``pool`` replica within
        ``cap × tie_epsilon`` of the best remaining; the winner is drawn
        uniformly from that set. epsilon=0 reduces to strict argmax.
        """
        if not pool:
            raise StrategyError("near-top pick requires a non-empty pool")
        best = max(rows[i]["remaining"] for i in pool)
        eps = cap * self.tie_epsilon
        if eps > 0:
            top_set = [i for i in pool if rows[i]["remaining"] >= best - eps]
        else:
            top_set = [max(pool, key=lambda i: rows[i]["remaining"])]
        pick = random.choice(top_set) if len(top_set) > 1 else top_set[0]
        logger.info(
            f"score(): CAPACITY_TOKEN_AWARE {tag} near-top set={len(top_set)} "
            f"(eps={eps:.0f}, best={best:.0f}) → winner idx={pick}"
        )
        return pick

    def _first_bind_top(self, counts: list[int]) -> int:
        """Pick the first-bind winner index from per-replica session counts.

        Candidate window = every replica within ``first_bind_window`` sessions
        of the minimum (0 = strict min, reproducing the pre-P2 behavior minus
        its iteration-order bias). Inside the window: weighted random — weight
        ``window + 1 − count`` tilts toward the emptier replicas while keeping
        co-tied ones reachable; ``first_bind_weighted=False`` flattens to
        uniform. Determinism note: the router actor is single-threaded, so a
        seeded ``random`` module state makes runs reproducible; unseeded runs
        randomize — the point of P2 (small-integer ties should not resolve by
        pool order).
        """
        lo = min(counts)
        window = self.first_bind_window
        if window <= 0:
            return min(range(len(counts)), key=lambda i: counts[i])
        candidates = [i for i, c in enumerate(counts) if c <= lo + window]
        if len(candidates) == 1:
            return candidates[0]
        if self.first_bind_weighted:
            weights = [lo + window + 1 - counts[i] for i in candidates]
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def _compute_load(
        self,
        kv_usage: float,
        running: int | float,
        waiting: int | float,
        inflight: int | float = 0,
    ) -> float:
        """load = a·kv_usage + b·running/max + c·waiting/max + d·inflight/max (∈ [0,1], bigger = more loaded).

        Weights ``(a, b, c, d) = self.load_weights``; ``max = self._max_num_seqs``.
        ``inflight`` (the Balancer's own acquire/release counter, maintained
        synchronously) is the only term non-zero at cold start — the other three
        come from async-polled vLLM metrics, still 0 before the first poll lands.
        Its weight ``d`` keeps the first wave of requests from collapsing onto
        ``pool[0]`` when the polled terms are tied at 0.
        """
        if self._max_num_seqs is None:
            raise StrategyError("set_capacity() must be called before routing")
        a, b, c, d = self.load_weights
        max_num_seqs = self._max_num_seqs
        return (
            a * float(kv_usage)
            + b * min(1.0, float(running) / max_num_seqs)
            + c * min(1.0, float(waiting) / max_num_seqs)
            + d * min(1.0, float(inflight) / max_num_seqs)
        )

    def is_overloaded(
        self,
        store: DataStore,
        replica: ReplicaInfo,
        replicas: list[ReplicaInfo] | None = None,
    ) -> bool:
        """Return True if ``replica`` is overloaded (mode-specific verdict).

        Used only by the sticky short-circuit to decide whether to send a
        returning session back to its bound replica. Combined scoring never
        consults overload. ``replicas`` (the candidate pool) is required by
        SKEW — its medians are pool-relative.
        """
        if self.overload_mode == OverloadMode.NONE:
            return False
        if self.overload_mode == OverloadMode.KV_CACHE_USAGE_PERC:
            kv_perc = store.get_metric(replica.replica_id, MetricKey.KV_CACHE_USAGE_PERC) or 0.0
            logger.info(f"is-overload replica={replica.replica_id} kv_perc={kv_perc:.4f}")
            return kv_perc > self.sticky_overload_threshold
        if self.overload_mode == OverloadMode.KV_LOAD:
            m = store.get_metrics(replica.replica_id)
            kv_usage = store.kv_cache_load(replica.replica_id)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            inflight = m.get(MetricKey.INFLIGHT_COUNT, 0)
            load = self._compute_load(kv_usage, running, waiting, inflight)
            # Emit the load the sticky check used (the bound replica).
            logger.info(f"is-overload replica={replica.replica_id} kv_load={load:.4f}")
            return load > self.sticky_overload_threshold
        if self.overload_mode == OverloadMode.SKEW:
            return self._is_skew_overloaded(store, replica, replicas)
        raise ValueError(
            f"There is no {self.overload_mode}, "
            "please set overload_mode in ['None', 'kv_cache_usage_perc', 'kv_load', 'skew']"
        )

    def _is_skew_overloaded(
        self,
        store: DataStore,
        replica: ReplicaInfo,
        replicas: list[ReplicaInfo] | None,
    ) -> bool:
        """SKEW verdict: sustained pool-relative skew (P4).

        Per call, snapshot the pool's (active_sessions, running,
        inflight_tokens) and record whether ``replica`` is beyond the skew
        lines; the verdict fires only when the replica has been skewed in ALL
        of the last ``skew_window`` snapshots. One clean sample resets the
        streak. Snapshots ride the sticky-shortcut cadence (sub-second under
        load), so window ≈ samples × request cadence — no timer thread.
        """
        if not replicas:
            return False

        def med(values: list[float]) -> float:
            ordered = sorted(values)
            mid = len(ordered) // 2
            return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0

        ids = [r.replica_id for r in replicas]
        sessions = [store.get_metric(rid, MetricKey.ACTIVE_SESSIONS) or 0 for rid in ids]
        running = [store.get_metric(rid, MetricKey.NUM_REQUESTS_RUNNING) or 0 for rid in ids]
        tokens = [store.get_metric(rid, MetricKey.INFLIGHT_TOKENS) or 0 for rid in ids]
        med_sessions = med(sessions)
        med_running = med(running)
        med_tokens = med(tokens)

        idx = ids.index(replica.replica_id)
        session_skew = sessions[idx] > med_sessions + self.skew_delta
        combo_skew = running[idx] > self.skew_factor * med_running and tokens[idx] > self.skew_factor * med_tokens
        skewed = session_skew or combo_skew

        streak = self._skew_streak.get(replica.replica_id, 0)
        streak = streak + 1 if skewed else 0
        self._skew_streak[replica.replica_id] = streak

        if streak >= self.skew_window:
            logger.info(
                f"is-overload replica={replica.replica_id} skew streak={streak}/{self.skew_window} "
                f"(sessions {sessions[idx]} vs med {med_sessions}, running {running[idx]} vs med {med_running}, "
                f"tokens {tokens[idx]} vs med {med_tokens}) → OVERLOADED"
            )
            return True
        logger.debug(
            f"is-overload replica={replica.replica_id} skew streak={streak}/{self.skew_window} "
            f"(sessions={sessions[idx]}/med={med_sessions} running={running[idx]}/med={med_running} "
            f"tokens={tokens[idx]}/med={med_tokens})"
        )
        return False

    def _sticky_shortcut(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None,
    ) -> list[float] | None:
        """Return a pre-built score list if a sticky replica should win, else None.

        Sticky wins when ``request_id`` is provided and the bound replica (from
        ``store.get_sticky_binding``) is present in ``replicas``. When
        ``memory_overload_filter`` is set the bound replica must also NOT be
        overloaded; otherwise the overload check is skipped. On win, returns a
        list with ``STICKY_TOP_SCORE`` at the bound index and ``0.0`` elsewhere;
        else ``None`` (fall through).
        """
        if not request_id:
            return None
        sticky_id = store.get_sticky_binding(request_id)
        if sticky_id is None:
            return None
        for idx, replica in enumerate(replicas):
            if replica.replica_id == sticky_id:
                if self.is_overloaded(store, replica, replicas):
                    logger.info(f"score(): STICKY replica={sticky_id} OVERLOADED → fallback")
                    return None
                logger.info(f"score(): STICKY replica={sticky_id} HIT → short-circuit (top score)")
                scores = [0.0] * len(replicas)
                scores[idx] = STICKY_TOP_SCORE
                return scores
        logger.info(f"score(): sticky replica={sticky_id} not in pool → fallback")
        return None

    def score(
        self,
        prompt_ids: list[int] | None,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None = None,
    ) -> list[float]:
        """Score each replica. Larger is better.

        After the sticky short-circuit misses, the ``slow_cut`` mode selects the
        fallback scoring: ``prefix-load-aware`` → ``S = α·S_cache + (1-α)·S_load``;
        ``least-inflight`` → ``-INFLIGHT_COUNT`` (verl GlobalRequestLoadBalancer-style).
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []
        t0 = time.perf_counter()
        try:
            # Sticky short-circuit.
            shortcut: list[float] | None = None
            if self.do_shortcut:
                shortcut = self._sticky_shortcut(store, replicas, request_id)
                if shortcut is not None:
                    return shortcut
            if self.slow_cut == SlowCut.LEAST_INFLIGHT:
                # First-bind aware: sessions already bound to a replica stay
                # lifted in ACTIVE_SESSIONS through their tool/sandbox phases
                # (where INFLIGHT_COUNT momentarily reads 0), so the first
                # request of a NEW session lands on the replica with the
                # fewest live sessions instead of on whichever looks idle
                # mid-tool-call. Small-integer ties are the norm here — resolve
                # them by windowed random (P2), not pool iteration order.
                counts = [store.get_metric(r.replica_id, MetricKey.ACTIVE_SESSIONS) or 0 for r in replicas]
                top = self._first_bind_top(counts)
                scores = [0.0] * len(replicas)
                scores[top] = STICKY_TOP_SCORE
                return scores
            # Hash-resolving slow_cuts share one resolution across all replicas.
            gpu_hash_strs = resolve_prefix_hashes(prompt_ids or [], request_id, store)
            if self.slow_cut == SlowCut.PREFIX_LOAD_AWARE:
                return self._prefix_load_aware(store, replicas, gpu_hash_strs)
            if self.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE:
                return self._capacity_token_scores(store, replicas, request_id, prompt_ids or [], gpu_hash_strs)
            raise ValueError(f"Unknow slowcut type {self.slow_cut}")
        finally:
            emitter.on_route(time.perf_counter() - t0)

    def _prefix_load_aware(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        gpu_hash_strs: list[str],
    ) -> list[float]:
        """Prefix-cache + blended-load combined score (``slow_cut=prefix-load-aware``).

            S = α·S_cache + (1-α)·S_load

        ``S_cache`` is the three-layer weighted prefix-hit score;
        ``S_load = 1 - load`` where ``load`` blends kv_usage /
        running / waiting / inflight via ``_compute_load``. Returns one score
        per replica.
        """
        result: list[float] = []
        loads: dict[str, float] = {}
        for replica in replicas:
            m = store.get_metrics(replica.replica_id)
            kv_usage = store.kv_cache_load(replica.replica_id)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            inflight = m.get(MetricKey.INFLIGHT_COUNT, 0)
            load = self._compute_load(kv_usage, running, waiting, inflight)
            loads[replica.replica_id] = load
            s_load = 1.0 - load
            s_cache, gpu_hit = self._cache_score(store, replica, gpu_hash_strs)
            if emitter.enabled():
                emitter.on_score(replica.replica_id, {"load": load, "s_cache": s_cache})
            score = self.alpha * s_cache + (1 - self.alpha) * s_load
            result.append(score)
            logger.info(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} "
                f"inflight={inflight} → load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f} "
                f"s_cache={s_cache:.4f} ({self.alpha:.2f}·cache + {1 - self.alpha:.2f}·load) → score={score:.4f}"
            )
        scores_str = ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas))
        logger.info(f"score(): COMBINED scores: {scores_str}")
        # Per-replica load that drove this combined-score dispatch (all replicas,
        # reused from the loop above — no extra computation). The plot parses this
        # into a per-replica load panel; sticky-win / least-inflight dispatches do
        # not reach here, so they emit no line (panel omits those dispatches).
        logger.info(f"route-load loads={loads}")
        return result

    def _cache_score(
        self,
        store: DataStore,
        replica: ReplicaInfo,
        hash_strs: list[str],
    ) -> tuple[float, float]:
        """Three-layer weighted prefix-cache hit score ∈ [0, 1].

            S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit

        ``hash_strs`` is the caller-computed prefix-hash chain (shared across
        replicas, per-request cached) so the per-replica cost is just the index
        walk. CPU/SSD return 0.0 today (see ``KVCacheStore.get_layer_prefix_hit_rate``).
        Returns ``(s_cache, gpu_hit)`` so the caller logs gpu_hit without re-querying.
        """
        gpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, hash_strs, Layer.GPU)
        cpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, hash_strs, Layer.CPU)
        ssd_hit = store.get_layer_prefix_hit_rate(replica.replica_id, hash_strs, Layer.SSD)
        w = self.layer_weights
        s_cache = w[Layer.GPU] * gpu_hit + w[Layer.CPU] * cpu_hit + w[Layer.SSD] * ssd_hit
        return s_cache, gpu_hit

    # ── Capacity-gated token routing (CAPACITY_TOKEN_AWARE) ───────────

    def _total_token_capacity(self, store: DataStore) -> int:
        """Per-replica KV-cache token capacity = ``num_gpu_blocks × block_size``.

        ``num_gpu_blocks`` is a per-replica gauge (constant across replicas in
        practice); ``block_size`` is learned from the first KV event (defaults
        to 16 if not yet seen). Returns 0 when unavailable, in which case the
        caller falls back to least-inflight.
        """
        for node_id in store.get_metric_node_ids():
            nblk = store.get_metric(node_id, MetricKey.NUM_GPU_BLOCKS)
            if nblk and nblk > 0:
                block_size = store.get_block_size() or 16
                return int(nblk) * int(block_size)
        return 0

    def _capacity_token_scores(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None,
        prompt_ids: list[int],
        gpu_hash_strs: list[str],
    ) -> list[float]:
        """Capacity-gated token routing (discrete: winner=STICKY_TOP_SCORE, rest 0).

        For each replica ``i``::

            avail[i]     = cap × (1 - kv_cache_usage_perc[i])   # free tokens (no cache)
            need[i]      = len(prompt_ids) × (1 - gpu_hit[i])    # prefill this req adds
            remaining[i] = avail[i] - need[i]                    # free tokens after assign
            eligible[i]  = avail[i] >= cap × (1 - capacity_reserve_threshold)   # pure capacity gate

        pick ``argmin(active_sessions)`` with a tie window — the session-count
        gauge stays lifted through tool/sandbox phases, so the first wave
        spreads by true session load instead of by whatever looks momentarily
        idle; co-tied replicas (within ``first_bind_window`` of the min) are
        resolved by weighted random rather than pool order (see
        :meth:`_first_bind_top`).
        Otherwise pick ``argmax(eligible, remaining)`` widened by
        ``tie_epsilon``: every eligible replica within ``cap × epsilon`` of
        the best remaining joins the top set, drawn uniformly at random (see
        :meth:`_near_top_pick`); the all-overloaded fallback applies the same
        treatment to the full pool.
        """
        n = len(replicas)
        cap = self._total_token_capacity(store)
        plen = len(prompt_ids) if prompt_ids else 0
        rows: list[dict] = []
        for replica in replicas:
            kv_perc = store.get_metric(replica.replica_id, MetricKey.KV_CACHE_USAGE_PERC) or 0.0
            inflight = store.get_metric(replica.replica_id, MetricKey.INFLIGHT_COUNT) or 0
            inflight_tokens = store.get_metric(replica.replica_id, MetricKey.INFLIGHT_TOKENS) or 0
            active_sessions = store.get_metric(replica.replica_id, MetricKey.ACTIVE_SESSIONS) or 0
            s_cache, gpu_hit = self._cache_score(store, replica, gpu_hash_strs)
            avail = cap * (1.0 - kv_perc)
            need = plen * (1.0 - gpu_hit)
            remaining = avail - need
            if emitter.enabled():
                emitter.on_score(replica.replica_id, {"avail": avail, "need": need, "remaining": remaining})
            rows.append(
                {
                    "replica": replica,
                    "kv_perc": kv_perc,
                    "inflight": inflight,
                    "inflight_tokens": inflight_tokens,
                    "active_sessions": active_sessions,
                    "gpu_hit": gpu_hit,
                    "s_cache": s_cache,
                    "avail": avail,
                    "need": need,
                    "remaining": remaining,
                }
            )

        thresh = cap * (1.0 - self.capacity_reserve_threshold)
        cold_start = store.get_sticky_binding(request_id) is None
        if cold_start:
            top = self._first_bind_top([rows[i]["active_sessions"] for i in range(n)])
            logger.info(
                "score(): CAPACITY_TOKEN_AWARE cold start → windowed first-bind "
                f"(window={self.first_bind_window}, weighted={self.first_bind_weighted}) "
                f"active_sessions={[rows[i]['active_sessions'] for i in range(n)]}"
            )
        else:
            eligible = [i for i in range(n) if rows[i]["avail"] >= thresh]
            if not eligible:
                top = self._near_top_pick(cap, rows, list(range(n)), tag="no-eligible")
            else:
                top = self._near_top_pick(cap, rows, eligible, tag="eligible")

        for i, row in enumerate(rows):
            tag = " ← WINNER" if i == top else ""
            logger.info(
                f"score(): replica={row['replica'].replica_id} kv_perc={row['kv_perc']:.3f} "
                f"gpu_hit={row['gpu_hit']:.3f} inflight={row['inflight']} "
                f"avail={row['avail']:.0f} need={row['need']:.0f} "
                f"max_num_batched_tokens={self._max_num_batched_tokens} inflight_tokens={row['inflight_tokens']:} "
                f"active_sessions={row['active_sessions']} "
                f"remaining={row['remaining']:.0f}{tag}"
            )
        winner = rows[top]["replica"].replica_id
        logger.info(
            f"score(): CAPACITY_TOKEN_AWARE winner={winner} "
            f"(kv_perc={rows[top]['kv_perc']:.3f}, remaining={rows[top]['remaining']:.0f})"
        )
        # Per-replica capacity signal for the plot (mirrors route-load in prefix-load-aware).
        cap_loads = {row["replica"].replica_id: row["remaining"] for row in rows}
        logger.info(f"route-capacity remaining={cap_loads}")
        scores = [0.0] * n
        scores[top] = STICKY_TOP_SCORE
        return scores


StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
