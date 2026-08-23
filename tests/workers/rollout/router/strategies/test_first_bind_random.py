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

"""Tests for P2 windowed random first-bind (``_first_bind_top`` + call sites).

Agentic first binds read small-integer session counts — ties are the norm.
The window widens "tied" from strict-equality to within-``window`` of the
min; inside the window a weighted random choice tilts toward emptier
replicas while keeping co-tied ones reachable.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from verl.workers.rollout.router.kvcaware.strategies.base import ReplicaInfo
from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import STICKY_TOP_SCORE, KVCacheAwareStrategy
from verl.workers.rollout.router.kvcaware.types import SlowCut

from .test_kvc_aware_strategy import PROMPT_IDS, FakeRouteDataProvider

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _strat(**kwargs) -> KVCacheAwareStrategy:
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        collector_names=["vllm_zmq"],
        weight=1.0,
        load_weights=(0.4, 0.2, 0.1, 0.3),
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


# ── _first_bind_top unit behavior ──────────────────────────────────────


class TestFirstBindTop:
    def test_window_zero_is_strict_min(self):
        strat = _strat(first_bind_window=0)
        assert strat._first_bind_top([5, 2, 3]) == 1

    def test_single_candidate_returns_it(self):
        strat = _strat(first_bind_window=1)
        assert strat._first_bind_top([0, 9, 8]) == 0  # only index 0 within window

    def test_tied_candidates_share_draws(self):
        """All-equal counts + uniform → every index wins a fair share."""
        strat = _strat(first_bind_window=1, first_bind_weighted=False)
        random.seed(0)
        winners = Counter(strat._first_bind_top([2, 2, 2]) for _ in range(600))
        assert set(winners) == {0, 1, 2}

    def test_weighted_tilts_toward_emptier(self):
        """weights = window+1-count: emptier replicas win more often."""
        strat = _strat(first_bind_window=1)  # weighted by default
        random.seed(0)
        winners = Counter(strat._first_bind_top([0, 1, 1]) for _ in range(600))
        assert winners[0] > winners[1]
        assert winners[1] > 0  # co-tied replicas stay reachable

    def test_window_excludes_far_replicas(self):
        """A replica beyond the window never wins."""
        strat = _strat(first_bind_window=1, first_bind_weighted=False)
        random.seed(0)
        winners = {strat._first_bind_top([0, 1, 5]) for _ in range(200)}
        assert winners <= {0, 1}

    def test_weighted_never_picks_zero_weight(self):
        """A replica at the window edge (weight 0 via count = lo+window+1) can't
        appear as a candidate at all — candidates are pre-filtered by window."""
        strat = _strat(first_bind_window=1)
        random.seed(0)
        for _ in range(100):
            assert strat._first_bind_top([3, 4]) in (0, 1)
            assert strat._first_bind_top([3, 5]) == 0


# ── least-inflight call site ───────────────────────────────────────────


class TestLeastInflightFirstBind:
    def test_routes_within_window(self):
        """Counts (1, 2): both within window 1 — either may win; never a third."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"active_sessions": 1}, "rep_b": {"active_sessions": 2}, "rep_c": {"active_sessions": 7}},
        )
        random.seed(1)
        winners = set()
        for _ in range(60):
            scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"))
            winners.add(scores.index(STICKY_TOP_SCORE))
        assert winners <= {0, 1}
        assert winners == {0, 1}  # both reachable across draws

    def test_strict_when_window_zero(self):
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT, first_bind_window=0)
        provider = FakeRouteDataProvider(
            {"rep_a": {"active_sessions": 4}, "rep_b": {"active_sessions": 2}},
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores[1] == STICKY_TOP_SCORE


# ── capacity cold-start call site ──────────────────────────────────────


class TestCapacityColdStartFirstBind:
    def _provider(self):
        return FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.0, "active_sessions": 0},
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.0, "active_sessions": 1},
                "rep_c": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.0, "active_sessions": 6},
            },
        )

    def test_cold_start_uses_windowed_pick(self):
        strat = _strat(
            slow_cut=SlowCut.CAPACITY_TOKEN_AWARE,
            first_bind_window=1,
            first_bind_weighted=False,
        )
        random.seed(2)
        winners = set()
        for _ in range(60):
            scores = strat.score(PROMPT_IDS, self._provider(), _replicas("rep_a", "rep_b", "rep_c"))
            winners.add(scores.index(STICKY_TOP_SCORE))
        assert winners <= {0, 1}  # rep_c (6) outside the window never wins
        assert winners == {0, 1}


# ── config validation ──────────────────────────────────────────────────


class TestFirstBindConfig:
    def test_config_defaults(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(collector_names=["x"], weight=1.0)
        assert cfg.first_bind_window == 1
        assert cfg.first_bind_weighted is True

    def test_config_rejects_negative_window(self):
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, first_bind_window=-1)

    def test_config_rejects_non_bool_weighted(self):
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, first_bind_weighted="yes")
