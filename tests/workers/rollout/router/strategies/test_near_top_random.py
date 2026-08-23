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

"""Tests for P5 near-top random in the capacity branch.

``remaining`` is continuous — strict-equality ties never fire (exp2: 75 in
40,292 routes, all cold-start), so routing was a deterministic argmax whose
feedback concentrates traffic. The top set widens to within cap×epsilon of
the best, drawn uniformly.
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
        slow_cut=SlowCut.CAPACITY_TOKEN_AWARE,
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


def _winner(strat, provider, reps) -> int:
    """Score with a stale binding (forces the eligible branch) → winner idx."""
    scores = strat.score(PROMPT_IDS, provider, reps, "r1")
    return scores.index(STICKY_TOP_SCORE)


class TestNearTopPick:
    def test_near_tied_replicas_share_draws(self):
        """avail 1440 vs 1424 (gap 16 = cap 1600 × 0.01): both in the top set."""
        strat = _strat(tie_epsilon=0.01)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.10},  # avail 1440
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.11},  # avail 1424
                "rep_c": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.5},  # avail 800 — out
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b", "rep_c")
        random.seed(3)
        winners = Counter(_winner(strat, provider, reps) for _ in range(80))
        assert set(winners) == {0, 1}
        assert winners[2] == 0 if 2 in winners else True

    def test_far_replica_excluded(self):
        """gap beyond eps → deterministic argmax even across draws."""
        strat = _strat(tie_epsilon=0.01)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.1},  # avail 1440
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.4},  # avail 960 — out
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b")
        random.seed(4)
        winners = {_winner(strat, provider, reps) for _ in range(50)}
        assert winners == {0}

    def test_epsilon_zero_is_strict_argmax(self):
        strat = _strat(tie_epsilon=0.0)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.10},
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.11},  # gap 16 > 0
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b")
        random.seed(5)
        winners = {_winner(strat, provider, reps) for _ in range(50)}
        assert winners == {0}

    def test_ineligible_replica_never_enters_top_set(self):
        """Replica within eps of best remaining but below the reserve gate
        stays excluded — the gate is a hard filter, epsilon only widens ties."""
        strat = _strat(tie_epsilon=0.01, capacity_reserve_threshold=0.85)  # gate: avail >= 240
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.849},  # avail 241.6 — eligible
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.851},  # avail 238.4 — gated out
                "rep_c": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.86},  # avail 224 — gated out
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b", "rep_c")
        random.seed(6)
        winners = {_winner(strat, provider, reps) for _ in range(50)}
        assert winners == {0}

    def test_all_overloaded_fallback_uses_near_top(self):
        """No replica clears the gate → near-top over the FULL pool."""
        strat = _strat(tie_epsilon=0.01, capacity_reserve_threshold=0.1)  # gate: avail >= 1440
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.30},  # avail 1120
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.31},  # avail 1104 (within 16)
                "rep_c": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.8},  # avail 320 — out
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b", "rep_c")
        random.seed(7)
        winners = set(_winner(strat, provider, reps) for _ in range(80))
        assert winners == {0, 1}

    def test_single_candidate_deterministic(self):
        strat = _strat(tie_epsilon=0.01)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.1},
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.9},  # gated (avail 160 < 240? no—0.9×)
            },
            sticky={"r1": "rep_gone"},
        )
        reps = _replicas("rep_a", "rep_b")
        for _ in range(10):
            assert _winner(strat, provider, reps) == 0


class TestTieEpsilonConfig:
    def test_config_roundtrip(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, tie_epsilon=0.02)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.tie_epsilon == 0.02

    def test_config_rejects_negative(self):
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, tie_epsilon=-0.01)
