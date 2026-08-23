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

"""Tests for P3 threshold semantics split.

load_threshold historically fed two unrelated decisions — the sticky
overload trigger and the capacity reserve gate. exp1 showed the conflict:
at 0.9 the overload channel never fires (peak KV usage 0.22) while the
reserve semantics want ~0.9. The two knobs are now independent, with
None → load_threshold preserving exact legacy behavior.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.strategies.base import ReplicaInfo
from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import STICKY_TOP_SCORE, KVCacheAwareStrategy
from verl.workers.rollout.router.kvcaware.types import OverloadMode, SlowCut

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


class TestThresholdResolution:
    def test_none_falls_back_to_load_threshold(self):
        strat = _strat(load_threshold=0.6)
        assert strat.sticky_overload_threshold == 0.6
        assert strat.capacity_reserve_threshold == 0.6

    def test_explicit_overrides_win(self):
        strat = _strat(
            load_threshold=0.9,
            sticky_overload_threshold=0.3,
            capacity_reserve_threshold=0.8,
        )
        assert strat.sticky_overload_threshold == 0.3
        assert strat.capacity_reserve_threshold == 0.8

    def test_rejects_out_of_range(self):
        from verl.workers.rollout.router.kvcaware.strategies.kvc_aware import StrategyError

        with pytest.raises(StrategyError):
            _strat(sticky_overload_threshold=1.5)
        with pytest.raises(StrategyError):
            _strat(capacity_reserve_threshold=0.0)


class TestStickyOverloadChannel:
    """kv_cache_usage_perc overload mode reads sticky_overload_threshold."""

    def _overloaded(self, strat, kv_perc: float, sticky: str) -> bool:
        provider = FakeRouteDataProvider({"rep_a": {"kv_cache_usage_perc": kv_perc}}, sticky={"r1": sticky})
        # The overload verdict shows up as: sticky hit (not overloaded) vs fallback.
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        return scores[0] != STICKY_TOP_SCORE

    def test_low_threshold_unbinds_at_moderate_kv(self):
        """The dead channel revived: kv=0.4 > 0.3 → sticky falls back."""
        strat = _strat(
            overload_mode=OverloadMode.KV_CACHE_USAGE_PERC,
            load_threshold=0.9,
            sticky_overload_threshold=0.3,
        )
        assert self._overloaded(strat, 0.4, "rep_a") is True

    def test_default_threshold_still_tolerates_moderate_kv(self):
        """None → 0.9 legacy: kv=0.4 stays bound (exp1 behavior preserved)."""
        strat = _strat(overload_mode=OverloadMode.KV_CACHE_USAGE_PERC, load_threshold=0.9)
        assert self._overloaded(strat, 0.4, "rep_a") is False


class TestCapacityReserveGate:
    """Capacity eligibility reads capacity_reserve_threshold, independently."""

    def _scores(self, strat, rep_a_kv: float, rep_b_kv: float):
        provider = FakeRouteDataProvider(
            {
                # cap=1600. rep_a avail = 1600·(1-kv); rep_b always ample.
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": rep_a_kv},
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": rep_b_kv},
            },
            sticky={"r1": "rep_gone"},
        )
        return strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")

    def test_gate_independent_of_overload_threshold(self):
        """reserve=0.85 → rep_a (avail=160 < 240) filtered even though the
        overload knob was lowered to 0.3."""
        strat = _strat(
            slow_cut=SlowCut.CAPACITY_TOKEN_AWARE,
            load_threshold=0.9,
            sticky_overload_threshold=0.3,
            capacity_reserve_threshold=0.85,
        )
        scores = self._scores(strat, 0.9, 0.1)
        assert scores[1] == STICKY_TOP_SCORE  # rep_b (avail=1440) wins the gate

    def test_low_reserve_keeps_marginal_replica_eligible(self):
        """reserve=0.99 → thresh=16; rep_a (avail=160) eligible again and wins
        on remaining (gpu_hit absent → need equals plen on both)."""
        strat = _strat(
            slow_cut=SlowCut.CAPACITY_TOKEN_AWARE,
            load_threshold=0.9,
            capacity_reserve_threshold=0.99,
        )
        scores = self._scores(strat, 0.9, 0.95)  # rep_a avail=160 > rep_b avail=80
        assert scores[0] == STICKY_TOP_SCORE


class TestConfigCompat:
    def test_config_accepts_split_thresholds(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            collector_names=["x"],
            weight=1.0,
            load_threshold=0.9,
            sticky_overload_threshold=0.3,
            capacity_reserve_threshold=0.8,
        )
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.sticky_overload_threshold == 0.3
        assert strat.capacity_reserve_threshold == 0.8

    def test_config_defaults_none(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(collector_names=["x"], weight=1.0)
        assert cfg.sticky_overload_threshold is None
        assert cfg.capacity_reserve_threshold is None

    def test_config_rejects_out_of_range(self):
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, sticky_overload_threshold=0)
        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, capacity_reserve_threshold=1.0)
