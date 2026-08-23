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

"""Tests for P4 SKEW overload mode — sustained pool-relative skew.

The 27.24 shape (exp1): running ~13 vs pool median ~3, inflight-token CV 0.68,
yet KV usage only 0.2 — absolute KV/load thresholds structurally never fire.
SKEW compares the bound replica against the POOL MEDIAN and requires the skew
to persist across ``skew_window`` consecutive snapshots before unbinding.
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
        overload_mode=OverloadMode.SKEW,
        slow_cut=SlowCut.LEAST_INFLIGHT,
        # Small window so tests exercise the streak without 60 iterations.
        skew_window=3,
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


def _pool(rep_hot: dict, others: list[dict], n_others: int = 3) -> dict:
    """Build a pool where one hot replica stands against uniform others."""
    data = {"rep_hot": rep_hot}
    for i in range(n_others):
        data[f"rep_{i}"] = dict(others)
    return data


HOT_2724 = {"active_sessions": 13, "num_requests_running": 13, "inflight_tokens": 60000}
CALM = {"active_sessions": 3, "num_requests_running": 3, "inflight_tokens": 9000}


class TestSkewVerdict:
    def test_hotspot_unbinds_after_sustained_skew(self):
        """The 27.24 shape: after skew_window samples, sticky falls back."""
        strat = _strat()
        provider = FakeRouteDataProvider(_pool(HOT_2724, CALM), sticky={"r1": "rep_hot"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        for i in range(strat.skew_window - 1):
            scores = strat.score(PROMPT_IDS, provider, reps, "r1")
            assert scores[0] == STICKY_TOP_SCORE  # not yet sustained
        scores = strat.score(PROMPT_IDS, provider, reps, "r1")
        assert scores[0] != STICKY_TOP_SCORE  # streak complete → fallback

    def test_brief_spike_never_unbinds(self):
        """Skew below the window (transient burst) keeps the binding."""
        strat = _strat(skew_window=5)
        provider = FakeRouteDataProvider(_pool(HOT_2724, CALM), sticky={"r1": "rep_hot"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        for _ in range(4):
            strat.score(PROMPT_IDS, provider, reps, "r1")
        calm_provider = FakeRouteDataProvider(_pool(CALM, CALM), sticky={"r1": "rep_hot"})
        for _ in range(10):
            scores = strat.score(PROMPT_IDS, calm_provider, reps, "r1")
            assert scores[0] == STICKY_TOP_SCORE

    def test_one_clean_sample_resets_streak(self):
        """skew → calm → skew: the streak restarts from zero."""
        strat = _strat(skew_window=3)
        hot = FakeRouteDataProvider(_pool(HOT_2724, CALM), sticky={"r1": "rep_hot"})
        calm = FakeRouteDataProvider(_pool(CALM, CALM), sticky={"r1": "rep_hot"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        strat.score(PROMPT_IDS, hot, reps, "r1")  # streak 1
        strat.score(PROMPT_IDS, hot, reps, "r1")  # streak 2
        strat.score(PROMPT_IDS, calm, reps, "r1")  # reset to 0
        scores = strat.score(PROMPT_IDS, hot, reps, "r1")  # streak 1 again
        assert scores[0] == STICKY_TOP_SCORE

    def test_combo_signal_running_and_tokens(self):
        """running AND inflight_tokens both > factor×median, sessions level."""
        strat = _strat(skew_delta=99)  # disable the sessions line
        combo = {"active_sessions": 4, "num_requests_running": 9, "inflight_tokens": 40000}
        provider = FakeRouteDataProvider(_pool(combo, CALM), sticky={"r1": "rep_hot"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        for _ in range(strat.skew_window - 1):
            strat.score(PROMPT_IDS, provider, reps, "r1")
        scores = strat.score(PROMPT_IDS, provider, reps, "r1")
        assert scores[0] != STICKY_TOP_SCORE

    def test_running_alone_is_not_enough(self):
        """running skewed but tokens level → no combo verdict, stays bound."""
        strat = _strat(skew_delta=99)
        half = {"active_sessions": 4, "num_requests_running": 9, "inflight_tokens": 9000}
        provider = FakeRouteDataProvider(_pool(half, CALM), sticky={"r1": "rep_hot"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        for _ in range(strat.skew_window + 2):
            scores = strat.score(PROMPT_IDS, provider, reps, "r1")
            assert scores[0] == STICKY_TOP_SCORE

    def test_calm_replica_never_flags(self):
        """A median-level bound replica stays bound indefinitely."""
        strat = _strat()
        provider = FakeRouteDataProvider(_pool(HOT_2724, CALM), sticky={"r1": "rep_0"})
        reps = _replicas("rep_hot", "rep_0", "rep_1", "rep_2")
        for _ in range(strat.skew_window + 5):
            scores = strat.score(PROMPT_IDS, provider, reps, "r1")
            assert scores[1] == STICKY_TOP_SCORE

    def test_no_replicas_no_verdict(self):
        strat = _strat()
        provider = FakeRouteDataProvider({"rep_hot": dict(HOT_2724)}, sticky={"r1": "rep_hot"})
        assert strat.is_overloaded(provider, ReplicaInfo("rep_hot"), []) is False


class TestSkewConfig:
    def test_config_roundtrip(self):
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            collector_names=["x"],
            weight=1.0,
            overload_mode="skew",
            skew_window=30,
            skew_delta=3,
            skew_factor=2.5,
        )
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.overload_mode == OverloadMode.SKEW
        assert strat.skew_window == 30
        assert strat.skew_delta == 3
        assert strat.skew_factor == 2.5

    def test_config_rejects_bad_values(self):
        from verl.workers.rollout.router.kvcaware.config.base import ConfigError
        from verl.workers.rollout.router.kvcaware.config.strategy import KVCAwareStrategyConfig

        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, skew_window=0)
        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, skew_delta=-1)
        with pytest.raises(ConfigError):
            KVCAwareStrategyConfig(collector_names=["x"], weight=1.0, skew_factor=1.0)
