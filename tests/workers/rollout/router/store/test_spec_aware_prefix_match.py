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

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.kv_cache_store import KVCacheStore
from verl.workers.rollout.router.kvcaware.types import (
    KVCacheEventObservation,
    KVCacheGroupMetadata,
    KVCacheRegistryMetadata,
    PrefixHashChain,
)
from verl.workers.rollout.router.kvcaware.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _registry(
    groups: tuple[KVCacheGroupMetadata, ...],
    *,
    scheduler: int,
    hash_size: int,
    partial: bool,
    source: str = "engine_core",
) -> KVCacheRegistryMetadata:
    return KVCacheRegistryMetadata(
        groups=groups,
        scheduler_block_size=scheduler,
        hash_block_size=hash_size,
        mamba_cache_mode="align" if any(group.spec_kind == "mamba" for group in groups) else None,
        partial_hash_hits_enabled=partial,
        source_vllm_version="0.26.0",
        source=source,
    )


def _chain(store: KVCacheStore, prompt_tokens: int, hash_size: int) -> PrefixHashChain:
    values, _ = get_prefix_hashes_incremental(list(range(prompt_tokens)), hash_size, 0, 0)
    return PrefixHashChain(
        prompt_token_count=prompt_tokens,
        hash_block_size=hash_size,
        hashes=tuple(str(value) for value in values),
        registry_generation=store.cache_group_registry_generation(),
    )


def _add_boundaries(
    store: KVCacheStore,
    chain: PrefixHashChain,
    group_idx: int,
    boundaries: list[int],
    node_id: str = "replica",
) -> None:
    store.add_blocks(
        node_id,
        [chain.hashes[boundary // chain.hash_block_size - 1] for boundary in boundaries],
        group_idx=group_idx,
    )


def test_full_attention_partial_tail_and_last_prompt_token_exclusion() -> None:
    store = KVCacheStore()
    store.install_cache_group_registry(
        _registry(
            (
                KVCacheGroupMetadata(0, 8, "full_attention"),
                KVCacheGroupMetadata(1, 8, "mamba"),
            ),
            scheduler=8,
            hash_size=4,
            partial=True,
        )
    )
    chain = _chain(store, prompt_tokens=14, hash_size=4)
    _add_boundaries(store, chain, 0, [8, 12])
    _add_boundaries(store, chain, 1, [12])

    assert store.get_gpu_prefix_hit_rate("replica", chain) == 1.0

    # prompt_token_count - 1 caps the query at 12 rather than hashing/reusing
    # all 16 tokens of a longer backing hash chain.
    capped = PrefixHashChain(13, 4, chain.hashes, chain.registry_generation)
    assert store.get_gpu_prefix_hit_rate("replica", capped) == 1.0


def test_sliding_window_accepts_complete_window_without_older_blocks() -> None:
    store = KVCacheStore()
    store.install_cache_group_registry(
        _registry(
            (KVCacheGroupMetadata(0, 8, "sliding_window", 17),),
            scheduler=8,
            hash_size=8,
            partial=False,
        )
    )
    chain = _chain(store, prompt_tokens=25, hash_size=8)
    _add_boundaries(store, chain, 0, [16, 24])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 1.0


def test_sliding_window_short_prefix_must_start_at_block_zero() -> None:
    store = KVCacheStore()
    store.install_cache_group_registry(
        _registry(
            (KVCacheGroupMetadata(0, 8, "sliding_window", 25),),
            scheduler=8,
            hash_size=8,
            partial=False,
        )
    )
    chain = _chain(store, prompt_tokens=17, hash_size=8)
    _add_boundaries(store, chain, 0, [16])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 0.0

    _add_boundaries(store, chain, 0, [8])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 1.0


def test_mamba_uses_nearest_checkpoint_without_continuity() -> None:
    store = KVCacheStore()
    store.install_cache_group_registry(
        _registry(
            (
                KVCacheGroupMetadata(0, 8, "full_attention"),
                KVCacheGroupMetadata(1, 8, "mamba"),
            ),
            scheduler=8,
            hash_size=4,
            partial=True,
        )
    )
    chain = _chain(store, prompt_tokens=14, hash_size=4)
    _add_boundaries(store, chain, 0, [8, 12])
    _add_boundaries(store, chain, 1, [12])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 1.0


def test_three_spec_fixed_point_rechecks_after_each_shrink() -> None:
    store = KVCacheStore()
    store.install_cache_group_registry(
        _registry(
            (
                KVCacheGroupMetadata(0, 8, "full_attention"),
                KVCacheGroupMetadata(1, 8, "mamba"),
                KVCacheGroupMetadata(2, 8, "sliding_window", 9),
            ),
            scheduler=8,
            hash_size=4,
            partial=True,
        )
    )
    chain = _chain(store, prompt_tokens=14, hash_size=4)
    _add_boundaries(store, chain, 0, [8, 12])
    _add_boundaries(store, chain, 1, [12])
    _add_boundaries(store, chain, 2, [8])

    # SWA shrinks 12 -> 8; restarting then discovers that the only Mamba
    # checkpoint was at 12 and cannot serve the new common boundary.
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 0.0

    _add_boundaries(store, chain, 1, [8])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == pytest.approx(8 / 12)


def test_registry_generation_mismatch_invalidates_request_chain() -> None:
    store = KVCacheStore()
    registry = _registry(
        (KVCacheGroupMetadata(0, 8, "full_attention"),),
        scheduler=8,
        hash_size=8,
        partial=False,
    )
    store.install_cache_group_registry(registry)
    chain = _chain(store, prompt_tokens=17, hash_size=8)
    _add_boundaries(store, chain, 0, [8, 16])
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 1.0

    store.invalidate_cache_group_registry()
    store.install_cache_group_registry(registry)
    assert store.get_gpu_prefix_hit_rate("replica", chain) == 0.0


def test_legacy_fallback_activates_then_rejects_permanently() -> None:
    store = KVCacheStore()
    store.reset_registry_discovery(["a", "b"], legacy_fallback_enabled=True)
    store.begin_legacy_fallback(["0.21.0", "0.21.0"])
    observation = KVCacheEventObservation(0, 16, None, None)

    assert store.observe_event("a", observation)
    assert not store.cache_group_registry_ready()
    assert store.observe_event("b", observation)
    assert store.get_registry_status()["legacy_single_group_fallback"] == "active"

    generation = store.cache_group_registry_generation()
    assert not store.observe_event("a", KVCacheEventObservation(1, 16, None, None))
    status = store.get_registry_status()
    assert status["legacy_single_group_fallback"] == "rejected"
    assert not status["ready"]
    assert status["generation"] > generation
    assert not store.observe_event("a", observation)


@pytest.mark.parametrize(
    "observation",
    [
        KVCacheEventObservation(0, 32, None, None),
        KVCacheEventObservation(0, 16, "mamba", None),
        KVCacheEventObservation(0, 16, "full_attention", 1024),
    ],
)
def test_legacy_fallback_rejects_replica_conflicts(observation: KVCacheEventObservation) -> None:
    store = KVCacheStore()
    store.reset_registry_discovery(["a", "b"], legacy_fallback_enabled=True)
    store.begin_legacy_fallback(["0.21.0", "0.21.0"])
    assert store.observe_event("a", KVCacheEventObservation(0, 16, None, None))
    assert not store.observe_event("b", observation)
    assert store.get_registry_status()["legacy_single_group_fallback"] == "rejected"


def test_clear_preserves_verified_registry_and_fallback_state() -> None:
    store = KVCacheStore()
    store.reset_registry_discovery(["a"], legacy_fallback_enabled=True)
    store.begin_legacy_fallback(["0.21.0"])
    assert store.observe_event("a", KVCacheEventObservation(0, 16, None, None))
    store.add_blocks("a", ["hash"], group_idx=0)
    generation = store.cache_group_registry_generation()

    store.clear_replica("a")

    assert store.cache_group_registry_ready()
    assert store.get_registry_status()["legacy_single_group_fallback"] == "active"
    assert store.cache_group_registry_generation() == generation
    assert not store.replicas_by_block
