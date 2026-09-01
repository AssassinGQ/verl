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
    get_vllm_cache_capabilities,
    resolve_registry_metadata,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]

_GROUPS = (
    KVCacheGroupMetadata(0, 8, "full_attention"),
    KVCacheGroupMetadata(1, 16, "mamba"),
)


@pytest.mark.parametrize(
    ("version", "partial"),
    [
        ("0.22.0", False),
        ("0.23.3", False),
        ("0.24.0", True),
        ("0.25.1", True),
        ("0.26.0", True),
        ("0.31.0", True),
    ],
)
def test_version_tiers_preserve_full_registry_but_gate_partial(version: str, partial: bool) -> None:
    capabilities = get_vllm_cache_capabilities(version)
    registry = resolve_registry_metadata(
        _GROUPS,
        source_vllm_version=version,
        prefix_caching_enabled=True,
        configured_hash_block_size=4,
        mamba_cache_mode="align",
        partial_event_capable=capabilities.supports_partial_events,
    )

    assert registry.scheduler_block_size == 16
    assert registry.hash_block_size == 4
    assert registry.partial_hash_hits_enabled is partial


def test_hash_config_field_changes_at_v026() -> None:
    assert get_vllm_cache_capabilities("0.25.2").hash_config_field == "hash_block_size"
    assert get_vllm_cache_capabilities("0.26.0").hash_config_field == "prefix_match_unit"


def test_legacy_versions_and_unknown_versions_have_no_metadata_capability() -> None:
    assert not get_vllm_cache_capabilities("0.21.2").supports_group_metadata
    assert not get_vllm_cache_capabilities(None).supports_group_metadata
    assert not get_vllm_cache_capabilities("not-a-version").supports_group_metadata


@pytest.mark.parametrize(
    "groups",
    [
        (
            KVCacheGroupMetadata(0, 8, "full_attention"),
            KVCacheGroupMetadata(1, 10, "mamba"),
        ),
        (
            KVCacheGroupMetadata(0, 8, "unknown_future_spec"),
            KVCacheGroupMetadata(1, 8, "mamba"),
        ),
    ],
)
def test_invalid_divisibility_or_unknown_spec_fails_closed(groups) -> None:
    with pytest.raises(ValueError):
        resolve_registry_metadata(
            groups,
            source_vllm_version="0.26.0",
            prefix_caching_enabled=True,
            configured_hash_block_size=4,
            mamba_cache_mode="align",
            partial_event_capable=True,
        )


def test_non_align_mamba_disables_fine_hashing() -> None:
    registry = resolve_registry_metadata(
        (
            KVCacheGroupMetadata(0, 8, "full_attention"),
            KVCacheGroupMetadata(1, 8, "mamba"),
        ),
        source_vllm_version="0.26.0",
        prefix_caching_enabled=True,
        configured_hash_block_size=4,
        mamba_cache_mode="all",
        partial_event_capable=True,
    )
    assert registry.hash_block_size == registry.scheduler_block_size == 8
    assert not registry.partial_hash_hits_enabled


def test_partial_event_span_does_not_replace_physical_group_size() -> None:
    registry = resolve_registry_metadata(
        _GROUPS,
        source_vllm_version="0.26.0",
        prefix_caching_enabled=True,
        configured_hash_block_size=4,
        mamba_cache_mode="align",
        partial_event_capable=True,
    )
    store = KVCacheStore()
    store.install_cache_group_registry(registry)

    assert store.observe_event("replica", KVCacheEventObservation(1, 4, "mamba", None))
    assert store.get_group_block_size(1) == 16


def test_swa_rejects_partial_event_and_invalidates_registry() -> None:
    registry = resolve_registry_metadata(
        (
            KVCacheGroupMetadata(0, 8, "full_attention"),
            KVCacheGroupMetadata(1, 8, "sliding_window", 1024),
        ),
        source_vllm_version="0.26.0",
        prefix_caching_enabled=True,
        configured_hash_block_size=4,
        mamba_cache_mode=None,
        partial_event_capable=True,
    )
    store = KVCacheStore()
    store.install_cache_group_registry(registry)

    assert not store.observe_event(
        "replica",
        KVCacheEventObservation(1, 4, "sliding_window", 1024),
    )
    assert not store.cache_group_registry_ready()


def test_legacy_fallback_is_disabled_by_default() -> None:
    store = KVCacheStore()
    store.reset_registry_discovery(["replica"], legacy_fallback_enabled=False)
    store.begin_legacy_fallback(["0.21.0"])
    assert store.get_registry_status()["legacy_single_group_fallback"] == "disabled"
    assert not store.observe_event("replica", KVCacheEventObservation(0, 16, None, None))


def test_enum_align_mamba_mode_keeps_fine_hashing() -> None:
    from enum import Enum

    class Mode(Enum):
        ALIGN = "align"

    registry = resolve_registry_metadata(
        (
            KVCacheGroupMetadata(0, 8, "full_attention"),
            KVCacheGroupMetadata(1, 8, "mamba"),
        ),
        source_vllm_version="0.26.0",
        prefix_caching_enabled=True,
        configured_hash_block_size=4,
        mamba_cache_mode=Mode.ALIGN,
        partial_event_capable=True,
    )

    assert registry.mamba_cache_mode == "align"
    assert registry.hash_block_size == 4
    assert registry.partial_hash_hits_enabled
