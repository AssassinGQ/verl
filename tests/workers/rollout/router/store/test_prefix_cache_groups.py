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

from verl.workers.rollout.router.kvcaware.types import (
    KVCacheGroupMetadata,
    KVCacheRegistryMetadata,
    PrefixHashChain,
)
from verl.workers.rollout.router.kvcaware.utils import prefix_cache

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _registry(hash_block_size: int = 4) -> KVCacheRegistryMetadata:
    return KVCacheRegistryMetadata(
        groups=(
            KVCacheGroupMetadata(0, 8, "full_attention"),
            KVCacheGroupMetadata(1, 8, "mamba"),
        ),
        scheduler_block_size=8,
        hash_block_size=hash_block_size,
        mamba_cache_mode="align",
        partial_hash_hits_enabled=hash_block_size < 8,
        source_vllm_version="0.26.0",
        source="engine_core",
    )


class _RegistryStore:
    def __init__(self) -> None:
        self.ready = True
        self.generation = 1
        self.registry = _registry()
        self.requests: dict[str, dict] = {}

    def is_cache_group_registry_ready(self) -> bool:
        return self.ready

    def get_cache_group_registry(self):
        return self.registry

    def get_cache_group_registry_generation(self) -> int:
        return self.generation

    def get_per_request(self, request_id, key, default=None):
        return self.requests.get(request_id, {}).get(key, default)

    def set_per_request(self, request_id, key, value):
        self.requests.setdefault(request_id, {})[key] = value


def test_common_hash_chain_extends_incrementally(monkeypatch) -> None:
    store = _RegistryStore()
    starts: list[int] = []
    original = prefix_cache.get_prefix_hashes_incremental

    def spy(prompt_ids, block_size, parent_hash, start_block_idx, seed=0):
        starts.append(start_block_idx)
        return original(prompt_ids, block_size, parent_hash, start_block_idx, seed)

    monkeypatch.setattr(prefix_cache, "get_prefix_hashes_incremental", spy)
    first = prefix_cache.resolve_prefix_hashes(list(range(8)), "request", store)
    second = prefix_cache.resolve_prefix_hashes(list(range(12)), "request", store)

    assert isinstance(first, PrefixHashChain)
    assert isinstance(second, PrefixHashChain)
    assert second.hashes[: len(first.hashes)] == first.hashes
    assert (first.prompt_token_count, second.prompt_token_count) == (8, 12)
    assert starts == [0, 2]


def test_generation_or_hash_size_change_recomputes(monkeypatch) -> None:
    store = _RegistryStore()
    starts: list[int] = []
    original = prefix_cache.get_prefix_hashes_incremental

    def spy(prompt_ids, block_size, parent_hash, start_block_idx, seed=0):
        starts.append(start_block_idx)
        return original(prompt_ids, block_size, parent_hash, start_block_idx, seed)

    monkeypatch.setattr(prefix_cache, "get_prefix_hashes_incremental", spy)
    prefix_cache.resolve_prefix_hashes(list(range(16)), "request", store)
    store.generation += 1
    store.registry = _registry(hash_block_size=8)
    result = prefix_cache.resolve_prefix_hashes(list(range(16)), "request", store)

    assert starts == [0, 0]
    assert result is not None
    assert result.hash_block_size == 8
    assert len(result.hashes) == 2


def test_not_ready_returns_none() -> None:
    store = _RegistryStore()
    store.ready = False
    assert prefix_cache.resolve_prefix_hashes(list(range(12)), "request", store) is None
