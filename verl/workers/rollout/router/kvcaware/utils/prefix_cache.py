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

"""Per-request checkpointing for the common fine-grained prefix hash chain."""

from __future__ import annotations

from typing import Any

from ..types import PrefixHashChain
from .hash import get_prefix_hashes_incremental

_PREFIX_HASH_KEY = "prefix_hashes"


def resolve_prefix_hashes(
    prompt_ids: list[int],
    request_id: str | None,
    store: Any,
) -> PrefixHashChain | None:
    """Return the registry-generation-bound token-only prefix hash chain."""
    if hasattr(store, "is_cache_group_registry_ready"):
        if not store.is_cache_group_registry_ready():
            return None
        registry = store.get_cache_group_registry()
        if registry is None:
            return None
        hash_block_size = registry.hash_block_size
        generation = store.get_cache_group_registry_generation()
    else:
        # Narrow compatibility for non-production strategy test providers.
        hash_block_size = store.get_block_size()
        generation = 0
    if not hash_block_size or hash_block_size <= 0:
        return None

    cached = store.get_per_request(request_id, _PREFIX_HASH_KEY) if request_id else None
    if (
        not cached
        or cached.get("generation") != generation
        or cached.get("hash_block_size") != hash_block_size
        or len(prompt_ids) // hash_block_size < len(cached.get("hashes", ()))
    ):
        cached = None

    if cached is not None:
        tail, parent = get_prefix_hashes_incremental(
            prompt_ids,
            hash_block_size,
            cached["parent_hash"],
            len(cached["hashes"]),
        )
        hashes = (*cached["hashes"], *(str(value) for value in tail))
    else:
        values, parent = get_prefix_hashes_incremental(prompt_ids, hash_block_size, 0, 0)
        hashes = tuple(str(value) for value in values)

    if request_id:
        store.set_per_request(
            request_id,
            _PREFIX_HASH_KEY,
            {
                "generation": generation,
                "hash_block_size": hash_block_size,
                "hashes": hashes,
                "parent_hash": parent,
            },
        )
    return PrefixHashChain(
        prompt_token_count=len(prompt_ids),
        hash_block_size=hash_block_size,
        hashes=hashes,
        registry_generation=generation,
    )
