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

"""Backend-agnostic KV-cache index and spec-aware prefix matcher."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from math import ceil, gcd, lcm
from typing import Literal

from ..types import (
    FULL_ATTENTION,
    MAMBA,
    SLIDING_WINDOW,
    KVCacheEventObservation,
    KVCacheGroupMetadata,
    KVCacheRegistryMetadata,
    Layer,
    PrefixHashChain,
    normalize_spec_kind,
    validate_registry,
)

LegacyFallbackState = Literal["disabled", "pending", "active", "rejected"]


class KVCacheStore:
    """Own the group-isolated GPU block index and its trusted registry."""

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self.replicas_by_block: dict[tuple[int, str], set[str]] = {}
        self._registry: KVCacheRegistryMetadata | None = None
        self._registry_generation = 0
        self._legacy_fallback_state: LegacyFallbackState = "disabled"
        self._legacy_fallback_enabled = False
        self._configured_replicas: set[str] = set()
        self._legacy_observed_block_sizes: dict[str, int] = {}
        self._legacy_source_versions: tuple[str | None, ...] = ()
        self._replica_layer_counts: dict[Layer, dict[str, int]] = {
            Layer.GPU: {},
            Layer.CPU: {},
            Layer.SSD: {},
        }
        self._lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Replica and block management

    def clear_replica(self, replica_id: str) -> None:
        """Clear one replica's blocks without changing registry/fallback state."""
        with self._lock:
            stale: list[tuple[int, str]] = []
            for key, replicas in self.replicas_by_block.items():
                replicas.discard(replica_id)
                if not replicas:
                    stale.append(key)
            for key in stale:
                del self.replicas_by_block[key]
            for counts in self._replica_layer_counts.values():
                counts.pop(replica_id, None)

    def add_blocks(
        self,
        replica_id: str,
        block_hashes: Iterable[str],
        layer: Layer = Layer.GPU,
        group_idx: int = 0,
    ) -> None:
        with self._lock:
            counts = self._replica_layer_counts.setdefault(layer, {})
            for block_hash in block_hashes:
                counts[replica_id] = counts.get(replica_id, 0) + 1
                if layer == Layer.GPU:
                    self.replicas_by_block.setdefault((group_idx, block_hash), set()).add(replica_id)

    def remove_blocks(
        self,
        replica_id: str,
        block_hashes: Iterable[str],
        layer: Layer = Layer.GPU,
        group_idx: int = 0,
    ) -> None:
        with self._lock:
            counts = self._replica_layer_counts.setdefault(layer, {})
            for block_hash in block_hashes:
                key = (group_idx, block_hash)
                replicas = self.replicas_by_block.get(key)
                existed = layer == Layer.GPU and replicas is not None and replica_id in replicas
                if existed:
                    replicas.discard(replica_id)
                    if not replicas:
                        del self.replicas_by_block[key]
                if layer != Layer.GPU or existed:
                    counts[replica_id] = max(0, counts.get(replica_id, 0) - 1)

    # Registry discovery and legacy fallback

    def reset_registry_discovery(self, replica_ids: Iterable[str], legacy_fallback_enabled: bool) -> None:
        """Reset discovery state for a newly initialized static server set."""
        with self._lock:
            self._registry = None
            self.block_size = None
            self._configured_replicas = set(replica_ids)
            self._legacy_fallback_enabled = legacy_fallback_enabled
            self._legacy_fallback_state = "disabled"
            self._legacy_observed_block_sizes.clear()
            self._legacy_source_versions = ()
            self._registry_generation += 1

    def begin_legacy_fallback(self, source_versions: Iterable[str | None]) -> None:
        """Enter pending only after every configured replica reports unsupported."""
        with self._lock:
            if self._legacy_fallback_state == "rejected" or not self._legacy_fallback_enabled:
                return
            self._registry = None
            self.block_size = None
            self._legacy_observed_block_sizes.clear()
            self._legacy_source_versions = tuple(source_versions)
            self._legacy_fallback_state = "pending"
            self._registry_generation += 1

    def install_cache_group_registry(
        self,
        metadata: KVCacheRegistryMetadata | Iterable[KVCacheGroupMetadata],
    ) -> None:
        """Atomically install a complete registry.

        The iterable form is retained for local callers from before the complete
        registry RPC. It deliberately disables partial matching.
        """
        if isinstance(metadata, KVCacheRegistryMetadata):
            registry = validate_registry(metadata)
        else:
            groups = tuple(KVCacheGroupMetadata.from_raw(item) for item in metadata)
            groups = tuple(
                group
                if group.spec_kind is not None
                else KVCacheGroupMetadata(group.group_idx, group.block_size, FULL_ATTENTION, group.sliding_window)
                for group in groups
            )
            sizes = [group.block_size for group in groups]
            registry = validate_registry(
                KVCacheRegistryMetadata(
                    groups=groups,
                    scheduler_block_size=lcm(*sizes),
                    hash_block_size=gcd(*sizes),
                    mamba_cache_mode=None,
                    partial_hash_hits_enabled=False,
                    source_vllm_version=None,
                    source="engine_core",
                )
            )
        with self._lock:
            self._install_registry_locked(registry)

    def _install_registry_locked(self, registry: KVCacheRegistryMetadata) -> None:
        self._registry = registry
        self.block_size = registry.scheduler_block_size
        self._legacy_fallback_state = "active" if registry.source == "legacy_single_group" else "disabled"
        self._registry_generation += 1

    def invalidate_cache_group_registry(self) -> None:
        with self._lock:
            self._registry = None
            self.block_size = None
            self._registry_generation += 1

    def _reject_legacy_locked(self) -> None:
        self._registry = None
        self.block_size = None
        self._legacy_fallback_state = "rejected"
        self._legacy_observed_block_sizes.clear()
        self._registry_generation += 1

    def cache_group_registry_ready(self) -> bool:
        with self._lock:
            return self._registry is not None

    def cache_group_registry_generation(self) -> int:
        with self._lock:
            return self._registry_generation

    def get_cache_group_registry(self) -> KVCacheRegistryMetadata | None:
        with self._lock:
            return self._registry

    def get_cache_group_metadata(self) -> tuple[KVCacheGroupMetadata, ...]:
        with self._lock:
            return self._registry.groups if self._registry is not None else ()

    def get_group_block_size(self, group_idx: int) -> int | None:
        with self._lock:
            if self._registry is None or group_idx >= len(self._registry.groups):
                return None
            return self._registry.groups[group_idx].block_size

    def get_registry_status(self) -> dict[str, object]:
        with self._lock:
            registry = self._registry
            return {
                "ready": registry is not None,
                "source": registry.source if registry is not None else None,
                "generation": self._registry_generation,
                "partial_hash_hits_enabled": (registry.partial_hash_hits_enabled if registry is not None else False),
                "legacy_single_group_fallback": self._legacy_fallback_state,
                "group_count": len(registry.groups) if registry is not None else 0,
                "scheduler_block_size": registry.scheduler_block_size if registry is not None else None,
                "hash_block_size": registry.hash_block_size if registry is not None else None,
                "source_vllm_version": registry.source_vllm_version if registry is not None else None,
            }

    def observe_event(self, replica_id: str, observation: KVCacheEventObservation) -> bool:
        """Validate one GPU BlockStored observation and advance fallback state."""
        observation = KVCacheEventObservation(
            group_idx=observation.group_idx,
            event_block_size=observation.event_block_size,
            spec_kind=normalize_spec_kind(observation.spec_kind),
            sliding_window=observation.sliding_window,
        )
        with self._lock:
            if self._legacy_fallback_state in ("pending", "active"):
                return self._observe_legacy_locked(replica_id, observation)
            if self._registry is None:
                return False
            if not self._validate_event_locked(observation):
                self._registry = None
                self.block_size = None
                self._registry_generation += 1
                return False
            return True

    def validate_observed_group_metadata(
        self,
        metadata: Iterable[KVCacheEventObservation | KVCacheGroupMetadata],
        replica_id: str = "",
    ) -> bool:
        """Compatibility batch wrapper around :meth:`observe_event`."""
        for item in metadata:
            if isinstance(item, KVCacheEventObservation):
                observation = item
            else:
                group = KVCacheGroupMetadata.from_raw(item)
                observation = KVCacheEventObservation(
                    group.group_idx,
                    group.block_size,
                    group.spec_kind,
                    group.sliding_window,
                )
            if not self.observe_event(replica_id, observation):
                return False
        return True

    def _observe_legacy_locked(self, replica_id: str, observation: KVCacheEventObservation) -> bool:
        valid = (
            replica_id in self._configured_replicas
            and observation.group_idx == 0
            and observation.event_block_size is not None
            and observation.event_block_size > 0
            and observation.spec_kind in (None, FULL_ATTENTION)
            and observation.sliding_window is None
        )
        expected_sizes = set(self._legacy_observed_block_sizes.values())
        if expected_sizes and observation.event_block_size not in expected_sizes:
            valid = False
        if not valid:
            self._reject_legacy_locked()
            return False
        assert observation.event_block_size is not None
        self._legacy_observed_block_sizes[replica_id] = observation.event_block_size
        if self._legacy_fallback_state == "active":
            registry = self._registry
            if registry is None or registry.hash_block_size != observation.event_block_size:
                self._reject_legacy_locked()
                return False
            return True
        if self._configured_replicas and self._configured_replicas <= self._legacy_observed_block_sizes.keys():
            versions = set(self._legacy_source_versions)
            source_version = next(iter(versions)) if len(versions) == 1 else None
            size = observation.event_block_size
            self._install_registry_locked(
                validate_registry(
                    KVCacheRegistryMetadata(
                        groups=(KVCacheGroupMetadata(0, size, FULL_ATTENTION, None),),
                        scheduler_block_size=size,
                        hash_block_size=size,
                        mamba_cache_mode=None,
                        partial_hash_hits_enabled=False,
                        source_vllm_version=source_version,
                        source="legacy_single_group",
                    )
                )
            )
        return True

    def _validate_event_locked(self, observation: KVCacheEventObservation) -> bool:
        assert self._registry is not None
        if observation.group_idx < 0 or observation.group_idx >= len(self._registry.groups):
            return False
        group = self._registry.groups[observation.group_idx]
        if observation.event_block_size is None or observation.event_block_size <= 0:
            return False
        if observation.spec_kind is not None and observation.spec_kind != group.spec_kind:
            return False
        if observation.sliding_window is not None and observation.sliding_window != group.sliding_window:
            return False
        if observation.event_block_size == group.block_size:
            return True
        return (
            self._registry.partial_hash_hits_enabled
            and observation.event_block_size == self._registry.hash_block_size
            and observation.event_block_size < group.block_size
            and group.spec_kind in (FULL_ATTENTION, MAMBA)
        )

    # Retained-cache size

    def per_replica_block_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._replica_layer_counts.get(Layer.GPU, {}))

    # Prefix queries

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        hash_strs: list[str],
        layer: Layer = Layer.GPU,
    ) -> float:
        if layer != Layer.GPU or self.block_size is None:
            return 0.0
        with self._lock:
            if not hash_strs:
                return 0.0
            matched = 0
            for idx, hash_str in enumerate(hash_strs):
                if not self._cached_locked(node_id, 0, hash_str):
                    break
                matched = idx + 1
            return matched / len(hash_strs)

    def get_gpu_prefix_hit_rate(
        self,
        node_id: str,
        chain: PrefixHashChain | dict[int, list[str]],
    ) -> float:
        """Return the exact common prefix hit for every required group."""
        with self._lock:
            if self._registry is None:
                return 0.0
            if isinstance(chain, dict):
                return self._compat_group_query_locked(node_id, chain)
            registry = self._registry
            if (
                chain.registry_generation != self._registry_generation
                or chain.hash_block_size != registry.hash_block_size
            ):
                return 0.0
            alignment = (
                registry.hash_block_size if registry.partial_hash_hits_enabled else registry.scheduler_block_size
            )
            candidate = max(0, chain.prompt_token_count - 1)
            query_tokens = candidate // alignment * alignment
            query_tokens = min(query_tokens, len(chain.hashes) * chain.hash_block_size)
            query_tokens = query_tokens // alignment * alignment
            if query_tokens == 0:
                return 0.0

            full_groups = sorted(
                (group for group in registry.groups if group.spec_kind == FULL_ATTENTION),
                key=lambda group: group.group_idx,
            )
            other_groups = sorted(
                (group for group in registry.groups if group.spec_kind != FULL_ATTENTION),
                key=lambda group: (group.spec_kind or "", group.group_idx),
            )
            ordered = [*full_groups, *other_groups]
            common = query_tokens
            while True:
                changed = False
                for group in ordered:
                    narrowed = self._group_hit_locked(node_id, chain, group, common, registry)
                    if narrowed < common:
                        common = narrowed
                        changed = True
                        break
                if not changed or common == 0:
                    break
            return common / query_tokens

    def _cached_locked(self, node_id: str, group_idx: int, hash_str: str) -> bool:
        replicas = self.replicas_by_block.get((group_idx, hash_str))
        return replicas is not None and node_id in replicas

    @staticmethod
    def _hash_at(chain: PrefixHashChain, token_boundary: int) -> str | None:
        if token_boundary <= 0 or token_boundary % chain.hash_block_size:
            return None
        index = token_boundary // chain.hash_block_size - 1
        return chain.hashes[index] if 0 <= index < len(chain.hashes) else None

    def _has_boundary_locked(
        self,
        node_id: str,
        group_idx: int,
        chain: PrefixHashChain,
        token_boundary: int,
    ) -> bool:
        hash_str = self._hash_at(chain, token_boundary)
        return hash_str is not None and self._cached_locked(node_id, group_idx, hash_str)

    def _group_hit_locked(
        self,
        node_id: str,
        chain: PrefixHashChain,
        group: KVCacheGroupMetadata,
        candidate: int,
        registry: KVCacheRegistryMetadata,
    ) -> int:
        if group.spec_kind == FULL_ATTENTION:
            return self._full_attention_hit_locked(node_id, chain, group, candidate, registry)
        if group.spec_kind == SLIDING_WINDOW:
            return self._sliding_window_hit_locked(node_id, chain, group, candidate)
        if group.spec_kind == MAMBA:
            return self._mamba_hit_locked(node_id, chain, group, candidate, registry)
        return 0

    def _full_attention_hit_locked(
        self,
        node_id: str,
        chain: PrefixHashChain,
        group: KVCacheGroupMetadata,
        candidate: int,
        registry: KVCacheRegistryMetadata,
    ) -> int:
        full_boundary = 0
        for boundary in range(group.block_size, candidate + 1, group.block_size):
            if not self._has_boundary_locked(node_id, group.group_idx, chain, boundary):
                break
            full_boundary = boundary
        else:
            if candidate % group.block_size == 0:
                return candidate
        if not registry.partial_hash_hits_enabled:
            return full_boundary
        upper = min(candidate, full_boundary + group.block_size - registry.hash_block_size)
        upper = upper // registry.hash_block_size * registry.hash_block_size
        for boundary in range(upper, full_boundary, -registry.hash_block_size):
            if self._has_boundary_locked(node_id, group.group_idx, chain, boundary):
                return boundary
        return full_boundary

    def _sliding_window_hit_locked(
        self,
        node_id: str,
        chain: PrefixHashChain,
        group: KVCacheGroupMetadata,
        candidate: int,
    ) -> int:
        assert group.sliding_window is not None
        end_blocks = candidate // group.block_size
        required = ceil((group.sliding_window - 1) / group.block_size)
        for end in range(end_blocks, 0, -1):
            start = max(0, end - required)
            boundaries = range((start + 1) * group.block_size, (end + 1) * group.block_size, group.block_size)
            if all(self._has_boundary_locked(node_id, group.group_idx, chain, boundary) for boundary in boundaries):
                return end * group.block_size
        return 0

    def _mamba_hit_locked(
        self,
        node_id: str,
        chain: PrefixHashChain,
        group: KVCacheGroupMetadata,
        candidate: int,
        registry: KVCacheRegistryMetadata,
    ) -> int:
        step = registry.hash_block_size if registry.partial_hash_hits_enabled else group.block_size
        start_boundary = candidate // step * step
        for boundary in range(start_boundary, 0, -step):
            if self._has_boundary_locked(node_id, group.group_idx, chain, boundary):
                return boundary
        return 0

    def _compat_group_query_locked(self, node_id: str, hashes_by_group: dict[int, list[str]]) -> float:
        """Compatibility for the pre-common-chain internal API."""
        assert self._registry is not None
        if any(group.group_idx not in hashes_by_group for group in self._registry.groups):
            return 0.0
        boundary = self._registry.scheduler_block_size
        query = min(len(hashes_by_group[group.group_idx]) * group.block_size for group in self._registry.groups)
        query = query // boundary * boundary
        if not query:
            return 0.0
        hits = []
        for group in self._registry.groups:
            matched = 0
            for hash_str in hashes_by_group[group.group_idx]:
                if not self._cached_locked(node_id, group.group_idx, hash_str):
                    break
                matched += 1
            hits.append(matched * group.block_size)
        return (min(hits) // boundary * boundary) / query
