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

"""KV-cache metadata shared by discovery, event collection, and routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import gcd, lcm
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from .layer import Layer

CacheScope = tuple[Layer, int]
RegistrySource = Literal["engine_core", "legacy_single_group"]

FULL_ATTENTION = "full_attention"
SLIDING_WINDOW = "sliding_window"
MAMBA = "mamba"
KNOWN_SPEC_KINDS = frozenset({FULL_ATTENTION, SLIDING_WINDOW, MAMBA})


def normalize_spec_kind(value: Any) -> str | None:
    """Normalize the stable vLLM spec names accepted by this router."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    kind = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "full": FULL_ATTENTION,
        "attention": FULL_ATTENTION,
        "fullattentionspec": FULL_ATTENTION,
        "slidingwindow": SLIDING_WINDOW,
        "slidingwindowspec": SLIDING_WINDOW,
        "mambaspec": MAMBA,
    }
    return aliases.get(kind, kind)


@dataclass(frozen=True)
class KVCacheGroupMetadata:
    """Final physical configuration for one vLLM KV-cache group."""

    group_idx: int
    block_size: int
    spec_kind: str | None = None
    sliding_window: int | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> KVCacheGroupMetadata:
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, Mapping):
            group_idx = raw.get("group_idx", raw.get("group_id"))
            block_size = raw.get("block_size")
            spec_kind = raw.get("spec_kind", raw.get("kind", raw.get("kv_cache_spec_kind")))
            sliding_window = raw.get("sliding_window", raw.get("kv_cache_spec_sliding_window"))
        else:
            group_idx = getattr(raw, "group_idx", getattr(raw, "group_id", None))
            block_size = getattr(raw, "block_size", None)
            spec_kind = getattr(
                raw,
                "spec_kind",
                getattr(raw, "kind", getattr(raw, "kv_cache_spec_kind", None)),
            )
            sliding_window = getattr(raw, "sliding_window", getattr(raw, "kv_cache_spec_sliding_window", None))
        if group_idx is None or block_size is None:
            raise ValueError(f"cache-group metadata is missing group_idx or block_size: {raw!r}")
        return cls(
            group_idx=int(group_idx),
            block_size=int(block_size),
            spec_kind=normalize_spec_kind(spec_kind),
            sliding_window=int(sliding_window) if sliding_window is not None else None,
        )


@dataclass(frozen=True)
class KVCacheRegistryMetadata:
    """Complete, replica-consistent configuration used for GPU prefix queries."""

    groups: tuple[KVCacheGroupMetadata, ...]
    scheduler_block_size: int
    hash_block_size: int
    mamba_cache_mode: str | None
    partial_hash_hits_enabled: bool
    source_vllm_version: str | None
    source: RegistrySource

    @classmethod
    def from_raw(cls, raw: Any) -> KVCacheRegistryMetadata:
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"cache registry must be a mapping, got {type(raw).__name__}")
        groups_raw = raw.get("groups")
        if not isinstance(groups_raw, Sequence) or isinstance(groups_raw, str | bytes):
            raise ValueError("cache registry groups must be a sequence")
        return cls(
            groups=tuple(KVCacheGroupMetadata.from_raw(item) for item in groups_raw),
            scheduler_block_size=int(raw["scheduler_block_size"]),
            hash_block_size=int(raw["hash_block_size"]),
            mamba_cache_mode=(str(raw["mamba_cache_mode"]) if raw.get("mamba_cache_mode") is not None else None),
            partial_hash_hits_enabled=bool(raw["partial_hash_hits_enabled"]),
            source_vllm_version=(
                str(raw["source_vllm_version"]) if raw.get("source_vllm_version") is not None else None
            ),
            source=str(raw.get("source", "engine_core")),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "group_idx": group.group_idx,
                    "block_size": group.block_size,
                    "spec_kind": group.spec_kind,
                    "sliding_window": group.sliding_window,
                }
                for group in self.groups
            ],
            "scheduler_block_size": self.scheduler_block_size,
            "hash_block_size": self.hash_block_size,
            "mamba_cache_mode": self.mamba_cache_mode,
            "partial_hash_hits_enabled": self.partial_hash_hits_enabled,
            "source_vllm_version": self.source_vllm_version,
            "source": self.source,
        }


@dataclass(frozen=True)
class PrefixHashChain:
    """Token-only chained hashes at the registry's finest supported boundary."""

    prompt_token_count: int
    hash_block_size: int
    hashes: tuple[str, ...]
    registry_generation: int


@dataclass(frozen=True)
class KVCacheEventObservation:
    """Event fields used to validate, but never discover, physical metadata."""

    group_idx: int
    event_block_size: int | None
    spec_kind: str | None
    sliding_window: int | None


@dataclass(frozen=True)
class VLLMCacheCapabilities:
    supports_group_metadata: bool
    supports_partial_events: bool
    hash_config_field: str


def get_vllm_cache_capabilities(source_vllm_version: str | None) -> VLLMCacheCapabilities:
    """Return the compatibility tier for a vLLM version string."""
    try:
        parsed = Version(source_vllm_version) if source_vllm_version else None
    except InvalidVersion:
        parsed = None
    if parsed is None:
        return VLLMCacheCapabilities(False, False, "hash_block_size")
    return VLLMCacheCapabilities(
        supports_group_metadata=parsed >= Version("0.22.0"),
        supports_partial_events=parsed >= Version("0.24.0"),
        hash_config_field="prefix_match_unit" if parsed >= Version("0.26.0") else "hash_block_size",
    )


def validate_registry(metadata: KVCacheRegistryMetadata) -> KVCacheRegistryMetadata:
    """Validate all invariants required by the query and event paths."""
    groups = tuple(sorted(metadata.groups, key=lambda item: item.group_idx))
    if not groups:
        raise ValueError("cache registry groups must be non-empty")
    if [group.group_idx for group in groups] != list(range(len(groups))):
        raise ValueError(f"cache-group indices must be contiguous from 0: {groups!r}")
    if metadata.source not in ("engine_core", "legacy_single_group"):
        raise ValueError(f"unknown cache registry source={metadata.source!r}")
    if metadata.scheduler_block_size <= 0 or metadata.hash_block_size <= 0:
        raise ValueError("scheduler/hash block sizes must be positive")
    if metadata.scheduler_block_size % metadata.hash_block_size:
        raise ValueError("scheduler_block_size must be divisible by hash_block_size")
    for group in groups:
        if group.block_size <= 0 or group.block_size % metadata.hash_block_size:
            raise ValueError(
                f"group {group.group_idx} block_size={group.block_size} is not divisible by "
                f"hash_block_size={metadata.hash_block_size}"
            )
        if group.spec_kind not in KNOWN_SPEC_KINDS:
            raise ValueError(f"group {group.group_idx} has unknown spec_kind={group.spec_kind!r}")
        if group.spec_kind == SLIDING_WINDOW:
            if group.sliding_window is None or group.sliding_window <= 0:
                raise ValueError(f"sliding-window group {group.group_idx} requires a positive window")
        elif group.sliding_window is not None:
            raise ValueError(f"non-sliding group {group.group_idx} cannot declare sliding_window")
    if metadata.partial_hash_hits_enabled and not any(metadata.hash_block_size < group.block_size for group in groups):
        raise ValueError("partial hash hits require a hash size finer than a physical group block")
    if metadata.source == "legacy_single_group":
        expected = (KVCacheGroupMetadata(0, metadata.hash_block_size, FULL_ATTENTION, None),)
        if groups != expected or metadata.partial_hash_hits_enabled:
            raise ValueError("legacy registry must be one full-attention group without partial hits")
    if groups != metadata.groups:
        metadata = KVCacheRegistryMetadata(
            groups=groups,
            scheduler_block_size=metadata.scheduler_block_size,
            hash_block_size=metadata.hash_block_size,
            mamba_cache_mode=metadata.mamba_cache_mode,
            partial_hash_hits_enabled=metadata.partial_hash_hits_enabled,
            source_vllm_version=metadata.source_vllm_version,
            source=metadata.source,
        )
    return metadata


def resolve_registry_metadata(
    groups: Sequence[Any],
    *,
    source_vllm_version: str | None,
    prefix_caching_enabled: bool,
    configured_hash_block_size: int | None,
    mamba_cache_mode: str | None,
    partial_event_capable: bool,
) -> KVCacheRegistryMetadata:
    """Mirror vLLM's DP=1/context-parallel=1 cache block-size resolver."""
    normalized = tuple(
        sorted((KVCacheGroupMetadata.from_raw(item) for item in groups), key=lambda item: item.group_idx)
    )
    mamba_cache_mode = _normalize_mamba_cache_mode(mamba_cache_mode)
    if not normalized:
        raise ValueError("EngineCore returned empty cache-group metadata")
    group_sizes = [group.block_size for group in normalized]
    if len(normalized) == 1:
        scheduler_block_size = hash_block_size = group_sizes[0]
    else:
        scheduler_block_size = lcm(*group_sizes)
        has_non_align_mamba = any(group.spec_kind == MAMBA for group in normalized) and mamba_cache_mode != "align"
        if not prefix_caching_enabled or has_non_align_mamba:
            hash_block_size = scheduler_block_size
        else:
            hash_block_size = configured_hash_block_size or gcd(*group_sizes)
    if hash_block_size <= 0 or any(size % hash_block_size for size in group_sizes):
        raise ValueError(
            f"invalid hash_block_size={hash_block_size}; group block sizes must be divisible: {group_sizes}"
        )
    all_specs_known = all(group.spec_kind in KNOWN_SPEC_KINDS for group in normalized)
    partial_enabled = (
        partial_event_capable
        and all_specs_known
        and hash_block_size < max(group_sizes)
        and prefix_caching_enabled
        and not (any(group.spec_kind == MAMBA for group in normalized) and mamba_cache_mode != "align")
    )
    return validate_registry(
        KVCacheRegistryMetadata(
            groups=normalized,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            mamba_cache_mode=mamba_cache_mode,
            partial_hash_hits_enabled=partial_enabled,
            source_vllm_version=source_vllm_version,
            source="engine_core",
        )
    )


def _normalize_mamba_cache_mode(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    return normalized
