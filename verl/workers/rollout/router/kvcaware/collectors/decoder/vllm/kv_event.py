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

"""
KVCacheEvent — standardized KV cache event data structure.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVCacheEvent:
    """Standardized KV cache event — normalized from backend-specific ZMQ payloads.

    Attributes:
        event_type: ``"stored"`` / ``"removed"`` / ``"clear"``.
        node_id: Source endpoint from the ZMQ connection.
        block_hashes: Block hashes involved in the event (list from msgpack).
        parent_block_hash: Parent block hash — single value shared by all
                           block_hashes in a BlockStored event.
        token_ids: Pre-chopped block token bytes (``list[bytes]``, only present
                   in BlockStored events). Each element is one full event span
                   encoded as uint32 big-endian (4 bytes per token). Sparse
                   groups may carry fewer block hashes than token chunks.
        block_size: Block size (only present in BlockStored events).
        group_idx: KV cache group that owns the block.
        extra_keys: Additional block-identity keys, for example multimodal inputs.
        kv_cache_spec_kind: KV cache specification kind for the group.
        kv_cache_spec_sliding_window: Sliding-window size when applicable.
        locality: Optional cache locality reported by vLLM.
        data_parallel_rank: vLLM data-parallel rank from the event-batch envelope.
    """

    event_type: str
    node_id: str
    block_hashes: list[str]
    parent_block_hash: str | None
    token_ids: list[bytes] | None
    block_size: int | None
    medium: str | None = None
    group_idx: int | None = None
    extra_keys: list[Any] | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    data_parallel_rank: int | None = None

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw_data: Any, default_node_id: str | None = None) -> list[KVCacheEvent]:
        """Parse a msgpack-decoded vLLM event batch.

        Current vLLM batches use ``[timestamp, [event_dict, ...],\n        data_parallel_rank]``.
        Legacy batches use positional event lists and may omit ``data_parallel_rank``.

        Args:
            raw_data: Msgpack-decoded event batch.
            default_node_id: Source endpoint used as the event node_id.

        Returns:
            A list of normalized events. Malformed or unknown events are skipped.

        Raises:
            ValueError: If the top-level format is invalid.
        """
        if not isinstance(raw_data, list | tuple) or len(raw_data) < 2:
            raise ValueError(f"Expected list with >= 2 elements, got {type(raw_data)}")

        event_list = raw_data[1]
        if not isinstance(event_list, list | tuple):
            raise ValueError(f"Expected raw_data[1] as event list, got {type(event_list)}")

        data_parallel_rank = cls._opt_int(raw_data, 2)
        results: list[KVCacheEvent] = []
        for event_entry in event_list:
            try:
                node_id = default_node_id or ""
                if isinstance(event_entry, Mapping):
                    event_type = cls._resolve_event_type(event_entry.get("type"))
                    event = cls._build_event(event_type, event_entry, node_id, data_parallel_rank)
                elif isinstance(event_entry, list | tuple) and event_entry:
                    event_type = cls._resolve_event_type(event_entry[0])
                    event = cls._build_legacy_event(
                        event_type,
                        event_entry[1:],
                        node_id,
                        data_parallel_rank,
                    )
                else:
                    continue

                if event is not None:
                    results.append(event)
            except (KeyError, ValueError, TypeError, IndexError):
                continue

        return results

    # ── Build helpers ────────────────────────────────────────────────────

    @classmethod
    def _build_legacy_event(
        cls,
        event_type: str,
        fields: list | tuple,
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent | None:
        """Build an event from the legacy positional vLLM format."""
        if event_type == "stored":
            return cls._build_legacy_block_stored(fields, node_id, data_parallel_rank)
        elif event_type == "removed":
            return cls._build_legacy_block_removed(fields, node_id, data_parallel_rank)
        elif event_type == "clear":
            return cls._build_all_blocks_cleared(node_id, data_parallel_rank)
        return None

    @classmethod
    def _build_event(
        cls,
        event_type: str,
        fields: Mapping[str, Any],
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent | None:
        """Build an event from the current dictionary vLLM format."""
        if event_type == "stored":
            return cls._build_block_stored(fields, node_id, data_parallel_rank)
        elif event_type == "removed":
            return cls._build_block_removed(fields, node_id, data_parallel_rank)
        elif event_type == "clear":
            return cls._build_all_blocks_cleared(node_id, data_parallel_rank)
        return None

    @staticmethod
    def _opt_str(fields: list | tuple, idx: int) -> str | None:
        """Return a positional field as str, or None when absent."""
        if idx >= len(fields) or fields[idx] is None:
            return None
        return str(fields[idx])

    @staticmethod
    def _opt_int(fields: list | tuple, idx: int) -> int | None:
        """Return a positional field as int, or None when absent."""
        if idx >= len(fields) or fields[idx] is None:
            return None
        return int(fields[idx])

    @staticmethod
    def _mapping_opt_str(fields: Mapping[str, Any], key: str) -> str | None:
        """Return a mapping field as str, or None when absent."""
        value = fields.get(key)
        return str(value) if value is not None else None

    @staticmethod
    def _mapping_opt_int(fields: Mapping[str, Any], key: str) -> int | None:
        """Return a mapping field as int, or None when absent."""
        value = fields.get(key)
        return int(value) if value is not None else None

    @classmethod
    def _build_legacy_block_stored(
        cls,
        fields: list | tuple,
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent:
        """Build a BlockStored event from its legacy positional fields."""
        if len(fields) < 4:
            raise ValueError(f"BlockStored needs >= 4 fields, got {len(fields)}")

        block_hashes = [str(block_hash) for block_hash in fields[0]]
        parent_block_hash = str(fields[1]) if fields[1] is not None else None
        raw_token_ids = list(fields[2]) if fields[2] is not None else None
        block_size = int(fields[3])
        token_ids = _convert_token_ids(raw_token_ids, block_size) if raw_token_ids is not None else None
        raw_extra_keys = fields[7] if len(fields) > 7 else None

        return cls(
            event_type="stored",
            node_id=node_id,
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            medium=cls._opt_str(fields, 5),
            data_parallel_rank=data_parallel_rank,
            group_idx=cls._opt_int(fields, 8),
            extra_keys=list(raw_extra_keys) if raw_extra_keys is not None else None,
        )

    @classmethod
    def _build_legacy_block_removed(
        cls,
        fields: list | tuple,
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent:
        """Build a BlockRemoved event from its legacy positional fields."""
        return cls(
            event_type="removed",
            node_id=node_id,
            block_hashes=[str(block_hash) for block_hash in fields[0]],
            parent_block_hash=None,
            token_ids=None,
            block_size=None,
            medium=cls._opt_str(fields, 1),
            data_parallel_rank=data_parallel_rank,
            group_idx=cls._opt_int(fields, 2),
        )

    @classmethod
    def _build_block_stored(
        cls,
        fields: Mapping[str, Any],
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent:
        """Build a BlockStored event from the current dictionary format."""
        raw_token_ids = fields.get("token_ids")
        block_size = int(fields["block_size"])
        raw_extra_keys = fields.get("extra_keys")

        return cls(
            event_type="stored",
            node_id=node_id,
            block_hashes=[str(block_hash) for block_hash in fields["block_hashes"]],
            parent_block_hash=cls._mapping_opt_str(fields, "parent_block_hash"),
            token_ids=(_convert_token_ids(list(raw_token_ids), block_size) if raw_token_ids is not None else None),
            block_size=block_size,
            medium=cls._mapping_opt_str(fields, "medium"),
            data_parallel_rank=data_parallel_rank,
            group_idx=cls._mapping_opt_int(fields, "group_idx"),
            extra_keys=list(raw_extra_keys) if raw_extra_keys is not None else None,
            kv_cache_spec_kind=cls._mapping_opt_str(fields, "kv_cache_spec_kind"),
            kv_cache_spec_sliding_window=cls._mapping_opt_int(fields, "kv_cache_spec_sliding_window"),
            locality=cls._mapping_opt_str(fields, "locality"),
        )

    @classmethod
    def _build_block_removed(
        cls,
        fields: Mapping[str, Any],
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent:
        """Build a BlockRemoved event from the current dictionary format."""
        return cls(
            event_type="removed",
            node_id=node_id,
            block_hashes=[str(block_hash) for block_hash in fields["block_hashes"]],
            parent_block_hash=None,
            token_ids=None,
            block_size=None,
            medium=cls._mapping_opt_str(fields, "medium"),
            data_parallel_rank=data_parallel_rank,
            group_idx=cls._mapping_opt_int(fields, "group_idx"),
            locality=cls._mapping_opt_str(fields, "locality"),
        )

    @classmethod
    def _build_all_blocks_cleared(
        cls,
        node_id: str,
        data_parallel_rank: int | None,
    ) -> KVCacheEvent:
        """Build an AllBlocksCleared event."""
        return cls(
            event_type="clear",
            node_id=node_id,
            block_hashes=[],
            parent_block_hash=None,
            token_ids=None,
            block_size=None,
            data_parallel_rank=data_parallel_rank,
        )

    # ── Tag resolution ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_event_type(tag: Any) -> str:
        """Map msgspec struct tag to canonical event type string.

        vLLM msgspec uses numeric tags:
          - 0 → BlockStored
          - 1 → BlockRemoved
          - 2 → AllBlocksCleared

        String tags are also supported.
        """
        if isinstance(tag, int):
            return {0: "stored", 1: "removed", 2: "clear"}.get(tag, f"unknown_{tag}")

        tag_str = str(tag).lower()
        if "stored" in tag_str:
            return "stored"
        elif "removed" in tag_str or "evicted" in tag_str:
            return "removed"
        elif "clear" in tag_str:
            return "clear"
        return f"unknown_{tag}"

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def is_store(self) -> bool:
        """True if this is a block-stored event."""
        return "stored" in self.event_type.lower()

    @property
    def is_remove(self) -> bool:
        """True if this is a block-removed event."""
        return any(k in self.event_type.lower() for k in ("removed", "evicted"))

    @property
    def is_clear(self) -> bool:
        """True if this is an all-blocks-cleared event."""
        return "clear" in self.event_type.lower()


# ── Token ID conversion ────────────────────────────────────────────────────


def _convert_token_ids(raw_ids: list[int], block_size: int) -> list[bytes]:
    """Convert raw token IDs to list of block-sized uint32 big-endian byte chunks.

    Each block of token IDs is encoded as uint32 big-endian (4 bytes per token),
    matching aibrix's ``convertTokenIDs`` / ``tokenIDsToBytes``.

    Args:
        raw_ids: Raw token IDs as ``list[int]``.
        block_size: Number of tokens per block (must be > 0).

    Returns:
        List of bytes objects, one per full block.

    Raises:
        ValueError: If ``block_size <= 0``, ``len(raw_ids)`` is not
            divisible by ``block_size``, or any token ID is out of the
            uint32 range (``[0, 2**32)``).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if len(raw_ids) % block_size != 0:
        raise ValueError(f"token_ids len={len(raw_ids)} not divisible by block_size={block_size}")
    num_blocks = len(raw_ids) // block_size
    result: list[bytes] = []
    try:
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            result.append(struct.pack(f">{block_size}I", *raw_ids[start:end]))
    except struct.error as e:
        raise ValueError(f"Invalid token ID value for struct packing: {e}") from e
    return result
