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

"""vLLM KV event decoder with registry-aware token-only hash reconstruction."""

from __future__ import annotations

from typing import Any

import msgpack

from ....collectors.decoder import Decoder, KVCacheUpdate
from ....collectors.decoder.vllm.kv_event import KVCacheEvent
from ....logging import get_router_logger
from ....types import FULL_ATTENTION, KVCacheEventObservation, KVCacheRegistryMetadata, Layer
from ....utils.hash import compute_hash

logger = get_router_logger("vllm-kv")

RemoteBlockKey = tuple[str, int | None, str]
OwnedRemoteBlockKey = tuple[str, int | None, int, str]
WarningScope = tuple[str, int | None, int]


class VLLMKVDecoder(Decoder):
    """Decode vLLM cache events into group-isolated local prefix hashes."""

    _MEDIUM_TO_LAYER: dict[str, Layer] = {"GPU": Layer.GPU, "cpu": Layer.CPU}

    def __init__(self) -> None:
        # Parent lookup intentionally omits group_idx: vLLM's chained token hash
        # is common across groups. Ownership remains group-scoped below.
        self.remote_to_local_block_hash: dict[RemoteBlockKey, str] = {}
        self._owned_remote_to_local: dict[OwnedRemoteBlockKey, str] = {}
        self._registry: KVCacheRegistryMetadata | None = None
        self._warned_extra_key_scopes: set[WarningScope] = set()

    def set_cache_group_registry(self, registry: KVCacheRegistryMetadata | None) -> None:
        self._registry = registry

    def decode(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | None:
        if isinstance(raw_data, str):
            logger.debug("VLLMKVDecoder received string data, expected bytes - skipping")
            return None
        try:
            raw = msgpack.unpackb(raw_data, raw=False)
            if not isinstance(raw, list) or not raw:
                logger.warning(f"Unexpected msgpack format from node {node_id} (type={type(raw).__name__})")
                return None
            event_payloads = raw if isinstance(raw[0], list) else [raw]
            update = KVCacheUpdate(node_id=node_id)
            for payload in event_payloads:
                events = KVCacheEvent.from_raw(payload, default_node_id=node_id)
                for event in self._order_events(events):
                    if event.event_type == "stored":
                        self._on_block_stored(event, update)
                    elif event.event_type == "removed":
                        self._on_block_removed(event, update)
                    elif event.event_type == "clear":
                        update.clear()
                    else:
                        raise ValueError(f"Unknown event.event_type {event.event_type}.")
            return update
        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            preview = bytes(raw_data[:32]).hex() if isinstance(raw_data, bytes | bytearray) else str(raw_data)[:64]
            logger.warning(
                f"Failed to decode msgpack payload from node {node_id}: {exc!r} (len={len(raw_data)}, head={preview})"
            )
            return None

    @staticmethod
    def _order_events(events: list[KVCacheEvent]) -> list[KVCacheEvent]:
        """Process dense store chains before sparse group checkpoints.

        vLLM can emit one BlockStored per cache group for the same token span.
        Full-attention groups normally carry every block hash, while Mamba
        groups may carry only the latest state checkpoint. The dense event must
        establish the shared remote-to-local mapping before sparse groups reuse
        it. Non-store events remain ordering barriers.
        """
        ordered: list[KVCacheEvent] = []
        stored_run: list[KVCacheEvent] = []

        def flush_stored_run() -> None:
            stored_run.sort(
                key=lambda event: (
                    len(event.block_hashes) != len(event.token_ids or ()),
                    event.kv_cache_spec_kind != FULL_ATTENTION,
                )
            )
            ordered.extend(stored_run)
            stored_run.clear()

        for event in events:
            if event.event_type == "stored":
                stored_run.append(event)
            else:
                flush_stored_run()
                ordered.append(event)
        flush_stored_run()
        return ordered

    @staticmethod
    def _remote_block_key(event: KVCacheEvent, block_hash: str) -> RemoteBlockKey:
        return event.node_id, event.data_parallel_rank, block_hash

    @staticmethod
    def _owned_remote_block_key(event: KVCacheEvent, block_hash: str) -> OwnedRemoteBlockKey:
        return (
            event.node_id,
            event.data_parallel_rank,
            event.group_idx if event.group_idx is not None else 0,
            block_hash,
        )

    @classmethod
    def _medium_to_layer(cls, medium: str | None) -> Layer:
        return cls._MEDIUM_TO_LAYER.get(medium, Layer.GPU)

    @staticmethod
    def _has_extra_keys(value: Any) -> bool:
        return value is not None and value != [] and value != () and value != {}

    def _block_extra_keys(self, event: KVCacheEvent, index: int) -> Any:
        if not event.extra_keys or index >= len(event.extra_keys):
            return None
        return event.extra_keys[index]

    def _warn_extra_keys_once(self, event: KVCacheEvent, group_idx: int) -> None:
        scope = (event.node_id, event.data_parallel_rank, group_idx)
        if scope in self._warned_extra_key_scopes:
            return
        self._warned_extra_key_scopes.add(scope)
        logger.warning(
            "KV cache event contains extra_keys; multimodal, LoRA, cache salt, "
            "or prompt embeddings requests will not receive cache-aware routing benefit "
            f"for node={event.node_id}, dp_rank={event.data_parallel_rank}, group_idx={group_idx}"
        )

    def _on_block_stored(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        if event.token_ids is None:
            logger.debug("Stored event has no token_ids - skipping")
            return
        group_idx = event.group_idx if event.group_idx is not None else 0
        layer = self._medium_to_layer(event.medium)
        if layer == Layer.GPU:
            update.observe_group_metadata(
                KVCacheEventObservation(
                    group_idx=group_idx,
                    event_block_size=event.block_size,
                    spec_kind=event.kv_cache_spec_kind,
                    sliding_window=event.kv_cache_spec_sliding_window,
                )
            )

        event_span = event.block_size
        hash_block_size = self._registry.hash_block_size if self._registry is not None else event_span
        if (
            event_span is None
            or hash_block_size is None
            or event_span <= 0
            or hash_block_size <= 0
            or event_span % hash_block_size
        ):
            logger.warning(
                f"Invalid KV event span from node={event.node_id}, group_idx={group_idx}: "
                f"event_block_size={event_span}, hash_block_size={hash_block_size}"
            )
            return

        if len(event.block_hashes) != len(event.token_ids):
            self._on_sparse_block_stored(event, update, group_idx, layer)
            return

        parent_hash = 0
        if event.parent_block_hash is not None:
            parent = self.remote_to_local_block_hash.get(self._remote_block_key(event, event.parent_block_hash))
            if parent is None:
                logger.debug(
                    f"Skipping KV event chain with unknown parent from node={event.node_id}, "
                    f"dp_rank={event.data_parallel_rank}, group_idx={group_idx}"
                )
                return
            parent_hash = int(parent)

        local_hashes: list[str] = []
        chain_excluded = False
        bytes_per_hash = hash_block_size * 4
        for index, (remote_hash, block_bytes) in enumerate(zip(event.block_hashes, event.token_ids, strict=False)):
            if chain_excluded or self._has_extra_keys(self._block_extra_keys(event, index)):
                chain_excluded = True
                self._warn_extra_keys_once(event, group_idx)
                continue
            if len(block_bytes) != event_span * 4:
                logger.warning(f"KV event token span mismatch from node={event.node_id}, group_idx={group_idx}")
                chain_excluded = True
                continue
            for offset in range(0, len(block_bytes), bytes_per_hash):
                parent_hash = compute_hash(parent_hash, block_bytes[offset : offset + bytes_per_hash], seed=0)
            local_hash = str(parent_hash)
            shared_key = self._remote_block_key(event, remote_hash)
            existing = self.remote_to_local_block_hash.get(shared_key)
            if existing is not None and existing != local_hash:
                logger.warning(
                    f"Conflicting remote KV hash mapping from node={event.node_id}, "
                    f"dp_rank={event.data_parallel_rank}; skipping chain"
                )
                chain_excluded = True
                continue
            self.remote_to_local_block_hash[shared_key] = local_hash
            self._owned_remote_to_local[self._owned_remote_block_key(event, remote_hash)] = local_hash
            local_hashes.append(local_hash)

        update.add(layer, local_hashes, group_idx=group_idx)

    def _on_sparse_block_stored(
        self,
        event: KVCacheEvent,
        update: KVCacheUpdate,
        group_idx: int,
        layer: Layer,
    ) -> None:
        """Register sparse group checkpoints through a dense sibling chain."""
        assert event.token_ids is not None
        if len(event.block_hashes) > len(event.token_ids):
            logger.warning(
                f"KV event has more hashes than token blocks from node={event.node_id}, group_idx={group_idx}"
            )
            return
        if (
            event.parent_block_hash is not None
            and self.remote_to_local_block_hash.get(self._remote_block_key(event, event.parent_block_hash)) is None
        ):
            logger.debug(
                f"Skipping sparse KV event chain with unknown parent from node={event.node_id}, "
                f"dp_rank={event.data_parallel_rank}, group_idx={group_idx}"
            )
            return

        resolved: list[tuple[str, str]] = []
        chain_excluded = False
        for index, remote_hash in enumerate(event.block_hashes):
            if chain_excluded or self._has_extra_keys(self._block_extra_keys(event, index)):
                chain_excluded = True
                self._warn_extra_keys_once(event, group_idx)
                continue
            local_hash = self.remote_to_local_block_hash.get(self._remote_block_key(event, remote_hash))
            if local_hash is None:
                logger.debug(
                    f"Skipping sparse KV event without a dense hash mapping from node={event.node_id}, "
                    f"dp_rank={event.data_parallel_rank}, group_idx={group_idx}"
                )
                return
            resolved.append((remote_hash, local_hash))

        for remote_hash, local_hash in resolved:
            self._owned_remote_to_local[self._owned_remote_block_key(event, remote_hash)] = local_hash
        update.add(layer, [local_hash for _, local_hash in resolved], group_idx=group_idx)

    def _on_block_removed(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        group_idx = event.group_idx if event.group_idx is not None else 0
        local_hashes = [
            local_hash
            for block_hash in event.block_hashes
            if (local_hash := self._owned_remote_to_local.pop(self._owned_remote_block_key(event, block_hash), None))
            is not None
        ]
        update.remove(self._medium_to_layer(event.medium), local_hashes, group_idx=group_idx)
