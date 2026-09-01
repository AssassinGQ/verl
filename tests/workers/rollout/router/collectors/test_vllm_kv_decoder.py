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

"""Unit tests for VLLMKVDecoder layer bucketing (mixed-medium frames)."""

from __future__ import annotations

import msgpack
import pytest

from verl.workers.rollout.router.kvcaware.collectors.decoder.vllm.kv import VLLMKVDecoder
from verl.workers.rollout.router.kvcaware.types import (
    KVCacheGroupMetadata,
    KVCacheRegistryMetadata,
    Layer,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _stored_event(block_hash, parent, token_ids, block_size, medium):
    """A stored event entry: [tag, block_hashes, parent, token_ids, block_size, <unused>, medium]."""
    return ["stored", [block_hash], parent, token_ids, block_size, None, medium]


def _dict_stored_event(block_hash, token_ids, *, group_idx, parent=None, medium="GPU"):
    return {
        "type": "BlockStored",
        "block_hashes": [block_hash],
        "parent_block_hash": parent,
        "token_ids": token_ids,
        "block_size": 2,
        "medium": medium,
        "group_idx": group_idx,
    }


def _dict_removed_event(block_hash, *, group_idx, medium="GPU"):
    return {
        "type": "BlockRemoved",
        "block_hashes": [block_hash],
        "medium": medium,
        "group_idx": group_idx,
    }


def _decode_event(decoder, event, node_id, data_parallel_rank):
    payload = [0, [event], data_parallel_rank]
    return decoder.decode(msgpack.packb(payload), node_id)


def test_mixed_medium_frame_buckets_per_layer():
    """A single frame with a GPU and a cpu BlockStored keeps layers distinct.

    Regression: the old scalar medium_add aggregation let the later event's
    medium overwrite the earlier one, so the whole batch was written under one
    layer. Per-layer dict bucketing must keep each event's blocks in its layer.
    """
    decoder = VLLMKVDecoder()
    payload = [
        1234567890,  # timestamp
        [
            _stored_event("rh_gpu", None, [1, 2], 2, "GPU"),
            _stored_event("rh_cpu", None, [3, 4], 2, "cpu"),
        ],
    ]

    update = decoder.decode(msgpack.packb(payload), "node1")

    assert update is not None
    # Both layers present — no cross-layer overwrite.
    assert (Layer.GPU, 0) in update.add_blocks
    assert (Layer.CPU, 0) in update.add_blocks
    assert len(update.add_blocks[(Layer.GPU, 0)]) == 1
    assert len(update.add_blocks[(Layer.CPU, 0)]) == 1
    # Different token ids → different local hashes per layer.
    assert update.add_blocks[(Layer.GPU, 0)] != update.add_blocks[(Layer.CPU, 0)]


def test_none_medium_defaults_to_gpu():
    """Older vLLM events without medium default to the GPU layer."""
    decoder = VLLMKVDecoder()
    payload = [0, [_stored_event("rh", None, [1, 2], 2, None)]]

    update = decoder.decode(msgpack.packb(payload), "node1")

    assert update is not None
    assert (Layer.GPU, 0) in update.add_blocks
    assert (Layer.CPU, 0) not in update.add_blocks
    assert set(decoder.remote_to_local_block_hash) == {("node1", None, "rh")}


def test_clear_event_sets_clear_all():
    """An AllBlocksCleared event marks the update for a full replica clear."""
    decoder = VLLMKVDecoder()
    payload = [0, [["clear"]]]

    update = decoder.decode(msgpack.packb(payload), "node1")

    assert update is not None
    assert update.clear_all is True


def test_decode_failure_surfaces_exception_not_swallowed():
    """A malformed payload returns None and logs the real error.

    Regression: the ``except`` used a non-f-string with an unbound ``{exc}``,
    so the warning rendered the literal text ``"{exc}"`` and the actual decode
    error was silently lost.
    """
    from loguru import logger as loguru_logger

    decoder = VLLMKVDecoder()
    # 0xc1 is a reserved/invalid msgpack byte → unpackb raises UnpackException,
    # exercising the failed-to-decode branch (not the unexpected-format branch).
    garbage = b"\xc1\xc1\xc1"
    msgs: list[str] = []
    sink_id = loguru_logger.add(msgs.append, level="WARNING", format="{message}")
    try:
        update = decoder.decode(garbage, "node1")
    finally:
        loguru_logger.remove(sink_id)

    assert update is None
    text = "\n".join(msgs)
    assert "{exc}" not in text  # placeholder must be gone
    assert "node1" in text  # node_id must interpolate
    assert "len=" in text and "head=c1c1c1" in text  # diagnostic preview present


def test_remote_parent_mapping_is_shared_across_groups_but_scoped_by_replica_and_dp():
    decoder = VLLMKVDecoder()
    scopes = [
        ("replica-a", 0, 0, [1, 2]),
        ("replica-a", 0, 1, [1, 2]),
        ("replica-b", 0, 0, [3, 4]),
        ("replica-a", 1, 0, [5, 6]),
    ]
    local_hashes = {}

    for node_id, dp_rank, group_idx, token_ids in scopes:
        update = _decode_event(
            decoder,
            _dict_stored_event("shared-remote-hash", token_ids, group_idx=group_idx),
            node_id,
            dp_rank,
        )
        assert update is not None
        local_hashes[(node_id, dp_rank, group_idx)] = update.add_blocks[(Layer.GPU, group_idx)][0]

    assert local_hashes[("replica-a", 0, 0)] == local_hashes[("replica-a", 0, 1)]
    assert len(decoder.remote_to_local_block_hash) == 3
    assert set(decoder.remote_to_local_block_hash) == {
        ("replica-a", 0, "shared-remote-hash"),
        ("replica-b", 0, "shared-remote-hash"),
        ("replica-a", 1, "shared-remote-hash"),
    }

    for group_idx in (0, 1):
        update = _decode_event(
            decoder,
            _dict_removed_event("shared-remote-hash", group_idx=group_idx),
            "replica-a",
            0,
        )
        assert update is not None
        assert update.remove_blocks[(Layer.GPU, group_idx)] == [local_hashes[("replica-a", 0, group_idx)]]
    # Parent identity remains reusable for another group after one owner removes.
    assert ("replica-a", 0, "shared-remote-hash") in decoder.remote_to_local_block_hash


def test_parent_lookup_can_cross_group_and_unknown_parent_is_skipped():
    decoder = VLLMKVDecoder()
    parent = _decode_event(
        decoder,
        _dict_stored_event("parent", [1, 2], group_idx=0),
        "replica-a",
        0,
    )
    child = _decode_event(
        decoder,
        _dict_stored_event("child", [3, 4], group_idx=1, parent="parent"),
        "replica-a",
        0,
    )
    unknown = _decode_event(
        decoder,
        _dict_stored_event("unknown-child", [5, 6], group_idx=1, parent="missing"),
        "replica-a",
        0,
    )

    assert parent is not None and parent.add_blocks[(Layer.GPU, 0)]
    assert child is not None and child.add_blocks[(Layer.GPU, 1)]
    assert unknown is not None
    assert unknown.add_blocks.get((Layer.GPU, 1), []) == []


def test_legacy_group_scope_is_used_for_store_and_remove():
    decoder = VLLMKVDecoder()
    stored_payload = [
        0,
        [["stored", ["legacy-hash"], None, [1, 2], 2, None, "GPU", None, [None], 5]],
    ]
    removed_payload = [0, [["removed", ["legacy-hash"], "GPU", 5]]]

    stored_update = decoder.decode(msgpack.packb(stored_payload), "replica-a")
    removed_update = decoder.decode(msgpack.packb(removed_payload), "replica-a")

    assert stored_update is not None
    assert removed_update is not None
    assert removed_update.remove_blocks[(Layer.GPU, 5)] == stored_update.add_blocks[(Layer.GPU, 5)]


def test_legacy_and_current_events_compute_the_same_local_hash():
    legacy_decoder = VLLMKVDecoder()
    current_decoder = VLLMKVDecoder()

    legacy_update = legacy_decoder.decode(
        msgpack.packb([0, [_stored_event("legacy-hash", None, [1, 2], 2, "GPU")]]),
        "replica-a",
    )
    current_update = _decode_event(
        current_decoder,
        _dict_stored_event("current-hash", [1, 2], group_idx=0),
        "replica-a",
        0,
    )

    assert legacy_update is not None
    assert current_update is not None
    assert legacy_update.add_blocks[(Layer.GPU, 0)] == current_update.add_blocks[(Layer.GPU, 0)]


def test_stored_event_propagates_group_metadata():
    decoder = VLLMKVDecoder()
    event = {
        **_dict_stored_event("hash", [1, 2], group_idx=3),
        "kv_cache_spec_kind": "sliding_window",
        "kv_cache_spec_sliding_window": 4096,
    }

    update = _decode_event(decoder, event, "replica-a", 0)

    assert update is not None
    assert len(update.observed_group_metadata) == 1
    metadata = update.observed_group_metadata[0]
    assert (metadata.group_idx, metadata.event_block_size) == (3, 2)
    assert (metadata.spec_kind, metadata.sliding_window) == ("sliding_window", 4096)


def test_registry_fine_hashes_full_and_partial_events() -> None:
    registry = KVCacheRegistryMetadata(
        groups=(KVCacheGroupMetadata(0, 4, "full_attention", None),),
        scheduler_block_size=4,
        hash_block_size=2,
        mamba_cache_mode=None,
        partial_hash_hits_enabled=True,
        source_vllm_version="0.26.0",
        source="engine_core",
    )
    decoder = VLLMKVDecoder()
    decoder.set_cache_group_registry(registry)

    full = {
        **_dict_stored_event("full", [1, 2, 3, 4], group_idx=0),
        "block_size": 4,
    }
    partial = _dict_stored_event("partial", [5, 6], group_idx=0, parent="full")
    full_update = _decode_event(decoder, full, "replica-a", 0)
    partial_update = _decode_event(decoder, partial, "replica-a", 0)

    assert full_update is not None
    assert partial_update is not None
    assert len(full_update.add_blocks[(Layer.GPU, 0)]) == 1
    assert len(partial_update.add_blocks[(Layer.GPU, 0)]) == 1
    assert full_update.observed_group_metadata[0].event_block_size == 4
    assert partial_update.observed_group_metadata[0].event_block_size == 2


def test_extra_keys_exclude_chain_warn_once_and_do_not_leak() -> None:
    from loguru import logger as loguru_logger

    decoder = VLLMKVDecoder()
    secret = "secret-cache-salt"
    excluded_root = {
        **_dict_stored_event("excluded", [1, 2], group_idx=0),
        "extra_keys": [secret],
    }
    descendant = _dict_stored_event("descendant", [3, 4], group_idx=0, parent="excluded")
    fresh_root = _dict_stored_event("fresh", [5, 6], group_idx=0)
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        first = _decode_event(decoder, excluded_root, "replica-a", 0)
        replay = _decode_event(decoder, excluded_root, "replica-a", 0)
        child = _decode_event(decoder, descendant, "replica-a", 0)
        fresh = _decode_event(decoder, fresh_root, "replica-a", 0)
    finally:
        loguru_logger.remove(sink_id)

    assert first is not None and first.add_blocks.get((Layer.GPU, 0), []) == []
    assert replay is not None and replay.add_blocks.get((Layer.GPU, 0), []) == []
    assert child is not None and child.add_blocks.get((Layer.GPU, 0), []) == []
    assert fresh is not None and fresh.add_blocks[(Layer.GPU, 0)]
    warnings = [message for message in messages if "extra_keys" in message]
    assert len(warnings) == 1
    assert secret not in warnings[0]


def test_sparse_mamba_checkpoint_reuses_dense_full_chain_in_same_batch():
    decoder = VLLMKVDecoder()
    sparse_mamba = {
        **_dict_stored_event("tail", [1, 2, 3, 4], group_idx=0),
        "kv_cache_spec_kind": "mamba",
    }
    dense_full = {
        **_dict_stored_event("unused", [1, 2, 3, 4], group_idx=1),
        "block_hashes": ["head", "tail"],
        "kv_cache_spec_kind": "full_attention",
    }
    payload = [0, [sparse_mamba, dense_full], 0]

    update = decoder.decode(msgpack.packb(payload), "replica-a")

    assert update is not None
    full_hashes = update.add_blocks[(Layer.GPU, 1)]
    assert len(full_hashes) == 2
    assert update.add_blocks[(Layer.GPU, 0)] == [full_hashes[-1]]


def test_sparse_checkpoint_without_dense_mapping_is_skipped():
    decoder = VLLMKVDecoder()
    sparse = {
        **_dict_stored_event("tail", [1, 2, 3, 4], group_idx=0),
        "kv_cache_spec_kind": "mamba",
    }

    update = _decode_event(decoder, sparse, "replica-a", 0)

    assert update is not None
    assert update.add_blocks.get((Layer.GPU, 0), []) == []
    assert ("replica-a", 0, "tail") not in decoder.remote_to_local_block_hash
