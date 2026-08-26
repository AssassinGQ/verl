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

"""Unit tests for normalizing vLLM KV-cache event wire formats."""

from __future__ import annotations

import struct

import pytest

from verl.workers.rollout.router.kvcaware.collectors.decoder.vllm.kv_event import KVCacheEvent

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def test_current_dict_block_stored_parses_metadata_and_stringifies_hashes():
    payload = [
        1234.5,
        [
            {
                "type": "BlockStored",
                "block_hashes": [2**64 - 1, 123],
                "parent_block_hash": 456,
                "token_ids": [1, 2, 3, 4],
                "block_size": 2,
                "lora_id": None,
                "medium": "GPU",
                "lora_name": None,
                "extra_keys": [None, "image-a"],
                "group_idx": 3,
                "kv_cache_spec_kind": "sliding_window",
                "kv_cache_spec_sliding_window": 512,
                "locality": "LOCAL",
                "future_field": "ignored",
            }
        ],
        7,
    ]

    events = KVCacheEvent.from_raw(payload, default_node_id="replica-a")

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "stored"
    assert event.node_id == "replica-a"
    assert event.block_hashes == [str(2**64 - 1), "123"]
    assert event.parent_block_hash == "456"
    assert event.token_ids == [struct.pack(">2I", 1, 2), struct.pack(">2I", 3, 4)]
    assert event.block_size == 2
    assert event.medium == "GPU"
    assert event.data_parallel_rank == 7
    assert event.group_idx == 3
    assert event.extra_keys == [None, "image-a"]
    assert event.kv_cache_spec_kind == "sliding_window"
    assert event.kv_cache_spec_sliding_window == 512
    assert event.locality == "LOCAL"


def test_mooncake_cpu_block_stored_allows_sparse_metadata():
    payload = [
        1234.5,
        [
            {
                "type": "BlockStored",
                "block_hashes": [987],
                "parent_block_hash": None,
                "token_ids": [5, 6],
                "block_size": 2,
                "lora_id": None,
                "medium": "cpu",
                "lora_name": None,
                "group_idx": 0,
            }
        ],
        0,
    ]

    event = KVCacheEvent.from_raw(payload, default_node_id="replica-a")[0]

    assert event.block_hashes == ["987"]
    assert event.medium == "cpu"
    assert event.data_parallel_rank == 0
    assert event.group_idx == 0
    assert event.extra_keys is None
    assert event.kv_cache_spec_kind is None
    assert event.kv_cache_spec_sliding_window is None
    assert event.locality is None


def test_current_dict_removed_and_clear_parse_batch_rank():
    payload = [
        1234.5,
        [
            {
                "type": "BlockRemoved",
                "block_hashes": [111, 222],
                "medium": "GPU",
                "group_idx": 2,
                "locality": "REMOTE",
                "future_field": "ignored",
            },
            {"type": "AllBlocksCleared"},
        ],
        4,
    ]

    removed, cleared = KVCacheEvent.from_raw(payload, default_node_id="replica-b")

    assert removed.event_type == "removed"
    assert removed.block_hashes == ["111", "222"]
    assert removed.medium == "GPU"
    assert removed.data_parallel_rank == 4
    assert removed.group_idx == 2
    assert removed.locality == "REMOTE"

    assert cleared.event_type == "clear"
    assert cleared.data_parallel_rank == 4
    assert cleared.block_hashes == []


def test_legacy_positional_events_remain_supported():
    payload = [
        1234.5,
        [
            ["stored", [321], None, [1, 2], 2, None, "GPU", None, [None], 5],
            ["removed", [321], "GPU", 5],
            ["clear"],
        ],
    ]

    stored, removed, cleared = KVCacheEvent.from_raw(payload, default_node_id="legacy-node")

    assert stored.event_type == "stored"
    assert stored.block_hashes == ["321"]
    assert stored.medium == "GPU"
    assert stored.data_parallel_rank is None
    assert stored.group_idx == 5
    assert stored.extra_keys == [None]
    assert stored.token_ids == [struct.pack(">2I", 1, 2)]

    assert removed.event_type == "removed"
    assert removed.block_hashes == ["321"]
    assert removed.group_idx == 5
    assert removed.data_parallel_rank is None
    assert cleared.event_type == "clear"


def test_unknown_and_malformed_dict_events_are_skipped():
    payload = [
        1234.5,
        [
            {"type": "FutureEvent", "future_field": "ignored"},
            {"type": "BlockStored", "block_hashes": [1]},
        ],
        0,
    ]

    assert KVCacheEvent.from_raw(payload, default_node_id="replica-a") == []
