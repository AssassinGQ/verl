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

"""Tests for vLLMHttpServer cache-group metadata discovery."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.workers.rollout.vllm_rollout import vllm_async_server as server_module
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


def _make_server() -> vLLMHttpServer:
    server = object.__new__(vLLMHttpServer)
    server._kv_cache_group_metadata = None
    server._kv_cache_registry_result = None
    return server


def _vllm_config(**cache_overrides):
    cache_config = {
        "enable_prefix_caching": True,
        "hash_block_size": None,
        "prefix_match_unit": None,
        "mamba_cache_mode": None,
        **cache_overrides,
    }
    return SimpleNamespace(
        cache_config=SimpleNamespace(**cache_config),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def test_fetch_kv_cache_group_metadata_uses_engine_core_utility(monkeypatch) -> None:
    metadata = [
        {
            "group_idx": 0,
            "block_size": 16,
            "kind": "full_attention",
            "sliding_window": None,
        },
        {
            "group_idx": 1,
            "block_size": 16,
            "kind": "sliding_window",
            "sliding_window": 1024,
        },
    ]
    monkeypatch.setattr(server_module, "_VLLM_VERSION", server_module.version.parse("0.26.0"))
    call_utility_async = AsyncMock(return_value=metadata)
    engine_client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=call_utility_async))
    server = _make_server()

    asyncio.run(server._fetch_kv_cache_group_metadata(engine_client, _vllm_config(prefix_match_unit=8)))

    call_utility_async.assert_awaited_once_with("get_kv_cache_group_metadata")
    result = server.get_kv_cache_registry_metadata()
    assert result["status"] == "ok"
    assert result["metadata"]["scheduler_block_size"] == 16
    assert result["metadata"]["hash_block_size"] == 8
    assert result["metadata"]["partial_hash_hits_enabled"] is True
    assert result["metadata"]["source"] == "engine_core"
    assert len(server.get_kv_cache_group_metadata()) == 2


def test_fetch_kv_cache_group_metadata_failure_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "_VLLM_VERSION", server_module.version.parse("0.26.0"))
    call_utility_async = AsyncMock(side_effect=RuntimeError("metadata RPC failed"))
    engine_client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=call_utility_async))
    server = _make_server()

    asyncio.run(server._fetch_kv_cache_group_metadata(engine_client, _vllm_config()))

    call_utility_async.assert_awaited_once_with("get_kv_cache_group_metadata")
    assert server.get_kv_cache_group_metadata() is None
    assert server.get_kv_cache_registry_metadata()["status"] == "error"


def test_old_vllm_reports_metadata_rpc_as_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "_VLLM_VERSION", server_module.version.parse("0.21.0"))
    call_utility_async = AsyncMock()
    engine_client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=call_utility_async))
    server = _make_server()

    asyncio.run(server._fetch_kv_cache_group_metadata(engine_client, _vllm_config()))

    call_utility_async.assert_not_awaited()
    assert server.get_kv_cache_registry_metadata()["status"] == "unsupported"
