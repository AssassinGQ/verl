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

"""GPU smoke test for EngineCore registry discovery and prefix hits.

Set VLLM_MODEL at runtime. Optional expectations:
EXPECTED_KV_CACHE_GROUP_COUNT, EXPECTED_KV_CACHE_SPEC_KINDS, and
VLLM_PREFIX_MATCH_UNIT. Set VLLM_TENSOR_PARALLEL_SIZE for models that do not
fit on one GPU.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers import AutoTokenizer

from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.router.kvcaware.balancer import KVCAwareBalancer
from verl.workers.rollout.router.kvcaware.utils.prefix_cache import resolve_prefix_hashes

VLLM_MODEL = os.environ.get("VLLM_MODEL")
EXPECTED_GROUP_COUNT = (
    int(os.environ["EXPECTED_KV_CACHE_GROUP_COUNT"]) if "EXPECTED_KV_CACHE_GROUP_COUNT" in os.environ else None
)
EXPECTED_SPEC_KINDS = set(filter(None, os.environ.get("EXPECTED_KV_CACHE_SPEC_KINDS", "").split(",")))
EVENT_TIMEOUT_S = float(os.environ.get("KV_EVENT_TIMEOUT_S", "30"))
PREFIX_MATCH_UNIT = int(os.environ["VLLM_PREFIX_MATCH_UNIT"]) if "VLLM_PREFIX_MATCH_UNIT" in os.environ else None
TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(not VLLM_MODEL, reason="set VLLM_MODEL to run the cache-registry GPU E2E"),
]


@ray.remote(num_cpus=0)
class _InspectableKVCAwareBalancer(KVCAwareBalancer):
    """Test-only read access to the Balancer's process-local DataStore."""

    def inspect_prefix(self, replica_id: str, prompt_ids: list[int]) -> dict:
        registry = self._store.get_cache_group_registry()
        chain = resolve_prefix_hashes(prompt_ids, None, self._store)
        return {
            "registry_ready": self._store.is_cache_group_registry_ready(),
            "registry": registry.as_dict() if registry is not None else None,
            "hash_count": len(chain.hashes) if chain is not None else 0,
            "gpu_hit": self._store.get_gpu_prefix_hit_rate(replica_id, chain) if chain is not None else 0.0,
        }


def _build_config():
    assert VLLM_MODEL is not None
    config_dir = Path(__file__).resolve().parents[5] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")

    rollout = config.actor_rollout_ref.rollout
    config.trainer.n_gpus_per_node = TENSOR_PARALLEL_SIZE
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = VLLM_MODEL
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.nnodes = 1
    rollout.n_gpus_per_node = TENSOR_PARALLEL_SIZE
    rollout.tensor_model_parallel_size = TENSOR_PARALLEL_SIZE
    rollout.data_parallel_size = 1
    rollout.pipeline_model_parallel_size = 1
    rollout.load_format = "auto"
    rollout.dtype = "bfloat16"
    rollout.max_model_len = 2048
    rollout.max_num_batched_tokens = 2048
    rollout.max_num_seqs = 32
    rollout.gpu_memory_utilization = 0.6
    rollout.enforce_eager = True
    rollout.enable_prefix_caching = True
    rollout.disable_log_stats = False
    rollout.router_strategy = "kvcaware"
    rollout.router_config = OmegaConf.load(config_dir / "rollout" / "router" / "kvcaware.yaml")
    with open_dict(rollout.engine_kwargs.vllm):
        rollout.engine_kwargs.vllm["kv-events-config"] = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
        }
        if PREFIX_MATCH_UNIT is not None:
            rollout.engine_kwargs.vllm["prefix-match-unit"] = PREFIX_MATCH_UNIT
    return config


async def _wait_for_gpu_hit(inspector, replica_id: str, prompt_ids: list[int]) -> dict:
    deadline = time.monotonic() + EVENT_TIMEOUT_S
    snapshot = {}
    while time.monotonic() < deadline:
        snapshot = await inspector.inspect_prefix.remote(replica_id, prompt_ids)
        if snapshot["gpu_hit"] > 0:
            return snapshot
        await asyncio.sleep(0.5)
    return snapshot


@pytest.mark.asyncio
async def test_enginecore_registry_and_group_aware_gpu_hit():
    """Install a complete registry and observe a nonzero repeated-prefix hit."""
    assert VLLM_MODEL is not None
    ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "NCCL_P2P_DISABLE": "1",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VERL_KV_EVENTS_BASE_PORT": os.environ.get("VERL_KV_EVENTS_BASE_PORT", "30000"),
            }
        }
    )

    try:
        config = _build_config()
        manager = await LLMServerManager.create(
            config=config,
            worker_group=None,
            rollout_resource_pool=None,
        )
        assert len(manager.server_handles) == 1

        replica_id = manager.server_addresses[0]
        server_handle = manager.server_handles[0]
        discovery = await server_handle.get_kv_cache_registry_metadata.remote()
        assert discovery["status"] == "ok", discovery
        enginecore_registry = discovery["metadata"]
        group_count = len(enginecore_registry["groups"])
        if EXPECTED_GROUP_COUNT is not None:
            assert group_count == EXPECTED_GROUP_COUNT, enginecore_registry

        router_status = await manager.router_handle.get_status.remote()
        assert router_status["cache_group_registry_ready"] is True, router_status
        assert router_status["cache_group_registry_source"] == "engine_core", router_status
        assert router_status["cache_group_count"] == group_count, router_status
        assert router_status["cache_group_hash_block_size"] > 0, router_status
        assert router_status["cache_group_scheduler_block_size"] > 0, router_status

        inspector = _InspectableKVCAwareBalancer.remote(
            {replica_id: server_handle},
            config.actor_rollout_ref.rollout.router_config,
        )
        initial = await inspector.inspect_prefix.remote(replica_id, [])
        assert initial["registry_ready"] is True, initial
        assert initial["registry"] is not None, initial
        assert len(initial["registry"]["groups"]) == group_count, initial
        actual_specs = {group["spec_kind"] for group in initial["registry"]["groups"]}
        if EXPECTED_SPEC_KINDS:
            assert actual_specs == EXPECTED_SPEC_KINDS, initial

        scheduler_size = initial["registry"]["scheduler_block_size"]
        hash_size = initial["registry"]["hash_block_size"]
        partial_enabled = initial["registry"]["partial_hash_hits_enabled"]
        if PREFIX_MATCH_UNIT is not None:
            assert hash_size == PREFIX_MATCH_UNIT, initial
            assert partial_enabled is True, initial
        query_boundary = 2 * scheduler_size
        if partial_enabled:
            query_boundary += hash_size
            assert query_boundary % scheduler_size != 0

        tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL, trust_remote_code=True)
        prompt = "Cache group routing validates a shared prefix across every attention group. " * 512
        all_prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        prompt_ids = all_prompt_ids[: query_boundary + 1]
        assert len(prompt_ids) == query_boundary + 1

        client = manager.get_client()
        output = await client.generate(
            "cache-group-registry-warmup",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_tokens": 1},
        )
        assert output.token_ids

        snapshot = await _wait_for_gpu_hit(inspector, replica_id, prompt_ids)
        assert snapshot["hash_count"] > 0, snapshot
        assert snapshot["gpu_hit"] > 0, snapshot
        print(
            "cache-group smoke: "
            f"registry_ready={snapshot['registry_ready']} "
            f"group_count={len(snapshot['registry']['groups'])} "
            f"partial={partial_enabled} gpu_hit={snapshot['gpu_hit']:.4f}"
        )
    finally:
        ray.shutdown()
