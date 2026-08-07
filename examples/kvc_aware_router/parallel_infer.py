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
"""Standalone inference runner for the kvc_aware_router example (blackbox framework).

Spins up vLLM replicas behind the KV-cache-aware router, plus gateway + reward
worker, runs blackbox mini-swe-agent sessions in parallel, and reports resolve
rate. Does NOT start the Megatron trainer.

Config source: config/kvc_aware_router_infer.yaml (base ppo_megatron_trainer +
router@router_config: kvcaware); megatron/optimizer sections are inert here.

The agent runner / reward / dataset classes are reused from the installed
`uni-agent` package (examples.blackbox_recipes.mini_swe_agent.*).

Usage:
    python examples/kvc_aware_router/parallel_infer.py \
        --model-path ~/models/Qwen3.5-9B \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --max-samples 1 --kv-events
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from uni_agent.framework.entry import build_agent_framework, build_gateway_manager

from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import tq
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    force=True,
)
logger = logging.getLogger(__name__)

# ── Recipe-specific constants ───────────────────────────────────────────────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
_CONFIG_NAME = "kvc_aware_router_infer"
_DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest"

# Ray's default idle-worker reaper (~10 s) kills agent workers between dispatch
# gaps, ending the job prematurely. Use a very large threshold so long-running
# agent loops are not interrupted.
_RAY_IDLE_WORKER_TIMEOUT_MS = int(os.getenv("RAY_IDLE_WORKER_TIMEOUT_MS", str(2**30 - 1)))


# =====================================================================
# Dataset loading (inlined; keeps the driver self-contained)
# =====================================================================


def _remap_image_to_local(image_name: str) -> str:
    parts = image_name.split("/")
    if len(parts) > 1 and "." in parts[0]:
        basename = parts[-1]
    else:
        basename = image_name
    basename = basename.replace("_1776_", "__")
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return f"{basename}:latest"


def _remap_sample_images(sample: dict[str, Any]) -> dict[str, Any]:
    extra_info = sample.get("extra_info")
    if not extra_info:
        return sample
    tools_kwargs = extra_info.get("tools_kwargs", {})
    env = tools_kwargs.get("env", {})
    image = env.get("image")
    if not image:
        return sample
    local_image = _remap_image_to_local(image)
    if local_image != image:
        logger.debug("Remapping image: %s -> %s", image, local_image)
        env["image"] = local_image
    return sample


def _inject_reward_fields(sample: dict[str, Any]) -> None:
    extra_info = sample.get("extra_info", {})
    tools_kwargs = extra_info.get("tools_kwargs", {})
    reward_config = tools_kwargs.get("reward", {})
    sample.setdefault("data_source", reward_config.get("name", "unknown"))
    sample.setdefault("reward_model", {"ground_truth": {}})


def load_swe_dataset(
    data_path: str, max_samples: int = -1, shuffle: bool = False, seed: int = 42
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = os.path.expanduser(data_path)
    logger.info("Loading dataset from: %s", path)
    samples = pq.read_table(path).to_pylist()
    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])
    if shuffle:
        import random

        logger.info("Shuffling dataset (seed=%d) before sampling", seed)
        rng = random.Random(seed)
        rng.shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    logger.info("Loaded %d samples", len(samples))
    return samples


# =====================================================================
# Config
# =====================================================================


def _load_config(
    *,
    model_path: str,
    engine: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    enable_mooncake: bool,
    device: str,
    kv_events: bool,
    alpha: float | None,
    load_threshold: float | None,
    slow_cut: str | None,
    overload_mode: str | None,
    do_shortcut: bool | None,
) -> Any:
    """Compose the recipe's training config and override inference fields.

    The megatron/actor/optimizer sections are left untouched and never read.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        config = compose(config_name=_CONFIG_NAME)

    OmegaConf.set_struct(config, False)

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)

    ro = config.actor_rollout_ref.rollout
    ro.name = engine
    ro.mode = "async"
    if max_model_len is not None:
        # Preserve the previous kvcaware runner's context-budget logic: reserve
        # 100 tokens and derive the prompt budget from model context minus the
        # requested response budget.
        prompt_length = max_model_len - response_length - 100
        if prompt_length <= 0:
            raise ValueError(
                "--max-model-len must exceed --response-length by at least 100 tokens "
                f"(got {max_model_len=} and {response_length=})"
            )
        ro.max_model_len = max_model_len
    else:
        # Clamp so the computed context never exceeds the model-native context
        # declared by the recipe config.
        ro.max_model_len = min(prompt_length + response_length + 1024, ro.max_model_len)
    ro.prompt_length = prompt_length
    ro.response_length = response_length
    ro.max_num_batched_tokens = ro.max_model_len
    ro.n = n
    ro.temperature = temperature
    ro.top_p = top_p
    ro.tensor_model_parallel_size = tensor_parallel_size
    ro.gpu_memory_utilization = gpu_memory_utilization
    ro.nnodes = nnodes
    ro.n_gpus_per_node = n_gpus_per_node
    ro.calculate_log_probs = True
    ro.enable_sleep_mode = False
    ro.disable_log_stats = False  # expose engine /metrics for the kvcaware collector

    # vLLM engine kwargs: MFU metric (always on) + optional mooncake connector /
    # kv-events (the kvcaware router's retained-cache load signal).
    vllm_kwargs: dict = {"enable_mfu_metrics": True}
    if enable_mooncake:
        # The connector class differs by backend: GPU build uses
        # "MooncakeStoreConnector"; vllm-ascend uses "MooncakeConnectorStoreV1".
        mooncake_connector = "MooncakeConnectorStoreV1" if device == "ascend" else "MooncakeStoreConnector"
        vllm_kwargs["kv_transfer_config"] = {
            "kv_connector": mooncake_connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {},
        }
    if kv_events:
        # vLLM kv-events (zmq publisher). Endpoint ports are placeholders
        # (verl assigns ephemeral).
        vllm_kwargs["kv-events-config"] = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:0",
            "replay_endpoint": "tcp://*:0",
        }
    ro.engine_kwargs = {"vllm": vllm_kwargs}

    # KVCAware router strategy[0] overrides — each falls back to the
    # kvcaware.yaml value when the flag is omitted.
    strat0 = ro.router_config.strategies[0]
    if alpha is not None:
        strat0.alpha = alpha
    if load_threshold is not None:
        strat0.load_threshold = load_threshold
    if slow_cut is not None:
        strat0.slow_cut = slow_cut
    if overload_mode is not None:
        strat0.overload_mode = overload_mode
    if do_shortcut is not None:
        strat0.do_shortcut = do_shortcut

    af = ro.custom.agent_framework
    af.gateway_count = gateway_count
    runner_name = next(iter(af.agent_runners.keys()))
    runner_cfg = af.agent_runners[runner_name]
    runner_cfg.max_concurrent_sessions = max_concurrent_sessions
    if tool_image:
        runner_cfg.runner_kwargs.tool_image = tool_image
    runner_cfg.runner_kwargs.run_timeout = run_timeout

    config.trainer.nnodes = nnodes
    config.trainer.n_gpus_per_node = n_gpus_per_node

    OmegaConf.set_struct(config, True)
    return config


# =====================================================================
# Batch + score capture
# =====================================================================


def _build_prompts(samples: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    raw_prompts = [sample["prompt"] for sample in samples]
    uids = [str(uuid4()) for _ in samples]
    tools_kwargs_list = [dict((sample.get("extra_info") or {}).get("tools_kwargs", {})) for sample in samples]
    prompts = tu.get_tensordict(
        tensor_dict={
            "raw_prompt": raw_prompts,
            "uid": uids,
            "data_source": [sample["data_source"] for sample in samples],
            "reward_model": [sample["reward_model"] for sample in samples],
            "tools_kwargs": tools_kwargs_list,
        },
        non_tensor_dict={"global_steps": 0},
    )
    return prompts, uids


def _install_tq_capture() -> tuple[dict[str, float], dict[str, str]]:
    """Monkeypatch the process-local TransferQueue to capture rm_scores in-memory.

    Runner dispatch is a Ray task, but session finalize/score/TQ-writes happen
    in this driver process, so patching ``tq`` here captures every write.
    """
    captured_scores: dict[str, float] = {}
    uid_status: dict[str, str] = {}

    async def _fake_put(*, key, partition_id=None, tag=None, **kwargs):
        if isinstance(tag, dict) and "status" in tag:
            uid_status[str(key)] = str(tag["status"])

    async def _fake_batch_put(*, keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        if fields is None or keys is None or "rm_scores" not in fields:
            return
        rm = fields["rm_scores"]  # nested tensor; rm[i] is trajectory i's response scores
        for i, key in enumerate(keys):
            row = rm[i]
            captured_scores[str(key)] = float(row[-1].item()) if row.numel() else 0.0

    tq.async_kv_put = _fake_put
    tq.async_kv_batch_put = _fake_batch_put
    return captured_scores, uid_status


def _report(samples, uids, captured_scores) -> dict[str, Any]:
    uid_to_index = {uid: i for i, uid in enumerate(uids)}
    per_sample_sum = [0.0] * len(samples)
    per_sample_cnt = [0] * len(samples)
    for key, score in captured_scores.items():
        # key format: {uid}_{session_index}_{index}
        uid = key.rsplit("_", 2)[0]
        idx = uid_to_index.get(uid)
        if idx is None:
            continue
        per_sample_sum[idx] += score
        per_sample_cnt[idx] += 1
    per_sample_scores = [
        per_sample_sum[i] / per_sample_cnt[i] if per_sample_cnt[i] else 0.0 for i in range(len(samples))
    ]
    resolved = sum(1 for s in per_sample_scores if s > 0)
    mean = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0
    logger.info(
        "Resolved %d / %d samples (%.2f%%), mean score: %.4f",
        resolved,
        len(samples),
        100.0 * resolved / max(len(samples), 1),
        mean,
    )
    return {"resolved": resolved, "total": len(samples), "mean_score": mean, "per_sample_scores": per_sample_scores}


# =====================================================================
# Runner
# =====================================================================


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    max_samples: int,
    shuffle: bool,
    seed: int,
    engine: str,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
    result_path: str | None,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    mooncake_config_path: str | None,
    enable_mooncake: bool,
    device: str,
    kv_events: bool,
    alpha: float | None,
    load_threshold: float | None,
    slow_cut: str | None,
    overload_mode: str | None,
    do_shortcut: bool | None,
) -> dict[str, Any]:
    # vLLM's mooncake connector reads MOONCAKE_CONFIG_PATH (not extra_config).
    # Set before ray.init so Ray-spawned workers inherit it.
    if enable_mooncake and mooncake_config_path:
        os.environ["MOONCAKE_CONFIG_PATH"] = os.path.expanduser(mooncake_config_path)

    if not ray.is_initialized():
        ray.init(_system_config={"idle_worker_killing_time_threshold_ms": _RAY_IDLE_WORKER_TIMEOUT_MS})

    config = _load_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        temperature=temperature,
        top_p=top_p,
        n=n,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
        gateway_count=gateway_count,
        max_concurrent_sessions=max_concurrent_sessions,
        tool_image=tool_image,
        run_timeout=run_timeout,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_mooncake=enable_mooncake,
        device=device,
        kv_events=kv_events,
        alpha=alpha,
        load_threshold=load_threshold,
        slow_cut=slow_cut,
        overload_mode=overload_mode,
        do_shortcut=do_shortcut,
    )

    samples = load_swe_dataset(data_path, max_samples=max_samples, shuffle=shuffle, seed=seed)
    if not samples:
        raise ValueError("No samples to process")

    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)
    llm_client = llm_server_manager.get_client()

    gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
    reward_worker = ray.remote(RewardLoopWorker).remote(config, None)
    framework = build_agent_framework(
        config=config,
        gateway_manager=gateway_manager,
        reward_loop_worker_handles=[reward_worker],
    )

    prompts, uids = _build_prompts(samples)
    captured_scores, _uid_status = _install_tq_capture()

    logger.info("Starting %d sample(s), %d session(s) each...", len(samples), n)
    try:
        asyncio.run(framework.generate_sequences(prompts))
    except RuntimeError as exc:
        logger.warning("generate_sequences failed: %s", exc)

    if not captured_scores:
        logger.warning(
            "No trajectory scores captured — all rollouts may have failed (see the "
            "generate_sequences summary above), or the TransferQueue monkeypatch did not "
            "reach the writer; resolve rate will be reported as 0."
        )

    result = _report(samples, uids, captured_scores)

    if result_path:
        out = os.path.expanduser(result_path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model_path": os.path.expanduser(model_path), "data_path": data_path, **result}, f, indent=2)
        logger.info("Wrote result file to: %s", out)

    asyncio.run(gateway_manager.shutdown())
    return result


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Blackbox mini-swe-agent standalone inference")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3.5-9B")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before --max-samples slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle.")
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM maximum context length. Defaults to the config-native context clamp.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization (fraction, 0-1).",
    )
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-count", type=int, default=1)
    parser.add_argument("--max-concurrent-sessions", type=int, default=128)
    parser.add_argument("--tool-image", type=str, default=_DEFAULT_TOOL_IMAGE)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Optional path to write a JSON result file (resolve rate + per-sample scores).",
    )
    # KVCAware router options
    parser.add_argument(
        "--kv-events",
        action="store_true",
        help="Enable vLLM kv-events zmq publisher (kvcaware router retained-cache load signal).",
    )
    parser.add_argument(
        "--enable-mooncake",
        action="store_true",
        help="Attach mooncake connector for cross-replica KV sharing (mooncake master runs separately).",
    )
    parser.add_argument(
        "--mooncake-config-path",
        type=str,
        default="mooncake_config.json",
        help="Path to the mooncake config JSON (used with --enable-mooncake).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "ascend"],
        help="Target backend: 'gpu' -> MooncakeStoreConnector; 'ascend' -> MooncakeConnectorStoreV1.",
    )
    # KVCAware strategy[0] overrides (fall back to kvcaware.yaml when omitted)
    parser.add_argument("--alpha", type=float, default=None, help="strategy[0] alpha (cache vs load blend, [0,1]).")
    parser.add_argument(
        "--load-threshold",
        type=float,
        default=None,
        help="strategy[0] load_threshold (overload when load > threshold, (0,1)).",
    )
    parser.add_argument(
        "--slow-cut",
        type=str,
        default=None,
        choices=["prefix-load-aware", "least-inflight", "capacity-token-aware"],
        help="strategy[0] slow_cut fallback scoring mode.",
    )
    parser.add_argument(
        "--overload-mode",
        type=str,
        default=None,
        choices=["None", "kv_cache_usage_perc", "kv_load"],
        help="strategy[0] overload_mode for the sticky short-circuit.",
    )
    parser.add_argument(
        "--do-shortcut",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="strategy[0] do_shortcut master switch (--do-shortcut / --no-do-shortcut).",
    )
    args = parser.parse_args()

    # Set before ray.init so runner Ray tasks inherit it.
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        gateway_count=args.gateway_count,
        max_concurrent_sessions=args.max_concurrent_sessions,
        tool_image=args.tool_image,
        run_timeout=args.run_timeout,
        result_path=args.result_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        mooncake_config_path=args.mooncake_config_path,
        enable_mooncake=args.enable_mooncake,
        device=args.device,
        kv_events=args.kv_events,
        alpha=args.alpha,
        load_threshold=args.load_threshold,
        slow_cut=args.slow_cut,
        overload_mode=args.overload_mode,
        do_shortcut=args.do_shortcut,
    )


if __name__ == "__main__":
    main()
