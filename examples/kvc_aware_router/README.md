# kvc_aware_router Inference Quick Start

Last updated: 08/07/2026

## What is this

This example runs SWE-bench agentic inference on verl's **KV-cache-aware
router** (`kvcaware`) using the uni-agent **blackbox framework**: vLLM replicas
sit behind the kvcaware router (KV-cache hit rate + load aware dispatch), a
gateway pool drives `mini_swe_agent` sessions in AKernel remote sandboxes, and
a reward worker reports resolve rate. No trainer is started.

Core components:
- `KVCAwareBalancer` — routing framework, manages component lifecycle and routing decisions
- `Collector` — collects vLLM KV events and Prometheus metrics for decisions
- `Strategy` — scoring strategy, combines KV cache hit rate and load
- `Store` — singleton storage for collected metrics and KV block states

The agent runner / reward / dataset code is reused from the installed
`uni-agent` package (`examples.blackbox_recipes.mini_swe_agent.*`).

## Prerequisites

1. This repo (verl, router-dev) and `pip install uni-agent`.
2. An AKernel remote-sandbox endpoint (`AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN`).
3. Dataset parquet (SWE-bench verified). Default path:
   `examples/kvc_aware_router/swe_bench_verified_modal.parquet` — generate with
   uni-agent's `examples/data_preprocess/swe_bench_verified.py`, or
   point `--data-path` at any compatible parquet.

## Run

`run_infer.sh` is a thin wrapper: it exports the Ray-worker environment (AKernel
creds + observability env) and forwards all CLI flags to `parallel_infer.py`.
See the full flag list with defaults via `--help`:

```bash
bash examples/kvc_aware_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --max-samples 1 --kv-events

# Full 8-GPU data-parallel
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --max-samples -1 --kv-events

# With mooncake cross-replica KV sharing (mooncake master runs separately)
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --enable-mooncake --kv-events

# Ascend (vllm-ascend) backend
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --device ascend --enable-mooncake
```

Key CLI flags (see `--help` for the complete list with defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `~/models/Qwen3.5-9B` | Model path |
| `--data-path` | `<example>/swe_bench_verified_modal.parquet` | Dataset parquet |
| `--max-samples` | `-1` | Samples to run (-1 = all) |
| `--shuffle` / `--seed` | off / `42` | Shuffle before sampling, with a reproducible seed |
| `--prompt-length` / `--response-length` | `4096` / `131072` | Token lengths |
| `--max-model-len` | config-native clamp | vLLM maximum context length; with this flag, prompt length becomes `max_model_len - response_length - 100` |
| `--n` | `1` | Sessions per sample |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | `4` / `8` / `1` | Parallelism |
| `--gateway-count` / `--max-concurrent-sessions` | `1` / `128` | Gateway pool / session concurrency |
| `--gpu-memory-utilization` | `0.8` | vLLM GPU memory utilization (0-1) |
| `--tool-image` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | Sandbox sidecar tool image |
| `--run-timeout` | `7200` | Per-session sandbox run timeout (s) |
| `--max-turns` | `100` | Max agent turns per session |
| `--kv-events` | off | Enable vLLM kv-events (kvcaware load signal) |
| `--enable-mooncake` / `--mooncake-config-path` | off / `mooncake_config.json` | Cross-replica KV sharing |
| `--device` | `gpu` | `gpu` or `ascend` (selects mooncake connector class) |
| `--alpha` / `--load-threshold` / `--slow-cut` / `--overload-mode` / `--do-shortcut` | from `kvcaware.yaml` | kvcaware strategy[0] overrides |

## Environment variables

Only variables read via `os.environ` inside Ray-spawned workers (not CLI
flags) — `run_infer.sh` exports them; set them in the shell before invoking:

| Var | Default | Description |
|-----|---------|-------------|
| `AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN` | empty | AKernel remote sandbox auth |
| `AKERNEL_TUNNEL_SSL_VERIFY` | `0` | AKernel tunnel TLS verify (0 = disabled) |
| `VERL_LOGGING_LEVEL` | `INFO` | verl logging level |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout in the sandbox (s) |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight observability server |

## Observability

Router logs (`vllm-evidence`, `router-dispatch`, `score():`, `is-overload`)
are parsed by `plot_metrics.py` into a 24-panel time-aligned figure:

```bash
python examples/kvc_aware_router/plot_metrics.py /path/to/run.log
```

## Notes

- The blackbox runner requires an AKernel remote sandbox; without it sessions
  fail fast. The old swe-rex localdocker/simulated agent configs were removed
  in the blackbox-framework migration.
- Dataset schema must carry `extra_info.tools_kwargs` with `env.image` /
  `reward` fields (uni-agent blackbox format). If your parquet predates it,
  regenerate with the uni-agent data_preprocess script.
- Keep `--prompt-length + --response-length` comfortably below the model's native
  context length; `max_model_len` is clamped to the config-declared value so it
  never exceeds the model's context.
