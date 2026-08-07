# llm-router 推理快速开始

Last updated: 08/07/2026

## 这是什么

本示例在 verl 的 **KV-cache-aware router**（`kvcaware`）上运行 SWE-bench agentic 推理,使用 uni-agent 的 **blackbox 框架**:vLLM 副本位于 kvcaware 路由器之后(KV-cache 命中率 + 负载感知的调度),gateway 池在 AKernel 远程沙箱中驱动 `mini_swe_agent` 会话,reward worker 上报 resolve rate。不启动 trainer。

核心组件:
- `KVCAwareBalancer` — 路由框架,管理组件生命周期与路由决策
- `Collector` — 采集 vLLM KV 事件与 Prometheus 指标用于决策
- `Strategy` — 评分策略,综合 KV-cache 命中率与负载
- `Store` — 单例存储,缓存采集到的指标与 KV block 状态

agent runner / reward / dataset 代码复用自已安装的 `uni-agent` 包
(`examples.blackbox_recipes.mini_swe_agent.*`)。

## 前置条件

1. 本仓库(verl, router-dev)和 `pip install uni-agent`。
2. 一个 AKernel 远程沙箱端点(`AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN`)。
3. 数据集 parquet(SWE-bench verified)。默认路径:
   `examples/kvc_aware_router/swe_bench_verified_modal.parquet` — 使用
   uni-agent 的 `examples/data_preprocess/swe_bench_verified.py` 生成,
   或将 `--data-path` 指向任意兼容的 parquet。

## 运行

`run_infer.sh` 是一个薄包装:导出 Ray worker 环境变量(AKernel 凭据 +
可观测性 env),然后把所有 CLI flag 透传给 `parallel_infer.py`。完整 flag
列表与默认值用 `--help` 查看:

```bash
bash examples/kvc_aware_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --max-samples 1 --kv-events

# 全量 8-GPU data-parallel
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --max-samples -1 --kv-events

# 带 mooncake 跨副本 KV 共享(mooncake master 单独起)
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --enable-mooncake --kv-events

# Ascend（vllm-ascend）后端
bash examples/kvc_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3.5-9B --device ascend --enable-mooncake
```

主要 CLI flag(完整列表见 `--help`):

| Flag | 默认值 | 说明 |
|------|---------|------|
| `--model-path` | `~/models/Qwen3.5-9B` | 模型路径 |
| `--data-path` | `<example>/swe_bench_verified_modal.parquet` | 数据集 parquet |
| `--max-samples` | `-1` | 运行的样本数(-1 = 全部) |
| `--shuffle` / `--seed` | 关 / `42` | 采样前打乱数据,并指定可复现的随机种子 |
| `--prompt-length` / `--response-length` | `4096` / `131072` | Token 长度 |
| `--max-model-len` | config 原生上限 | vLLM 最大上下文长度;设置后 prompt 长度变为 `max_model_len - response_length - 100` |
| `--n` | `1` | 每个样本的会话数 |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | `4` / `8` / `1` | 并行度 |
| `--gateway-count` / `--max-concurrent-sessions` | `1` / `128` | Gateway 池 / 会话并发度 |
| `--gpu-memory-utilization` | `0.8` | vLLM GPU 显存利用率(0-1) |
| `--tool-image` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | 沙箱 sidecar 工具镜像 |
| `--run-timeout` | `7200` | 每个会话沙箱运行超时(秒) |
| `--max-turns` | `100` | 每个会话的最大 agent 轮次 |
| `--kv-events` | 关 | 启用 vLLM kv-events(kvcaware 负载信号) |
| `--enable-mooncake` / `--mooncake-config-path` | 关 / `mooncake_config.json` | 跨副本 KV 共享 |
| `--device` | `gpu` | `gpu` 或 `ascend`(选择 mooncake connector 类) |
| `--alpha` / `--load-threshold` / `--slow-cut` / `--overload-mode` / `--do-shortcut` | 取自 `kvcaware.yaml` | kvcaware strategy[0] 覆盖 |

## 环境变量

仅列出 Ray worker 内部通过 `os.environ` 读取的变量(非 CLI flag)——
`run_infer.sh` 会 export 它们,调用前在 shell 里设置:

| 变量 | 默认值 | 说明 |
|-----|---------|------|
| `AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN` | 空 | AKernel 远程沙箱认证 |
| `AKERNEL_TUNNEL_SSL_VERIFY` | `0` | AKernel 隧道 TLS 校验(0 = 禁用) |
| `VERL_LOGGING_LEVEL` | `INFO` | verl 日志级别 |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | 沙箱内 reward 评估超时(秒) |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight 可观测性服务 |

## 可观测性

路由器日志(`vllm-evidence`、`router-dispatch`、`score():`、`is-overload`)
由 `plot_metrics.py` 解析为 24 面板时间对齐图:

```bash
python examples/kvc_aware_router/plot_metrics.py /path/to/run.log
```

## 注意事项

- blackbox runner 需要一个 AKernel 远程沙箱;没有它会快速失败。旧的
  swe-rex localdocker/simulated agent 配置已在 blackbox 框架迁移中被移除。
- 数据集 schema 必须携带 `extra_info.tools_kwargs`,包含 `env.image` /
  `reward` 字段(uni-agent blackbox 格式)。如果你的 parquet 早于该格式,
  请使用 uni-agent 的 data_preprocess 脚本重新生成。
- 请保持 `--prompt-length + --response-length` 明显低于模型的原生上下文长度;
  `max_model_len` 会被钳制到配置文件声明的值,确保不超过模型上下文。
