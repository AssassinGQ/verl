# Router 指标、rl-insight 与日志抓手（当前代码）

> 依据当前 `verl/workers/rollout/router/kvcaware/` 源码整理。指标的出站定义以 `types/emit_spec.py` 为单一事实来源；vLLM `/metrics` 的入站解析以 `collectors/decoder/vllm/metrics.py` 为准。

---

## 1. 数据链路

```text
vLLM /metrics ──HTTP poll──→ VLLMMetricsDecoder ─→ DataStore
Balancer acquire/release ─callback───────────────→ InflightDecoder ─→ DataStore
vLLM KV events ──ZMQ────────────────────────────→ VLLMKVDecoder ───→ DataStore

DataStore 写入
  ├─ router 日志：vllm-metrics / router-dispatch / vllm-evidence
  └─ Emitter（VERL_RL_INSIGHT_ENABLE=1）
       └─ rl-insight → Prometheus → Grafana

strategy.score()
  └─ Emitter：策略分数成分与 route latency
```

Collector 配置的默认 HTTP poll 间隔是 5 秒；运行配置可覆盖它。实际使用哪个周期，应从对应实验日志相邻 `vllm-metrics` 时间戳核实，不能只看 yaml/default。

---

## 2. rl-insight 发射的指标

所有带 replica label 的指标都使用：

```text
replica=<ip:port>
```

唯一不带 replica label 的指标是 `route_latency_seconds`。

### 2.1 B 类：DataStore 写入时发射

| 指标 | 类型 | 发射时机 | 含义 / 正确使用方式 |
|---|---|---|---|
| `kv_cache_usage_perc` | gauge | HTTP poll | vLLM 报告的 KV usage；capacity 策略计算 `avail` 的输入。 |
| `num_requests_running` | gauge | HTTP poll | vLLM 当前运行 request 数；观察引擎是否仍在工作。 |
| `num_requests_waiting` | gauge | HTTP poll | vLLM 当前等待 request 数。 |
| `kv_cache_load` | gauge | HTTP poll | router 自算的 retained blocks / num_gpu_blocks；不同于 `kv_cache_usage_perc`。 |
| `prompt_tokens` | gauge（累计值） | HTTP poll | vLLM 累计本地计算的 prefill token（cache miss）。用相邻样本差分求窗口量。 |
| `prompt_tokens_cached` | gauge（累计值） | HTTP poll | vLLM 累计由本地 prefix cache 命中的 prompt token。与 `prompt_tokens` 的差分组合算 token hit ratio。 |
| `external_prefix_cache_hits` | gauge（累计值） | HTTP poll | vLLM 累计外部 prefix cache hit 数。 |
| `estimated_flops_per_gpu` | gauge（累计值） | HTTP poll | vLLM 累计估算 FLOPs；差分/rate 可估计 MFU。 |
| `dispatched_count` | counter | router acquire | 每次 router dispatch +1；看派发份额和 router 请求量。 |
| `completed_count` | counter | router release | 每次 router release +1；与 dispatched 配对。 |
| `prompt_len_sum` | counter | router acquire | 每次 dispatch 加该 request prompt 长度；`Δprompt_len_sum / Δdispatched_count` 是窗口平均 prompt 长。 |
| `inflight_tokens` | gauge | router acquire/release | acquire 加 prompt 长度，release 减同一 prompt 长度；router 侧 token 加权在途量。 |
| `inflight_avg_turn` | gauge | router acquire/release | `inflight_turn_sum / inflight_count`；空闲时定义为 0。 |
| `active_sessions` | gauge | sticky binding 生命周期 | 该 replica 的存活 sticky binding 数（router request_id == gateway session_id）：首绑 +1、改绑平移、session finalize/abort 或 LRU 逐出 −1。agentic 首绑信号——session 在工具/sandbox 阶段 `inflight_count` 归零但它保持抬升。 |
| `kv_evictions` | counter | KV block removed event | vLLM 移除的 block 数；**不是纯 eviction**，包含 request 完成时的释放。 |

### 2.2 A 类：策略 `score()` 时发射

| 指标 | 类型 | 仅在哪种 slow-cut 产生 | 含义 |
|---|---|---|---|
| `load` | histogram | `prefix-load-aware` | `_compute_load(kv_usage, running, waiting, inflight)` 的负载成分。 |
| `s_cache` | histogram | `prefix-load-aware` | 三层加权 prefix-hit score。 |
| `avail` | histogram | `capacity-token-aware` | `cap × (1 - kv_cache_usage_perc)`。 |
| `need` | histogram | `capacity-token-aware` | `prompt_len × (1 - gpu_hit)`。 |
| `remaining` | histogram | `capacity-token-aware` | `avail - need`，可为负。 |
| `route_latency_seconds` | histogram | 所有 `score()` | 策略 score() 的自计时，不含模型推理时间。 |

`avail`、`need`、`remaining` 使用 Prometheus 默认 histogram buckets；token 数通常远大于默认 bucket 设计范围。因此不要从其 p50/p95 直接推导真实 token 分布，应使用 `score(): replica=... avail=... need=... remaining=...` 日志。

---

## 3. 重要派生指标

### 3.1 Router 原始在途请求数

```text
raw_inflight = dispatched_count - completed_count
```

它表示 router 已记录 acquire、尚未记录 release 的 LLM request 数。它适合观察 router 账本与派发分布，**不等于**：

- 尚未结束的 agent session 数；
- vLLM 当前正在跑的 request 数；
- GPU 仍有有效计算工作。

有效长尾分析还要结合 session timeout 对账，详见 `analysis.md` 和 `exp1.md`。

### 3.2 Prefix-hit

有两种不同但都有效的口径，报告时必须写清：

```text
replica hit ratio
= prefix_cache_hits / prefix_cache_queries
```

这是 vLLM prefix query 的请求级统计，适合与策略 `gpu_hit` 的含义对照。

```text
token hit ratio（窗口）
= Δprompt_tokens_cached / (Δprompt_tokens_cached + Δprompt_tokens)
```

这是 token 级统计，`vllm-evidence` 会按窗口直接打印 `hit=<x>%`。分母为零时无定义。

注意：`gpu_hit` 是 router 根据 prefix hash 和自身 KV store 在当前请求、当前 replica 上算出的即时估计；它不是上述 vLLM 累计 query hit ratio，也不是 token hit ratio。

### 3.3 平均 prompt 长度与派发份额

```text
window_avg_prompt_len
= Δprompt_len_sum / Δdispatched_count

replica_dispatch_share
= Δdispatched_count(replica) / Σ Δdispatched_count(all replicas)
```

用于识别：副本接到的请求是否更长、派发是否集中、低命中副本是否被冷落。

---

## 4. 日志抓手

日志可能混有 ANSI 控制符和二进制内容；提取时使用能处理二进制的工具，例如 `grep -a`，并按需去掉 ANSI 色码。

| 分析目标 | 日志关键字 | 关键字段 / 解释 |
|---|---|---|
| 每次路由排名 | `route(): replicas=` | `ranking=[replica=score,...]`；策略异常时会走随机 fallback。 |
| 实际派发目的地 | `routed to server=` | `request=`、`server=`、`ranking=`、`route=<ms>`。 |
| sticky 命中/回退 | `score(): STICKY` | `HIT`、`OVERLOADED → fallback`。仅 shortcut 开启时出现。 |
| capacity 逐副本打分 | `score(): replica=` | `kv_perc`、`gpu_hit`、`inflight`、`avail`、`need`、`inflight_tokens`、`active_sessions`、`remaining`、`WINNER`。 |
| capacity 最终选择 | `CAPACITY_TOKEN_AWARE` | `cold start → min active_sessions`、`no eligible → max remaining`、`winner=`。 |
| capacity 全副本 remaining | `route-capacity remaining=` | 一次 routing 时全部 replica 的 remaining map。 |
| vLLM 原始采样 | `vllm-metrics replica=` | `/metrics` 解码后的所有 canonical 值，包括 prefix query/hit、running、KV usage、GPU blocks。 |
| router 生命周期账本 | `router-dispatch replica=` | `dispatched`、`completed`、`inflight_turn_sum`、`prompt_len_sum`、`active_sessions`；`dispatched-completed` 为 raw inflight。 |
| 窗口化 vLLM 摘要 | `vllm-evidence replica=` | `kv`、`usage`、`run`、`wait`、TTFT、queue、prefill、TPOT、token hit、decode。 |
| KV event 汇总 | `kv-events tally:` | stored/removed/clear event 与 block 计数；不能直接当成纯 eviction。 |
| 策略初始化确认 | `KVCacheAwareStrategy created:` | `load_threshold`、`do_shortcut`、`slow_cut`、`overload_mode` 等最终策略参数。 |
| 策略容量注入 | `KVCacheAwareStrategy capacity set:` | `max_num_seqs`、`max_num_batched_tokens`。 |
| session 生命周期 | `agent_runner_ray_task start` / `agent_runner_ray_task end` | `sample_index`、`session_index`、`elapsed`；识别接近 runner timeout 的 session。 |
| sandbox OOM | `sandbox was oom-killed by the kernel` | memory cgroup OOM，需与 session/replica 对账。 |
| 调用失败 | `invoke response failed` / `instance not exist` | OOM 或 sandbox 生命周期故障的后续症状。 |
| session / score 汇总 | `generate_sequences summary` / `=> Resolved` | `Resolved` 是 score 汇总哨兵，不是 router 清零判据。 |

---

## 5. 日志与 rl-insight 的边界

| 信号 | 推荐用途 | 不应做的解释 |
|---|---|---|
| `router-dispatch` | 重建 router dispatch/complete 账本、长尾候选边界 | 当作完整 agent session 生命周期 |
| `vllm-metrics.num_requests_running` | 判断 vLLM 是否仍有实际引擎请求 | 取代 router raw inflight |
| `vllm-evidence` | 约 30 个 poll 的窗口性能与 token hit 趋势 | 当作逐 request 精确记录 |
| A 类 histogram | 查看策略成分趋势 / 分布 | 对 token 量使用默认 buckets 的 p50/p95 作为精确值 |
| `Resolved` | 确认已有 score 汇总、实验脚本成功哨兵 | 证明全部 session 成功或 router 已收敛 |

---

## 6. 代码位置

| 内容 | 源码 |
|---|---|
| 入口指标键与描述 | `types/metric_spec.py` |
| rl-insight 出站指标契约 | `types/emit_spec.py` |
| Emitter 的发射时机与类型映射 | `insight/emitter.py` |
| HTTP poll、日志与 DataStore 写入 | `collectors/collector.py` |
| vLLM Prometheus 解析 | `collectors/decoder/vllm/metrics.py` |
| acquire/release → inflight 计数 | `collectors/decoder/basic/inflight.py` |
| 策略 A 类分数计算 | `strategies/kvc_aware.py` |
