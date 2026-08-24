# Exp1：128 并发 sticky vs kvcaware 对比实验

> **范围固定**：`results/result-0820/` 下，`concurrency=128` 的 **2 策略 × 4 context** 矩阵，共 8 次实验。
>
> 本文是 exp1 的单一事实来源。通用指标见 [`analysis.md`](analysis.md)，64×8 session 和 `Resolved` 语义见 [`framework.md`](framework.md)，指标与日志字段见 [`metric.md`](metric.md)。

---

# 1. 环境与实验范围

## 1.1 运行环境

| 项目 | 配置 |
|---|---|
| 集群 / 推理副本 | 6 节点 Ascend；12 replica（`6 × 8 NPU / TP=4`） |
| 模型 | Qwen3.5-9B（以实际容器配置为准） |
| 运行版本 | 容器 `e60b08d2 localize`；策略行为以日志验证为准 |
| 采样与 session | `MAX_SAMPLES=64`，`--n 8`，即 512 session |
| session 并发上限 | `max_concurrent_sessions=128` |
| runner timeout | `run_timeout=7200s` |
| sandbox command deadline | `timeout=600s`，`hard_deadline=630s` |
| rl-insight | 已启用；实际 poll 周期以实验日志/Prometheus 数据为准 |

`prompt64x8` 表示 driver 输入 64 条 sample，AgentFramework 再为每条 sample 展开 8 个独立 session；不是一次输入 512 条 prompt。

## 1.3 数据位置与时间对齐

| 数据 | 路径 | 用途 |
|---|---|---|
| 实验日志 | `results/result-0820/infer-{sticky,kvcaware-lt0.9}-prompt64x8-128x{16384,32768,64000,128000}-n6.log` | session、router、vLLM、OOM/timeout 的文本证据。 |
| rl-insight | `results/result-0820/rl-insight/` | exp1 同期的 Prometheus、Grafana、Tempo 数据。 |
| Prometheus | `results/result-0820/rl-insight/prometheus/` | per-replica dispatch、running、KV、prefix-hit 等时序。 |
| Grafana | `results/result-0820/rl-insight/grafana/` | `grafana.db` 与服务日志。 |
| Tempo | `results/result-0820/rl-insight/tempo/` | 如存在，可补充 session 链路。 |

已用 32K 两组验证时间轴：将日志中的时间字符串直接作为 **UTC** 查询 Prometheus，可命中同次实验新 replica port 从零开始的 counter，开始时间只差秒级。

| 实验 | 日志首个真实派发 | Prometheus 新 replica 指标开始 |
|---|---|---|
| sticky / 32K | `2026-08-19 21:10:38` | `2026-08-19 21:10:xx UTC` |
| kvcaware / 32K | `2026-08-19 23:36:00` | `2026-08-19 23:36:xx UTC` |

因此 exp1 采用：

```text
日志时间 = Prometheus 查询 UTC 时间
```

本机终端可能显示 UTC+8，不应据此再次换算日志时间。其它实验或数据目录仍须用 `dispatched_count` 起点、`num_requests_running` 上升或 token counter 起点复核。

## 1.4 八个实验

| 策略 | Context | 日志 |
|---|---:|---|
| sticky | 16K | `infer-sticky-prompt64x8-128x16384-n6.log` |
| kvcaware lt=0.9 | 16K | `infer-kvcaware-lt0.9-prompt64x8-128x16384-n6.log` |
| sticky | 32K | `infer-sticky-prompt64x8-128x32768-n6.log` |
| kvcaware lt=0.9 | 32K | `infer-kvcaware-lt0.9-prompt64x8-128x32768-n6.log` |
| sticky | 64K | `infer-sticky-prompt64x8-128x64000-n6.log` |
| kvcaware lt=0.9 | 64K | `infer-kvcaware-lt0.9-prompt64x8-128x64000-n6.log` |
| sticky | 128K | `infer-sticky-prompt64x8-128x128000-n6.log` |
| kvcaware lt=0.9 | 128K | `infer-kvcaware-lt0.9-prompt64x8-128x128000-n6.log` |

不属于 exp1：`16×*`、`96×128K`、其他 `lt` 值和 `results/result-0821/` 重跑。

---

# 2. 策略配置

## 2.1 Sticky：least-inflight + 全生命周期粘性

首次 request 选择：

```text
min(inflight_count)
```

随后同一 `request_id` sticky 回首次 replica。exp1 中 `overload_mode=None`，不会触发过载 fallback，因此绑定不会迁移。

## 2.2 KVC-aware：capacity-token-aware 初始分配 + sticky shortcut

日志中实际生效的配置：

```text
slow_cut               = capacity-token-aware
overload_mode          = kv_cache_usage_perc
load_threshold         = 0.9
do_shortcut            = true
memory_overload_filter = true
```

首次 request 没有 binding，进入 cold start：

```text
min(inflight_tokens)
```

常规 capacity 分支的计算是：

```text
avail     = cap × (1 - kv_cache_usage_perc)
need      = prompt_len × (1 - gpu_hit)
remaining = avail - need
```

并先筛选：

```text
avail >= cap × (1 - load_threshold)
```

即 `lt=0.9` 时优先选择至少有 10% KV 容量可用的副本；再选 `remaining` 最大者。

但 exp1 中该常规分支并未承担后续轮次分流：每个 context 有 512 次 cold start/winner，正好对应 512 session；后续均为 `STICKY HIT`，`OVERLOADED → fallback` 为 0。因此 kvcaware 的实际语义是：

```text
首次 request：min(inflight_tokens)
后续 request：sticky 回首次 replica
```

## 2.3 Exp1 的真实策略对比

| 实验标签 | 首次绑定 | 后续 request | 实际比较对象 |
|---|---|---|---|
| sticky | `min(inflight_count)` | sticky 回原 replica | 按当前 request 数建立会话绑定 |
| kvcaware lt=0.9 | `min(inflight_tokens)` | sticky 回原 replica | 按当前 prompt token 量建立会话绑定 |

exp1 比较的是两种**全生命周期粘性**的初始绑定方式，不是“粘性 vs 无粘性容量路由”。两组 prefix-hit 接近是这一事实的自然结果。

## 2.4 当前代码的实现风险

当前 host 代码的常规 capacity 分支直接选择 `max(remaining)`，是确定性单赢家；没有多候选随机选择。该风险应在关闭 shortcut 或触发 overload fallback 的后续实验中验证，不能用来解释 exp1 的后续轮次。

---

# 3. 实验结果

## 3.1 性能表（日志重建，当前阶段）

- Walltime：首条 `router-dispatch` 到最后一轮 `vllm-metrics`。
- Prefix-hit：每 replica 最终 `prefix_cache_hits / prefix_cache_queries` 的算术平均。
- 原始阶段时间基于：
  ```text
  raw_inflight = dispatched_count - completed_count
  ```
  尚未扣除 timeout-terminated session，故不是最终有效调度时间。

| 策略 / Context | Prefix-hit | Walltime（min） | 原始非长尾（min） | 原始长尾（min） | 原始长尾后等待（min） | 超时 session（已观察） |
|---|---:|---:|---:|---:|---:|---:|
| sticky / 16K | 85.84% | 155.4 | 42.0 | 90.7 | 22.6 | 7 |
| kvcaware lt=0.9 / 16K | 85.99% | 121.1 | 5.8 | — | — | 15 |
| sticky / 32K | 91.99% | 120.6 | 74.6 | — | — | 9 |
| kvcaware lt=0.9 / 32K | 91.77% | 179.0 | 82.4 | 58.2 | 38.4 | 20 |
| sticky / 64K | 93.25% | 188.3 | 116.7 | — | — | 10 |
| kvcaware lt=0.9 / 64K | 93.47% | 192.8 | 128.0 | — | — | 24 |
| sticky / 128K | 90.24% | 208.0 | 124.1 | 43.3 | 40.7 | 11 |
| kvcaware lt=0.9 / 128K | 90.35% | 180.2 | 140.2 | 27.9 | 12.2 | 31 |

“超时 session（已观察）”取日志中 `agent_runner_ray_task end` 的 `elapsed >= 7000s`；它只是可解析下界，不是完整 512 session 的终态计数。

## 3.2 各阶段耗时对比

此表仅保留阶段耗时，仍是基于 `raw_inflight` 的初步结果。`—` 表示尚未完成 timeout session → request → replica 对账，不能补为 0 或当作有效长尾结束。

| 策略 / Context | Walltime（min） | 原始非长尾（min） | 原始长尾（min） | 原始长尾后等待（min） |
|---|---:|---:|---:|---:|
| sticky / 16K | 155.4 | 42.0 | 90.7 | 22.6 |
| kvcaware lt=0.9 / 16K | 121.1 | 5.8 | — | — |
| sticky / 32K | 120.6 | 74.6 | — | — |
| kvcaware lt=0.9 / 32K | 179.0 | 82.4 | 58.2 | 38.4 |
| sticky / 64K | 188.3 | 116.7 | — | — |
| kvcaware lt=0.9 / 64K | 192.8 | 128.0 | — | — |
| sticky / 128K | 208.0 | 124.1 | 43.3 | 40.7 |
| kvcaware lt=0.9 / 128K | 180.2 | 140.2 | 27.9 | 12.2 |

## 3.3 有效阶段耗时（session 对账后）

已完成 `session → request → replica` 对账：router 日志的 `request=session-<sample>-<session>-<uuid>` 直接携带 `(sample_index, session_index)`，与 runner start/end join 后可按 session 级终态重建。

口径：

```text
timeout session：agent_runner_ray_task end 的 elapsed >= 7000s
                 （runner run_timeout=7200s 的挂死收敛）
有效路由：      非 timeout session 的 routed to server=
tail_start：    第 6/12 个 replica 完成其最后一个有效路由的时刻
tail_end：      最后一个 replica 完成其最后一个有效路由的时刻
idle：          tail_end 之后到实验结束（挂死 timeout 等待 + 收尾）
```

| 策略 / Context | Walltime | 有效非长尾 | 有效长尾 | 长尾后无效等待 | timeout session | 挂死路由数 |
|---|---:|---:|---:|---:|---:|---:|
| sticky / 16K | 155:22 | 47:41 | 63:49 | 43:52 | 7 | 250 |
| kvcaware / 16K | 121:08 | 55:17 | 65:07 | 0:45 | 15 | 376 |
| sticky / 32K | 120:36 | 79:43 | 40:21 | 0:32 | 9 | 372 |
| kvcaware / 32K | 179:02 | 86:42 | 33:23 | 58:58 | 20 | 758 |
| sticky / 64K | 188:15 | 119:55 | 44:26 | 23:55 | 10 | 479 |
| kvcaware / 64K | 192:51 | 130:45 | 28:18 | 33:47 | 24 | 946 |
| sticky / 128K | 208:03 | 125:16 | 12:57 | 69:50 | 11 | 493 |
| kvcaware / 128K | 180:15 | 143:54 | 20:57 | 15:24 | 31 | 922 |

对账同时确认：

- 未解析到 `end` 的 session（Ray 日志折叠）为 11～70 个不等；其路由计入有效路由但无终态，量级 <12%。
- timeout session 的**首个绑定 replica** 高度集中：exp1 全部 8 组中，sticky 的 37 个 timeout session 有 24 个首绑到 27.24 两台；kvcaware 的 90 个有 69 个首绑到 27.24 两台。
- 32K kvcaware 的 20 个 timeout 中 17 个首绑 27.24；再次印证 4.6 的热点锁定机制。

## 3.4 同 context 对比

| Context | 更短 walltime | walltime 差（kvcaware - sticky） | Prefix-hit 差（kvcaware - sticky） | timeout session 差（kvcaware - sticky） |
|---|---|---:|---:|---:|
| 16K | kvcaware | -34.2 min | +0.15 pct | +8 |
| 32K | sticky | +58.4 min | -0.22 pct | +11 |
| 64K | sticky | +4.6 min | +0.22 pct | +14 |
| 128K | kvcaware | -27.8 min | +0.11 pct | +20 |

四个 context 中 kvcaware 的已观察 timeout session 都更多；因此较短 walltime 不能直接称为调度收益。

---

# 4. 关键发现

## 4.1 Prefix-hit 不是 exp1 walltime 差异的主要解释

同 context 下策略间的最终 prefix-hit 只差约 0.1～0.2 个百分点，但 walltime 可相差数十分钟。应优先检查初始绑定、运行期负载、长尾、OOM、timeout 与收尾等待。

## 4.2 Exp1 的差异来自初始绑定与异常收敛，而非无粘性迁移

两组后续均 sticky，因此都保持 session-local cache。唯一策略差异是首次绑定：

```text
sticky：   min(inflight_count)
kvcaware： min(inflight_tokens)
```

初始 prompt token 只能描述当前 KV/prefill 压力，不能预测 session 的未来轮数、上下文增长、工具耗时或 OOM/timeout 风险；sticky 建立后也不能迁移修正。

## 4.3 KVC-aware 的常规 capacity 分支在 exp1 中未承担后续分流

每组 kvcaware 都有 512 次 cold start、无 overload fallback，之后为 sticky hit。因此 `eligible → max(remaining)` 只是一条未被 exp1 后续轮次使用的代码路径。

## 4.4 OOM、timeout 与 router 残留 inflight 必须分层解释

`Resolved` 代表 AgentFramework 已完成 session 成功/失败收敛且捕获到至少一条 score；它不要求全部 session 成功，也不要求 router `dispatched == completed`。因此 `Resolved` 不能作为长尾结束或 router 收敛判据。

## 4.5 32K：kvcaware 为什么慢 58.4 分钟

### 排除 cache / 重计算原因

| 指标 | sticky | kvcaware |
|---|---:|---:|
| Walltime | 120.6 min | 179.0 min |
| dispatch | 30,613 | 29,575 |
| 本地重计算 prefill | 36.37M | 35.30M |
| token-level prefix hit | 92.33% | 92.16% |
| OOM 日志 | 4,424 | 5,380 |
| 已观察 timeout session | 9 | 20 |

kvcaware 不是因更低 hit、更高重计算或更多 request 而慢，而是在更长时间里完成更少工作，并出现更多 OOM/timeout。

### 冷启动数量均衡更好，但长期负载更差

首次 512 session 的绑定数量：

| 指标 | sticky / count | kvcaware / token |
|---|---:|---:|
| 最少 / 最多 session | 10 / 63 | 17 / 60 |
| 标准差 | 17.92 | 13.36 |
| CV | 0.420 | 0.313 |

所以 kvcaware 的 `min(inflight_tokens)` 在本次冷启动中确实让**session 数量**更均匀。但这没有变成长期有效工作量均衡。

两个 `10.170.27.24` replica 获得的初始 session 是：

```text
sticky：20 / 512（3.9%）
kvcaware：36 / 512（7.0%）
```

kvcaware 多绑定了 16 个 session；随后全生命周期 sticky，使这些 session 无法迁移。它们成为长期热点：

| 指标 | sticky | kvcaware |
|---|---:|---:|
| 两台 27.24 平均 running | 9.48 / 8.58 | 12.98 / 11.73 |
| 两台 27.24 peak running | 10 / 10 | 19 / 17 |
| 各 replica 平均 running CV | 0.389 | 0.870 |
| 各 replica 平均 inflight-token CV | 0.269 | 0.681 |
| 最大 inflight-token 峰值 | 278K | 386K |

kvcaware 的现象是：两个热点 replica 长期高 running/high token，其余多数 replica 仅低负载；集群有效并行度下降，进一步放大 OOM、timeout 与长尾。

当前最合理的因果链：

```text
min(inflight_tokens) 使冷启动 session 数更平均
→ 但当前 token 不能预测 session 未来总成本
→ 两台后续慢的 replica 多获得 16 个 session
→ 后续 sticky 不能迁移
→ 运行期 request/token 负载高度集中
→ 吞吐下降、OOM 与 timeout 增加
→ kvcaware 32K 多耗时 58.4 min
```

尚待 session→replica 对账确认的是：这两台 replica 是因为拿到更难 session 而慢，还是节点/sandbox 状态使正常 session 变慢。

但 Prometheus 的 vLLM 延迟指标已排除“仅 agent session 难度不同”的弱解释：两台 27.24 在两种策略下均有显著更高的引擎 prefill 与 decode 时间，且 queue time 近零。例如 32K：

| 指标（每 request 平均） | sticky 27.24 | kvcaware 27.24 | 其它多数 replica |
|---|---:|---:|---:|
| prefill time | 3.14 / 3.37 s | 3.15 / 3.48 s | 约 0.42～0.77 s |
| TPOT | 0.71 / 0.73 s | 0.75 / 0.81 s | 约 0.03～0.08 s |
| TTFT | 3.59 / 3.84 s | 3.61 / 4.02 s | 约 0.50～0.93 s |
| queue time | 近零 | 近零 | 近零 |

即 27.24 的慢发生在 vLLM 引擎 prefill/decode 本身，而非排队；TPOT 约为多数副本的 10～25 倍。这强烈指向该节点/两副本的 NPU、vLLM worker、通信或共享宿主资源异常。仍需节点级 NPU/CPU/网络/worker 日志确认具体根因。

## 4.6 27.24 长尾集中：随机性与可证实部分

32K 两组的 27.24 replica 都呈现“累计 dispatch 少、但 running 与 inflight-token 长期最高”的模式；kvcaware 只是进一步把其初始绑定从 20 个提高到 36 个 session。它们承载的是慢推进/异常 session，而不是大量短 request。

哪些 session 先绑定到 27.24 具有随机性：session 异步启动，模型采样、工具路径、任务难度和 sandbox 资源峰值都会变化。但“27.24 在两组中都长期成为热点”不能简单归为一次随机波动。现有数据尚不能区分两种原因：

```text
A. 初始绑定偶然把更难、更多轮或更易 OOM/timeout 的 session 给了 27.24；
B. 27.24 所在节点/副本对正常 session 也更慢。
```

需要用 `(sample_index, session_index) → 初始 replica → elapsed / OOM / timeout` 的对账，并用独立重复实验验证。

## 4.7 Exp1 的 overload 再均衡机制未触发

kvcaware 并非设计上永远不能修正 sticky：绑定副本若过载，shortcut 会 fallback，下一轮 request 才会重新走 capacity routing。

但 exp1 的触发条件只有：

```text
kv_cache_usage_perc > 0.9
```

32K 热点 27.24 的 KV usage 峰值仅约 0.22，远低于 0.9；四组日志的 `OVERLOADED → fallback` 均为 0。因此本次没有任何中后期重新均衡机会。

这暴露的是信号错配：KV usage 适合“KV 快满”的容量保护，却不能感知“KV 未满、但慢 session / tool / OOM / timeout 让副本长期高 running、高 inflight-token”的 agentic 长尾热点。

---

# 5. 启发

## 5.1 评价调度器要分离吞吐、收敛和异常恢复

```text
walltime
= 非长尾有效并行工作
+ 长尾低利用率有效工作
+ 长尾后无效等待
```

前两项评价调度数据面；最后一项评价 timeout、OOM、sandbox 回收、异步 release、评分和汇总。只比较 walltime 容易把异常更快收敛误判为调度更好。

## 5.2 Sticky 与 kvcaware 的公平比较单位是初始绑定质量与有效长尾

两组都是 sticky，关键不在“是否迁移”，而在首次绑定是否把未来成本更均匀地分散。每个 context 应并排比较：

```text
首次绑定的 session 数 / prompt token 总量
有效活跃 replica 曲线与有效长尾
prefix-hit
OOM、instance failure、timeout session
长尾后无效等待
```

只有 `min(inflight_tokens)` 在不恶化异常/timeout 的前提下缩短有效长尾，才能认定为 token-aware 初始分配收益。

## 5.3 Overload 应拆分 KV 保护与 agentic 长尾感知

不建议单纯降低 `lt`。当前 `load_threshold` 同时控制：

```text
sticky overload：kv_cache_usage_perc > lt
capacity eligibility：avail >= cap × (1-lt)
```

例如将 `lt` 从 0.9 降到 0.2，虽然会更早解除 sticky，却也要求 fallback 副本保留 80% KV 容量；两个语义被耦合，难以调参。

应拆为独立参数：

```text
sticky_overload_threshold：何时解除 sticky
capacity_reserve_threshold：fallback 时至少保留多少 KV 容量
```

保留 KV usage 作为容量安全信号，并加入持续偏斜的 agentic 热点信号，例如：

```text
active_sessions 高
或持续的 running 高 + inflight_tokens 高
或近期 timeout / OOM 高
```

`active_sessions` 最贴近 sticky 的长期归属：session 首次绑定时加一，成功/失败/timeout 终止时减一。当前 `inflight_count`、`inflight_tokens` 和 `running` 都只观察正在进行的 LLM request；session 在 tool/sandbox 阶段可能不占这些值，却仍会回到原 replica。

新判据应采用相对阈值和持续窗口，避免短 spike 破坏 cache 局部性；只在 session 的**下一轮 LLM request**到来时解除 sticky 并 fallback。它不能迁移已经卡在 sandbox/tool 的 session，后者仍需 timeout 与异常回收处理。

## 5.4 后续 capacity 实验需要显式验证 fallback / 无 shortcut 行为

若要验证真正的 `remaining` capacity 路由，应显式配置 `do_shortcut=false` 或可控触发 overload fallback，并记录 capacity 分支调用次数、后续 request 的 replica 分布、prefix-hit、有效长尾及异常变化。当前常规分支为确定性 `max(remaining)`，应评估是否需要 top-k、容差或概率化分流。

## 5.5 代码已演进（2026-08-22/23 落地）

> **本文档记录的是 exp1 实验当时的代码行为（容器 `e60b08d2`）。** 本节启发已落地为 P1-P5 五项改进并推送：

| 启发来源 | 落地 |
|---|---|
| §5.2 首绑质量（count 信号 + 挂死 session 不可见） | P1 `active_sessions` 首绑（binding 表派生 + gateway 终态桥接）+ P2 窗口加权随机 |
| §5.3 overload 拆分（KV 保护 vs agentic 长尾感知、相对阈值 + 持续窗口） | P3 阈值双语义拆分 + P4 `OverloadMode.SKEW`（池中位数持续偏斜） |
| §5.4 容差/概率化分流 | P5 near-top 随机（`tie_epsilon`） |

当前策略语义见 `strategy.md`；commit 号与参数表见 `README.md`"当前代码状态"；实现方案与决策记录见 `planning/20260822-router-policy-improvements/`。**Phase 7 真机重跑后，本文的基线表是对照组。**

---

# 6. 分析方法记录

## 6.1 数据来源

| 目标 | 日志 / 指标 |
|---|---|
| router 原始账本 | `router-dispatch ... dispatched=... completed=...` |
| 路由目的地与首次绑定 | `routed to server=`、`CAPACITY_TOKEN_AWARE winner=` |
| vLLM 运行态 / prefix | `vllm-metrics` 或 rl-insight Prometheus 的 `running`、token、KV 指标 |
| session 生命周期 | `agent_runner_ray_task start/end ... sample_index=... session_index=... elapsed=...` |
| 异常 | OOM、`invoke response failed`、`instance not exist` |
| 汇总 | `generate_sequences summary`、`=> Resolved`（仅辅助） |

## 6.2 重建流程

1. 从 `router-dispatch` 合并完整 replica 快照，计算 `raw_inflight = dispatched - completed`。
2. 以 `(sample_index, session_index)` 建 session 表，记录 start/end/elapsed 及 OOM/timeout 证据。
3. 标记接近 `run_timeout` 的 timeout-terminated session。
4. 用 `routed to server=` 建立 session → request → replica 映射，得到每 replica 的 `hung_inflight`。
5. 重建：
   ```text
   effective_inflight = raw_inflight - hung_inflight
   ```
6. 用有效 inflight 计算非长尾、有效长尾和长尾后无效等待；再按相同 context 对比两策略。

## 6.3 对账方法与结果

对账已按以下方式完成：

```text
1. router 日志 request=session-<sample>-<session>-<uuid>
   直接解析出 (sample_index, session_index)；
2. 与 agent_runner_ray_task start/end 按 (sample, session) join；
3. elapsed >= 7000s 判为 timeout-terminated（挂死）session；
4. 挂死 session 的全部 routed to server= 路由从有效工作剔除；
5. 每 replica 取最后一个有效路由时刻；
6. 按空闲比例（半数 replica 完成 → 长尾开始；全部完成 → 长尾结束）
   计算 3.3 的有效阶段表。
```

剩余限制：runner 日志存在 Ray 折叠/丢失（11～70 个 session 无 end 行），这类 session 的路由仍计入有效工作；其比例低于 12%，对边界影响为分钟级。
