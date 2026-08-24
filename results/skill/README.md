# kvc-aware Router：分析指南（README / 索引）

> 为 **6 节点 Ascend 多并发 × 多 context 矩阵实验**（`multi-node-ascend-exps.sh`）
> 的负载均衡/路由行为分析准备。供后续读日志 / rl-insight / 轨迹时快速对齐，
> 不重复部署与排查过程（见 `../README_zh.md` / `../ASCEND_QUICKSTART_zh.md`）。
>
> 策略与指标文档（`strategy.md` / `metric.md`）跟随 host 分支 `router-dev` 当前代码；
> 实验记录（`exp1.md` / `exp2.md`）描述**实验当时**运行的代码（exp1/exp2 为容器
> `e60b08d2` + exp2 叠加 commit `3cfd111f`），文档内均有版本说明。
> 记录时间：2026-08-20；最近更新：2026-08-24（P1-P5 落地后同步）。

---

## 当前代码状态（P1-P5，2026-08-22/23 落地）

exp1/exp2 分析定下的五项策略改进已全部实现并推送（方案与决策：
`planning/20260822-router-policy-improvements/`）：

| 项 | commit | 内容 | 参数（默认） |
|---|---|---|---|
| P1 active_sessions | verl `0bd3ea8f` + uni-agent `44f1b84` | 首绑按存活 session 数选（binding 表单一事实源；gateway 终态回调 → Ray fire-and-forget `on_session_end` → binding 失效） | 无（默认生效） |
| P2 窗口随机首绑 | `87faf46a` | 候选 = `min(active_sessions) ± window`，窗内加权随机 | `first_bind_window=1`，`first_bind_weighted=true` |
| P3 阈值双语义拆分 | `c875d1f7` | overload 判据与容量门槛各自独立阈值 | `sticky_overload_threshold` / `capacity_reserve_threshold`（None → 回落 `load_threshold`） |
| P4 SKEW overload | `e93e049b` | 池中位数持续偏斜判过载（sessions 绝对差 OR running+tokens 比值 AND） | `overload_mode=skew` 时生效；`skew_window=60` / `skew_delta=2` / `skew_factor=2.0` |
| P5 near-top 随机 | `29ec0085` | capacity top 集放宽为 `remaining ≥ best − cap×ε`，均匀随机——**修复 exp2 发现的单赢家崩塌** | `tie_epsilon=0.01` |

全部参数已暴露到 `examples/kvc_aware_router/parallel_infer.py` CLI（`3a13c487`）。
**降级语义**：uni-agent 侧无桥接（旧版）时 `active_sessions` 退化为累计首绑数——仍是
合法首绑信号，但失去"避开挂死 session 所在副本"的存活语义。

---

## 文档索引

| 文件 | 内容 |
|---|---|
| `strategy.md` | 当前 `verl/workers/rollout/router/` 的 shortcut、overload 与 slow-cut 策略流程和关键公式（含 P1-P5） |
| `metric.md` | 当前源码定义的 rl-insight 指标、采集链路、派生口径与日志关键字（含 `active_sessions`） |
| `framework.md` | **64×8 session 展开**、`Resolved` 判据、session 终态桥接（P1），以及它与 router inflight 生命周期的区别 |
| `analysis.md` | 调度长尾、长尾后无效等待等**通用指标定义**、判读建议，以及 exp1 的关键调度结论 |
| `exp1.md` | **exp1 单一事实来源**：环境、策略配置、结果表、关键发现、启发与可复现分析方法 |
| `exp2.md` | **exp2 单一事实来源**：128K sticky vs 无 shortcut capacity routing，以及 commit `3cfd111f` 的实测影响 |

推荐阅读顺序：`strategy → metric`（了解当前策略与数据），需要判读时用 `analysis`，
复盘 exp1/exp2 分别看对应的单一事实来源文档。

---

## Exp1 范围（固定口径）

后续提及 **exp1** 时，统一指 `results/result-0820/` 下、固定 `concurrency=128` 的完整 **2 策略 × 4 context** 对比矩阵，**共 8 个日志**；不包含 `16×*`、`96×128K`，也不包含 `results/result-0821/` 的重跑。

| 策略 | context | 日志 |
|---|---|---|
| sticky（least-inflight） | 16K / 32K / 64K / 128K | `infer-sticky-prompt64x8-128x{16384,32768,64000,128000}-n6.log` |
| kvcaware（capacity-token-aware，`lt=0.9`） | 16K / 32K / 64K / 128K | `infer-kvcaware-lt0.9-prompt64x8-128x{16384,32768,64000,128000}-n6.log` |

exp1 的长尾分析以 router 生命周期内的 `raw_inflight = dispatched_count - completed_count` 为原始信号；最终性能口径使用 `effective_inflight = raw_inflight - hung_inflight`。`hung_inflight` 由事后识别的 runner-level timeout session 校正：其 `agent_runner_ray_task end` 的 `elapsed` 接近配置的 `run_timeout`。长尾开始为有效 inflight 已归零的 replica 比例首次达到报告阈值，长尾结束为所有 replica 的有效 inflight 均为零；长尾结束后至实验结束的时间归为长尾后无效等待。prefix-hit 使用每 replica 的累计 `prefix_cache_hits / prefix_cache_queries`；报告时必须明确是 replica 算术均值还是 query 加权总体值。详见 `analysis.md` 与 `framework.md`。

### Exp1 性能统计（日志重建）

时间统一以分钟计。开始点是首条 `router-dispatch`（首次真实派发），结束点是最后一轮 `vllm-metrics`；prefix-hit 为末尾累计计数的 **12 replica 算术平均**。

当前表的长尾列仍是**原始 router 账本的初步重建**：边界从按时间合并的完整 12-replica `router-dispatch` 快照恢复。由于 release 是异步 RPC，且日志已确认存在接近 runner `run_timeout≈7200s` 的 timeout-terminated session，原始 `raw_inflight` 会把无有效进展的等待误算为工作；因此 `—` 不是“没有长尾”，而是尚未完成 timeout session → router request/replica 的事后对账，不能可靠填入有效长尾与长尾后无效等待。

“超时 session（已观察）”按日志中 `agent_runner_ray_task end` 的 `elapsed ≥ 7000s` 统计，仅表示可解析到的下界（部分 Ray 日志可能被聚合/折叠），不等同于全部 512 个 session 的完整终态计数。

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

> 下一版有效性能表将以 `(sample_index, session_index)` 对账 session 的 start/end/timeout/OOM 状态，并通过 `routed to server=` 将 timeout session 映射到 replica，重建 `effective_inflight` 后填写“有效长尾”和“长尾后无效等待”。在此之前，不能把 `Resolved` 或 `raw_inflight=0` 当作 session 级工作收敛的充分证明。

---

## 快速结论速查（TL;DR）

> 以下为 exp2（0821，容器 `e60b08d2` + commit `3cfd111f`）时的结论；**两处实现问题已修复**，见上方"当前代码状态"。

- **两套策略**：sticky = 全生命周期粘性（实验当时首绑按 inflight 个数最少；当前代码已改为 active_sessions 窗口随机）；kvcaware = 无粘性（按 `remaining = avail - need` 最大选）。
- **128×128000 实测**：两者 resolve 都是 `3/64`（OOM 主导）；walltime ~3h55m vs ~4h39m。
- **prefix-hit**：sticky ~90.4% vs kvcaware ~39.8%（粘性带来缓存局部性，kvcaware 因无粘性 + 单赢家问题前缀分散）。
- ~~发现一个策略实现 bug：`top_idx` 浮点严格相等 → 单赢家崩塌~~ **已修**（P5 near-top 随机，`tie_epsilon=0.01`；exp2 实测 40,292 次路由中严格相等仅 75 次）。详见 `exp2.md` §4.3。

---

*版本说明：策略/指标文档跟随 `router-dev` 当前代码（P1-P5 已落地）；exp1/exp2 记录实验当时的代码行为，文档内有各自版本说明。代码再变时同步更新 `strategy.md`、`metric.md` 与本页"当前代码状态"。*
