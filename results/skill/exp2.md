# Exp2：128K sticky vs 无 shortcut capacity routing

> **范围**：`results/result-0821/` 下的两个 128 并发、128K context 实验。exp2 在 exp1 基础上叠加了 commit [`3cfd111f`](https://github.com/yyyyrrrrif/verl/commit/3cfd111f51402cf71214f969a77e14b7d6adba5e)：移除 capacity 冷启动分支、为真实 top tie 增加随机选择，并在派发时立即记录 prefix reverse index。

---

# 1. 环境与实验范围

## 1.1 运行环境

| 项目 | 配置 |
|---|---|
| 集群 / 推理副本 | 6 节点 Ascend；12 replica（`6 × 8 NPU / TP=4`） |
| 模型 | Qwen3.5-9B（以实际容器配置为准） |
| 样本与 session | `MAX_SAMPLES=64`，`--n 8`，共 512 session |
| session 并发上限 | 128 |
| runner / sandbox timeout | 7200s / 600s（hard deadline 630s） |
| context / 并发 | 128K / 128 |
| 叠加代码 | `3cfd111f remove cold start and add random choice in routing` |

## 1.2 两个实验

| 策略 | 日志 |
|---|---|
| sticky | `results/result-0821/infer-sticky-prompt64x8-128x128000-n6.log` |
| kvcaware lt=0.9 | `results/result-0821/infer-kvcaware-lt0.9-prompt64x8-128x128000-n6.log` |

## 1.3 数据位置与时间对齐

| 数据 | 路径 |
|---|---|
| 原始日志 | `results/result-0821/infer-*.log` |
| rl-insight 根目录 | `results/result-0821/rl-insight/` |
| Prometheus TSDB | `results/result-0821/rl-insight/prometheus/` |
| Grafana | `results/result-0821/rl-insight/grafana/` |

沿用 exp1 已验证的口径：日志时间字符串直接作为 Prometheus 的 UTC 查询时间；使用新 replica port 的 `dispatched_count` 从零增长作锚点复核。

---

# 2. 策略与 commit

## 2.1 实际运行配置

| 字段 | sticky | kvcaware |
|---|---|---|
| `slow_cut` | `least-inflight` | `capacity-token-aware` |
| `do_shortcut` | `true` | `false` |
| `overload_mode` | `None` | `None` |
| `load_threshold` | 0.6 | 0.9 |
| collector poll 配置 | 0.05s | 0.05s |

日志行为验证：

- sticky 有 `38,606` 条 `STICKY HIT`；
- kvcaware 的 `STICKY HIT=0`，有 `40,292` 次 capacity routing；
- 因此 exp2 是有效的：**全生命周期 sticky** 对比 **每轮无 sticky 的 capacity routing**。

## 2.2 Commit `3cfd111f` 的影响

1. **移除 cold start**：capacity 策略不再以 `min(inflight_tokens)` 分配 session 首轮；每个 request 都走容量排序。
2. **真实 top 随机选择**：相同最高分副本会随机选一个，避免稳定排序恒选 pool 第一个。
3. **即时记录 dispatched prefix**：请求派发时立刻写入 prefix→replica reverse index；不再等待约 6～9 秒批量 KV event，后续相同 prefix 更快看到 `gpu_hit`。
4. **poll 配置改为 0.05s**：日志配置 dump 确认已生效；实际周期仍应从时序数据核验。

capacity 计算仍是：

```text
avail     = cap × (1 - kv_cache_usage_perc)
need      = prompt_len × (1 - gpu_hit)
remaining = avail - need
```

先按容量门槛 `avail >= cap × (1-lt)` 分组，再按 `remaining` 排序；严格浮点 `remaining == best_remaining` 使绝大多数调度仍只有一个 top，随机选择仅在真实相等时生效。

---

# 3. 实验结果

## 3.1 总体性能

| 策略 | Resolved | Prefix-hit（replica 均值） | Prefix-hit（query 加权） | Walltime | 原始非长尾 | 原始长尾 | 原始长尾后等待 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sticky | 3/64 | 90.19% | 90.44% | 209.7 min | 124.9 min | — | — |
| kvcaware | 3/64 | 38.08% | 39.81% | 255.9 min | 205.9 min | 23.5 min | 26.5 min |

时间从首条 `router-dispatch` 到最后一轮 `vllm-metrics`。原始长尾以 `raw_inflight = dispatched-completed` 恢复；sticky 组 raw 账本在日志窗口内未稳定清零（`—`）。

## 3.1b 有效阶段耗时（session 对账后）

按 exp1 已验证的对账口径（`request=session-<sample>-<session>` join runner start/end；`elapsed>=7000s` 判挂死并剔除其路由；按每 replica 最后有效路由划界）：

| 策略 | Walltime | 有效非长尾 | 有效长尾 | 长尾后无效等待 | timeout session | 挂死路由数 |
|---|---:|---:|---:|---:|---:|---:|
| sticky | 209:39 | 123:43 | 45:20 | 40:37 | 12 | 553 |
| kvcaware | 255:53 | 212:15 | 8:55 | 34:42 | 15 | 1,417 |

有效口径下的解读与 raw 口径相反：kvcaware 的**有效长尾反而更短**（8:55 vs 45:20），但其有效非长尾极长（212 min vs 124 min）——无 sticky 的逐轮 capacity routing 让集群更久保持“半数以上 replica 活跃”的状态，代价是整体吞吐更慢（低 prefix-hit、重 prefill），最终 walltime 仍多 46 分钟。且其挂死路由（1,417）是 sticky 的 2.6 倍，逐轮重路由把大量请求花在了最终挂死的 session 上。

kvcaware 相比 sticky：

```text
walltime：+46.2 min（+22.0%）
prefix-hit：-52.1 pct（replica 均值）
Resolved：相同（3/64）
```

## 3.2 路由分布

| 指标 | sticky | kvcaware |
|---|---:|---:|
| 全部 LLM request 路由数 | 39,118 | 40,292 |
| 唯一 request/session id | 512 | 512 |
| 后续 request 改变 replica | 0 | 36,044 |
| 两台 27.24 的全部路由数 | 902（2.3%） | 882（2.2%） |
| 全部路由数最小 / 最大 | 439 / 4,534 | 439 / 4,525 |

两种策略的总体 request 分布都很不均；27.24 仍然显著被冷落。kvcaware 虽然逐轮重路由，但并没有修复这一副本偏斜。

---

# 4. 关键发现

## 4.1 无 sticky 明显降低 prefix cache 局部性，却没有提高 resolve

kvcaware 每轮重新选 replica，prefix cache 不跨 replica 共享；其 prefix-hit 从 sticky 的约 90% 降至约 38%～40%。这意味着更多 prompt token 必须重新 prefill，KV/locality 收益被显著削弱。

尽管作了即时 prefix reverse-index 写入，kvcaware 的最终 hit 仍远低于 sticky，说明“同一 session 留在同一 replica”的局部性收益远大于该补偿。两组 Resolved 都是 3/64，未见质量收益。

## 4.2 移除 cold start 后，capacity 分支没有带来更快的收敛

commit 移除 `min(inflight_tokens)` cold start 后，kvcaware 的每轮选择由 `remaining` 决定。它比 sticky 多耗时 46.2 分钟，非长尾阶段也更长：205.9 vs 124.9 分钟。

这不是单纯长尾收尾问题；在大部分有效运行阶段，capacity routing 已经更慢。结合低 prefix-hit，主要解释是频繁跨 replica 路由造成的 prefill 重计算与 KV locality 损失。

## 4.3 随机 tie-break 对连续 remaining 几乎没有实际覆盖

commit 只在多个 replica 的最终 score 严格相等时随机。`remaining` 是连续浮点；日志中 capacity 路由共有 40,292 次，但极少出现可并列的 top。因此随机选择避免了真正同分时的 pool[0] 偏置，却不能平滑“接近同分”的副本，也不能防止单一 `max(remaining)` 决策。

## 4.4 27.24 仍被冷落：capacity 信号存在正反馈

两台 27.24 在 kvcaware 中仅收到 882/40,292（2.2%）路由，和 sticky 的 902/39,118（2.3%）几乎相同。它们不是被重路由机制重新利用，而是持续被排除。

机制是：

```text
prefix-hit 低
→ need 高
→ remaining 低
→ 更少被选
→ 难以积累本地 prefix
→ prefix-hit 继续低
```

数字取证（0821 原始分析保留）：27.24 每次都参与打分（`score(): emit replica=` 计数 17,794/17,795，与其他 replica 相同）但几乎从不进 `winners=`（499/19,391，2.6%）；其 `need` 均值 20,883/21,125 比其余 replica 的 16,828~17,919 大 ~25%（`gpu_hit` 仅 0.245/0.316，全集群最低）。注意其 `avail`/`remaining` 均值与其余 replica 相当（679K vs 676~687K）——**不是真的容量满**，是低命中 → 高 need 的策略性冷落。

即时 dispatch prefix 记录缩短了“刚派发后尚未收到 KV event”的盲窗，但不能解决已形成的低命中副本正反馈。

> **修复状态**：本节描述的“严格浮点相等 → 单赢家”已在 host 代码修复（P5 near-top 随机，`tie_epsilon=0.01`）；但正反馈本身（低命中 → 高 need → 少选）是 capacity 信号的结构性质，P5 只消除其被浮点比较放大的部分。低命中副本的保护（探索配额/最小份额）仍是开放问题（见 §5.3）。

## 4.5 OOM 仍是低 resolve 的主导因素，且 kvcaware 更重

| 信号 | sticky | kvcaware |
|---|---:|---:|
| sandbox OOM 日志 | 14,841 | 15,769 |
| `invoke response failed` | 18,452 | 24,306 |
| `instance not exist` | 3,621 | 8,549 |
| 已观察 runner timeout log | 12 | 15 |

两组均是 128K × 128 下的系统性 OOM；kvcaware 的失败连锁事件更多。故 3/64 resolve 首先是 sandbox 内存/异常收敛问题，不能解读为正常路由质量。

---

# 5. 启发

## 5.1 对 agentic sticky，局部性是重要基线

exp2 直接表明：仅把每轮请求改为 `max(remaining)` 重新路由，会显著损失跨轮 prefix cache；若没有可验证的长尾/吞吐收益，不应关闭 shortcut。

## 5.2 随机应覆盖近似最优候选，而非严格相等

连续 `remaining` 的严格相等罕见。若目标是平滑分流，应定义近似 top 集合，例如相对 capacity gap、top-k 或按 remaining 加权抽样；然后用 prefix-hit、有效长尾和失败率共同评估。

## 5.3 需要防止低命中副本永久冷却

capacity 策略应区分“当前 prefix 少”与“节点不健康”。对于低 hit replica，可考虑探索配额、最小派发份额或基于健康指标的保护；否则 `need` 会持续惩罚冷副本。

## 5.4 OOM 与调度应分层处理

128K × 128 的 OOM 使任何策略比较都受噪声主导。后续应降低并发或 context 后先验证有效长尾，再在高压配置下验证异常恢复；同时保留 timeout/OOM 作为独立指标。

---

# 6. 分析方法记录

## 6.1 数据来源

| 目标 | 信号 |
|---|---|
| 实际策略 | `KVCAwareStrategyConfig`、`STICKY HIT`、`CAPACITY_TOKEN_AWARE winners=` |
| 路由分布 | `routed to server=` |
| router 账本 / 长尾 | `router-dispatch` 的 dispatched / completed |
| prefix-hit | `vllm-metrics` 末尾 `prefix_cache_hits / prefix_cache_queries` |
| 引擎状态 | `vllm-metrics` / rl-insight 的 running、KV usage、token counters |
| session 与 timeout | `agent_runner_ray_task start/end`，`elapsed` |
| 异常 | OOM、invoke failed、instance not exist |

## 6.2 计算口径

- 解析最后一条 metrics 计算每 replica query hit ratio；实验主值为 replica 算术平均，另报 query 加权值。
- 从首个 `router-dispatch` 到最后一条 `vllm-metrics` 计算 walltime。
- 将 `routed to server=` 按 request id 分组；同一 id 跨 replica 表示后续轮次发生重路由。
- 长尾使用完整 replica `router-dispatch` 快照；timeout session 尚未映射回 request/replica，因此当前仅为原始账本结果。

## 6.3 对账方法

与 exp1 相同：`request=session-<sample>-<session>-<uuid>` 直接解析 `(sample_index, session_index)`，与 runner start/end join；`elapsed>=7000s` 判为挂死并剔除其全部路由；按每 replica 最后有效路由划长尾边界。exp2 两组的 timeout session 首绑 replica 分布较分散（sticky 12 个分布在 5 台，kvcaware 15 个分布在 6 台），未复现 exp1 的 27.24 集中现象——这与 exp2 无 sticky 逐轮重路由一致。

## 6.4 当前限制

1. 当前 commit 不在本地历史，已从指定 GitHub 仓库按 SHA fetch 并审阅 patch；exp2 行为同时以日志验证。
2. runner 日志存在 Ray 折叠/丢失（sticky 12 个、kvcaware 11 个 session 无 end 行），其路由计入有效工作，影响为分钟级。
3. 未对 Prometheus 逐窗口展开 vLLM prefill/TPOT；现有结论以日志、最终 counters 和 router 分布为主。
