# 调度性能分析：指标与实验结论

> 本文定义用于多 replica、多 session agent rollout 的调度分析口径，并记录 exp1 的阶段性结论。指标定义不绑定特定实验的 replica 数、并发或 context；具体实验将其代入自身的副本数和阈值。

---

# 一、关键指标的定义和采集计算方式

## 1. 分析对象与三层生命周期

一次 agent rollout 同时存在三个不同的生命周期；分析时不得混用它们。

```text
Agent session 生命周期
  session 创建 → runner / tool / sandbox → finalize → score → TQ 写入

LLM request 生命周期
  router acquire → vLLM generate → router release

实验生命周期
  首个有效工作开始 → framework 收敛 → 结果汇总 / 资源清理
```

- **session** 是端到端任务执行单元；一个 sample 可展开多个 session。
- **LLM request** 是 session 的一次模型调用；单个 session 可以产生多轮 request。
- **实验结束**是 driver/framework 的控制面事件，不天然等价于前两者均已收敛。

## 2. Router inflight：调度工作量的基础信号

对每个 replica：

```text
raw_inflight = dispatched_count - completed_count
```

数据来自 `router-dispatch` 日志或 rl-insight 的累计 counter：

```text
router-dispatch replica=<id> dispatched=<N> completed=<M>
```

它表示 router 已记录 acquire、但尚未记录对应 release 的 LLM request 数。它适合恢复**调度面工作分布**，但不等价于：

- 当前仍在 vLLM 执行的 request（使用 `num_requests_running` 观察）；
- 尚未结束的完整 agent session；
- GPU 一定仍有有效工作。

原因是 router release 是异步 RPC；实验/agent session 可以收敛，而部分 release 尚未在 LoadBalancer actor 中完成记账。

## 3. 有效 inflight 与超时挂死 session

为分析真正的调度长尾，应从 `raw_inflight` 中区分仍在有效推进的请求与已无有效进展、仅等待超时回收的请求。

```text
effective_inflight(replica)
= raw_inflight(replica) - hung_inflight(replica)
```

其中 `hung_inflight` 是通过 session 生命周期**事后识别**的挂死请求。建议的最小判据：

```text
agent_runner_ray_task 的 elapsed 接近 runner 级 run_timeout
```

例如，若配置 `run_timeout=7200s`，可使用带容差的 `elapsed >= 0.98 × run_timeout` 标记 timeout-terminated session；不要机械匹配一个固定秒数。可用以下证据增强归因：

- `sandbox was oom-killed by the kernel`；
- `instance not exist`；
- `invoke response failed`；
- sandbox command 的 timeout / hard deadline；
- `(sample_index, session_index)` 的 runner start/end 对账。

注意：**OOM 不是挂死的必要条件**。无 OOM 的 session 也可能达到 runner 级 timeout；反过来 OOM 后的 session 也可能及时异常退出。

## 4. 长尾：调度收敛效率指标

### 4.1 调度含义

长尾不是简单的“最后几个任务耗时很长”，而是：

> 集群已从高并行工作状态转入多数 replica 提前闲置、少数 replica 被少量残留有效工作拖住的低利用率阶段。

它衡量调度把工作从全局并行阶段收敛到低利用率收尾阶段时的损失。长尾越长，说明少数慢 session、异常恢复、粘性绑定或路由集中越强地支配实验 walltime。

### 4.2 边界定义

设实验共有 `R` 个 replica，`effective_inflight(r, t)` 为 replica `r` 在时刻 `t` 的有效在途请求数。设长尾进入比例为 `p`（默认可取 `0.5`，但应在每个实验报告中显式记录）。

```text
tail_start
= 首次满足：空闲 replica 数 / R ≥ p 的时刻

其中，空闲 replica 条件为：effective_inflight(replica) = 0

tail_end
= 首次满足：所有 replica 的 effective_inflight = 0 的时刻
```

由此得到：

```text
non_tail_duration = tail_start - workload_start

tail_duration = tail_end - tail_start

tail_ratio = tail_duration / (tail_end - workload_start)
```

`workload_start` 建议取首个有效 router dispatch；边界从按时间合并的完整 replica 快照中恢复，避免同一轮逐 replica 写日志带来的毫秒级偏差。

### 4.3 长尾后无效等待时间

```text
post_tail_invalid_wait = experiment_end - tail_end
```

它表示所有**有效调度工作**已结束后，实验仍未结束的控制面等待。常见组成：

```text
挂死 session 等待 runner timeout
+ OOM / instance 异常后的清理
+ 异步 router release 未落账
+ Gateway / sandbox 回收
+ score、TQ、driver 汇总与日志 flush
```

因此它不应被解释为模型正常推理耗时，也不应混入长尾以评价路由数据面的吞吐能力。它是异常恢复与控制面收敛质量的独立指标。

## 5. 其他配套指标

| 指标 | 计算 / 来源 | 调度意义 |
|---|---|---|
| prefix-hit | 每 replica `prefix_cache_hits / prefix_cache_queries`；报告 replica 均值和 query 加权值 | 衡量 KV/prefix 局部性；不能单独解释 walltime |
| `num_requests_running` | `vllm-metrics` poll gauge | 引擎实际运行请求；用于确认 GPU 侧是否仍忙 |
| dispatch share | 每 replica `Δdispatched / ΣΔdispatched` | 检查派发集中或副本被冷落 |
| effective active replicas | `count(effective_inflight > 0)` | 绘制长尾期间集群利用率衰减 |
| OOM / invoke failure | sandbox / runner 日志关键字 | 区分正常慢 session 与失败恢复导致的尾部 |
| timeout-terminated session 数 | runner `elapsed` 对齐 `run_timeout` | 量化长尾后无效等待的主要来源 |

## 6. 统计与解释原则

1. **walltime 分解**：
   ```text
   walltime = 非长尾有效并行 + 长尾低利用率 + 长尾后无效等待
   ```
2. 评价路由数据面时，重点比较前两项；第三项单列为稳定性/回收问题。
3. `Resolved` 仅是已捕获 score 的结果汇总哨兵，不是 router inflight 清零判据，也不是所有 session 成功判据。
4. `raw_inflight` 只能作为原始调度账本；涉及有效工作结束时，必须结合 timeout-terminated session 做事后校正。
5. 同 context 的策略对比必须同时报告 prefix-hit、长尾、异常/timeout 指标；仅比较 walltime 容易把“失败更快收敛”误判为“调度更好”。

## 7. 判读速查（面板与日志）

实验进行中 / 事后判读的检查顺序：

1. **先看收敛**：是否 `=> Resolved`；`dispatched/completed` 差是否归零。
2. **负载均衡**：`num_requests_running` 每 replica 一条线——热点 = 一条持续高、其余低位；`inflight_tokens` "单边爬升不回落"是长驻长尾预警；"派发少但 running 高"= 被长驻拖住，"派发多且 running 高"= 被灌满。
3. **capacity 路由实验**：`routed to server=` 时序分布是否总指向同一台；`gpu_hit` 是否偏斜（前缀集中 → need 偏斜 → 选人偏斜）；多 replica `avail` 接近时胜负只看 remaining。
4. **sticky 对比实验**：比较 running/派发分布差异，验证"无粘性是否缓解单点长尾"。
5. **面板陷阱**：`avail/need/remaining` 是 histogram 且桶为 None——p50 可能失真，用日志 `score(): emit replica=... avail= need= remaining=` 的精确值。
6. **128K 慢速判读**：`num_gpu_blocks`(~914-931)/token 容量小，长 context 单请求占用大；`inflight_tokens` 对比 `max_num_seqs=512` 判断是否 token-受限（而非数量受限）。
7. **崩塌分布排查**：若某 replica 派发接近 0、其余均分——先核实该 replica 的 `need`/`gpu_hit`/`remaining` 是否其实与别人相当（策略性冷落 vs 实现偏差），再下"满/空"结论。

---

# 二、实验中的关键结论

## 1. Exp1 的分析范围

exp1 是 `results/result-0820/` 中固定高并发的完整策略 × context 对比矩阵：sticky（least-inflight）与 kvcaware（capacity-token-aware, `lt=0.9`）在相同 context 下配对比较。具体日志集合和已有性能表见 `README.md` 的“Exp1 性能统计”。

## 2. Prefix cache 局部性不是 exp1 walltime 差异的主解释

相同 context 下，sticky 与 kvcaware 的最终 prefix-hit 非常接近，差异为小数百分点量级；两种策略的总体命中率均处在较高水平。

因此，对 exp1 中数十分钟级的 walltime 差异，不能简单归因为 prefix-hit。更需要检查：

- session 的完成时长分布；
- 长尾中有效活跃 replica 是否集中；
- OOM、instance 消失与 runner timeout；
- 长尾结束后的无效等待；
- router release 账本是否收敛。

## 3. Sticky 的核心取舍：缓存局部性与不可迁移长尾

sticky 的首次请求按最小 inflight 个数选择 replica，随后同一 request/session 的后续轮次回到既定副本。它的收益是 KV/prefix 局部性；代价是：

```text
快 session 已结束 → 其 replica 提前空闲
慢 session / 异常 session 仍绑定原 replica → 空闲 replica 无法接管后续轮次
```

因此 sticky 的长尾反映的是**会话生命周期不均衡被粘性固定放大**。长尾越长，不表示 prefix cache 一定无效，而是说明局部性收益没有抵消会话时长方差和异常恢复成本。

## 4. KVC-aware 的理论优势与实测风险

kvcaware 在关闭 shortcut 时，每轮请求可重新选择 replica；理论上应减少一个 session 被固定在慢副本造成的尾部，并可依据 KV 可用容量和 prefix 命中估计做转移。

但 exp1 的实际实现和运行条件有两个风险：

1. **近似确定性单赢家派发**：`remaining` 是连续浮点，而 top 集合用严格相等判断；实际常只有一个 top，随机 tie-break 几乎不生效。相近容量副本未被平滑分流，可能形成派发集中。
2. **状态滞后与正反馈**：路由所见 `kv_perc` / running 来自约百毫秒量级 poll；并发脉冲可同时选择同一“最优”副本。低 prefix-hit 的副本会有更大的 `need`，从而更少被选，进一步难以积累 prefix cache。

所以 kvcaware 的长尾缩短才可视为调度改进；若 walltime 更短但只是 timeout/OOM 更早收敛，不能称为策略收益。

## 5. OOM、timeout 与 router 残留 inflight 必须分层解释

exp1 中可见 sandbox OOM、`invoke response failed`、`instance not exist` 和接近 runner `run_timeout` 的长 session。它们可导致 session 在有效工作停止后仍消耗实验时间。

同时，router release 是 fire-and-forget RPC；即使 session 已在 AgentFramework 中成功或失败收敛，`dispatched_count - completed_count` 仍可能短暂或永久残留。

因此：

```text
Resolved
≠ 所有 session 成功
≠ 所有 session 有效工作仍在进行
≠ router inflight 已清零
```

分析时应先把接近 `run_timeout` 的 session 标记为 timeout-terminated，再用有效 inflight 定位长尾结束；剩余等待归入“长尾后无效等待”。

## 6. 后续分析优先级

1. 为每个实验建立 `(sample_index, session_index)` 的 session 表：start、end、elapsed、成功/失败、OOM/instance/timeout 证据。
2. 用 session 与 `routed to server=` 的 request/replica 信息对齐，计算每 replica 的 `hung_inflight`，得到有效 inflight 时间线。
3. 以有效 inflight 重算长尾、长尾比例与长尾后无效等待，并对同 context 的两种策略并排比较。
4. 对长尾最严重的实验，单独检查长尾阶段活跃 replica 数、dispatch share、`num_requests_running` 和 OOM/timeout 事件时间线。
5. 修复/规避容量策略的单赢家集中后，以相同指标复跑，验证改进是否来自更短的有效长尾，而非失败路径变化。
