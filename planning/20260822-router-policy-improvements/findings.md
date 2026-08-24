# Findings: kvcaware 路由策略改进

> 本 plan 的研究发现。exp1/exp2 的完整分析见 `results/skill/exp1.md`、`exp2.md`；本文件只记与实现直接相关的代码事实与推导。

## §1 数据依据（为什么是这 5 项）

| 改进项 | exp 证据 |
|---|---|
| P1 active_sessions 首绑 | exp1 首绑分布 10–63（sticky CV 0.420 / kvcaware 0.313），远差于均匀 ~43；根因是 inflight_count 在工具阶段归零 → "空台"持续吸绑 |
| P2 加权随机首绑 | 计数信号取值小、并列常态；确定性 tie-break 使落点取决于遍历顺序 |
| P3 阈值拆分 | exp1 `lt=0.9` 下 sticky overload 永不触发（KV 峰值 0.22，四组 fallback=0）；同一 lt 又决定 capacity 门槛，调一个动两个 |
| P4 组合 overload | 27.24 形态：running 12.98/11.73（集群其余 1.9–4.0）、inflight-token CV 0.68、KV 仅 0.11–0.22 → 单 KV 判据结构性抓不到 |
| P5 近似 top 随机 | exp2 实测：40,292 次 capacity 路由，严格 `==` 仅 75 次多赢家（0.2%），且全在冷启动 15s（metrics 全零并列）；运行期仍是 argmax(remaining) |

## §2 有效口径下的策略基线（对账后，写死在案供 Phase 7 对比）

exp1 有效 walltime（非长尾+长尾，剔除 idle）：

| Context | sticky | kvcaware(token) | kvcaware 相对 |
|---|---:|---:|---:|
| 16K | 111:30 | 120:24 | 慢 8:54 |
| 32K | 120:04 | 120:05 | 持平 |
| 64K | 164:21 | 159:03 | 快 5:18 |
| 128K | 138:13 | 164:51 | 慢 26:38 |
| 合计 | 534.2 | 564.4 | 慢 30.2 |

→ 原始 walltime 的"16K/128K kvcaware 更快"全部由 idle 差异（43:52 vs 0:45 等）贡献，有效吞吐无净收益。**count 系首绑是数据上的胜者。**

## §3 现有代码触点（Phase 0 前置核对清单）

| 文件 | 角色 | 与各项的关系 |
|---|---|---|
| `strategies/kvc_aware.py` | score() 入口、sticky shortcut、三种 slow_cut | P1 首绑分支、P3 两处读阈值、P4 is_overloaded、P5 top_idx 判定 |
| `strategies/routing.py` | 加权排序 + random tie-break（骨架已有） | P5 只需候选集变宽；P2 若复用则在此加窗口 |
| `balancer.py` | acquire/release server、on_acquire/on_release 回调、sticky binding | P1 新增 `on_session_end` Ray 方法（invalidate + _fire） |
| `collectors/decoder/basic/sticky.py` | on_acquire→put binding | P1 不改（put 路径已在 collector `_write_sticky_update` → DataStore） |
| `store/data_store.py` | put/invalidate/get binding | P1 三个写路径内同步维护 ACTIVE_SESSIONS（put 换 replica 时 −旧+新；invalidate −1） |
| `store/per_request_store.py` | LRU(10000) 逐出 | P1 逐出钩子同步 −1（计数不泄漏）；512 session 远不及上限 |
| `store/per_replica_store.py` | per-replica 计数 | ACTIVE_SESSIONS 落点（METRIC_SPECS 注册即可 incr） |
| `config/strategy.py` | KVCAwareStrategyConfig | P3/P4/P5 新配置项 + 校验 |
| `types/metric_spec.py` / `emit_spec.py` | MetricKey/EMIT_SPECS | P1 新增 ACTIVE_SESSIONS（B-class gauge） |
| `types/overload_mode.py` | OverloadMode 枚举 | P4 新增 SKEW |
| `config/kvcaware.yaml`（trainer config） | 策略默认参数 | P3 示例默认值 |
| `../uni-agent/uni_agent/gateway/manager.py` | GatewayManager：create/finalize/abort_session | P1 终态回调注册点（finalize/abort + 注册 API，默认空、异常吞掉） |
| `../uni-agent/uni_agent/framework/entry.py` | rollout adapter（本就 verl-coupled） | P1 桥接：注册回调 → `balancer_handle.on_session_end.remote(session_id)` |
| `../uni-agent/uni_agent/llm_router/` | kvcaware 迁移副本（测试中） | **不动**（D6） |

## §4 P1 session 终态信号：binding 派生（定案，修订于 2026/08/22 二次核对）

决定性事实（核对两侧源码）：

```text
a. ../uni-agent/uni_agent/gateway/session/session.py:181
     backend.generate(request_id=self.handle.session_id, ...)
   → router 的 request_id 就是 session_id；sticky binding 表天然 = session→replica 映射。
b. verl 侧 invalidate_sticky_binding 无生产调用方 → binding 表只增不删；
   让 finalize/abort 触发失效即得"存活绑定"语义。
c. KVCAwareBalancer 是 Ray actor（base.py get_router_handle ray.remote），
   GatewayManager 在 driver 进程 → 事件只能走 Ray remote fire-and-forget
   （与 llm_server._release_server 同模式），进程内 callback 无法直达。
d. GatewayManager（manager.py:75-108）create（同步预约+回滚）/finalize/abort（幂等）
   三出口齐全，是 driver 侧唯一且正确的终态挂点。
e. ../uni-agent/uni_agent/llm_router/ 是 kvcaware 的迁移副本（balancer 有
   manager→provider 改名差异，strategies/store/types 与 verl4 一致），测试中，不动。
```

方案（详见 task_plan P1 与 D5）：ACTIVE_SESSIONS 由 binding put/替换/invalidate 同步 ±1（单一事实源派生）；GatewayManager 加通用终态回调（默认空）→ `framework/entry.py` 桥接 → `balancer.on_session_end.remote(session_id)`（fire-and-forget）→ invalidate + `_fire("on_session_end")`。

幂等性论证：重复 abort → 重复 invalidate → no-op；rebind → 表内替换自动 −旧+新；LRU 逐出 → 同步 −1。无 ±1 事件漂移面。

降级：回调未注册（非 uni-agent 部署/测试）时 binding 不失效，ACTIVE_SESSIONS ≈ 累计首绑数（dispatched 语义）——合法首绑信号，仅失"存活"语义。

## §5 P5 的实现细节（exp2 commit 对照）

commit `3cfd111f`（exp2 叠加）已落地的部分：
- capacity 分支 top_idx 为列表（多赢家同分）；
- `route()` 对 top ties `random.choice`。

需改的仅一处判定宽度：
```python
# 现状（严格相等，实测 0.2% 触发）
top_idx = [i for i in order if rows[i]["remaining"] == best_remaining and ...]
# 目标（相对 gap）
eps = cap * tie_epsilon   # 默认 0.01 → 12 replica 下窗口 ≈ cap 的 1%
top_idx = [i for i in order if rows[i]["remaining"] >= best_remaining - eps and ...]
```
注意 host 分支当前还是**单赢家版**（`top = max(...)`，无列表）——P5 需先合入 exp2 的多赢家骨架再放宽判定，或直接在 host 上实现等价逻辑。

## §6 P4 的"持续"实现选择

collector 实际 poll ≈150ms（exp1 实测，yaml 写 0.05s 有 ~3× 放大）。判"持续 60s"两个方案：
- **样本窗**：per-replica 保留最近 ~400 个 poll 的指标数组，判窗内全部超阈 —— 内存 12×400×4 float，可接受；
- **一阶 EMA**：`ema = α·new + (1−α)·ema`，判 ema 超阈 —— 实现最简，但"持续 60s"语义弱化。

倾向样本窗（判据可解释、回放可验证）；窗口长度做成 config。

## §7 验证资产（Phase 6 直接可用）

- exp1 32K 双组日志：`results/result-0820/infer-{sticky,kvcaware-lt0.9}-prompt64x8-128x32768-n6.log`
- 对账脚本思路已验证：`request=session-<sample>-<session>-<uuid>` 直接解析 (sample, session)，join runner start/end（`/tmp/reconcile.py`，正式化后入 repo scripts/）
- Prometheus 已可本地起（promtool/prometheus 二进制在 `~/.rl-insight/services/prometheus/2.54.1/`），日志时间 = UTC 查询时间已验证
- 回放判据基线：首绑 CV（0.420/0.313）、27.24 形态（running 12.98 vs median ~3）、timeout 首绑集中度（sticky 24/37、kvcaware 69/90）
## §8 文档同步：过时点审计（2026-08-24）

### README.md
- L8 "依据代码版本：容器 e60b08d2…见 01" —— 01 已删除，引用悬空
- L19-25 索引表：`strategy.md` 出现两次（重复行）；04/05 仍在索引
- L28 阅读顺序 "01 → 02 → 03" —— 三个文件都已删除/合并
- L70-71 TL;DR："单赢家 bug"（浮点严格相等）—— P5（`29ec0085`）已修：top 集放宽为 `remaining >= best − cap×tie_epsilon`
- TL;DR "sticky = 首个按 inflight 个数最少接" —— P1 后是 `min(active_sessions)` 窗口随机
- 缺：P1-P5 已落地的任何信息

### 04_分析建议.md（15 行）
通用判读建议（负载均衡/前缀命中/面板），无历史独有数据 → 并入 analysis.md。

### 05_128K实测对比（117 行）
与 exp2.md 同题材（0821 128×128000）。exp2.md 是对账后的完整版（有效阶段表、挂死路由数、session 分布）；
需核对 05 是否有 exp2 未覆盖的增量（初步 diff：05 的"27.24 单赢家案例"细节 exp2 §4.4 已覆盖）。

### framework.md
描述 session 展开与 Resolved，无 P1 桥接信息（callback → on_session_end → binding 失效）。

### exp1.md
L17 "运行版本 容器 e60b08d2" 正确（历史事实，不改）；L132 单赢家描述正确（当时代码确实如此）。
缺一个"代码已演进"指针，否则读者会以为问题仍在。

### metric.md / strategy.md
P1/P5 时已同步（active_sessions、skew、tie_epsilon 都在）——无需再改，除非 Phase 1-4 触及。

## §9 文档同步：当前代码状态速记（写 README 用）

| 项 | commit | 一句话 |
|---|---|---|
| P1 active_sessions | verl `0bd3ea8f` + uni-agent `44f1b84` | 首绑按存活 session 数；binding 表单一事实源；gateway 终态回调 → Ray fire-and-forget |
| P2 窗口随机首绑 | `87faf46a` | min±first_bind_window(1) 内加权随机 |
| P3 阈值拆分 | `c875d1f7` | sticky_overload_threshold / capacity_reserve_threshold（None 回落） |
| P4 SKEW | `e93e049b` | 池中位数持续偏斜判过载（sessions 绝对差 / running+tokens 比值 AND） |
| P5 near-top | `29ec0085` | top 集放宽 cap×tie_epsilon(0.01)，均匀随机；修掉单赢家崩塌 |
| CLI 暴露 | `3a13c487` | parallel_infer.py 全部新参数 flag |

降级语义：uni-agent 侧无桥接时 active_sessions 退化为累计首绑数（合法首绑信号，失去存活语义）。

## §10 文档同步：git 状态

results/skill/ 下 untracked：README.md、analysis.md、exp1.md、exp2.md、framework.md、04、05
（metric.md、strategy.md 已随 P1/P5 commit 入库）。Phase 5 一并提交。
