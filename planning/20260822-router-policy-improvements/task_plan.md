# Plan: kvcaware 路由策略改进（exp1/exp2 数据驱动，除 prefix-aware 外全部项）

> 起：2026-08-22。依据 `results/skill/exp1.md`、`exp2.md` 的对账结论，把讨论定下的 5 项策略改进（**排除 prefix-aware 首绑**，用户明确不做）整理成可实现方案，并在当前仓库落地。
>
> 代码事实源：`verl/workers/rollout/router/kvcaware/`（当前 host `router-dev` 分支）。分析文档事实源：`results/skill/`。

## 0. 架构原则（最高优先约束，任何 P 不得违背）

> **所有实现决策先过这五条，冲突时以本节为准。**

1. **单一事实源，不建平行账本**。已有的状态表优先复用、派生，而不是为新信号新建一条独立的采集/计数链。实例：session→replica 映射已由 sticky binding 表持有（`request_id == session_id`，gateway session 直接以 session_id 作为 router request_id），`active_sessions` 必须从 binding 生命周期派生（put/替换/失效时 ±1），**不新建** SessionTransport/SessionDecoder 事件链。
2. **进程边界清晰**。状态归属：KVCAwareBalancer 是 Ray actor（router 状态唯一持有者）；GatewayManager 在 driver 进程（session 生命周期唯一持有者）。跨边界只走 Ray remote fire-and-forget（与 `_release_server` 同模式），不做进程内回调穿透，不在 driver 侧镜像 router 状态。
3. **事件链最少化**。能复用既有写路径（sticky put/invalidate）就不加新 Transport/Decoder；新语义挂 `_fire` 钩子（如 `on_session_end`）供后续（P4）按需订阅，默认无人监听、零成本。
4. **迁移方向兼容**。kvcaware 模块后续整体迁往 uni-agent：verl 侧改动收敛在 `kvcaware/` 内自包含（不向 kvcaware 外新增反向依赖）；uni-agent 侧只加**通用**挂点（session 终态回调注册，默认空、不感知 verl），桥接代码放 `framework/entry.py`（该层本就 verl-coupled）。`uni_agent/llm_router/` 是测试中的迁移副本，**本 plan 一律不动**（用户 2026/08/22 明确）。
5. **幂等与降级**。跨进程事件按 at-least-once 设计，接收端幂等（invalidate 对 absent binding no-op）；信号缺失时**降级而非失效**——无 gateway 终态事件时 `active_sessions` 退化为累计绑定数（≈dispatched 语义，仍是合法首绑信号），不为极端路径加补偿复杂度。

**扩展性落点**（后续项按此挂接，不再改骨架）：
- P2（加权随机首绑）：挂 `first_bind_signal` 选择器内部，候选集逻辑不动 store；
- P3（阈值拆分）：只动 `config/strategy.py` + 两处读取点；
- P4（偏斜 overload）：订阅 `on_session_end`/`on_acquire` 钩子 + 读 ACTIVE_SESSIONS，时间窗放 strategy 内；
- P5（近似 top 随机）：只动 `_capacity_token_scores` 候选集判定；
- 新指标：`METRIC_SPECS`/`EMIT_SPECS` 各加一行即可（store/emitter 自动识别）。

## Goal

在纯路由策略层面（不隔离机器、不动 sandbox/timeout 机制）落地以下 5 项改进，使 sticky 初始绑定的落点方差下降、死掉的 overload 纠错通道复活：

1. `active_sessions` 信号 + 首绑 `min(active_sessions)`（P1）
2. 首绑加权随机 / quota（P2）
3. `load_threshold` 双语义拆分：`sticky_overload_threshold` + `capacity_reserve_threshold`（P3）
4. overload 判据从单一 KV usage 扩展为持续偏斜组合信号（P4）
5. 近似 top 随机：capacity tie-break 从严格 `==` 改为相对 gap（P5）

**成功判据**：
- 单元测试覆盖每个新行为（沿用 `tests/workers/rollout/router/` 现有 pytest 风格）；
- exp1 32K 场景的日志回放/仿真中：首绑分布 CV 显著下降（0.4 → <0.2）、27.24 类热点（sustained running 2×median）能触发 rebind；
- 真机重跑（用户执行）：有效 walltime / 有效长尾不劣于 exp1 sticky 基线，timeout 首绑集中度下降。

## Phases

| # | 阶段 | 状态 | 产出 |
|---|---|---|---|
| 0 | 现有代码结构确认 + 信号可得性核对 | ⏳ pending | 确认 acquire/release 回调链能挂 session 生命周期钩子；确认 store 能维护 per-replica active_sessions；列出每项改动的触点文件清单 |
| 1 | P1 `active_sessions` 信号 + 首绑 | ✅ done (verl `0bd3ea8f` + uni-agent `1cbdab0`) | binding 派生：ACTIVE_SESSIONS 由 sticky binding put/替换/invalidate 同步维护；GatewayManager 终态回调（默认空）→ entry.py 桥接 → balancer `on_session_end`（invalidate + _fire）；首绑分支读 min(ACTIVE_SESSIONS)；METRIC_SPECS/EMIT_SPECS 各 +1；单测 |
| 2 | P2 首绑加权随机 / quota | ✅ done (`87faf46a`) | 候选集 = 计数最低 ± 窗口；窗口内均匀或按 (max−count+1) 加权；quota 上限可选；单测 |
| 3 | P3 阈值双语义拆分 | ✅ done (`c875d1f7`) | config 新增 `sticky_overload_threshold` / `capacity_reserve_threshold`，`load_threshold` 保留向后兼容；capacity 分支与 overload 判据分开读各自阈值；单测 + config 兼容测试 |
| 4 | P4 overload 组合信号（持续偏斜） | ✅ done (`e93e049b`) | 新增 `OverloadMode.SKEW`：sustained(active_sessions>median+Δ 或 running>2×median 且 inflight_tokens>2×median，窗口 60s)；仅作用于 sticky shortcut 的 fallback 判定；单测 |
| 5 | P5 近似 top 随机 | ✅ done (`29ec0085`) | capacity 分支 tie-break 从 `remaining == best` 改 `remaining >= best − cap×δ`（δ 默认 0.01，可配）；单测 |
| 6 | 日志回放仿真验证 | ⏳ pending | 用 exp1 32K 日志回放：对比新首绑分布 CV、热点触发率、rebind 次数；离线确认参数（窗口/Δ/δ）不必真机调 |
| 7 | 真机矩阵重跑（用户执行） | ⏳ pending | 32K/64K 至少各一组 sticky+改进版；产出对齐 exp1 口径的有效阶段表，与 exp1 基线对比 |

## 各项方案细节

### P1：`active_sessions` 信号 + 首绑

**问题**：`inflight_count` 是瞬时 LLM 请求数；session 在工具/sandbox 阶段归零，导致"看起来空闲就继续绑"。exp1 首绑分布 10–63（CV 0.4）。

**方案**：
```text
PerReplicaStore / DataStore 新增 active_sessions 计数。
生命周期锚点：
  +1：request_id 首次 acquire（即 sticky binding put 时）
  −1：session 终态。
```

**难点（Phase 0 已解决）**：router 侧只看 LLM request 的 acquire/release，**不知道 session 终态**（AgentFramework 层事件）。

**定案（2026/08/22 二次核对后修订）：binding 派生 + 单个终态通知**，不做 SessionTransport/SessionDecoder 事件链（原方案作废，理由见 D5）：

```text
决定性事实（已核对源码）：
  a. gateway/session/session.py:181  backend.generate(request_id=self.handle.session_id)
     → router request_id 就是 session_id；sticky binding 表天然就是 session→replica 映射。
  b. binding 目前从不失效（invalidate_sticky_binding 无生产调用方）→ 表是只增的；
     finalize/abort 时让它失效即得到"存活绑定"语义。
  c. Balancer 是 Ray actor，GatewayManager 在 driver 进程 → 跨进程只能走 Ray remote。

方案（数据面 + 控制面）：
  +1  / 平移：binding put 路径（on_acquire → StickyUpdate put → DataStore）
       —— 新 replica +1；若 session 已绑在别的 replica（overload fallback 改绑），
       旧 replica −1、新 replica +1（表内替换 ⇒ gauge 天然正确）。
  −1  / 归零：GatewayManager.finalize_session / abort_session
       → 通用回调（driver 进程内，默认空）
       → 桥接（framework/entry.py）fire-and-forget Ray 调 balancer.on_session_end(session_id)
       → balancer：store.invalidate_sticky_binding(session_id)（幂等）+ _fire("on_session_end", ...)
       → ACTIVE_SESSIONS gauge 由 binding 计数派生维护（put/替换/invalidate 内部同步 ±1）。

不变量：active_sessions(replica) ≡ 该 replica 的存活 binding 数（每 replica 每时刻精确）。
降级：无终态事件（非 uni-agent gateway / 回调未注册）时 binding 不失效，ACTIVE_SESSIONS
     退化为累计首绑数 ≈ dispatched 语义 —— 仍是合法首绑信号，只是失去"存活"语义（原则 5）。
```

关键设计点：

1. **单一事实源**（原则 1）：session→replica 映射只在 binding 表里存在一份；`ACTIVE_SESSIONS` 是它的 per-replica 计数投 影，put/替换/invalidate 三个写路径同步维护，无第二账本。
2. **+1 锚在 binding put**（保留 D4）：create 早于首个 LLM request，锚在 put 规避空窗；rebind（overload fallback 后新 winner 覆盖旧 binding）由"表内替换"语义自动 −旧+新，无需特殊处理。
3. **uni-agent 侧只加通用挂点**（原则 4）：`GatewayManager` 三个生命周期方法加同步回调注册点（默认空、异常吞掉、不感知 verl）；`framework/entry.py` 的 rollout adapter 在构建 gateway_manager 后注册桥接回调 → `balancer_handle.on_session_end.remote(session_id)`。abort 幂等 ⇒ 重复通知 invalidate 为 no-op（原则 5）。
4. **首绑用法**：`least-inflight` 与 `capacity-token-aware`（cold-start 分支）的首绑选择读 `min(ACTIVE_SESSIONS)`，非首绑仍走原逻辑；`ACTIVE_SESSIONS` 进 `METRIC_SPECS`/`EMIT_SPECS`（B-class gauge，on_acquire/release/终态三类写点发出）。
5. **LRU 逐出兜底**：binding 表 10000 上限，512 session 远不及；逐出时该 binding 的 ACTIVE_SESSIONS 同步 −1（计数不泄漏）。

原"SessionTransport/SessionDecoder 事件链"与"超时近似 fallback"两案均作废（D5）。

### P2：首绑加权随机 / quota

**问题**：agentic 下计数信号是小的整数，并列是常态；确定性 tie-break 使落点取决于遍历顺序。

**方案**：
```text
候选集 = { r : active_sessions(r) <= min + window }，window 默认 1。
选择：候选集内均匀随机；或 weight = (min+window+1−count) 加权抽样。
quota（可选，默认关）：单台首绑上限 = ceil(活跃 session 总数 / R) + slack。
```
实现在首绑选择处，不动 route() 主流程。random 已 import（routing.py）。

### P3：阈值双语义拆分

**问题**：`load_threshold` 同时控制 sticky overload 判据（`kv_perc > lt`）和 capacity eligibility（`avail >= cap×(1−lt)`）。exp1 的 lt=0.9 使前者永不触发（KV 峰值 0.22）而后者语义又不可独立调。

**方案**：
```yaml
strategies:
  - name: kvcaware
    load_threshold: 0.9            # 保留：仅作 capacity_reserve 的缺省来源（向后兼容）
    sticky_overload_threshold: 0.3 # 新增：解除 sticky 的门槛
    capacity_reserve_threshold: null  # 新增：null → 回落 load_threshold
```
代码：`is_overloaded()` 读 `sticky_overload_threshold`；`_capacity_token_scores()` 的 `thresh` 读 `capacity_reserve_threshold`（缺省回落）。config 校验两条 in (0,1)。

### P4：overload 组合信号（持续偏斜）

**问题**：27.24 形态是"KV 不满（0.2）但 running 12 vs 集群 2–4"，单一 KV 判据永远抓不到。

**方案**：新增 `OverloadMode.SKEW`：
```text
sticky_overloaded(r) if 持续 >= skew_window(默认 60s):
    active_sessions(r) > median(active_sessions) + skew_delta(默认 2)
  OR ( running(r) > skew_factor×median(running)
       AND inflight_tokens(r) > skew_factor×median(inflight_tokens) )   # skew_factor 默认 2.0
```
- 只作用于 `_sticky_shortcut` 的 fallback 判定，不参与正常 capacity 排序；
- 需要在 store 或 strategy 内维护各 replica 指标的时间窗（简化实现：collector 已 ~150ms poll，保留最近 N 个样本或用一阶 EMA 判"持续"）；
- median 跨当前 pool 所有副本计算（相对阈值，context 无关）。

### P5：近似 top 随机

**问题**：commit `3cfd111f` 的随机用严格 `==`；exp2 实测 40,292 次路由仅 75 次触发（全在冷启动 15s metrics 全零期），运行期仍是确定性 argmax。

**方案**：capacity 分支的 top 集合判定改为：
```python
eps = cap * tie_epsilon          # tie_epsilon 默认 0.01
top_idx = [i for i in order
           if rows[i]["remaining"] >= best_remaining - eps
           and rows[i]["avail"] >= thresh]
```
`route()` 的 random.choice 骨架保持不变（已存在），只改候选集宽度。无 eligible 全过载时的 fallback 分支同样处理。

## Decision log

- **2026/08/22 — D0 排除 prefix-aware**：用户明确本 plan 不含 prefix-aware 首绑（同 sample 8 session 前缀聚集），其余 5 项全做。
- **2026/08/22 — D1 依赖顺序**：P3 先于 P4（拆阈值后 SKEW 模式才有干净的 threshold 可读）；P1 先于 P2（加权随机需要 active_sessions 作权重基础，P2 也可先用 inflight_count 落地后切换）；P5 独立，随时可做。
- **2026/08/22 — D2 session 终态信号定案：gateway 回调桥接**（用户提出，更新于当日）：照搬 inflight 的 CallbackTransport 模式——`../uni-agent` GatewayManager 三个生命周期方法（create/finalize/abort）注册回调，verl 侧新 SessionTransport/SessionDecoder 写入 store。依据：Manager 层已有精确的 `active_sessions_per_gateway` 维护逻辑（含失败回滚 + abort 幂等），只是粒度在 gateway actor 而非 LLM replica；桥接只差 session→replica 映射，而 sticky binding 已持有该映射。比超时近似精确、比 AgentFramework 大改侵入小。**推翻**先前"先做超时近似"的倾向，超时近似降级为 fallback。
- **2026/08/22 — D3 默认参数策略**：窗口/Δ/factor/ε 全部做成 config 项并给保守默认（60s / 2 / 2.0 / 0.01），Phase 6 日志回放校准，不真机盲调。
- **2026/08/22 — D4 +1 锚在 binding put 而非 create_session**：create_session 早于首个 LLM request（binding 尚不存在），若 +1 锚在 create 则时序上有窗口期计数为空；锚在 binding put 使 active_sessions 语义严格等于"绑定在该 replica 的 session 数"，且复用 on_acquire 已有链路。
- **2026/08/22 — D5 P1 改为"binding 派生 + 单个终态通知"，作废 SessionTransport/SessionDecoder 事件链**：二次核对发现 `request_id == session_id`（gateway session 以 session_id 作 backend.generate 的 request_id），sticky binding 表已是 session→replica 单一事实源；且 binding 当前从不失效，finalize/abort 让它失效即得"存活绑定"。派生方案下 rebind/重复 abort/bulk invalidate 全部天然幂等（表内替换/no-op），无 ±1 事件漂移风险；跨进程只剩一个 fire-and-forget `on_session_end`（Ray remote，与 `_release_server` 同模式）。原方案（新 Transport/Decoder + decoder 暂存快照 + 防负护栏）在漂移面和代码量上全面劣于派生方案，作废。
- **2026/08/22 — D6 `uni_agent/llm_router/` 不动**：该目录是 kvcaware 迁移到 uni-agent 的副本，用户明确"还在测试中"。P1 的 uni-agent 侧改动仅限 `gateway/manager.py`（通用回调挂点）与 `framework/entry.py`（桥接注册），不触碰 llm_router/。
