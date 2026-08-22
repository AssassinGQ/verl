# Progress: kvcaware 路由策略改进

## 2026/08/22 — plan 创建

### 背景
exp1（0820，8 实验）与 exp2（0821，2 实验 + commit `3cfd111f`）的分析与 session→replica 对账已完成（`results/skill/exp1.md`、`exp2.md`）。用户要求把讨论定下的策略改进整理成实现方案写入 planning；**prefix-aware 首绑明确排除**。

### 本 session 已完成
- 确认 exp1 真实语义：两组均 `do_shortcut=true`，对比 `min(inflight_count)` vs `min(inflight_tokens)` 首绑，后续全 sticky；常规 capacity 分支与 overload fallback（lt=0.9）均未触发。
- 确认 exp2 真实语义：kvcaware `do_shortcut=false`，每轮无 sticky capacity routing；prefix-hit 90%→38%，有效非长尾 124→212 min。
- 完成 session→request/replica 对账（`request=session-<s>-<se>-<uuid>` 直接 join runner 日志），补齐 exp1 八组 + exp2 两组的有效阶段表。
- 有效口径结论：kvcaware 无净收益（合计慢 30.2 min）；原始 walltime 的优势全由 idle 段差异贡献。
- 定位 27.24 引擎级慢（prefill 4–8×、TPOT 10–25×、queue≈0）；timeout 首绑高度集中（exp1 sticky 24/37、kvcaware 69/90 在 27.24）。

### 本 plan 范围（5 项，排除 prefix-aware）
P1 active_sessions 首绑 / P2 加权随机或 quota / P3 阈值双语义拆分 / P4 偏斜组合 overload / P5 近似 top 随机。环境类（慢机隔离、sandbox OOM、挂死提前终止）不在本 plan。

### 状态
- Phase 0（代码触点核对）部分完成：P1 的 session 终态信号已定案（gateway 回调桥接，见下）；其余触点清单已入 findings §3。
- 下一步：Phase 0 收尾（核对其余项触点）→ Phase 1 落地。

### 2026/08/22 — P1 方案更新：session 终态信号定案为 gateway 回调桥接

用户提出：类似 inflight 的 CallbackTransport，在 gateway 定义回调点，kvcaware 的 callback 注册进去，实现 active_sessions 精确获取。

核对 `../uni-agent` 源码后确认可行且优于原"超时近似"倾向：
- `GatewayManager`（manager.py:75-108）已维护 `active_sessions_per_gateway`，create（同步预约+失败回滚）/finalize/abort（幂等）三出口全覆盖；
- 插桩点定在 Manager 层（driver 进程），不进 Ray actor；
- +1 锚在 binding put（规避 create 早于首绑请求的时序窗）、−1 锚在 finalize/abort；session→replica 映射复用 sticky binding；
- 遗留时序细节：−1 时 binding 可能已被清理 → decoder 侧暂存 session→replica 快照或 binding invalidate 时补发终态事件（Phase 1 实现时定夺）。

plan 已更新：task_plan P1 方案 + Phase 1 产出描述 + D2 改写（原"优先近似"作废）+ 新增 D4（+1 锚点）；findings §4 重写 + §3 触点表补 `../uni-agent` 侧与新文件。

### 2026/08/22 — P1 方案二次修订：binding 派生 + 单个终态通知（作废事件链方案）

实现前逐文件核对两侧源码，三个决定性发现：

1. `gateway/session/session.py:181`：`backend.generate(request_id=self.handle.session_id)` → **request_id 就是 session_id**，sticky binding 表天然是 session→replica 映射；
2. verl 侧 `invalidate_sticky_binding` 无生产调用方 → binding 表只增；finalize/abort 让它失效即得"存活绑定"语义；
3. KVCAwareBalancer 是 Ray actor、GatewayManager 在 driver 进程 → 原方案的"进程内回调 → CallbackTransport"链在物理上不成立，必须走 Ray remote。

据此 P1 改为（D5）：ACTIVE_SESSIONS 由 binding put/替换/invalidate 同步 ±1（单一事实源派生，无第二账本）；uni-agent 侧 GatewayManager 加通用终态回调（默认空），`framework/entry.py` 桥接 fire-and-forget `balancer.on_session_end.remote(session_id)`。原 SessionTransport/SessionDecoder 链与超时近似 fallback 均作废。

另发现 `../uni-agent/uni_agent/llm_router/` 为 kvcaware 迁移副本（测试中，用户明确不动，D6）；P1 的 uni-agent 侧改动仅 `gateway/manager.py` + `framework/entry.py`。应用户要求，task_plan 开头新增"§0 架构原则"五条（单一事实源 / 进程边界 / 事件链最少化 / 迁移兼容 / 幂等降级）+ 各 P 扩展性落点。
