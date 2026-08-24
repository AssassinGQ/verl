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

### 2026/08/22 — P1 落地完成（两个 commit）

**verl4 `0bd3ea8f`**：store（binding 写路径维护 ACTIVE_SESSIONS、delete_where 返回删除数、LRU on_row_evicted 钩子）、balancer（`on_session_end` Ray 方法 + `_fire` 钩子）、strategy（least-inflight 与 capacity 冷启动改 `min(active_sessions)`）、llm_server（`router_handle` 属性）、insight（METRIC/EMIT_SPECS + ACQUIRE/RELEASE/SESSION_END 三写点发射、router-dispatch 行）、测试（store 生命周期 9 例 + balancer 5 例全绿）。
**uni-agent `1cbdab0`**：GatewayManager 终态回调（默认空、异常吞掉）+ `bridge_session_end_to_router`（entry.py adapter 与 llm_router example 两处接线）+ gateway 回调测试 4 例。

测试结果：router 全套 326 passed / 7 skipped；e2e 2 例失败为**既有环境问题**（venv 缺 `uni_agent` 模块、缺 `mooncake_http_metadata_server` 二进制，stash 基线同样失败）；strategy 4 例既有失败（测试未传 request_id 被 cold-start 分支截胡）按其本意改用 `sticky={"r1": "rep_gone"}` 钉住 eligible 排序路径后全绿。metric.md/strategy.md 已同步新语义。

下一步：P2（首绑加权随机/quota）。

### 2026/08/23 — P2–P5 落地完成

- **P2 `87faf46a`**：`_first_bind_top`——候选 = min±`first_bind_window`(默认1)，窗内 `random.choices` 按 `window+1−count` 加权（`first_bind_weighted=false` 则均匀）；接入 least-inflight 与 capacity 冷启动两个首绑点。window=0 退化为严格 min。
- **P3 `c875d1f7`**：`sticky_overload_threshold` / `capacity_reserve_threshold`（默认 None → 回落 load_threshold，行为不变）；`is_overloaded` 与 capacity 门槛各读各的；balancer summary 与 yaml 示例同步。
- **P4 `e93e049b`**：`OverloadMode.SKEW`——`active_sessions > median+skew_delta`（绝对差）或 `running>且 inflight_tokens> skew_factor×median`（比值），连续 `skew_window`(默认60，sticky-shortcut 节拍采样) 才判 overload；streak 状态在 strategy 内（router actor 单线程免锁）。`is_overloaded` 增可选 `replicas` 参数。
- **P5 `29ec0085`**：`_near_top_pick`——top 集 = `{i∈pool : remaining[i] >= best − cap×tie_epsilon(默认0.01)}`，均匀抽取；eligible 与 no-eligible 两分支都走；reserve 门槛保持硬过滤（ε 只放宽并列，不救被门槛排除者）。ε=0 退化严格 argmax。

测试：strategy 119 + balancer 41 + ray-integration 5 + config/router 108 全绿；quota（P2 可选项）未实现——窗口+加权已覆盖其目标，等 Phase 6 回放显示需要再加。既有 flaky：balancer 目录混跑时 ray-integration 5 例失败系 conftest session 级 monkeypatch 的条件判断粒度问题（P1 前即存在，单跑/文件级跑均过），未修。

### 2026/08/23 — 全部提交推送

- verl4 `router-dev` → `origin/router-dev`：`0bd3ea8f`(P1) / `87faf46a`(P2) / `c875d1f7`(P3) / `e93e049b`(P4) / `29ec0085`(P5) / `d4147d54`(docs) 全部推送。
- uni-agent `router` → `origin/router`：P1 桥接 commit `1cbdab0` 已在远端（用户后续提交 `0618505`/`f54f14e` 叠于其上）。

下一步：Phase 6 日志回放（exp1 32K 日志 + `/tmp/reconcile.py` 思路正式化），校准 window/Δ/factor/ε。

## 2026/08/24 — Phase 8：skill 文档同步

审计 results/skill/ 八个文档与 P1-P5 代码的差异（完整审计见 findings §8）：README 三处悬空引用 +
TL;DR 两处过时（单赢家 bug 已被 P5 修复、首绑语义已变）+ 无 P1-P5 信息；04/05 遗留待合并；
framework.md 缺 P1 桥接；exp1.md 缺演进指针；五个文档 untracked。README 头部/索引/当前代码状态节已改完，
剩 TL;DR 与尾部版本说明。（注：曾误开 planning/20260824-skill-docs-sync/，已并入本文件组并删除。）

### 2026/08/24 — Phase 8 完成（`0c907471`，已推送）

- README：悬空引用/重复索引行清除；新增"当前代码状态"（P1-P5 commit + 参数 + 降级语义）；TL;DR 标注两处已修（P5 单赢家、P1 首绑语义）。
- 05 并入 exp2 §4.4（数字取证链 + avail 相当的澄清 + 修复状态注记）后删除；04 并入 analysis §7"判读速查"（去实验特定化）后删除。
- framework 新增 §2.4（P1 桥接链 + 幂等 + 降级 + 409 时序安全）；exp1 新增 §5.5 演进指针（启发 → P 项映射，基线表留作 Phase 7 对照）。
- 五个 untracked 文档全部入库。
