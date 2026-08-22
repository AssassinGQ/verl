# Progress: mooncake 在 Ascend 跑通

## 2026/07/14 — plan 创建（从主线拆分）

### 背景
主线 `20260707-kvcare-router-ascend910b3` A/B（no-mc）已跑通，C/D（mooncake）是唯一卡结论成立的技术债：两台机都跑不通，搁置中（§15）。用户提议把 mooncake 攻关单独开 plan，聚焦不阻塞主线。

### 拆分约束（用户明确）
- mooncake bring-up 独立 plan（本文件）
- plot_eviction 两项优化独立 plan（`20260714-plot-eviction-bandwidth-html`）
- **排除**："C/D 改用其它 KV transfer 方案验证 mc 净负向"——**不做**，不进任何 plan。本 plan 硬约束：只攻关 mooncake 本身在 Ascend 跑通，不换方案/connector/mock。

### 继承的已坐实结论（不重复，索引到主线 findings §8/§10/§12/§15）
- §8：connector 名 `MooncakeConnectorStoreV1`，P2PHANDSHAKE 无需 daemon
- §10：pip 0.3.9 aarch64 可用，但 register_buffer 挂
- §12 🔬：P2PHANDSHAKE config 修好（`store.setup ret=0` 无 garbage），暴露 RDMA 根因
- §15：910C 源码 build 装通但 `import mooncake.store` 堆损坏；torch_npu 无关；C/D 双重受阻搁置

### 起点状态
- mooncake_config.json 已对齐官方（commit b7f97e4 前 b7f0168），config 层不再是卡点。
- 主攻方向：910C 源码 build 的堆损坏（首攻换版本，次攻 ASan）。
- 910B3 RDMA 路是死结（架构限制，Phase 4 兜底评估大概率放弃）。

### 待办
- [ ] Phase 0：证据固化（ldd/gdb/隔离脚本/环境指纹 → 提 issue 最小复现包）
- [ ] Phase 1：换版本源码 build（v0.3.8 / main / v0.4.x 试，看堆损坏是否消失）
- [ ] Phase 2：ASan 定位堆损坏点（`-fsanitize=address` + `halt_on_error=0`，报首个越界写）
- [ ] Phase 3：910C fabric 路端到端（堆损坏解决后，C 组单副本 SMOKE → `Mooncake memory registered`）
- [ ] Phase 4：910B3 RDMA 路评估（兜底，大概率放弃）
- [ ] Phase 5：C/D 矩阵跑通 + 对比（补全 ABCD 结论表 §9）

### 当前不动，按有空推进
主线 A/B 结论分析（A/B 主线结果对比 3090 walltime 收益）是当前主线下一步，mooncake 攻关是独立技术债，不抢主线资源。
