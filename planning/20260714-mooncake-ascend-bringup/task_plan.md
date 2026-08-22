# Plan: mooncake 在 Ascend 跑通（C/D 解锁技术债）

> 起：2026-07-14。从 `20260707-kvcare-router-ascend910b3` 拆出来——主线 A/B（no-mc）已跑通并出结论，**C/D（mooncake）是唯一卡结论成立的技术债**：两台机都跑不通，搁置中。本 plan 专门收 mooncake bring-up 的根因定位 + 修复尝试，让 C/D 能跑，补全 "C−A = mc 纯贡献 / D−A = 总收益" 两格结论。
>
> **排除项**（用户明确不做）：不换 KV transfer 方案、不换 connector、不用 mock 验 mc 净负向——**只攻关 mooncake 本身在 Ascend 跑通**。

## Goal

让 mooncake KV-transfer 在 Ascend（910C fabric 路优先 / 910B3 RDMA 路兜底）跑通 `register_memory(NPU_ptr)`，使 vLLM-Ascend `MooncakeConnectorStoreV1` 能跨 replica 传 KV，C/D 矩阵组可跑。

**成功判据**：C 组（sticky + mc）单副本 SMOKE，vLLM 启动日志 `Mooncake memory registered` 成功（非 `Mooncake memory registration failed.`），无 `corrupted size vs. prev_size` / 无 `Invalid argument [22]`。

## 故障模式全梳理（两台机两条路，4 个卡点）

| 机器 | 路径 | 卡点 | 根因 | 已确认？ |
|---|---|---|---|---|
| 910C | 源码 build `-DUSE_ASCEND_DIRECT=ON`（有 ascend-direct/fabric mem）| **import mooncake.store 堆损坏** | `corrupted size vs. prev_size`（glibc malloc），exit 阶段检测但运行期已写坏；隔离实验 `import torch_npu` 干净 / `import mooncake.store` 崩 → mooncake 自身写坏堆（§15） | ✅ 两次用户纠正归因后确认 |
| 910C | pip wheel 0.3.9（干净，无堆损坏）| 无 ascend-direct → 910C 没 fabric mem → register_buffer 过不了 | pip wheel 不带 `-DUSE_ASCEND_DIRECT`，store.so/nvlink_allocator.so 是 GPU 版 | ✅ §15 |
| 910B3 | 非 fabric（默认）→ RDMA transport | `register_memory(NPU_ptr)` → `Invalid argument [22]` | RoCE NIC 注册不了 NPU 设备内存（无 P2P/IOMMU peer access）；fabric 路代码注释明写 A3-only | ✅ §12 🔬 |
| 910B3 | config 层（已在 §12 修好，非真卡点）| P2PHANDSHAKE config → `store.setup ret=0` 无 garbage | 之前误用 `metadata_server=host:port`（多节点格式），单机应 P2PHANDSHAKE | ✅ 已修 |

**本质矛盾**：源码 build 有 ascend-direct（910C 要的 fabric mem）但堆损坏；pip 干净但无 ascend-direct。**要么修好源码 build 的堆损坏（首选），要么让 pip wheel 带 ascend-direct（次要）。**

## Phases

| # | 阶段 | 状态 | 产出 |
|---|---|---|---|
| 0 | 证据固化：ldd + gdb + 环境指纹 | ⏳ pending | `ldd libmooncake_store.so`、`gdb -batch -ex bt` 全栈、`python -c "import mooncake.store"` 隔离复现脚本、容器 CANN/glibc/gcc 版本 → 提 issue 的最小可复现包 |
| 1 | 换 mooncake 版本试源码 build | ⏳ pending | 试 `v0.3.8` / `main` HEAD / `v0.4.x`（若有）源码 build `-DUSE_ASCEND_DIRECT=ON`，看堆损坏是否消失（可能是 v0.3.9 特定 commit 的回归） |
| 2 | ASan 定位堆损坏点 | ⏳ pending | mooncake 源码 build 加 `-fsanitize=address`，`ASAN_OPTIONS=halt_on_error=0`，跑 `import mooncake.store` → ASan 报首个越界写位置 → 定位到具体 .cpp（静态初始化越界 / CANN 版本错位 / glibc ABI 之一） |
| 3 | 910C fabric 路端到端 | ⏳ pending | 堆损坏解决后，`ASCEND_ENABLE_USE_FABRIC_MEM=1` + MC_FABRIC_MEM=1，C 组单副本 SMOKE → `Mooncake memory registered` 成功 |
| 4 | 910B3 RDMA 路评估（兜底，可能放弃）| ⏳ pending | 评估 RoCE 注册 NPU 内存是否真无解（查 910B3 是否有 P2P/IOMMU 路径）；若 fabric-only 则 910B3 放弃 C/D，只 910C 跑 |
| 5 | C/D 矩阵跑通 + 对比 | ⏳ pending | C/D SMOKE → 真跑 → C−A mc 纯贡献 / D−A 总收益，补全 ABCD 结论表（主线 plan §9） |

## Decision log

- **2026/07/14 — D0 拆 plan + 约束**：mooncake bring-up 从主线拆出独立 plan（用户提议）。**硬约束：不换 KV transfer 方案/connector/mock**（用户明确不做"改用其它方案验证 mc 净负向"），只攻关 mooncake 本身。即本 plan 成败 = mooncake 是否在 Ascend 跑通，不绕道。
- **2026/07/14 — D1 首攻源码 build 堆损坏（v0.3.9 回归假设）**：堆损坏最可能是 v0.3.9 某个 commit 的回归（静态初始化越界），而非架构性死结——因为 pip wheel 同源不崩，源码 build 崩，差异在 build 配置/链接的 CANN 版本。优先试换版本（Phase 1，成本最低），再 ASan（Phase 2，定位最准但 build 成本高）。
- **2026/07/14 — D2 910C fabric 路为主攻方向**：910B3 RDMA 注册 NPU 内存是架构限制（RoCE 无 P2P/IOMMU peer access，§12 fabric 路代码注释明写 A3-only），修不了；910C fabric 路（`ASCEND_ENABLE_USE_FABRIC_MEM`）才是正路。故 910B3 作兜底评估（Phase 4，大概率放弃），主攻 910C（Phase 3）。
- **2026/07/14 — D3 config 已修好，不再纠缠**：§12 🔬 已确认 `metadata_server=P2PHANDSHAKE`（单机）/`http://host:port`（多节点）是**不同字段**（metadata_server=transfer-engine 层 vs master_server_address=store 层），合并=P2PHANDSHAKE+9422+起 master，`store.setup ret=0` 无 garbage。config 层不再重复查。
- **2026/07/14 — D4 不阻塞主线**：A/B（no-mc）结论已成立（§9，低压无收益 / 910C 16-replica 待对比），C/D 是补全 ABCD 表的两格，mooncake 搁置不影响 A/B 结论交付。本 plan 是独立技术债攻关，按有空才推进。
- **2026/07/14 — D5 提 issue 留证据**：Phase 0 产出最小可复现包（ldd/gdb/隔离脚本/环境指纹），无论能否自修，都提 issue 给 kvcache-ai/Mooncake——堆损坏是 mooncake 源码 build 在 Ascend 的真 bug，非用户配置问题。
