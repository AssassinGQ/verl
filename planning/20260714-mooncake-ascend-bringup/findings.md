# Findings: mooncake 在 Ascend 跑通

> 本 plan 的研究发现。继承主线 plan §8/§10/§12/§15 已坐实结论，本文件只记**本 plan 新增**发现。

## §1 继承自主线 plan 的已坐实结论（不重复，只索引）

- **§8**（07/07）：vllm-ascend mooncake 差异——connector 名 `MooncakeConnectorStoreV1`（非 GPU 的 `MooncakeStoreConnector`），P2PHANDSHAKE 无需 daemon，config 补 `local_buffer_size`。
- **§10**（07/09）：mooncake pip 0.3.9 aarch64 可用（推翻 source-build 假设）；GPU-v3 topology 让 master 起 + 8 client 注册，但 register_buffer 仍挂（transfer engine 握手读垃圾）。
- **§12**（07/11）：P2PHANDSHAKE config 根因确认（权威源背书）——`metadata_server` 两合法值：`P2PHANDSHAKE`（单机）/`http://host:port`（多节点）。🔬 910B3 验证 `store.setup ret=0` 无 garbage，但暴露更深 RDMA 根因（见下）。
- **§15**（07/11 夜 3）：910C 源码 build `-DUSE_ASCEND_DIRECT=ON` 装通但 `import mooncake.store` 堆损坏；隔离实验 `import torch_npu` 干净 / `import mooncake.store` 崩 → mooncake 源码 build 自身写坏堆，torch_npu 无关。C/D 双重受阻，搁置。

## §2 两条路径的本质矛盾（v0.3.9 当下）

| 路径 | ascend-direct/fabric mem | 堆损坏 | C/D 能跑？ |
|---|---|---|---|
| 910C 源码 build `-DUSE_ASCEND_DIRECT=ON` | ✅ 有 | ✅ 崩 | ❌ |
| 910C pip wheel 0.3.9 | ❌ 无 | ❌ 干净 | ❌（无 fabric mem → register_buffer 过不了）|
| 910B3 非 fabric → RDMA | — | — | ❌（RoCE 注册不了 NPU 内存，EINVAL，fabric-only）|

**矛盾点**：要 ascend-direct（910C fabric mem 必需）必须源码 build，但源码 build 堆损坏；要干净必须 pip，但 pip 无 ascend-direct。

## §3 堆损坏的三个候选根因（待 Phase 1/2 验证）

systematic-debugging 定位到"mooncake 源码 build 自身加载就写坏堆"（§15），但未定位到具体 .cpp。候选：

1. **v0.3.9 特定 commit 的回归**（最可能）：静态初始化越界。证据：pip wheel 同源不崩，源码 build 崩 → 差异在 build 配置/链接，非架构死结。**Phase 1 换版本试，成本最低。**
2. **链接的 CANN 版本与容器 cann-9.0.0 错位**：源码 build `-DUSE_ASCEND_DIRECT=ON` 链接某 CANN lib，容器是 cann-9.0.0，ABI 不匹配 → 静态初始化写坏。需 Phase 2 ASan 定位 + 核对 CANN 版本。
3. **glibc ABI 不匹配**：mooncake build 用的 glibc 与容器 glibc 版本差异。概率低（pip wheel 同 glibc 不崩），但 ASan 能排除。

## §4 关键事实：检测点 ≠ 损坏点（用户纠正，重要）

gdb 显示崩在 **exit 阶段**（`__exit(0)` → `dl_fini` → `libunified_dlog.so` 析构 "Dlog finalize" → `vsyslog→open_memstream→calloc` → `malloc_consolidate` 检测 `corrupted size vs. prev_size`）。import 本身已成功（到了 exit 才崩）。

**但 consolidate 检测点 ≠ 损坏点**：consolidate 扫描发现某 chunk 的 size/prev_size 对不上，说明堆在**运行期就被写坏了**，exit 只是踩到。**不能因为崩在 exit 就以为 exit 才是问题**——静态初始化（import 时）就越界写了。这一认知决定 Phase 2 必须用 ASan（halt_on_error=0 让它报首个越界写），而非只看 exit backtrace。

## §5 官方指南背书（源码 build 是 Ascend 唯一正路）

[vLLM-Ascend KV Pool 官方指南](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/kv_pool.html) 明确：**Ascend 必须源码 build `-DUSE_ASCEND_DIRECT=ON`**，非 pip。`dependencies.sh` 自动装 rdma-core/libibverbs，`make install` 把 libasio.so + ascend-direct .so + master 一次性装到位。

→ 含义：pip wheel 在 Ascend 上残缺（缺 libibverbs/libasio + 无 ascend-direct）**不是 bug 是设计**，官方就没打算让 pip wheel 在 Ascend 用。故 Phase 1（换版本源码 build）/ Phase 2（ASan 修源码 build）是正攻方向，pip wheel 路是死路。

## §6 910B3 RDMA 路是死结（架构限制，非配置可修）

§12 🔬 隔离实验：`get_transfer_engine(device_name=None)`（非 fabric）→ RDMA transport（`transfer_engine_impl.cpp:336 installTransport, type=rdma`）→ `rdma_context.cpp:334 Failed to register memory 0x...: Invalid argument [22]`。RoCE NIC（rocep189s0f0）**注册不了 NPU 设备内存**（无 P2P/IOMMU peer access）。fabric 路代码注释明写 **A3-only**。

→ 含义：910B3 跑 C/D 是死路（非 fabric 必走 RDMA，RDMA 注册不了 NPU 内存），Phase 4 只是"确认死结 + 留证据"，大概率放弃 910B3，只攻 910C fabric 路（Phase 3）。

## §7 mooncake_config.json 已对齐官方（commit b7f97e4 前 b7f0168）

config 层已修好（不是卡点）：
- `preferred_segment:false` + `prefer_alloc_in_same_node:true`，去掉 `local_buffer_size`/`local_hostname`（fabric 路用 0/自动）
- master：`--rpc_port` → `--port`（源码 build 二进制）+ `--eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1` + lease TTL 11000（>ASCEND_TRANSFER_TIMEOUT）
- C/D env：`ACL_OP_INIT_MODE=1` + `HCCL_RDMA_TIMEOUT=17` + `ASCEND_CONNECT_TIMEOUT=10000` + `ASCEND_TRANSFER_TIMEOUT=10000` + `PYTHONHASHSEED=0`

→ 含义：config 不再是变量，卡点纯在 mooncake 二进制（堆损坏 / 无 ascend-direct）。
