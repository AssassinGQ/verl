# task_plan — 复现官方 vLLM×Mooncake 博客(代码版本对齐,144 TCP 验证机制)

## 目标
官方博客 https://vllm.ai/blog/2026-05-06-mooncake-store 报 2x+ 吞吐(3.8×/46×TTFT↓)。
**我方 exp3/exp4 在 3090+TCP loopback 下 mc 净负收益**(C>A +7%、D>B +20-30%、External hit=0)。
本计划目标:**代码和版本跟官方博客对齐**(ivanium/vllm:feat/mooncake-store-int @ 72d90b9),在 144(8×3090+TCP,无 RDMA)上验证"为什么官方 2x、我方负收益"的机制差异——**不追官方数字**(硬件差一代,数字不可比;博客 2x 前提 RDMA+PD分离+Kimi 144 物理上没有)。

## ⭐ 已验证的可行性(2026-07-10)
- ✅ **代码已就位**:`/data1/hgq/vllm-blog`(浅克隆 ivanium/vllm feat/mooncake-store-int @ 72d90b9,121M)
- ✅ **博客架构确认**(方向纠正):博客**不走 verl**,用 `vllm serve` + Rust `vllm-router` + python benchmark 三件套
- ✅ **vllm serve 在 blog vllm 下可用**:`vllm serve --help` OK,`MooncakeStoreConnector` 在 factory 注册(名解析 OK)
- ✅ **mooncake runtime 可用**:容器内 mooncake 0.3.11,`MooncakeDistributedStore.setup()` ret=0 TCP 通,daemon up(metadata@9527 + master@9422)
- ✅ **torch 匹配**:blog 要求 torch==2.11.0,容器内 2.11.0+cu128
- ✅ **Rust toolchain 装好**(host):cargo 1.97 + rustc 1.97(经 TUNA 镜像装,绕过 static.rust-lang.org 不通)
- ✅ **vllm-project/router 已 clone** `/data1/hgq/vllm-router-src`,cargo config 配 TUNA 镜像
- ✅ **cargo build 完成**(容器内):host py3.6<pyo3 → 改容器内 build(py3.12);openssl-sys 缺 pkg-config → `apt install pkg-config libssl-dev` 后重跑过。**二进制就位** `/data1/hgq/vllm-router-src/target/release/vllm-router`(24.5MB),`--help` OK:支持 colocated `--worker-urls w1 w2` + `--policy {random,round_robin,cache_aware,power_of_two,consistent_hash,rendezvous_hash}`(默认 cache_aware)→ **Phase 2 两对照策略(round_robin vs cache_aware)都支持**
- ✅ **Phase 3 源码级差异完成**(findings §8):0.21 CPU staging(2× cudaMemcpy)vs blog GPU 直读是根因 #2 源码实锤;lookup 两边同逻辑(根因 #1 是路由非代码)
- ⚠️ 144 uni-agent 是旧 checkout(非 git),已 sync 本地 llm-router 分支的 llm_router 包 + examples(虽然博客不用 verl,但备着)
- ⚠️ **NixlConnector 跑不了**(flash_attn .so ABI)——但 colocated 模式不用 Nixl,不影响 store 验证
- ⚠️ **网络坑**:容器 unreachable proxy(`HTTP_PROXY=8.92.10.60:7890`)+ `static.rust-lang.org`/`index.crates.io` 慢→全用 TUNA 镜像 + unset proxy 绕过

## 博客真实配置(查证,作为对齐基准)
- **代码**:ivanium/vllm `feat/mooncake-store-int` @ `72d90b9a627a`(2026-05-11)= vllm main 前身(0.21+358)
- **架构三件套**(全程不走 verl):
  1. `vllm serve Kimi-K2.5 -tp 4 --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector",...},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"enable_cross_layers_blocks":true}}]}}'`
  2. `vllm-router`(Rust,`vllm-project/router` 仓库,`cargo build --release`)`--policy round_robin --vllm-pd-disaggregation --prefill URL --decode URL`(也支持 colocated `--worker-urls w1 w2 --policy round_robin`)
  3. python benchmark:`benchmarks/multi_turn/{gen_random_multi_turn,benchmark_serving_multi_turn}.py`
- **硬件**:Kimi-K2.5-NVFP4,CUDA 13.1,3 节点 Slurm,1P(1node×4GPU TP4)+1D(2node×4GPU DEP8),UCX_TLS=cuda_ipc,cuda_copy,tcp,NCCL_IB_HCA=mlx5_0/1/3/4(真 RDMA)
- **博客 2x 收益**:cache hit 1.7%→92.2%(跨实例 prefix 不重算)+ RDMA 高带宽(传输<重算)

## 我方负收益的根因(exp4 §5,需对照验证)
1. **kvcare prefix 聚合→同 prompt 全在同 replica→External hit 恒=0**(mc 无 L2 命中可省)→ memory `mooncake-external-hit-no-cross-replica`
2. **TCP+CPU staging 传输开销 > prefill 节省**(无 RDMA,传输慢)
3. **mc L2 staging 碎块写入破坏 prefix 批量化**(blocks/event B 32.2→D 1.80,+656 万额外 GPU 分配)

## Phase 0:环境就绪(在 hgq-swe 容器内,不改镜像) ✅ 完成(2026-07-10)
- [x] 备份 0.21 editable 指针:`__editable__.vllm-0.21.0+cu129.pth.bak.021`
- [x] vllm-blog 切换脚本 `use_blog_vllm.sh`(/data1/hgq/),验证门过(import vllm + MooncakeStoreConnector OK,纯 Python 不重编 .so)
- [x] mooncake_config.json 改 store 模式(metadata@9527 + master@9422,从 P2P 切走)
- [x] 起 mooncake daemon:metadata@9527 + master@9422(lease_ttl=5000, protocol=tcp)
- [x] GPU 全空确认(8 卡 4MiB/0%)
- [x] **端到端 sanity**:`MooncakeDistributedStore.setup()` ret=0,TcpTransport 起来,auto-detect TCP-only memcpy,注册 4GB buffer——blog store 路径在 144 TCP 全通
- 坑:`mooncake.store` 原生库不 parse "4GB"(报 -600);blog worker 经 `_parse_size` 转 OK,sanity 手动 parse 验 daemon 通

## Phase 1:用博客三件套跑 baseline(验证 store 起得来)—— 方向纠正后
**⚠️ 方向纠正(2026-07-10,用户提醒)**:博客**不走 verl**,用 `vllm serve` + Rust `vllm-router` + python benchmark 三件套。之前把 blog vllm 塞 verl parallel_infer → verl 0.8.0.dev 只认 vllm 0.21 连环碎。改用博客原生架构。
**目的**:用 blog `vllm serve` 起 1 个 colocated 实例(**TP=1**,Qwen3-8B,MooncakeStoreConnector TCP),确认 store 在 144 TCP 下能起 vllm。
**⚠️ TP=1 而非 TP2**(源码 diff 结论,findings §8):blog worker 无 CPU staging(TP>1 正确性 workaround),纯 TCP + TP>1 会撞 mooncake asio 线程设错 device 的 bug;0.21 用 `MOONCAKE_CPU_STAGING` 绕,blog 没绕 → 先 TP=1 验。
**⏸️ GPU 待命(2026-07-10 用户决策)**:144 全 8 卡被 yrf 占满,147 GPU6 空但 0-5 跑 256-mc;**不抢 GPU**,本阶段只做 CPU 活(router build + 源码 diff 已完成),vllm-serve 待 GPU 空再起。
- [ ] 装 Rust toolchain(rustup + cargo)到 hgq-swe
- [ ] clone vllm-project/router + `cargo build --release`
- [ ] 写 `vllm serve` 启动:`vllm serve Qwen3-8B -tp 2 --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{}}'` + blog PYTHONPATH + mooncake config
- **验证门**:① vllm serve 起得来不崩 ② store worker `register_buffer` 不报错 ③ mooncake master 有 client 注册 ④ health/ready
- 用 benchmark 客户端发 1 条 multi-turn 请求验通链路

## Phase 2:核心对照(验证机制差异)
**科学问题**:blog store connector + 144 TCP,路由策略决定 External hit 有无。
- 两组(都 colocated,1 router + 2-3 vllm-serve 实例):
  - **R-rr**:vllm-router `--policy round_robin`(博客用)→ 预期跨实例查找→External hit>0
  - **R-kvc**:vllm-router cache-aware policy 或单实例对照 → 预期 External hit≈0
- benchmark:`gen_random_multi_turn.py`(prefix-len 20000/input 10000/turn 2048/30 turns)对照博客 131:1 特征
- 抓:external hit rate / prefix hit / prefillT / walltime / store send/recv/register_buffer 次数
- **判定门**:R-rr External hit>0 且 R-kvc≈0 → 坐实"路由决定 mc 是否有用"

## Phase 3:差异归因(对照表,源码级)
产出文档 `findings.md` §博客对照:
1. **路由**:博客 round-robin(跨节点)vs 我方 kvcare(聚合)→ External hit 有无
2. **传输**:博客 GPUDirect RDMA(零拷贝)vs 我方 TCP+CPU staging → 传输开销量级
3. **拓扑**:博客 PD 分离(1P1D)vs 我方 colocated → store 角色(跨实例 vs 同实例)
4. **工作负载**:博客真实 agentic trace(131:1,hit94%)vs 我方 32×8 simulated → hit 基线
5. **代码版本**:blog 72d90b9 重构版 store vs 0.21 旧版 store → API/机制差异

## 已知限制(声明在结论里)
- **不可复现博客 2x 数字**:144 无 RDMA/无 GB200/无 Kimi-K2.5/单机无法 PD 分离
- **NixlConnector 跑不了**:flash_attn .so ABI 不兼容,但 144 无 RDMA 本就做不了 PD,不影响 colocated store 验证
- **本计划验证的是"机制方向"**:为什么我方负收益、博客正收益的前提差异,非数字复现

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| blog vllm + verl 0.8.0.dev:`No module 'vllm.lora.models'` | 1 | 方向纠正:博客不走 verl,改用 vllm-serve+Rust router+benchmark 三件套 |
| Rust 装:`static.rust-lang.org` timeout(容器+host 都不通) | 2 | 换 TUNA 镜像 `RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup`,装上 cargo/rustc 1.97 |
| cargo build 卡 "Updating crates.io index" | 1 | cargo config 配 TUNA `sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/` 镜像,crates 下载通 |
| mooncake.store `Invalid global_segment_size: 4`(sanity) | 1 | 原生库不 parse "4GB";blog worker 经 `_parse_size` 转 OK(实际跑 vllm 自动 parse,sanity 手动 parse 验证 daemon 通) |
| ssh 复合命令 255(cargo/rustc 进程占资源) | 几次 | 拆简单命令 + pkill 精准杀 pid 避免杀 ssh 自身 |
| 容器 proxy `8.92.10.60:7890` 劫持 | 持续 | 所有网络命令 `unset http_proxy https_proxy` + `no_proxy='*'`(memory `kvc-router-proxy`) |
| pyo3 build:host python3.6 < pyo3 最低 3.7 | 2 | host 无 py3.7+;改在 hgq-swe 容器内 build(python3.12 + Python.h + libpython3.12 齐全),pyo3 过 |
| openssl-sys build:容器缺 pkg-config | 1(待解) | `apt-get install -y pkg-config libssl-dev`(reqwest 默认 native-tls→openssl-sys);装完重跑 cargo build |
| ssh 144 timeout/255(host 满载 ~150 容器+yrf 8 卡 vllm) | 几次 | 拆简单命令,脚本文件化(scp+exec)避免复合命令;重试 |

## 配置速查
- 容器:144 `hgq-swe`(hgq-swe-vllm021-mooncake:stage4),host 144 跑 Rust build
- 代码:blog = `/data1/hgq/vllm-blog` @ 72d90b9;router = `/data1/hgq/vllm-router-src`(Rust)
- 切 blog:`PYTHONPATH=/data1/hgq/vllm-blog`(不改 editable 指针)
- mooncake:metadata@9527 + master@9422,`protocol:"tcp"`,`MOONCAKE_CONFIG_PATH=/data1/hgq/mooncake_config.json`
- model:`/data1/models/Qwen/Qwen3-8B`
- env:`MC_TCP_ENABLE_CONNECTION_POOL=1` + `MOONCAKE_CPU_STAGING=1` + `VLLM_HOST_IP=127.0.0.1` + unset proxy
- 网络镜像:rustup/cargo 全用 TUNA(github/crates.io/rust-lang.org 部分不通)

## 决策门
- Phase 1 vllm serve 起不来(store 初始化错)→ 查 store worker blog 版初始化报错(已 import OK + sanity OK,风险低)
- Phase 2 R-rr External hit 仍=0 → 说明 144 TCP 下 round-robin 也不触发跨 replica 命中,改抓 store send/recv 次数证明跨 replica 查找发生过
- 任何 phase 若证伪我方"负收益"结论 → 如实记录
