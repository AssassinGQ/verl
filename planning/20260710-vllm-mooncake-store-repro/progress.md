# Progress — 复现官方 vLLM×Mooncake 博客(144 TCP 验证机制)

## 2026-07-10

### 计划启动 + 可行性验证(完成)
- 官方博客 https://vllm.ai/blog/2026-05-06-mooncake-store 查证完成:2x+ 吞吐(3.8×/46×TTFT↓),前提=GB200 RDMA + PD 分离 + round-robin + 真实 agentic trace(131:1,hit94%)
- PR #40900 已合入 vllm main(2026-05-12);博客真实代码分支=ivanium/vllm `feat/mooncake-store-int` @ `72d90b9a627a`(2026-05-11)=0.21+358 重构版
- **144 现有 vllm0.21 stage4 镜像 ≠ 博客版本**(store 旧版 API 子目录 vs 博客扁平布局)
- 用户决策:**代码和版本跟官方一致** + **只在 hgq 容器操作不改别人 docker**

### Phase 0 前置验证(完成,见 task_plan §已验证可行性)
- ✅ 浅克隆博客代码:`/data1/hgq/vllm-blog` @ 72d90b9(121M,144 hgq-swe 容器内)
- ✅ `PYTHONPATH=/data1/hgq/vllm-blog` import vllm + `MooncakeStoreWorker` 全 OK(store 纯 Python,不依赖重编 .so)
- ✅ 容器内 mooncake 0.3.11 `from mooncake.store import MooncakeDistributedStore` OK
- ✅ torch 2.11.0 匹配(blog pyproject 要求 ==2.11.0,容器内正是 2.11.0+cu128)
- ✅ 博客 store 默认 protocol="tcp"(line 88),144 无 RDMA 正好走 TCP
- ⚠️ NixlConnector import 失败(flash_attn _vllm_fa2_C/_vllm_fa3_C ABI 不兼容 0.21 编译 .so)——但 144 无 RDMA,PD 分离做不了,不影响 store 验证

### Phase 0 实施(✅ 完成)
- ✅ 备份 0.21 editable 指针 → `__editable__.vllm-0.21.0+cu129.pth.bak.021`
- ✅ 部署 `use_blog_vllm.sh`(/data1/hgq/,bind-mount 进容器)→ 验证门过:`PYTHONPATH=/data1/hgq/vllm-blog` import vllm + MooncakeStoreConnector 全 OK,store 纯 Python 不依赖重编 .so
- ✅ mooncake_config.json 改 store 模式(metadata http://127.0.0.1:9527/metadata + master 9422,从旧 P2P 切走)
- ✅ 起 mooncake daemon:metadata@9527(pid 18181)+ master@9422(lease_ttl=5000, rpc_port=9422, protocol=tcp,pid 18206)
- ✅ GPU 全空(8 卡 4MiB/0%,无 compute apps;ray 残留全 zombie 无活跃 worker)
- ✅ **端到端 sanity**:`MooncakeDistributedStore.setup()` ret=0,TcpTransport listen 15602,auto-detected TCP-only memcpy,注册 4GB local buffer——**blog store 路径在 144 TCP loopback 全通**

**坑(已解)**:`mooncake.store` 原生库不解析 "4GB"(当成 int 4→报错 -600);blog worker 经 `_parse_size` 转字节才 OK。sanity 手动 parse_size 验证 daemon 路径本身没问题;vllm 跑时 blog worker 自动 parse。

### Phase 1 进行中
- ✅ sync 144 uni-agent(非 git、旧 checkout,缺 examples/llm_router):从本地 llm-router 分支 tar 打包 sync `uni_agent/llm_router/`(balancer.py 更新到 66761fe)+ `examples/llm_router/`(parallel_infer/run_infer/agent_config_simulated/standalone_collector)
- ✅ 清理 144 stale 文件:`collectors/collector/` 包目录 shadow 了 `collector.py`(import 冲突)→ 删整个 collectors 重解;`strategies/load_score.py` 已删(local 没有)
- ✅ 全 import chain 验证:`PYTHONPATH=/data1/hgq/vllm-blog:/data1/hgq/uni-agent` 下 vllm(blog)+ MooncakeStoreConnector + verl + uni_agent + KVCAwareBalancer 全 OK
- 🔄 跑 Phase 1 最小 mc 实验(detached):1 replica TP2,Qwen3-8B,`--max-samples 4 --n 2 --max-turns 3`,验 `MooncakeStoreConnector` 真能起 vllm + 产 KV 事件。log: `/tmp/phase1_run.log`
- 验证门:① vllm 起得来不崩 ② mooncake master 有注册 ③ store worker register_buffer 不报错 ④ 产 BlockStored 事件

### ⚠️ Phase 1 阻塞:blog vllm 与容器 verl 版本不匹配(2026-07-10)
- 跑 parallel_infer + blog PYTHONPATH 失败:`ModuleNotFoundError: No module named 'vllm.lora.models'`
- 根因:容器内 **verl 0.8.0.dev ↔ vllm 0.21 是匹配对**。blog vllm(0.21+358)重构了 `vllm.lora`(0.21 `vllm.lora.models.LoRAModel` → blog `vllm.lora.lora_model.py`+`model_manager.py`),verl `utils/vllm/utils.py:22` import 不到
- 范围:verl→vllm import 点 30+(`vllm_async_server.py`/`utils.py`/`patch.py`),逐个 patch 是兔子洞

### 🔄 方向纠正:博客根本不用 verl,用 vllm-serve + Rust router + benchmark 三件套(2026-07-10,用户提醒)
- 之前错误:把 blog vllm 塞进 verl `parallel_infer.py` → verl 0.8.0.dev 只认 vllm 0.21 → 连环碎
- **博客真实架构(查证)**:三个独立组件,全不走 verl:
  1. **vllm serve**:`exec vllm serve Kimi-K2.5 ... --kv-transfer-config 'MultiConnector...'`(prefill+decode 各一实例)
  2. **vllm-router(Rust 二进制)**:`ROUTER_REPO/target/release/vllm-router --policy round_robin --vllm-pd-disaggregation --prefill ... --decode ...`(仓库=`vllm-project/router`,Rust)
  3. **benchmark 客户端**(python):`benchmarks/multi_turn/gen_random_multi_turn.py` 造数据 + `benchmark_serving_multi_turn.py` 发请求
- **关键**:vllm-router 支持非 PD colocated 模式 `--worker-urls w1 w2 --policy round_robin`,适配 144 单机
- 144 现状:blog vllm ✅(已 clone),benchmark 脚本 ✅(vllm-blog 里),**Rust/cargo ❌**(需装)

### Phase 1 重新规划(用博客三件套,不用 verl)
- [ ] 装 Rust toolchain(rustup + cargo)到 hgq-swe 容器
- [ ] clone vllm-project/router + `cargo build --release` 编 vllm-router 二进制
- [ ] 用 blog vllm 起 1 个 colocated vllm-serve 实例(TP2,Qwen3-8B,`--kv-transfer-config MooncakeStoreConnector`),验 store 起得来
- [ ] (后续 Phase 2)起 2-3 个 vllm-serve + vllm-router round_robin,跑 benchmark 对照

### ⚠️ Phase 1 阻塞:Rust toolchain 装不上(2026-07-10)→ 已解
- 容器 `hgq-swe` 无 cargo/rustc,博客 vllm-router 是 Rust,需 `cargo build --release`
- 容器有 unreachable proxy `HTTP_PROXY=8.92.10.60:7890`(memory `kvc-router-proxy`),已 unset 绕过
- 但 `static.rust-lang.org`(rustup 下载 toolchain 处)从容器和 host 都 timeout——github/crates.io/sh.rustup.rs 通,唯独 rust-lang.org 不通
- **解=换 TUNA 镜像**:`RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup`,host 装上 `cargo 1.97.0 + rustc 1.97.0`(够新,支持 pyo3 0.26/axum 0.8)
- 🔄 cargo build vllm-router(host 上,release)进行中

### Phase 1 验证(待 cargo 完成后跑)
- ✅ `vllm serve --help` 在 blog vllm 下可用(api_server importable)
- ✅ `MooncakeStoreConnector` 在 factory 注册(名解析 OK),`vllm serve --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector",...}'` 将可用
- ✅ mooncake daemon up(metadata@9527 + master@9422,store.setup() sanity ret=0 TCP 通)
- [ ] cargo build vllm-router → 二进制
- [ ] 起 1 个 colocated vllm-serve 实例 验 store 起得来

### 续会话(2026-07-10 晚):cargo build 连环坑 + GPU 调度决策 + 源码级差异实锤
- **cargo build 第 1 次失败**:host 跑(detached pid 44129)pyo3 检测到 host python3=3.6(<3.7 最低)。host 只有 py3.6(Ubuntu18.04),无 conda。**根因**:pyo3 是 hard dep(Cargo.toml `pyo3=0.26 features=["extension-module"]`,crate-type cdylib,不能跳)。
- **改在容器内 build**(hgq-swe 有 python3.12 + Python.h + libpython3.12.so):写 `build_router_in_container.sh`(TUNA 装 rustup + cargo TUNA config + cargo build --release),scp 到 /data1/hgq,docker exec -d 跑,log=/data1/hgq/build_container.log。
- **cargo build 第 2 次失败**:crates 全下完(TUNA OK),pyo3 这次过了(python3.12 满足),但 `openssl-sys` 编译缺 `pkg-config`。**根因**:reqwest 默认 native-tls→openssl-sys,容器没装 pkg-config + libssl-dev。
- **修法(待执行,ssh 144 突然 timeout)**:容器内 `apt-get install -y pkg-config libssl-dev`(脚本 /data1/hgq/check_install_deps.sh 已 scp 但 ssh 断未确认),装完重跑 cargo build。ssh 144 此刻 timeout(host 上 ~150 容器 + yrf 8 卡 vllm 满载,memory 记 ssh 255 易发)。
- **mooncake daemon 已死**:之前 session 起的 metadata@9527 + master@9422 都不在了(host/容器都没 listen),Phase 1 起 vllm-serve 前需重启。
- **benchmark 脚本**:`benchmark_serving_multi_turn.py` 在 vllm-blog 有,但 `gen_random_multi_turn.py` **不在**(blog 用的是别的造数据方式,Phase 2 再查)。

#### GPU 调度决策(用户拍板)
- 144 全 8 卡被 `yrf-swe-mooncake`(别人容器)占满(100%/~22.3GB 每卡),不能动。
- 147 GPU6 空闲(0MiB)但 0-5 卡在跑我方 256-mc 矩阵(parallel_infer/kvc_aware_router,06:22 启,钉 0-5)。
- **用户决策:先只做 CPU 活,GPU 实验待命**(不抢 147 GPU6 干扰 256-mc)。
- → 当前阶段:把 router build 跑完(CPU)+ 完成 Phase 3 源码差异归因(不需 GPU)。vllm-serve 起不来待 GPU 空。

#### Phase 3 源码级差异(✅ 完成,详见 findings §8)
对照 blog worker(1192 行)vs 0.21 worker(1253 行),**根因源码实锤**:
1. **传输开销(根因 #2)**:0.21 gated `MOONCAKE_CPU_STAGING=1` 下每次传输 2× 同步 cudaMemcpy(D→H send + H→D recv,是 TP>1 正确性 workaround,注释 line 289-297);blog **0 个 staging symbol**,直接把 GPU 地址交 mooncake 靠 GPUDirect 读。→ blog 2x 传输前提是 GPUDirect,144/147 TCP 不满足;且 blog 代码无 TP>1 staging workaround → **Phase 1 验证须 TP=1**。
2. **External hit(根因 #1)**:blog/0.21 lookup 核心逻辑相同(都 `batch_is_exist` 查共享 pool);0.21 的 ZMQ LookupKeyClient/Server 仅是 verl 多进程 IPC 管道非语义差异。→ External hit=0 是路由(kvcare 聚合)非 store 代码。
3. blog 新增 disk offload(0.21 无),属容量机制,主线无关。

#### vllm-router 二进制 build ✅ 完成
- host python3.6 < pyo3 最低 3.7 → 改 hgq-swe 容器内 build(python3.12 + Python.h + libpython3.12 齐全,pyo3 过)。
- 第 3 处坑:openssl-sys 缺 pkg-config → 容器内 `apt install pkg-config libssl-dev` 解,重跑 cargo build 过。
- **二进制**:`/data1/hgq/vllm-router-src/target/release/vllm-router`(24.5MB),`--help` 验通:colocated `--worker-urls w1 w2` + `--policy {random,round_robin,cache_aware,power_of_two,consistent_hash,rendezvous_hash}`(默认 cache_aware)。→ Phase 2 R-rr(round_robin)vs R-kvc(cache_aware)对照策略齐备。
- Phase 1 启动脚本备好(TP=1):`start_vllm_store.sh`(已按源码 diff 改 TP2→TP1、去 MOONCAKE_CPU_STAGING、加 GPU pin)+ `start_mooncake_daemon.sh`(daemon 死了,Phase 1 起前先跑)。两脚本在本机 /tmp,待 scp 到 144。
