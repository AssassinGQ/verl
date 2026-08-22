# Findings — 复现官方 vLLM×Mooncake 博客(144 TCP 验证机制)

## §1. 官方博客真实配置(查证,2026-07-10)

**博客**: https://vllm.ai/blog/2026-05-06-mooncake-store 《Serving Agentic Workloads at Scale with vLLM x Mooncake》
- TL;DR: 3.8× 吞吐、46× P50 TTFT↓、8.6× E2E latency↓,线性扩展到 60 GB200 GPU
- 驱动 = cache hit 1.7%→92.2%(真实 trace)/ >95%(全规模)

**代码**: ivanium/vllm `feat/mooncake-store-int` @ `72d90b9a627a`(2026-05-11)
- 相对 v0.21.0: diverged, ahead 28 / behind 358 → 博客基于 0.21 之后 +358 commit 的 main 前身
- PR #40900(`feat/mooncake-store-connector`,作者 LCAIZJ)2026-05-12 合入 vllm main(base=main),merge_commit=ebeb09d
- store 布局**扁平**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_{connector,worker,scheduler,data,metrics}.py`(vs 0.21 旧版子目录 `mooncake/store/`)
- store worker 默认 `protocol=config.get("protocol", "tcp")`(line 88)——**博客默认 tcp 非 rdma**

**硬件/软件**:
- Kimi-K2.5-NVFP4,CUDA 13.1,3 节点 Slurm
- 1P(1 node×4 GPU TP4)+ 1D(2 node×4 GPU DEP8 decode)= 12 GPU,扩展到 60
- 传输: UCX_TLS=cuda_ipc,cuda_copy,tcp; NCCL_IB_HCA=mlx5_0/1/3/4(真 IB/RDMA NIC); MC_ENABLE_DEST_DEVICE_AFFINITY=1; MC_WORKERS_PER_CTX=4; MC_SLICE_SIZE=524288
- kv-transfer-config: **MultiConnector** = NixlConnector(PD,kv_buffer_device=cuda)+ MooncakeStoreConnector(store pool,load_async=true, enable_cross_layers_blocks=true)
- vllm serve 关键: --block-size 64 --max-model-len 118400 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --disable-hybrid-kv-cache-manager --enforce-eager -O3

**工作负载**: Codex/GPT-5.4 SWE-bench Pro 610 traces,median 33 turns/trace,turn30 ctx~80K(最长>180K),input:output=131:1,平均 2242 tok/turn,cache hit 94.2%
- dataset: huggingface.co/datasets/Inferact/codex_swebenchpro_traces
- scaling synthetic: 20K common+10K 首input+2048/turn+900 output+30 turns,sessions 75→375 随 GPU 数

**博客 2x 收益来源**: ①跨实例 prefix cache hit(避免迁 replica 重算 prefill)②RDMA 高带宽让 KV 传输开销 < 重算 prefill
- 路由用 round-robin(刻意制造跨节点流量压测 datapath)

## §2. 我方负收益结论(exp3/exp4,findings §20/§24)

3090 TCP loopback + colocated + kvcare 聚合下 mc **净负向**:
- exp3: C−A walltime +7.4%、D−B walltime +30.6%,External hit(C 0.69%/D 0.00%)
- exp4: C₁/A +14.8%、D/B +20.9%,External hit=0(D)

**根因(exp4 §5 源码级实锤)**:
1. kvcare prefix 聚合→同 prompt 全在同 replica→External hit 恒=0(mc 无 L2 命中可省)
2. TCP+CPU staging 传输开销 > prefill 节省(无 RDMA)
3. mc L2 staging 碎块写入破坏 prefix 批量化(blocks/event B 32.2→D 1.80,+656 万额外 GPU 分配)

## §3. 差异归因(博客 vs 我方,待 Phase 2/3 验证)

| 收益前提 | 博客 | 我方(3090 loopback) | 后果 |
|---|---|---|---|
| ①跨实例 prefix hit | round-robin→跨节点→mc L2 命中省重算 | kvcare 聚合→同 prompt 同 replica→External=0 | mc 无 L2 命中可省 |
| ②传输 < 重算 | GPUDirect RDMA 零拷贝不经 SM | TCP loopback + CPU staging | 传输开销 > prefill 节省 |
| ③工作负载 hit 高 | 真实 agentic trace 131:1 hit94% | 32×8 simulated 单轮假设 hit29-51% | mc 省得少 |

reconcile: 我方"负收益"是 3090 TCP-loopback + colocated + kvcare 聚合 regime 下正确结论,不可外推真集群 RDMA。博客 "What's next" 自己把 cache-aware routing 列未来工作——我方 kvcare 已做(prefix 聚合),这反而消灭 mc External hit 前提。我方主线价值与官方路线不矛盾,互补两层。

## §5. 博客架构 = vllm-serve + Rust router + benchmark 三件套(不用 verl!2026-07-10 查证)

**之前的错误**:我把 blog vllm 塞进 verl `parallel_infer.py` 跑,verl 0.8.0.dev 只认 vllm 0.21,连环碎 30+ 处 import。用户提醒"用 blog 里官方的配置"后查证:

博客真实架构是**三个独立组件,全程不走 verl**:
1. **vllm serve**(`start_1p1d_prefill.sh`):`exec vllm serve Kimi-K2.5 ... -tp 4 --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector",...},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"enable_cross_layers_blocks":true}}]}}'`
2. **vllm-router**(Rust 二进制,`start_1p1d_router.sh`):`ROUTER_REPO/target/release/vllm-router --policy round_robin --vllm-pd-disaggregation --prefill URL --decode URL ...`
   - 仓库 = **github.com/vllm-project/router**(Rust),`cargo build --release` 编译
   - 支持 policies: cache-aware, power-of-two, consistent-hash, random, round-robin
   - **支持非 PD colocated 模式**:`--worker-urls http://w1 http://w2 --policy round_robin`(不需 `--vllm-pd-disaggregation`)
3. **benchmark 客户端**(python,`run_1p1d_load_test.sh`):
   - `benchmarks/multi_turn/gen_random_multi_turn.py`(造数据:--random-prefix-len 20000 --random-input-len 10000 --per-turn-input-len 2048 --num-prompts 75 --multi-turn-num-turns 30)
   - `benchmarks/multi_turn/benchmark_serving_multi_turn.py`(发请求:--max-active-conversations 75 --limit-min/max-tokens 900)

**博客真实环境**:`/usr/local/cuda-13.1`,Kimi-K2.5-NVFP4,3 节点 Slurm,UCX_TLS=cuda_ipc,cuda_copy,tcp,NCCL_IB_HCA=mlx5_0/1/3/4(真 RDMA)

## §7. 144 环境/网络坑 + 镜像方案(2026-07-10,关键会话知识)

**网络可达性矩阵**(144 容器 + host,unset proxy 后):
| 站点 | 容器 | host | 备注 |
|---|---|---|---|
| github.com / api.github.com / raw.githubusercontent | ✅ | ✅ | clone/fetch 正常 |
| sh.rustup.rs | ✅ | ✅ | rustup-init stub 可下 |
| crates.io 根 | 403(正常) | — | cargo 用 index.crates.io |
| index.crates.io | — | ✅ 200 | sparse index 可达但慢 |
| **static.rust-lang.org** | ❌ timeout | ❌ timeout | **rustup toolchain 下载处,不通** |
| **TUNA 镜像** mirrors.tuna.tsinghua.edu.cn | ✅ | ✅ | **rustup + crates 全走这里** |

**容器 proxy 劫持**:`hgq-swe` 设了 `HTTP_PROXY=http://8.92.10.60:7890`(unreachable,memory `kvc-router-proxy`)。所有外网命令必须 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` + `export no_proxy='*'`。

**Rust toolchain 装法**(host):
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
curl -sSL --noproxy '*' -o /tmp/rustup-init-host https://sh.rustup.rs
/tmp/rustup-init-host -y --default-toolchain stable --profile minimal --no-modify-path
# → cargo 1.97 + rustc 1.97(host,够新)
```

**cargo 镜像配置** `/root/.cargo/config.toml`:
```toml
[source.crates-io]
replace-with = "tuna"
[source.tuna]
registry = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"
[net]
git-fetch-with-cli = true
```
否则 `cargo build` 卡在 "Updating crates.io index"。

**ssh 坑**:144 上跑 `pkill -9 -f rustc` 会杀到 ssh 自身或致连接 255。应精准按 PID 杀,或拆简单命令。

## §6. 144/147 适配性(2026-07-10)
- 144(8×3090)与 147(8×3090)都**无 RDMA/无 GPUDirect**,只有 TCP loopback。blog 2x 的传输前提(GPU 直读显存)在两台机器上**物理上不满足**。
- blog vllm-serve 把 GPU 地址直接交给 mooncake(无 CPU staging,见 §8)。在纯 TCP 上,要么靠 mooncake TcpTransport 自己处理 GPU 地址(TP=1 单卡可能 OK),要么撞 0.21 注释里"mooncake asio 线程 TP>1 设错 device"那个 bug(blog 代码没带 staging workaround)。
- **GPU 占用现状(2026-07-10)**:144 全 8 卡被 `yrf-swe-mooncake`(别人容器,不能动)占满(100% util,~22.3GB/卡);147 的 GPU6 空闲(0MiB)但 0-5 卡在跑我方 256-mc 矩阵(parallel_infer/kvc_aware_router,06:22 启,`--n-gpus-per-node 6` 钉 0-5)。用户决策:**先只做 CPU 活,GPU 实验待命**(不抢 147 GPU6 干扰 256-mc)。
- 可行的 Phase 1 验证(待 GPU 空):blog `vllm serve` Qwen3-8B **TP=1** + `MooncakeStoreConnector`(独占 mooncake 端口避开 256-mc),验 store 起得来。

## §8. 源码级差异(2026-07-10):blog store worker vs 0.21 store worker —— 负收益机制实锤
对照(已 scp 本地):
- blog:`/data1/hgq/vllm-blog/.../mooncake/mooncake_store_worker.py`(1192 行,扁平布局)
- 0.21:容器 `/data1/hgq/vllm-src/.../mooncake/store/worker.py`(1253 行,子目录,带 `.bak.tokenids` vendored 补丁)

**① 传输路径:0.21 多一层 CPU staging,blog 没有(根因 #2 源码实锤)**
- 0.21(gated `MOONCAKE_CPU_STAGING=1`,我们实验都开了):
  - send(line 588-603):`_stage_put_batch` 先 GPU→CPU `cudaMemcpy`,再 `batch_put_from_multi_buffers(keys, staging_addrs=CPU_addrs, sizes)`
  - recv(line 711-748):`_stage_get_batch` 分配 CPU target → `batch_get_into_multi_buffers` 收到 CPU → `_unstage_get_batch` CPU→GPU `cudaMemcpy`
  - 注释(line 289-297):"mooncake's asio thread does cudaMemcpy on the wrong device under TP>1" → CPU staging 是 **TP>1 正确性 workaround**(把 copy 挪到 worker 线程的正确 CUDA context)
  - 每次传输 = **2 次额外同步 cudaMemcpy**(D→H send + H→D recv)叠加在 TCP 上 → 这正是我们测到的 staging 开销
- blog:`grep -E "_stage_put_batch|_stage_get_batch|_unstage_get_batch|MOONCAKE_CPU_STAGING|_cuda_memcpy|_CpuStagingBuffer"` blog worker = **0 命中**。blog send(line 486)直接 `batch_put_from_multi_buffers(keys, addrs=GPU_addrs, sizes)`,addrs 来自 `token_database.prepare_value()` 的 **GPU 地址**。**blog 把 GPU 地址直接交给 mooncake,靠 GPUDirect(cuda_copy/RDMA)直接读显存,无 CPU 中转**。
- 结论:blog 2x 的传输前提是 GPUDirect(GB200 RDMA)。144/147 只有 TCP loopback,blog 这条无 staging 路径在 TP>1 下会撞 0.21 注释里那个 multi-device bug → **Phase 1 起博客 vllm-serve 应先用 TP=1 规避**(blog 代码没带 TP>1 staging workaround)。

**② lookup 跨实例命中机制:两边一样(根因 #1 = 路由,非 store 代码)**
- blog lookup(line 1070-1130)与 0.21 lookup(line 1098-1144)**核心逻辑相同**:都 `self.store.batch_is_exist(multi_tp_keys)` 查全局共享 store pool(metadata server)。任何 replica PUT 的 block 对所有 replica lookup 可见 → 跨实例 prefix hit = pool 里有。pool 不区分来源(memory `mooncake-external-hit-no-cross-replica` 同一点)。
- 0.21 额外有 `LookupKeyServer`(worker rank 0)+ `LookupKeyClient` + `get_zmq_rpc_path_lookup`(line 1166-1247)—— ZMQ IPC,让 connector 进程经 ZMQ 调 worker 的 lookup。这是 verl/colocated 多进程布局的**进程间通信管道**,非命中语义差异。blog 单进程 `vllm serve` 直接调 `self.store_worker.lookup()`,不需这层 ZMQ。
- 结论:我们 External hit=0 **不是 store 代码差异,是路由**(kvcare 聚合 → 同 prompt 同 replica → prefix 在本地 L1,永远不到 store 查)。round-robin 才会把同 prefix 打到不同 replica 触发跨实例 store 命中。

**③ blog 新增 disk offload,0.21 没有**
- blog 有 `_get_disk_offload_buffer_budget_bytes` / `_estimate_disk_offload_staging_bytes` / `_split_disk_offload_load_batches`(line 104-215),支持 KV 落盘做显存分级。0.21 无。属容量机制,与负收益主线无关。

**总结(源码闭环)**:
| 根因 | blog | 我方(0.21+MOONCAKE_CPU_STAGING=1) | 源码证据 |
|---|---|---|---|
| ①传输开销 | GPU 直读(GPUDirect),无 CPU staging | 2× cudaMemcpy(D→H+H→D)+TCP | blog 0 staging symbol;0.21 line 588-603/711-748 |
| ②External hit | lookup 查共享 pool,同 0.21 | 路由聚合→prefix 不出本地 L1→pool 查不到 | 两边 lookup 同逻辑;0.21 ZMQ client 仅 IPC |
| ③容量 | disk offload | 无 | blog line 104-215 |
