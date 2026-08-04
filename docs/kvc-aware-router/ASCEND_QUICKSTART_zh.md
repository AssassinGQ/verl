# KV-Cache-Aware Router — Ascend 全流程部署(镜像 → 推理 → 观测)

Last updated: 08/04/2026.

从一张**基础镜像**出发,在 Ascend NPU 服务器上跑通 kvc-aware router 推理 +
rl-insight 在线观测的端到端指南。合并了 [`examples/kvc_aware_router/README_cn.md`](../../examples/kvc_aware_router/README_cn.md)(推理部署)
与 [`README_zh.md`](README_zh.md)(观测),并把 NVIDIA 流程改写到 Ascend。

> 本文只覆盖 **kvc-aware router 推理 + 观测**这条线。verl 训练 / 通用 NPU 用法见
> [`docs/ascend_tutorial/`](../ascend_tutorial/README.md)(权威版本矩阵、宿主 CANN 安装、
> 性能调优都在那)。

---

## ⚠️ 先读:版本兼容(Ascend 上最容易翻车的地方)

Ascend 软件栈是**强版本绑定**的——CANN ↔ torch_npu ↔ torch ↔ vLLM ↔ vLLM-Ascend 必须自洽,
错一档 vLLM worker 直接起不来。verl 官方验证过的组合(来自
[`install_guidance.rst`](../ascend_tutorial/get_start/install_guidance.rst),2026/05 更新):

| 组件 | verl 推荐版本 |
|------|--------------|
| HDK | `26.0.rc1` |
| CANN | `9.0.0` |
| torch / torch_npu | `2.9.0` / `2.9.0.post2` |
| triton-ascend | `3.2.1` |
| **vLLM** | **`0.18.0`** |
| **vLLM-Ascend** | **`0.18.0`** |
| transformers | `5.3.0` |
| Python | `3.11` |

**你手里的 `0.23.0` 镜像不在 verl 验证矩阵里**(矩阵最高 0.18)。两种走法:

- **求稳(推荐)**:忽略 0.23.0,用 verl 一键脚本装官方 0.18.0 组合(本文 §3.3)。vllm-ascend
  的 API 在 0.18↔0.23 之间有变动,kvc-aware router 依赖的 `kv-events` / `/metrics` 在 0.18 已验证。
- **坚持 0.23.0**:先验证镜像里 CANN / torch_npu / vllm 的实际版本(§3.2),再把
  vllm-ascend 对到 `releases/v0.23.0`(vllm-ascend 版本号**必须**等于 vllm 版本号),并做好
  kv-events 接口对不上的心理准备。

---

## 1. 前置(宿主机)

- NPU 驱动 + CANN 已装在**宿主**:`npu-smi info` 能出卡;`/usr/local/Ascend/driver`、
  `/usr/local/Ascend/firmware` 存在(容器要挂)。
- 宿主有 docker,且 `/var/run/docker.sock`、`/usr/bin/docker` 可用(local docker 沙箱要 docker-in-docker)。
- 模型权重、数据盘都在宿主某个 `<DATA_DIR>` 下(如 `/data1`)。

> 宿主 CANN 没装?按 [`docs/ascend_tutorial/get_start/install_guidance.rst`](../ascend_tutorial/get_start/install_guidance.rst)
> §"HDK + CANN 宿主安装"走(yum 装 `Atlas-A3-hdk-npu-driver` + `Ascend-cann-toolkit`)。

---

## 2. 起 Ascend 容器(`--privileged` + driver 挂载)

verl 的 Ascend Docker **不用** `--device /dev/davinci*` 那套,而是 `--privileged` + 把宿主
Ascend driver/firmware 挂进去(见 [`dockerfile_build_guidance.rst`](../ascend_tutorial/get_start/dockerfile_build_guidance.rst))。
再叠上 local docker 沙箱需要的 docker.sock / fuse 挂载:

```bash
# <IMAGE_NAME> = 你的基础镜像(如 cann:9.0.0-a3-... 或你下载的 0.23.0 镜像)
# <DATA_DIR>   = 宿主数据盘(模型/数据集/wheels 都放这),如 /data1
IMAGE_NAME=<你的镜像> DATA_DIR=/data1 CONTAINER_NAME=swe-ascend SHM_SIZE=10g \
docker run -dit \
  --ipc=host --network host \
  --name ${CONTAINER_NAME:-swe-ascend} \
  --privileged \
  --cap-add SYS_ADMIN --device /dev/fuse \
  --shm-size=${SHM_SIZE:-10g} \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v ${DATA_DIR}:${DATA_DIR} \
  -v /tmp:/tmp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  ${IMAGE_NAME} /bin/bash
```

> `--privileged` 已覆盖 device/cap,显式写 `--cap-add SYS_ADMIN --device /dev/fuse` 是为了让
> fuse sandbox 能起(沿用 NVIDIA 流程的沙箱需求)。

进容器 + 验证 NPU + 激活 CANN 环境(**每次进容器都要 source**):

```bash
docker exec -it swe-ascend bash
npu-smi info                                     # 看到 NPU 卡 = driver 挂载 OK
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true   # atb 可能没装,忽略
python -c "import torch_npu; print(torch_npu.__version__)"         # 有输出 = torch_npu 在
```

---

## 3. 容器内:基础镜像 → 可跑 verl + vllm-ascend

### 3.1 激活 CANN env(同 §2 末尾,新 shell 必做)

### 3.2 先验证镜像现状,决定装多少

```bash
python -c "import torch,torch_npu; print('torch',torch.__version__,'npu',torch_npu.__version__)"
python -c "import vllm; print('vllm',vllm.__version__)"
python -c "import vllm_ascend; print('vllm_ascend',vllm_ascend.__version__)"
```

- 三个都有且版本自洽 → 跳到 §3.4(只装 verl + swe-rex)。
- 缺 vllm/vllm-ascend,或版本对不上 → 走 §3.3 一键脚本重装。

### 3.3 装 vllm + vllm-ascend(verl 官方一键脚本)

verl 提供 `scripts/install_vllm_mcore_npu.sh`,`USE_MEGATRON=0` = 只要 FSDP(推理不需要 Megatron)。
它从源码装 **vllm 0.18.0 + vllm-ascend 0.18.0 + triton-ascend 3.2.1 + transformers 5.3.0** 一套:

```bash
cd /path/to/uni-agent        # 见 §3.4,先把 verl 源码弄进来
USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh
```

> **要 0.23.0 的话**:脚本里把 `v0.18.0` 改成 `v0.23.0`(vllm 和 vllm-ascend 两处 git checkout
> 都改),其余按你镜像的 CANN/torch_npu 版本对齐。建议先用 0.18.0 baseline 跑通,再升 0.23。

### 3.4 装 verl + swe-rex 等推理依赖

```bash
# 拿到 uni-agent 源码(含 verl submodule)
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
git submodule update --init --recursive

# verl(NPU 用 requirements-npu.txt,不是 requirements.txt)
pip install -e verl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r verl/requirements-npu.txt \
    --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ \
    --trusted-host triton-ascend.osinfra.cn

# swe-rex + 推理侧依赖
pip install swe-rex loguru pydantic pydantic_settings boto3 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir swebench -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> `requirements-npu.txt` 已钉 `numpy<2.0.0`(NPU 上 numpy 2.x 会触发 dtype repr 递归)和
> `triton-ascend==3.2.1`,不用再手动处理 numpy。

### 3.5 关键 env(Ray on NPU 必设)

```bash
# 用哪几张卡(按你的卡数改,A3 常见 8 卡或 16 卡)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 绕过 Ray 的 is_npu_available 误检测 —— verl 文档/CI 都设,务必带上
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
```

> 注意:`ascend-exps.sh` 里**漏了** `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`(历史
> 遗漏)。手动跑时加上更稳;若多 worker 抢卡/卡识别异常,第一个就查这个。

---

## 4. 装 rl-insight(在线观测)

观测依赖一个运行中的 rl-insight server(Prometheus + Grafana + api)。本 router 推送的
dashboard + 自定义桶透传在 `feat/verl-agentic-rollout-dashboard` 分支——**必须用这个分支**。

### 4.1 标准安装(GitHub + dl.grafana.com 可达)

```bash
git clone https://github.com/touch869/rl-insight.git
cd rl-insight && git checkout feat/verl-agentic-rollout-dashboard
pip install -e .
rl-insight server install     # 一次性:下 Prometheus/Grafana/Tempo
rl-insight server start       # Prometheus :9090、Grafana :3000、api :18080
```

### 4.2 受限环境(GitHub 被墙 / 代理 MITM 致 TLS 失败)

三道坎:

- **clone 失败** → git bundle:可达机 `git bundle create rl-insight.bundle --all`,scp 过去再 clone。
- **pip 被墙** → `pip install -e . --no-build-isolation`(用本机 setuptools)。
- **`server install` TLS/超时** → 别处下好包放目录,用自建 config 钉版本(`--local-archive`
  不扫目录,rl-insight 仍要 `install_version` 拼文件名 `prometheus-{ver}.linux-amd64.tar.gz`):
  ```bash
  mkdir -p <DATA_DIR>/rl-archives
  # 放: prometheus-2.54.1.linux-amd64.tar.gz、grafana-13.0.0.linux-amd64.tar.gz
  cat > <DATA_DIR>/rl-archives/install-config.yaml <<'EOF'
  prometheus: { enable: true, install_version: "2.54.1" }
  grafana:    { enable: true, install_version: "13.0.0" }
  tempo:      { enable: false }
  EOF
  rl-insight server install --local-archive <DATA_DIR>/rl-archives \
      --config <DATA_DIR>/rl-archives/install-config.yaml
  ```

详见 [`README_zh.md`](README_zh.md) *受限环境*。

---

## 5. 准备数据 + 沙箱镜像(local docker 真实)

### 5.1 生成 parquet

```bash
# local docker 沙箱用 modal(默认),docker 会自动拉 swebench/sweb.eval.x86_64.*
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py \
    --local-save-dir examples/kvc_aware_router
```

输出 `examples/kvc_aware_router/swe_bench_verified_modal.parquet`。

### 5.2 预下载 swe-rex wheels(避免 500 沙箱并发 pip 打爆源)

```bash
# <WHEELS_DIR> 放在挂载的 <DATA_DIR> 下,容器内可见,如 /data1/swe_wheels
pip download swe-rex -d <WHEELS_DIR> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

把 [`examples/kvc_aware_router/agent_config_localdocker.yaml`](../../examples/kvc_aware_router/agent_config_localdocker.yaml)
里 swe-rex wheels 的挂载源路径改成你的 `<WHEELS_DIR>`。

### 5.3 拉 SWE-bench 镜像(国内用火山引擎 CR)

镜像名在 parquet 的 `extra_info` 字段(`swebench/sweb.eval.x86_64.<instance>`)。先解析出列表
(脚本见 [`examples/kvc_aware_router/README_cn.md`](../../examples/kvc_aware_router/README_cn.md) §4.5),
再国内拉取 + tag:

```bash
docker pull enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.<instance>:v2
docker tag  enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.<instance>:v2 \
            swebench/sweb.eval.x86_64.<instance>:latest
```

---

## 6. 运行(推理 + 观测一起跑)

### 6.1 先起 server,再跑推理

```bash
rl-insight server start
# 观测开关 + URL(run_infer.sh 已默认 RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080)
export VERL_RL_INSIGHT_ENABLE=1
```

### 6.2 冒烟测试(小卡数、1 样本)

> **`--max-model-len` 是必填**(`parallel_infer.py` 里 `required=True`),`run_infer.sh` 默认不传它,
> 所以冒烟命令必须显式带——旧 README 的 `bash run_infer.sh /model` 直接跑会 argparse 报错。

```bash
# 2 卡、TP=1、1 样本。按你的卡改 ASCEND_RT_VISIBLE_DEVICES 和 --n-gpus-per-node
ASCEND_RT_VISIBLE_DEVICES=0,1 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
VERL_RL_INSIGHT_ENABLE=1 \
bash examples/kvc_aware_router/run_infer.sh /data1/models/Qwen3-4B \
    --device ascend --n-gpus-per-node 2 --tensor-parallel-size 1 \
    --max-model-len 8192 --max-samples 1 --kv-events
```

### 6.3 全量 8/16 卡 data-parallel

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
VERL_RL_INSIGHT_ENABLE=1 \
bash examples/kvc_aware_router/run_infer.sh /data1/models/Qwen3-4B \
    --device ascend --num-workers 8 --n-gpus-per-node 8 --tensor-parallel-size 1 \
    --max-num-seqs 64 --max-samples -1 \
    --max-model-len 39936 --response-length 8192 --kv-events
```

> **1-token 退化**:`max_tokens = min(response_length, prompt_length+response_length-prompt)`。
> `prompt_length = max_model_len - response_length - 100`(代码内派生)。多轮累积超 `max_model_len`
> 时 `max_tokens` 塌缩到 1。把 `max_model_len` 控制在模型原生上下文内(如 Qwen3-8B 40960 → 用 39936)。

### 6.4 看板 + ssh 隧道(远程访问)

浏览器开 `http://<server 机>:3000`(`admin`/`admin`,首次改密提示点 Skip)。server 在远程、
本地经跳板机访问用 `-J`:

```bash
ssh -J <跳板user>@<跳板host> -L 3000:localhost:3000 root@<server机>
# 浏览器 http://localhost:3000;要查 PromQL 原始值加 -L 9090:localhost:9090
```

---

## 7. 已知问题 / 排查

### Ascend 专属

| 现象 | 根因 | 解决 |
|------|------|------|
| vLLM worker 起不来 / 卡识别错 | CANN env 没 source / `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` 没设 | 每次进容器 `source .../set_env.sh`;设上 `RAY_EXPERIMENTAL_NOSET...=1` |
| `import vllm_ascend` 报 API 不匹配 | vllm 与 vllm-ascend 版本不自洽 | 两边版本号必须相等;对齐 torch_npu/CANN |
| 沙箱起不来 | 没预下载 wheels / 没拉 swebench 镜像 | 走完 §5.2 + §5.3 |

### 通用(transformers / numpy)

`requirements-npu.txt` 已钉 `numpy<2.0.0`,通常无需额外处理。旧环境下若仍报
`Backend should be defined in BACKENDS_MAPPING`,降 `pip install "transformers==4.57.6"`
(详见 [`examples/kvc_aware_router/README_cn.md`](../../examples/kvc_aware_router/README_cn.md) §6)。

### 观测:Grafana 显示 "No data"

数据只在 **router 跑起来后**才流动——server 单独只是存储+展示,hub `:9092` 是 router 起的。
"server 起了、grafana 空" 几乎都是 router 还没推数据。自上而下查(完整三步见
[`README_zh.md`](README_zh.md) *排查*):

1. router 在路由?`grep -E 'score\(\)|route\(\)' <router-log>`
2. hub 注册了?`curl -s localhost:9090/api/v1/targets | grep 9092`
3. hub 有业务 metric?`curl -s localhost:9092/metrics | grep kv_cache_load`

**首条数据有延迟**(prometheus `scrape_interval: 10s` + router 首 `score()` + hub 注册)约 30s–1min。

---

## 附:与 NVIDIA 流程的差异速查

| 维度 | NVIDIA([examples README](../../examples/kvc_aware_router/README_cn.md)) | Ascend(本文) |
|------|----------|---------|
| Docker 设备 | `--gpus all` | `--privileged` + 挂载 `/usr/local/Ascend/{driver,firmware}` |
| 卡可见 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` + `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` |
| 状态命令 | `nvidia-smi` | `npu-smi info` |
| requirements | `requirements.txt`(numpy≥2) | `requirements-npu.txt`(numpy<2 + triton-ascend) |
| 安装脚本 | (镜像自带) | `USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh` |
| 进程清理 | `nvidia-smi` 查 PID 杀 | `fuser -k /dev/davinci*` + `ray stop --force` |
| `--device` flag | `gpu`(默认) | `ascend`——但**仅在 `--enable-mooncake` 时**改 KV connector 类,不带 mooncake 时对 vLLM 配置无影响;ascend 真正生效靠环境 |
| gpu mem util | 0.8(代码恒定) | 0.8(代码恒定,与 device 无关) |

---

相关:[推理 README(中文)](../../examples/kvc_aware_router/README_cn.md) ｜ [观测 README(中文)](README_zh.md) ｜ [verl Ascend 教程总入口](../ascend_tutorial/README.md) ｜ [Ascend 安装权威版本矩阵](../ascend_tutorial/get_start/install_guidance.rst)
