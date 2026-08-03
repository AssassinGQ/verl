# KV-Cache-Aware Router 在线观测

Last updated: 08/03/2026.

kvc-aware router 把路由层内部信号——KV 占用、dispatch/completed 计数、route
延迟、score 分量——实时推到 [rl-insight](https://github.com/verl-project/rl-insight)
（Prometheus → Grafana）。用于在 rollout 运行时查看前缀缓存命中率、副本负载倾斜、
路由决策开销。

> env 门控 + 懒加载：开关关闭（默认）时 router 几乎零开销——每条 emit 路径在
> `VERL_RL_INSIGHT_ENABLE` 上短路。

## 前置：运行中的 rl-insight server

观测依赖一个运行中的 rl-insight server（Prometheus + Grafana + api）。本 router
推送的 dashboard + histogram 自定义桶透传在 `feat/verl-agentic-rollout-dashboard`
分支——**必须用这个分支，不是 PyPI 默认版**（默认版缺 verl dashboard 和桶透传，
router 推的自定义桶落不进 histogram、grafana 看不到面板）。

### 标准安装（GitHub + dl.grafana.com 可达）

```bash
git clone https://github.com/touch869/rl-insight.git
cd rl-insight && git checkout feat/verl-agentic-rollout-dashboard
pip install -e .
rl-insight server install     # 一次性：下载 Prometheus/Grafana/Tempo
rl-insight server start       # Prometheus :9090、Grafana :3000、api :18080
```

rl-insight 的通用用法（训练 metric、TransferQueue、trace）见
[`docs/advance/rl_insight.md`](../advance/rl_insight.md)。

### 受限环境（GitHub 被墙 / 代理 MITM 导致 TLS 失败）

三道坎，逐个破：

- **clone 失败** → git bundle：可达机 `git bundle create rl-insight.bundle --all`，
  scp 过去，`git clone rl-insight.bundle`。
- **`pip install` 被墙（PyPI 不可达）** → `pip install -e . --no-build-isolation`
  （用本机自带 setuptools）。
- **`server install` TLS/超时（github release + dl.grafana.com）** → 在别处下好包，
  放到目录，指向它。`--local-archive` 不扫描目录——rl-insight 仍需要版本号才能拼
  文件名（`prometheus-{版本}.linux-amd64.tar.gz`）；版本来自 `install_version` 或走
  https 查 `latest`（证书失败的环境同样不通）。所以用**你自建的 config** 固定版本
  （rl-insight 不自带 config，也没有 `--version` CLI 参数）：
  ```bash
  mkdir -p /data1/rl-archives
  #   prometheus-2.54.1.linux-amd64.tar.gz   ← 文件名版本必须和 install_version 一致
  #   grafana-13.0.0.linux-amd64.tar.gz
  cat > /data1/rl-archives/install-config.yaml <<'EOF'
  prometheus: { enable: true, install_version: "2.54.1" }
  grafana:    { enable: true, install_version: "13.0.0" }
  tempo:      { enable: false }
  EOF
  rl-insight server install --local-archive /data1/rl-archives \
      --config /data1/rl-archives/install-config.yaml
  ```
  替代：修通 CA（`pip install -U certifi` / `update-ca-certificates`，或装 MITM 代理
  的根证书）让 `latest` 能查——就不用 config 了。

## 启用

rl-insight 是 env 门控——没有专用 flag，启动前 export 让每个 Ray 起的
router/vLLM worker 继承。**先起 server**，再跑推理：

```bash
rl-insight server start
VERL_RL_INSIGHT_ENABLE=1 bash examples/kvc_aware_router/run_infer.sh /data/models/Qwen3-4B --kv-events
```

`run_infer.sh` 已默认 export `RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080`，所以只需
`VERL_RL_INSIGHT_ENABLE=1`。只有 server 在别的机子/端口才显式设 `RL_INSIGHT_SERVER_URL`。
（`ascend-exps.sh` 更进一步——自动起/停 server，且默认 `VERL_RL_INSIGHT_ENABLE=1`。）

| env 变量 | 默认 | 用途 |
|---------|------|------|
| `VERL_RL_INSIGHT_ENABLE` | 未设（关） | 设 `1` 开启 kvc-aware emitter。和 trainer 用同一个开关（`RLInsightLogger`），一个 flag 同时点亮 trainer + router。 |
| `RL_INSIGHT_SERVER_URL` | `run_infer.sh` 默认 `http://127.0.0.1:18080`；rl-insight 本身必填（不设 → init 打 `"server URL is required"` 并静默关闭）。只有 server 在别处才覆盖。`127.0.0.1` 也避开共享盒子的死代理，见*坑*。 |

## 观测信号（20 个 primitive）

emitter 在两个挂载点扇出；合约（名/类型/label/桶）单一事实源在
[`verl/workers/rollout/router/kvcaware/types/emit_spec.py`](../../verl/workers/rollout/router/kvcaware/types/emit_spec.py)。

- **B 类——store 写入（14）：**
  - *poll*（vLLM `/metrics` 快照）→ KV 占用、running、waiting、累计 prompt/cached/external token、flops
  - *acquire* → `dispatched_count`、`prompt_len_sum`、`inflight_tokens`、`inflight_avg_turn`
  - *release* → `completed_count`、`inflight_tokens`、`inflight_avg_turn`
  - *kv-removed* → `kv_evictions`
- **A 类——策略 `score()`（6）：** `load` / `s_cache`（prefix-load-aware）、
  `avail` / `need` / `remaining`（capacity-token）、`route_latency_seconds`

类型：gauge（瞬时水位 + vLLM 累计量原样转发）、counter（dispatched/completed/prompt_len_sum/kv_evictions）、histogram（5 个 score 分量 + route latency）。

## 触发链路

```
router score() / acquire / release / poll
  → emitter 单例  (VERL_RL_INSIGHT_ENABLE=1)
  → rl_insight hub actor  (:9092)
  → Prometheus scrape     (10s)
  → Grafana "agentic rollout" dashboard
```

## Dashboard

rl-insight 自带 `verl_agentic_rollout` dashboard（24 面板）；server 启动时自动
provision。浏览器开 `http://<server 机>:3000`，登录（`admin`/`admin`；Grafana
首次登录的改密提示可点 Skip）。

**远程看板**：在能 ssh 到 server 机的本地，转发 Grafana 端口（经跳板机用 `-J`）：
```bash
ssh -J <跳板user>@<跳板host> -L 3000:localhost:3000 root@<server机>
# 浏览器 http://localhost:3000
```
`-L` 里的 `localhost` 是在 **server 机**上解析的，所以转的就是 server 机上的 3000。
只看 dashboard 转发 3000；要写 PromQL 查原始值加 `-L 9090:localhost:9090`。

## 共享 GPU 盒子的坑

1. **死代理**——盒子的 `http_proxy` 不通时 hub 注册失败（Connection refused）。用
   `RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080`（`127.0.0.1` 在 no_proxy 放行，直连）。
2. **`:9092` 残留 hub**——上一轮的 hub actor 在 ray 退出后仍占 :9092，下一轮 router
   hub bind 失败（`OSError: Address already in use`）→ grafana 无数据。重启前清：
   ```bash
   ray stop --force
   for pid in $(ss -tlnP | grep ':9092' | grep -oP 'pid=\K[0-9]+'); do kill -9 "$pid"; done
   ```
3. **server 先于 router**——router hub 在启动时注册；server 没起，注册静默失败。
   永远先 `rl-insight server start`，再跑推理。
4. **本地 Grafana 占 `:3000`**——ssh 隧道 `-L 3000:localhost:3000` 会落到本地
   rl-insight grafana（无远端数据）如果本地也跑着。先停本地：`rl-insight server stop`，
   或换本地端口 `-L 3001:localhost:3000`，浏览器开 `:3001`。

## 排查：grafana 显示 "No data"

数据只在 **router 跑起来后** 才流动——server 单独只是存储 + 展示，hub `:9092`
是 router 起的（不是 server）。所以"server 起了、grafana 空"几乎都是 router 还没
推数据。自上而下查：

1. **router 在跑 + 在路由？**
   `grep -E 'score\(\)|route\(\)' <router-log>`——无输出 = router 还没路由（还在加载，
   或没发推理请求）。
2. **hub 注册到 prometheus 了？**
   `curl -s localhost:9090/api/v1/targets | grep 9092`——没有 `:9092` target = router
   没起 hub。通常是 `VERL_RL_INSIGHT_ENABLE=1` 没进 ray worker（旧 ray session 带的是
   旧 env）→ `ray stop --force` 后重跑。
3. **hub 有业务 metric？**
   `curl -s localhost:9092/metrics | grep kv_cache_load`——有 `process_*` 但无
   `kv_cache_load` = emitter 没推（env 没进 worker）；完全空 = hub 起了但还没注册
   （router 还没跑到 `score()`）。

**首次数据有延迟**：prometheus `scrape_interval: 10s` + router 出第一个 `score()` +
hub 注册——router 起来后约 30s–1min grafana 才出第一条曲线。刚启动别急着狂刷新。

---

英文版：[README.md](README.md)。
