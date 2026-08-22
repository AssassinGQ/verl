# Router 策略逻辑（当前代码）

> 依据当前目录 `verl/workers/rollout/router/` 的实现整理；不以历史实验文档或容器旧版本描述作为代码事实来源。

一次 LLM request 进入 router 后，策略分两步：先尝试 **shortcut + overload**，不能 shortcut 时再执行 **slow-cut**。

## 1. shortcut + overload

- 有历史绑定且原副本未过载：继续发给原副本。
- 是否过载取决于配置，例如：
  ```text
  kv_cache_usage_perc > load_threshold
  ```
- 原副本过载或没有绑定：进入 slow-cut。

## 2. slow-cut

- `least-inflight`：
  ```text
  选 active_sessions 最小的副本（首绑感知）
  ```
  `active_sessions` = 该副本的存活 sticky binding 数（session 首绑 +1、
  session finalize/abort −1）。session 在工具/sandbox 阶段 `inflight_count`
  瞬时归零，但它保持抬升——首绑不再被"工具间隙的空台"吸走。

- `prefix-load-aware`：
  ```text
  score = α × cache_hit + (1-α) × (1-load)
  ```

- `capacity-token-aware`：
  ```text
  avail     = cap × (1-kv_usage)
  remaining = avail - prompt_len × (1-prefix_hit)
  ```
  - **冷启动**：没有历史绑定时，选 `active_sessions` 最小的副本；
  - **正常情况**：先筛选：
    ```text
    avail >= cap × (1-lt)
    ```
    再从满足门槛的副本中选 `remaining` 最大的；
  - `lt=0.9` 表示副本至少还剩 **10% KV 容量**时才优先参与选择；
  - 如果没有副本满足门槛，则在所有副本中选 `remaining` 最大的。

## 代码位置

| 逻辑 | 文件 |
|---|---|
| 请求 acquire/release | `verl/workers/rollout/llm_server.py` |
| 副本选择与回调 | `verl/workers/rollout/router/kvcaware/balancer.py` |
| shortcut、三类 slow-cut | `verl/workers/rollout/router/kvcaware/strategies/kvc_aware.py` |
| score 排序 | `verl/workers/rollout/router/kvcaware/strategies/routing.py` |
