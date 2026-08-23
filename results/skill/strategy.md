# Router 策略逻辑（当前代码）

> 依据当前目录 `verl/workers/rollout/router/` 的实现整理；不以历史实验文档或容器旧版本描述作为代码事实来源。

一次 LLM request 进入 router 后，策略分两步：先尝试 **shortcut + overload**，不能 shortcut 时再执行 **slow-cut**。

## 1. shortcut + overload

- 有历史绑定且原副本未过载：继续发给原副本。
- 是否过载取决于 `overload_mode`：
  - `kv_cache_usage_perc`：`kv_perc > sticky_overload_threshold`；
  - `kv_load`：加权 load（kv+inflight）`> sticky_overload_threshold`；
  - `skew`：相对池中位数的持续偏斜——`active_sessions > median + skew_delta`（绝对差）
    或 `running > 且 inflight_tokens > skew_factor × median`（比值），连续
    `skew_window` 次采样（默认 60，sticky-shortcut 节拍）才判过载，一次干净样本即清零。
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
  - **冷启动**：没有历史绑定时，`active_sessions` 最小 ± `first_bind_window`（默认 1）为候选，窗内按 `window+1−count` 加权随机（`first_bind_weighted=false` 则均匀）；
  - **正常情况**：先筛选：
    ```text
    avail >= cap × (1-crt)    # crt = capacity_reserve_threshold
    ```
    再从满足门槛的副本中选 `remaining` 最大的——top 集放宽为
    `remaining >= best − cap × tie_epsilon`（默认 0.01），集内均匀随机；
  - 阈值双语义：`sticky_overload_threshold`（overload 判据）与
    `capacity_reserve_threshold`（容量门槛）独立配置，均默认回落 `load_threshold`；
  - 如果没有副本满足门槛，则在所有副本中做同样的 near-top 随机。

## 代码位置

| 逻辑 | 文件 |
|---|---|
| 请求 acquire/release | `verl/workers/rollout/llm_server.py` |
| 副本选择与回调 | `verl/workers/rollout/router/kvcaware/balancer.py` |
| shortcut、三类 slow-cut | `verl/workers/rollout/router/kvcaware/strategies/kvc_aware.py` |
| score 排序 | `verl/workers/rollout/router/kvcaware/strategies/routing.py` |
