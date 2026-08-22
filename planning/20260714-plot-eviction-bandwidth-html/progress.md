# Progress: plot_eviction.py 两项优化

## 2026/07/14 — plan 创建

### 拆分
从 `20260707-kvcare-router-ascend910b3` 拆出两个 backlog 项独立 plan（用户提议）：
- 本 plan：plot_eviction.py 加带宽 panel + plotly HTML 输出
- 平行 plan：`20260714-mooncake-ascend-bringup`（mooncake 技术栈）
- **排除项**："C/D 改用其它 KV transfer 方案验证 mc 净负向"——用户明确不做，不进任何 plan。

### 起点状态
- plot_eviction.py 当前 9 面板（KV Load / usage / MFU / running / waiting / 累计驱逐 / gpu prefix hit / prefill / external-hit），HEAD=ca7e197。
- collector evidence 行现状：`... external=N flops=N [poll #K]`（flops 已通）。
- 910C A/B 真跑已通（用户），但本 plan 两项是增量，不影响 A/B 结论分析，随时可做。

### 确认的接入点（开工即用，不用再查）
- vLLM 暴露 `vllm:estimated_read/write_bytes_per_gpu_total`（Counter，`--enable-mfu-metrics` gate，同 flops 系列）。
- collector 三处 wiring 行号：`metric_spec.py:50/139`、`metrics.py:52`、`collector.py:28/39/214/224/228`（详见主线 plan §17 / perfetto plan）。

### 待办
- [ ] Phase 0：collector wiring read/write_bytes → evidence
- [ ] Phase 1：plot 带宽 panel（第 10 面）
- [ ] Phase 2：plotly HTML 输出（`--html`）
- [ ] Phase 3：910C 真数据验证（下次 A/B 跑时）
