# Plan: plot_eviction.py 两项优化（带宽 panel + plotly 交互 HTML）

> 起：2026-07-14。从 `20260707-kvcare-router-ascend910b3` 拆出来——主线 A/B 已跑通，监控可视化工具链（plot 9 面板 + perfetto）也已闭环（见 `../workhis/planning-history/20260714-plot-perfetto-toolchain/`）。本 plan 只收**两个 backlog 项**，都是 plot_eviction.py 的增量优化，不阻塞任何主线。

## Goal

给 `examples/llm_router/plot_eviction.py` 加两个能力，补 PNG 的两个短板：

1. **带宽利用率面板**：decode 阶段是带宽受限的，`read/write_bytes ÷ 3.2TB/s` 比 MFU 更能反映"忙不忙"；且带宽分母（HBM 3.2TB/s）**无 800I/800T 歧义**（MFU 有 560 vs 750 的 I/T 区分问题）。vLLM 已暴露 `vllm:estimated_read_bytes_per_gpu_total` / `vllm:estimated_write_bytes_per_gpu_total`（Counter，同 flops 系列，gate 在 `--enable-mfu-metrics`）。
2. **plotly 交互式 HTML 输出**：弥补 PNG 不能缩放/悬停的痛点。Perfetto 试过因同色/同轴冲突出局（D2 结论），plotly 是更轻的替代——`plotly.graph_objects` 多 subplot sharex，hover 显示 replica+时刻+数值，导出单 HTML 文件，浏览器打开即用。

## 数据流（复用现有，不新开采集）

```
vllm --enable-mfu-metrics
  → /metrics 暴露 vllm:estimated_{flops,read_bytes,write_bytes}_per_gpu_total (Counter)
  → collector scrape (metrics.py _PROMETHEUS_MAP 加 read/write_bytes 映射)
  → _CUMULATIVE_KEYS 加 read/write_bytes（算 delta）
  → evidence 行加 read= / write= 字段（照 flops= 模式）
  → plot_eviction.py parse_signal_line EVIDENCE_PAT 加 read= / write= 捕获组
  → 新 panel: bandwidth utilization = (read+write bytes/s) / 3.2e12
  → --peak-hbm-tb 配置分母（默认 3.2，800I/800T 同值）
```

## Phases

| # | 阶段 | 状态 | 产出 |
|---|---|---|---|
| 0 | collector wiring：read/write_bytes 进 evidence | ⏳ pending | `metric_spec.py` 加 `ESTIMATED_READ/WRITE_BYTES_PER_GPU` + METRIC_SPECS；`metrics.py` `_PROMETHEUS_MAP` 加映射；`collector.py` `_CUMULATIVE_KEYS` 加两项 + evidence 行加 `read=` `write=` |
| 1 | plot 带宽 panel | ⏳ pending | `plot_eviction.py` EVIDENCE_PAT 加 `read=` `write=` 捕获组；`_bw_series(bytes_pts, window_s, peak_bytes)` 照 `_mfu_series` 写（滑窗 delta/duration/peak）；新增第 10 面板 bandwidth util（实线 1min 实时 / 虚线总平均，同 MFU 模式）；`--peak-hbm-tb`（默认 3.2） |
| 2 | plotly HTML 输出 | ⏳ pending | `--html out.html` flag；`_render_plotly(signals, out)` 用 `make_subplots(rows=10, shared_xaxes=True)` + per-replica trace + hovertemplate；与 PNG 路径互斥可选（`--out .png` 保留）；不引入 CJK 字体依赖（plotly 浏览器渲染，无此问题） |
| 3 | 910C 真数据验证 | ⏳ pending | 用户下次 910C 跑 A/B 时，新 log 直接 `python plot_eviction.py G.log --html G.html`，确认带宽 panel 有值、HTML 能打开悬停 |

## Decision log

- **2026/07/14 — D0 拆 plan**：这两项是 plot_eviction 增量，从主线拆出独立 plan 聚焦（用户提议）。不阻塞 A/B 结论分析。
- **2026/07/14 — D1 带宽分母用 HBM 规格 3.2TB/s，不用实测**：同 MFU 决策（D-MFU）——benchmark 测的是可达带宽，拿它当分母算的是带宽效率比，不是利用率。800I/800T 都是 3.2TB/s HBM，无 I/T 歧义，正好补 MFU 的分母短板。
- **2026/07/14 — D2 plotly 选 plotly.graph_objects 不用 dash/server**：要单 HTML 文件离线打开，`fig.write_html(..., include_plotlyjs=True)` 内嵌 JS，不要 server。轻、可邮件传、910C 容器外浏览器直接看。
- **2026/07/14 — D3 不替换 PNG，只加 `--html` 选项**：PNG 是日常主力（D-PNG 主力），plotly 是"要 zoom/悬停细看"时的补充，两者并存，`--out` 扩展名决定输出。
- **2026/07/14 — D4 read+write 合算一路带宽**：decode 阶段 read 远多于 write（KV 读），但合算"总带宽利用率"更直观；如需分拆后续加 `--bw-split`，初版合一。
