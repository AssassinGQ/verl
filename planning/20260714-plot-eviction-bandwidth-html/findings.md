# Findings: plot_eviction.py 两项优化

> 本 plan 的研究发现。每个发现一条，按时间顺序。

## §1 带宽 metric 在 vLLM 的暴露（确认存在，复用 flops wiring）

vLLM `vllm/v1/metrics/perf.py` 在 `--enable-mfu-metrics` gate 下暴露三个 Counter：
- `vllm:estimated_flops_per_gpu_total`（已用，commit cb5fc75）
- `vllm:estimated_read_bytes_per_gpu_total`（本 plan 要用）
- `vllm:estimated_write_bytes_per_gpu_total`（本 plan 要用）

三者同系列：都是 Counter（单调递增累计量）、per-GPU（已 ÷ TP/PP）、由 vLLM perf 估算器在每次 forward 更新。**采集链路与 flops 完全同构**——只需在 collector 的三处（`metric_spec.py` 加 MetricKey + METRIC_SPECS、`metrics.py` `_PROMETHEUS_MAP` 加映射、`collector.py` `_CUMULATIVE_KEYS` 加项 + evidence 行加字段）照抄 flops 的写法各加两条即可。evidence 行最终形如：
```
... external=N flops=N read=N write=N [poll #K]
```

## §2 带宽分母：HBM 3.2TB/s，无 I/T 歧义（补 MFU 短板）

- Atlas 800I A3 / 800T A3 同 A3 die，HBM 都是 **3.2 TB/s**（80GB HBM-2e）。
- MFU 分母有 I/T 歧义（800I=560 / 800T=750 TFLOPS FP16），device id `0xD803` 分不出（见 perfetto plan D0/D-it）。
- 带宽分母无此问题——**带宽 panel 正好补 MFU 的分母短板**，decode 阶段（带宽受限）更能反映"卡忙不忙"。
- 默认 `--peak-hbm-tb 3.2`，可配（未来若上 HBM-3e 机型改）。

## §3 MFU 滑窗 overcount bug 的教训（带宽 panel 必须照修）

`_mfu_series` 原版滑窗 bug（cb5fc75 修复）：滑窗 60s 内每个 evidence 点的 `d_flops` 是一个 ~30s evidence-window 的 delta，60s 窗里有 2-3 个点 → 直接 sum 等于把 1.5x 的量算进去。

**带宽 panel 必须照同样方式修**：`_bw_series` 滑窗算 `real_bw = (bytes_sum / dur_sum) / peak_bytes`，其中 `dur` 用 evidence-window 的 duration（`t1 - t_prev`），不是 wall-clock 60s 直接除。总平均 `total_bw = (cum_bytes / elapsed) / peak_bytes`。

## §4 plotly 比 Perfetto 更适合本场景（D2 结论的延伸）

Perfetto 试过出局（perfetto plan D2）：proto 无 color 字段，UI 按父 process 组配色 → 同父组同色；而共享 Y 轴（`y_axis_share_key`）要求同父组 → "共享轴"和"按 replica 配色"在 Perfetto 里**互斥**，16 个 replica 无法既同轴又分色。

**plotly 无此限制**：
- `make_subplots(rows=N, shared_xaxes=True)` 每行一个 panel，panel 内多条 trace（per-replica）天然共享 Y 轴 + 各自配色（plotly 默认 tab10，可配）。
- hover 显示 replica + 时刻 + 数值，正好补 PNG 静态短板。
- `fig.write_html(out, include_plotlyjs=True)` 单 HTML 内嵌 JS，离线浏览器打开，不依赖 server。
- 910C 容器外（用户本地浏览器）直接看，不需要容器装 plotly（plot 只读 log，可本地跑）。

## §5 现有 plot_eviction.py 接入点（已确认，等开工）

```
plot_eviction.py:
  EVIDENCE_PAT (re):      加 read= / write= 捕获组（照 flops= 模式，可选组）
  parse_signal_line():    无需改（regex 更新即可）
  _mfu_series():          照抄写 _bw_series
  main() 信号 buffer:     加 bandwidth buffer
  _draw() / 子图:          加第 10 个 subplot
  argparse:               加 --peak-hbm-tb / --html
```
`metric_spec.py` / `metrics.py` / `collector.py` 的 flops wiring 行号已在主线 plan §17 记录，照抄。
