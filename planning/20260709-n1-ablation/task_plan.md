# Task Plan — n=1 消融实验

## 目标

隔离"同 prompt 跨 sample 共享"是否为 KVCAware 的 B−A 驱动。
n=1（每 prompt 只采 1 sample）→ 无跨 sample 的 prefix 共享。
若 B−A 优势消失 → 坐实 M1（同 prompt 跨 sample 聚合）是驱动；若还在 → 跨 prompt 公共前缀。

## 前置依赖

- standalone collector 完成（A 组用）

## 配置

`--n 1 --max-samples 128`（128 个不同 prompt，无跨 sample 共享）
其余同 exp4：α=0.3, load_threshold=0.6, TP=2, 3 replica, simulated, Qwen3-8B

---

## Phase 1 — A vs B @ n=1 [ ]
- A sticky/nomc n=1, B kvcare α0.3/nomc n=1, 各 128 samples
- 判定：B−A walltime / prefix-hit 差异

## Phase 2 — 对比 n=8（exp4）[ ]
- n=1 B−A vs n=8 B−A（exp4: B−A −23.7%）
- 若 n=1 B−A≈0：M1 是驱动（跨 sample 聚合）
- 若 n=1 B−A 仍显著：M1 非唯一驱动（跨 prompt 公共前缀也贡献）

## Phase 3 — 报告 [ ]
因果结论。

## 进度日志
见 progress.md。
