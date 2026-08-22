# Findings — n=1 消融

## 假设 H4
当前矩阵 n=8（每 prompt 8 sample），kvcare 聚合同 prompt → 跨 sample 共享 KV（M1）。
n=1：无跨 sample 共享。
- B−A 消失 → M1 是驱动
- B−A 仍在 → 跨 prompt 公共前缀也贡献

## 配置
--n 1 --max-samples 128（128 不同 prompt）
