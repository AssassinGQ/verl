# AgentFramework：exp1 session 展开与完成判据

> 适用范围：`results/result-0820/` 的 exp1（`concurrency=128`，64 samples × `n=8`）；§2.4 为当前代码（P1 后）的行为。
>
> 目的：说明 `prompt64x8` 的实际执行模型，以及为何日志出现 `=> Resolved` 时 router 侧仍可能有未归零的 inflight。

---

## 1. `prompt64x8` 的含义：64 条数据在 AgentFramework 内部展开为 512 个 session

实验脚本传入：

```bash
--max-samples 64 --n 8
```

`parallel_infer.py` 读取 64 条数据，构建的 `prompts` batch 本身只有 **64 条 sample**。`--n 8` 被写入：

```python
ro.n = n  # actor_rollout_ref.rollout.n = 8
```

实际的 `×8` 不在 `parallel_infer.py` 中提前 repeat prompt，而是在 `../uni-agent` 的 `OpenAICompatibleAgentFramework` 内部展开：

```python
num_sessions = int(self._rollout_config.get("n"))
```

随后对每个 sample 创建 `num_sessions` 个独立 session：

```python
tasks = [
    self._run_session_with_concurrency_limit(
        sample_fields=sample_fields,
        sample_index=sample_index,
        session_index=session_index,
    )
    for session_index in range(num_sessions)
]
outcomes = await asyncio.gather(*tasks, return_exceptions=True)
```

因此 exp1 的实际目标规模为：

```text
64 samples × 8 sessions / sample = 512 agent sessions
```

session 数受 `max_concurrent_sessions=128` 的 semaphore 限制；512 个 session 被分批运行，而不是同时运行 512 个。

关键代码位置：

- `examples/kvc_aware_router/parallel_infer.py`：`ro.n = n`
- `../uni-agent/uni_agent/framework/framework.py`：`generate_sequences()`、`_run_batch_to_tq()`、`_run_prompt_sessions_to_tq()`

---

## 2. `=> Resolved` 的判据与 router inflight 不同

### 2.1 AgentFramework 实际等待的内容

主 driver 在判断 `captured_scores` 前，先阻塞于：

```python
asyncio.run(framework.generate_sequences(prompts))
```

AgentFramework 的正常完成链路是：

```text
64 prompts
  → 每条 prompt 的 8 个 session
    → agent runner（Ray task）
    → gateway.finalize_session()
    → RewardLoopWorker.compute_score()
    → 写入 TQ 的 trajectory / rm_scores
```

它使用 `asyncio.gather(..., return_exceptions=True)` 聚合 64 个 prompt 和每 prompt 的 8 个 session。普通 session 异常会被标为失败，但不会中断整个 batch；只有所有 session 都没有成功输出时，框架才抛出 `RuntimeError("All rollouts failed ...")`。

### 2.2 `Resolved` 实际表示什么

`parallel_infer.py` monkeypatch 当前 driver 进程的 TransferQueue 写入接口。任一成功 trajectory 在写入 `rm_scores` 时，会把最终 score 放入内存字典 `captured_scores`。

在 `framework.generate_sequences()` 返回后，driver 汇总当前的分数：

```python
resolved = sum(1 for s in per_sample_scores if s > 0)

if captured_scores:
    print(f"=> Resolved {resolved}/{total} samples ...")
```

所以：

```text
=> Resolved X/64
```

表示：

- 512 个 session 已完成成功/失败的框架级收敛；
- 至少有一条成功 trajectory 写入了 `rm_scores`；
- 64 个原始 sample 中，X 个 sample 的已捕获 score 聚合后大于 0。

它**不表示**：

- 512 个 session 都成功；
- 64 个 sample 都完成；
- 所有 sandbox 正常退出；
- 所有 LLM 请求都已在 router 完成 release；
- `dispatched_count == completed_count`；
- 12/12 replica 的 router inflight 均为 0。

### 2.3 为什么 Resolved 后仍可能有 router inflight

router inflight 的定义是：

```text
inflight = dispatched_count - completed_count
```

一次 LLM 请求 acquire 时增加 `dispatched` / inflight；release 时增加 `completed` / 减少 inflight。

但 release 是 fire-and-forget 的 Ray RPC：

```python
self._load_balancer.release_server.remote(...)
```

调用方不 `await` LB actor 实际处理该 RPC。因此存在两个独立的收敛点：

```text
A. AgentFramework：session 成功或失败、trajectory 已 finalize / score 已写入
B. Router：所有异步 release_server RPC 已被 LB actor 执行并记入 completed_count
```

exp1 等待 A，再基于 score 输出 `Resolved`；它没有在输出前等待 B 的 drain barrier。OOM、sandbox/instance 异常或 shutdown 时序会进一步放大 B 未落账的概率。

故应将 `Resolved` 视为**部分/全部 session 已完成评分后的结果汇总哨兵**，不要把它用作 router inflight 已收敛的判据。

### 2.4 session 终态如何到达 router（P1 桥接，当前代码）

exp1/exp2 时代的代码里 router 完全看不到 session 终态（binding 只增不减）。当前代码补上了这条链：

```text
GatewayManager.finalize_session / abort_session   （driver 进程）
  → _fire_session_end(session_id)                  通用回调，默认空
  → bridge_session_end_to_router（framework/entry.py 注册）
      → router.on_session_end.remote(session_id)   Ray fire-and-forget（与 release_server 同模式）
          → balancer：store.invalidate_sticky_binding(session_id)（幂等）+ _fire("on_session_end")
              → binding 失效 → ACTIVE_SESSIONS(replica) −1
```

要点：

- **request_id 就是 session_id**（gateway session 以 session_id 作为 `backend.generate` 的 request_id），所以 sticky binding 表天然就是 session→replica 映射，桥接只需传 id。
- **幂等**：abort-after-finalize 重发 → invalidate 对已失效 binding no-op；rebind（overload fallback）→ 表内替换自动 −旧+新。
- **降级**：uni-agent 侧无桥接（版本不对齐 / 非 uni-agent 部署）时 binding 不失效，`active_sessions` 退化为累计首绑数——仍是合法首绑信号，但失去"避开挂死 session 所在副本"的存活语义。
- 时序安全：session 非 `ACTIVE` 时 gateway 拒绝新请求（409），session_end 之后不可能再有该 session 的 acquire。

---

## 3. 可用于后续分析的判读

| 信号 | 表示什么 | 不能表示什么 |
|---|---|---|
| `=> Resolved X/64` | 已有 score 被捕获，X 个原始 sample 的已捕获聚合 score > 0 | 512 session 全成功、router 已清零 |
| `router-dispatch: dispatched - completed` | router release 账本尚未落账的 LLM request 数 | 尚未完成的完整 agent session 数、GPU 当前运行请求数 |
| `vllm-metrics: num_requests_running` | vLLM 此刻运行中的请求数 | agent/sandbox 的完整生命周期 |

