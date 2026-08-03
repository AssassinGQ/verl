# KV-Cache-Aware Router Online Observability

Last updated: 08/03/2026.

The kvc-aware router pushes its internal signals — KV occupancy, dispatch/completed
counts, route latency, score components — live to [rl-insight](https://github.com/verl-project/rl-insight)
(Prometheus → Grafana). Use it to see prefix-cache hit rates, per-replica load
skew, and routing-decision cost as a rollout runs.

> Env-gated and lazily wired: with the switch off (the default) the router pays
> ~zero cost — every emit path short-circuits on `VERL_RL_INSIGHT_ENABLE`.

## Prerequisite: a running rl-insight server

Observability needs an rl-insight server (Prometheus + Grafana + api) already up.
The dashboard + histogram-bucket passthrough this router emits live in the
`feat/verl-agentic-rollout-dashboard` branch — **use that, not the default PyPI build**.

### Standard install (GitHub + dl.grafana.com reachable)

```bash
git clone https://github.com/touch869/rl-insight.git
cd rl-insight && git checkout feat/verl-agentic-rollout-dashboard
pip install -e .
rl-insight server install     # one-time: fetches Prometheus/Grafana/Tempo
rl-insight server start       # Prometheus :9090, Grafana :3000, api :18080
```

General rl-insight usage (trainer metrics, TransferQueue, traces) is in
[`docs/advance/rl_insight.md`](../advance/rl_insight.md).

### Restricted environment (GitHub blocked / proxy MITM breaking TLS)

Three obstacles, each with a workaround:

- **clone fails** → git bundle: on a reachable machine `git bundle create rl-insight.bundle --all`, scp it over, `git clone rl-insight.bundle`.
- **`pip install` blocked (PyPI unreachable)** → `pip install -e . --no-build-isolation` (uses the box's own setuptools).
- **`server install` TLS/timeout (github releases + dl.grafana.com)** → download the
  archives elsewhere, drop them in a dir, point install at it. `--local-archive` does
  not scan the dir — rl-insight still needs a version to build the filename
  (`prometheus-{ver}.linux-amd64.tar.gz`); it gets that from `install_version` or by
  querying `latest` over https (which is also broken here). So pin the version via a
  **config you create** (rl-insight ships none, and has no `--version` CLI flag):
  ```bash
  mkdir -p /data1/rl-archives
  #   prometheus-2.54.1.linux-amd64.tar.gz   ← filename version must match install_version
  #   grafana-13.0.0.linux-amd64.tar.gz
  cat > /data1/rl-archives/install-config.yaml <<'EOF'
  prometheus: { enable: true, install_version: "2.54.1" }
  grafana:    { enable: true, install_version: "13.0.0" }
  tempo:      { enable: false }
  EOF
  rl-insight server install --local-archive /data1/rl-archives \
      --config /data1/rl-archives/install-config.yaml
  ```
  Alternative: fix the CA (`pip install -U certifi` / `update-ca-certificates`, or
  install a MITM proxy's root) so `latest` resolves — then no config needed.

## Turn it on

rl-insight is env-gated — no special flag, just export the env before launching so
every Ray-spawned router/vLLM worker inherits it. Start the server **first**, then
run inference:

```bash
rl-insight server start
VERL_RL_INSIGHT_ENABLE=1 bash examples/kvc_aware_router/run_infer.sh /data/models/Qwen3-4B --kv-events
```

`run_infer.sh` already exports `RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080` by default,
so you only need `VERL_RL_INSIGHT_ENABLE=1`. Set `RL_INSIGHT_SERVER_URL` explicitly only
if the server is on another host/port. (`ascend-exps.sh` goes further — it starts/stops
the server itself and defaults `VERL_RL_INSIGHT_ENABLE=1`.)

| env var | default | purpose |
|---------|---------|---------|
| `VERL_RL_INSIGHT_ENABLE` | unset (off) | Set to `1` to turn on the kvc-aware emitter. Same switch the trainer uses (`RLInsightLogger`), so one flag lights up trainer + router together. |
| `RL_INSIGHT_SERVER_URL` | `run_infer.sh` defaults `http://127.0.0.1:18080`; rl-insight itself requires it (unset → init logs `"server URL is required"` and stays off). Override only if the server is elsewhere. `127.0.0.1` also sidesteps dead-proxy pitfalls — see *gotchas*. |

## What you see (20 primitives)

The emitter fans out at two mount points; the contract (names/types/labels/buckets)
is the single source of truth in
[`verl/workers/rollout/router/kvcaware/types/emit_spec.py`](../../verl/workers/rollout/router/kvcaware/types/emit_spec.py).

- **B-class — store writes (14):**
  - *poll* (vLLM `/metrics` snapshot) → KV usage, running, waiting, cumulative prompt/cached/external tokens, flops
  - *acquire* → `dispatched_count`, `prompt_len_sum`, `inflight_tokens`, `inflight_avg_turn`
  - *release* → `completed_count`, `inflight_tokens`, `inflight_avg_turn`
  - *kv-removed* → `kv_evictions`
- **A-class — strategy `score()` (6):** `load` / `s_cache` (prefix-load-aware),
  `avail` / `need` / `remaining` (capacity-token), `route_latency_seconds`

Types: gauges (instantaneous levels + vLLM cumulative counters forwarded as-is),
counters (dispatched/completed/prompt_len_sum/kv_evictions), histograms (the five
score components + route latency).

## Emit path

```
router score() / acquire / release / poll
  → emitter singleton  (VERL_RL_INSIGHT_ENABLE=1)
  → rl_insight hub actor  (:9092)
  → Prometheus scrape     (10 s)
  → Grafana "agentic rollout" dashboard
```

## Dashboard

rl-insight ships a `verl_agentic_rollout` dashboard (24 panels); the server
auto-provisions it on start. Open `http://<server-host>:3000` and log in
(`admin`/`admin`; Grafana's first-login password-change prompt can be skipped).

## Gotchas (shared GPU boxes)

1. **Dead proxy** — if the box's `http_proxy` is unreachable, hub registration
   fails with Connection refused. Use `RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080`
   (`127.0.0.1` is in `no_proxy`, so it connects directly).
2. **Stale hub on `:9092`** — a hub actor from a previous run keeps `:9092`
   listening after Ray exits, so the next router's hub fails to bind
   (`OSError: Address already in use`) → Grafana shows no data. Clear before
   relaunch:
   ```bash
   ray stop --force
   for pid in $(ss -tlnP | grep ':9092' | grep -oP 'pid=\K[0-9]+'); do kill -9 "$pid"; done
   ```
3. **Server before router** — the router hub registers on startup; if the
   rl-insight server isn't up yet, registration silently fails. Always
   `rl-insight server start` first.
4. **Local Grafana hogging `:3000`** — an ssh tunnel `-L 3000:localhost:3000`
   falls back to a local rl-insight Grafana (no remote data) if one is running.
   Stop the local stack first: `rl-insight server stop`.

## Troubleshooting: grafana shows "No data"

Data only flows once the **router** is running — the server alone is storage +
display, and the hub `:9092` is started by the router (not the server). So
"server started, grafana empty" almost always means the router isn't pushing yet.
Check top-down:

1. **Router running + routing?**
   `grep -E 'score\(\)|route\(\)' <router-log>` — no output = router hasn't routed
   yet (still loading, or no inference requests sent).
2. **Hub registered to prometheus?**
   `curl -s localhost:9090/api/v1/targets | grep 9092` — no `:9092` target = router
   didn't start the hub. Usually `VERL_RL_INSIGHT_ENABLE=1` didn't reach the ray
   worker (a stale ray session carries the old env) → `ray stop --force` and rerun.
3. **Hub has business metrics?**
   `curl -s localhost:9092/metrics | grep kv_cache_load` — has `process_*` but no
   `kv_cache_load` = emitter not pushing (env didn't reach worker); completely empty
   = hub up but nothing registered yet (router hasn't hit `score()`).

**First data has latency**: prometheus `scrape_interval: 10s` + the router's first
`score()` + hub registration — expect ~30s–1min after the router starts before
grafana shows the first curve. Don't refresh frantically right after launch.
