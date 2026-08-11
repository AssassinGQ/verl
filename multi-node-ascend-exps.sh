#!/usr/bin/env bash
# Multi-node Ascend experiment driver.
#
# Topology: 6 nodes (this host = Ray head, + 5 passwordless-SSH workers).
# Every run_infer in the ascend-exps matrix runs as ONE distributed Ray job
# spanning all 6 nodes: 6 * 8 NPU = 48 cards, TP=4 → 12 replicas across nodes.
# The matrix iterates sequentially; each experiment uses all 6 machines.
#
# Worker access model: ssh to the worker HOST, then `docker exec` into the
# WORKER_CONTAINER (named hgq-verl-ascend) where REPO lives. The head (this
# host) is assumed to already be inside the same container — its commands run
# directly. Containers must use --network host so ray/rl-insight ports are
# reachable across hosts via the host IP.
#
# Flow: ssh-check → ray cluster up (head + workers) → rl-insight on head →
#       ascend-exps matrix loop (run_infer.sh --nnodes 6 each time).
set -euo pipefail

# =====================================================================
# Configuration
# =====================================================================
WORKERS=(
    "root@10.22.22.22"
    "root@10.22.22.23"
    "root@10.22.22.24"
    "root@10.22.22.25"
    "root@10.22.22.26"
)

WORKER_CONTAINER="hgq-verl-ascend"        # docker container name on each worker
NNODES=6                                 # head + len(WORKERS)
N_GPUS_PER_NODE=8                        # NPU per node
TP=4                                     # tensor-parallel → 48/4 = 12 replicas
RAY_PORT=6379
RL_INSIGHT_PORT=18080

# REPO is the in-container path on every node (head + workers). The head must
# already be inside the container; workers are reached via ssh + docker exec.
REPO=/root/hgq/ws/verl
MODEL=/root/hgq/ws/models/Llama3.1-8B-Instruct
DATASET=/root/hgq/ws/data/swe_bench_train_model.parquet
LOG_BASE=/tmp
MAX_SAMPLES=64
RES_LEN=8000
GPU_MEM_UTIL=0.8

# AKernel remote sandbox (required by the blackbox runner on every node).
# NOTE: values are passed via `docker exec -e`; if a token contains shell
# metacharacters ($ ` " ') it must be escaped or this will break.
: "${AKERNEL_SERVER_ADDRESS:?Set AKERNEL_SERVER_ADDRESS}"
: "${AKERNEL_TOKEN:?Set AKERNEL_TOKEN}"

# Tool image registry (network-dependent, like AKERNEL)
TOOL_IMAGE_REGISTRY="${TOOL_IMAGE_REGISTRY:-xx.xx.xx.xx:xxxx}"
TOOL_IMAGE="${TOOL_IMAGE_REGISTRY}/openyuanrong/mini-swe-agent-tool:latest"

# MAX_TURNS caveat (multi-node): the mini_swe_agent_runner reads AGENT_MAX_TURNS
# from the *worker process* env, which is fixed at `ray start` time and does NOT
# track the per-context MAX_TURNS computed below. We set the worker env to the
# largest value in the matrix (smallest context → most turns) so no experiment
# is truncated early. For exact per-context max_turns, the runner must be changed
# to read it from a Ray task argument instead of os.environ.
AGENT_MAX_TURNS_FIXED=150

# =====================================================================
# Helpers
# =====================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

head_ip() {
    # First non-loopback IPv4 of this host (workers reach head via it).
    # Requires --network host on the container so this is the host IP.
    hostname -I | awk '{print $1}'
}

# Run a command inside a worker's container over ssh.
#   $1 = worker host, $2.. = argv passed to `docker exec <container>`
worker_exec() {
    local host=$1; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
        "docker exec ${WORKER_CONTAINER} $*"
}

# Same, but with per-node env injected via `docker exec -e` (Ray workers inherit
# these for the whole cluster lifetime). $1 = worker host, $2 = head ip.
worker_exec_with_env() {
    local host=$1 hip=$2; shift 2
    ssh -o StrictHostKeyChecking=no "${host}" "docker exec \
        -e VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO} \
        -e VERL_RL_INSIGHT_ENABLE=1 \
        -e RL_INSIGHT_SERVER_URL=http://${hip}:${RL_INSIGHT_PORT} \
        -e AKERNEL_SERVER_ADDRESS=${AKERNEL_SERVER_ADDRESS} \
        -e AKERNEL_TOKEN=${AKERNEL_TOKEN} \
        -e AKERNEL_TUNNEL_SSL_VERIFY=${AKERNEL_TUNNEL_SSL_VERIFY:-0} \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e PYTHONHASHSEED=0 \
        -e AGENT_MAX_TURNS=${AGENT_MAX_TURNS_FIXED} \
        -e PYTHONPATH=${REPO} \
        ${WORKER_CONTAINER} $*"
}

# Fan-out a plain (no-env) docker exec across all workers in parallel.
worker_exec_all() {
    local cmd=$1
    for w in "${WORKERS[@]}"; do
        log "  → ${w}: ${cmd}"
        worker_exec "${w}" "${cmd}" &
    done
    wait
}

# =====================================================================
# Step 1: passwordless-SSH + container reachability check
# =====================================================================
step1_ssh_check() {
    log "=== Step 1: SSH + container reachability check (${#WORKERS[@]} workers) ==="
    local ok=1
    for w in "${WORKERS[@]}"; do
        if worker_exec "${w}" "true" 2>/dev/null; then
            log "  ✓ ${w} (container ${WORKER_CONTAINER})"
        else
            log "  ✗ ${w} (ssh or docker exec ${WORKER_CONTAINER} failed)"
            ok=0
        fi
    done
    [[ ${ok} -eq 1 ]] || { log "ERROR: not all workers reachable; fix ssh/docker first."; exit 1; }
}

# =====================================================================
# Step 2: Ray cluster up (head + workers)
# =====================================================================
step2_ray_up() {
    log "=== Step 2: bring up Ray cluster (nnodes=${NNODES}) ==="
    local hip; hip=$(head_ip); HEAD_IP=${hip}
    log "head IP = ${HEAD_IP}"

    log "stopping any existing ray on all nodes..."
    worker_exec_all "ray stop -f 2>/dev/null || true" >/dev/null
    pkill -9 -f 'ray::' 2>/dev/null || true
    worker_exec_all "bash -lc 'fuser -k /dev/davinci* 2>/dev/null || true'" >/dev/null
    ray stop -f 2>/dev/null || true

    log "starting head node (this container)..."
    export VERL_RL_INSIGHT_ENABLE=1
    export RL_INSIGHT_SERVER_URL="http://${HEAD_IP}:${RL_INSIGHT_PORT}"
    export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export PYTHONHASHSEED=0
    export AGENT_MAX_TURNS="${AGENT_MAX_TURNS_FIXED}"
    export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
    # Cluster-level system config: disable the idle-worker reaper so long-running
    # agent workers survive dispatch gaps. The driver's ray.init(_system_config=...)
    # is IGNORED when connecting to an already-running cluster, so this MUST be set
    # at `ray start --head` time to take effect across the 6-node cluster.
    ray start --head --port="${RAY_PORT}" --temp-dir=/tmp/ray_head \
        --system-config='{"idle_worker_killing_time_threshold_ms": 2147483647}'

    log "workers joining cluster (inside ${WORKER_CONTAINER})..."
    for w in "${WORKERS[@]}"; do
        log "  → ${w} joining"
        worker_exec_with_env "${w}" "${HEAD_IP}" \
            "ray start --address=${HEAD_IP}:${RAY_PORT} --temp-dir=/tmp/ray_worker" &
    done
    wait

    log "waiting for cluster to settle..."
    sleep 10
    ray status || true
    log "alive nodes: $(ray nodes 2>/dev/null | grep -c alive)"
}

# =====================================================================
# Step 3: rl-insight on head
# =====================================================================
step3_rl_insight() {
    log "=== Step 3: rl-insight server on head (${HEAD_IP}:${RL_INSIGHT_PORT}) ==="
    rl-insight server start 2>/dev/null || true   # already-running is fine
    trap 'rl-insight server stop 2>/dev/null || true' EXIT
    log "rl-insight up; workers scrape via RL_INSIGHT_SERVER_URL=${RL_INSIGHT_SERVER_URL}"
}

# =====================================================================
# Step 4: ascend-exps matrix (each run_infer spans all 6 nodes)
# =====================================================================
run_experiment() {
    local log_file=$1
    shift

    while ! grep -q "${TARGET}" "${log_file}" 2>/dev/null; do
        # Clean only driver processes + davinci handles across all nodes.
        # Do NOT kill ray:: actors — the cluster must stay up across retries.
        pkill -9 -f 'parallel_infer.py' 2>/dev/null || true
        worker_exec_all "bash -lc 'fuser -k /dev/davinci* 2>/dev/null || true'" >/dev/null
        npu-smi info 2>/dev/null | tail -3 || true
        log "  running → ${log_file}"

        bash "${REPO}/examples/kvc_aware_router/run_infer.sh" \
            --model-path "${MODEL}" \
            --data-path "${DATASET}" \
            --device ascend \
            --nnodes "${NNODES}" \
            --n-gpus-per-node "${N_GPUS_PER_NODE}" \
            --tp "${TP}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --response-length "${RES_LEN}" \
            --max-model-len "${CONTEXT}" \
            --max-samples "${MAX_SAMPLES}" \
            --n 8 \
            --shuffle \
            --max-concurrent-sessions "${CONCURRENCY}" \
            --max-turns "${MAX_TURNS}" \
            --tool-image "${TOOL_IMAGE}" \
            --kv-events \
            "$@" > "${log_file}" 2>&1 || log "  (run failed, will retry)"
    done
}

step4_matrix() {
    log "=== Step 4: ascend-exps matrix (nnodes=${NNODES}, tp=${TP}, replicas=$((NNODES*N_GPUS_PER_NODE/TP))) ==="
    log "WARNING: AGENT_MAX_TURNS fixed at ${AGENT_MAX_TURNS_FIXED} on workers (multi-node env limit); per-context MAX_TURNS below is recorded but NOT enforced in-sandbox."

    local concurrencys=(16 24 32 128 192 256)
    local contexts=(16384 32768 64000 128000)
    export TARGET="Resolved"

    for CONCURRENCY in "${concurrencys[@]}"; do
        for CONTEXT in "${contexts[@]}"; do
            local RECTOR=$((128000 / CONTEXT))
            local MAX_TURNS=$((150 / RECTOR))

            local LOG_FILE="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}-n${NNODES}.log"
            log "sticky concurrency=${CONCURRENCY} context=${CONTEXT} max_turns=${MAX_TURNS}"
            run_experiment "${LOG_FILE}" \
                --slow-cut least-inflight \
                --overload-mode None

            local lts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
            for lt in "${lts[@]}"; do
                LOG_FILE="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}-n${NNODES}.log"
                log "kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT} max_turns=${MAX_TURNS}"
                run_experiment "${LOG_FILE}" \
                    --slow-cut capacity-token-aware \
                    --overload-mode kv_cache_usage_perc \
                    --load-threshold "${lt}"
            done
        done
    done
    log "=== matrix complete ==="
}

# =====================================================================
# Main
# =====================================================================
step1_ssh_check
step2_ray_up
step3_rl_insight
step4_matrix
