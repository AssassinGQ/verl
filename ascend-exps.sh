#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONHASHSEED=0

MODEL=/path/to/Qwen/Llama3.1-8B-Instruct
DATASET=/path/to/swe_bench_train_model.parquet
MAX_SAMPLES=64
RES_LEN=8000
LOG_BASE=/tmp

# Blackbox runner prerequisite: each agent session creates a real AKernel sandbox.
: "${AKERNEL_SERVER_ADDRESS:?Set AKERNEL_SERVER_ADDRESS to the AKernel endpoint}"
: "${AKERNEL_TOKEN:?Set AKERNEL_TOKEN to the AKernel access token}"
export AKERNEL_SERVER_ADDRESS AKERNEL_TOKEN
export AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"

# Tool image registry — default to the Huawei SWR CN-east-3 endpoint; override
# TOOL_IMAGE_REGISTRY to point at a mirror reachable from your network. The
# sidecar tool image path under that registry is fixed.
TOOL_IMAGE_REGISTRY="${TOOL_IMAGE_REGISTRY:-xx.xx.xx.xx:xxxx}"
TOOL_IMAGE="${TOOL_IMAGE_REGISTRY}/openyuanrong/mini-swe-agent-tool:latest"

# rl-insight online observability — start the server stack (Prometheus/Grafana/api)
# on entry and stop it on exit (incl. Ctrl-C). Requires a one-time `rl-insight server install`.
# Override VERL_RL_INSIGHT_ENABLE=0 to disable the router emitter.
export VERL_RL_INSIGHT_ENABLE="${VERL_RL_INSIGHT_ENABLE:-1}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"
rl-insight server start --detach 2>/dev/null || true
trap 'rl-insight server stop 2>/dev/null || true' EXIT

export SWE_AGENT_DUMP_TRAJECTORIES=1
export UNI_AGENT_GATEWAY_DEBUG_MESSAGE_MAX_CHARS=99999999

TARGET="Resolved"

concurrencys=(16 24 32 128 192 256)
contexts=(16384 32768 64000 128000)

run_experiment() {
    local log_file=$1
    shift

    while ! grep -q "$TARGET" "$log_file" 2>/dev/null; do
        pkill -9 -f 'parallel_infer.py|ray::' || true
        ps -aux | grep run_infer.sh | grep -v grep | awk -F ' ' '{print$2}' | xargs -r -I {} kill -9 {} || true
        ray stop || true
        fuser -k /dev/davinci* || true
        npu-smi info
        bash examples/kvc_aware_router/run_infer.sh \
            --model-path "$MODEL" \
            --data-path "$DATASET" \
            --device ascend \
            --n-gpus-per-node 8 \
            --tp 4 \
            --response-length "$RES_LEN" \
            --max-model-len "$CONTEXT" \
            --max-samples "$MAX_SAMPLES" \
            --n 8 \
            --shuffle \
            --tool-image "$TOOL_IMAGE" \
            --max-concurrent-sessions "$CONCURRENCY" \
            --max-turns "$MAX_TURNS" \
            --kv-events \
            "$@" > "$log_file" 2>&1
    done
}

for CONCURRENCY in "${concurrencys[@]}"; do
    for CONTEXT in "${contexts[@]}"; do
        RECTOR=$((128000 / CONTEXT))
        MAX_TURNS=$((150 / RECTOR))

        LOG_FILE="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
        export SWE_AGENT_TRAJECTORY_DIR="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.traj"
        echo "Running sticky concurrency=${CONCURRENCY} context=${CONTEXT} max_turns=${MAX_TURNS}"
        run_experiment "$LOG_FILE" \
            --slow-cut least-inflight \
            --overload-mode None

        lts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
        for lt in "${lts[@]}"; do
            LOG_FILE="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
            export SWE_AGENT_TRAJECTORY_DIR="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.traj"
            echo "Running kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT} max_turns=${MAX_TURNS}"
            run_experiment "$LOG_FILE" \
                --slow-cut capacity-token-aware \
                --overload-mode kv_cache_usage_perc \
                --load-threshold "$lt"
        done
    done
done
