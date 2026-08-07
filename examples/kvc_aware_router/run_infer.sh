#!/usr/bin/env bash
# KV-cache-aware router standalone inference (blackbox framework).
#
# A thin wrapper around parallel_infer.py: sets up the Ray-worker environment
# (AKernel sandbox creds + observability env), then forwards all CLI flags
# verbatim to parallel_infer.py's argparse — see `bash run_infer.sh --help`.
#
#   bash examples/kvc_aware_router/run_infer.sh --model-path /path/to/model \
#       --max-samples 1 --kv-events
#   bash examples/kvc_aware_router/run_infer.sh --help   # full flag list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── Environment variables (exported for Ray-spawned workers) ──────────────
# These are read via os.environ inside Ray tasks, not CLI flags.
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-}"
export AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

# Forward every CLI flag to parallel_infer.py (the single source of flags,
# defaults, types, and --help). No shell-side defaults — argparse owns them.
python examples/kvc_aware_router/parallel_infer.py "$@"
