#!/usr/bin/env bash
# Launch the SGLang server for the MCTS runners (--llm sglang).
#
# Override anything via env vars, e.g.:
#   PORT=30001 MEM_FRACTION=0.86 scripts/serve_sglang.sh
# Anything after -- is passed straight to launch_server.
#
# sglang lives in ~/virtual_envs/SGLEnv (requirements-sglang.txt), NOT in .local_env, so this
# script does not use whatever python3 is on PATH. Override with PY=... if needed.

set -euo pipefail

MODEL="${MODEL:-TheBloke/vicuna-13B-v1.5-AWQ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MEM_FRACTION="${MEM_FRACTION:-0.82}"          # ~11.5 GB of KV (15k tokens) on the 24 GB 4090
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"    # one chunk per prompt; 512 split each into 5 rounds
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"  # covers n=15 sampling plus --num_workers concurrency
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"     # run queued requests sharing a prefix back to back
PY="${PY:-$HOME/virtual_envs/SGLEnv/bin/python}"   # server env: requirements-sglang.txt

if ! "$PY" -c "import sglang" 2>/dev/null; then
  echo "error: no sglang in $PY" >&2
  echo "  create it:  python3 -m venv ~/virtual_envs/SGLEnv && \\" >&2
  echo "              ~/virtual_envs/SGLEnv/bin/pip install -r requirements-sglang.txt" >&2
  exit 1
fi

echo "=== sglang: ${MODEL} on ${HOST}:${PORT} ==="

exec "$PY" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --schedule-policy "${SCHEDULE_POLICY}" \
  --enable-metrics \
  "$@"
