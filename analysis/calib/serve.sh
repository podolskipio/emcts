#!/usr/bin/env bash
# Launch the SGLang server used by the Phase-1 calibration runs.
#
#   calib/serve.sh 8b      -> hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
#   calib/serve.sh 13b     -> TheBloke/vicuna-13B-v1.5-AWQ
#
# Everything except --model-path is held constant across the three calibration runs, so
# the only thing the worker sweep varies is --num_workers on the client.
set -euo pipefail

case "${1:-8b}" in
  8b)  MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" ;;
  13b) MODEL="TheBloke/vicuna-13B-v1.5-AWQ" ;;
  *)   MODEL="$1" ;;
esac

PY="${PY:-$HOME/virtual_envs/SGLEnv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"      # vicuna's hard ceiling; held for the 8B too
MEM_FRACTION="${MEM_FRACTION:-0.82}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"  # must cover the 24-worker arm, so it is not
                                              # the thing that caps the sweep (see BUDGET.md)
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"

echo "=== sglang: ${MODEL} on ${HOST}:${PORT} (ctx ${CONTEXT_LENGTH}, mem ${MEM_FRACTION}, cg-bs ${CUDA_GRAPH_MAX_BS}) ==="

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
  "${@:2}"
