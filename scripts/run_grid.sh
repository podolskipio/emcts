#!/usr/bin/env bash
# Start the SGLang server (if needed) and run the §4c grid.
#
#   scripts/run_grid.sh                    # first start
#   scripts/run_grid.sh --continue         # after Ctrl+C / a crash / a stopped run
#   scripts/run_grid.sh --dry_run          # list the runs, check the configs, run nothing
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ scripts/run_grid.sh --continue    # Order step 5 (Qwen)
#
# Every argument is passed to scripts/run_grid.py. The server runs under the restart supervisor
# (analysis/wed/scripts/serve_supervised.sh) in ~/virtual_envs/SGLEnv, in its own session, so
# Ctrl+C stops the grid but not the server. It keeps running afterwards so --continue starts
# straight away. Stop it with:
#   touch analysis/thu/runs/sglang.stop && pkill -f "m sglang[.]launch_server"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-TheBloke/vicuna-13B-v1.5-AWQ}"
CLIENT_PY="${CLIENT_PY:-$REPO/.local_env/bin/python}"      # runners: requirements.txt
SERVER_PY="${SERVER_PY:-$HOME/virtual_envs/SGLEnv/bin/python}"   # server: requirements-sglang.txt
SERVER_URL="http://127.0.0.1:30000"
LOG_DIR="$REPO/analysis/thu/runs"
STOP_FILE="$LOG_DIR/sglang.stop"

served_model() {
  curl -s --max-time 5 "$SERVER_URL/v1/models" | "$CLIENT_PY" -c \
    "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true
}

stop_server() {
  touch "$STOP_FILE"                                        # the supervisor must not restart it
  pkill -f "m sglang[.]launch_server" 2>/dev/null || true
  while pgrep -f "serve_supervised.sh|m sglang[.]launch_server" >/dev/null; do sleep 2; done
}

# --dry_run needs no server
if [[ " $* " == *" --dry_run "* ]]; then
  exec "$CLIENT_PY" scripts/run_grid.py "$@"
fi

# 1. The server: reuse it if it already serves $MODEL, otherwise (re)start it.
current="$(served_model)"
if [[ "$current" == "$MODEL" ]]; then
  echo "SGLang already serving $MODEL"
else
  if [[ -n "$current" ]] || pgrep -f "serve_supervised.sh|m sglang[.]launch_server" >/dev/null; then
    echo "Stopping the SGLang server ($current) to switch to $MODEL"
    stop_server
  fi
  mkdir -p "$LOG_DIR"
  echo "Starting SGLang on $MODEL (log: analysis/thu/runs/sglang_server.log)"
  LOG_DIR="$LOG_DIR" STOP_FILE="$STOP_FILE" PY="$SERVER_PY" \
    setsid nohup bash analysis/wed/scripts/serve_supervised.sh "$MODEL" >/dev/null 2>&1 &

  # 2. Wait until it answers (loading + CUDA-graph capture takes a few minutes).
  for i in $(seq 1 120); do
    [[ "$(served_model)" == "$MODEL" ]] && break
    sleep 5
  done
  if [[ "$(served_model)" != "$MODEL" ]]; then
    echo "SGLang did not come up within 10 minutes -- see analysis/thu/runs/sglang_server.log"
    exit 1
  fi
  echo "SGLang ready"
fi

# 3. The grid.
exec "$CLIENT_PY" scripts/run_grid.py "$@"
