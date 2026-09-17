#!/usr/bin/env bash
# analysis/calib/serve.sh under a restart loop: a WSL dxgkrnl GPU stall (dmesg
# dxgvmb_send_wait_sync_object_gpu, 2026-09-15 15:11) hung the scheduler until its watchdog killed
# the server mid-run. Same server args as every Phase-1 run; only the restart is new.
#   serve_supervised.sh 13b | Qwen/Qwen2.5-7B-Instruct-AWQ        (stop: touch $STOP_FILE)
set -uo pipefail
MODEL_ARG="${1:-13b}"
LOG_DIR="${LOG_DIR:-/tmp}"
STOP_FILE="${STOP_FILE:-$LOG_DIR/sglang.stop}"
rm -f "$STOP_FILE"
HERE="$(cd "$(dirname "$0")" && pwd)"
i=0
while [ ! -f "$STOP_FILE" ]; do
  i=$((i+1))
  echo "[supervisor] $(date -Is) start #$i ($MODEL_ARG)" | tee -a "$LOG_DIR/sglang_supervisor.log"
  "$HERE/../../calib/serve.sh" "$MODEL_ARG" >> "$LOG_DIR/sglang_server.log" 2>&1
  echo "[supervisor] $(date -Is) server exited rc=$?" | tee -a "$LOG_DIR/sglang_supervisor.log"
  pkill -f "m sglang[.]launch_server" 2>/dev/null; sleep 5
done
