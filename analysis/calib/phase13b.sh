#!/usr/bin/env bash
# After run 2 finishes: swap the server to Vicuna-13B AWQ, then
#   run 3            -- 13B + persona, 24 dialogues  (prices the 3 13B/persona cells)
#   confirmation     -- 13B, no persona, 100 dialogues
# The confirmation is both the SR check against the published 0.700 and the measured
# 13B/no-persona anchor cell, so its GPU time is not spent twice.
set -x
cd "$(dirname "$0")/.."
W=$(cat calib/adopted_workers.txt)

while pgrep -f "calib/after_sweep.sh" > /dev/null; do sleep 30; done

# swap the server; only one may be resident at a time on a 24 GB card
pkill -f "sglang.launch_server" || true
sleep 20
nohup calib/serve.sh 13b > calib/logs/server_13b.log 2>&1 &
until curl -sf http://127.0.0.1:30000/health > /dev/null; do sleep 10; done
sleep 10
curl -s http://127.0.0.1:30000/get_model_info

python3 calib/run_calib.py --run 3 --arch 13b --workers "$W" --dialogues 24 --persona \
  --tag "run3_d24_w$W" --note "13B + persona at the adopted worker count" \
  > calib/logs/driver_run3.log 2>&1
echo "RUN3 DONE"

python3 calib/run_calib.py --run 4 --arch 13b --workers "$W" --dialogues 100 \
  --tag "confirm_13b_nopersona_d100" \
  --note "SR confirmation vs the published 0.700, and the measured 13B/no-persona anchor cell" \
  > calib/logs/driver_confirm.log 2>&1
echo "CONFIRM DONE"
