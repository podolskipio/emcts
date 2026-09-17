#!/usr/bin/env bash
# 1.2 worker-count sweep, run 1 only. 24 dialogues per arm: rollout.py clamps
# workers = min(--num_workers, n_dialogues), so a 10-dialogue run cannot exercise 16 or 24.
set -x
cd "$(dirname "$0")/.."
for W in 10 16 24; do
  python3 calib/run_calib.py --run 1 --arch 8b --workers $W --dialogues 24 \
    --tag "run1_d24_w$W" --note "1.2 worker sweep, 24 dialogues so the clamp does not bind" \
    > "calib/logs/driver_run1_d24_w$W.log" 2>&1
done
echo "SWEEP DONE"
