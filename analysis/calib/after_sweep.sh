#!/usr/bin/env bash
# Waits for the 1.2 worker sweep to finish, adopts the highest *stable* worker count, and
# runs calibration run 2 (8B + persona) at it. Keeps the GPU busy without a manual step.
set -x
cd "$(dirname "$0")/.."
while pgrep -f "calib/sweep.sh" > /dev/null; do sleep 30; done
W=$(python3 - <<'PY'
import json
rows=[r for r in json.load(open('calib/throughput.json')) if r['run']==1 and r['dialogues']==24 and r['exit_code']==0]
# "stable" = no queue backlog and KV pool not exhausted; among those, the fastest.
ok=[r for r in rows if r['gpu'].get('peak_kv_token_usage',1) < 0.95]
print(max(ok or rows, key=lambda r: r['generations_per_s'])['workers'])
PY
)
echo "ADOPTED WORKERS=$W"
echo "$W" > calib/adopted_workers.txt
python3 calib/run_calib.py --run 2 --arch 8b --workers "$W" --dialogues 24 --persona \
  --tag "run2_d24_w$W" --note "8B + persona at the adopted worker count" \
  > "calib/logs/driver_run2.log" 2>&1
echo "RUN2 DONE"
