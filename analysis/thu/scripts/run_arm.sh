#!/usr/bin/env bash
# Thursday runs. Identical to analysis/wed/scripts/run_arm.sh (the Phase-1 diagnostic config) except
# that output lands in analysis/thu/runs/. Usage:
#   run_arm.sh <tag> <beta_emo> <n_dialogues> <data.jsonl> [extra runner flags...]
set -euo pipefail
TAG=$1; BETA=$2; N=$3; DATA=$4; shift 4; EXTRA=("$@")
MODEL="${MODEL:-TheBloke/vicuna-13B-v1.5-AWQ}"
WORKERS="${WORKERS:-10}"
SIMS="${SIMS:-50}"
MAXT="${MAXT:-10}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$REPO/analysis/thu/runs/$TAG/$TAG.pkl"
mkdir -p "$REPO/analysis/thu/runs/$TAG"
cd "$REPO/src"
CMD=(python3 runners/rollout.py --game emo_p4g --algo emomcts
  --llm sglang --sglang_model "$MODEL"
  --emotion_classifier hf --num_mcts_sims "$SIMS" --llm_prior_topk 5 --max_realizations 4
  --beta_emo "$BETA" --Q_0 0.0 --cpuct 1.0 --emo_signal level --emo_risk_lambda 0.0
  --emo_valence_table soft --logit_scoring off --max_turns "$MAXT"
  --max_conv "$N" --num_workers "$WORKERS" --seed 0 --p4g_persona
  --data "$DATA" --output "$OUT" "${EXTRA[@]}")
echo "${CMD[*]}" > "$REPO/analysis/thu/runs/$TAG/command.txt"
START=$(date +%s); date -Is > "$REPO/analysis/thu/runs/$TAG/started_at.txt"
"${CMD[@]}"
END=$(date +%s); echo "wall_clock_s=$((END-START))" > "$REPO/analysis/thu/runs/$TAG/wall_clock.txt"
