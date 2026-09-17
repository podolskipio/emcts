#!/usr/bin/env bash
# Wednesday runs on the Phase-1 diagnostic config (analysis/phase1/scripts/run_diag.sh), with a
# chosen model / dialogue file. Usage:
#   run_arm.sh <tag> <beta_emo> <n_dialogues> <data.jsonl> [extra runner flags...]
# MODEL env var picks the served model (default vicuna-13B AWQ). Output: analysis/wed/runs/<tag>/.
set -euo pipefail
TAG=$1; BETA=$2; N=$3; DATA=$4; shift 4; EXTRA=("$@")
MODEL="${MODEL:-TheBloke/vicuna-13B-v1.5-AWQ}"
WORKERS="${WORKERS:-10}"
SIMS="${SIMS:-50}"
MAXT="${MAXT:-10}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$REPO/analysis/wed/runs/$TAG/$TAG.pkl"
mkdir -p "$REPO/analysis/wed/runs/$TAG"
cd "$REPO/src"
CMD=(python3 runners/rollout.py --game emo_p4g --algo emomcts
  --llm sglang --sglang_model "$MODEL"
  --emotion_classifier hf --num_mcts_sims "$SIMS" --llm_prior_topk 5 --max_realizations 4
  --beta_emo "$BETA" --Q_0 0.0 --cpuct 1.0 --emo_signal level --emo_risk_lambda 0.0
  --emo_valence_table soft --logit_scoring off --max_turns "$MAXT"
  --max_conv "$N" --num_workers "$WORKERS" --seed 0 --p4g_persona
  --data "$DATA" --output "$OUT" "${EXTRA[@]}")
echo "${CMD[*]}" > "$REPO/analysis/wed/runs/$TAG/command.txt"
START=$(date +%s); date -Is > "$REPO/analysis/wed/runs/$TAG/started_at.txt"
"${CMD[@]}"
END=$(date +%s); echo "wall_clock_s=$((END-START))" > "$REPO/analysis/wed/runs/$TAG/wall_clock.txt"
