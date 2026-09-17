#!/usr/bin/env bash
# Phase-1 diagnostic run. Usage: run_diag.sh <tag> <beta_emo> <n_dialogues> [extra runner flags...]
#   run_diag.sh D1 0.7 30     run_diag.sh D2 0.0 30     run_diag.sh T1 0.7 1  (timing)
# Config (user decision 2026-09-14): Vicuna-13B AWQ, sampled value+prior (logit scoring OFF),
# R=4, persona, n_sims 50, K 5, Tmax=depth cap 10, hf classifier, level, soft, cpuct 1, Q_0 0,
# 10 workers, seed 0, dialogues 101-130 of the non-annotated pool.
set -euo pipefail
TAG=$1; BETA=$2; N=$3; shift 3; EXTRA=("$@")   # e.g. --search_horizon episode (pilot, SEARCH_HORIZON_BUG.md)
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$REPO/analysis/phase1/runs/$TAG/$TAG.pkl"
mkdir -p "$REPO/analysis/phase1/runs/$TAG"
cd "$REPO/src"
CMD=(python3 runners/rollout.py --game emo_p4g --algo emomcts
  --llm sglang --sglang_model TheBloke/vicuna-13B-v1.5-AWQ
  --emotion_classifier hf --num_mcts_sims 50 --llm_prior_topk 5 --max_realizations 4
  --beta_emo "$BETA" --Q_0 0.0 --cpuct 1.0 --emo_signal level --emo_risk_lambda 0.0
  --emo_valence_table soft --logit_scoring off --max_turns 10
  --max_conv "$N" --num_workers 10 --seed 0 --p4g_persona
  --data "$REPO/analysis/phase1/runs/dialogues_101_130.jsonl"
  --output "$OUT" "${EXTRA[@]}")
echo "${CMD[*]}" > "$REPO/analysis/phase1/runs/$TAG/command.txt"
START=$(date +%s); date -Is > "$REPO/analysis/phase1/runs/$TAG/started_at.txt"
"${CMD[@]}"
END=$(date +%s); echo "wall_clock_s=$((END-START))" > "$REPO/analysis/phase1/runs/$TAG/wall_clock.txt"
