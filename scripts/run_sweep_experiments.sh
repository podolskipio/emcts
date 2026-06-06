#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/src"

# tweakable via env
LLM="${LLM:-ollama}"
OLLAMA_MODEL="${OLLAMA_MODEL:-vicuna:13b}"
NUM_DIALOGS="${NUM_DIALOGS:-100}"
SIMS="${SIMS:-20 50}"
MAX_REALIZATIONS="${MAX_REALIZATIONS:-3}"
MAX_TURNS="${MAX_TURNS:-10}"
Q_0="${Q_0:-0.0}"
CPUCT="${CPUCT:-1.0}"
TOPK="${TOPK:-5}"
EMOTION_CLASSIFIER="${EMOTION_CLASSIFIER:-hf}"
BETA_EMO="${BETA_EMO:-0.7}"
OUT="${OUT:-outputs/sweep}"
TAG="${TAG:-vicuna}"   # short backbone tag for run-ids

# common args shared by every rollout invocation
common_args=(
    --llm "$LLM" --ollama_model "$OLLAMA_MODEL"
    --max_conv "$NUM_DIALOGS"
    --max_turns "$MAX_TURNS"
    --max_realizations "$MAX_REALIZATIONS"
    --Q_0 "$Q_0"
    --cpuct "$CPUCT"
)

echo "=============================================================="
echo "Sim-budget sweep: gdpzero / gdpzero+topk / emomcts+topk"
echo "  backbone=$LLM/$OLLAMA_MODEL  dialogs=$NUM_DIALOGS  sims={$SIMS}  topk=$TOPK  beta_emo=$BETA_EMO"
echo "  output dir=$OUT"
echo "=============================================================="

for S in $SIMS; do
    echo
    echo "######################## num_mcts_sims=$S ########################"

    # 1. GDPZero — base (no top-k pruning)
    RUN="rollout_p4g_gdpzero_${TAG}_${NUM_DIALOGS}d_${S}s"
    echo "--- [$S sims] gdpzero (no topk) -> $RUN ---"
    python runners/rollout.py \
        --game p4g --algo gdpzero \
        --num_mcts_sims "$S" \
        "${common_args[@]}" \
        --output "$OUT/${RUN}/${RUN}.pkl"

    # 2. EMOMCTS (multi-objective Q, beta_emo) — top-k=5 prior pruning
    RUN="rollout_emo_p4g_multiobjq_beta${BETA_EMO//./_}_${TAG}_${NUM_DIALOGS}d_${S}s_topk${TOPK}"
    echo "--- [$S sims] emomcts beta_emo=$BETA_EMO topk=$TOPK -> $RUN ---"
    python runners/rollout.py \
        --game emo_p4g --algo emomcts \
        --num_mcts_sims "$S" \
        --llm_prior_topk "$TOPK" \
        --emotion_classifier "$EMOTION_CLASSIFIER" \
        --beta_emo "$BETA_EMO" \
        "${common_args[@]}" \
        --output "$OUT/${RUN}/${RUN}.pkl"

    # 3. GDPZero — top-k=5 prior pruning
    RUN="rollout_p4g_gdpzero_${TAG}_${NUM_DIALOGS}d_${S}s_topk${TOPK}"
    echo "--- [$S sims] gdpzero topk=$TOPK -> $RUN ---"
    python runners/rollout.py \
        --game p4g --algo gdpzero \
        --num_mcts_sims "$S" \
        --llm_prior_topk "$TOPK" \
        "${common_args[@]}" \
        --output "$OUT/${RUN}/${RUN}.pkl"

done

echo
echo "=============================================================="
echo "Sweep complete. Per-run pickles + metadata.json under $OUT/"
echo "=============================================================="
