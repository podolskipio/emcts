#!/usr/bin/env bash
# Run emotion-aware MCTS (EmotionAwareDiscountQOpenLoopMCTS) with a ChatGPT backbone.
#
# Override defaults via environment variables, e.g.:
#   GAME=emo_p4g LAMBDA_EMO=0.5 scripts/mcts/run_emomcts_chatgpt.sh
#
# Extra arguments after -- are passed through to runners/emomcts.py:
#   scripts/mcts/run_emomcts_chatgpt.sh --debug
#
# Results are written into a per-run subdirectory under OUTPUT (see _common.setup_output_dir):
#   outputs/<run-id>/<run-id>.pkl
#   outputs/<run-id>/<run-id>_emotions.json
#   outputs/<run-id>/<run-id>_da_emotions.json
#   outputs/<run-id>/metadata.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/src"

# tweakable via env
LLM="${LLM:-chatgpt}"                       # 'chatgpt' = Azure ChatGPT, 'gpt-3.5-turbo' = OpenAI public
GAME="${GAME:-emo_p4g}"                     # emomcts requires an emotion-aware task
NUM_DIALOGS="${NUM_DIALOGS:-20}"
NUM_MCTS_SIMS="${NUM_MCTS_SIMS:-20}"
MAX_REALIZATIONS="${MAX_REALIZATIONS:-3}"
Q_0="${Q_0:-0.0}"
LAMBDA_EMO="${LAMBDA_EMO:-0.3}"             # 0.0 = GDPZero-equivalent
GEN_SENTENCES="${GEN_SENTENCES:--1}"

# default run-id encodes the key knobs so two runs don't collide
RUN_ID="${RUN_ID:-emomcts_${GAME}_${LLM}_d${NUM_DIALOGS}_s${NUM_MCTS_SIMS}_lambda${LAMBDA_EMO}}"
OUTPUT="${OUTPUT:-outputs/${RUN_ID}.pkl}"

echo "=== emomcts / ChatGPT ==="
echo "  llm=${LLM}  game=${GAME}  num_dialogs=${NUM_DIALOGS}  num_mcts_sims=${NUM_MCTS_SIMS}  lambda_emo=${LAMBDA_EMO}"
echo "  output=${OUTPUT}"
echo

exec python runners/emomcts.py \
    --llm "${LLM}" \
    --game "${GAME}" \
    --num_dialogs "${NUM_DIALOGS}" \
    --num_mcts_sims "${NUM_MCTS_SIMS}" \
    --max_realizations "${MAX_REALIZATIONS}" \
    --Q_0 "${Q_0}" \
    --lambda_emo "${LAMBDA_EMO}" \
    --gen_sentences "${GEN_SENTENCES}" \
    --output "${OUTPUT}" \
    "$@"
