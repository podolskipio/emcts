#!/usr/bin/env bash
# Run GDP-Zero MCTS with a ChatGPT backbone.
#
# Override defaults via environment variables, e.g.:
#   GAME=esc NUM_DIALOGS=10 scripts/mcts/run_gdpzero_chatgpt.sh
#
# Extra arguments after -- are passed through to runners/gdpzero.py:
#   scripts/mcts/run_gdpzero_chatgpt.sh --debug
#
# Results are written into a per-run subdirectory under OUTPUT (see _common.setup_output_dir):
#   outputs/<run-id>/<run-id>.pkl
#   outputs/<run-id>/metadata.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/src"

# tweakable via env
LLM="${LLM:-chatgpt}"                       # 'chatgpt' = Azure ChatGPT, 'gpt-3.5-turbo' = OpenAI public
GAME="${GAME:-p4g}"
NUM_DIALOGS="${NUM_DIALOGS:-20}"
NUM_MCTS_SIMS="${NUM_MCTS_SIMS:-20}"
MAX_REALIZATIONS="${MAX_REALIZATIONS:-3}"
Q_0="${Q_0:-0.0}"
GEN_SENTENCES="${GEN_SENTENCES:--1}"

# default run-id encodes the key knobs so two runs don't collide
RUN_ID="${RUN_ID:-gdpzero_${GAME}_${LLM}_d${NUM_DIALOGS}_s${NUM_MCTS_SIMS}}"
OUTPUT="${OUTPUT:-outputs/${RUN_ID}.pkl}"

echo "=== gdpzero / ChatGPT ==="
echo "  llm=${LLM}  game=${GAME}  num_dialogs=${NUM_DIALOGS}  num_mcts_sims=${NUM_MCTS_SIMS}"
echo "  output=${OUTPUT}"
echo

exec python runners/gdpzero.py \
    --llm "${LLM}" \
    --game "${GAME}" \
    --num_dialogs "${NUM_DIALOGS}" \
    --num_mcts_sims "${NUM_MCTS_SIMS}" \
    --max_realizations "${MAX_REALIZATIONS}" \
    --Q_0 "${Q_0}" \
    --gen_sentences "${GEN_SENTENCES}" \
    --output "${OUTPUT}" \
    "$@"
