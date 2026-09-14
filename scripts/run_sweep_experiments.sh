#!/usr/bin/env bash
#
# Grid A — the paper's factorial evaluation grid.
#
#   3 models (AWQ) x 2 persona conditions x 3 methods x n_sims=50 = 18 runs of 100 dialogues,
#   plus the A2 low-budget replicate grid and the A3 no-MCTS baseline.
#
#     model tag   backbone (AWQ)                                    persona   method
#     ---------   ----------------------------------------------    -------   ---------------------
#     vicuna      TheBloke/vicuna-13B-v1.5-AWQ                       no / yes  gdpzero
#     llama31     hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 no / yes  gdpzero_topk
#     qwen25      Qwen/Qwen2.5-7B-Instruct-AWQ                       no / yes  emomcts_topk
#
#   Seeds: 3 on the vicuna / no-persona cell (statistical power — the paper's critical fix),
#          1 everywhere else.
#
# !! The pre-freeze 0.70 SR / 6.72 AvgT numbers are DISCARDED. They were produced with the
# !! argmax-mined w(.) table. Every run below pins --emo_valence_table soft, and the anchor
# !! cell is re-run from scratch into gridA/runs/.
#
# Run order (stages run in this order; it matters if you run out of time):
#
#   a1  vicuna / no-persona, 3 methods x 3 seeds, n_sims=50   (~30 h)  ANCHOR CELL.
#       Minimum viable result, and the annotation study samples from its logs, so it goes first.
#   a2  n_sims=10, 3 methods x 3 models x 3 seeds              (~18 h)  error bars, kills the
#       "all three read exactly 0.58" problem.
#   a4  the remaining 5 factorial cells at n_sims=50           (~80 h)
#
# DROPPED FOR NOW (a3): the DialogXpert slot is NOT in the default run order. DialogXpert is not
# implemented in this repository — there is no trained dialogue policy here, only llm_raw /
# gdpzero / emomcts — so stage a3 would run the nearest no-search baseline (--algo llm_raw) under
# the anchor cell's conditions, tagged `llm_raw`, NOT `dialogxpert`. The stage is still defined:
# run it with STAGES=a3 to put it back. Until then RESULTS.md §3 reports the published
# 0.8132 SR / 5.07 AvgT with no reproduction beside them.
#
# ---------------------------------------------------------------------------------------------
# One SGLang server serves one model, so a full grid is three passes. Point SGLANG_HOST at the
# server and run this script once per backbone:
#
#     scripts/serve_sglang.sh MODEL=TheBloke/vicuna-13B-v1.5-AWQ      # terminal 1
#     MODELS=vicuna  scripts/run_sweep_experiments.sh                 # terminal 2
#     ...then restart the server on llama31 / qwen25 and re-run with MODELS=<tag>.
#
# Cells whose backbone is not the one currently being served are SKIPPED and recorded in
# gridA/SKIPPED.log — never silently substituted (SGLangChatModel would otherwise fall back to
# whatever the server happens to hold, which would quietly destroy the model factor).
#
# Everything is resumable: a run whose pickle already holds NUM_DIALOGS episodes is skipped.
# ---------------------------------------------------------------------------------------------

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRID_DIR="${GRID_DIR:-$REPO_ROOT/gridA}"
OUT="${OUT:-$GRID_DIR/runs}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Frozen grid parameters. DO NOT change any of these mid-grid: the comparability
# rule and the paired tests both assume every cell was produced by one batch.
# If one must change, the BATCH.json guard below stops the sweep and you restart
# under NEW_BATCH=1, which treats everything already on disk as a separate batch.
# ---------------------------------------------------------------------------
NUM_DIALOGS="${NUM_DIALOGS:-100}"
SIMS="${SIMS:-50}"                 # main factorial budget
A2_SIMS="${A2_SIMS:-10}"           # low-budget replicate grid
MAX_TURNS="${MAX_TURNS:-10}"
MAX_REALIZATIONS="${MAX_REALIZATIONS:-3}"
Q_0="${Q_0:-0.0}"
CPUCT="${CPUCT:-1.0}"
TOPK="${TOPK:-5}"
BETA_EMO="${BETA_EMO:-0.7}"
EMOTION_CLASSIFIER="${EMOTION_CLASSIFIER:-hf}"
EMO_SIGNAL="${EMO_SIGNAL:-level}"          # spec default — do not vary in this grid
EMO_RISK_LAMBDA="${EMO_RISK_LAMBDA:-0.0}"  # spec default — do not vary in this grid
EMO_VALENCE_TABLE="${EMO_VALENCE_TABLE:-soft}"  # post-freeze table; argmax is discarded
ANCHOR_SEEDS="${ANCHOR_SEEDS:-0 1 2}"      # the vicuna/no-persona cell gets 3 seeds
A2_SEEDS="${A2_SEEDS:-0 1 2}"
SEED="${SEED:-0}"                          # the single seed everywhere else

# execution knobs (not part of the frozen grid: they change speed, not the sampled distribution)
LLM="${LLM:-sglang}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SGLANG_HOST="${SGLANG_HOST:-http://127.0.0.1:30000}"
MIN_CACHE_HIT="${MIN_CACHE_HIT:-0.50}"     # stop condition: realization-cache hit rate
STAGES="${STAGES:-a1 a2 a4}"
MODELS="${MODELS:-vicuna llama31 qwen25}"
DRY_RUN="${DRY_RUN:-0}"
ON_MODEL_MISMATCH="${ON_MODEL_MISMATCH:-skip}"   # skip | abort

VICUNA_MODEL="${VICUNA_MODEL:-TheBloke/vicuna-13B-v1.5-AWQ}"
LLAMA31_MODEL="${LLAMA31_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
QWEN25_MODEL="${QWEN25_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

STOP_FILE="$GRID_DIR/STOP"
SKIPPED_LOG="$GRID_DIR/SKIPPED.log"
TIMING_TSV="$GRID_DIR/TIMING.tsv"
BATCH_JSON="$GRID_DIR/BATCH.json"

model_path() {
    case "$1" in
        vicuna)  echo "$VICUNA_MODEL" ;;
        llama31) echo "$LLAMA31_MODEL" ;;
        qwen25)  echo "$QWEN25_MODEL" ;;
        *) echo "unknown model tag '$1'" >&2; return 1 ;;
    esac
}

# Ollama tags for the smoke-test path (--llm ollama); the grid itself runs on SGLang/AWQ.
ollama_path() {
    case "$1" in
        vicuna)  echo "${OLLAMA_VICUNA_MODEL:-vicuna:13b}" ;;
        llama31) echo "${OLLAMA_LLAMA31_MODEL:-llama3.1:8b}" ;;
        qwen25)  echo "${OLLAMA_QWEN25_MODEL:-qwen2.5:7b}" ;;
        *) echo "unknown model tag '$1'" >&2; return 1 ;;
    esac
}

# The three grid methods plus the a3 baseline. Every method sees the same scenarios in the same
# order (--data default, --max_conv NUM_DIALOGS over a file-ordered corpus) and the same persona
# assignment (--p4g_persona keys off the dialogue id), which is what makes McNemar paired.
method_args() {
    case "$1" in
        gdpzero)      echo "--game p4g --algo gdpzero" ;;
        gdpzero_topk) echo "--game p4g --algo gdpzero --llm_prior_topk $TOPK" ;;
        emomcts_topk) echo "--game emo_p4g --algo emomcts --llm_prior_topk $TOPK \
                            --emotion_classifier $EMOTION_CLASSIFIER --beta_emo $BETA_EMO \
                            --emo_signal $EMO_SIGNAL --emo_risk_lambda $EMO_RISK_LAMBDA \
                            --emo_valence_table $EMO_VALENCE_TABLE" ;;
        llm_raw)      echo "--game p4g --algo llm_raw" ;;
        *) echo "unknown method '$1'" >&2; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# batch guard — "do not change any parameter mid-grid"
# ---------------------------------------------------------------------------
batch_fingerprint() {
    cat <<EOF
{
  "num_dialogs": $NUM_DIALOGS,
  "sims": $SIMS,
  "a2_sims": $A2_SIMS,
  "max_turns": $MAX_TURNS,
  "max_realizations": $MAX_REALIZATIONS,
  "Q_0": $Q_0,
  "cpuct": $CPUCT,
  "topk": $TOPK,
  "beta_emo": $BETA_EMO,
  "emotion_classifier": "$EMOTION_CLASSIFIER",
  "emo_signal": "$EMO_SIGNAL",
  "emo_risk_lambda": $EMO_RISK_LAMBDA,
  "emo_valence_table": "$EMO_VALENCE_TABLE",
  "anchor_seeds": "$ANCHOR_SEEDS",
  "a2_seeds": "$A2_SEEDS",
  "seed": $SEED,
  "llm": "$LLM",
  "models": {"vicuna": "$VICUNA_MODEL", "llama31": "$LLAMA31_MODEL", "qwen25": "$QWEN25_MODEL"}
}
EOF
}

check_batch() {
    local new="$GRID_DIR/.batch.new"
    batch_fingerprint > "$new"
    if [[ -f "$BATCH_JSON" ]] && ! diff -q "$BATCH_JSON" "$new" >/dev/null 2>&1; then
        echo
        echo "!!! GRID PARAMETERS CHANGED MID-GRID !!!"
        echo "  $BATCH_JSON was written by an earlier pass and does not match this invocation:"
        diff "$BATCH_JSON" "$new" || true
        echo
        echo "  Runs already on disk belong to a different batch and are NOT comparable to what"
        echo "  this invocation would produce. Either restore the old settings, or open a new"
        echo "  batch in a fresh grid dir:"
        echo "      GRID_DIR=$GRID_DIR.batch2 $0"
        if [[ "${NEW_BATCH:-0}" != "1" ]]; then rm -f "$new"; exit 3; fi
        echo "  NEW_BATCH=1 set — continuing and rewriting the fingerprint."
    fi
    mv "$new" "$BATCH_JSON"
    # provenance is appended to, never compared: the code revision may legitimately advance
    # between passes (a bug fix in an unrelated runner), and the report prints every entry so a
    # revision change across the grid is visible rather than silent.
    printf '%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
        "stages='$STAGES' models='$MODELS' workers=$NUM_WORKERS" >> "$GRID_DIR/PROVENANCE.tsv"
}


# ---------------------------------------------------------------------------
# which backbone is the server actually holding?
# ---------------------------------------------------------------------------
SERVED_MODEL=""
probe_served_model() {
    [[ "$LLM" == "sglang" ]] || { SERVED_MODEL="(not sglang)"; return 0; }
    SERVED_MODEL="$(curl -s --max-time 10 "${SGLANG_HOST%/}/v1/models" \
        | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["data"][0]["id"])
except Exception:
    print("")' 2>/dev/null)"
    if [[ -z "$SERVED_MODEL" ]]; then
        echo "!!! no SGLang server reachable at $SGLANG_HOST"
        echo "    start one:  MODEL=<hf-path> scripts/serve_sglang.sh"
        exit 4
    fi
    echo "SGLang at $SGLANG_HOST serves: $SERVED_MODEL"
}

model_is_served() {
    [[ "$LLM" != "sglang" ]] && return 0
    [[ "$(model_path "$1")" == "$SERVED_MODEL" ]]
}

# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
n_episodes() {
    python3 - "$1" <<'PY' 2>/dev/null || echo 0
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f:
        print(len(pickle.load(f)))
except Exception:
    print(0)
PY
}

run_cell() {
    local stage="$1" tag="$2" persona="$3" method="$4" sims="$5" seed="$6"
    local run="${stage}__${tag}__${persona}persona__${method}__${sims}s__seed${seed}"
    local run_dir="$OUT/$run"
    local pkl="$run_dir/${run}.pkl"

    [[ -f "$STOP_FILE" ]] && return 0

    if ! model_is_served "$tag"; then
        local msg="SKIP $run — needs $(model_path "$tag"), server holds $SERVED_MODEL"
        echo "### $msg"
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $msg" >> "$SKIPPED_LOG"
        [[ "$ON_MODEL_MISMATCH" == "abort" ]] && { echo "ON_MODEL_MISMATCH=abort"; exit 5; }
        return 0
    fi

    local have; have="$(n_episodes "$pkl")"
    if [[ "$have" -ge "$NUM_DIALOGS" ]]; then
        echo "--- [$stage] $run: already complete ($have episodes) — skipping"
        return 0
    fi

    # --llm sglang is the grid backend; --llm ollama is supported for a local smoke test, where
    # the model tag maps onto an Ollama tag instead (OLLAMA_<TAG>_MODEL).
    local backbone=(--llm "$LLM")
    if [[ "$LLM" == "ollama" ]]; then
        backbone+=(--ollama_model "$(ollama_path "$tag")")
        [[ -n "${OLLAMA_HOST:-}" ]] && backbone+=(--ollama_host "$OLLAMA_HOST")
    else
        backbone+=(--sglang_model "$(model_path "$tag")")
    fi

    local args=(
        "${backbone[@]}"
        $(method_args "$method")
        --num_mcts_sims "$sims"
        --max_conv "$NUM_DIALOGS"
        --max_turns "$MAX_TURNS"
        --max_realizations "$MAX_REALIZATIONS"
        --Q_0 "$Q_0" --cpuct "$CPUCT"
        --seed "$seed"
        --num_workers "$NUM_WORKERS"
        --profile_roles
        --output "$OUT/${run}.pkl"
    )
    [[ "$persona" == "yes" ]] && args+=(--p4g_persona)

    echo
    echo "=============================================================================="
    echo "[$stage] $run"
    echo "  backbone=${backbone[3]}  persona=$persona  method=$method  sims=$sims  seed=$seed"
    echo "=============================================================================="
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY_RUN: python runners/rollout.py ${args[*]}"
        return 0
    fi

    local t0 t1 status=0
    t0=$(date +%s)
    ( cd "$REPO_ROOT/src" && python runners/rollout.py "${args[@]}" ) 2>&1 \
        | tee "$OUT/${run}.log" || status=${PIPESTATUS[0]}
    t1=$(date +%s)

    have="$(n_episodes "$pkl")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stage" "$tag" "$persona" "$method" "$sims" "$seed" "$((t1-t0))" "$have" "$status" \
        >> "$TIMING_TSV"

    # ---- stop condition 1: the cell did not complete -------------------------------------
    if [[ "$status" -ne 0 || "$have" -lt "$NUM_DIALOGS" ]]; then
        {
            echo "CELL FAILED: $run"
            echo "  exit status : $status"
            echo "  episodes    : $have / $NUM_DIALOGS"
            echo "  log         : $OUT/${run}.log"
            echo "  A missing cell destroys the model x persona interaction. Do NOT substitute a"
            echo "  different configuration — fix this cell and re-run (the sweep is resumable)."
        } | tee "$STOP_FILE"
        return 1
    fi

    # ---- stop condition 2: realization-cache hit rate collapsed ---------------------------
    if ! python3 "$REPO_ROOT/scripts/gridA_cache_hit.py" "$run_dir" --min "$MIN_CACHE_HIT"; then
        {
            echo "CACHE HIT RATE BELOW ${MIN_CACHE_HIT} in $run"
            echo "  The MCTS realization cache keys on the serialized prompt/state string, so a drop"
            echo "  this large means prompt serialization changed. Diagnose before running anything"
            echo "  else — runs on either side of the change are not one batch."
        } | tee "$STOP_FILE"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
METHODS=(gdpzero gdpzero_topk emomcts_topk)

stage_a1() {   # anchor: vicuna / no-persona, 3 methods x 3 seeds, n_sims=50
    echo; echo "######################## STAGE a1 — ANCHOR CELL ########################"
    for seed in $ANCHOR_SEEDS; do
        for m in "${METHODS[@]}"; do
            run_cell a1 vicuna no "$m" "$SIMS" "$seed" || return 1
        done
    done
}

stage_a3() {   # no-MCTS baseline on vicuna. NOT in the default STAGES -- run with STAGES=a3.
    echo; echo "######################## STAGE a3 — no-MCTS baseline ########################"
    run_cell a3 vicuna no llm_raw 0 "$SEED" || return 1
}

stage_a2() {   # low-budget replicate grid: n_sims=10, 3 methods x 3 models x 3 seeds
    echo; echo "######################## STAGE a2 — n_sims=$A2_SIMS replicates ########################"
    for tag in $MODELS; do
        for seed in $A2_SEEDS; do
            for m in "${METHODS[@]}"; do
                run_cell a2 "$tag" no "$m" "$A2_SIMS" "$seed" || return 1
            done
        done
    done
}

stage_a4() {   # the remaining 5 factorial cells at n_sims=50, 1 seed
    echo; echo "######################## STAGE a4 — remaining factorial cells ########################"
    for tag in $MODELS; do
        for persona in no yes; do
            # vicuna/no-persona is the anchor cell, already covered by a1
            [[ "$tag" == "vicuna" && "$persona" == "no" ]] && continue
            for m in "${METHODS[@]}"; do
                run_cell a4 "$tag" "$persona" "$m" "$SIMS" "$SEED" || return 1
            done
        done
    done
}

# ---------------------------------------------------------------------------
main() {
    if [[ -f "$STOP_FILE" ]]; then
        echo "!!! a previous pass hit a stop condition:"; echo; cat "$STOP_FILE"; echo
        echo "Resolve it, then remove $STOP_FILE to continue."
        exit 2
    fi
    check_batch
    probe_served_model
    [[ -f "$TIMING_TSV" ]] || printf 'stage\tmodel\tpersona\tmethod\tsims\tseed\twall_s\tepisodes\tstatus\n' > "$TIMING_TSV"

    echo "=============================================================================="
    echo "Grid A   dialogs=$NUM_DIALOGS  sims=$SIMS (a2: $A2_SIMS)  topk=$TOPK  beta_emo=$BETA_EMO"
    echo "         emo_signal=$EMO_SIGNAL  emo_risk_lambda=$EMO_RISK_LAMBDA  w(e)=$EMO_VALENCE_TABLE"
    echo "         stages='$STAGES'  models='$MODELS'  workers=$NUM_WORKERS"
    echo "         out=$OUT"
    echo "=============================================================================="

    for stage in $STAGES; do
        case "$stage" in
            a1) stage_a1 || break ;;
            a2) stage_a2 || break ;;
            a3) stage_a3 || break ;;
            a4) stage_a4 || break ;;
            *) echo "unknown stage '$stage' (choose from a1 a2 a3 a4)"; exit 1 ;;
        esac
    done

    echo
    if [[ -f "$STOP_FILE" ]]; then
        echo "########## SWEEP STOPPED ##########"; cat "$STOP_FILE"
        echo "###################################"
    else
        echo "Stages complete: $STAGES"
    fi
    [[ -s "$SKIPPED_LOG" ]] && { echo; echo "Skipped cells (wrong backbone served) — see $SKIPPED_LOG:"; cat "$SKIPPED_LOG"; }
    echo
    echo "Now build the report:"
    echo "    python3 scripts/gridA_report.py --grid-dir $GRID_DIR"
    [[ -f "$STOP_FILE" ]] && exit 2 || exit 0
}

main "$@"
