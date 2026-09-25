# EmoMCTS — Emotion-Aware Monte-Carlo Tree Search for Dialogue Policy Planning

EmoMCTS plans goal-oriented dialogue with **open-loop Monte-Carlo Tree Search over
LLM-prompted simulations**, and makes the user's **emotion a first-class search
coordinate**. Alongside the usual task-value estimate, it maintains a *parallel*
action-value channel that tracks the expected **emotional valence** of the user's
reaction, and folds it into the PUCT selection rule through a single weight `β`.

The method is applied to **PersuasionForGood** (persuade a user to donate to *Save the
Children*).

Dialogue simulators are prompted LLMs — OpenAI, Azure OpenAI, local 🤗 Transformers,
[SGLang](https://github.com/sgl-project/sglang), or local [Ollama](https://ollama.com). All reported
results use an **open-source Vicuna-13B** backbone, so they are fully reproducible without a
proprietary API. The 2026-09 runs serve `TheBloke/vicuna-13B-v1.5-AWQ` with SGLang 0.5.9;
Qwen2.5-7B-Instruct-AWQ is the second backbone.

> **Status — frozen 2026-09-17 (`git tag thu-freeze`).** The evaluation grid is pre-registered in
> [`PREREG.md`](PREREG.md) and every frozen value is in [`analysis/FREEZE_NOTES.md`](analysis/FREEZE_NOTES.md) §10.
> Three findings changed the design and are worth knowing before reading anything older:
> the mined valence table was partly fitted on **post-decision** turns, so the grid runs the
> never-fitted `generic` table; tree search **ignored the episode horizon** and now stops at it
> (`--search_horizon episode`); and the P4G success detector accepts **hedged non-commitments**.
> See [`analysis/README.md`](analysis/README.md) for what is current and what is the record.

**Contents:** [Method](#method) · [Layout](#repository-layout) · [Setup](#setup) ·
[Data](#data) · [Running EmoMCTS](#running-emomcts) ·
[Self-play metrics](#self-play-metrics--sr--at) · [LLM judge](#pairwise-llm-judge) ·
[Reproducing the paper](#reproducing-the-paper) · [Interactive demo](#interactive-demo)

## Method

EmoMCTS (`EmotionAwareMultiObjectiveQ`) keeps **two** value tables per
`(state, action)`:

- `Q[s][a]` — the donation-rollout value (standard GDP-Zero behaviour), and
- `Q_emo[s][a]` — the running mean **emotional valence** of the user's reaction.

Selection uses a PUCT rule that adds the emotion channel to the task channel:

```
score(a) = Q[s][a] + β · Q_emo[s][a] + c_puct · P[s][a] · √N(s) / (1 + N(s,a))
```

The valence weights `w(e)` per emotion come from `--emo_valence_table`:

| table | what it is | use |
|---|---|---|
| `soft` (default) | mined: `w(e) ∝ P(donate \| emotion = e) − base_rate`, all turns (`src/emotion_mining/mine_emotion_donation_p4g.py`) | the shipped default; **reported as a finding, not used by grid arms** |
| `generic` | textbook valence signs, never fitted to outcomes | **what the frozen grid runs** |
| `predecision` | the same mining recipe restricted to turns **before** the donation decision | ablation only (C4) |
| `argmax` | the mined table under hard label assignment | comparison |

**Why the mined table is not the default for the grid.** It is fitted partly on turns at or after the
user accepts: about two-thirds of `w(happiness)` comes from them ("I'll donate" is a happy turn). Mined
on turns strictly *before* the decision, **no emotion's confidence interval excludes the base rate** —
under uniform, last-turn or recency weighting (42 tests, 0 hits after multiplicity correction) — while
the same estimator still detects the post-decision effect. So on this corpus, persuadee emotion before
the decision does not measurably predict donation, and the grid uses signs that were never fitted to
outcomes. [`analysis/thu/remine.md`](analysis/thu/remine.md),
[`analysis/thu/construct_test.md`](analysis/thu/construct_test.md) §3b.

Two further levers, shared with the GDP-Zero baseline:

- **Top-`K` prior pruning** (`--llm_prior_topk`): the search is hard-pruned to the `K`
  highest-prior dialogue acts per node, concentrating the simulation budget.
- **Emotion classifier** (`--emotion_classifier hf`): a deterministic encoder
  (`j-hartmann/emotion-english-distilroberta-base`) labels each user reaction — no LLM
  cost, fully reproducible.
- **Search horizon** (`--search_horizon {legacy,episode}`, all MCTS runners): which
  `get_dialog_ended` values end a simulated branch.
  - `legacy` (default, GDP-Zero's rule) treats only a donation as terminal. The −1 the game
    returns at the turn limit or on a repetition loop is ignored, so search keeps expanding
    dialogue states past `--max_turns` that no episode can reach.
  - `episode` stops a branch on any non-zero `get_dialog_ended` and backs up that value
    (+1 donate, −1 turn limit or stall), so search stops exactly where the episode loop stops.
  - See [`analysis/phase1/SEARCH_HORIZON_BUG.md`](analysis/phase1/SEARCH_HORIZON_BUG.md) and the
    pilot [`analysis/thu/p1_pilot.md`](analysis/thu/p1_pilot.md). **`episode` is the frozen grid default.**
- **Success criterion** (`--p4g_success {tag,committed,amount}`, `rollout.py`): which `[donate]`-tagged
  persuadee turn ends the episode *and the search branch*. `tag` (default, GDP-Zero's rule) accepts
  hedged non-commitments; `committed` requires no hedge; `amount` also requires a named amount. The grid
  runs `tag` and reports `committed` offline — a strict-but-imperfect detector inside the tree corrupts
  backups. [`analysis/thu/success_criterion_fix.md`](analysis/thu/success_criterion_fix.md).
- **Config integrity** (`--frozen_config <json>`): refuses to start a run whose *resolved* arguments
  differ from a frozen template, and refuses arm flags passed where they are inert. Runner defaults are
  not the grid config (`--num_mcts_sims` 20, `--max_realizations` 3, `--max_conv` 20, `--emotion_classifier llm`,
  no top-K, no seed), so every grid config writes every value out explicitly.

### Selection and pooling arms

All default off, so the shipped planner is unchanged. Each is measured in `analysis/thu/`.

| arm | flags | what it changes |
|---|---|---|
| **Bias** | `--beta_emo β` | the shipped emotion channel: `β · Q_emo` in PUCT |
| **CenteredBias** | `--beta_emo β --emo_centre` | centres `Q_emo` over expanded siblings, removing the flat visited-edge bonus |
| **Momentum** | `--beta_emo β --emo_signal delta` | backs up the **average local affective change** per edge, `(ν(child) − ν(parent))/2`, instead of the level |
| **AffPool** | `--aff_pool --aff_pool_bias b --aff_pool_tau τ` | RAVE-style pooling of the task return, keyed `(affect bucket, act)` |
| **ActPool** | `--aff_pool --aff_pool_key act` | the same pool keyed by act alone — the control that makes AffPool's affective claim testable |

The three selection arms are compared at **matched dose** (equal rate of changing the baseline's chosen
action), not at equal β: `β` 0.70 / 1.03 / 1.11 for Bias / CenteredBias / Momentum, all ≈15.2 %.
[`analysis/thu/momentum.md`](analysis/thu/momentum.md) §3.

### Realization-cache fixes

The open-loop tree keys a node by its system-act prefix and caches up to `--max_realizations` (R)
user replies per node. Once a child's pool is full, every later visit **replays** a cached reply
instead of generating. [`analysis/phase1/p_depth.md`](analysis/phase1/p_depth.md) found three problems
with this:
- The draw ignores which parent a reply was written for, so at depth ≥ 2 about 74 % of served replies
  were generated under a different parent realization.
- Depth-1 edges replay the same R replies on every visit, because the root pool holds one state.
- Replies that end the episode never enter the pool, so the cache never serves one.

The fixes below all default **off**, so the frozen grid is bit-identical (the golden fingerprints in
`tests/test_wed_arms.py` still pass). They are not part of the frozen grid.

| fix | flags | what it changes |
|---|---|---|
| **Ended children** | `--cache_ended_children` | a generated reply that ends the search is filed under its node when it is generated, so the cache can serve it. It sits in a separate table and is never sampled as a parent, so nothing is searched past the end of an episode. This fixes a bug: without the flag, only a non-terminal re-entry adds to the pool |
| **Fresh depth 1** | `--cache_fresh_depth1` | the root's outgoing edges always generate, and their child pools keep every reply (uncapped). Deeper edges still cache |
| **Bucket draw** | `--cache_draw bucket --cache_bucket_tau τ` | a cache hit draws only among replies whose generating parent's ν was on the same side of τ as the current parent's. If that side has no replies (a bucket miss), it draws uniformly |
| **Kernel draw** | `--cache_draw kernel --cache_kernel_h h` | a cache hit weights reply *j* by `exp(−(ν_now − ν_gen,j)² / h²)` |
| **Both** | `--cache_draw bucket_kernel` (needs τ and h) | the kernel weights within the matching bucket |

The first two flags apply to `--algo gdpzero` and `--algo emomcts`. `--cache_draw` needs a parent ν, so
it applies to `emomcts` only. None of the draws generates more than the default: the rule for when to use
the cache is unchanged. τ and h have no defaults because neither is fitted (0.263 is the frozen generic
τ_med). When `--cache_draw` is not `uniform`, each `sim_steps` record also logs
`child_generating_parent_nu`, `cache_draw_bucket_miss`, `cache_draw_prob` and `cache_draw_prob_uniform`,
so the new mismatch rate can be measured directly. Tests: `tests/test_cache_fixes.py`.

## Results

> ⚠ **These are pre-freeze numbers and are not the paper's results.** They were produced under
> `--search_horizon legacy` (search did not stop at the turn limit) with the mined `soft` valence
> table, before the arms were dose-matched. They are kept as the record of where the project stood.
> The frozen grid that supersedes them is specified in [`PREREG.md`](PREREG.md) and
> [`analysis/thu/plan_4c_run_table.md`](analysis/thu/plan_4c_run_table.md); its configs are generated and
> validated in `analysis/grid/configs/`. Two later measurements bear on the table directly:
> under `episode`, NoEmo reaches **SR 0.78** on 100 held-out dialogues
> ([`analysis/thu/success_criterion_fix.md`](analysis/thu/success_criterion_fix.md)), and a
> 10-dialogue pilot in this environment can be off by more than 0.2 SR, so small-n rows should not be
> read as effects.

PersuasionForGood, **100 dialogues** per cell, Vicuna-13B backbone, `max_turns = 10`.
SR = success rate (↑), AvgT = average turns to resolution (↓). Best SR per budget in **bold**.

| Sims | Method            |  SR   | AvgT |
|:----:|-------------------|:-----:|:----:|
|  10  | GDP-Zero          | 0.580 | 7.31 |
|  10  | GDP-Zero + top-K  | 0.580 | 7.28 |
|  10  | **EmoMCTS + top-K** | 0.580 | 7.25 |
|  20  | GDP-Zero          | 0.530 | 7.43 |
|  20  | GDP-Zero + top-K  | 0.610 | 7.21 |
|  20  | **EmoMCTS + top-K** | **0.640** | 6.91 |
|  50  | GDP-Zero          | 0.560 | 7.40 |
|  50  | GDP-Zero + top-K  | 0.594 | 7.36 |
|  50  | **EmoMCTS + top-K** | **0.700** | 6.72 |

At a low simulation budget (10 sims) all methods sit at the same success floor; the emotion
channel separates from the baselines as the budget grows, with EmoMCTS reaching the goal more
often **and** in fewer turns at 20–50 sims. (`EmoMCTS + top-K` uses `β = 0.7`, `K = 5`.)

> **Note.** All results in this table were produced with `--search_horizon legacy`, for both GDP-Zero
> and EmoMCTS: simulated branches were not stopped at the turn limit. The pilot has since run
> ([`analysis/thu/p1_pilot.md`](analysis/thu/p1_pilot.md)): on 10 paired dialogues `episode` succeeded on
> 10/10 against 6/10 for `legacy`, at 42 % lower wall clock, and it is now the frozen default. The
> legacy horizon survives as a disclosed ablation (B2), re-measured at n = 100.

## Repository layout

```
src/
  games/        DialogGame + PersuasionGame (p4g) / EmotionalSupportGame / CBGame
  players/      system/user agents + planners per task
  mcts/         mcts.py            MCTS / OpenLoopMCTS (GDP-Zero base)
                emotion_mcts.py    EmotionAwareOpenLoopMCTS + EmotionAwareMultiObjectiveQ
                                   (the published Double-Q) + the mined valence map
  emotion_classifiers/  hf_emotion.py (encoder) · llm_emotion.py (prompt-based)
  utils/        gen_models (OpenAI/Azure/HF/Ollama), sessions, rewards, prompts, loaders
  runners/      gdpzero.py / emomcts.py    turn-by-turn response comparison
                rollout.py                 self-play episodes (--algo llm_raw|gdpzero|emomcts)
                _common.py                 task registry + dataset readers
  metrics/      dialog_metrics.py + run_metrics.py   SR / AT (/ SL)
  evaluators/   resp_ranker + {p4g,esc,cb}_evaluator + run_judge.py   pairwise LLM judge
  emotion_mining/  mine_emotion_donation_p4g.py   mine w(e) from the corpus
                   mining_corpus.py               corpus readers, splits, turn weighting
                   mine_emotion_da_bonus_p4g.py · learn_emotion_transition_p4g.py
scripts/
  run_sweep_experiments.sh       Grid A: 3 models x 2 persona x 3 methods factorial (SR/AT)
  gridA_report.py                Grid A analysis -> gridA/RESULTS.md (SR, AvgT, delta, McNemar)
  gridA_cache_hit.py             realization-cache hit rate (Grid A stop condition)
  sweep_judge.sh                 sweep + LLM-judge comparison (vs human / raw / gdpzero)
  plot_da_histogram.py           per-turn dialogue-act distribution figure
  plot_emotion_conditioned_actions.py   action choice vs. user emotion figure
data/
  p4g/  300_dialog_turn_based.pkl · p4g-valid.txt
        rollout_evalset_nonannotated.jsonl   the 100 eval dialogues (positions 1-100 of the pool)
  p4g_personas/  full_dialog.csv · full_info.csv   real persuadee surveys (--p4g_persona)
  esc/  esc-{train,valid,test}.txt   ·   cb/  cb-{train,valid,test}.txt
tests/        bit-identity + arm acceptance (test_wed_arms, test_thu_arms, test_search_horizon,
              test_emo_channel_freeze, test_cache_fixes) and the stub end-to-end regression driver
analysis/     see analysis/README.md -- what is current, what is the record
PREREG.md     the grid's pre-registration (append-only, timestamped entries)
```

Each task's `*Game` / `*SystemPlanner` / `*Model` triple exposes a common API
(`get_dialog_ended`, `get_next_state`, `predict`, `get_valid_moves`,
`get_utterance[_w_da]`, …) so the planner and MCTS code is task-agnostic.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# OpenAI (optional — only if you use an OpenAI/Azure backbone or judge)
export OPENAI_API_KEY=sk-...
# Azure OpenAI (only for --llm chatgpt)
export MS_OPENAI_API_KEY=... MS_OPENAI_API_BASE="https://...openai.azure.com"
export MS_OPENAI_API_VERSION=... MS_OPENAI_API_CHAT_VERSION=...

# Open-source backbone. All 2026-09 runs use SGLang serving an AWQ checkpoint:
#   analysis/calib/serve.sh 13b     -> TheBloke/vicuna-13B-v1.5-AWQ on 127.0.0.1:30000
#   analysis/wed/scripts/serve_supervised.sh 13b   (same, under a restart loop)
# then pass --llm sglang --sglang_model TheBloke/vicuna-13B-v1.5-AWQ
#
# Ollama is still supported for a quick local run (--llm ollama --ollama_model vicuna:13b):
ollama serve && ollama pull vicuna:13b
```

`torch` / `transformers` are needed for the HF emotion classifier and the local-HF
backend; `requests` covers OpenAI and Ollama. Modules use absolute imports rooted at
`src/` — run from `src/` or set `PYTHONPATH=$PWD/src`. Entry-point scripts self-bootstrap
`src/` and resolve relative `--data` / `--output` paths against the repo root.

## Data

The repo ships pre-converted splits; `--data` defaults to the validation file of the
selected `--game`.

| Task  | File(s)                                | Format                                                                |
|-------|----------------------------------------|-----------------------------------------------------------------------|
| `p4g` | `data/p4g/300_dialog_turn_based.pkl`   | GDP-Zero pickle: `{did: {dialog:[{er,ee}], label:[{er,ee}]}}`         |
| `p4g` | `data/p4g/p4g-valid.txt`               | JSON-lines `{id, dialog:[{speaker,text,strategy}]}` (from the converter) |
| `p4g` | `data/p4g/rollout_evalset_nonannotated.jsonl` | the **evaluation set**: positions 1–100 of the non-annotated pool. Disjoint by construction from the annotated 300 (which the mining and corpus analyses use) and from the pilot / saturation sets; asserted in `src/emotion_mining/build_p4g_rollout_evalset.py` |
| `esc` | `data/esc/esc-{train,valid,test}.txt`  | DPDP JSON-lines                                                        |
| `cb`  | `data/cb/cb-{train,valid,test}.txt`    | DPDP JSON-lines                                                        |

Regenerate the JSON-lines P4G file with `python src/utils/convert_p4g_to_jsonl.py`.
Dataset readers and the task registry live in `runners/_common.py`. A Hugging Face Hub
loader is available via `--data hf:<repo>[:<config>[:<split>]]` (`pip install datasets`).

## Running EmoMCTS

EmoMCTS runs on the emotion-aware task `emo_p4g`. The published runner exposes exactly
two emotion-relevant knobs: **`--beta_emo`** (the emotion-channel weight) and
**`--llm_prior_topk`** (top-`K` pruning), plus the emotion classifier choice.

```bash
cd src

# EmoMCTS (Double-Q), Vicuna-13B backbone
python runners/emomcts.py --game emo_p4g \
       --llm ollama --ollama_model vicuna:13b \
       --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 \
       --num_mcts_sims 50 --num_dialogs 50 \
       --output outputs/emomcts_p4g.pkl

# GDP-Zero baseline (β = 0, same backbone)
python runners/gdpzero.py --game p4g \
       --llm ollama --ollama_model vicuna:13b \
       --num_mcts_sims 50 --num_dialogs 50 --llm_prior_topk 5 \
       --output outputs/gdpzero_p4g.pkl
```

Both runners write the same per-turn pickle schema, so `run_judge.py --h2h` can compare
them directly (see [LLM judge](#pairwise-llm-judge)). Both also accept
`--search_horizon {legacy,episode}` (see [Method](#method)). `--max_turns` (default 10) is the episode
horizon for **every** runner: it bounds the rollout loop and is passed to the game, so the replay
runners no longer inherit the old `max_conv_turns = 15` default.

## Self-play metrics — SR / AT

`runners/rollout.py` plays *full* self-play episodes (system policy ↔ user simulator
until the goal or `--max_turns`) and writes one record per dialog
(`{did, task, algo, success, num_turns, history}`). The action selector is pluggable via
`--algo`:

| `--algo`  | Planner                                                                 |
|-----------|-------------------------------------------------------------------------|
| `llm_raw` | single LLM call per turn (`argmax(planner.predict(state))`)             |
| `gdpzero` | `OpenLoopMCTS` — GDP-Zero open-loop search                              |
| `emomcts` | `EmotionAwareMultiObjectiveQ` — the Double-Q (`--beta_emo`, `--emotion_classifier`) |

```bash
cd src
python runners/rollout.py --game p4g     --algo gdpzero \
       --llm ollama --ollama_model vicuna:13b \
       --num_mcts_sims 50 --max_conv 100 --llm_prior_topk 5 \
       --output outputs/rollout_gdpzero_p4g.pkl

python runners/rollout.py --game emo_p4g --algo emomcts \
       --llm ollama --ollama_model vicuna:13b \
       --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 \
       --num_mcts_sims 50 --max_conv 100 \
       --output outputs/rollout_emomcts_p4g.pkl

# same, with search bounded at the episode horizon (--max_turns, default 10)
python runners/rollout.py --game emo_p4g --algo emomcts \
       --llm ollama --ollama_model vicuna:13b \
       --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 \
       --num_mcts_sims 50 --max_conv 100 --search_horizon episode \
       --output outputs/rollout_emomcts_p4g_episode.pkl

python metrics/run_metrics.py --episodes outputs/rollout_emomcts_p4g.pkl --max_turns 10
```

`--search_horizon` applies to `--algo gdpzero` and `--algo emomcts`; `llm_raw` builds no tree.
The value is recorded in the run's `metadata.json` (`mcts_args.search_horizon`). The default,
`legacy`, is exactly the pre-flag search rule: verified bit-identical on the deterministic stub
backbone (`tests/run_e2e_regression.py`). Under `episode`, search never simulates past
`--max_turns` (`tests/test_search_horizon.py`).

`rollout.py` prints a cumulative SR / AT summary every 10 dialogs and a final summary.

| metric                  | meaning                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------|
| **SR** — Success Rate   | fraction of episodes reaching the goal within `--max_turns`                               |
| **AT** — Average Turn   | mean #turns; failed / over-limit count as `--max_turns` (PPDPP convention)                |
| **SL** — Sale-to-List   | CraigslistBargain only — `(deal − seller_list) / (buyer_target − seller_list)`, clipped   |

Implementations: `metrics/dialog_metrics.py`.

## Pairwise LLM judge

`evaluators/run_judge.py` reads per-turn pickles and asks an LLM judge which response
wins (A/B-swapped to debias, majority vote over `n` samples):

- **vs. human** (default): `-f`'s response vs. the human reference.
- **head-to-head** (`--h2h <other.pkl>`): `-f`'s response vs. `--h2h`'s response.

A "win" always means the `-f` model won.

```bash
cd src
# EmoMCTS vs human
python evaluators/run_judge.py --task p4g --judge gpt-3.5-turbo \
       -f outputs/emomcts_p4g.pkl --out_json outputs/emomcts_vs_human.json
# EmoMCTS vs GDP-Zero (head-to-head)
python evaluators/run_judge.py --task p4g --judge gpt-3.5-turbo \
       -f outputs/emomcts_p4g.pkl --h2h outputs/gdpzero_p4g.pkl \
       --output outputs/emomcts_vs_gdpzero.pkl --out_json outputs/emomcts_vs_gdpzero.json
```

Output: a pickle with per-record decisions plus a printed `{win, draw, lose, n, win_rate}`
summary (`--out_json` dumps the summary; `--limit N` caps records;
`--judge ollama` runs fully offline).

## Reproducing the paper

**The frozen grid.** Every run is generated from one template and validated before it starts:

```bash
# regenerate the 37 grid configs (each is parsed by the real runner argparse and checked
# against its own --frozen_config JSON; a hand-edited config is refused, not silently run)
python analysis/thu/scripts/gen_grid_configs.py --p4g_success tag --out analysis/grid/configs
bash analysis/calib/serve.sh 13b &          # SGLang, vicuna-13B-AWQ
bash analysis/grid/configs/commands.sh      # or run individual blocks; see plan_4c_run_table.md
```

Frozen values, and the evidence for each, are in [`analysis/FREEZE_NOTES.md`](analysis/FREEZE_NOTES.md) §10.
The run order, budgets and the optional blocks are in
[`analysis/thu/plan_4c_run_table.md`](analysis/thu/plan_4c_run_table.md).

**Earlier pipeline** (pre-freeze; kept because the mined tables and the Grid A sweep are still
referenced):

```bash
# 1. Mine the emotion-valence map from the corpus (writes outputs/emotion_donation_analysis.json)
python src/emotion_mining/mine_emotion_donation_p4g.py

# 2. Grid A -- the factorial evaluation grid (3 AWQ models x 2 persona conditions x 3 methods,
#    n_sims=50, 100 dialogues each; 3 seeds on the Vicuna / no-persona anchor cell).
#    One SGLang server serves one model, so run it once per backbone:
MODEL=TheBloke/vicuna-13B-v1.5-AWQ scripts/serve_sglang.sh &      # terminal 1
MODELS=vicuna scripts/run_sweep_experiments.sh                     # terminal 2
python3 scripts/gridA_report.py                                    # -> gridA/RESULTS.md

# 3. Sweep + LLM-judge comparison (emomcts vs human / raw / gdpzero), judged with gpt-3.5
scripts/sweep_judge.sh

# 4. Policy-behaviour figures
python scripts/plot_da_histogram.py                  # dialogue-act distribution by turn
python scripts/plot_emotion_conditioned_actions.py   # action choice vs. user emotion
```

Each script is env-overridable (e.g. `NUM_DIALOGS=20 STAGES=a1 scripts/run_sweep_experiments.sh`),
though Grid A's *frozen* parameters are guarded: changing one mid-grid aborts the sweep rather
than mixing two batches. Grid A runs land under `gridA/runs/<run-id>/`, everything else under
`src/outputs/<run-id>/`, each with a `metadata.json` snapshot of the exact arguments, so any run
is reproducible from its directory.

## Interactive demo

Converse with the planner; you play the user.

```bash
cd src
python interactive/emcts/interactive.py --game p4g --algo raw-prompt
python interactive/gdpzero/interactive.py --algo raw-prompt        # P4G-only, GDP-Zero-faithful
```

Type `q` to quit, `r` to restart; `-h` lists all flags. Use `--llm ollama
--ollama_model vicuna:13b` for the local backbone.

## Acknowledgements

- GDP-Zero — Yu et al., *Prompt-Based MCTS for Goal-Oriented Dialogue Policy Planning*, EMNLP 2023 ([paper](https://arxiv.org/abs/2305.13660)).
- PPDPP — Deng et al., *Plug-and-Play Policy Planner for LLM Dialogue Agents*, ICLR 2024 ([paper](https://arxiv.org/abs/2311.00262)).
- Datasets: PersuasionForGood, ESConv, CraigslistBargain.
- Emotion classifier: `j-hartmann/emotion-english-distilroberta-base`.
