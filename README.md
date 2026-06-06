# EmoMCTS — Emotion-Aware Monte-Carlo Tree Search for Dialogue Policy Planning

EmoMCTS plans goal-oriented dialogue with **open-loop Monte-Carlo Tree Search over
LLM-prompted simulations**, and makes the user's **emotion a first-class search
coordinate**. Alongside the usual task-value estimate, it maintains a *parallel*
action-value channel that tracks the expected **emotional valence** of the user's
reaction, and folds it into the PUCT selection rule through a single weight `β`.

The method is applied to **PersuasionForGood** (persuade a user to donate to *Save the
Children*).

Dialogue simulators are prompted LLMs — OpenAI, Azure OpenAI, local 🤗 Transformers, or
local [Ollama](https://ollama.com). All reported results use an **open-source
Vicuna-13B** backbone (via Ollama) so they are fully reproducible without a proprietary
API.

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

The valence weights `w(e)` per emotion are **mined from the PersuasionForGood corpus**,
not hand-set: `w(e) ∝ P(donate | user emotion = e) − base_rate`
(`scripts/mine_emotion_donation_p4g.py`). The mined weights overturn naive affective
valence — *fear* is the strongest positive predictor of donation, while *neutral*
(apathy) is the main negative signal.

Two further levers, shared with the GDP-Zero baseline:

- **Top-`K` prior pruning** (`--llm_prior_topk`): the search is hard-pruned to the `K`
  highest-prior dialogue acts per node, concentrating the simulation budget.
- **Emotion classifier** (`--emotion_classifier hf`): a deterministic encoder
  (`j-hartmann/emotion-english-distilroberta-base`) labels each user reaction — no LLM
  cost, fully reproducible.

## Results

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
scripts/
  mine_emotion_donation_p4g.py   mine the emotion-valence map from the corpus
  run_sweep_experiments.sh       simulation-budget sweep (SR/AT)
  sweep_judge.sh                 sweep + LLM-judge comparison (vs human / raw / gdpzero)
  plot_da_histogram.py           per-turn dialogue-act distribution figure
  plot_emotion_conditioned_actions.py   action choice vs. user emotion figure
data/
  p4g/  300_dialog_turn_based.pkl · p4g-valid.txt
  esc/  esc-{train,valid,test}.txt   ·   cb/  cb-{train,valid,test}.txt
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

# Open-source backbone used for all reported results
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
them directly (see [LLM judge](#pairwise-llm-judge)).

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

python metrics/run_metrics.py --episodes outputs/rollout_emomcts_p4g.pkl --max_turns 10
```

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

```bash
# 1. Mine the emotion-valence map from the corpus (writes outputs/emotion_donation_analysis.json)
python scripts/mine_emotion_donation_p4g.py

# 2. Simulation-budget sweep: gdpzero / gdpzero+topk / emomcts across num_sims (SR / AT)
scripts/run_sweep_experiments.sh

# 3. Sweep + LLM-judge comparison (emomcts vs human / raw / gdpzero), judged with gpt-3.5
scripts/sweep_judge.sh

# 4. Policy-behaviour figures
python scripts/plot_da_histogram.py                  # dialogue-act distribution by turn
python scripts/plot_emotion_conditioned_actions.py   # action choice vs. user emotion
```

Each script is env-overridable (e.g. `SIMS="20 50" NUM_DIALOGS=100 scripts/run_sweep_experiments.sh`).
Per-run outputs land under `src/outputs/<run-id>/` with a `metadata.json` snapshot of the
exact arguments, so any run is reproducible from its directory.

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
