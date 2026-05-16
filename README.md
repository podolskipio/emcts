# EMCTS — Prompt-based MCTS Dialogue Policy Planning

Monte-Carlo Tree Search over LLM-prompted dialogue simulations for goal-oriented
dialogue policy planning, applied to three tasks:

| Task  | Game class             | System / User         | Goal                                                  |
|-------|------------------------|-----------------------|-------------------------------------------------------|
| `p4g` | `PersuasionGame`       | Persuader / Persuadee | persuade the user to donate to *Save the Children*    |
| `esc` | `EmotionalSupportGame` | Therapist / Patient   | reduce the user's emotional distress                  |
| `cb`  | `CBGame`               | Buyer / Seller        | close a deal at a favourable price (CraigslistBargain) |

Builds on [GDP-Zero](https://github.com/jasonyux/GDPZero) (EMNLP 2023) and borrows
task / reward setups from [PPDPP](https://github.com/dengyang17/PPDPP). The runner
defaults are wired to be **GDPZero-faithful** so SR / AT / SL numbers are directly
comparable to that paper. Dialogue simulators are prompted LLMs — OpenAI, Azure
OpenAI, local 🤗 Transformers, or local [Ollama](https://ollama.com).

**Contents:** [Layout](#repository-layout) · [Setup](#setup) ·
[Data](#data) · [Interactive demo](#interactive-demo) ·
[Offline runners](#offline-evaluation-runners) ·
[Self-play & SR/AT/SL](#self-play-metrics--sr--at--sl) ·
[LLM judge](#pairwise-llm-judge) · [Status](#status--known-limitations)

## Repository layout

```
src/
  games/        DialogGame + PersuasionGame / EmotionalSupportGame / CBGame
  players/      system/user agents + planners per task (p4g / esc / cb)
  mcts/         MCTS, OpenLoopMCTS, OpenLoopMCTSParallel; emotion_mcts.py
  utils/        gen_models (OpenAI/Azure/HF/Ollama), sessions, rewards, prompts,
                hf_loaders.py             Hugging Face Hub dataset loaders
                convert_p4g_to_jsonl.py   P4G pickle -> ESC/CB-style JSON-lines
  interactive/  interactive.py — interactive demo
  runners/      raw_prompting.py / gdpzero*.py  turn-by-turn response comparison
                rollout.py                       self-play episodes (pluggable --algo)
                _common.py                       task registry + dataset readers
  metrics/      dialog_metrics.py + run_metrics.py   SR / AT / SL
  evaluators/   resp_ranker + {p4g,esc,cb}_evaluator   pairwise LLM rankers
                run_judge.py              CLI: vs-human or head-to-head judge
data/
  p4g/  300_dialog_turn_based.pkl · p4g-valid.txt (JSON-lines, generated)
  esc/  esc-{train,valid,test}.txt
  cb/   cb-{train,valid,test}.txt
```

Each task's `*Game` / `*SystemPlanner` / `*Model` triple exposes a common API
(`get_game_ontology`, `get_dialog_ended`, `get_next_state`, `predict`,
`get_valid_moves`, `get_utterance[_w_da]`, …) so the planner and MCTS code is
task-agnostic.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# OpenAI
export OPENAI_API_KEY=sk-...
# Azure OpenAI (only for --llm chatgpt)
export MS_OPENAI_API_KEY=... MS_OPENAI_API_BASE="https://...openai.azure.com"
export MS_OPENAI_API_VERSION=... MS_OPENAI_API_CHAT_VERSION=...
```

`torch` / `transformers` are only needed for the local-HF backend; `requests`
covers OpenAI and Ollama. Pin the `openai` package per `requirements.txt`.

Modules use absolute imports rooted at `src/`. Run from `src/` or set
`PYTHONPATH=$PWD/src`; the entry-point scripts also self-bootstrap `src/` and
resolve relative `--data` / `--output` paths against the repo root, so you can
invoke them from any working directory.

## Data

The repo ships pre-converted splits — `--data` defaults to the validation file
of the selected `--game`, so most invocations don't need to pass it explicitly.

| Task  | File(s)                                       | Source / format                                                                                |
|-------|-----------------------------------------------|------------------------------------------------------------------------------------------------|
| `p4g` | `data/p4g/300_dialog_turn_based.pkl`           | GDP-Zero pickle: `{did: {dialog:[{er,ee}], label:[{er,ee}]}}`                                  |
| `p4g` | `data/p4g/p4g-valid.txt`                      | JSON-lines, one `{id, dialog:[{speaker,text,strategy}]}` per dialog — produced by the converter |
| `esc` | `data/esc/esc-{train,valid,test}.txt`         | DPDP JSON-lines: `{emotion_type, problem_type, situation, dialog:[{text,speaker,strategy?}]}`  |
| `cb`  | `data/cb/cb-{train,valid,test}.txt`           | DPDP JSON-lines: `{item_name, buyer_*, seller_*, dialog:[{text,speaker,strategy}]}`            |

`read_p4g` auto-detects pickle vs JSON-lines from the suffix. Regenerate the
JSON-lines P4G file at any time:

```bash
python src/utils/convert_p4g_to_jsonl.py
# --input/--output override; relative paths resolve from the repo root
```

The dataset readers and task registry live in `runners/_common.py`
(`TASKS`, `read_p4g/esc/cb`).

### Hugging Face Hub

`--data hf:<repo>[:<token>[:<split>]]` loads a dataset straight from the Hub
(`pip install datasets`). Each call loads **one split**.

| `--data` argument                                 | What gets loaded                                                                                   |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `hf:repo`                                         | first split of the default config (usually `train`)                                                |
| `hf:repo:<token>`                                 | tries `<token>` as a split first; if not a split, treats it as a **config** and takes its first split |
| `hf:repo:<config>:<split>`                        | explicit                                                                                           |

```bash
python runners/raw_prompting.py --game esc --data hf:thu-coai/esconv:validation
python runners/raw_prompting.py --game cb  --data hf:stanfordnlp/craigslist_bargains:validation
python runners/raw_prompting.py --game p4g --data hf:spawn99/PersuasionForGood:FullDialog:train
```

(Note `FullDialog` is a **config name**, not a split — three segments needed to
nail it down.) `spawn99/PersuasionForGood` has no DA labels, so the P4G HF loader
defaults `sys_da="other"` and `usr_da=U_Neutral`. ESConv and CB preserve their
strategy labels.

## Interactive demo

Converse with the policy planner; you play the user side.

```bash
cd src
python interactive/interactive.py --game p4g --algo raw-prompt
python interactive/interactive.py --game esc --algo raw-prompt --emotion_type anxiety --problem_type "job crisis"
python interactive/interactive.py --game cb  --algo raw-prompt --cb_buyer_price 80 --cb_seller_price 150
```

Type `q` to quit, `r` to restart. Key flags (`-h` for the full list):

| flag                                                            | default          | meaning                                                                       |
|-----------------------------------------------------------------|------------------|-------------------------------------------------------------------------------|
| `--game {p4g,esc,cb}`                                           | `p4g`            | which dialog game                                                             |
| `--algo {gdpzero,raw-prompt}`                                   | `gdpzero`        | planning algorithm                                                            |
| `--llm {gpt-3.5-turbo,chatgpt,code-davinci-002,text-davinci-002,ollama}` | `gpt-3.5-turbo` | backbone LLM                                                          |
| `--ollama_model`, `--ollama_host`                               | `llama3.1`, env  | local model / server URL when `--llm ollama`                                  |
| `--zero_shot {0,1}`                                             | `1`              | interactive-only — user DA from planner heuristic (`1`) vs `get_utterance_w_da` (`0`); the runners hardcode `0` |
| `--num_mcts_sims`, `--max_realizations`, `--Q_0`                | `10`, `3`, `0.25`| MCTS hyper-parameters (used by `gdpzero`)                                     |
| `--emotion_type`, `--problem_type`                              | `anxiety`, `job crisis` | ESConv scenario (`esc` only)                                            |
| `--cb_item`, `--cb_buyer_desc`, `--cb_buyer_price`, `--cb_seller_desc`, `--cb_seller_price` | bike listing | CraigslistBargain scenario (`cb` only)                       |

### Local model via Ollama

Pass `--llm ollama` to talk to a local [Ollama](https://ollama.com) server
instead of OpenAI:

```bash
ollama serve && ollama pull llama3.1
python interactive/interactive.py --game cb --algo raw-prompt --llm ollama --ollama_model llama3.1
# remote: --ollama_host http://host:11434 (or $OLLAMA_HOST)
```

`utils/gen_models.py` provides `OllamaModel` (`/api/generate`) and
`OllamaChatModel` (`/api/chat`); both accept the standard kwargs
(`max_new_tokens`, `temperature`, `do_sample`, `repetition_penalty`, `stop`, …)
translated to Ollama's `options`. Multiple samples come from repeated calls.

## Offline evaluation (runners)

`src/runners/` replays a held-out dataset; at every turn the planner picks the
next system DA + utterance, saved alongside the ground-truth response in a pickle
for later judging.

| runner                          | planner                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| `runners/raw_prompting.py`      | greedy one-step (chat planner prior → argmax)                            |
| `runners/gdpzero.py`            | open-loop MCTS + realization selection                                   |
| `runners/gdpzero_noopenloop.py` | closed-loop MCTS                                                         |
| `runners/gdpzero_noRS.py`       | open-loop MCTS, no realization selection                                 |

```bash
cd src
python runners/raw_prompting.py --game p4g                                       # uses data/p4g/p4g-valid.txt by default
python runners/raw_prompting.py --game esc
python runners/raw_prompting.py --game cb --llm ollama --ollama_model llama3.1
python runners/gdpzero.py       --game p4g --num_mcts_sims 20
```

The runners are **GDPZero-faithful**: the user agent emits its own DA via
`get_utterance_w_da` (no planner-heuristic shortcut), so `get_dialog_ended` reads
DAs the simulator itself produced — directly comparable to GDPZero's
`core/game.py`. There is no `--zero_shot` flag; the parameter still exists on
every `*Model` / game / planner constructor (`build_agents(..., zero_shot=False)`
by default) for in-code experimentation.

This is *turn-by-turn response comparison*. For *episode-level* SR / AT / SL,
see the next section.

## Self-play metrics — SR / AT / SL

`runners/rollout.py` plays *full* self-play episodes (system policy ↔ user
simulator until the game ends or `--max_turns`) and writes one episode record
per dialog: `{did, task, algo, success, num_turns, history, [deal_price, buyer_price, seller_price]}`.
The action-selection step is **pluggable** via `--algo`:

| `--algo`  | Class             | What it does                                                                     |
|-----------|-------------------|----------------------------------------------------------------------------------|
| `llm_raw` | `LLMRawPolicy`    | `argmax(planner.predict(state))` — single LLM call per turn; mirrors `raw_prompting` |
| `gdpzero` | `GDPZeroPolicy`   | `OpenLoopMCTS` per turn → argmax of visit-count policy                            |
| `emomcts` | `EmoMCTSPolicy`   | `EmotionAwareOpenLoopMCTS` — same MCTS loop with an injected `emotion_classifier` |

Adding a new policy: subclass `RolloutPolicy` (any class with
`pick_action(state, *, game, system, planner) -> int` works), register it in
`POLICIES`, extend `add_policy_args` / `make_policy` for any CLI flags —
`rollout_one` and `main` stay untouched.

```bash
cd src
python runners/rollout.py --game cb                                           # llm_raw, all CB dialogs (data/cb/cb-valid.txt)
python runners/rollout.py --game cb  --algo llm_raw --max_conv 10             # first 10 only
python runners/rollout.py --game esc --algo emomcts --num_mcts_sims 20        # emotion-aware MCTS
python metrics/run_metrics.py --episodes outputs/rollout.pkl --max_turns 8
```

| flag                                              | default                  | meaning                                                                    |
|---------------------------------------------------|--------------------------|----------------------------------------------------------------------------|
| `--algo {llm_raw,gdpzero,emomcts}`                | `llm_raw`                | which policy picks each system action                                      |
| `--max_turns`                                     | `10`                     | hard cap on turns per episode                                              |
| `--max_conv N`                                    | all dialogs              | rollout only the first N conversations                                     |
| `--num_mcts_sims`, `--max_realizations`, `--Q_0`, `--cpuct` | `20`, `3`, `0.0`, `1.0` | MCTS hyper-parameters (for `gdpzero` / `emomcts`)                   |

Episode metrics:

| metric                        | meaning                                                                                                                                                                                |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SR** — Success Rate         | fraction of episodes reaching the goal (donate / solved / deal) within `--max_turns`                                                                                                   |
| **AT** — Average Turn         | mean #turns; failed / over-limit count as `--max_turns` (PPDPP convention; `--at_successes_only` averages successes only)                                                              |
| **SL** — Sale-to-List (CB)    | `(deal − seller_list) / (buyer_target − seller_list)`, clipped to `[0,1]`; **higher = better deal**; failed negotiations get `0`                                                       |

Implementations: `metrics/dialog_metrics.py` (`success_rate`, `average_turn`,
`sale_to_list_ratio`, `compute_metrics`). The CB deal price is best-effort
extracted (last number mentioned in the dialog).

## Pairwise LLM judge

`evaluators/` ships LLM-based pairwise rankers — given a context and two
responses, the judge picks A / B / can't-tell (A/B-swapped to debias,
majority-vote over `n` samples). One per task (`P4GEvaluator`, `ESCEvaluator`,
`CBEvaluator`); shared in `resp_ranker.py`.

`evaluators/run_judge.py` is the CLI. It reads per-turn pickles from the offline
runners and asks the judge which response wins:

- **vs. human** (default): A = `ori_resp`, B = `new_resp` (from `-f`).
- **head-to-head** (`--h2h <other.pkl>`): A = `--h2h`'s `new_resp`, B = `-f`'s `new_resp`.

A "win" always means the `-f` model beat the reference.

```bash
cd src
python evaluators/run_judge.py --task p4g -f outputs/gdpzero_p4g.pkl --output outputs/eval_p4g.pkl
python evaluators/run_judge.py --task esc -f outputs/gdpzero_esc.pkl --h2h outputs/raw_esc.pkl --output outputs/h2h_esc.pkl
python evaluators/run_judge.py --task cb  -f outputs/gdpzero_cb.pkl  --judge ollama --ollama_model llama3.1
```

Output: a pickle with per-record decisions (`winner`, `choices`, `rationales`,
`do_swap`, `did`, `context`, `resp_a`, `resp_b`) + a printed
`{win, draw, lose, n, win_rate}` summary. `--out_json` dumps the summary;
`--limit N` caps records.

| flag                                                  | default          | meaning                                                                |
|-------------------------------------------------------|------------------|------------------------------------------------------------------------|
| `--task {p4g,esc,cb}`                                 | *required*       | which task evaluator / prompt to use                                   |
| `-f <pkl>`                                            | *required*       | runner pickle to evaluate — B = its `new_resp`                         |
| `--h2h <pkl>`                                         | (vs. human)      | second runner pickle — A = its `new_resp`; requires `--output`         |
| `--judge {gpt-3.5-turbo,chatgpt,ollama}`              | `gpt-3.5-turbo`  | judge LLM; Ollama works fully offline                                  |
| `--ollama_model`, `--ollama_host`                     | `llama3.1`, env  | judge server when `--judge ollama`                                     |
| `--output`, `--out_json`, `--limit`, `--debug`        | —                | per-record pickle / summary JSON / record cap / verbose ranker logs    |

## Status / known limitations

- `--algo llm_raw` works end-to-end for all three games (CB's `SellerChatModel.get_utterance_w_da` was added so the GDPZero-faithful user-DA path runs).
- `--algo gdpzero` / `emomcts` for `rollout.py` (and `runners/gdpzero*.py`) are wired but not yet runnable end-to-end: `mcts/mcts.py` calls `game.get_next_state(state, action)` with two positional args and `get_next_state_batched(...)`, which the games don't implement. Reconciling those is the next step.
- `emomcts` additionally needs an emotion-aware game producing `EmotionAwareDialogSession` and a real `emotion_classifier` (a stub with Ekman labels is wired by default so the policy *constructs*).
- `players/p4g_players.py` is a near-direct port of GDP-Zero's `core/players.py`; some dead/commented blocks remain.
- `utils/gen_models.py` mixes pre-1.0 and ≥1.0 `openai` SDK call styles — pin the version that matches your backend.
- No training entry point (policy-network fine-tuning) yet.

## Acknowledgements

- GDP-Zero — Yu et al., *Prompt-Based MCTS for Goal-Oriented Dialogue Policy Planning*, EMNLP 2023 ([paper](https://arxiv.org/abs/2305.13660)).
- PPDPP — Deng et al., *Plug-and-Play Policy Planner for LLM Dialogue Agents* ([paper](https://arxiv.org/abs/2311.00262)).
- Datasets: PersuasionForGood, ESConv, CraigslistBargain.
