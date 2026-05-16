# EMCTS — Prompt-based MCTS Dialogue Policy Planning

Monte-Carlo Tree Search over LLM-prompted dialogue simulations for goal-oriented
dialogue policy planning, applied to three tasks:

| Task  | Game class             | System / User         | Goal                                                  |
|-------|------------------------|-----------------------|-------------------------------------------------------|
| `p4g` | `PersuasionGame`       | Persuader / Persuadee | persuade the user to donate to *Save the Children*    |
| `esc` | `EmotionalSupportGame` | Therapist / Patient   | reduce the user's emotional distress                  |
| `cb`  | `CBGame`               | Buyer / Seller        | close a deal at a favourable price (CraigslistBargain) |

Builds on [GDP-Zero](https://github.com/jasonyux/GDPZero) (EMNLP 2023) and borrows
task / reward setups from [PPDPP](https://github.com/dengyang17/PPDPP). Dialogue
simulators are prompted LLMs — OpenAI, Azure OpenAI, local 🤗 Transformers, or
local [Ollama](https://ollama.com).

**Contents:** [Layout](#repository-layout) · [Setup](#setup) ·
[Data](#data) · [Interactive demo](#interactive-demo) ·
[Offline runners](#offline-evaluation-runners) ·
[Metrics — SR / AT / SL](#metrics-sr--at--sl) ·
[LLM judge (pairwise)](#pairwise-llm-judge) ·
[Status](#status--known-limitations)

## Repository layout

```
src/
  games/        DialogGame + PersuasionGame / EmotionalSupportGame / CBGame
  players/      system/user agents + planners per task (p4g / esc / cb)
  mcts/         MCTS, OpenLoopMCTS, OpenLoopMCTSParallel; emotion_mcts.py
  utils/        gen_models (OpenAI/Azure/HF/Ollama), sessions, rewards, prompts
  interactive/  interactive.py — interactive demo
  runners/      raw_prompting.py / gdpzero*.py  turn-by-turn response comparison
                rollout.py                       self-play episodes -> metrics records
                convert_p4g_to_jsonl.py          P4G pickle -> ESC/CB-style JSON-lines
                _common.py                       task registry + dataset readers
                hf_loaders.py                    Hugging Face Hub dataset loaders
  metrics/      dialog_metrics.py + run_metrics.py   SR / AT / SL
  evaluators/   resp_ranker + {p4g,esc,cb}_evaluator   pairwise LLM rankers
                run_judge.py                     CLI: vs-human or head-to-head judge
data/
  p4g/  300_dialog_turn_based.pkl · p4g-valid.txt (JSON-lines, generated)
  esc/  esc-{train,valid,test}.txt
  cb/   cb-{train,valid,test}.tx
```

Each task's `*Game` / `*SystemPlanner` / `*Model` triple exposes a common API
(`get_game_ontology`, `get_dialog_ended`, `get_next_state`, `predict`,
`get_valid_moves`, `get_utterance[_w_da]`, …) so the planner and MCTS code is
task-agnostic.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

`torch` / `transformers` are only needed for the local-HF backend; `requests`
covers OpenAI and Ollama. Pin the `openai` package per `requirements.txt`.

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
```

Modules use absolute imports rooted at `src/`. Run from `src/` or set
`PYTHONPATH=$PWD/src`; the entry-point scripts (`interactive.py`, the runners,
`run_metrics.py`, `run_judge.py`, `convert_p4g_to_jsonl.py`) also self-bootstrap
`src/` and resolve relative paths against the repo root.

## Data

The repo ships pre-converted validation/train/test splits:

| Task  | File(s)                                       | Source / format                                                                                |
|-------|-----------------------------------------------|------------------------------------------------------------------------------------------------|
| `p4g` | `data/p4g/300_dialog_turn_based.pkl`           | GDP-Zero pickle: `{did: {dialog:[{er,ee}], label:[{er,ee}]}}`                                  |
| `p4g` | `data/p4g/p4g-valid.txt`                      | JSON-lines, one `{id, dialog:[{speaker,text,strategy}]}` per dialog — produced by the converter |
| `esc` | `data/esc/esc-{train,valid,test}.txt`         | JSON-lines: `{emotion_type, problem_type, situation, dialog:[{text,speaker,strategy?}]}`       |
| `cb`  | `data/cb/cb-{train,valid,test}.txt`           | JSON-lines: `{item_name, buyer_*, seller_*, dialog:[{text,speaker,strategy}]}`             |

`read_p4g` accepts either format (auto-detected by suffix). Re-create the
JSON-lines P4G file at any time:

The dataset readers and task registry live in `runners/_common.py`
(`TASKS`, `read_p4g/esc/cb`).

### Hugging Face Hub

`--data hf:<repo>[:<split>]` loads a dataset straight from the Hub
(`pip install datasets`):

```bash
python runners/raw_prompting.py --game p4g --data hf:spawn99/PersuasionForGood:FullDialog
python runners/raw_prompting.py --game esc --data hf:thu-coai/esconv:validation
python runners/raw_prompting.py --game cb  --data hf:stanfordnlp/craigslist_bargains:validation
```

`spawn99/PersuasionForGood` carries no DA labels so `sys_da` defaults to
`"other"` and `usr_da` to `U_Neutral`. ESConv and CB preserve their strategy
labels.

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
| `--zero_shot {0,1}`                                             | `1`              | `1`: user = user-LLM + planner heuristic · `0`: user = `get_utterance_w_da`   |
| `--num_mcts_sims`, `--max_realizations`, `--Q_0`                | `10`, `3`, `0.25`| MCTS hyper-parameters (used by `gdpzero`)                                     |
| `--emotion_type`, `--problem_type`                              | `anxiety`, `job crisis` | ESConv scenario (`esc` only)                                            |
| `--cb_item`, `--cb_buyer_desc`, `--cb_buyer_price`, `--cb_seller_desc`, `--cb_seller_price` | bike listing | CraigslistBargain scenario (`cb` only)                       |

### Local model via Ollama

Pass `--llm ollama` to talk to a local [Ollama](https://ollama.com) server
instead of OpenAI:

```bash
ollama serve && ollama pull llama3.1
python interactive/interactive.py --game cb --algo raw-prompt --llm ollama --ollama_model llama3.1
# remote server: --ollama_host http://host:11434 (or $OLLAMA_HOST)
```

`utils/gen_models.py` provides `OllamaModel` (`/api/generate`) and
`OllamaChatModel` (`/api/chat`); both accept the standard kwargs (`max_new_tokens`,
`temperature`, `do_sample`, `repetition_penalty`, `stop`, …) translated to
Ollama's `options`. Multiple samples come from repeated calls.

## Offline evaluation (runners)

`src/runners/` replays a dataset; at every turn the planner picks the next
system DA + utterance, recorded alongside the ground-truth response in a pickle
for later scoring.

| runner                          | planner                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| `runners/raw_prompting.py`      | greedy one-step (chat planner prior → argmax)                            |
| `runners/gdpzero.py`            | open-loop MCTS + realization selection                                   |
| `runners/gdpzero_noopenloop.py` | closed-loop MCTS                                                         |
| `runners/gdpzero_noRS.py`       | open-loop MCTS, no realization selection (re-generates the utterance)    |

```bash
cd src
python runners/raw_prompting.py --game p4g --data data/p4g/p4g-valid.txt --num_dialogs 20
python runners/raw_prompting.py --game esc --data data/esc/esc-valid.txt
python runners/raw_prompting.py --game cb  --data data/cb/cb-valid.txt --llm ollama --ollama_model llama3.1
python runners/gdpzero.py       --game p4g --data data/p4g/300_dialog_turn_based.pkl --num_mcts_sims 20
```

This is *turn-by-turn response comparison*. For *episode-level* SR / AT / SL,
see [Metrics](#metrics-sr--at--sl) below.

## Metrics (SR / AT / SL)

`runners/rollout.py` plays *full* self-play episodes (planner ↔ user simulator,
until the game ends or `--max_turns`) and writes one episode record per dialog:
`{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}`.
`metrics/run_metrics.py` scores them:

```bash
cd src
python runners/rollout.py     --game cb  --data data/cb/cb-valid.txt   --max_turns 8 --output outputs/rollout_cb.pkl
python runners/rollout.py     --game esc --data data/esc/esc-valid.txt --max_turns 8 --output outputs/rollout_esc.pkl
python metrics/run_metrics.py --episodes outputs/rollout_cb.pkl --task cb --max_turns 8
```

| metric                        | meaning                                                                                                                                                                                |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SR** — Success Rate         | fraction of episodes reaching the goal (donate / solved / deal) within `--max_turns`                                                                                                   |
| **AT** — Average Turn         | mean #turns; failed / over-limit count as `--max_turns` (PPDPP convention; `--at_successes_only` averages successes only)                                                              |
| **SL** — Sale-to-List (CB)    | `(deal − seller_list) / (buyer_target − seller_list)`, clipped to `[0,1]`; **higher = better deal**; failed negotiations get `0`                                                       |

Implementations: `metrics/dialog_metrics.py` (`success_rate`, `average_turn`,
`sale_to_list_ratio`, `compute_metrics`). The CB deal price is best-effort
extracted (last number mentioned). Rollouts currently use the greedy planner;
an MCTS rollout needs the `mcts/mcts.py` ↔ game `get_next_state` reconciliation
(see [Status](#status--known-limitations)).

## Pairwise LLM judge

`evaluators/` ships LLM-based pairwise rankers — given a context and two
responses, the judge picks A / B / can't-tell (A/B-swapped to debias,
majority-vote over `n` samples). One per task (`P4GEvaluator`,
`ESCEvaluator`, `CBEvaluator`); shared in `resp_ranker.py`.

`evaluators/run_judge.py` is the CLI. It reads the per-turn pickles written by
the offline runners and asks the judge which response wins:

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

- `raw-prompt` paths work for all three games (`interactive.py --algo raw-prompt`, `runners/raw_prompting.py`).
- MCTS paths (`--algo gdpzero`, `runners/gdpzero*.py`) are wired but not yet runnable end-to-end: `mcts/mcts.py` calls `game.get_next_state(state, action)` with two positional args and `get_next_state_batched(...)`, which the games don't implement. Reconciling those is the next step (and also unlocks an MCTS rollout for SR / AT).
- `players/p4g_players.py` is a near-direct port of GDP-Zero's `core/players.py`; some dead/commented blocks remain.
- `utils/gen_models.py` mixes pre-1.0 and ≥1.0 `openai` SDK call styles — pin the version that matches your backend.
- No training entry point (policy-network fine-tuning) yet.

## Acknowledgements

- GDP-Zero — Yu et al., *Prompt-Based MCTS for Goal-Oriented Dialogue Policy Planning*, EMNLP 2023 ([paper](https://arxiv.org/abs/2305.13660)).
- PPDPP — Deng et al., *Plug-and-Play Policy Planner for LLM Dialogue Agents* ([paper](https://arxiv.org/abs/2311.00262)).
- Datasets: PersuasionForGood, ESConv, CraigslistBargain.
