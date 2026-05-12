# EMCTS — Prompt-based MCTS Dialogue Policy Planning

Monte-Carlo Tree Search over LLM-prompted dialogue simulations for goal-oriented
dialogue policy planning, applied to three tasks:

| Task  | Game class            | System / User       | Goal                                              |
|-------|-----------------------|---------------------|---------------------------------------------------|
| `p4g` | `PersuasionGame`      | Persuader / Persuadee | persuade the user to donate to *Save the Children* |
| `esc` | `EmotionalSupportGame`| Therapist / Patient | reduce the user's emotional distress              |
| `cb`  | `CBGame`              | Buyer / Seller      | close a deal at a favourable price (CraigslistBargain) |

It builds on [GDP-Zero](https://github.com/jasonyux/GDPZero) ("Prompt-Based
Monte-Carlo Tree Search for Goal-Oriented Dialogue Policy Planning", EMNLP 2023)
and borrows task setups / reward shaping from
[PPDPP](https://github.com/dengyang17/PPDPP). The dialogue simulators are
prompted LLMs — OpenAI, Azure OpenAI, a local 🤗 Transformers model, or a local
[Ollama](https://ollama.com) server — so planning speed is bound by whichever
backend you pick.

**Contents:** [Repository layout](#repository-layout) · [Setup](#setup) ·
[Interactive demo](#interactive-demo) · [Local model (Ollama)](#running-against-a-local-model-ollama) ·
[Offline evaluation (runners)](#offline-evaluation-runners) · [Metrics (SR / AT / SL)](#metrics-sr--at--sl) ·
[Pairwise response evaluation](#pairwise-response-evaluation) · [Status / known limitations](#status--known-limitations)

## Repository layout

```
src/
  games/        DialogGame + PersuasionGame / EmotionalSupportGame / CBGame
                (game ontology, terminal conditions, get_next_state simulation)
  players/      Dialog models (system/user agents) and planners per task:
                  p4g_players.py  PersuaderModel / PersuadeeModel / P4G*SystemPlanner
                  esc_players.py  TherapistModel / PatientModel / ESC*SystemPlanner
                  cb_players.py   BuyerModel    / SellerModel  / CB*SystemPlanner
                planner.py        DialogPlanner abstract base
  mcts/         MCTS, OpenLoopMCTS, OpenLoopMCTSParallel (mcts.py); emotion_mcts.py
  utils/        gen_models.py     LLM backends (OpenAI / Azure / local HF / Ollama) + DialogModel base
                sessions.py       DialogSession variants (plain / ESConv / CB)
                rewards.py         reward_dict for esc / cima / p4g heuristic scores
                prompt_examples.py few-shot example dialogs per task
                utils.py           dotdict, seeding, entropy helpers
  interactive/  interactive.py     interactive demo (see below)
  runners/      raw_prompting.py / gdpzero*.py  turn-by-turn response-comparison eval over a dataset;
                rollout.py  self-play episodes -> episode records for metrics;
                _common.py  shared task registry + dataset readers + eval loop
  metrics/      dialog_metrics.py  SR / AT / SL implementations; run_metrics.py  CLI
  evaluators/   resp_ranker.py + {p4g,esc,cb}_evaluator.py  pairwise LLM response rankers
```

The three `*Game` / `*SystemPlanner` / `*Model` families expose a common API, so a
game/planner/model triple from any task is interchangeable in the planner and MCTS
code:

- `DialogGame(system_agent, user_agent, planner, zero_shot, max_conv_turns=15, success_base=0.1)`
  with `get_game_ontology()`, `get_dialog_ended(state) -> float`,
  `map_user_action(v, sampled_das) -> str`,
  `get_next_state(state, action, agent_state=[], mode='train') -> (next_state, next_agent_state, v)`,
  and a task-specific `init_dialog(...)`.
- `DialogPlanner.predict(state[, policy, ent_bound, agent_state]) -> (prob, v)`,
  `heuristic(state) -> (float, sampled_das)`, `get_valid_moves(state) -> np.ndarray`.
- `DialogModel.get_utterance(state, action, mode='train') -> str`,
  `get_utterance_w_da(state, action, mode='train') -> (da, utt)`, plus batched / `predict_da` variants.

## Setup

1. **Python deps**:

   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt')"   # sentence tokenizer used by gen_models
   ```

   (`torch` / `transformers` are only needed for the local-HF backend and the
   policy-network paths; `requests` is what the OpenAI-API and Ollama backends use.
   See the note in `requirements.txt` about the `openai` package version.)

2. **OpenAI / Azure OpenAI credentials** — the LLM backends in `utils/gen_models.py`
   read these from the environment:

   ```bash
   # OpenAI
   export OPENAI_API_KEY=sk-xxxx

   # Azure OpenAI (only if using --llm chatgpt)
   export MS_OPENAI_API_KEY=xxxx
   export MS_OPENAI_API_BASE="https://xxx.openai.azure.com"
   export MS_OPENAI_API_VERSION="xxx"
   export MS_OPENAI_API_CHAT_VERSION="xxx"
   ```

3. **Import path** — modules use absolute imports rooted at `src/`. Either run from
   `src/` or add it to `PYTHONPATH`:

   ```bash
   cd src                 # then `python -m interactive.interactive ...`
   # or
   export PYTHONPATH="$PWD/src"
   ```

   (`interactive/interactive.py` also self-bootstraps `src/` onto `sys.path`, so
   `python src/interactive/interactive.py ...` works from the repo root too.)

## Interactive demo

Converse with the policy planner. You play the **user** side of the game; the system
side's next dialog act + utterance is chosen either by MCTS (`gdpzero`) or by greedy
one-step prompting (`raw-prompt`).

```bash
cd src
python interactive/interactive.py --game p4g --algo raw-prompt
python interactive/interactive.py --game esc --algo raw-prompt --emotion_type anxiety --problem_type "job crisis"
python interactive/interactive.py --game cb  --algo raw-prompt --cb_buyer_price 80 --cb_seller_price 150
```

Type `q` to quit, `r` to restart the conversation.

Key flags (`python interactive/interactive.py -h` for the full list):

| flag | default | meaning |
|------|---------|---------|
| `--game {p4g,esc,cb}` | `p4g` | which dialog game |
| `--algo {gdpzero,raw-prompt}` | `gdpzero` | planning algorithm |
| `--llm {gpt-3.5-turbo,chatgpt,code-davinci-002,text-davinci-002,ollama}` | `gpt-3.5-turbo` | backbone LLM (`ollama` = a local Ollama server, see below) |
| `--ollama_model`, `--ollama_host` | `llama3.1`, `$OLLAMA_HOST` or `http://localhost:11434` | model name / server URL when `--llm ollama` |
| `--zero_shot {0,1}` | `1` | `1`: simulate the user with the user LLM + planner heuristic; `0`: use the user model's `get_utterance_w_da` |
| `--num_mcts_sims`, `--max_realizations`, `--Q_0` | `10`, `3`, `0.25` | MCTS hyper-parameters (used by `gdpzero`) |
| `--emotion_type`, `--problem_type` | `anxiety`, `job crisis` | ESConv scenario (`esc` only) |
| `--cb_item`, `--cb_buyer_desc`, `--cb_buyer_price`, `--cb_seller_desc`, `--cb_seller_price` | bike listing | CraigslistBargain scenario (`cb` only) |

### Running against a local model (Ollama)

Instead of the OpenAI API you can point the backbone at a model served locally by
[Ollama](https://ollama.com) — no API key, no network. Start the server and pull a
model once, then pass `--llm ollama`:

```bash
ollama serve                 # often already running as a background service
ollama pull llama3.1         # or qwen2.5, mistral, phi3, ...

cd src
python interactive/interactive.py --game cb --algo raw-prompt --llm ollama --ollama_model llama3.1
# point at a remote/non-default server with --ollama_host http://host:11434 (or set $OLLAMA_HOST)
```

`utils/gen_models.py` provides `OllamaModel` (completion, `/api/generate`) and
`OllamaChatModel` (chat, `/api/chat`); both accept the same generation kwargs as
the other backends (`max_new_tokens`, `temperature`, `do_sample`,
`num_return_sequences`, `repetition_penalty`, `stop`, …), translated to Ollama's
`options`. Since Ollama has no native `n`, multiple samples are produced by
repeated calls (and a single call, replicated, when `do_sample=False`). It only
needs `requests` (already a dependency).

## Offline evaluation (runners)

`src/runners/` replays a held-out dataset and, at every turn, has the planner pick the next
system dialog act + utterance, recording it next to the ground-truth response in a pickle for
later scoring. All four runners take `--game {p4g,esc,cb}` and `--data <path>`:

| runner | planner |
|--------|---------|
| `runners/raw_prompting.py` | greedy one-step (chat planner prior → argmax) |
| `runners/gdpzero.py` | open-loop MCTS + realization selection |
| `runners/gdpzero_noopenloop.py` | closed-loop MCTS |
| `runners/gdpzero_noRS.py` | open-loop MCTS, no realization selection (re-generates the utterance) |

```bash
cd src
python runners/raw_prompting.py --game p4g --data data/p4g/300_dialog_turn_based.pkl --num_dialogs 20
python runners/raw_prompting.py --game esc --data data/esc/esc-valid.json
python runners/raw_prompting.py --game cb  --data data/cb/cb-valid.txt --llm ollama --ollama_model llama3.1
python runners/gdpzero.py       --game p4g --data data/p4g/300_dialog_turn_based.pkl --num_mcts_sims 20
```

These four runners do *turn-by-turn response comparison* (predicted next response vs. the
dataset's ground-truth one) — for the *episode-level* metrics below see `runners/rollout.py`.
The dataset files are **not** shipped with this repo — supply them via `--data`. Expected
formats: P4G = GDP-Zero's `300_dialog_turn_based.pkl` (`{did: {dialog, label}}`); ESConv =
JSON / JSON-lines of `{emotion_type, problem_type, situation, dialog: [{text, speaker, strategy?}]}`;
CraigslistBargain = JSON-lines of `{item_name, buyer_*, seller_*, dialog: [{text, speaker, strategy}]}`.
The task wiring + dataset readers live in `runners/_common.py` (`TASKS`, `read_p4g/esc/cb`).

## Metrics (SR / AT / SL)

`runners/rollout.py` plays *full* self-play episodes (planner ↔ user simulator, until the game
ends or `--max_turns`) and writes one episode record per dialog
(`{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}`).
`metrics/run_metrics.py` then scores a file of those records:

```bash
cd src
python runners/rollout.py     --game cb  --data data/cb/cb-valid.txt --max_turns 8 --output outputs/rollout_cb.pkl
python metrics/run_metrics.py --episodes outputs/rollout_cb.pkl --task cb --max_turns 8
```

| metric | meaning |
|--------|---------|
| **SR** — Success Rate | fraction of episodes that reached the goal (donate / solved / deal) within the turn limit |
| **AT** — Average Turn | mean #turns; failed / over-limit episodes count as `--max_turns` (PPDPP convention; `--at_successes_only` averages over successes instead) |
| **SL** — Sale-to-List ratio (CB only) | buyer's-perspective negotiation quality, `(deal_price − seller_list_price) / (buyer_target_price − seller_list_price)`, clipped to `[0, 1]`; **higher = better deal**, a failed negotiation gets `0` |

The computation lives in `metrics/dialog_metrics.py` (`success_rate`, `average_turn`,
`sale_to_list_ratio`, `compute_metrics`) and works on any list of episode records of that shape.
The CB deal price is best-effort extracted from the dialog (last number mentioned), and the
greedy planner is used for the rollout — an MCTS rollout needs the `mcts/mcts.py` ↔ game
`get_next_state` reconciliation (see below).

### Pairwise response evaluation

`evaluators/` provides LLM-based **pairwise rankers** — given a dialog `context` and two
candidate system responses, they ask an LLM which one better serves the task goal (A/B-swapped
to debias, majority vote over `n` samples), returning `0` (A better) / `1` (B better) / `2`
(can't tell). One per task: `P4GEvaluator` ("better persuades to donate?"), `ESCEvaluator`
("better supports the patient?"), `CBEvaluator` ("better reaches a lower price while closing?").
Shared logic is in `evaluators/resp_ranker.py` (`RespRanker` / `LLMRespRanker`); pick one with
`evaluators.get_evaluator(task, gen_model)`.

```python
from utils.gen_models import OpenAIChatModel
from evaluators import get_evaluator
ev = get_evaluator("cb", OpenAIChatModel("gpt-3.5-turbo"))
pref, debug = ev.evaluate(dialog_context, response_a, response_b)   # pref in {0, 1, 2}
```

## Status / known limitations

This is an in-progress research codebase. Currently:

- **The `raw-prompt` paths work for all three games** (`interactive.py --algo raw-prompt`, `runners/raw_prompting.py`).
- **The MCTS paths (`--algo gdpzero`, `runners/gdpzero*.py`) are wired to the current APIs but not yet runnable end-to-end**: `mcts/mcts.py` calls `game.get_next_state(state, action)` with only two arguments (the games' `get_next_state` expects `agent_state` to be a list) and `game.get_next_state_batched(...)`, which the game classes don't implement. Reconciling `mcts/mcts.py` with the game API is the next step.
- `players/p4g_players.py` is largely a direct port of GDP-Zero's `core/players.py`; some of its dead/commented blocks remain.
- `utils/gen_models.py` mixes the pre-1.0 and ≥1.0 `openai` SDK call styles (see `requirements.txt`); pick/pin the version that matches the backend you use.
- No training entry point (policy-network fine-tuning) is checked in yet.

## Acknowledgements

- GDP-Zero — Yu et al., *Prompt-Based Monte-Carlo Tree Search for Goal-Oriented Dialogue Policy Planning*, EMNLP 2023. ([paper](https://arxiv.org/abs/2305.13660))
- PPDPP — Deng et al., *Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents*. ([paper](https://arxiv.org/abs/2311.00262))
- Datasets: PersuasionForGood, ESConv, CraigslistBargain.
