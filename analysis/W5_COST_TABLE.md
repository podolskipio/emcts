# §W5 cost table — per-role wall-clock

All rows measured on the same machine (RTX 4090, 24 GB), `emomcts`, `n_sims=50`, `max_turns=10`,
`--emotion_classifier hf`, `--llm_prior_topk 5`, `--beta_emo 0.7`, `max_hist_num_turns=5`. Only one
server was resident at a time — with the other running, the second would fall back to CPU and the
row would be meaningless.

Three runs are reported. The two **serving** rows are both `--num_workers 1` (sequential, so
per-call latency is service time and stays comparable between backends). A third **throughput** run
repeats the SGLang configuration with `--num_workers 10`; it answers "how long does the batch take"
rather than "how fast is a call", and is kept in its own section for that reason.

Measured 2026-09-04 with `--max_conv 10 --max_turns 10`: ten scenarios per backend, the same ten in
both rows. Taken after **all** the prompt-construction fixes listed at the bottom of this file
(1–17), including fix 9 (SGLang silently dropped the task prompt), fix 10 (p4g ended the dialogue on
the opening pleasantry), fix 11 (verbatim repetition loops) and fix 17 (`top_p`/`top_k` were left to
each backend's own default). Every measurement predating those is superseded and has been removed.

**The tree-size confound that inverted the earlier headline is gone.** The previous pair realized 31
Ollama turns against 65 SGLang turns, so seconds/turn was measuring dialogue dynamics rather than
serving speed and the per-turn "speedup" came out at 0.66×. With fixes 10, 11 and 17 in place the two
backends realize **62 and 61 turns** — near-identical denominators — and the per-turn column is a
like-for-like comparison again. It agrees with the per-output-token column in both sign and rough
magnitude, which is the check that was failing before.

## Outcomes (10 dialogues, `max_turns 10`)

| | Ollama (Q4 GGUF)<br>`--num_workers 1` | SGLang (AWQ)<br>`--num_workers 1` | SGLang (AWQ)<br>`--num_workers 10` |
|---|---|---|---|
| episodes | 10 | 10 | 10 |
| SR | 0.50 | 0.80 | 0.70 |
| AT (PPDPP convention, failures = `max_turns`) | 7.10 | 6.10 | 6.40 |
| mean *actual* turns | 6.20 | 6.10 | 6.40 |
| turns per episode (sorted) | 3, 4, 4, 5, 5, 6, 7, 8, 10, 10 | 4, 4, 4, 4, 5, 6, 10, 10, 10, 10 | 4, 4, 5, 5, 5, 5, 6, 10, 10, 10 |
| realized turns (profiler denominator) | 62 | 61 | 64 |
| **wall clock** | **3 h 18 min** | **2 h 30 min** | **52 min** |
| **wall seconds / turn** | **192.0** | **147.6** | **49.1** |
| total calls | 31,326 | 16,100 | 16,551 |
| seconds inside LLM calls | 11,894.6 | 8,994.1 | 17,646.0 *(overlapped)* |
| output throughput (per wall second) | 76.0 tok/s | 107.5 tok/s | 273.5 tok/s |

The third column is the same SGLang configuration run with ten dialogues in flight at once
(`--num_workers 10`); see **Throughput mode** below for what it does and does not measure. The
first two columns are the like-for-like *serving* comparison — both sequential, so per-call latency
is service time in each.

The two backends now produce **near-identical dialogue lengths** (6.20 vs 6.10 mean actual turns)
where the previous pair differed 2.1×. The termination divergence the earlier write-up attributed to
Q4-GGUF-vs-AWQ quantization was substantially fixes 9, 10 and 17, exactly as that write-up predicted.

They still differ on *outcome*: SGLang converts 8/10 against Ollama's 5/10. How they fail differs too:

* **Ollama:** 2 failures run to the cap, and 3 are caught by fix 11's stall detector at turns 6, 7
  and 8 — i.e. the verbatim loops did not disappear with the sampling fix, they moved to the
  backend that previously had none, and the detector now ends them early instead of letting them
  burn to the cap. Those three episodes avoided 9 turns of futile search between them.
* **SGLang:** both failures run to the cap with **zero** verbatim repeats — near-loops (fix 11's
  documented residual), not fixed points. 1 duplicate utterance in the entire run, against 3
  episodes previously spending 46% of the denominator on loops.

`AT` remains **not** a valid cost denominator once SR < 1: `average_turn()` counts a failed episode
as `max_turns`. With the stall detector now ending failures *before* the cap, it diverges again — it
reports 7.10 for Ollama against 6.20 actual. Both tables below divide by the profiler's realized-turn
count, verified equal to `sum(num_turns)` over the episode pickle (62 and 61).

## "Before": Ollama (vicuna:13b, Q4 GGUF) — 62 turns

| role | calls | samples | calls/turn | mean latency/call | tokens in/call | tokens out/call | total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy prior | 1785 | 1785 | 28.8 | 1.52 s | 1278 | 125 | 2719.0 |
| value estimator | 17850 | 17850 | 287.9 | 0.15 s | 1105 | 12 | 2753.6 |
| user simulator | 4248 | 4248 | 68.5 | 0.65 s | 1129 | 46 | 2757.7 |
| system utterance | 4248 | 4248 | 68.5 | 0.86 s | 1210 | 64 | 3650.1 |
| emotion classifier (HF, local) | 3195 | 3195 | 51.5 | 0.00 s | — | — | 14.2 |
| **total** | **31326** | | **505** | | | | **11894.6** |

Run totals: 31,934,450 tokens in, 904,353 tokens out. Per turn: 505 calls, ~515k tokens in, 192 s.

## "After": SGLang (vicuna-13B-v1.5-AWQ, 4-bit) — 61 turns

| role | calls | samples | calls/turn | mean latency/call | tokens in/call | tokens out/call | total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy prior | 1452 | 1452 | 23.8 | 0.88 s | 1403 | 82 | 1276.2 |
| value estimator | 1452 | 14520 | 23.8 | 0.28 s | 1233 | 120 | 403.5 |
| user simulator | 4900 | 4900 | 80.3 | 0.64 s | 1315 | 58 | 3133.2 |
| system utterance | 4900 | 4900 | 80.3 | 0.85 s | 1371 | 79 | 4159.9 |
| emotion classifier (HF, local) | 3396 | 3396 | 55.7 | 0.01 s | — | — | 21.5 |
| **total** | **16100** | | **264** | | | | **8994.1** |

Run totals: 16,989,965 tokens in, 967,470 tokens out. Per turn: 264 calls, ~279k tokens in, 147 s.

The `samples` column carries the structural win: the value estimator's `num_return_sequences: 10`
costs SGLang **1,452 calls for 14,520 samples**, where Ollama pays **17,850 calls for 17,850
samples** — 287.9 calls/turn against 23.8, a 12× difference in call count for the same job. Ollama
has no native `n`, so it issues ten sequential HTTP requests per invocation, each re-sending the
full ~1.1k-token prompt.

## Per turn

Denominators are comparable here (62 vs 61 realized turns), so unlike the previous pair this table
is a like-for-like comparison rather than an artefact of dialogue length.

| role | calls/turn | seconds/turn | speedup |
|---|---|---|---:|
| policy prior | 28.8 → 23.8 | 43.9 → 20.9 | 2.1× |
| value estimator | 287.9 → 23.8 | 44.4 → 6.6 | **6.7×** |
| user simulator | 68.5 → 80.3 | 44.5 → 51.4 | 0.9× |
| system utterance | 68.5 → 80.3 | 58.9 → 68.2 | 0.9× |
| emotion classifier | 51.5 → 55.7 | 0.2 → 0.4 | 0.6× |
| **wall** | | **192 → 147 s** | **1.30×** |

The two simulator roles sit slightly *below* parity (0.9×) because SGLang issued more of those calls
per turn (80.3 vs 68.5) — it searched wider trees, having converted 8/10 dialogues instead of 5/10.
Per generated token those same roles are 1.25–1.29× in SGLang's favour, which is the cleaner read.

## Per generated token — the confound-free comparison

Dividing each role's total latency by its total output tokens removes both residual tree-size
difference and the differing generation lengths. Prompt sizes are close enough across backends
(1278 vs 1403, 1105 vs 1233, 1129 vs 1315, 1210 vs 1371 tokens in/call) for this to be like-for-like.

| role | Ollama ms/output token | SGLang ms/output token | decode speedup |
|---|---:|---:|---:|
| policy prior | 12.2 | 10.8 | 1.13× |
| value estimator | 12.9 | 2.3 | **5.55×** |
| user simulator | 14.2 | 11.0 | 1.29× |
| system utterance | 13.4 | 10.7 | 1.25× |

**SGLang decodes ~1.13–1.29× faster per token on the single-sample roles, and 5.55× on the one role
that asks for ten samples.** Per *invocation* the value estimator costs 10 × 0.154 s = 1.54 s under
Ollama against a single 0.278 s call under SGLang — the same 5.55×. This is the structural win of the
migration and the result that is invariant to tree size: `heuristic` asks for
`num_return_sequences: 10`, Ollama issues ten sequential requests each re-sending the full prompt,
while SGLang forks all ten from one cached prefix in a single request.

Aggregate throughput over the whole run: output **76.0 → 107.5 tokens/s**. Prompt throughput reads
2,685 → 1,889 tokens/s, which looks like a loss but is not one — with `--num_workers 1` the client is
a sequential chain of dependent requests, so prompt throughput measures prefill of a mostly *cached*
prefix rather than real work. Measured from the server log over this run: 31,899,352 of 35,710,384
prefill tokens were radix-cache hits, an **89.3% prefix hit rate**.

## Throughput mode: SGLang with 10 concurrent dialogues

`--num_workers 10` plays all ten dialogues at once on threads against the same server. Nothing else
changes — same model, same `n_sims=50`, same sampling, same scenarios.

| | SGLang `--num_workers 1` | SGLang `--num_workers 10` | gain |
|---|---:|---:|---:|
| wall clock, 10 dialogues | 2 h 30 min (9,001 s) | **52 min 22 s (3,142 s)** | **2.86×** |
| wall seconds / turn | 147.6 | **49.1** | **3.01×** |
| output throughput | 107.5 tok/s | **273.5 tok/s** | **2.54×** |
| realized turns | 61 | 64 | — |
| total calls | 16,100 | 16,551 | — |

### Ollama vs SGLang at `--num_workers 10`

Against the Ollama baseline the SGLang throughput run is **3.79× on wall clock, 3.91× per turn,
3.60× on output throughput** (11,902 → 3,142 s; 192.0 → 49.1 s/turn; 76.0 → 273.5 tok/s). Per turn
that factors exactly into the two independent wins:

| step | speedup |
|---|---:|
| backend migration — Ollama → SGLang, both `--num_workers 1` | 1.30× |
| concurrency — SGLang `--num_workers 1` → `--num_workers 10` | 3.01× |
| **combined** | **3.91×** |

1.30 × 3.01 = 3.91, which is the measured end-to-end figure. Concurrency is worth appreciably more
than the backend migration here.

**There is no measured Ollama `--num_workers 10` row**, and the comparison above therefore uses
Ollama's *sequential* row as the baseline. The reason is that the flag buys Ollama nothing at the
default configuration: with no `OLLAMA_NUM_PARALLEL` set, the daemon auto-selected a **single** slot
for this 13B model — `runner.parallel=1` and `n_seq_max = 1` in the server log, with `refCount=9`
showing the other nine requests queued behind it — so ten client threads serialize server-side.
That is read from the daemon log, **not** measured end-to-end; raising `OLLAMA_NUM_PARALLEL` would
change it, at the cost of splitting the KV cache across slots.

So read 3.79× as "SGLang with batching against Ollama as configured out of the box", not as a
like-for-like parallel-vs-parallel result. The 1.30× row above is the clean serving comparison.


### Per-role, 10 workers — 64 turns

| role | calls | samples | calls/turn | queued latency/call | tokens in/call | tokens out/call | total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy prior | 1633 | 1633 | 25.5 | 1.99 s | 1302 | 79 | 3245.9 |
| value estimator | 1633 | 16330 | 25.5 | 0.47 s | 1132 | 120 | 774.7 |
| user simulator | 4784 | 4784 | 74.8 | 1.13 s | 1178 | 44 | 5413.6 |
| system utterance | 4784 | 4784 | 74.8 | 1.71 s | 1246 | 68 | 8183.8 |
| emotion classifier (HF, local) | 3717 | 3717 | 58.1 | 0.01 s | — | — | 28.0 |
| **total** | **16551** | | **259** | | | | **17646.0** |

Run totals: 15,574,029 tokens in, 859,331 tokens out. Per turn: 259 calls, ~243k tokens in, and
**49 s of wall clock**.

**Read the latency column as queued time, not service time.** With ten dialogues in flight the calls
overlap, so the per-role seconds sum to 17,646 s against 3,142 s of wall clock — a 5.6× overlap
factor — and they no longer add up to anything. Latency per call roughly doubles (policy prior
0.88 → 1.99 s, system utterance 0.85 → 1.71 s) because each request now waits behind nine others.
That it only doubles under 10× the load is the whole point: the sequential runner is one dependent
chain of ~0.2-1 s requests with the GPU idle in between, so most of the added concurrency is
absorbed by capacity that was previously going to waste.

Only the **calls**, **samples**, **tokens** and **calls/turn** columns stay comparable to the
sequential tables — those count work, not time. `--profile_roles` warns about exactly this in its
help text, which is why the two rows quoted as the serving comparison are both `--num_workers 1`.

SR lands at 0.70 against the sequential 0.80. That is 7/10 vs 8/10 — one episode, well inside noise
at this sample size, and there is no mechanism by which thread count would change dialogue quality:
each dialogue builds its own MCTS object per turn and the agents hold no mutable state.

## What to take from this pair

The migration's mechanism holds and is now cleanly measurable: prefix caching plus native
`n`-sampling make SGLang faster per unit of generation, and with the dialogue-dynamics confound
removed the per-turn wall clock agrees — **1.30× end-to-end, 5.55× on the value estimator**, which
is where the structural advantage lives. Both the per-turn and per-output-token tables are now
quotable; the earlier guidance to quote only the decode-rate table no longer applies, because the
turn histograms are comparable (62 vs 61).

Two things to state alongside any headline number. First, the value estimator's 6.7× per-turn figure
is a *call-count* win (287.9 → 23.8 calls/turn), not purely a decode win, and it is the single
largest contributor to the wall-clock gap. Second, the two simulator roles come out at 0.9× per turn
purely because SGLang searched more turns' worth of successful dialogue; per token they are ahead.

Separately from the serving comparison, the practical cost of running this pipeline is set mostly by
**concurrency, not by decode speed**. Ten dialogues in flight cut the batch from 2 h 30 min to 52 min
on the identical server — a further 2.86× on top of the 1.30× from the backend migration, for
**3.79× against the sequential Ollama baseline**. The sequential runner is a dependent chain of
sub-second requests that leaves the GPU idle most of the time, so this is reclaimed waste rather
than extra hardware. Quote it as a throughput result, kept separate from the per-call numbers.

## Comparability caveats

- **Denominators are now comparable** (62 vs 61 realized turns), which was not true of the earlier
  pair (31 vs 65) — that is what makes the per-turn column meaningful this time.
- **Different quantizations of the same model**: Q4 GGUF under Ollama vs 4-bit AWQ under SGLang.
  Same family and size class, not bit-identical weights. After fixes 9/10/17 this no longer drives
  divergent termination behaviour, but it plausibly still contributes to the SR gap (0.50 vs 0.80),
  which is an outcome difference, not a cost one.
- **Sampling is now pinned identically on both** (`top_p=0.9`, `top_k=40`, fix 17). Before that each
  backend used its own default and the comparison was between two different samplers.
- **The emotion classifier is effectively free** in both rows (4.4 ms and 6.3 ms/call) — it is the
  local distilroberta (`--emotion_classifier hf`), not an LLM. The dash in the tables means "not an
  LLM call", not missing data.
- **The emotion-classifier call count** is cached per utterance string, so it tracks how many
  *distinct* user utterances each backend's tree produced (51.5 vs 55.7 per turn).
- **SR at n=10 carries a wide interval.** 0.50 vs 0.80 is 5/10 against 8/10; do not read it as a
  settled outcome difference without more scenarios.
- **The `--num_workers 10` column is a throughput measurement, not a latency one.** Its per-call
  latency includes queueing and its per-role seconds overlap 5.6×; only wall clock, call counts and
  token counts are meaningful there. Never compare it against a backend measured sequentially as if
  it were a serving result.


## Method

`src/utils/role_profiler.py` holds a ContextVar role stack. Call sites tag themselves
(`with role(POLICY_PRIOR): ...` — nine sites in `src/players/p4g_players.py`, plus both emotion
classifiers), and the generation model reports each call's latency and token counts, so attribution
is explicit rather than inferred from prompt shapes. Recording happens in `OllamaModel._post` (where
Ollama returns `prompt_eval_count` / `eval_count`) and in `SGLangChatModel.chat_generate` (from
`usage`, with `samples=len(choices)`). Inert unless `--profile_roles` is passed.

Reproduce:

```bash
cd src
# before -- Ollama resident; `ollama stop vicuna:13b` afterwards to free the VRAM
python runners/rollout.py --game emo_p4g --algo emomcts --llm ollama --ollama_model vicuna:13b \
  --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 --num_mcts_sims 50 \
  --max_conv 10 --max_turns 10 --num_workers 1 --profile_roles \
  --output outputs/ollama_10d_50sims_fixed/ollama_10d_50sims_fixed.pkl
# after -- start the server first, with only one backend GPU-resident at a time:
#   PATH=/home/piotr/virtual_envs/SGLEnv/bin:$PATH scripts/serve_sglang.sh
python runners/rollout.py --game emo_p4g --algo emomcts --llm sglang \
  --sglang_model TheBloke/vicuna-13B-v1.5-AWQ \
  --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 --num_mcts_sims 50 \
  --max_conv 10 --max_turns 10 --num_workers 1 --profile_roles \
  --output outputs/sglang_10d_50sims_fixed/sglang_10d_50sims_fixed.pkl
# throughput mode -- same server, ten dialogues in flight at once
python runners/rollout.py --game emo_p4g --algo emomcts --llm sglang \
  --sglang_model TheBloke/vicuna-13B-v1.5-AWQ \
  --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 --num_mcts_sims 50 \
  --max_conv 10 --max_turns 10 --num_workers 10 --profile_roles \
  --output outputs/sglang_10d_50sims_par10/sglang_10d_50sims_par10.pkl
```

Artifacts per run in `src/outputs/<name>/`: episode pickle, `_role_profile.json` (source for these
tables), `_emotions.json`, `metadata.json`.

## Prompt-construction fixes these rows depend on

All eight were pre-existing. Fixes 1, 2, 4, 5, 7 and 8 affect every backend and task; fix 3 is p4g-only, fix 6 esc/cb-only.

**1. History dropped for states ending mid-turn** (9 sites: 3 each in `p4g_players.py`,
`esc_players.py`, `cb_players.py`). `len(DialogSession)` returns `history // 2`, so a state holding
an odd number of entries — exactly the state a user simulator is asked about — tested as `len == 0`
and the builder returned `[]`, dropping the entire conversation from the prompt. The model then
continued the few-shot examples as the *wrong speaker*, and `_cleaned_chat_resp` truncated the reply
to `''`. Measured on SGLang: 6/6 empty before, 0/6 after. Guard now tests `len(exp.history) == 0`.

Under Ollama this produced 16% empty user turns; under SGLang, 99.8% — dialogues collapsed into
empty-turn loops. Any SGLang run predating this fix is invalid.

**2. Truncation kept ~half the dialogue instead of `max_hist_num_turns`** (12 sites, 4 per file —
the completion-path builders have it too). `max(0, len(exp) // 2 - K)` halves an already-halved turn
count. With fix 1 restoring real utterances, prompts grew until SGLang rejected them
(`400: requested 4181 tokens, 3925 from input messages` against a 4096 context) and the run produced
zero episodes. Now `max(0, len(exp) - K)`; a full 10-turn p4g dialogue builds an ~800-token prompt.

**3. The p4g persuader ignored the dialogue act MCTS chose** (`p4g_players.py`,
`PersuaderChatModel.get_utterance_batched`). `da_prompt` was computed from `action` and then never
used for any non-empty state: `__proccess_chat_exp` attaches a DA instruction to each user turn by
reading the *next* system DA out of the history, and the final user turn — the one being answered —
has no next turn, so it fell through to the branch that emits the bare utterance. Only turn 0 (the
`len(state) == 0` path) conditioned on the action. Verified by building the prompt for all 7
persuader DAs at turn 1: **7 actions produced 1 byte-identical prompt** before the fix, 7 distinct
prompts after.

Consequence: for every turn after the first, all children of a node were sampled from the *same*
distribution and differed only by temperature noise, while `get_next_state` still stamped
`sys_da = dialog_acts[action]` onto the node. The search selected DAs that never reached the model,
and the DA labels on the resulting episodes describe an action the utterance was not conditioned on
— which also propagates into the DA×emotion mining scripts. `esc_players.py` and `cb_players.py`
already thread `da_prompt` into that last user message; p4g now matches them. Present since the
initial commit, so it affects every p4g run including both rows above; the per-call latency and
token counts stay representative, but call counts and tree shape will change on a re-run.

**4. The planner ignored both of its truncation limits** (6 sites, 2 per player file).
`P4GChatSystemPlanner` and its esc/cb twins stored `max_hist_num_turns` / `user_max_hist_num_turns`
but passed neither into their history builders, so the **policy prior and value estimator saw the
whole dialogue while the system-utterance model and user simulator saw the last 5 turns** — the
roles disagreed about what "the conversation" is, and the two planner prompts were the only ones
that grew without bound. `_format_history_for_topk` (the top-K prior) had no truncation at all.
All four roles now share the same window. Measured on the p4g example dialogue, prompt tokens at
turn 15: policy prior 2153 → 1247, value estimator 2019 → 1079; both now plateau at turn 5 the way
`sys utt` (1223) and `user sim` (1085) already did. This removes the remaining path back to the
`400: requested 4181 tokens` overflow.

**5. `max_conv_turns` was never wired to `--max_turns`.** `build_agents` constructed the game
positionally and took the class default of 15, while `rollout.py`'s loop stops at `--max_turns`
(default 10). MCTS therefore spent calls simulating turns 11–15, which the dialogue can never
reach — the source of the `Persuader: Goodbye. / Persuadee: Goodbye.` branches and the empty user
utterances at context depths 15–29 noted above. `build_agents` now takes `max_conv_turns` and
`rollout.py` passes its own horizon; the other runners replay dataset turns rather than rolling
out, so they keep the default.

> **Erratum (2026-09-14).** `max_conv_turns` / `--max_turns` is **not** a search depth cap. Tree search treats only donation as terminal, so the −1.0 the game returns at the turn limit is ignored and search expands states past Tmax (19.7 % of simulation steps in the Phase-1 D1 run). Costs and behaviour in this document were measured under that behaviour. Fixed behind `--search_horizon episode`; see `analysis/phase1/SEARCH_HORIZON_BUG.md`.


**6. The esc/cb value estimator ran an unbounded zero-shot branch.** `build_agents` never passes
`zero_shot` to the planner, and `ESCChatSystemPlanner` / `CBChatSystemPlanner` default it to
`True` — so while the *players* run few-shot (`zero_shot=False`), the *planner's* `heuristic` takes
its `else` branch. That branch built the conversation with no `max_hist_num_turns`, making it the
one prompt still growing with the whole dialogue: **4100 tokens at 15 turns** of max-length
utterances, past vicuna's 4096. Now windowed like the rest. (This branch is also the only place an
esc/cb *scenario* — `emotion_type`/`problem_type`, item and prices — reaches the model at all; the
simulators never see it, because their scenario text sits behind `zero_shot=True`, which no runner
enables. That asymmetry is untouched and worth a decision of its own.)

### Window size: `max_hist_num_turns` stays at 5

Fix 2 exposed a subtlety worth recording. The old `len(exp) // 2 - K` never truncated at all below
10 turns, so at p4g dialogue lengths the chat path effectively had **no** history window — fixes 2
and 4 together therefore turn truncation *on* for the first time rather than repairing it.

`K=10` was tried, on the argument that it reproduces the original's effective behaviour. It does,
and it is safe on context, but it costs ~4x the wall-clock (prompts roughly double at depth: a
partial 50-sim run took ~9 min/turn and rising, against the ~88 s/turn the same n=1 SGLang
configuration showed at `K=5`). `K` is therefore
**5** — the value the parameter has always declared, the value the rows above were measured at, and
the value that keeps prompts bounded. `K` is now a real knob rather than dead code, so whether more
history helps persuasion is a sweep worth running, not a default worth guessing.

Headroom verified two ways against vicuna's 4096 context (prompt + 128 generated), at `K=10` — the
*worse* case, so `K=5` is strictly safer:

| task | worst over the real dataset (200 dialogs, horizon 15) | adversarial bound (every utterance at the 128-token cap) |
|---|---:|---:|
| p4g / emo_p4g | 2177 | 3731 (headroom 365) |
| esc | 2760 | 3921 (headroom 175) |
| cb | 985 | 3263 (headroom 833) |

The adversarial column is a true ceiling — no utterance can exceed `max_new_tokens=128`. esc's
175-token margin at `K=10` is the binding constraint if `K` is ever raised.

**7. The top-K policy prior never worked.** `_build_topk_messages` sent its instruction as a
trailing **system** message. Under vicuna's template that renders as plain text after the few-shot
dialogue, so the model ignored the ranking task and simply continued role-playing
(`" Persuader: [emotion appeal] That's a great start. Every bit helps..."`).
`_parse_topk_response` then scraped the single DA out of that role-play and returned it as the
"top-K" list — so the prior put ~96.5% of its mass on one action. Measured over a partial 50-sim
run: **90 of 97 calls returned exactly one action, always `emotion appeal`**. Sending the
instruction as a **user** message fixes it; verified live, returning five ranked actions that shift
with dialogue state. Since the prior drives branching, every call count in any run predating this
fix describes a search that was never working as designed.

**8. The greeting turn was missing from the profiler's denominator.** `rollout.py` realizes turn 0
(the forced greeting) *outside* the planning loop, and `mark_turn()` was only called inside it. That
turn's system-utterance, user-simulator and emotion-classifier calls were counted in the numerator
regardless, so every `calls/turn` figure was inflated by `(turns+1)/turns` — 33% on a 4-turn
episode, 11% on a 9-turn one, i.e. by *different* amounts in the two rows being compared. Fixed by
marking the greeting turn. The rows above post-date the fix and divide by the profiler's own
realized-turn count, verified equal to `sum(num_turns)` over the episode pickles (62 and 61).

Note that `AT` is *not* a substitute for that denominator once `SR < 1`: `average_turn()` scores a
failed episode as `max_turns` (the PPDPP convention), which for the Ollama row reports 7.10 against
6.20 actual turns.

Note for methodology: **Ollama silently truncates oversized prompts where SGLang errors**, so the
pre-fix "before" row was running on quietly clipped context. This formula is inherited from GDPZero,
so the corrected version diverges from that baseline — but the original cannot run this pipeline at
vicuna's 4096 context once the simulator actually speaks.

---

# Second round of fixes (2026-09-04) — the tables above post-date all of them

An earlier version of this file carried the pair measured *before* these fixes, with a defect that
**silently deleted the task prompt on SGLang** and reshaped it on Ollama. That pair has been
removed and re-measured; the tables at the top of this file are the re-measurement, taken with
fixes 9-17 in place. What follows is that second round of fixes, numbered on from the eight
already listed.

## 9. SGLang discarded every system message except the last (SGLang-only, all tasks — the big one)

Every prompt builder in `players/` separates the few-shot demo from the live conversation with a
**mid-conversation** `{"role": "system"}` marker:

```python
messages = [
    {'role': 'system',  'content': self.task_prompt},      # role + the list of dialog acts
    *self.prompt_examples,                                  # the few-shot demo
    {'role': 'system',  'content': self.new_task_prompt},   # "The following is a NEW conversation"
]
messages += <the live dialogue>
```

That is a GDP-Zero inheritance and it is correct against the OpenAI API, which keeps such a
message where it sits. It is **not** correct against either local backend:

* **SGLang.** `TheBloke/vicuna-13B-v1.5-AWQ` ships no `chat_template` in `tokenizer_config.json`,
  so SGLang falls back to its built-in `vicuna_v1.1` conversation template. In
  `sglang/srt/parser/conversation.py::generate_chat_conv` the entire handling of a system message
  is `conv.system_message = message.content` — **the last one wins and is hoisted to the top of
  the prompt**. So the leading task prompt was deleted outright, and the separator was moved out
  of its position, leaving the demo running straight into the live conversation with no boundary.
  Rendered, before vs after the fix:

  ```
  BEFORE:  SEPARATOR: The following is a NEW conversation. USER: Persuader: demo-sys-1
           ASSISTANT: Persuadee: [neutral] demo-usr-1</s>USER: Persuader: live-turn-1 ASSISTANT:
  AFTER:   TASK: you are a Persuadee. You can choose [no donation] [neutral] [donate]
           USER: Persuader: demo-sys-1 ASSISTANT: Persuadee: [neutral] demo-usr-1</s>
           USER: SEPARATOR: The following is a NEW conversation.
           Persuader: live-turn-1 ASSISTANT:
  ```

  Verified live against the running server with a two-system-message probe (secret planted in the
  first, question asked after the second): SGLang answered *"I'm sorry, I don't know what you're
  referring to"*; Ollama answered *"The secret code is XYZZY123."*

* **Ollama.** `vicuna:13b`'s Modelfile template is single-turn (`{{ .System }}\nUSER: {{ .Prompt }}`),
  so Ollama's fallback path concatenates **all** system messages into the one `.System` slot. The
  text survives, but the separator still loses its position.

**Consequences for the SGLang row above.** The persuadee never received
`[no donation] [negative reaction] [neutral] [positive reaction] [donate]`; the persuader never
received the Save-the-Children background or its role; the policy prior and value estimator never
received theirs. And because the demo and the live conversation were glued together with no
boundary, the model read the demo as *the conversation it was in* and continued it. That is
directly visible in the episodes: dialogue `20180723-042344_940_live` is **8/8 utterances copied
verbatim from `EXP_DIALOG`**, ending on the demo's own
`"I would donate 1 dollar to this charity and feel good about it I think."` — and was scored a
success. Across the SGLang row, 13.1% of all utterances were verbatim copies of a *substantive*
(non-greeting) demo turn.

**Fix.** `GenerationModel._normalize_chat_messages`, applied in `OllamaChatModel.chat_generate`
and `SGLangChatModel.chat_generate` (the OpenAI/Azure paths are left alone — they honour the
messages natively, and normalizing them would diverge from the published GDP-Zero baseline):

1. leading system messages merge into the single system message the template has a slot for;
2. a later system message is folded into the text of the *next* non-system message — it
   introduces what follows — or the previous one if it is trailing;
3. consecutive same-role messages merge, so user/assistant strictly alternate (`ADD_COLON_TWO`
   picks its `</s>` separator by message *index*, so a repeated role desynchronizes turn
   boundaries).

Checked against all 39 distinct prompt shapes the three tasks build (system utterance, user
simulator, policy prior, value estimator, top-K prior, deal classifier): exactly one system
message and it is first, strict alternation after it, no text dropped.

Measured effect on p4g (SGLang, same configuration): substantive demo copying **13.1% → 4.7%**.

## 10. p4g ended the dialogue in failure on the opening pleasantry

`PersuasionGame.get_dialog_ended` returned `-1.0` as soon as any persuadee turn carried
`U_NoDonation`. In GDP-Zero that rule is never exercised in this shape — GDP-Zero scores one next
response against a *dataset* prefix, it never self-plays a dialogue from scratch. `rollout.py`
does, and at turn 0 the persuadee has been asked nothing, so the DA it tags itself with is
ungrounded. Four of the ten Ollama episodes died on their first turn:

```
Persuader: [greeting]     Hello. How are you today?
Persuadee: [no donation]  I'm doing fine, how are you doing today?     <- episode over, SR miss
```

That is 4/10 episodes and the entire reason Ollama's mean actual turns was 3.10 against SGLang's
6.50 *in the removed pre-fix pair* — the "dialogue dynamics" confound that write-up attributed to
quantization differences was substantially this bug plus fixes 9 and 17. Re-measured with all three
fixed, the two backends land at 6.20 and 6.10 mean actual turns.

Both sibling games in this repo (`esc`, `cb`) and the PPDPP/DPDP reference env only ever end in
failure at the turn cap, so p4g now matches them. `end_on_no_donation=True` restores GDP-Zero's
rule. While there, the success scan was moved **before** the turn-cap check: previously a donation
on the final turn was scored as a failure, which esc/cb never did.

## 11. Dialogues fell into verbatim repetition loops with no way out

Both simulators run on a fixed `max_hist_num_turns` window, so once that window fills with a
repeated exchange the prompt is a **fixed point** — the same context yields the same reply, which
rebuilds the same context. Three of ten SGLang episodes ran turns 4–10 re-emitting one exchange
(45%, 45%, 30% verbatim repeats), contributing 30 of the 65 realized turns in the pre-fix pair.

`DialogGame.is_stalled` now fires when the latest system utterance *and* the latest user utterance
each repeat an earlier utterance by the same speaker, and all three games treat it as a failure
terminal. Requiring both sides to repeat keeps it conservative. Replayed over the existing
episode pickles it fires on **exactly the four looping SGLang episodes and on nothing else** — no
Ollama episode, no successful episode — and avoids 11 of 65 realized turns (17%), each of which
is a full MCTS search (~330 LLM calls at these settings). Making it a *terminal* rather than a
rollout-loop guard also lets MCTS learn that a looping branch is worthless instead of expanding it.

SR and AT are unchanged by this: `average_turn()` scores a failed episode as `max_turns` whatever
its realized length, and a looping episode was already going to fail.

**Residual.** The rule is verbatim-only, so it does not catch a *near*-loop — a persuadee that
keeps deferring in slightly reworded sentences ("I will definitely consider it, thank you for
bringing this to my attention") still runs to the cap. Loosening to a similarity threshold would
catch those but risks firing on healthy dialogue; left as-is deliberately.

## 12. The CB few-shot example was an emotional-support conversation

`CB_EXP_DIALOG` was the ESConv counselling dialogue with negotiation act labels pasted on top —
inherited verbatim from DPDP, and the **only** in-context example the CB buyer and seller ever saw:

```
Buyer:  [counter]  Why are you not feeling very good about yourself, lately?
                   ^ act instruction: "Please propose a new price or a new price range."
Seller: [deal]     Thank you, but I feel like everyone says that.
```

Replaced with a real CraigslistBargain dialogue (`cb-valid.txt` #94: a studio listed at $2440,
buyer target $1854, deal struck at $2000), covering greet / inquire / deny / confirm / propose /
agree and ending on the seller's `[deal]`. Seller acts are mapped onto CBGame's two-DA user
ontology exactly the way `read_cb()` maps the dataset (agree/affirm/accept → deal, else no deal).

## 13. esc/cb never told the simulators what the scenario was

`EmotionSupportDialogSession` carries the case's `emotion_type`/`problem_type` and
`CBDialogSession` carries the item, both reservation prices and both listing descriptions — but
they only reached the model through the players' `zero_shot=True` branches, and `build_agents`
always constructs the players with `zero_shot=False`. So:

* **every ESC episode simulated the same patient** — the single mother from the few-shot demo —
  whatever scenario the dataset row held;
* **CB was a negotiation over nothing**: neither side knew the item, the listed price, or its own
  target, which also makes `rollout.py::_extract_cb_price` ("the last number mentioned")
  meaningless.

The scenario is now attached to the separator message in the few-shot path too, in the same words
the PPDPP/DPDP env uses. The ESC *therapist* deliberately still does not get it — it is supposed
to discover the issue by asking, which is what the reference does. Confirmed working: the first
post-fix ESC episode opens
`Patient: "I'm struggling with depression and anxiety because of academic pressure"` against a
dataset row of `(depression, academic pressure)`.

## 14. ESC success was unreachable by construction — the critic was never consulted

`EmotionalSupportGame.get_dialog_ended` only returns success on `U_Solved`, and with
`zero_shot=False` (which is what `build_agents` always passes) `get_next_state` took the user's
dialog act from `PatientModel.get_utterance_w_da` — i.e. it asked the **patient to label its own
emotional state**. The patient is simultaneously instructed "you are the patient who is looking for
help ... because you have the emotional issue about X regarding Y", and it role-plays that
persistently, so it never reports itself solved. ESC SR was therefore **0 whatever the planner
did**, on every run.

Probed against the live model on a conversation where the patient had *just said*
*"Thank you so much, I feel much better now and I know exactly what to do"*:

| asked | result |
|---|---|
| the patient, 12 samples (`get_utterance_w_da`) | `Feel the same` ×6, `Feel worse` ×6 — **never** `Feel better`, never `Solved` |
| the critic, resolved conversation (`planner.heuristic`) | `Solved` ×10, **v = +1.000** |
| the critic, unresolved conversation | `Feel worse` ×10, **v = −1.000** |

The critic separates the two cases perfectly. It is `ESCChatSystemPlanner.heuristic` — the same
zero-shot judge PPDPP/DPDP's `env.compute_reward` uses — and the repo already ran it on every MCTS
node for the leaf value. It simply never decided the *dialogue act*.

`EmotionalSupportGame.get_next_state` now takes the utterance from the few-shot patient (tag
stripped) and the **label from the critic**, in both branches; they differ only in who writes the
utterance. This is what PPDPP does, and structurally what cb already did — `SellerChatModel`
runs its own yes/no deal classifier rather than trusting the seller's self-tag. p4g is deliberately
left on GDP-Zero's self-tagging, where the persuadee does emit `[donate]` readily (SR 0.60).

Verified end to end: on a resolved conversation `get_next_state` now stamps
`Patient: [Solved] ...` and `get_dialog_ended` returns `1.0`.

**Result, and a caveat to check before quoting it.** A 3-dialogue ESC run went from **SR 0.00 ->
1.00** (AT 5.33, mean actual turns 5.33, 0% demo copying). The transcripts are coherent and close
at plausible resolution points, but n=3 is far too small to call an SR, and the bar is loose: with
`reward_dict['esc']` = {worse -1.0, same -0.5, better +0.1, solved +1.0} and `success_base` 0.1,
`map_user_action` returns `Solved` on

| the other nine votes are | `Solved` votes needed |
|---|---:|
| `better` | **1 / 10** |
| `same` | 5 / 10 |
| `worse` | 6 / 10 |

so a *single* `Solved` sample flips a conversation the critic otherwise reads as merely improving.
That threshold is PPDPP/DPDP's own (`if reward > 0.1: done = 1`, with the identical reward table),
so it has been left exactly as the reference has it -- but ESC SR should be read alongside
transcripts rather than quoted on its own until a full-size run confirms it.

Cost: one extra critic call per realized user turn. On SGLang that is a single `n=10` request
forked from a cached prefix (~0.26 s) — the structural win the decode table above measures at
5.9×, so it is cheap on this backend and expensive on Ollama.

**Superseded intermediate attempt, recorded because it is a tempting wrong turn.** Relabelling the
demo's closing patient turn `[Feel better]` → `[Solved]` (so the patient would have an in-context
example of the terminal act, as p4g's demo has for `[donate]`) does **not** work and has been
reverted: an ESC re-run with that change still returned SR=0.00, and the 12-sample probe above
explains why — the persona instruction dominates the demo. `map_user_action` also gained the
empty-`da_dict` guard that `PersuasionGame`'s twin already had; without it `max()` raises.

## 15. CB scored a closed deal as "no deal" — SR was 0 there too

Same shape as fix 14, different mechanism. `CBGame.map_user_action` decided whether a deal had been
reached by testing the critic's **value** against `success_base` (0.1). But that value is the
price-normalized reward

```
v = (deal_price - seller_price) / (buyer_price - seller_price)
```

— i.e. how good the deal was **for the buyer**. A deal closed at the seller's asking price scores
exactly `0.0`, fails `v > 0.1`, and was recorded as `no deal`; the negotiation then ran on to the
turn cap and the episode was scored a failure. Probed against the live model on a post-fix CB run:

| state | critic verdict | v | old `map_user_action` |
|---|---|---:|---|
| Buyer *"Yes, that's within my budget."* / Seller *"Perfect! To confirm your purchase, please send me your name, address ..."* | `deal` ×10 | −0.000 | **`no deal`** |
| Seller *"$147 is significantly lower than my asking price"* | `no deal` ×10 | −1.000 | `no deal` |
| Seller *"I'm sorry, that's not possible. The rent is $2495 and I can't accept lower"* | `no deal` ×10 | −1.000 | `no deal` |

The critic discriminates perfectly; `map_user_action` returned `no deal` on all three, and CB SR was
0 across the run.

"Was a deal reached" and "was it a good deal" are two different questions, and the second already
has its own home — `metrics/dialog_metrics.py` computes **SL** from `deal_price` with this exact
formula. PPDPP/DPDP agrees: its CB test-time threshold is `reward >= 0.0` (any deal counts); the
0.4 threshold is train-time RL shaping. Termination now reads the critic's own deal/no-deal answer
by majority vote, and `v` stays the MCTS value and the SL numerator.

`CBGame.get_next_state` also switched to the critic for the label, like esc. `SellerChatModel`'s
bespoke yes/no classifier under-fires on exactly the states that matter (it answered NO on the
"Perfect! To confirm your purchase ..." turn) and cost a *third* LLM call per user turn on top of
the critic the search already runs — so dropping it fixes the labelling and removes a call. The
seller's `[deal]`/`[no deal]` tag strip moved into `get_utterance`, which is the method the game
now calls.

Verified after the change: `deal` → DA `deal`, `get_dialog_ended` = 1.0; both refusals → `no deal`,
`get_dialog_ended` = 0.0.
**Result.** A 3-dialogue CB run went from **SR 0.00 -> 1.00**, AT 5.67, **SL 0.667**, 0% demo
copying, 0% verbatim repeats. Per episode: GoPro closed at the seller's full $265 ask (SL 0.00 --
the buyer conceded completely), the apartment at the buyer's $1497 target against a $2495 listing
(SL 1.00), the bike at the buyer's $147 target against $160 (SL 1.00). Every deal price is a real
number from the dataset scenario, which was not possible before fix 13.

Two things in those transcripts are simulation-quality residuals rather than code defects, and they
mean **SL 0.667 is probably optimistic**:

* the seller conceded from $2495 to $1497 -- a 40% cut -- despite being told it is "trying to sell
  at the highest price you can, and it is listed at 2495". The seller simulator has no enforced
  reservation price, so a buyer that simply restates its target tends to win. Two of the three
  deals landed exactly on the buyer's target, which is what drives SL to 0.667.
* in one dialogue the buyer prefixed **6 of its 7 turns** with its opening line
  ("Hi, I am interested in your property. ..."), a prompting artefact rather than negotiation.

## 16. Two smaller ones

* **`self.inference_args['temperature'] = 0.0` mutated shared state** (`esc_players.py`,
  `cb_players.py`). `build_agents` hands the same dict to every player, so one `mode != 'train'`
  call switched the whole run to greedy decoding permanently — which collapses MCTS's open-loop
  realizations to a single child per action. Now copies the dict.
* **CB leaked the seller's DA tag into the transcript.** The few-shot seller turns render as
  `Seller: [no deal] <text>`, so the model emits the tag; p4g's `PersuadeeModel` and esc's
  `PatientModel` both strip it in `get_utterance_w_da`, CB's replacement (which classifies the
  deal with a separate yes/no call) did not. The literal `[deal]`/`[no deal]` went into the stored
  utterance, into the deal classifier's own snippet, and back into every later prompt as history.

## 17. `top_p`/`top_k` were never set, so each backend sampled differently

Neither parameter appeared anywhere in the codebase, so every server applied its own default:
Ollama `top_p=0.9`/`top_k=40`, while SGLang adopted the model's `generation_config`, which for
`vicuna-13b-v1.5` ships **`top_p=0.6`**. The backend comparison was therefore between two different
samplers, which is a validity problem independent of anything it caused.

What it caused is fix 11's loops. A 2×2 over both backends × `top_p` ∈ {0.6, 0.9}, n=30 samples per
cell drawn from one mid-dialogue context taken from the longest looping episode, with the system
dialogue act held fixed:

| backend | `top_p` | distinct system utterances /30 | distinct persuadee replies /30 | `[no donation]` at turn 1 |
|---|---|---:|---:|---:|
| Ollama | 0.9 *(its default)* | 23 | 30 | 10/30 |
| Ollama | 0.6 | 9 | 20 | 0/30 |
| SGLang | 0.6 *(its default)* | 10 | 25 | 0/30 |
| SGLang | 0.9 | 26 | 30 | 0/30 |
| SGLang | 0.9 + `top_k=40` | 26 | 30 | 0/30 |

The effect tracks `top_p`, not the backend: at 0.6 **Ollama collapses too** (9 distinct system
utterances in 30, 22 of them identical), and at 0.9 SGLang recovers to 26. A 0.6 nucleus is narrow
enough that the fixed `max_hist_num_turns` window becomes a fixed point. `top_k` contributes
nothing.

The turn-1 `[no donation]` mislabel of fix 10 is a *separate* axis: it is a tail event that a 0.6
nucleus also suppresses, but at matched `top_p=0.9` the two backends still differ decisively —
19/90 samples on Ollama's Q4 GGUF against 0/90 on AWQ. That one is quantization.

Fixed in `build_agents` (`runners/_common.py`), which now pins `top_p=0.9, top_k=40` into both the
system and user inference-arg defaults. Plumbing had to follow: `top_k` is not an OpenAI chat
parameter, so `SGLangChatModel._update_args` routes it (and `repetition_penalty`) through
`extra_body`, merging with any caller-supplied `extra_body`, and the OpenAI path drops it outright.

## What was re-measured, and what still stands

**p4g has been re-run** with fixes 9-17 in place, and the tables at the top of this file are that
re-measurement (`outputs/{ollama,sglang}_10d_50sims_fixed/`). The prediction made here before the
re-run held: the two backends came much closer together once they saw the same prompt and the same
sampler, landing at 62 and 61 realized turns against the pre-fix 31 and 65. That removes the
confound that had inverted the per-turn headline, so **the outcome table, the per-turn table and
the call counts are all quotable again** — not only the per-output-token table.

The **mechanism** of the migration is untouched: prefix caching plus native `n`-sampling make
SGLang faster per unit of generation, and the value estimator's `num_return_sequences: 10` is a
structural 5.55× that no prompt bug explains away.

**Still outstanding: `esc` and `cb` have not been re-measured.** Fixes 9, 12-16 and 17 all touch
them, and fixes 14 and 15 made their success criteria reachable at all, so any esc/cb number
predating this round should be treated as void. Reproduce with the same commands as above, adding
`--game esc --algo gdpzero` and `--game cb --algo gdpzero`.

## A configuration trap worth knowing about

`get_action_prob` returns MCTS visit counts and the runners take `np.argmax` of them. When
`--num_mcts_sims` is at or below the number of actions, every action gets ~1 visit, the counts are
effectively uniform, and `argmax` breaks the tie by **lowest index** — so the planner silently
returns action 0 on every turn. Action 0 is `credibility appeal` for p4g (7 actions),
`Affirmation and Reassurance` for esc (8), and `affirm` for cb (11). A 10-sim CB smoke run
produced `greet` followed by `affirm` seven times in all three dialogues, which is not a policy at
all — and since `affirm` can never name a price, no deal was reachable. The paper's 50-sim runs are
clear of this, but keep `--num_mcts_sims` comfortably above the action count for any quick run,
especially on cb.

## Known residuals (not fixed)

* **Few-shot copying is reduced, not eliminated** (13.1% → 4.7% substantive). The reference
  implementation (PPDPP/DPDP) avoids it entirely by giving the players *zero-shot* role prompts
  and no demo dialogue at all — the `zero_shot=True` branches for that already exist in
  `esc_players.py` / `cb_players.py` and are never enabled. Switching the players to zero-shot is
  the principled cure and a one-line change in `build_agents`; it is a research-design decision,
  so it has been left alone.
* **Near-loops** (fix 11's residual) still run to the cap.
* **`_sglang_cached_chat_completion` is an unbounded `lru_cache`**; memory grows over long runs.
* **Empty utterances when the model answers as the wrong speaker.** `_cleaned_chat_resp` truncates
  at the user role marker, so a reply that opens `"Buyer: ..."` when the seller was asked collapses
  to `''` and a blank turn is stored. Seen once in a post-fix CB run, right after the dialogue had
  reached a natural close ("Bye!") and was pushed past it by the turn budget. Fix 11's stall
  detector does catch a *run* of them (two empty utterances compare equal), but the first blank
  turn still lands in the transcript. A retry-on-empty in the players would be the cure; it is a
  design choice, so it has been left alone.

---

# Optimisation: logit-scoring the value estimator and the policy prior (2026-09-05)

Everything above measures the pipeline as it *generates*. Two of its five roles do not actually
need to generate: the value estimator and the policy prior both ask the model for a **label out of
a closed set** and throw the rest of the completion away.

* The **value estimator** samples 10 persuadee turns at temperature 1.1, reads the bracketed DA out
  of each, and averages `reward_dict['p4g']` over them. That is a 10-sample Monte-Carlo estimate of
  an expectation the model's logits give exactly.
* The **policy prior** samples 15 persuader turns and histograms their DA (or, under
  `--llm_prior_topk`, makes one 256-token ranking call instead).

`--logit_scoring` replaces both with a scored read of the same prompt. Nothing is generated.

**Findings, in one place:**

| | |
|---|---|
| value agreement, prompt as shipped | r = 0.933 / 0.945 / 0.939 over three 200-state samples — **misses** the 0.95 bar |
| value agreement, `--explicit_value_labels` | r = **0.950 / 0.954 / 0.957** — **clears** it; the brief's prescribed fallback works |
| what the fix actually does | de-noises the *sampled* baseline (test–retest 0.955 → 0.981); naming the answer format matters, listing the labels alone does not |
| policy prior vs the 15-sample histogram | logit **r = 0.969**, top-1 0.955 — against **r = −0.28**, top-1 0.055 for the `--llm_prior_topk` ranking call it replaces |
| estimator cost, sequential | value **2.3×**, prior-as-histogram **2.5×**, prior-as-top-K-ranking **15.0×** |
| end to end, 10 dialogues, equal 64-turn denominators | **52 min 22 s → 36 min 44 s (1.43×)**, 48% fewer tokens generated |
| reproducibility | run-to-run spread of `v` falls 11–40×, but **not** to zero; and it is not the same estimand (disattenuated r = 0.965, not 1.0) |
| act-set correction | the persuader set as wired is **7**, not the 13 in the ontology |

## What it does

`GenerationModel.score_labels(messages, labels, prefill, close)` — implemented on
`SGLangChatModel`, `NotImplementedError` elsewhere — returns P(label | prompt) over a closed set.
Two HTTP requests:

1. **Warm.** `/generate` on the rendered prompt with `max_new_tokens=0`, which is a pure prefill,
   plus `token_ids_logprob` over each label's first token (free, and returned as
   `first_token_probs`).
2. **Score.** One batched `/generate` carrying one item per label — prompt + that label's own
   tokens — read back through `logprob_start_len`, giving each label's exact sequence logprob.
   `probs` is the softmax over those.

The prompts are unchanged: the sampling and scoring paths share `_build_value_messages` /
`_build_prior_messages`, so the only thing that differs is how the answer is read.

**Why two requests and not one.** Sending request 2 on a cold prefix is the trap: the items arrive
together, each misses the radix cache, and each prefills the whole ~800-token prompt. Measured on
one prompt, RTX 4090, vicuna-13B-AWQ:

| | 5 labels | 13 labels |
|---|---:|---:|
| batch alone, cold prefix | 0.874 s | 1.057 s |
| warm then batch | **0.236 s** | **0.249 s** |
| (the warm call by itself) | 0.181 s | 0.181 s |
| the generation call it replaces | 0.362 s (n=10) | 0.388 s (n=15) |

So the warm call is not overhead — it is the prefill the batch would otherwise pay five or thirteen
times over. It doubles the *call* count in the profile below while cutting the seconds.

**First-token scoring is not a safe shortcut**, which is why request 2 exists. It is a good
approximation only when the label set is explicit in the prompt: on the real value prompt (which
lists the five labels) first-token and full-sequence distributions correlate at r = 0.9998; on a
probe prompt that did not list them they disagreed on the argmax — `no donation` leads on the token
`no` and loses once `ation` is charged for. `first_token_probs` is returned so that gap stays
visible rather than assumed away.

**Scoring needs the raw prompt**, so `SGLangChatModel.render_prompt` reproduces the server's own
rendering: a tokenizer `chat_template` when there is one, otherwise SGLang's built-in `vicuna_v1.1`
(`SeparatorStyle.ADD_COLON_TWO`) — the same template fix 9 is about. Anything else raises rather
than guessing. Verified against the live server: the local render and the chat endpoint produce
byte-identical greedy continuations.

## Correction to the brief: the persuader act set is 7, not 13

The p4g ontology declares 13 system dialog acts, but `PersuaderModel.__init__` keeps only those
with an entry in `da_prompts_mapping` — `self.dialog_acts = [da for da in dialog_acts if da in
self.da_prompts_mapping]` — which is **7**: credibility appeal, emotion appeal, proposition of
donation, logical appeal, task related inquiry, greeting, other. The planner is constructed from
`system.dialog_acts`, so the prior scores 7 tokens. The other six ontology acts (personal story,
foot in the door, self modeling, source/personal related inquiry, neutral to inquiry) have no
prompt and never reach the model in any path. The persuadee label set is 5, as expected.

## Validation: does the scored value agree with the sampled one?

`scripts/validate_logit_scoring.py`, 200 dialogue prefixes drawn one per dialog from the p4g
dataset (depths 1–12, 200 distinct dialogs), `--num_workers 10`.

The target was r ≥ 0.95 against the generation-based value. **With the value prompt as it ships it
lands at 0.9447 and misses.** The fix is a prompt change and is in the next subsection; this one is
about why the raw number is hard to read, which matters for interpreting the fixed one too.

The difficulty is that the baseline does not reach 0.95 against *itself*:

| | r |
|---|---:|
| logit vs one 10-sample generation run | 0.9447 |
| **two independent generation runs vs each other** (the baseline's own reliability) | **0.9577** |
| logit vs the mean of 2 runs (20 samples) | 0.9499 |
| logit vs the mean of 3 runs (30 samples) | 0.9568 |
| logit vs the mean of 5 runs (50 samples) | **0.9586** |
| logit vs one run, disattenuated (`r / sqrt(reliability)`) | **0.9653** |

A 10-sample mean of a 5-valued variable is a noisy thing to correlate against: with reliability
0.9577, a *noiseless* estimator could not score better than about **0.979** against one run. The
correlation rising monotonically as the baseline is averaged over more samples — 0.9447 → 0.9586 —
is the evidence that what remains is the baseline's sampling noise and not a disagreement about the
state. So on the as-shipped prompt the criterion is met on both noise-corrected readings and missed
on the raw one. Three independent 200-state runs gave 0.9290 / 0.9385 / 0.9447, so the run-to-run
spread on this figure is itself about ±0.01 — which is why the fix below is checked on three
independent samples rather than one.

Two readings that are unaffected by any of this:

* **P(donate)**, which is what the brief asks the value head for, correlates at **0.9462**. The
  scored path reads it directly; the sampled path can only estimate it as a fraction of 10 draws,
  so it is quantized to multiples of 0.1.
* **Resolution.** Over the same 200 states the sampled value takes **34 distinct values**; the
  scored value takes **200**. The 10-sample estimator cannot represent a value between two tenths,
  which is most of what the remaining disagreement is.

### The brief's fallback works, and clears the bar

"If lower, the prompt needs the label set made explicit." The first reading of the diagnostics
argued against that — 98.2% of the model's next-token mass already sat on the five labels, so the
closed-set assumption looked free. **That reasoning was wrong, and testing it settled the question
the other way.** In the value prompt as it ships, the labels appear exactly once, in the leading
system message ~1,000 tokens before the position where the answer is read, and the few-shot demo
exhibits only four of the five (`no donation` never occurs in it).

`scripts/explicit_labels_ablation.py` restates the set in the final user message and applies the
change to **both** arms — sampling and scoring share `_build_value_messages`, so neither is given a
prompt the other does not see. Three independent 200-state samples, three generation replicates
each:

| variant | r vs one gen run (seed 0 / 1 / 2) | gen test–retest ceiling | label-set mass |
|---|---:|---:|---:|
| as-is | 0.9331 / 0.9452 / 0.9388 | 0.957 / 0.956 / 0.952 | 0.982 |
| labels restated only | 0.9465 / 0.9442 / 0.9304 | 0.973 / 0.967 / 0.959 | 0.994 |
| **labels + answer framing** | **0.9504 / 0.9544 / 0.9570** | **0.979 / 0.982 / 0.982** | 0.994 |

**The criterion is met.** The winning variant clears 0.95 on all three samples; the prompt as
shipped clears it on none. It is `--explicit_value_labels`, and the text it appends is:

> Answer with your reaction to that question, beginning with exactly one of these labels:
> `[no donation] [negative reaction] [neutral] [positive reaction] [donate]`. Use `[donate]` only
> if you are agreeing to donate, and `[no donation]` only if you are refusing.

Two things about *how* it works are worth more than the headline:

* **Listing the labels is not what does it.** The middle row restates the set and gains almost
  nothing — it is worse than as-is on one of the three samples. Naming *what the label answers* is
  what carries the effect. So "make the label set explicit" is better read as "make the answer
  format explicit" than as "repeat the list".
* **It works mainly by de-noising the baseline, not by moving the scored value.** The generation
  path's test–retest reliability goes 0.955 → 0.981; correlation against the *mean* of three runs
  barely moves (0.953 → 0.959). The 10-sample histogram was spending part of its variance on
  deciding whether to answer in the label format at all, and that is what was capping agreement.
  Worst-case next-token mass on the label set goes from 88.7% to 98.2%, which is the same story
  measured at the other end.

That second point is why the flag is worth having independently of logit scoring: it makes the
**sampled** value estimator materially more reliable, and the sampled estimator is what every row
in this file was measured with.

**Default off.** It changes the MCTS leaf value for every p4g run, so every row above predates it
and turning it on breaks comparison against them, and against GDP-Zero's inherited prompt. Flipping
the default is a one-line change and probably the right one — but it should happen alongside a
re-measurement, not silently, which is the same rule the seventeen fixes above followed.

### A note on read-back temperature

A read-back temperature sweep was run because the sampling path draws at temperature 1.1, so it is
in principle estimating a *flattened* distribution. It is not: agreement improves slightly as the
read-back temperature goes **down** (T = 0.8 gives 0.9485 against one run, T = 1.1 gives 0.9425),
which is the `top_p=0.9 / top_k=40` truncation sharpening the sampler rather than temperature
flattening it. The gain is ~0.005 and fitting it would be tuning the estimator to reproduce an
artefact of the estimator it replaces, so **the deployed default is the raw softmax, T = 1.0**.
`LabelScores.at_temperature()` exposes the knob for anyone who wants it.

## Validation: the policy prior, and a finding about `--llm_prior_topk`

Same 200 states, comparing each candidate prior against the **15-sample histogram** — the prior
both of the cheap paths are standing in for.

| against the 15-sample histogram prior | logit-scored prior | `--llm_prior_topk 5` ranking call |
|---|---:|---:|
| pearson over all state × act cells | **0.9693** | **−0.2835** |
| mean per-state pearson | 0.9710 | — |
| top-1 agreement | **0.955** | **0.055** |
| mean total variation | 0.206 | 0.700 |

**The top-K ranking call is anti-correlated with the prior it was introduced to approximate.** This
is not a bug in either — they answer different questions, and inspecting the distributions makes it
obvious. The histogram and logit priors both say "what does the persuader say next", and on real
dataset states that is overwhelmingly `[other]` or `[task related inquiry]`. The top-K prompt asks
"which actions are most promising for landing a donation", so it ranks the persuasion strategies and
leaves `other` and `greeting` on the 0.5% floor. Example (dialog `20180723-042421_113_live`, 3 turns):

| act | histogram (15 samples) | top-K k=5 | logit |
|---|---:|---:|---:|
| credibility appeal | 0.045 | **0.428** | 0.009 |
| emotion appeal | 0.045 | 0.216 | 0.007 |
| logical appeal | 0.045 | 0.111 | 0.033 |
| other | **0.727** | 0.005 | **0.914** |

So `--logit_scoring prior` is **not** a pure cost change: it restores the behavioural-clone prior
that `--llm_prior_topk` had replaced with an advisory one. Which of the two is the better *search*
prior is a research question this measurement makes explicit rather than settles — but it should no
longer be decided by which one is cheaper, because the logit prior is now the cheapest of the three.
The two compose: `--logit_scoring` produces the full 7-way distribution and MCTS still hard-prunes
to the `--llm_prior_topk` highest-prior actions, at zero extra LLM calls.

## Estimator cost in isolation (`--num_workers 1`, so this is service time)

100 identical dataset states, each estimator asked the same question about each. Sequential, so
per-call latency is service time and none of the queueing caveats below apply.

| role | sampled | scored | speedup |
|---|---:|---:|---:|
| value estimator (10 completions → histogram) | 0.231 s/state | **0.101 s/state** | **2.29×** |
| policy prior (15 completions → histogram) | 0.252 s/state | **0.100 s/state** | **2.53×** |
| policy prior (`--llm_prior_topk 5` ranking call) | 1.501 s/state | **0.100 s/state** | **15.0×** |

The 15× is the one that matters for the deployed configuration: the top-K call generates up to 256
tokens of a numbered list, which is by far the most expensive thing either planner role did.

## End to end: 10 dialogues, `--num_workers 10`

Same configuration as the throughput row above (`emo_p4g`, `emomcts`, `n_sims=50`,
`max_turns=10`, `--emotion_classifier hf`, `--beta_emo 0.7`, `--llm_prior_topk 5`), same ten
scenarios, same server. Only `--logit_scoring` changes.

| | sampled (baseline) | `--logit_scoring value` | `--logit_scoring both` |
|---|---:|---:|---:|
| SR | 0.70 | 0.90 | 0.80 |
| AT | 6.40 | 4.60 | 6.40 |
| realized turns (profiler denominator) | 64 | 44 | 64 |
| turns per episode (sorted) | 4,4,5,5,5,5,6,10,10,10 | 1,3,3,4,4,5,5,5,6,8 | 4,5,5,5,5,6,6,8,10,10 |
| **wall clock** | **52 min 22 s** | **31 min 35 s** | **36 min 44 s** |
| **wall seconds / turn** | **49.1** | 43.1 *(see caveat)* | **34.4** |
| total calls | 16,551 | 10,108 | 19,076 |
| tokens in | 15,574,029 | 8,794,479 | 14,491,548 |
| **tokens out** | **859,331** | 424,815 | **447,554** |

**The like-for-like pair is the first and third columns** — both realize exactly 64 turns, so the
per-turn column is a real comparison and not an artefact of dialogue length. There,
`--logit_scoring both` is **1.43× on wall clock** (3,142 → 2,204 s; 49.1 → 34.4 s/turn) and
generates **48% fewer tokens** (859k → 448k), because two of the five roles stopped generating
entirely.

The middle column's 44-turn denominator is **not** comparable to the other two, so its seconds/turn
is not quotable; it is included for its call counts and per-call latencies, which count work rather
than time.

### Per role, `--num_workers 10`, 64 turns each

| role | calls/turn | s/call *(queued)* | tokens out/call | seconds/turn |
|---|---|---|---|---|
| policy prior | 25.5 → 49.2 | 1.99 → **0.14** | 79.3 → **0** | 50.7 → **6.7** |
| value estimator | 25.5 → 49.2 | 0.47 → **0.13** | 120.0 → **0** | 12.1 → **6.4** |
| user simulator | 74.8 → 72.1 | 1.13 → 1.33 | 43.6 → 37.0 | 84.6 → 95.6 |
| system utterance | 74.8 → 72.1 | 1.71 → 2.05 | 68.0 → 60.0 | 127.9 → 147.9 |
| emotion classifier | 58.1 → 55.4 | 0.01 → 0.01 | — | 0.4 → 0.4 |

Calls per turn double for the two scored roles because each scoring operation is two HTTP requests
(warm + batch); the *operation* counts are 1,633 → 1,576, i.e. essentially unchanged. Both roles
now generate **zero** output tokens.

**The two simulator roles read as slower, and that is the queueing artefact this file warns about
for every `--num_workers 10` row.** Per-role seconds sum to 16,449 s against 2,204 s of wall clock —
a **7.5× overlap**, up from the baseline's 5.6× — so a call's recorded latency is mostly time spent
waiting behind other calls. The simulators did not get slower; they got a larger share of a
better-saturated GPU, because the planner roles no longer stall the pipeline with long generations.
Only the calls, samples, tokens and calls/turn columns count work rather than time.

## Reproducibility: what scoring does and does not fix

Scoring is often described as making a stochastic step deterministic. That is half right, and the
half that is wrong matters for a paper.

### It is not a 1:1 replacement of the same quantity

The sampling path estimates the label distribution *as the sampler realizes it* — temperature 1.1,
`top_p=0.9`, `top_k=40`, with the label emerging from a free generation. The scored path computes
the raw softmax over the label set at temperature 1.0, no truncation, with `Persuadee: [` forced as
a prefill. Those are different estimands, and the histogram does not converge to the scored
distribution as samples → ∞; it converges to the tempered-and-truncated one.

The evidence that the gap is structural rather than noise is the disattenuated correlation:
**0.965, not 1.0** (0.9653 at n=200, 0.9630 at n=100 — stable). The temperature sweep points the
same way: agreement improves as the read-back temperature goes *down*, which is `top_p`/`top_k`
sharpening the sampler, not temperature flattening it.

Smaller divergences in the same direction: scoring renormalizes over the closed set (1.8% of
next-token mass discarded on average with the shipped prompt, 0.6% with `--explicit_value_labels`),
while sampling silently drops off-label completions and returns `v = 0.0` if all ten are dropped.

**Practical consequence: numbers produced with the sampling path do not carry over.** Anything
quoted from the rows above this section has to be re-run, not re-cited, if the pipeline switches.
The resolution change alone (34 → 200 distinct values over 200 states) alters PUCT tie-breaking, not
just precision.

### It is far more repeatable, but not bit-exact

`scripts/logit_stability.py`, 30 states × 5 repeats, measuring the spread of the *same* state's
value across identical repeated passes:

| condition | logprob spread (mean / max) | `v` spread (mean / max) | states identical across all 5 |
|---|---:|---:|---:|
| scored, warm cache, sequential | 2.8e−4 / 8.7e−3 | 3.2e−5 / 9.4e−4 | 28/30 |
| scored, `/flush_cache` before each pass | 6.9e−2 / 3.0e−1 | 1.6e−2 / 3.6e−2 | 0/30 |
| scored, 10 requests in flight | 2.2e−2 / 1.8e−1 | 4.2e−3 / 1.2e−2 | 2/30 |
| **sampled (what it replaces)** | — | **1.7e−1 / 6.0e−1** | 9/30 |

So the scored value is **11–40× tighter** than the sampled one in mean spread, and on a resident
warm cache it is very nearly exact (28/30 states bit-identical across five passes). But it is not
deterministic: the logprobs come off kernels whose reduction order depends on how the request was
batched and how much of the prefix was already resident, and **prefix-cache state matters more than
concurrency does** (6.9e−2 flushed against 2.2e−2 concurrent).

Read the worst case against the scale that matters: `v` ∈ [−1, 1] and `PersuasionGame.success_base`
is 0.1, so a 3.6e−2 drift can in principle move a state across the donation threshold — rare, but
not impossible, and worth knowing before attributing a run difference to a code change. The sampled
estimator's 6.0e−1 worst case makes the same point 17× more loudly, which is the honest framing:
scoring buys a large reduction in run-to-run variance, not its elimination.

### It adds one reproducibility dependency

`render_prompt` is a second copy of SGLang's `vicuna_v1.1` template, living in this repo. The
generation path asks the server to render the prompt; the scoring path renders it locally and sends
tokens. They agree today (verified: byte-identical greedy continuations), but if SGLang changes that
template the generation path follows it and the scoring path silently does not. Pin the SGLang
version alongside the model when quoting scored numbers.

## Caveats

* **Do not read SR off these rows.** 0.70 / 0.90 / 0.80 is 7, 9 and 8 episodes out of ten. The
  value estimator does not label user turns in this configuration: `build_agents` defaults to
  `zero_shot=False`, so `get_next_state` takes the persuadee's DA from `get_utterance_w_da` and
  discards `v`; the scored value reaches the search only as the MCTS leaf value. The one-turn
  episode in the middle column illustrates the noise directly — its *greeting*, which `rollout.py`
  realizes outside the planning loop, opened with `"Hello! I'm glad you're open to donating. Is
  there a specific reason you've chosen to donate $1?"` and the persuadee agreed on the spot. No
  planner decision was involved in that episode at all.
* **`--logit_scoring prior` changes what the prior means**, as measured above (r = −0.28 between the
  top-K ranking call and the histogram it replaced). The cost rows are like-for-like on
  configuration, not on prior semantics. `--logit_scoring value` is the semantically neutral half.
* **`_as_pseudo_samples` is off the path here.** `heuristic` returns a sample list because
  `PersuasionGame.map_user_action` takes the modal non-donate DA out of it; the scored path
  reproduces it by largest-remainder apportionment of the distribution over 10 slots. That only
  runs under `zero_shot=True` (the interactive runner), never in `rollout.py`.
* **Every cost row here was measured without `--explicit_value_labels`**, i.e. on the prompt as it
  ships. The flag is what clears the 0.95 bar, so the passing correlation and the measured wall
  clock come from two different prompts. They are separable — the flag adds one short sentence to
  a ~1.2k-token prompt and no calls — but a run quoted for both should set it and be re-measured.
* **SGLang only.** Ollama and the OpenAI chat API expose no way to score a continuation, so
  `--logit_scoring` warns and falls back to the sampling paths there. This is now the second
  structural advantage of the SGLang migration, after native `n`-sampling.
* **`render_prompt` is a second copy of the server's template.** It is checked against the server
  (byte-identical greedy continuations) but it will silently drift if SGLang changes
  `vicuna_v1.1`, and it refuses outright for any model that is neither vicuna-like nor shipping a
  `chat_template`.
* **No memoization was added.** The scored calls are repeatable enough to memoize (see
  Reproducibility below) and MCTS revisits nodes, so an `lru_cache` would help — but
  `_sglang_cached_chat_completion` is already listed above as an unbounded-memory residual, and a
  second one seemed the wrong trade to make silently. Note that memoizing would *increase*
  reproducibility here, by pinning each state's value to whichever cache state it was first
  computed under.

## Reproduce

```bash
cd src
# validation: value + prior, against the sampling paths and against --llm_prior_topk
python ../scripts/validate_logit_scoring.py --n_states 200 --num_workers 10 \
  --gen_replicates 5 --prior_topk 5 --out outputs/logit_scoring_validation.json
# the prompt fix: does making the label set explicit clear the 0.95 bar?
python ../scripts/explicit_labels_ablation.py --n_states 200 --num_workers 10 \
  --gen_replicates 3 --seed 0 --out outputs/explicit_labels_ablation.json
# reproducibility: spread of the same state's value across repeated identical passes
python ../scripts/logit_stability.py --n_states 30 --repeats 5 --out outputs/logit_stability.json
# service-time version of the same comparison
python ../scripts/validate_logit_scoring.py --n_states 100 --num_workers 1 \
  --gen_replicates 2 --prior_topk 5 --out outputs/logit_scoring_validation_seq.json
# end-to-end, value + prior
python runners/rollout.py --game emo_p4g --algo emomcts --llm sglang \
  --sglang_model TheBloke/vicuna-13B-v1.5-AWQ \
  --emotion_classifier hf --beta_emo 0.7 --llm_prior_topk 5 --num_mcts_sims 50 \
  --logit_scoring --max_conv 10 --max_turns 10 --num_workers 10 --profile_roles \
  --output outputs/sglang_10d_50sims_logit_par10/sglang_10d_50sims_logit_par10.pkl
# end-to-end, value only (prior left on the --llm_prior_topk ranking call)
python runners/rollout.py ... --logit_scoring value \
  --output outputs/sglang_10d_50sims_logitvalue_par10/sglang_10d_50sims_logitvalue_par10.pkl
```

`--logit_scoring` takes `off` (default) / `value` / `prior` / `both`; bare `--logit_scoring` means
`both`. `--explicit_value_labels` is the prompt fix, and is off by default for the reason given
above; add it to both the validation and the rollout commands to reproduce the passing numbers.
