# 1.1 Persona penalty — the number that prices half the grid

Like-for-like: 8B, 24 dialogues, 10 workers, identical grid config, only `--p4g_persona` differs.

| | no persona | persona | ratio |
|---|---:|---:|---:|
| **cache hit rate** | **0.9246** | **0.9254** | **1.00** |
| wall clock, 24 dialogues | 1,341 s | 3,253 s | **2.43x** |
| realized turns | 105 | 162 | 1.54x |
| seconds / turn | 12.78 | 20.08 | 1.57x |
| tokens in / call | 1,070 | 1,115 | 1.04x |
| calls/turn, value + prior | 15.96 each | 44.60 each | 2.79x |
| calls/turn, user sim + system | 49.39 each | 63.57 each | 1.29x |
| SR | 1.000 | 0.708 | |
| AT | 4.38 | 6.75 | |

## There is no cache penalty

The persona costs **nothing** on prefix caching: 0.9246 -> 0.9254, i.e. inside noise. The
persona text is per-dialogue, so it lengthens the prefix each worker reuses rather than
fragmenting it, and prompt length per call moves only 4% (1,070 -> 1,115 tokens). The
mechanism §1.1 warns about -- a rebuilt prompt collapsing radix hits -- is not present in
either arm.

## The penalty is search depth, not prompt size

The 2.43x is almost entirely **call volume**: the planner roles issue 2.79x the calls per
turn. The reason is visible in the SR column. Without a persona the persuadee agrees almost
immediately, so simulated rollouts hit the `U_Donate` terminal state after a couple of plies
and cost nothing further. With the real survey persona the persuadee resists, rollouts run to
the horizon, and the same 50 simulations therefore buy far more LLM calls.

Two consequences:

1. **The depth cap is the dominant cost lever in the persona cells**, exactly as the plan
   warned. Tmax = 10 is what makes an unterminated rollout expensive; the no-persona cells
   barely feel it because they terminate early on their own.
2. **The persona penalty is not transferable across models.** It is mediated by how readily
   that model's persuadee simulator concedes, which is a property of the checkpoint. It must
   be measured per model, which is why runs 2 and 3 both exist.

## Model penalty, both with persona

| | 8B persona | 13B persona | ratio |
|---|---:|---:|---:|
| wall clock, 24 dialogues | 3,253 s | 7,878 s | **2.42x** |
| seconds / turn | 20.08 | 49.54 | 2.47x |
| realized turns | 162 | 159 | 0.98 |
| cache hit rate | 0.9254 | 0.7934 | |
| peak KV pool | 0.118 | **0.704** | |

Two things to note. **The 13B uses 70% of its KV pool at 10 workers** against the 8B's 12%,
so the 16- and 24-worker arms would have met real KV pressure on the 13B even though they did
not on the 8B. Adopting 10 is safe on both; adopting more would not have been.

**The 13B's cache hit rate is 13 points lower** (0.793 vs 0.925). Both arms carry a persona,
so this is a model/template effect, not a persona effect. The 13B no-persona confirmation run
now in flight gives the baseline that isolates it. One hypothesis worth testing cheaply
afterwards: logit scoring reconstructs the raw prompt itself
(`gen_models.render_prompt`), taking the tokenizer's `chat_template` when there is one and a
hand-rolled `vicuna_v1.1` reproduction when there is not. The 8B has a chat template; vicuna
does not and takes the hand-rolled path. If that render differs from what the server builds
for a chat completion by even one token, scored prompts and generated prompts stop sharing a
radix prefix, and the hit rate falls exactly where it is observed to fall. Unverified.

> **Erratum (2026-09-14).** `max_conv_turns` / `--max_turns` is **not** a search depth cap. Tree search treats only donation as terminal, so the −1.0 the game returns at the turn limit is ignored and search expands states past Tmax (19.7 % of simulation steps in the Phase-1 D1 run). Costs and behaviour in this document were measured under that behaviour. Fixed behind `--search_horizon episode`; see `analysis/phase1/SEARCH_HORIZON_BUG.md`.
