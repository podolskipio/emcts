# Readiness Phase 4 — does coupling make comparisons resolvable? 🚦

Pre-registered in `PREREG.md` Entry 11 (`904bd51`). Script `scripts/phase4_gate.py`, numbers `phase4.json`.

**Gate: correlation 0.39 → the middle band ("limited").** Coupling roughly doubled the correlation
between arms (0.17 → 0.39), but did not reach 0.6. The pre-registered prediction (< 0.3) was wrong.
Under the brief, Phase 5 runs only the two retention comparisons (NoEmo ± #6, GDP-Zero ± #6), at 5 seeds.

## Runs

The three runs are coupled on one store (`coupling/p4.sqlite`). Shared settings: T 1.1, persona ON,
episode horizon, R 4, top-K 5, s20, the 100 eval dialogues. Commit `aad18c6`, which has the same `src/`
as `904bd51`.

| Run | Fixes | SR [Wilson] | AvgT | Calls served from the store | Wall |
|---|---|---|---|---|---|
| `P4_NoEmo_seed1` | #6 #5 | 0.68 [0.58, 0.76] | 6.61 | 1.0 %¹ | 2.91 h |
| `P4_ActPool_seed1` | #6 #5 + bucket_kernel draw | 0.70 [0.60, 0.78] | 6.64 | **25.9 %** (from NoEmo's draws) | 2.42 h |
| `P4_NoEmo_seed2` | #6 #5 | 0.75 [0.66, 0.83] | 6.46 | 0.8 %¹ | 2.83 h |

¹ These hits come from a first attempt that ran the three concurrently. That attempt hung the SGLang
scheduler: after ~6 min at ~90 % KV usage with ~48 running requests, the watchdog killed the server and
every dialogue failed. The failed folders are in `runs/_interrupted/`. Replies it had already stored are
valid single draws for their keys, so the store was kept, and the three runs then ran sequentially, as the
frozen grid did.

The success rates are reported, **not read as effects** (Rule 8).

## R1 — can the environment resolve?

| | Coupled (this phase) | Frozen, uncoupled, same dialogues |
|---|---|---|
| **corr (phi), NoEmo vs ActPool, per-dialogue success** | **0.39** [0.20, 0.59] | 0.17 [−0.03, 0.38] |
| κ | 0.39 [0.19, 0.58] | 0.17 [−0.02, 0.37] |
| agreement | 0.74 | 0.67 |
| dialogues where the arms never choose a different act | **35 %** | 3 % |
| first divergence: turn 1 / 2 / 3 / ≥ 4 | 29 / 15 / 12 / 9 | 55 / 25 / 16 / 1 |
| SR difference (NoEmo − ActPool) | −0.02 [−0.11, +0.07] | −0.11 [−0.22, 0.00] |
| detectable difference at n = 100, 80 % power (SR ≈ 0.69) | **0.143** (CI of corr → 0.118–0.164) | 0.183 at corr 0 |

**NoEmo seed spread under coupling:**
- **Seed rates:** 0.68 (seed 1) vs 0.75 (seed 2). The difference, −0.07 [−0.20, +0.05], sits inside the
  frozen seed spread of 0.65–0.76.
- **The two seeds are uncorrelated:** corr 0.05 [−0.13, +0.25]. That is the expected control: different
  seeds share no draws.

What it says:

- **Coupling works, and its reach is bounded by divergence.**
  - **Reach:** 35 % of dialogues are played identically by both arms. ActPool drew a quarter of its LLM
    calls from NoEmo's draws. The correlation more than doubled.
  - **Limit:** 29 % of dialogues still split at the very first planned turn. From there the two arms are
    nearly independent.
- **The brief's "before" of 0.02–0.09 is not what this pair shows.** The uncoupled frozen NoEmo/ActPool
  pair already correlates at 0.17, probably from dialogue difficulty (the persona), which both arms face.
  The 0.02–0.09 range may come from other pairs. This pair is the right reference for this contrast.
- **What resolution was bought.** At 0.39, a paired n = 100 comparison detects differences of about
  **0.14** instead of 0.18. That is better, but still larger than any plausible method effect here. The
  brief's GREEN band (≥ 0.6, ~0.10 detectable) is out of reach while arms diverge this early.
- **Why the prediction was wrong.** 3A's five dialogues all diverged by turn 2; at n = 100, 35 % never
  do. Five dialogues were a direction, not a result (Rule 11), and this is another pilot–population gap.

## Decision level: what retention (#6) does to Q (free, from these logs)

Script `scripts/phase5_decision.py`, numbers `phase5_decision.json`.

**Method.** Inside each run with #6 on, every edge that served a cache hit gets two values:
- **Q as planned:** the Q the planner actually used.
- **Q with the old draw rule:** the cached draws re-weighted the way the frozen cache would have drawn
  them, uniformly over live replies only. The frozen cache could never serve an ended reply.

| Run | Success-reachable edges | Q with the old draw rule → Q as planned | **Shift** | Ended share of cache draws there | Edges with no reachable ending |
|---|---|---|---|---|---|
| P2 NoEmo (25 dlg) | 98 | 0.524 → 0.669 | **+0.145** [0.119, 0.172] | 38 % | 78, shift 0 |
| P2 AffPool (25) | 63 | 0.598 → 0.719 | +0.121 [0.090, 0.156] | 30 % | 54, shift 0 |
| **P4 NoEmo s1 (100)** | 377 | 0.545 → 0.697 | **+0.151** [0.131, 0.173] | 40 % | 282, shift 0 |
| P4 ActPool s1 (100) | 244 | 0.583 → 0.704 | +0.120 [0.100, 0.143] | 31 % | 242, shift 0 |
| **P4 NoEmo s2 (100)** | 374 | 0.573 → 0.707 | **+0.133** [0.120, 0.148] | 39 % | 284, shift 0 |

**Under the frozen cache, Q on every edge where success was reachable was biased down by about 0.12–0.15**
(on a [−1, +1] scale). That is the "Q shifted up" the brief predicted, now measured on five runs. No
cached draw served an ended *failure*, so the correction runs one way. This is a first-order counterfactual:
it re-weights draws but does not replay the extra generations the frozen cache would have triggered, or
the different search that would have followed.

Whether that bias moved **SR** is the Phase 5 question.

## Phase 5 under this gate

The brief allows "Phase 5 on the two most important comparisons only (#6 on NoEmo, GDP-Zero ± #6), 5
seeds". The measured costs are far above the brief's estimate of ~40 h:

| Comparison | Runs | Measured cost per run | Total |
|---|---|---|---|
| NoEmo ± #6, s20, 5 coupled seeds | 10 | ~2.9 h (this phase) | ~29 h |
| GDP-Zero ± #6, s50, 5 coupled seeds | 10 | ~7.2 h (frozen `B1_GDPZero_plain`) | ~72 h |
| **total** | 20 | | **~100 GPU-h (~4.2 days)** |

Even at 5 coupled seeds and corr 0.39, the pooled detectable difference is about 0.14 / √5 ≈ 0.064.
That assumes the seeds are independent replicates, which they are, since they share no draws.

This is 2.5× the brief's estimate, and the verdict is already RED (Phase 3B). The scope of Phase 5 is
put to the human before any run starts.
