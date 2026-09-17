# TASK 1 — Headroom check: can the environment reward action selection?

**Answer: yes, at the shipped budget. The instrument is not broken.** At `n_sims` 50 the root
discriminates strongly — median top-two visit gap **27 of 49** (D1) and **24 of 49** (D2), ties at
**4.0 %** / **5.1 %**, and Q spread below 0.01 at **3.4 %** / **0.6 %** of roots. The
`[+0.200, +0.200]` 60-sim probe that motivated this check is **not representative of the root
population**; it is a tail case at 3.4 % / 0.6 %.

**One escalation, and it is a different one than expected: at `n_sims=10`, 49 % (D1) / 32 % (D2) of
roots are decided by tiebreak** (§4). Decided 2026-09-16: `n_sims=10` is dropped and the low-budget
cell is `n_sims=20` (PREREG Entries 2–3).

Sources: `analysis/wed/selections.parquet` (498,660 rows), `analysis/wed/steps.parquet`,
`analysis/phase1/{D1,D2}/turns.parquet`. D1 = β_emo 0.7, D2 = β_emo 0, both vicuna-13B-AWQ,
`search_horizon` legacy, K 5, R 4, Tmax 10, seed 0, dialogues 101–130 (outside the eval 100).
D1 and D2 are never averaged. CIs are cluster bootstrap over dialogues, 1000 replicates.
Machine-readable: `headroom.json`, `nsims_sensitivity.json` (keyed by exact `n_sims`), `budget_agreement.json`,
`roots_D1.parquet`, `roots_D2.parquet`. Script: `scripts/t1_headroom.py`.

**A note on counts.** A 50-simulation search logs **49** root selections and the final root visits
sum to 49, not 50 — the root expansion consumes one simulation without a logged root selection.
Every "of 49" below is that. Reconstructing final visit counts from `selections.parquet` reproduces
`turns.parquet`'s `root_visits_json` exactly: **0 mismatches over 175 (D1) and 177 (D2) roots.**

---

## 1. Root tie rate — low

Visit counts of the top two root actions, after all simulations.

| | D1 (β 0.7) | D2 (β 0) |
|---|---|---|
| roots | 175 | 177 |
| gap = 0 (decided purely by tiebreak) | **1.1 %** [0.0, 2.6] | **1.7 %** [0.0, 3.7] |
| gap ≤ 1 | 2.3 % [0.5, 4.5] | 1.7 % [0.0, 3.7] |
| gap ≤ 2 | 4.0 % [1.3, 7.1] | 5.1 % [2.4, 8.0] |
| gap ≤ 5 % of visits (≤ 2.45) | 4.0 % [1.3, 7.1] | 5.1 % [2.4, 8.0] |
| **median gap** | **27** | **24** |
| mean gap | 25.0 | 24.0 |
| top-1 visit share, median | 0.73 | 0.69 |

Gap distribution (quantiles, visits of 49):

| percentile | 5 | 25 | 50 | 75 | 95 |
|---|---|---|---|---|---|
| D1 | 3 | 11 | 27 | 38 | 46 |
| D2 | 3 | 10 | 24 | 38 | 44 |

**Split by `search_horizon`: not available.** Both runs predate the flag (`search_horizon` is absent
from D1/D2's logged args; D3 records `legacy`). Every row here is legacy. The `episode` comparison is
Task 3's job, not something these logs can answer.

**Ties are not concentrated anywhere.** By root turn, the median gap stays between 19 and 37 (D1) and
between 8.5 and 39 (D2), with no monotone decay into the late turns where the horizon bug bites
hardest. `corr(gap, n_expanded)` is 0.08 / 0.02 — the tie rate is not an artefact of how many actions
got expanded.

## 2. Root Q spread — not degenerate

`max(Q) − min(Q)` over expanded actions (N > 0), after folding in the final simulation's backup from
`steps.parquet` so the spread is the one the final decision saw.

| | D1 | D2 |
|---|---|---|
| **spread < 0.01** (tiebreak, not search) | **3.4 %** [1.1, 6.6] | **0.6 %** [0.0, 1.9] |
| median spread | **0.596** | **0.694** |
| mean spread | 0.567 | 0.665 |
| quantiles 5 / 25 / 75 / 95 | 0.03 / 0.27 / 0.82 / 1.05 | 0.12 / 0.38 / 0.91 / 1.29 |
| roots with only one expanded action | 0 % | 0 % |
| expanded actions per root (of 5) | 2:32, 3:55, 4:19, 5:69 | 2:9, 3:44, 4:32, 5:92 |

On a [−1, +1] value scale a median spread of 0.6–0.7 is a large signal. **No root had fewer than two
expanded actions.** The probe's `[+0.200, +0.200]` — two actions pinned to the leaf value — happens,
but in 3.4 % / 0.6 % of roots.

**Search is not just re-reading the prior.** The visit-argmax differs from the prior-argmax at
**31 %** (D1) and **45 %** (D2) of roots. It differs from the Q-argmax at 35 % / 23 %, which is
expected — visits follow UCT, not Q alone.

## 3. Does the action matter?

**I could not replay random action selection from the logs, and say so plainly.** The logs record the
tree the planner built, not the environment's response to a counterfactual act; scoring a random
policy needs the user simulator and the value model back in the loop, i.e. a GPU run, not an
analysis. That is a real gap in this answer.

What the logs do support:

| run | SR | Wilson 95 % | AvgT |
|---|---|---|---|
| D1 (β 0.7) | 0.600 (18/30) | [0.423, 0.754] | 5.83 |
| D2 (β 0) | 0.733 (22/30) | [0.556, 0.858] | 5.90 |

**The arms sit well inside each other's intervals at 30 dialogues**, exactly the concern the brief
raised. The gap is 0.133 SR against a Wilson half-width of roughly ±0.16 per arm. At 30 dialogues
this design cannot resolve differences of the size the arms are expected to produce. That is an
argument about **statistical power at the planned dialogue count**, and it is independent of the
instrument question — the instrument is fine; the sample is small.

Indirect evidence that the act choice is consequential: **the simulation budget changes the chosen
action** at 26.3 % (D1) and 35.0 % (D2) of roots between `n_sims=10` and 50 (§4). A budget that changed nothing downstream would not
move the argmax on a quarter to a third of decisions.

## 4. Where tree search starts working: tiebreak rate by budget

**This is a result in its own right.** It measures, on real LLM-planner trees (vicuna-13B, K = 5
actions, sampled value estimator, R = 4), the simulation budget below which a GDP-Zero-style open-loop
planner stops deciding by search and starts deciding by tiebreak. I have not checked the literature for
a published equivalent, so I make no priority claim.

**Method: exact truncation replay.** UCT selection at simulation *i* depends only on the statistics from
simulations 1…*i*−1. So the first *n* simulations of a logged 50-simulation search are the search an
*n*-simulation run would have made under the same seed. No re-running is needed. **Budget
bookkeeping:** a search of `n_sims` simulations makes `n_sims − 1` root selections, because one
simulation expands the root. The first version of this section tabulated 10 root **visits** as
`n_sims=10`, which is actually `n_sims=11`. The table below is at the exact `n_sims`, and PREREG
Entry 3 records the correction.

**Share of roots decided by tiebreak** (Q spread < 0.01 over expanded root actions), 95 % cluster
bootstrap CIs over 30 dialogues:

| `n_sims` | root visits | **D1 (β 0.7)** | **D2 (β 0, NoEmo)** | D1 gap ≤ 2 | D2 gap ≤ 2 | median expanded (of 5), D1 / D2 | median Q spread, D1 / D2 |
|---|---|---|---|---|---|---|---|
| 5 | 4 | **59.4 %** [48.6, 71.3] | **43.5 %** [33.7, 53.5] | 44.6 % | 60.5 % | 1 / 2 | 0.00 / 0.10 |
| **10** | 9 | **49.1 %** [37.6, 61.9] | **31.6 %** [24.9, 39.9] | 19.4 % | 27.1 % | 2 / 2 | 0.03 / 0.38 |
| 15 | 14 | 40.6 % [29.0, 53.0] | 22.6 % [16.7, 30.1] | 13.1 % | 14.7 % | 2 / 3 | 0.24 / 0.47 |
| **20** | 19 | **28.0 %** [18.0, 39.4] | **13.0 %** [7.8, 19.6] | 9.1 % | 10.2 % | 2 / 3 | 0.35 / 0.55 |
| 25 | 24 | 18.3 % [11.6, 26.4] | 5.1 % [2.1, 8.9] | 6.9 % | 11.3 % | 2 / 3 | 0.40 / 0.58 |
| 30 | 29 | 9.7 % [4.6, 15.7] | 1.7 % [0.0, 3.8] | 3.4 % | 9.0 % | 3 / 4 | 0.48 / 0.61 |
| 40 | 39 | 2.3 % [0.5, 4.8] | 0.6 % [0.0, 1.9] | 2.3 % | 6.8 % | 3 / 4 | 0.55 / 0.66 |
| **50** | 49 | **2.3 %** [0.5, 4.9] | **0.6 %** [0.0, 1.9] | 4.0 % | 5.1 % | 4 / 5 | 0.60 / 0.69 |

The 50 row is the truncation replay at 49 root visits, using the Q each root had when its last
selection was made. §2's 3.4 % / 0.6 % additionally folds in that last simulation's backup, which is
the Q the final decision actually saw. The two differ by one backup and one root (D1 2.3 % vs 3.4 %).
Within this table every row is computed the same way, so the rows compare like with like.

Reading it:

- **The transition is between 20 and 30 simulations for NoEmo** (13.0 % → 1.7 %), and between 30 and 40
  under Bias β 0.7 (9.7 % → 2.3 %). Below 15 at least a fifth of NoEmo roots, and 40 % of Bias roots, are
  decided without any Q signal.
- **The affect term delays the transition.** At every budget below 40, D1 has roughly twice D2's
  tiebreak share. This is the Bias narrowing documented in `p_var.md` §7.6: fewer siblings get expanded,
  so fewer have a Q to compare. At `n_sims=20` the gap is 28.0 % against 13.0 %.
- **The known hard trap does not fire at K = 5.** `W5_COST_TABLE.md`'s trap (`n_sims` ≤ #actions →
  uniform counts → argmax returns action 0) needs `n_sims` ≤ 5. Here the share of roots choosing action 0
  (`credibility appeal`) is flat across budgets: D1 18.3 / 16.6 / 16.6 % at 10 / 20 / 50, and D2 16.4 /
  15.3 / 14.7 %. **The failure at low budget is missing Q signal, not a pinned argmax.**
- **Budget changes the decision.** The `n_sims=10` pick agrees with the 50-simulation pick on 73.7 % (D1)
  / 65.0 % (D2) of roots, and the `n_sims=20` pick on 82.9 % / 73.4 %.

### Decision (human, 2026-09-16): `n_sims=10` dropped; low-budget cell is `n_sims=20`

E4 becomes `n_sims ∈ {20}`, and E1's 50 is the other point on the curve. PREREG Entry 2 amends the
prediction, and Entry 3 corrects the figures.

**⚠ Carry-over to the grid: 20 is much better than 10, but it is not clean.** At `n_sims=20`, **13 %** of
NoEmo roots and **28 %** of roots under Bias β 0.7 are still tiebreak-decided. For the pooling arms (β 0)
the relevant baseline is the 13 %. For any selection arm compared at `n_sims=20`, a meaningful share of
its decisions is still arbitrary. Budget 25 would bring NoEmo to 5.1 % and Bias to 18.3 %. **Recorded
here; not a request to revisit the decision.**

---

## What this means for Friday — reported, not decided

1. **The grid is worth starting at `n_sims=50`.** Root discrimination is healthy: ties at 4–5 %, median Q
   spread 0.6–0.7, and search overturning the prior on a third to half of decisions. The premise that
   "no arm can show anything" is **not supported** at the shipped budget.

2. **The low-budget cell is now `n_sims=20`** (§4 decision). TrajValue's "does NOT depend on `n_sims`"
   prediction is a clean contrast here, because the value channel is exactly what low budget starves.

3. **Power, not the instrument, is the binding constraint on the headline comparison.** D1 and D2 differ
   by 0.133 SR with fully overlapping Wilson intervals at n = 30. Plan the arm comparisons against the
   grid's dialogue count. The co-primary AvgT (PREREG) is the better-powered outcome.

4. **Question 3 is not fully answered.** Answering "does the action matter" directly, rather than by
   inference, needs a GPU run with a random-act arm on the same dialogues. It is not in the Thursday plan.
   I flag it as a known hole rather than silently substituting the arm-variance proxy.
