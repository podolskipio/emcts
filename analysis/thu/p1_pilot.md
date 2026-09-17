# TASK 3 — P1 search-horizon pilot: `legacy` vs `episode`

**⚠ Escalation (brief trigger: "`episode` moving SR materially").** On 10 paired dialogues,
**`episode` succeeds on all 10 and `legacy` on 6.** All 4 discordant pairs favour `episode` and none go
the other way. `episode` also ends dialogues sooner (AvgT 5.4 against 7.2) and **cuts wall clock by 42 %**.

**Recommendation: `--search_horizon episode` as the whole-grid default.** Not set — the choice is the
human's, and it must be the same for every grid run.

Design (supersedes `analysis/phase1/pilot_horizon/README.md`, see its header): NoEmo (β 0), the D1/D2
diagnostic config otherwise (vicuna-13B-AWQ, sampled scoring, n_sims 50, K 5, R 4, Tmax 10, persona, HF
classifier, 10 workers, seed 0). Dialogues **131–140**, disjoint from eval 1–100 and from D1–D3. Both
arms ran **fresh and serially** on one supervised server with no restarts: legacy 21:40–22:47, episode
22:47–23:26. Runs: `analysis/thu/runs/P1_{legacy,episode}`. Numbers: `p1_metrics.json`,
`p1/pilot_results.json`, `p1/pilot_tables.md`. Scripts: `scripts/t_pilot_metrics.py`,
`analysis/phase1/pilot_horizon/analyze_pilot.py`.

## A. Correctness of the fix — PASS

| check | result |
|---|---|
| `metadata.json` records `search_horizon = episode` | ✅ |
| steps whose child state is past Tmax | **0** of 6,360 (legacy: 12.3 % of 9,372) |
| selections at a state of length ≥ Tmax | **0** |
| simlog alignment check | PASS |
| max tree depth | **9** (legacy: 20) |

## B. Outcomes (paired, n = 10)

| | legacy | **episode** |
|---|---|---|
| **SR** | 0.60 (6/10) [0.31, 0.83] | **1.00 (10/10)** [0.72, 1.00] |
| **AvgT** (runner metric; failures count as Tmax) | 7.2 | **5.4** |
| mean turns of the episode record | 6.9 | 5.4 |
| discordant pairs: success only here | 0 | **4** |

| dialogue | legacy | episode | first divergent system turn |
|---|---|---|---|
| …144300_705 | ✗ T=10 | ✓ T=5 | 1 |
| …161442_405 | ✗ T=10 | ✓ T=6 | 1 |
| …185320_572 | ✗ T=10 | ✓ T=5 | 2 |
| …213210_462 | ✗ T=7 (user refused) | ✓ T=6 | 1 |
| …145857_925 | ✓ 5 | ✓ 4 | 2 |
| …155340_989 | ✓ 4 | ✓ 6 | 1 |
| …163111_795 | ✓ 8 | ✓ 8 | 2 |
| …164559_390 | ✓ 5 | ✓ 5 | identical |
| …190212_442 | ✓ 3 | ✓ 5 | 2 |
| …190447_206 | ✓ 7 | ✓ 4 | 1 |

**Read this as strong direction, weak magnitude.** An exact sign test on 4–0 discordant pairs gives
p = 0.125, two-sided. Ten dialogues cannot pin the size of the SR effect: the true gain could be a few
points or tens of points. What P1 does establish is that the two horizons produce **materially different
planners**. Trajectories diverge at system turn 1 or 2 in 9 of 10 dialogues.

### Mechanism: the legacy planner loops until the clock runs out

The mechanism is **not** "asks for the donation sooner". `episode` proposes donation *less*:
`proposition of donation` is 11.1 % of its acts against 13.0 %, and 4 of 10 episode dialogues never
propose at all, yet the user donates. What differs is **repetition**:

| failed under legacy | legacy system acts | episode acts (success) |
|---|---|---|
| 705 | greet/credi/**emoti/emoti**/task/**emoti/emoti/emoti**/task/**emoti** → T=10 | greet/emoti/credi/task/emoti |
| 405 | greet/propo/logic/greet/propo/logic/**emoti/emoti/emoti/emoti** → T=10 | greet/emoti/propo/credi/task/emoti |
| 572 | greet/credi/credi/task/propo/**emoti/emoti/emoti/emoti/emoti** → T=10 | greet/credi/emoti/emoti/other |

`emotion appeal` is 39.1 % of legacy's acts against 33.3 % for episode. Under `legacy` a simulated branch
that reaches the turn limit is neither stopped nor scored −1 (`SEARCH_HORIZON_BUG.md` §1). Its value
comes from `predict` on a longer and longer conversation, so a loop costs nothing inside the tree. Under
`episode` the same branch backs up −1, and search steers away from stalling. **The legacy planner does
not see the deadline that the environment enforces.**

**A caveat on what SR measures.** Four episode successes contain no donation proposal. The simulated
persuadee donates after engagement alone. SR on this simulator partly measures "avoid stalling until the
simulator agrees" rather than persuasion strategy. This does not change the horizon recommendation: the
bug is real and the fix is correct either way. It does bear on how arm differences in SR are read.

## C. Search structure and the root-Q mechanism

| | legacy | episode |
|---|---|---|
| search roots (planned turns) | 59 | 44 |
| steps per tree | 158.8 | 144.5 |
| share of steps with child exactly at Tmax | 7.0 % | 0.9 % |
| simulations ending in horizon failure (−1) | 0.3 % | 1.9 % |
| **root Q spread < 0.01** (tiebreak) | **5.1 %** | **4.5 %** |
| root Q spread, median | 0.588 | 0.631 |
| root top-two gap ≤ 2 / median | 3.4 % / 17 | 4.5 % / 20.5 |
| expanded root actions, median | 4 | 3.5 |
| **turn-1 roots only** (identical states, like-for-like), spread median / < 0.01 | 0.714 / 0 % | 0.807 / 0 % |
| selection points with ≥ 2 expanded siblings | 48.6 % | 49.2 % |
| fresh (non-cached) child share | 56.0 % | 54.3 % |

**The brief's predicted mechanism is not what happened.** The 60-sim probe suggested root Q would move
"from degenerate to discriminating". But the legacy root was not degenerate here (5.1 % sub-0.01, median
spread 0.59), and it barely changes under `episode` (4.5 %, 0.63). The SR effect runs through **which
branches look good** (loops stop looking free), not through whether the root can discriminate at all.
This agrees with Task 1 (`headroom.md`), which found legacy roots healthy at 50 simulations.

**On the "common 2669-root subset".** That figure belongs to the **replay** runners (`emomcts.py` over
corpus prefixes), where both horizons search from the same 2,697 corpus states and `episode` skips 28
past the horizon. P1 is **self-play** (`rollout.py`), where trajectories diverge by turn 2 and no common
state set exists beyond the turn-1 roots. Those are compared above. No past-horizon root occurs in
self-play, because the episode ends at Tmax.

## D. Cost — the number the grid budget depends on

| | legacy | **episode** | change |
|---|---|---|---|
| **wall clock, 10 dialogues** | 4,003 s | **2,306 s** | **−42 %** |
| per dialogue | 400 s | **231 s** | −42 % |
| per planned turn | 58.0 s | 42.7 s | −26 % |
| LLM `predict` calls per search root | 44.9–45.2 | **39.4–40.9** | −10 % |
| `predict` calls total | 2,652–2,668 | 1,733–1,798 | −34 % |
| SGLang prefill sequences | 32,414 | 18,402 | −43 % |
| **prefix-cache hit rate** | 80.8 % | 80.5 % | −0.3 pt |

(`predict` calls are reconstructed from tree structure: every simulation ends in one leaf expansion or
at a terminal. The bracket is the few leaves that cannot be classified; `t_pilot_metrics.py`.)

**The saving has two sources:** cheaper searches (−26 % per turn, from shallower trees and fewer
predicts) and shorter dialogues (5.4 against 7.2 turns). The second depends on the arm's outcomes, so
**a whole-grid saving of ~40 % is plausible for NoEmo-like arms but must be re-measured per arm**
(Task 9.1). Cache efficiency is unchanged.

## E. What else `episode` moves — freeze-relevant

Same `p_var.py` components on both arms (dialogue-clustered CIs; `p1/pilot_tables.md` §D):

| | legacy | episode |
|---|---|---|
| τ_med (median parent ν) | 0.404 | **0.279** |
| NodeKey ω² | 0.151 [0.081, 0.190] | **0.035** [0.023, 0.064] |
| **AffPool between/within ratio** (z) | 0.714 | **0.983** |
| AffPool evidence multiplier | 2.38 | 2.00 |
| Bias β 0.7 flip rate on these trees | 16.5 % | **11.9 %** |

1. **AffPool's homogeneity gate (≤ 1.0) is at 0.98 under `episode`**, on 10 dialogues, with z as the
   pooled quantity. Wednesday's gate on the task return (the quantity AffPool actually pools) was
   measured on legacy trees only. **Under `episode` that gate must be re-measured before AffPool/ActPool
   enter the grid.** The episode ActPool/AffPool pilots now queued provide the trees for it.
2. **τ_med drops from 0.40 to 0.28.** AffPool's frozen `--aff_pool_tau 0.35` is D1's legacy τ_med. Under
   `episode` it would no longer be a median split.
3. **Doses depend on the horizon.** Matched on episode trees, CenteredBias is β 1.06 and Momentum β 1.19
   against Bias 0.7 (11.9 % flips). On legacy trees the matches were 1.11 and 1.31. The relative dosing is
   similar, but **the β values in FREEZE_NOTES must come from trees grown under the chosen horizon**
   (`dose_replay_P1_episode.json`).
4. Per `SEARCH_HORIZON_BUG.md`, if P1 moves the Phase-1 gate components materially (it does: ω² falls 4×,
   the AffPool ratio approaches 1), **the D1/D2-based gate readings are not decision-grade under
   `episode`.**

## Recommendation (not applied)

**Adopt `--search_horizon episode` for the whole grid.**

- It is the correct environment: trees now respect the horizon that the rollout enforces. The pass on A
  is exact, not statistical.
- Every outcome difference points one way: 4–0 on SR, −1.8 turns AvgT.
- It is 42 % cheaper per NoEmo dialogue at an unchanged cache hit rate.
- **Mixing horizons across runs would invalidate every cross-arm comparison.** A legacy grid would bake
  in a planner that cannot see the deadline, and its arm differences would partly measure how each arm
  interacts with that blind spot.

**If adopted, these follow:** re-set AffPool τ from episode trees; re-measure the AffPool gate on the task
return under episode; take dose-matched β from episode trees; re-run the Task 9.1 cost measurement under
episode.

**Pilot queue switched to `episode`** (23:34) so these follow-ups have trees to work from: Bias β 0.7,
Momentum β 1.2, CenteredBias β 1.1 (episode-matched), ActPool, AffPool, all on 131–140. A partial legacy
Bias run, started 23:26 and stopped after one minute, is kept as `runs/M5_bias_legacy_ABORTED`. **This
choice concerns only which horizon the diagnostic pilots measure. It does not set the grid default.** If
the human chooses `legacy`, the legacy-tree doses (`dose_replay_P1_legacy.json`) and `P1_legacy` remain
the reference, and those pilots need re-running under legacy (~5 GPU-h).
