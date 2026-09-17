# Search-horizon bug: tree search does not stop at the episode horizon

**Status (2026-09-14).** Fixed behind `--search_horizon episode` (default `legacy`, bit-identical to
before). Verified by unit tests and stub end-to-end runs. **Pilot P1 (10 dialogues) is scheduled for
Thursday 2026-09-17.** Disclosed regardless of what the pilot shows.

## 1. The bug

`get_dialog_ended` returns +1.0 on donation, −1.0 on failure (turn limit `len(state) >=
max_conv_turns`, or a verbatim stall), and 0.0 otherwise. The rollout loop
(`runners/rollout.py:rollout_one`) ends an episode on any non-zero value, or at `--max_turns`.

Every tree search treats only success as terminal:

| planner | terminal test |
|---|---|
| `OpenLoopMCTS.search` (GDP-Zero) | `terminated_v == 1.0` |
| `EmotionAwareOpenLoopMCTS.search` | `terminated_v == 1.0` |
| `EmotionAwareMultiObjectiveQ.search` (EmoMCTS) | `terminated_v == 1.0` |
| `MCTS.search` (closed loop) | `terminals > 0` |

So a simulated branch that reaches the turn limit is neither stopped nor scored −1:

- the node is **expanded**, and `planner.predict` spends its LLM calls on a prior and a value for it;
- **selection continues below it**, generating system and user turns 11, 12, … up to the 50-simulation
  budget;
- values from those states are **backed up** into the Q (and Q_emo) of reachable ancestors.

The same happens after a verbatim stall. **`Tmax` does not bound search depth at all.** In D1 a
search rooted at turn 8 reached depth 33, a simulated dialogue 41 turns long under `--max_turns 10`.

**History.** Both halves are present in the initial commit (`bbfaa42`, 2026-05-12). That commit's
`get_dialog_ended` already returns −1.0 at the turn limit and on "no donation", while search
terminates only on `== 1.0`. W5 "fix 5" (`analysis/W5_COST_TABLE.md`) wired `max_conv_turns` to
`--max_turns` and described it as stopping MCTS from "simulating turns 11–15, which the dialogue can
never reach". It could not have had that effect on search, because the −1.0 it produces is ignored
there.

The false "depth cap" description is repeated in `analysis/calib/run_calib.py`, `BUDGET.md`,
`PERSONA_PENALTY.md`, the `build_agents` docstring and a comment in `rollout.py`. The two code
comments are corrected; the docs carry dated errata pointing here.

## 2. Correction of my own earlier statement

`phase1_report.md` and `p_var.md` (first versions, 2026-09-14) said the depth-cap failure touched
"under 1 % of visits". **That was wrong.** It counted rows at tree depth > 16. Tree depth is measured
from each turn's root, so a state is impossible once *root turn + depth* exceeds Tmax.

Measured correctly on the Phase-1 logs:

| | D1 (β 0.7) | D2 (β 0) |
|---|---|---|
| step rows whose child state is past Tmax (turn + depth > 10) | **19.7 %** | **12.7 %** |
| step rows whose child is exactly at Tmax (valued by `predict`, not −1) | 7.7 % | 6.9 % |
| simulations that reach an impossible state | **22.6 %** | 15.0 % |
| simulations touching a state at or past Tmax | 30.4 % | 22.4 % |
| trees containing an impossible state | **75 %** (132/175) | 50 % (89/177) |
| longest simulated dialogue | 41 turns | 27 turns |

Share of rows past Tmax, by root turn:

| root turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.02 | 0.04 | 0.10 | 0.17 | 0.20 | 0.22 | 0.33 | 0.59 | **0.76** |
| D2 | 0.00 | 0.01 | 0.05 | 0.07 | 0.11 | 0.16 | 0.24 | 0.44 | **0.71** |

Late-dialogue decisions are the most exposed. At turn 9, the last planned turn, three quarters of the
search happens in states that cannot occur.

## 3. What is affected

**Every MCTS result produced with this repository used the legacy rule**, for both GDP-Zero and
EmoMCTS:

- the Phase-1 diagnostics D1/D2;
- the calibration runs and the grid cost table (`analysis/calib/`);
- the README results;
- the paper's 40-sim runs behind §IV-D.

Those May runs predate fix 5: the game horizon was 15 against a rollout loop of 10, and search ignored
it anyway.

The replay runners (`emomcts.py`, `gdpzero*.py`) are affected the same way at their horizon of 15.

Because both planners share the rule, EmoMCTS-vs-GDP-Zero comparisons are like-for-like. **Each
planner, however, values actions partly on dialogue states the episode never reaches.** Whether that
changes decisions, outcomes or the comparison is what P1 measures.

## 4. Phase-1 sensitivity: gate components with the impossible rows removed

`scripts/build_steps.py --max_child_len 10` drops step rows whose child is past Tmax, recomputes
τ_med on the remaining rows, and reruns the same `p_var.py` (1000 dialogue-clustered replicates).
Results are in `horizon_sensitivity/`.

This is a **lower bound** on the bug's effect. The retained rows' Q values and selections were still
shaped by the dropped states. Only a run under `--search_horizon episode` removes that.

| median split | D1 all rows | D1 child ≤ Tmax | D2 all rows | D2 child ≤ Tmax | threshold |
|---|---|---|---|---|---|
| rows / edges | 33,903 / 5,355 | 27,228 / 3,779 | 28,129 / 5,312 | 24,543 / 4,219 | |
| τ_med | 0.350 | 0.297 | 0.313 | 0.293 | |
| discordance rate | 0.308 [0.284, 0.338] | **0.352** [0.325, 0.386] | 0.249 [0.220, 0.282] | 0.275 [0.247, 0.308] | ≥ 0.30 |
| within-prefix Δ | −0.055 [−0.067, −0.043] | −0.044 [−0.055, −0.031] | −0.055 [−0.069, −0.038] | −0.047 [−0.060, −0.034] | CI ∌ 0 |
| standardized effect | −0.275 [−0.33, −0.22] | **−0.215** [−0.27, −0.15] | −0.268 [−0.34, −0.19] | −0.232 [−0.29, −0.17] | \|·\| ≥ 0.2 |
| **pooled ω²** | 0.084 [0.055, 0.127] | **0.041** [0.029, 0.057] | 0.083 [0.063, 0.102] | **0.058** [0.045, 0.076] | ≥ 0.05, lower bound > 0 |
| edge-demeaned ω² | 0.0066 | 0.0037 | 0.0077 | 0.0030 | |
| fresh steps: std effect / ω² | −0.395 / 0.123 | −0.350 / 0.064 | −0.359 / 0.106 | −0.297 / 0.067 | |
| boundary fraction ±0.1 | 0.221 | 0.229 | 0.214 | 0.218 | ≤ 0.35 |
| bucket–depth r | −0.290 [−0.334, −0.245] | **−0.210** [−0.259, −0.155] | −0.240 | −0.166 | \|r\| ≤ 0.3 |
| depth-specific median split: Δ / ω² | −0.046 / 0.069 | −0.043 / 0.044 | −0.043 / 0.070 | −0.041 / 0.058 | |
| post-split cells with ≥ 3 visits | 43 % | 45 % | 36 % | 38 % | m = 3 |
| AffPool median n_prefixes | 3 | 3 [2, 3] | 3 | 3 [2, 3] | ≥ 4 |
| AffPool evidence multiplier | 2.59 [2.0, 3.0] | **2.00** [1.9, 2.0] | 2.50 | **2.00** | ≥ 3 |
| AffPool top-prefix share | 0.625 | **0.709** | 0.643 | 0.667 | ≤ 0.6 |
| AffPool between/within ratio | 0.785 | 0.703 | 0.783 | 0.759 | ≤ 1.0 |
| argmax flip, full PUCT | 0.316 | 0.328 | 0.155 (counterfactual) | 0.163 | |
| flips picking visited over unvisited | 83 % | 81 % | 43 % | 40 % | |
| label split: discordance / ω² | 0.061 / 0.007 | 0.077 / 0.008 | 0.054 / 0.012 | 0.062 / 0.009 | |
| σ_emo/\|Q_emo\| median / p90 | 0.45 / 1.45 | 0.55 / 1.55 | 0.50 / 1.46 | 0.55 / 1.53 | |

What changes:

- **Pooled ω² roughly halves in D1 and drops below the proposed 0.05.** Much of it came from the deep,
  impossible, happiness-saturated states (ν → 0.53), where bucket and depth are entangled. Its
  edge-demeaned part stays below 1 %.
- **The within-prefix effect weakens to about the 0.2 standardized threshold,** but still excludes 0
  in both runs.
- **Discordance rises**, because the dropped deep edges rarely saw both buckets. **The depth confound
  eases.**
- **AffPool gets worse on evidence:** multiplier 2.0, top-prefix share 0.71.
- **Channel-dominance conclusions are unchanged.**

**The Phase-1 gate readings in `phase1_report.md` §1 were computed on contaminated trees. The NodeKey
pooled-ω² component and the AffPool evidence components are sensitive to this bug.** No gate decision
should rest on the D1/D2 numbers until P1 shows how far a bounded search moves them. If they move
materially, D1/D2 must be re-run under `--search_horizon episode`.

## 5. The fix

`--search_horizon {legacy,episode}` on every MCTS runner (`rollout.py`, `emomcts.py`, `gdpzero.py`,
`gdpzero_noRS.py`, `gdpzero_noopenloop.py`), threaded through the MCTS configs and recorded in
`metadata.json`.

- **`legacy` (default):** exactly the old tests (`== 1.0`; `> 0` closed loop). No default flag value
  changed.
- **`episode`:** any non-zero `get_dialog_ended` ends the branch and its value is backed up. Donation
  gives +1. The turn limit or a stall gives −1, the same failure the episode is scored with.
  Implemented once as `MCTS._ends_search` and used by all four search paths.

**A design choice to note.** `episode` backs up −1 at the horizon, which matches how the episode is
actually scored (a dialogue that reaches Tmax without donating is a failure). The alternative is to
stop at Tmax but value the state with `planner.predict`. That would bound depth without the
failure signal, and it was not implemented. Under `episode`, the last planned turn's search becomes a
one-step lookahead against a ±1 outcome, so near-horizon decisions can shift toward acts that close.
P1 will show whether they do.

**Verification.**

- `tests/test_search_horizon.py`, **11 tests, all passing.** They drive the real games and planners
  with fake text sources:
  - legacy is exactly the pre-change comparison, value by value;
  - legacy search demonstrably goes past the turn limit, and `episode` never transitions out of a
    state at the limit;
  - `episode` backs up the −1;
  - `episode` stops on a verbatim stall;
  - donation is terminal under both;
  - the emotion channel still backs up the edge into a terminal child;
  - the plain open-loop and closed-loop planners respect the flag;
  - an unknown value is rejected.

  The 39 existing freeze tests still pass.
- **Stub end-to-end** (`tests/run_e2e_regression.py`, D1 flags, R 4, 3 dialogues, seed 0):
  - **default flag bit-identical to the pre-change tree** at β 0.7 and β 0: SR, AvgT, action sequences,
    full episode records, subtree logs, classifier records;
  - `--search_horizon episode`: **0 steps past Tmax**, against 484 (β 0.7) and 3,026 (β 0) under legacy
    on the same dialogues; simlog alignment passes; flag recorded in metadata.
- **Source files changed:** `src/mcts/mcts.py` and `src/mcts/emotion_mcts.py` (terminal rule);
  `src/runners/{rollout,emomcts,gdpzero,gdpzero_noRS,gdpzero_noopenloop}.py` (flag);
  `src/runners/_common.py` (docstring).

**Cost.** Bounded trees skip expanding unreachable nodes, so `episode` should be cheaper per turn.
The calib cost table was measured under legacy and would need re-pricing if the grid adopts
`episode`.

## 6. Pilot P1 (Thursday 2026-09-17)

The plan and commands are in `pilot_horizon/README.md`:

- D1's config + `--search_horizon episode` on D1's first 10 dialogues, paired against D1 itself;
- a correctness gate, paired outcomes (descriptive only at n = 10), search structure and cost;
- the P-VAR components recomputed with the same scripts;
- about 1 GPU-h.

*Result: pending. Append it here, whichever way it comes out.*

**Result (run 2026-09-16; full report `analysis/thu/p1_pilot.md`).** The design changed before the run:
NoEmo (β 0), with legacy and episode both run fresh on dialogues 131–140, so that the horizon is isolated
from the affect channel and wall clock is like-for-like. See the header of `pilot_horizon/README.md`.

- **Correctness gate: PASS.** 0 of 6,360 steps past Tmax (legacy: 12.3 %). Max depth 9 (legacy: 20).
  Alignment check passes.
- **Outcomes: SR 1.00 (10/10) under episode against 0.60 (6/10) under legacy.** All 4 discordant pairs
  favour episode (exact sign test p = 0.125). AvgT 5.4 against 7.2. Three of the four legacy failures
  are repetition loops (4–6 consecutive `emotion appeal`) that run to T = 10. Episode search stops
  treating those loops as free.
- **Cost: −42 % wall clock** (2,306 s against 4,003 s), −10 % `predict` calls per search root, prefix-cache
  hit rate unchanged (80.5 % against 80.8 %).
- **Gate components move materially:** NodeKey ω² 0.151 → 0.035, AffPool z-ratio 0.71 → 0.98, τ_med
  0.40 → 0.28. Per §4, the D1/D2-based gate readings are not decision-grade under the fix.
- **Recommendation (human decision pending):** `--search_horizon episode` for the whole grid.

## 7. Draft disclosure text (paper / appendix)

> **Search horizon.** In the open-loop MCTS inherited from GDP-Zero, simulated branches terminated
> only on user donation; reaching the dialogue turn limit (T = 10) or a repetition loop returned a
> failure value that the search did not treat as terminal. Tree search therefore expanded and valued
> dialogue states beyond the horizon that no episode can reach. In our diagnostic runs this affected
> 13–20 % of simulation steps and up to 76 % of the search at the final planned turn. The behaviour
> is identical for GDP-Zero and EmoMCTS, so the planners were compared under the same search. We
> provide a corrected search (`--search_horizon episode`) that terminates branches exactly where
> episodes terminate, and report [P1 / re-run results] under it.
