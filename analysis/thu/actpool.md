# TASK 4 — ActPool (`--aff_pool --aff_pool_key act`)

**Status: built, accepted, piloted under `episode`.** The flag defaults to AffPool exactly as before.

## 1. Build

`--aff_pool_key {affect,act}`, default `affect`, declared in both runners (`rollout.py`, `emomcts.py`)
and passed to `EmotionAwareMultiObjectiveQ(aff_pool_key=...)`. Under `act`, `_pool_bucket` returns 0
for every parent. The pool is then keyed by the act alone, and everything else is shared code: the RAVE
blend (`_pool_blend`), per-search scope, `--aff_pool_bias`, the backup of the task return into
`Q_pool`, and all AffPool diagnostics (`aff_bucket`, `pool_beta_selected`, `pool_flip`, and per-sibling
`Q_pool` / `N_pool` / `pool_beta`). An unknown key raises `ValueError` in the planner and an argparse
error in the runners.

## 2. Acceptance

| criterion | result |
|---|---|
| `--aff_pool` off ⇒ bit-identical to NoEmo | ✅ Whole-search sha256 equals the pre-change GOLDEN at β 0 and 0.7, with `aff_pool_key="act"` passed and ignored (`tests/test_thu_arms.py::test_pool_off_is_noemo_golden`) |
| `affect` ⇒ bit-identical to today's AffPool | ✅ Fingerprints equal the snapshot taken from the tree **before** the flag existed, at all 6 (β, bias) combinations (`prechange_affpool_fingerprints.json`). Explicit `affect` equals the implicit default on an extended fingerprint that includes `Q_pool`, `N_pool` and every step field |
| stub end-to-end regression, pre vs post tree | ✅ **IDENTICAL** at β 0, β 0.7 and AffPool β 0.7: SR, AvgT, action sequences, full episode records (`analysis/thu/regression/`) |
| `act` pools one cell per act | ✅ Every step has bucket 0. Each cell's `Q_pool` is the mean of its steps' `backup_value`, and `N_pool` is their count. The `act` cell is exactly the union of the two `affect` cells |
| `act` ⇒ generation count unchanged | ✅ **454 prefill sequences per search**, against 457 for AffPool (−0.7 %). New tokens per search are 101k against 82k (AffPool's dialogues were shorter, below) |
| `act` ⇒ prefix-cache hit rate unchanged | ≈ **83.2 %** against AffPool's 86.1 % (−2.9 pt), but +2.7 pt above NoEmo's 80.5 %. Across all six episode pilots the hit rate spans 79.2–86.1 % with dialogue composition, so the gap is inside the between-arm spread. There is no sign that the key change affects caching. Stated plainly: it is not within 2 points of AffPool on this pilot |
| test suite | 103 passed (69 pre-existing + 34 new) |

## 3. Pilot under `episode` (dialogues 131–140, β 0, bias 0.1)

Same dialogues, config and horizon for all three. NoEmo = `P1_episode`.

| | NoEmo | **ActPool** | AffPool |
|---|---|---|---|
| dialogues | 10/10 | 10/10 | 10/10 |
| SR | 1.00 | **0.60** | 0.90 |
| AvgT | 5.4 | 6.5 | 5.0 |
| wall clock / dialogue | 231 s | 280 s | 218 s |
| wall clock / search | 52.4 s | 50.8 s | 54.4 s |
| `predict` calls / search | 39.4–40.9 | 33.5–37.8 | 38.8–42.1 |
| mean RAVE β at selected edge | — | **0.67** | 0.60 |
| RAVE β by depth 1 / 2 / 3 / 4 / 5 | — | .47 / .75 / .85 / .92 / .97 | .44 / .61 / .76 / .85 / .91 |
| pool-term flip fraction | — | **37.2 %** | 30.2 % |
| cells per search | — | 5.0 | 7.5 |
| steps per cell, median | — | **17** | 11 |
| contributing prefixes per cell, median | — | **4** | 3 |
| cells with ≥ 3 prefixes | — | 68.2 % | 62.9 % |
| selection points with ≥ 2 expanded siblings | 49.2 % | **73.3 %** | 69.7 % |
| root Q spread < 0.01 / median | 4.5 % / 0.63 | 1.8 % / 0.62 | 0.0 % / 0.56 |

**What the diagnostics say.** ActPool behaves as designed: about 1.5× the evidence per cell (17 against 11
steps, 4 against 3 prefixes), a larger pool weight at every depth, and more flips from the pool term.
Both pooling arms widen search sharply: 70–73 % of selection points have ≥ 2 expanded siblings, against
49 % for NoEmo.

**⚠ SR is not interpretable at n = 10, and there is a ceiling.** ActPool's 0.60 against AffPool's 0.90
is 3 dialogues. NoEmo is at **10/10** under `episode` on these dialogues, so **no arm can beat NoEmo on
SR here**. If the grid adopts `episode`, SR may be near ceiling for the whole grid, and AvgT becomes the
effective primary outcome. PREREG already makes it co-primary. See `p1_pilot.md`.

## 4. The pooling gate under `episode` — passes on the quantity pooled

Overnight, `p1_pilot.md` flagged AffPool's between/within ratio rising to 0.98 under `episode`. That was
on **z**, which the pool does not carry. Re-measured on the **task return** with Wednesday's estimator
(`lib.affpool_block`), on the NoEmo P1 trees. The z/K0 numbers reproduce `analyze_pilot.py` (asserted).
Script `scripts/t4_affpool_gate_episode.py`, output `affpool_gate_episode.json`.

| NoEmo trees | key | ratio on **task return** [CI] | noise-corrected | evidence multiplier | ratio on z |
|---|---|---|---|---|---|
| legacy | K0 (AffPool) | 0.520 [0.411, 0.676] | −0.008 | 2.38 | 0.714 |
| legacy | unkeyed (ActPool) | 0.521 [0.445, 0.641] | 0.072 | **8.0** | 0.735 |
| **episode** | K0 (AffPool) | **0.637** [0.533, 0.753] | 0.145 | 2.0 | 0.983 |
| **episode** | unkeyed (ActPool) | **0.572** [0.549, 0.753] | 0.090 | **5.0** | 0.965 |

**The gate (≤ 1.0) passes under both horizons, with clear margin.** Under `episode` the upper CI is 0.75
for both keys. **Wednesday's finding holds on the new trees: the affective key buys no homogeneity on the
task return.** Unkeyed is as homogeneous as K0 (0.57 against 0.64) with 2.5× the evidence. This is what
makes ActPool the right control, and it is the reason PREREG predicts AffPool ≈ ActPool.

**τ note.** τ_med on episode NoEmo trees is 0.279, against 0.404 on legacy trees and the frozen
`--aff_pool_tau 0.35`. Under `episode` the default τ puts about 60 % of steps in the low bucket rather
than 50 %. This affects AffPool only, not ActPool. Whether to re-set τ is a freeze call.

Outputs: `episode_pilots_metrics.json`, `affpool_gate_episode.json`. Runs: `runs/E_actpool`,
`runs/E_affpool`.
