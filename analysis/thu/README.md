# `analysis/thu/` — the Thursday work (2026-09-16 → 17)

Everything that fed the freeze. Decisions live in [`../../PREREG.md`](../../PREREG.md) (Entries 1–6)
and [`../FREEZE_NOTES.md`](../FREEZE_NOTES.md) §10; this directory is the evidence.

## Reports

| file | question | answer |
|---|---|---|
| [`headroom.md`](headroom.md) | can the environment reward action selection at all? | Yes at n_sims 50 (ties 4–5 %, median root Q spread 0.6–0.7). §4 gives the **tiebreak rate by budget**: at n_sims 10, 49 % (D1) / 32 % (D2) of roots are decided by tiebreak → `n_sims=10` dropped |
| [`construct_test.md`](construct_test.md) | does lexical *resistance* predict donation better than emotion? | No. §3b is the **leakage cut**: whole-dialogue scoring inflates the lexical signal (AUC 0.73 → 0.50) and `w(happiness)` is ~⅔ post-decision |
| [`remine.md`](remine.md) | what does `w(e)` look like without outcome turns? | max \|Δw\| 0.41. **Addendum:** no emotion clears the base rate under uniform / last-turn / recency weighting (42 tests) → the grid runs `generic` |
| [`p1_pilot.md`](p1_pilot.md) | `legacy` vs `episode` search horizon | episode: 10/10 vs 6/10 SR, −42 % wall clock, correctness gate passes. **Recommended and frozen.** Mechanism: legacy loops until the clock runs out |
| [`actpool.md`](actpool.md) | ActPool, and does the affective key earn its place? | Built and accepted. The pooling gate passes on the task return under `episode` (0.64 keyed / 0.57 unkeyed with 2.5× the evidence) |
| [`momentum.md`](momentum.md) | what is `--emo_signal delta`, and how should the arms be dosed? | It is the per-edge **average local affective change**, not β·Δν. Arms are dose-matched on flip rate (§3), and Momentum keeps mean Q_emo ≈ 0 on live trees (§6) |
| [`tau_dose.md`](tau_dose.md) | τ and doses per valence table | τ_med 0.263 under `generic`; doses and the visited-edge bonus per table |
| [`trajvalue_gates.md`](trajvalue_gates.md) | TrajValue Gate A / Gate B | **Both fail** → the value arms die, nothing built. Momentum inside the value channel also points the wrong way |
| [`success_criterion_fix.md`](success_criterion_fix.md) | is the benchmark saturated, and is the success detector sound? | Not saturated (SR 0.78 at n=100). The detector accepts hedged non-commitments (~4 %); hedge detector v2 fixed; `tag` stays in the environment |
| [`b1_diagnosis.md`](b1_diagnosis.md) | is NoEmo actually GDP-Zero's search? | Yes at top-K 5 (0 flips in 120,085 real selections). With top-K off it diverges via PUCT tie rounding → **B1 runs GDP-Zero's own planner** |
| [`flags.md`](flags.md) | flag names, confirmed against the parsers | arm flags are per-runner, not in `add_common_args`; runner defaults ≠ grid values |
| [`plan_4c_run_table.md`](plan_4c_run_table.md) · [`plan_4c_review.md`](plan_4c_review.md) | the run table, and its review against the code | 37 configs; the plan's `--n_sims` / `--K` / `--Q0` spellings are wrong and `--max_conv 100` was missing |

## Scripts (`scripts/`)

`t1_headroom.py` · `t2_build_turns.py` · `t2_markers.py` · `t2_construct_test.py` · `t2_leakage.py` ·
`t5_momentum_replay.py` · `t5_dose_replay.py` · `t5_own_tree_flips.py` · `t6_trajvalue_gates.py` ·
`t4_affpool_gate_episode.py` · `t10_success_criteria.py` · `t_remine_variants.py` ·
`t_pilot_metrics.py` · `b1_planner_equivalence.py` · `gen_grid_configs.py` ·
`build_saturation_set.py` · `run_arm.sh`

Every replay script re-derives ν, `Q_emo` and visit counts from the logs and **asserts** they
reproduce what was logged before reporting anything.

## Runs (`runs/`)

| run | config | note |
|---|---|---|
| `P1_legacy`, `P1_episode` | NoEmo, dialogues 131–140 | the horizon pilot; `P1_episode` is the NoEmo reference and the dose-calibration tree set |
| `E_bias_b0.7`, `E_momentum_b1.2`, `E_centre_b1.1`, `E_actpool`, `E_affpool` | episode, dialogues 131–140 | arm pilots at the doses matched on episode trees |
| `SAT_noemo_episode_n100` | NoEmo, episode, n_sims 50, dialogues **141–240** | the saturation check; deliberately **not** the eval set |
| `M5_bias_legacy_ABORTED` | — | a legacy Bias run stopped after one minute when the pilots moved to `episode`; kept so the gap in the sequence is explained |

Dialogue sets are disjoint by construction and asserted: eval = positions 1–100, D1–D3 = 101–130,
pilots = 131–140, saturation = 141–240.

## Regeneration

`regression/` holds the stub end-to-end regression (pre- vs post-change, IDENTICAL at β 0 and 0.7).
`regression/src_pre/` is a copy of the source as it stood **before** the Thursday changes; it is
git-ignored and cannot be recovered from history (the Wednesday work was uncommitted until the freeze
commit), so it is kept on disk. Everything else here is regenerable from the committed inputs.
