# Plan §4c — review against the code (2026-09-17)

The plan is `plan_4c_run_table.md` (verbatim). Configs come **only** from
`scripts/gen_grid_configs.py`: 37 configs (Block A 28, B 5, C3 3, C4 1). Each one has been parsed by
`rollout.py`'s real argparse block and passed `finalize_args`: the inert-arm-flag check plus
`--frozen_config`. Draft output: `analysis/grid/DRAFT_success_pending/` (`commands.sh`,
`manifest.json`, `frozen/*.json`). It is a draft only because `--p4g_success` is still open.

## 1. Flag spellings — corrections

| plan says | code | status |
|---|---|---|
| `--n_sims 20` | **`--num_mcts_sims`** | ❌ wrong name. `--n_sims` does not exist; argparse would reject it |
| `--K 5` | **does not exist**. K is `--llm_prior_topk 5` | ❌ remove. The line below it already sets K |
| `--Q0 0.0` | **`--Q_0`** | ❌ wrong name |
| ⚠ persona flag | **`--p4g_persona`** (store_true) | ✅ confirmed |
| ⚠ eval-set flag | there is no eval-set flag. It is **`--data data/p4g/rollout_evalset_nonannotated.jsonl --max_conv 100`** | ⚠ see §2: `--max_conv` is another trap |
| `--aff_pool_key affect` / `act` ⚠ | `--aff_pool_key {affect,act}`, default affect | ✅ exactly as guessed |
| `--emo_centre` | `--emo_centre` (British; no `--emo_center`) | ✅ |
| `--emo_signal level` / `delta` | ✅ | ✅ |
| `--aff_pool --aff_pool_bias --aff_pool_tau` | ✅ | ✅ |
| `--search_horizon episode` / `legacy` | ✅ | ✅ |
| `--emo_valence_table generic` | ✅ (choices soft/argmax/generic/predecision) | ✅ |
| `--llm_prior_topk 0` (B1) | ✅ **0 = off**: no pruning (`mcts.py:222` needs 0 < K), 13-act space, 15-sample histogram prior | ✅ |

## 2. Missing from the shared block (would silently inherit)

| flag | runner default | grid value |
|---|---|---|
| **`--max_conv`** | **20** | **100**. Without it every run evaluates 20 dialogues |
| `--data` | `full_dialog.csv` (the 717 pool) | the eval-set file (first 100 non-annotated) |
| `--game` / `--algo` | p4g / `llm_raw` | `emo_p4g` / `emomcts` |
| `--llm` / `--sglang_model` | gpt-3.5-turbo / vicuna | sglang / vicuna (Qwen for B3–B5) |
| `--logit_scoring` | off | off. Written anyway |
| `--emo_risk_lambda` | 0.0 | 0.0. Written anyway |
| `--num_workers` | 1 | 10 (the pilots' value; the costs depend on it) |
| **`--p4g_success`** | tag | **open**, decided after the n=100 run |
| `--frozen_config` | none | per-run JSON |

The generator writes all of these out explicitly. Add `--max_conv` to the plan's table of runner
defaults: it is the most expensive one to inherit.

## 3. Contradiction check — built

"Pass τ to ActPool and the config check should flag it": `check_inert_arm_flags` (in `finalize_args`,
always on) refuses `--aff_pool_tau` unless the run is AffPool (`--aff_pool --aff_pool_key affect`). It
also refuses any `--aff_pool_*` without `--aff_pool`. It reads the flags actually passed, so a resolved
default is not flagged. Tested with `--flag=value` forms too.

## 4. Content issues

1. **B1 "plain GDP-Zero" is not verified to be GDP-Zero.** `emomcts --beta_emo 0 --llm_prior_topk 0`
   runs `EmotionAwareMultiObjectiveQ` on `emo_p4g`. The runner also has `--algo gdpzero`
   (`OpenLoopMCTS`). On the stub backbone (10 dialogues, 50 sims), `--algo gdpzero --game p4g` and
   `emomcts β 0` on `emo_p4g` gave **different action sequences on 10/10 dialogues**, SR 0.3 against 0.4.
   I could not isolate planner from game in the time: `gdpzero` on `emo_p4g` produced only 1 episode.
   **If the row is to read "the published baseline", decide which invocation it is.** The generator
   currently emits the plan's version (`emomcts β 0, top-K off`). Also unmeasured: B1's cost. With no
   top-K, the prior is 15 samples and there are 13 actions, so the 8 h estimate has no measurement
   behind it.
2. **B2 "compare on the common 2669-root subset" does not apply.** That subset belongs to the **replay**
   runners. Grid runs are self-play (`rollout.py`), where trajectories diverge by turn 2 (P1: 9 of 10
   dialogues). Compare B2 against NoEmo on paired dialogues: same eval ids, same seed.
3. **AffPool/ActPool bias 0.25 is unpiloted.** Every AffPool/ActPool number from Wednesday and Thursday,
   including the pooling gate, RAVE β by depth and cost, was measured at **b = 0.1**. Wednesday's sweep
   showed b = 0.25 is the only setting that leaves the edge's own Q dominant at depth 1 (RAVE β 0.26), so
   the choice is defensible. It is still unmeasured under `episode`.
4. **AffPool cost is measured, not assumed:** **218 s/dialogue** under `episode` at b = 0.1
   (`E_affpool`, AvgT 5.0), and 43.5 s per planned turn, the same as the other arms. The plan's 280 s
   assumption is conservative.
5. **The costs assume the lenient success detector.** Every s/dialogue figure comes from runs where the
   `[donate]` tag alone ends an episode. A stricter `--p4g_success` ends fewer episodes early, so
   dialogues run longer and cost more. For scale, legacy NoEmo at SR 0.6 cost 400 s/dialogue against
   231 s at SR 1.0. **If a strict criterion is chosen, re-measure cost before committing the grid
   budget**, or at least bound it with the 10-dialogue pilot set.
6. **`n_sims=20 ×0.4` is an estimate.** Nothing at 20 has been run. The first Block-A seed measures it.
7. **C5's text says "do not inherit Vicuna's 0.078".** Vicuna's τ under the frozen `generic` table is
   **0.263**; 0.078 was `predecision`'s. C5 needs no τ at all (NoEmo, Bias, ActPool have no bucket).
   C5's "Why" cell also repeats C4's text.
8. **B3–B5 on Qwen use Vicuna-calibrated doses.** Harmless for NoEmo and ActPool (no dose, no τ). Bias's
   β 0.7 follows the fixed-anchor rule, but its flip rate on Qwen trees is unmeasured, and D3 showed
   Qwen's ν runs lower. Report Qwen-Bias with its own measured flip rate (one replay on the NoEmo-Qwen
   trees, CPU only).
9. **Block C1/C2 (tagged cache) and C5 (Llama) have no configs.** The tagged cache is not implemented,
   and C5 needs its 5-dialogue pilot first.
10. **Block B total checks out:** A 145 h + B 31 h = 176 h, and the Order list sums to 176 h.
