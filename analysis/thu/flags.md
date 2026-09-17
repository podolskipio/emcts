# Task 9.2 — flag names, confirmed against the parsers

Dumped from the argparse blocks of `src/runners/rollout.py` and `src/runners/emomcts.py`. Both runners
declare the same arm flags with the same names and defaults (the dump is identical for every row below).
Every flag also has a hyphenated alias (`--aff-pool-key`, …) except the ones marked "none".

## Correction to the brief: the arm flags are **not** in `add_common_args`

`add_common_args` (`src/runners/_common.py:862`) holds only the shared plumbing: `--game`, `--llm`,
`--sglang_model`, `--llm_prior_topk`, `--explicit_value_labels`, `--logit_scoring`,
`--emotion_classifier`, `--seed`, `--num_workers`, `--max_turns`, `--p4g_persona`. **Every
arm flag is declared per runner**, in `rollout.py` and again in `emomcts.py`. A new arm flag has to be
added to both, and `tests/test_thu_arms.py::test_runner_aff_pool_key_flag` checks both.

## Arm → flags

| arm (brief name) | flags as they exist | status |
|---|---|---|
| NoEmo | `--beta_emo 0.0` (default) | ✅ |
| Bias | `--beta_emo β` with `--emo_signal level` (default) | ✅ |
| CenteredBias | `--beta_emo β --emo_centre` | ✅ (brief doesn't name it; British spelling, alias `--emo-centre`, no `--emo_center`) |
| Momentum | `--beta_emo β --emo_signal delta` | ✅ ships. See `momentum.md` for what it changes |
| AffPool | `--aff_pool --aff_pool_bias b --aff_pool_tau 0.35` | ✅ |
| **ActPool** | `--aff_pool --aff_pool_key act` | ✅ **built today**. Brief's ⚠ guess `--aff_pool_key {affect,act}` is exactly what was built |
| P1 horizon | `--search_horizon {legacy,episode}`; `--terminal_on_failure` is an alias for `episode` (same dest) | ✅ |
| TrajValue | `--traj_value`, `--traj_value_alpha`, `--traj_value_table` | ❌ **not built.** The spec `agent_task_trajvalue.md` is not in the repo |
| TrajPrompt | — | ❌ not built, same missing spec (§0.1b) |

Other emotion flags that exist and are off by default: `--emo_risk_lambda` (0.0),
`--emo_valence_table {soft,argmax,generic}` (soft), `--emo_constraint_tau` (unset),
`--emo_constraint_m_warm` (3).

## ⚠ Runner defaults are not the grid config

The Phase-1/Wednesday config passes these explicitly (`analysis/thu/scripts/run_arm.sh`). A grid config
that omits any of them silently runs something else:

| flag | runner default | grid value |
|---|---|---|
| `--num_mcts_sims` | **20** | 50 |
| `--max_realizations` | **3** | 4 (R) |
| `--llm_prior_topk` | **None** (15-sample histogram prior) | 5 (K) |
| `--emotion_classifier` | **llm** | hf |
| `--seed` | **None** (unseeded) | 0 |
| `--llm` | gpt-3.5-turbo | sglang |
| `--Q_0` (rollout) | 0.0 | 0.0 |

This is the reason Task 9.4 (generate every config from the frozen template) matters: generate every
config from the template and never rely on a runner default.
