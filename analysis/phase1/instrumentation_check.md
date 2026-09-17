# §4 instrumentation — bit-identity verification

**Result: PASS.** With the new logger on and no feature flags changed, 3-dialogue runs are
bit-identical in decisions, and in every other artifact the pre-change code writes, to a
pre-change run on the same seed. Checked at β = 0.7 (the D1 setting) and at β = 0.

## What changed (write-only)

Full diff: `runs/instrumentation_check/instrumentation.diff` (3 source files, +172/−5).

- `src/mcts/emotion_mcts.py`: `EmotionAwareMultiObjectiveQ` gains a `sim_steps` tape. In `search`
  it records the sibling statistics while computing each sibling's PUCT score (the existing loop,
  one `_calculate_uct` call per sibling as before), then the sampled parent realization and the
  child realization, then `z` and `v` after the backup. It adds three pure helpers
  (`_node_das`, `_dist_to_json`, `_realization_id` = sha1 of the history text). No random draws,
  no model calls, no writes to any table that selection or backup reads.
- `src/runners/_common.py`: `build_simlog_step_records` and `write_simlog_ndjson` write
  `<run_dir>/simlog/<dlg_id>.ndjson.gz`, a **new** carrier. The frozen 18-key subtree log is
  untouched.
- `src/runners/rollout.py`: `rollout_one` also returns the simlog records and appends one
  `turn` record per realized turn (reads states only). The episode pickle schema is unchanged.

No prompt, default flag value, or realization cache keying was changed.

## Why a stub backbone, not SGLang

SGLang sampling is not seeded per request (`SGLangChatModel.chat_generate` sends no seed), so two
runs of the *unchanged* code already diverge on the real model. "Bit-identical to a pre-change run
on the same seed" is only checkable against a deterministic token source. The check uses the
repo's own Part-3.2 harness (`tests/run_e2e_regression.py` + `tests/fake_backbone.py`). The stub
draws every response from one seeded global RNG, so an extra or reordered model call desynchronises
the stream and diverges the run. The real HF DistilRoBERTa classifier was used, not a stub, and
numpy/random were seeded as the runner does.

Configuration (D1's flags minus the backbone): `--game emo_p4g --algo emomcts --num_mcts_sims 50
--max_turns 10 --num_workers 1 --emotion_classifier hf --llm_prior_topk 5 --max_realizations 3
--cpuct 1.0 --Q_0 0.0 --emo_signal level --emo_valence_table soft --p4g_persona
--data analysis/phase1/runs/dialogues_101_130.jsonl --seed 0`, with `--beta_emo 0.7` and
`--beta_emo 0.0`. The check ran first at `--max_realizations 3` (the brief's R) and was repeated at
`--max_realizations 4` once R = 4 was chosen for the runs. Both pass.

Pre-change tree: a copy of `src/` taken before any edit (HEAD `2ed83c1`, working tree clean
under `src/`).

## Evidence (`runs/instrumentation_check/`)

`compare_regression.txt` (existing acceptance script):

| arm | SR pre → post | AvgT pre → post | action sequences | full episode records |
|---|---|---|---|---|
| β = 0.7 | 0.3333 → 0.3333 | 8.6667 → 8.6667 | identical (16 actions / 3 dialogues) | identical |
| β = 0.0 | 0.3333 → 0.3333 | 7.3333 → 7.3333 | identical (15 actions / 3 dialogues) | identical |
| **R = 4, β = 0.7** (`compare_regression_R4.txt`) | 0.6667 → 0.6667 | 5.6667 → 5.6667 | identical (11 actions) | identical |
| **R = 4, β = 0.0** | 0.3333 → 0.3333 | 7.0000 → 7.0000 | identical (13 actions) | identical |

The stub made 4008 (β = 0.7) and 3614 (β = 0) backbone calls, identical pre and post, so the call
sequence is unchanged.

`check_beta07.txt`, `check_beta0.txt`, `check_r4beta07.txt`, `check_r4beta0.txt` (`scripts/check_instrumentation.py`; all PASS):

- The frozen subtree NDJSON is record-for-record identical pre/post (3 files per arm).
- The classifier's utterance → emotion records are identical (1425 at β = 0.7).
- Simlog internal alignment (β = 0.7: 4838 steps, 567 edges; β = 0: 4400 steps, 509 edges):
  - (a) Each edge's `z` values, in simulation order, equal the frozen log's `per_step_valences`.
  - (b) Every step's `parent_realization_id` is traceable with no ordering assumption. At depth 1
    it is the observed root (`parent_nu == root_nu`, same distribution). At depth ≥ 2 it matches a
    depth−1 step in the same tree that generated that realization (`child_nu == parent_nu`, same
    distribution).
  - (c) `z == child_nu` on every step (`--emo_signal level`).

The 39 existing unit tests in `tests/test_emo_channel_freeze.py` still pass.

The same alignment checks are re-run on the D1/D2 logs before analysis (see `p_var.md`).
