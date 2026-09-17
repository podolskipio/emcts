# FREEZE NOTES — Week 1, Day 5–6 pre-freeze changes to the `Q_emo` backup

Status: **both changes landed; both acceptance tests in §3.2 passed; the grid may start.**

Covers:

* **Part 1** — Welford variance tracking on the emotion channel (`M2_emo` / `sigma_emo`)
  and the risk-adjusted selection rule `--emo_risk_lambda`
* **Part 1.5** — the frozen NDJSON subtree log schema
* **Part 2** — the delta-valence signal `--emo_signal {level,delta}`

---

## 1. Defaults (the freeze contract)

| flag | default | effect at the default |
|---|---|---|
| `--emo_risk_lambda` | `0.0` | `Q_emo - 0.0 * sigma_emo` is **bit-identical** to `Q_emo` (IEEE-754: `0.0 * finite = 0.0`, `x - 0.0 == x`, and `-0.0 - 0.0 == -0.0`). No branch, no epsilon. |
| `--emo_signal` | `level` | `z = ν(d_s′)` — byte-for-byte the pre-change quantity. |
| `--beta_emo` | `0.0` | unchanged; the emotion channel is off by default. |
| `--seed` | `None` | unchanged (unseeded), see §5.4. |
| `--emo_valence_table` | `soft` | resolves to the *same dict object* as `EMOTION_VALENCE_MINED`, so `ν(·)` is unchanged. |

Both new flags also accept the dashed spelling from the spec (`--emo-risk-lambda`,
`--emo-signal`); the underscored form is primary to match every other flag in the repo.
Both resolve to the same `dest`.

---

## 2. What landed

**`src/mcts/emotion_mcts.py`**

* `welford_update` / `welford_variance` / `welford_sigma` — module-level and pure, so the
  arithmetic is unit-testable without a game, planner or classifier.
* `EmotionAwareMultiObjectiveQ.M2_emo` — one extra float per edge, initialised to `0.0`
  alongside `Q_emo` in `_init_node`, so both channels stay in lock-step with the
  (possibly top-K-pruned) action set.
* `var_emo(s,a)` / `sigma_emo(s,a)` — derived at read time, never stored;
  `0.0` when `N < 2`, so nothing divides by `N-1 == 0`.
* `_update_emo_channel` — the Welford backup, taking the **pre-increment** `Nsa` exactly as
  the old running mean did. `delta` (deviation from the old mean) times `delta2`
  (deviation from the **updated** mean); the classic same-deviation-twice bug is guarded by
  a test that asserts the buggy form gives a different answer.
* `_emotion_signal(parent_state, next_state)` — the `level` / `delta` switch.
* `emo_valences` — the raw per-edge tape of every `z`, for the log (§4). Write-only.
* `check_emo_signal_flags` — the `delta` × n-step guard (§5.3).

**`src/mcts/mcts.py`** — two write-only additions, no behaviour: `node_V` (leaf value keyed
by the *tree* key, because `Vs`' key space cannot be joined against `Nsa`/`Q`) and
`cache_hits` (realization-cache hit/miss counts per node).

**Runners** — `--emo_risk_lambda` / `--emo_signal` / `--seed` on `runners/emomcts.py` and
`runners/rollout.py`; the flags are threaded into the planner and into `metadata.json`.
`rollout.pick_action` now returns `(action, planner)` so the rollout runner can log the tree
it would otherwise throw away.

---

## 3. §1.3 + §1.5 — subtree logging (SCHEMA FROZEN)

Two carriers, both written on **every edge, in every run, by every runner**, including the
GDP-Zero baselines — which have no emotion channel and report `0.0` / `[]` rather than
omitting the field. A missing key always means a bug, never a baseline.

**(a) NDJSON, gzip, one file per dialogue** — `<run_dir>/subtree/<dlg_id>.ndjson.gz`,
one line per edge per planned turn, 18 keys:

```
dlg_id, turn, node_id, parent_id, action_seq, utterances, emotion_dist, leaf_value,
N, Q, Q_emo, M2_emo, sigma_emo, root_visit_dist, cache_hit, seed, depth, per_step_valences
```

Frozen conventions, spelled out because they are part of the schema:

* `node_id` is the tree key: the `"__"`-joined system-DA prefix, which is what an open-loop
  node *is*. `""` is the empty prefix (dialogue start). `action_seq` is that prefix as a list.
* `N`, `Q`, `Q_emo`, `M2_emo`, `sigma_emo`, `per_step_valences` belong to the edge that
  **enters** `node_id`, i.e. `(parent_id, last action of action_seq)`. The search-root record
  has no incoming edge: it reports `N = Ns(root)`, `0.0` / `[]` for the rest, `parent_id = null`.
* `depth` is relative to the **search root** (root = 0), not the dialogue start.
* `emotion_dist` is the FULL softmax, **never** an argmax — averaged over the node's cached
  realizations, so one line summarises all R samples rather than an arbitrary one.
  `utterances` is those same R samples' system utterances.
* `cache_hit` = "at least one transition into this node was served from the realization
  cache" (the hit/miss counts behind it are on the planner).
* `sigma_emo` is derived at write time and is recomputable from `per_step_valences`; a test
  and the reporting script both assert that (0 mismatches over 673 edges, §5.3).
* Edges with `N == 0` are logged too. That is what makes "present on every edge" checkable.
* A dialogue that ends before any planning turn (the user agrees on the greeting) has no
  tree and therefore no file; its episode carries `subtree_log = ""`.
* The closed-loop ablation (`gdpzero_noopenloop.py`) keys nodes by the full dialogue string
  and cannot express parent/child, so it writes `parent_id = null`, `action_seq = []`,
  `depth = null`. Every other field is populated normally.

**(b) In-pickle** — the four replay runners additionally carry `M2_emo` / `sigma_emo` inside
`debug["search_tree"]`, so existing analysis that reads the pickle does not have to learn a
new file format. `rollout.py` stores only a pointer (`episode["subtree_log"]`) — the records
themselves would bloat the pickle.

---

## 4. §1.4 — risk-adjusted selection (implemented, OFF)

```
score(a) = Q(s,a) + beta_emo * (Q_emo(s,a) - lambda * sigma_emo(s,a))
         + cpuct * P(s,a) * sqrt(Ns) / (1 + Nsa(s,a))
```

Implemented unconditionally — no `if lambda == 0` branch — because the expression *reduces*
to `Q_emo` exactly at `lambda = 0.0`. A test asserts `.hex()` equality of the resulting UCT
scores against the verbatim pre-change formula over 50 randomised edge-statistic sets
(13 actions each), including the `Q_emo = -0.0` signed-zero case. A companion test asserts
that `lambda > 0` *does* move scores strictly downward on high-`sigma` edges, so the flag is
not silently inert. **Not swept.** Instrumentation for a later Tier-C experiment.

---

## 5. Acceptance tests

### 5.1 Unit tests — `tests/test_emo_channel_freeze.py`, **34 passed**

```
python -m pytest tests/test_emo_channel_freeze.py -q
```

| §3.1 requirement | test |
|---|---|
| Welford mean ≡ previous running mean | `test_welford_mean_matches_prechange_running_mean`, `..._on_random_sequences` (200 random sequences) |
| variance ≡ `numpy.var(zs, ddof=1)` | `test_welford_variance_matches_numpy_ddof1`, `..._on_random_sequences` (200) |
| `N < 2` → `var_emo == 0.0`, no ZeroDivision | `test_variance_is_zero_below_two_observations`, `test_planner_var_emo_is_zero_below_two_visits_and_on_unknown_edges` |
| `lambda = 0` → bit-identical selection | `test_lambda_zero_is_bit_identical_to_prechange_selection`, `..._across_many_random_edge_sets`, `..._negative_zero_q_emo` |
| `--emo_signal level` → `z` identical | `test_level_signal_matches_prechange_z`, `test_level_signal_ignores_the_parent_entirely` |
| `--emo_signal delta` at the root → `z == 0.0` | `test_delta_signal_is_zero_when_the_parent_has_no_distribution` |
| delta renormalisation keeps `Q_emo ∈ [−1,+1]` | `test_delta_renormalisation_keeps_z_and_q_emo_in_range` (2000 random pairs + 300 random edge histories), `test_delta_range_is_saturated_by_the_extreme_pair` |

Plus: the same-deviation-twice bug guard, the `delta` × n-step guard, constructor default
and validation tests, and seven tests on the frozen NDJSON schema (§3).

### 5.2 §3.2 End-to-end regression — **PASSED**

10 dialogues, `num_mcts_sims = 50`, fixed seed 0, `runners/rollout.py --game emo_p4g
--algo emomcts`, pre-change tree vs post-change tree.

| arm | SR (pre → post) | AvgT (pre → post) | action sequences | full episode records |
|---|---|---|---|---|
| **defaults** (`beta_emo=0.0`) | 0.4000 → **0.4000** | 7.200 → **7.200** | identical (53 actions / 10 dialogues) | identical |
| `beta_emo=0.3` | 0.4000 → **0.4000** | 7.700 → **7.700** | identical (49 actions / 10 dialogues) | identical |

`RESULT: IDENTICAL — strict generalization confirmed`

Reproduce:

```
python tests/run_e2e_regression.py --src <pre_src> --out <dir>/beta03_pre.pkl  \
       --dialogs 10 --sims 50 --seed 0 --beta_emo 0.3
python tests/run_e2e_regression.py --src src        --out <dir>/beta03_post.pkl \
       --dialogs 10 --sims 50 --seed 0 --beta_emo 0.3
python tests/compare_regression.py <dir>
```

The comparison is stricter than the spec asks: beyond SR/AvgT and the action sequences it
diffs the **complete episode records** (every utterance, every dialog act, every emotion
label, turn by turn). Nothing differs.

The `beta_emo = 0.3` arm is the one that carries the weight. At the `beta_emo = 0.0`
default the emotion channel is multiplied by zero, so a defaults-only regression cannot see
the Welford change at all and would pass vacuously. At `beta_emo = 0.3` the Welford mean
actually enters PUCT — and it still reproduces bit-for-bit, which is the claim that matters.
The two arms also produce genuinely different AvgT (7.200 vs 7.700), confirming the metric
is discriminative rather than pinned.

### 5.3 §3.3 Smoke test on the new instrumentation — **PASSED**

`python scripts/report_sigma_emo.py <run_dir>/subtree`, default arm:

```
673 edges with N >= 2   ->   670 have sigma_emo > 0  (99.6%)   PASS
mean 0.1178   median 0.1040
p10 0.0342   p25 0.0604   p75 0.1574   p90 0.2180   p99 0.3713
min 0.0000   max 0.4389

  [0.000, 0.044)  102  ##########################
  [0.044, 0.088)  171  ############################################
  [0.088, 0.132)  161  #########################################
  [0.132, 0.176)  116  #############################
  [0.176, 0.219)   58  ##############
  [0.219, 0.263)   28  #######
  [0.263, 0.307)   17  ####
  [0.307, 0.351)   11  ##
  [0.351, 0.395)    6  #
  [0.395, 0.439)    3

|Q_emo| on the same edges: mean 0.0826   max 0.2879
sigma_emo / |Q_emo|:       median 1.43   p90 9.10
cross-check vs per_step_valences: 673 edges, 0 mismatches
```

(`beta_emo = 0.3` is materially the same: 632 edges, 99.4% non-zero, mean 0.1132,
median 0.1009, σ/|Q_emo| median 1.41.)

**One-line read:** the within-node spread is not a rounding artefact — the median edge's
`sigma_emo` is **1.4× the magnitude of the `Q_emo` it summarises**, and at the 90th
percentile 9×. The reviewer's "severe information smoothing" objection describes something
real: a scalar running mean is discarding a spread larger than the signal it keeps. There
is ample headroom for the risk-adjusted rule of §1.4.

**Caveat, and it is a large one.** These σ values come from the deterministic stub
classifier used for the regression (§5.4), not from a real LLM. They demonstrate that the
instrumentation is wired correctly and that the analysis pipeline runs end to end; they are
**not** a measurement of real emotional variance and must not be quoted as one. The first
real numbers come out of the grid — which is exactly the point of computing this from the
main grid rather than a separate pass.

### 5.4 How the regression was run — deviation, with reasoning

The regression ran against a **deterministic stub backbone** (`tests/fake_backbone.py`),
not a live LLM. This is a deliberate deviation and it makes the test stronger, not weaker:

* §3.2 asks whether the *code* changed the run. A sampled LLM cannot answer that — two runs
  of the **unmodified** code already differ, so a difference would be indistinguishable from
  sampling noise and a match would be luck. "Assert SR and AvgT are identical" is not a
  well-formed test against a stochastic backbone.
* The stub is a real `GenerationModel`. The games, players, planner, user simulator and
  emotion classifier all call it through their normal interfaces and parse its output with
  their normal parsers. Only the token source is fake. The entire MCTS, backup, selection
  and logging path is the production one.
* It draws from **one seeded global stream**, not a per-prompt hash. A prompt-hashed stub
  returns the same utterance for the same prompt forever, which collapses every node to one
  realization, makes `max_realizations` meaningless, and — since p4g self-play starts every
  dialogue from the same empty scenario — makes all ten dialogues byte-identical (an earlier
  attempt gave SR = 1.0000 and AT = 4.000 for all ten, a vacuous comparison). The seeded
  stream reproduces exactly under the same seed while giving each node genuinely different
  realizations and each dialogue a different trajectory.
* Consequently, if a change alters *how many* RNG draws are taken, the streams desynchronise
  and the runs diverge visibly — which is precisely the failure mode the test exists to
  catch.

`--seed` (default `None` = previous unseeded behaviour) was added to `emomcts.py` and
`rollout.py` because §3.2 requires a fixed seed and no runner had one. The regression itself
seeds in the driver process rather than through the flag, because the pre-change tree does
not have the flag — seeding outside the runner keeps the two invocations identical in every
other respect.

**Not run:** a live-LLM regression. It would cost real API spend for a comparison that, per
the above, cannot be made to assert identity. Say the word and I will run one as a
*distributional* sanity check (same SR/AvgT within noise) instead.

---

## 5.5 Leakage control — resolved on the evaluation side (Day 6 decision)

Pre-flight found that `EMOTION_VALENCE_MINED` is mined over **all 300 annotated** dialogs
while 9 of the 12 grid invocations (`runners/emomcts.py` ×6, `runners/gdpzero.py` ×3) are
*replay* runs that default to that same corpus. Verified by re-running the miner off the
on-disk distribution cache and diffing against the deployed constant:

| emotion | DEPLOYED | soft / all-300 | soft / holdout-100 | argmax / all-300 (original) |
|---|---|---|---|---|
| happiness | +0.54 | **+0.54** | +0.41 | +0.56 |
| fear | +0.24 | **+0.24** | **−0.10** | +0.62 |
| disgust | +0.13 | **+0.13** | +0.04 | +0.25 |
| anger | +0.11 | **+0.11** | +0.11 | +0.26 |
| surprise | +0.09 | **+0.09** | +0.04 | +0.15 |
| neutral | −0.07 | **−0.07** | −0.12 | −0.09 |
| sadness | −0.14 | **−0.14** | −0.39 | −0.30 |
| contempt | −0.60 | −0.60 | −0.60 | −0.60 |

So the **soft** re-mining is confirmed deployed (exact match, `n=300 base=0.493 alpha=50`),
and these are *not* the original argmax weights. The eval-set exclusion was not applied.

**Decision: hold out on the evaluation side, not the mining side.** `w(·)` keeps all 300 —
mining on 200 flips the sign of `w(fear)` and nearly triples `w(sadness)`, because the thin
cells are unstable — and every runner instead evaluates on dialogs that carry no dialog-act
annotation at all and are therefore absent from the mining corpus.

* Self-play (`runners/rollout.py`) already did this: `P4G_ROLLOUT_DATA` is the 717
  non-annotated dialogs. Overlap with the annotated 300 verified = **0**.
* Replay runners defaulted to the annotated 300 and are now pointed explicitly at
  `data/p4g/rollout_evalset_nonannotated.jsonl` (100 dialogs, overlap with the mining
  corpus = 0). `scripts/run_paper_experiments.sh` sets `REPLAY_DATA` once and passes
  `--data "$REPLAY_DATA"` to all 9 replay invocations. Smoke-tested end to end: both
  runners complete, the frozen subtree schema and the in-pickle `M2_emo`/`sigma_emo`
  are intact.

**What this costs, stated plainly.** On non-annotated data the reader maps missing acts to
`other` / `U_Neutral`, so in replay:

* `ori_da` is `other` on 100% of turns. It was already `other` on **91%** of turns on the
  annotated set — a separate, pre-existing mismatch: the corpus labels are hyphenated
  (`credibility-appeal`) while the canonical inventory uses spaces (`credibility appeal`),
  so only 2 of the 27 raw labels (`greeting`, `other`) ever match in `_read_p4g_pickle`.
  The move therefore costs the remaining 9%, not a working signal. `ori_da` is used only as
  a fallback label in `run_judge.py:326` when `--h2h` is absent.
* `usr_da` on replayed user turns is constant `neutral`. Two consequences: the
  `usr_da == success_user_da` early-stop never fires, and the value-estimator prompt
  (`_build_value_messages`, `keep_user_da=True`) sees `[neutral]` where it would have seen
  `[positive reaction]` / `[donate]`. This is a real degradation — and it applies
  **identically to both arms**, so the GDP-Zero vs EmoMCTS head-to-head stays fair.
* `ori_resp` — the human next utterance, which is what `run_judge` actually scores — is
  real text and unaffected. The headline metric is intact.
* Replay tree keys become `other__other__...` prefixes. Pre-existing (see above), and
  harmless: a fresh MCTS is built per turn, so no cross-dialogue key collisions.

## 5.6 `--emo_valence_table {soft,argmax}` — added pre-freeze

Not in the original spec. Added because the Day-6 decision is to run on all-300 soft weights
"and maybe all-300 argmax": with `w(e)` living in a module constant, switching tables later
means editing `emotion_mcts.py` mid-grid — the exact failure the freeze exists to prevent, and
a full re-run. Same reasoning as Part 2: land the code path now so the sweep is a flag.

* `soft` (default) **is** `EMOTION_VALENCE_MINED` — the same object, not a copy, so the
  default path is bit-identical. Asserted by test (`... is EMOTION_VALENCE_MINED`).
* `argmax` is `EMOTION_VALENCE_ARGMAX`: same corpus (all 300 annotated), same base rate
  0.493, same `alpha=50`; the only difference is Algorithm 2 line 7's assignment rule.
  Reproduced from the miner, not transcribed: `fear +0.62, happiness +0.56, anger +0.26,
  disgust +0.25, surprise +0.15, neutral −0.09, sadness −0.30, contempt −0.60`.
* Both tables keep `ν` inside the documented `[−1,+1]` (soft `[−0.60,+0.54]`, argmax
  `[−0.60,+0.62]`), tested for the `level` and `delta` signals alike.
* The flag is live, not inert: same seed/config, `soft` gives SR 0.400 / AvgT 7.700 and
  `argmax` gives SR 0.300 / AvgT 7.600.

**Re-verified after this change:** 39 unit tests pass; the §3.2 regression is still
`IDENTICAL` in both arms (SR 0.4000/0.4000, AvgT 7.200 and 7.700, full episode records equal).

## 6. Deviations from the spec

1. **Flag spelling.** Primary names are `--emo_risk_lambda` / `--emo_signal`, matching every
   other flag in this repo (`--beta_emo`, `--num_mcts_sims`). The spec's dashed spellings are
   registered as aliases on the same `dest`, so both work.

2. **§2.3(b), "at the root → `z_delta = 0`" — implemented as "parent has no attached emotion
   distribution → `z_delta = 0`".** The operative rule is the spec's second clause; the
   root case falls out of it whenever the root genuinely lacks a distribution.

   In *this* codebase the search root usually does have one: the replayed prefix is
   classified before search begins, and in self-play the root is a full user turn. Treating
   the search root as parentless on principle would force `Q_emo ≡ 0` on **every root edge** —
   and the root edges are the only ones whose visit counts decide the action actually played.
   The emotion channel would be silently neutralised at the one decision that matters, making
   the `delta` arm nearly equivalent to `beta_emo = 0`. That is not what the reward change is
   meant to test.

   The spec's actual concern — never falling back to the level value, so the two signals are
   never mixed inside one run — is honoured exactly: a missing/empty parent distribution
   yields `0.0`, matching the `ν(∅) = 0` convention. Tested against four distinct
   parentless shapes (no parent object, unclassified parent turn, empty distribution,
   freshly-initialised session).

3. **§2.3(c) n-step guard: raises rather than warns, and is currently dormant.** No n-step
   horizon flag exists in the tree yet (B2 has not landed). `check_emo_signal_flags` is
   called by both runners with `getattr(cmd_args, "emo_n_step", 1)`, so it reads the default
   and never fires today — and goes live the moment B2 adds its flag, with no edit to backup
   code mid-grid. It raises `ValueError` rather than warning: a warning in a grid script's
   stdout is a warning nobody reads.

4. **§3.2 backbone.** See §5.4.

5. **Extra arm in §3.2.** The spec asks for defaults only; `beta_emo = 0.3` was run as well,
   because the defaults-only test is vacuous for the change being made (§5.2).

6. **`--seed` added.** Behaviour-neutral (default `None` = unseeded), required by §3.2.

7. **`rollout.pick_action` signature changed** to return `(action, planner)`. Needed so the
   self-play runner can emit the frozen subtree log; it throws the tree away otherwise. The
   only caller is `rollout_one` in the same file.

8. **§1.5 judgement calls not fixed by the spec**, all listed in §3 above: `emotion_dist` is
   the mean full softmax over cached realizations; the root record's edge-valued fields are
   `0.0` / `[]`; `cache_hit` is "≥ 1 cached transition into this node"; `depth` is relative
   to the search root; the closed-loop ablation degrades to `parent_id = null`.

---

## 7. Files

| file | what |
|---|---|
| `src/mcts/emotion_mcts.py` | Welford state + update, `sigma_emo`, risk-adjusted PUCT, `level`/`delta` signal, n-step guard |
| `src/mcts/mcts.py` | `node_V`, `cache_hits` (write-only bookkeeping for the log) |
| `src/runners/_common.py` | frozen NDJSON schema: `build_subtree_records`, `write_subtree_ndjson`, `subtree_emo_stats` |
| `src/runners/emomcts.py` | new flags, in-pickle `M2_emo`/`sigma_emo`, subtree log |
| `src/runners/rollout.py` | new flags, subtree log, `pick_action` returns the planner |
| `src/runners/gdpzero{,_noRS,_noopenloop}.py` | same schema on the baselines (all-zero emotion fields) |
| `tests/test_emo_channel_freeze.py` | the §3.1 acceptance tests (34) |
| `tests/fake_backbone.py` | deterministic stub backbone for the regression |
| `tests/run_e2e_regression.py` | §3.2 driver (runs either tree via `runpy`, so runner defaults are exercised, not re-declared) |
| `tests/compare_regression.py` | §3.2 comparison; exit code 0 only on full identity |
| `scripts/report_sigma_emo.py` | §3.3 smoke test + σ distribution report |

---

# §8 — Wednesday arms (2026-09-16 brief; landed 2026-09-15)

Status: **all §1 builds landed behind flags that default to the shipped planner; bit-identity
confirmed three ways; GPU smoke-tested.** Nothing here flips a default or applies a gate.

## 8.1 Flags and defaults

| flag | default | effect when set |
|---|---|---|
| `--aff_pool` | off | AffPool: `Q_eff = (1−β)Q + β·Q_pool[(bucket, a)]` replaces `Q` in PUCT, `β = N_pool/(N + N_pool + 4·N·N_pool·b²)` [Gelly & Silver 2011 §5] |
| `--aff_pool_bias` | 0.1 | RAVE bias `b`; sweep {0.05, 0.1, 0.25} on dialogues 131–140 (`analysis/wed/runs/SW_b*`) |
| `--aff_pool_tau` | 0.35 | bucket 1 ⇔ parent ν < τ (K0; 0.35 = D1 τ_med). Not tuned on eval dialogues |
| `--emo_centre` | off | CenteredBias: `β·(Q_emo − μ)` on **expanded** siblings (N > 0); unexpanded keep `β·Q_emo` (= 0) |
| `--emo_constraint_tau` | unset = off | Constrain: PUCT over `{a : Q_emo ≥ τ or N < m_warm}`, fallback = single argmax Q_emo; root decision = argmax N over the current feasible set |
| `--emo_constraint_m_warm` | 3 | visits before an action can be masked |
| `--emo_valence_table generic` | (`soft` stays default) | happiness +0.54, surprise 0, neutral 0, sadness/fear/anger/disgust −0.54, contempt −0.60 |
| `--terminal_on_failure` | off | alias for `--search_horizon episode` (same `dest`): −1.0 ends a branch. One code path, not two |

All flags exist on `runners/rollout.py` and `runners/emomcts.py`, are threaded into `metadata.json`
(`mcts_args`), and accept the dashed spelling.

## 8.2 What was confirmed in code

* **`Q_pool` carries the TASK RETURN, not `z`.** It is fed the same scalar `v` that the backup
  writes into `Q` (`emotion_mcts.py`, "AffPool backup"). It has to be: `Q_eff` blends `Q_pool` with
  `Q`, so the two must be in the same units. Scope is per search (a new planner per turn), updated
  on every backup from every node, keyed on the **sampled parent realization's** ν, the same
  realization whose ν Phase-1 logged as `parent_nu`. §5.1 therefore recomputes yesterday's
  homogeneity gate on `backup_value`: D1 0.50, noise-corrected 0.05, against 0.785 / 0.31 on z
  (`analysis/wed/affpool_gate_recheck.md`).
* **`backup_value`** is logged on every simlog step, at the call site that writes `z`, from the
  same `v` that enters `Q` (not the running mean). **It backfills D1/D2 exactly:** their `v` field
  is that variable. Replaying `v` into a running mean per edge reproduces every logged sibling
  `(N, Q)`: D1 33,903/33,903, D2 28,129/28,129, 0 mismatches (`build_tables.py:check_backfill`).
* **One reorder in `search`:** the parent realization is now drawn *before* PUCT, not after,
  so AffPool can key on it. Selection takes no random draws, so the RNG stream is unchanged. The
  three identity checks below cover exactly this.
* **CenteredBias interpretation.** Subtracting μ from *every* sibling is a per-node constant and
  changes no ranking, so the arm would be inert by construction. μ is subtracted from expanded edges only.
  The unit test `test_centering_cannot_reorder_expanded_siblings` pins the defining property: among
  expanded siblings the ranking is Bias's.
* **Constrain has no β term in the spec formula.** The code composes: `--beta_emo 0.0` gives the
  spec's `Q + U` over the feasible set; β > 0 would add `β·Q_emo` inside the mask. Run it at β = 0.

## 8.3 Bit-identity (flags off)

| check | result |
|---|---|
| `tests/test_wed_arms.py` (19 tests; 69 in `tests/`) | **pass** |
| whole-search fingerprint vs the pre-change tree (sha256 over Ns/Nsa/Q/P/Q_emo/M2_emo/tapes/every sim_steps field; real game, varying classifier; β 0 and 0.7) | **identical** (`GOLDEN` generated from the pre-change snapshot) |
| `_calculate_uct` with no arm vs pre-change formula, 200 random edge sets, λ ∈ {0, 0.3} | `.hex()`-identical |
| §3.2 end-to-end regression, stub backbone, 10 dialogues × 50 sims, β 0 and 0.7 | **IDENTICAL**: SR, AvgT, action sequences and full episode records |

## 8.4 Logging (simlog, only on runs where the arm is on)

* AffPool: step `aff_bucket`, `pool_beta_selected`, `pool_flip` (argmax with vs without the pool
  term); sibling `Q_pool`, `N_pool`, `pool_beta`. Cell counts and distinct prefixes per cell are
  derived from `(aff_bucket, action, action_prefix)`.
* CenteredBias: `emo_mu`, `n_expanded`, `centre_flip`, `centre_flip_to ∈ {visited, unexpanded}`.
* Constrain: `constraint_masked`, `constraint_violating`, `constraint_warm`, `constraint_fallback`,
  `constraint_flip`; turn record `root_constraint` = {action, unrestricted_action, disagree,
  feasible, root_fallback, root_Q_emo, root_N}.

## 8.5 GPU smoke tests (vicuna-13B AWQ, 1 dialogue, 10 sims, Tmax 4; `analysis/wed/runs/smoke/`)

All four configurations completed with the fields above present and the flags recorded in
`metadata.json`. Not a measurement. **Infrastructure note:** the first attempt died because
a WSL `dxgkrnl` GPU stall hung the SGLang scheduler until its watchdog killed it (dmesg
`dxgvmb_send_wait_sync_object_gpu`, 15:11). Runs are now served by
`analysis/wed/scripts/serve_supervised.sh`, which restarts the server; a dialogue that fails
while it is down is logged and can be re-run.

## 8.6 Proposed τ for Constrain — for confirmation, not hard-coded

`τ = 0.12`, the 25th percentile of `z` in D2 (β = 0, a tree the affect channel did not shape;
D1 gives 0.14). Counterfactual replay of the mask on D2's logged selections (exact, since
D2's uct is `Q + U`) at τ = 0.115: mask present at 10.8 % of selection points, fallback 0.1 %,
selection flips 6.4 %, **root disagreement 19.2 % of turns**. At τ = 0.2 the root
decision changes on 36 % of turns. Full table: `analysis/wed/s1_constraint_tau.json`.

---

## 9. Tmax is now one flag for every runner (2026-09-16)

### 9.1 The corrected value

**Tmax = 10 turns, on every runner.** `--max_turns` moved from `rollout.py`'s own parser into
`add_common_args` (`src/runners/_common.py`), and all six runners now pass it as
`build_agents(max_conv_turns=cmd_args.max_turns)`:

| runner | game horizon before | after |
|---|---|---|
| `rollout.py` | 10 (`--max_turns`, W5 fix 5) | 10 |
| `gdpzero.py` | **15** (game default, flag absent) | 10 |
| `emomcts.py` | **15** | 10 |
| `gdpzero_noRS.py` | **15** | 10 |
| `gdpzero_noopenloop.py` | **15** | 10 |
| `raw_prompting.py` | **15** (no search; affects `get_dialog_ended` only) | 10 |

The flag now lives in one place precisely so the two cannot drift again: the replay runners
inherited the `DialogGame` default of 15 only because they never named the parameter.

### 9.2 Why it matters from tomorrow, and not before

Under `--search_horizon legacy` the game horizon does **not** bound search at all: search
terminates only on success and ignores the −1.0 that `get_dialog_ended` returns at the turn
limit (§1 of `analysis/phase1/SEARCH_HORIZON_BUG.md`). So 15-vs-10 was inert in the replay
runners — it changed no tree, because neither number was ever consulted.

Under `--search_horizon episode` it becomes the environment definition. `len(state)` in
`_failure_or_continue` is the **absolute** turn count of the full history, so a tree rooted at
real turn *t* stops at *t + d = Tmax*; measured on the real game at Tmax 10, the deepest child
state is turn 10 from every root turn (1, 4, 6, 8, 9), i.e. root-relative depth exactly
`Tmax − t`. With Tmax 15 the replay runners would have searched five turns deeper than the grid.

### 9.3 Does this change Wednesday's numbers? **No — nothing to re-derive.**

Checked, not assumed. Wednesday's arms were characterized on `steps.parquet` /
`selections.parquet`, which come from D1/D2/D3 — and all three are `runners/rollout.py` runs at
`max_turns: 10`:

| evidence | result |
|---|---|
| `analysis/wed/constants.json` → `runs.{D1,D2,D3}.metadata` | `runner: runners/rollout.py`, `args.max_turns: 10` |
| `analysis/wed/runs/*/command.txt` (D3, SW_b0.05, SW_b0.1, SW_b0.25) | `runners/rollout.py … --max_turns 10` |
| any Wednesday script invoking `gdpzero*.py` / `emomcts.py` | none |
| max realized root turn in `steps.parquet` | 9 (consistent with Tmax 10, not 15) |

**CenteredBias and Constrain were not characterized by the replay runners.** Despite its name,
`analysis/wed/scripts/s1_centre_replay.py` is an offline re-scoring of logged selection points
(`load_selections("D1")`, pure pandas — no game, no runner, no MCTS); `s1_constraint_tau.py` is
the same against `steps.parquet`. "Replay" there means replaying the logged tree's selection
arithmetic, not running a replay runner. Both therefore inherit D1/D2's Tmax of 10.

One caution for reading those logs: 21.1 % of Wednesday's step rows have `turn_index + depth > 10`
and **5.3 % exceed 15**. That is the legacy-horizon bug (depth bounded only by the simulation
budget), not evidence of a Tmax-15 environment — under a binding horizon of 15 nothing could
exceed 15. The affect contamination it causes is unchanged by this fix and is what P1 measures.

### 9.4 Consequence of the lower horizon: past-horizon replay roots are now skipped

Lowering the replay runners to 10 exposes a mismatch the old 15 hid: **the p4g corpus is longer
than the horizon.** Dialogues run to 15 turns and the replay loop searches from every prefix, so
28 of 2697 search roots (1.0 %) now sit at or past Tmax. Under `episode` such a root is terminal,
so `search` returns −1.0 without expanding it — `Ns` stays empty, every `Nsa` is 0, and
`get_action_prob` divides 0/0. Verified: the policy comes back all-`NaN` and `np.argmax` then
silently returns action 0 (`personal story`), attributed to the planner, behind nothing louder
than a `logging.warn`.

`replay_root_is_terminal` (`src/runners/_common.py`) now stops the dialogue at such a root in all
four replay runners. Asking for the next system act there is meaningless — the environment has
already ended the dialogue. **It returns `False` under `legacy`**, which never consults the
horizon, so legacy replay output is unchanged.

Effect on an `episode` replay: 2669 of 2697 turns evaluated instead of 2697. Regression tests:
`test_max_turns_is_shared_and_reaches_the_game`, `test_replay_root_past_the_horizon_is_skipped_not_valued`
(`tests/`, 71 passed).

---

# §10 — Thursday freeze (2026-09-17)

Work is in `analysis/thu/`. The decisions are in `PREREG.md` Entries 2–6. Grid configs:
`analysis/grid/configs/` (generated, validated; §10.6).

## 10.1 Frozen values

| item | value | source |
|---|---|---|
| horizon | `--search_horizon episode` (legacy only as ablation B2) | `thu/p1_pilot.md` |
| valence table | `--emo_valence_table generic` | `thu/remine.md` + addendum |
| AffPool τ | **0.263** = median parent ν under `generic` on the episode NoEmo pilot trees; occupancy **50.0 %**. Superseded, must not be used: 0.35 (`soft`/legacy), 0.078 (`predecision`) | `thu/freeze_doses_generic.json` |
| AffPool / ActPool bias | per plan §4c: 0.25. **Pilots ran at 0.1**, so cost and gate at 0.25 are not measured | `thu/actpool.md` |
| Bias | `--beta_emo 0.70 --emo_signal level` → flip **15.20 %** [13.1, 17.9] | same |
| CenteredBias | `--beta_emo 1.03 --emo_signal level --emo_centre` → flip **15.24 %** [12.6, 18.6] | same |
| Momentum | `--beta_emo 1.11 --emo_signal delta` → flip **15.20 %** [13.4, 17.2] | same |
| **primary budget** | **`--num_mcts_sims 50`**; 20 is the contrast (Block A runs both) | PREREG Entry 6 |
| **success criterion** | **`--p4g_success tag`** (environment). `committed` (detector v2) is an offline sensitivity row | `thu/success_criterion_fix.md` |
| R / K / Tmax | `--max_realizations 4` / `--llm_prior_topk 5` (B1: off) / `--max_turns 10` | plan §4c |
| eval set | `--data data/p4g/rollout_evalset_nonannotated.jsonl --max_conv 100` (runner default for `max_conv` is 20) | `thu/plan_4c_review.md` |
| classifier | `--emotion_classifier hf` (j-hartmann/emotion-english-distilroberta-base) | |
| value / prior scoring | sampled: `--logit_scoring off`; 10-sample value estimator; top-K ranking-call prior | |
| seeds | 1, 2, 3 (pilots used 0) | plan §4c |
| backbone / server | vicuna-13B-v1.5-AWQ; Qwen2.5-7B-Instruct-AWQ for B3–B5. SGLang **0.5.9**, ctx 4096, mem-fraction 0.82, chunked prefill 4096, cuda-graph max bs 64, schedule `lpm`, `serve_supervised.sh` | `calib/serve.sh` |
| config integrity | every grid run passes `--frozen_config`; arm flags passed where they are inert are refused (`check_inert_arm_flags`) | `runners/_common.py` |

Flip rate = argmax flip against NoEmo, one-step replay on the P1 episode NoEmo trees (6,360 selection
points, dialogues 131–140). The two rounded doses match Bias to within
0.04 points.

## 10.2 The emotion term's level under each table (human request)

Mean `Q_emo` on **visited** edges, P1 episode NoEmo trees. This is how much of Bias is still a flat
exploration term, which CenteredBias and Momentum exist to remove.

| table | **visited-edge mean `Q_emo`, level** (sd) | visited-edge mean, delta | **Bias β 0.7 pick-change rate** | τ_med |
|---|---|---|---|---|
| **`generic` (frozen)** | **+0.193** (0.247) | +0.023 (0.172) | **15.2 %** [13.1, 17.9] | 0.263 |
| `soft` (shipped) | +0.267 (0.165) | +0.013 (0.108) | 11.9 % [10.4, 13.3] | 0.279 |
| `predecision` | +0.070 (0.098) | +0.000 (0.080) | 7.5 % [6.1, 8.9] | 0.078 |

CenteredBias's centred term averages **0.000** on visited edges under `generic`, by construction.

**Under `generic`, Bias is less of a pure exploration suppressor than under `soft`.** `generic` has
negative weights, so a visited edge can carry negative `Q_emo`. Bias's flips under `generic`:
visited-over-unvisited **30.9 %**, to unexpanded **4.8 %**, between visited **64.3 %**. On the same
trees under `soft`: 42.5 / 0.1 / 57.4 %. CenteredBias under `generic`: 9.9 / 9.0 / 81.1 %. Momentum:
18.4 / 29.4 / 52.2 %.

## 10.3 Code landed Thursday (all default-off; bit-identity kept)

| flag / change | default | tests |
|---|---|---|
| `--aff_pool_key {affect,act}` (ActPool) | affect | GOLDEN + pre-change AffPool snapshots at 6 (β, bias) |
| `--frozen_config PATH` | none | inherited-default rejection for classifier, R, n_sims, top-K, seed |
| `check_inert_arm_flags` (τ on ActPool, pool settings without `--aff_pool`) | always on; legal configs unchanged | yes |
| `--emo_valence_table predecision` | soft | equals mined output |
| `--p4g_success {tag,committed,amount}` (rollout.py → game → search) | tag | env == offline scorer on every logged pilot success |
| miner: `--pre_decision_only`, `--base_rate`, `--turn_weighting`, `--recency_gamma` | shipped | shipped table reproduces exactly |

Test suite: **114 passed**. The stub end-to-end regression against the pre-Thursday tree, at β 0 and
0.7, is **IDENTICAL** (SR, AvgT, action sequences, full episode records; `thu/regression/`).

## 10.4 Not in the grid

TrajValue (both gates failed, `thu/trajvalue_gates.md`), TrajPrompt (dropped), `predecision` as default
(C4 ablation only). PREREG Entry 4.

## 10.5 Late decisions (2026-09-17 afternoon)

- **Saturation check** (NoEmo, episode, n_sims 50, non-eval dialogs 141–240, n = 100): SR **0.78**
  [0.69, 0.85], so not saturated. Wall clock 242 s/dialog. The cumulative SR went 1.00 after 10 dialogs to
  0.78 after 100: **10-dialog pilots in this environment can be off by more than 0.2.** The P1 horizon
  contrast is re-measured at n = 100 in B2 before any number from it is reported.
- **Success criterion:** `tag` in the environment. Success ends search branches, so an imperfect strict
  detector corrupts backups. The genuine hedge rate is 3/78 tag-successes (3.8 %), or 6.4 % with the
  ambiguous turns. The hedge detector moved v1 → v2: the hedge must govern the donation verb, and a firm
  first-person commitment overrides it. On all 134 logged successes: v1 caught 5 of 6 hedges with
  2 false positives; v2 catches 6 of 6 with 0. **v2 is fitted, not validated** (single reader, written on
  those turns). A second reader labels grid `[donate]` turns before `committed` is cited.
- **B1:** `--algo gdpzero --game p4g --llm_prior_topk 0`. At top-K 5 (every other run) NoEmo is GDP-Zero's
  search bit for bit: 10/10 identical end to end, 0 bracketing flips in 120,085 real selection points.
  With top-K off, the PUCT bracketing difference breaks histogram-prior ties differently at 0.27 % of
  selections, so B1 uses GDP-Zero's own planner. `thu/b1_diagnosis.md`.
- **C4** (`predecision` ablation): Bias only, so no τ is needed. β stays 0.70 under the fixed-anchor rule
  (Entry 5). Its achieved flip rate on the episode pilot trees under `predecision` is **7.5 %**, and it is
  reported beside the result.
- **B3–B5 (Qwen):** run at n_sims 50 (primary). Vicuna doses. Bias's flip rate on Qwen trees is measured
  from B_Qwen_NoEmo's logs (CPU) and reported beside it.

## 10.6 Grid configs

`python analysis/thu/scripts/gen_grid_configs.py --p4g_success tag --out analysis/grid/configs`
produces **37 configs**: Block A 28 (6 arms × n_sims {20, 50} × 2–3 seeds), B1, B2, B3–B5, C3 ×3, C4.
C1/C2 have no config because the tagged cache is not built. C5 (Llama) needs its 5-dialog pilot first.

- Every config writes out the full shared block (no runner default relied on) and carries a
  `--frozen_config` JSON of 27–29 values.
- Before being written, every config is parsed by `rollout.py`'s real argparse block and passed through
  `finalize_args` (inert-arm-flag check + frozen-config check).
- Tamper test on a generated config: the untouched config is accepted. Each of these is refused before
  starting: R 4 → 3, the classifier flag deleted (would inherit `llm`), τ added to ActPool, the ActPool key
  flipped to `affect`, Bias β nudged by 1e-7.
- sha256: `commands.sh` 4490d173c035704ca8e1a81e6f0b866eb8f75817b3d0d56ee5f61504e4d434ff, `manifest.json` c265ef6ac2710a71ac3d508d727b94839841cba1f2940cd355764909f72892ca.
