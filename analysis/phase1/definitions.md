# Phase 1 — definitions confirmed in code (§3)

Code state: branch `paper-fixes` at `2ed83c1` plus the Phase-1 instrumentation (write-only; see
`instrumentation_check.md`). Line numbers refer to the instrumented files.

None of the five items turned out ambiguous. Three findings change how the brief's analysis has to
be read, and they're flagged **⚠** below.

---

## `z` — what gets backed up into `Q_emo`

**One definition, no normalization under `--emo_signal level`.**

- Code path: `EmotionAwareMultiObjectiveQ.search` (`src/mcts/emotion_mcts.py:611-612`). After
  the recursive `v = self.search(next_state)` returns:
  `emo_v = self._emotion_signal(state, next_state)` → `self._update_emo_channel(hashable_state, best_action, emo_v, nsa_old)`.
- `_emotion_signal` (`:455-475`): under `level` it returns
  `z_level = self._emotion_quality(self._attached_distribution(next_state))`, which is
  `ν(d_{s'})` of the **child realization** `next_state`, i.e. the distribution attached to its
  last turn (the simulated user reply). No rescaling. (The `/2` applies only to `delta`.)
- The parent argument is ignored under `level`. The existing unit test
  `test_level_signal_ignores_the_parent_entirely` covers that.
- `z` is local (the immediate child's user reaction), not the leaf's.
- The instrumentation logs `z` as the exact value passed to `_update_emo_channel`, and
  `check_instrumentation.py` asserts `z == child_nu` on every step.

**⚠ Range.** `ν ∈ [−1, +1]` holds only in principle. With the soft table and the HF classifier
(which never emits Contempt), `ν ∈ [min w, max w] = [−0.14, +0.54]`. So
`β·|ΔQ_emo| ≤ 0.7 × 0.68 = 0.48`, not 1.4.

## `ν(d)` — soft, not argmax

`_emotion_quality(dist)` (`:437`) = `Σ_e dist[e] · valence_weights[e]` over the full
distribution. `valence_weights` is `EMOTION_VALENCE_TABLES["soft"]`, the same object as
`EMOTION_VALENCE_MINED`. An empty or `None` distribution gives `0.0`.

The distribution is the HF pipeline's `top_k=None` score vector over its 7 labels, mapped to
`Emotions` and renormalised to sum 1 (`src/emotion_classifiers/hf_emotion.py:43`). `Contempt` is
present in the dict with probability 0. Attached in `EmotionAwarePersuasionGame.get_next_state`
(`src/games/p4g_game.py:159`). Logged vectors therefore have 8 keys, with Contempt ≡ 0: effectively
the 7-vector.

## parent ν — the realization sampled at the parent

In `search`, after PUCT selection at node `s`:
`state = self._sample_realization(hashable_state)` (`:562`), a **uniform draw
(`np.random.randint`) from the node's realization pool**. That `state` is passed to
`_get_next_state(state, best_action)`, so it is the dialogue the system utterance and the user
simulator are conditioned on when a fresh child is generated.

`parent_nu = ν(state.predicted_distribution())`, logged right there (`:583`) from the very
object that was drawn. No ordering assumption is involved. Alignment is verified by content id: every
depth-≥2 step's `parent_realization_id` resolves to a depth−1 step in the same tree that generated
it, with identical ν and distribution. Every depth-1 step's parent ν equals the turn's observed
`root_nu`.

**⚠ Parent ≠ the realization the simulation descended through.** The recursion passes
`next_state` down, and the child node adds it to its pool (`_add_new_realizations`). It then
*re-samples* uniformly from that pool before its own transition. On the stub run, only 1439 of
4201 depth-≥2 steps (34 %) have `parent_realization_id` equal to the previous step's
`child_realization_id` in the same simulation. The brief defines parent ν as "the realization
sampled at the parent during that simulation", and that is what's logged.

**⚠ Cached children are not conditioned on the current parent.** `_get_next_state` (`:109-111`)
returns a uniformly sampled **cached** child once the child pool holds `max_realizations` entries.
That cached child was generated from whichever parent realization was drawn when it was
generated, not necessarily the one drawn now. Only fresh generations
(`child_from_cache == False`) are causal (parent → child) observations; for cached steps the
`(parent_nu, z)` pairing is re-randomised. The realization cache is keyed on the action prefix
alone, and the brief forbids changing that keying. So **every NodeKey estimator is reported twice:
on all steps (what `Q_emo` actually averages, and what a NodeKey split would see under the current
cache) and on fresh generations only (the causal premise)**. For a cached step, the generating
parent's ν is recoverable exactly by joining `child_realization_id` to the fresh step that produced
it (`generating_parent_nu` in `steps.parquet`).

**Root.** `_init_node` seeds the root pool with `[state.copy()]` (`:44`). Every later root visit
calls `_add_new_realizations(root)`, whose `state in pool` test uses `__eq__` on the history, so
the pool stays size 1. The root bucket is constant within a tree, and depth-1 edges have no
within-tree bucket variation. Confirmed on data: depth-1 `parent_nu == root_nu`.

## `Q_emo`, `M2_emo` — Welford, independent of β

`_update_emo_channel` → `welford_update(n_old, mean, m2, z)` (`:243`):
`δ = z − mean; mean += δ/(n_old+1); δ₂ = z − mean; m2 += δ·δ₂`. `n_old` is `Nsa` before the
increment, so the mean is the running mean. `M2_emo` is the sum of squared deviations
(`var = M2/(N−1)`). Both are initialised to `0.0` (not `Q_0`) in `_init_node` (`:483`).

The call is **unconditional**: `beta_emo` appears only in `_calculate_uct`. At β = 0 the channel is
still accumulated, which is what makes D2's counterfactual §7.6 analysis possible. `Q_emo` and
`M2_emo` of a sibling are logged as they stood at the moment of choice (`siblings`).

`Q` ∈ **[−1, +1]** (`utils/rewards.py` p4g: no donation −1 … donate +1, terminal ±1), not [0, 1]
as §7.6 assumes.

## `Q_pool` scope — per search

**`Q_pool` / AffPool / NodeKey do not exist in the code.** The only scope evidence available is
architectural, and it is unambiguous: `runners/rollout.py:pick_action` constructs a **new**
`EmotionAwareMultiObjectiveQ` for every dialogue turn (`dp = EmotionAwareMultiObjectiveQ(...)`,
`:95`), runs `num_MCTS_sims` searches from the observed state, takes the argmax root visit, and
discards the object. Any table held on the planner is therefore per `(dialogue_id, turn_index)`
search. That matches the brief's §2 description. Per-turn persistence would need AffPool to be
built outside the planner (runner- or game-level), which nothing in the code or `analysis/*.md`
describes. §8 is computed per tree, as the brief specifies. If AffPool is later designed to persist
across turns, §8 has to be recomputed at corpus level. `p_var.py` has no switch for that today: the
cell key in `affpool_cells` would have to drop `tree_key`.

## ⚠ The depth cap does not bound search

`Game._failure_or_continue` returns −1.0 once `len(state) >= max_conv_turns`, which is `--max_turns`
= 10. The open-loop `search` (`OpenLoopMCTS.search` and `EmotionAwareMultiObjectiveQ.search`)
treats only `terminated_v == 1.0` (donation) as terminal. A branch that reaches the turn limit is
therefore expanded and valued by `planner.predict` like any other node, not scored −1 and stopped.
In D1's first finished dialogue a turn-2 tree reaches depth 10, absolute dialogue turn 12. This
pre-dates Phase 1 (it is in the initial commit). The policy default is left unchanged, and the fix
sits behind `--search_horizon episode`. Measured as root turn + depth > Tmax, it affects 19.7 % of D1
steps and 12.7 % of D2 steps. Full write-up and sensitivity: `SEARCH_HORIZON_BUG.md`.

## Conventions used in the analysis

| name | definition |
|---|---|
| edge | `(dialogue_id, turn_index, action_prefix, action)`; `action_prefix` = acts from the search root to the parent |
| `depth` | edge depth = `len(action_prefix) + 1`. Root's outgoing edges are depth 1; "depth ≥ 2" = parent is a sampled (non-root) realization |
| tree | `(dialogue_id, turn_index)` |
| `τ_med` | median of `parent_nu` over all step rows of the run (D1 and D2 each get their own) |
| `bucket_med` | `parent_nu < τ_med` |
| `bucket_lab` | argmax of `parent_emotion_dist` ∈ {fear, sadness, anger, disgust} |
| fresh step | `child_from_cache == False` (the child was generated from this parent realization) |
| realization id | sha1[:12] of the `(role, act, utterance)` history, so identical text gives an identical id, matching pool dedup |
