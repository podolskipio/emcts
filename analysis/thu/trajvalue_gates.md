# TASK 6 — TrajValue gates: **both fail**

🚦 **Gate A FAIL, Gate B FAIL ⇒ the value arm dies. Nothing was built.** TrajPrompt was already dropped
(human, 2026-09-17), so no value arm enters the grid. Per the brief, the gates test the channel, not an
implementation.

Script `scripts/t6_trajvalue_gates.py`, numbers `trajvalue_gates.json`. The spec is the human's
(2026-09-17). Any detail the spec does not fix is stated in the script header and repeated in §3.

## 1. Result

| gate | criterion | result | verdict |
|---|---|---|---|
| **A** incremental McFadden R² of `nu_mean + n_negative + neg_run_max + nu_slope` over `nu_final + n_turns` | CI excludes 0 **and** point ≥ 0.02 | **0.0125** [0.0039, 0.0551] | **FAIL** (below 0.02) |
| **B** out-of-fold Brier gain of `base + g(signature)` over `base` | CI excludes 0 | **−0.0024** [−0.0209, +0.0167] | **FAIL** (worse, and CI spans 0) |
| **B** sign stability | no cell flips in > 1 fold | two cells flip in 2 folds (`T1│1-2`, `T1│3+`) | **FAIL** |

Corpus: 285 annotated dialogues with at least one pre-decision turn (donation rate 0.46). ν is
**cross-fitted**: in each of 5 folds, w(e) is re-mined on the training dialogues with the `predecision`
recipe and applied to the held-out fold.

### Gate A in detail

| | value |
|---|---|
| McFadden R², base (`nu_final + n_turns`) | 0.100 |
| McFadden R², + trajectory | 0.113 |
| **cross-validated** log-loss gain | **−0.0072** [−0.0244, +0.0091] |
| cross-validated Brier gain | −0.0017 [−0.0083, +0.0047] |
| cross-validated AUC | 0.698 → **0.697** |

**The in-sample increment is positive only because the models are nested.** Four extra parameters on
285 dialogues always raise in-sample R², which is why the lower CI bound (0.004) sits above zero. Out of
fold, the trajectory features make prediction slightly **worse**. So the gate's "CI excludes 0" condition
was nearly automatic, and the 0.02 bar is what did the work.

**Per-feature contributions** (R² lost when each feature is dropped last; standardized coefficient):

| feature | Δ R² | coef |
|---|---|---|
| `neg_run_max` | 0.0061 | −0.49 |
| `nu_slope` | 0.0049 | −0.23 |
| `nu_mean` | 0.0014 | −0.15 |
| `n_negative` | 0.0007 | +0.24 |

The two **momentum summaries** (`neg_run_max`, `nu_slope`) carry most of what little there is, and both
point the **"wrong" way**: longer negative runs and *rising* ν go with less donation. This was the brief's
second, independent test of momentum inside the value channel, and **momentum is not supported here
either**. That matches Wednesday's corpus and simulator results.

### Sustained-negative vs recovering

| trajectory (pre-decision, cross-fitted ν) | n | donation rate [Wilson 95 %] |
|---|---|---|
| sustained negative (negative run ≥ 2, final turn ν < 0) | 156 | **0.49** [0.41, 0.56] |
| recovering (negative run ≥ 2, final turn ν ≥ 0) | 65 | **0.28** [0.18, 0.40] |
| never negative | 17 | 0.76 [0.53, 0.90] |

**The contrast runs opposite to the recovery hypothesis.** Users who recover from a negative run donate
*less* than users who stay negative (0.28 against 0.49; the 95 % intervals do not overlap). A value
adjustment that rewards recovery would push the planner the wrong way. Only 17 dialogues never go
negative, because the `predecision` table centres ν near 0 (§3).

### Gate B in detail

| cell (tercile ν_final │ neg_run_max) | g, folds 1–5 | training n, folds 1–5 |
|---|---|---|
| T1 │ 0 | 0 in every fold (**structurally empty**: ν_final < 0 implies a run ≥ 1) | 0 |
| T1 │ 1–2 | −0.00, −0.02, **+0.11, +0.10, +0.08** | 23–34 |
| T1 │ 3+ | +0.01, −0.08, −0.23, −0.01, **+0.03** | 42–53 |
| T2 │ 0 | +0.14, +0.25, +0.25, +0.00, **−0.25** | **1–4** |
| T2 │ 1–2 | +0.13, +0.14, +0.04, +0.19, +0.24 | 29–42 |
| T2 │ 3+ | −0.15, −0.18, −0.16, −0.20, −0.09 | 30–46 |
| T3 │ 0 | +0.25, +0.19, +0.25, +0.25, +0.25 | 9–13 |
| T3 │ 1–2 | −0.11, −0.09, −0.13, −0.12, **+0.06** | 29–36 |
| T3 │ 3+ | −0.23, −0.25, −0.25, −0.25, −0.16 | 31–37 |

Out-of-fold Brier: base 0.2495, with g 0.2519. Four cells are stable (T2│1–2, T2│3+, T3│0, T3│3+). The
cells that flip are the thin one (T2│0, n = 1–4) and the low-ν_final column. A 9-cell table on 228
training dialogues is at the edge of what the corpus supports.

## 2. Sensitivity (not a rescue): the frozen `predecision` table, not cross-fitted

The same gates with ν from the frozen table, which was mined on all 300 dialogues and so has seen every
outcome it scores:

| | Gate A increment | CV log-loss gain | Gate B Brier gain | sign flips | verdict |
|---|---|---|---|---|---|
| cross-fit (primary) | 0.0125 [0.004, 0.055] | −0.0072 | −0.0024 [−0.021, +0.017] | max 2 | A FAIL, B FAIL |
| frozen table (leaky) | 0.0053 [0.002, 0.045] | **−0.0143** [−0.025, −0.003] | +0.0082 [−0.008, +0.024] | 0 | A FAIL, B FAIL (CI spans 0) |

Even with the leak, neither gate passes. Under the frozen table the trajectory features hurt out of fold
with a CI that excludes zero, and g's signs become stable, which is expected when the table has seen the
labels. **No further variants were run.** Choosing among gate variants after seeing the failure would be
the forking-paths problem the pre-registration exists to prevent.

## 3. Implementation choices the spec left open

| choice | what was done | why |
|---|---|---|
| turn unit | whole persuadee turns (`user_turns.parquet`) | the planner classifies whole simulated replies |
| cut | strictly before the first decision act | Task 2 cut, as the spec requires |
| ν | cross-fitted `predecision` recipe (soft, α 50, unit base) at the turn unit | the frozen table; the spec requires cross-fitting w alongside g |
| "negative" | ν < 0 | the table's own zero. With a unit-base table ν is centred near 0, so negative runs are common (268 of 285 dialogues have one) |
| folds | stratified by `donated`, seed 20260917; the same folds for w and g | cross-fit w alongside g |
| g | cell rate − training base, centred (unweighted over 9 cells), clipped ±0.25, empty cell 0 | spec |
| held-out prediction | clip(base + g, 0, 1) against base | the Brier comparison the spec names |

**Two spec points for the record, even though nothing is built.**

1. **The value range is [−1, +1], not [0, 1].** `predict` returns the mean reward over 10 sampled
   persuadee acts on {−1, −0.5, 0, +0.5, +1} (`utils/rewards.py`). Pilot backups include −1 (17.6 % below
   zero). `clip(v + α·g, 0, 1)` would map every predicted failure to neutral. The build should clip to
   [−1, +1].
2. **Units.** g is a difference in donation probability, and v spans a range twice as wide. At α = 1 the
   adjustment is half as strong as the table suggests. α = 2 puts g in value units.

## 4. What this closes

- **TrajValue: dead.** Its prediction row in PREREG Entry 1 ("TrajValue > NoEmo; its advantage does NOT
  depend on n_sims") needs a dated entry recording that the arm was dropped at its gate before any grid
  run. Recording that is not a subgroup choice, but the entry must exist.
- **TrajPrompt: dropped** (human decision). Its PREREG row ("TrajPrompt ≥ TrajValue") needs the same
  kind of entry.
- **Momentum inside the value channel: not supported.** Negative-run length and slope predict weakly, in
  the opposite direction, and not out of fold.

---

## Addendum (2026-09-17): the recovery contrast, re-checked under the frozen table

**Human question:** is "recovering 0.28 vs sustained-negative 0.49" pre-cut or post-cut? It is one of
three convergent measurements, alongside the happiness sign flip and emotion appeals after falling
affect.

**It was pre-cut**: features stop before the first decision act, with ν cross-fitted. Recomputed under
each table, with a 2,000-replicate dialog bootstrap on the difference:

| ν table | window | recovering (run ≥ 2, final ν ≥ 0) | sustained negative (run ≥ 2, final ν < 0) | recovering − sustained, 95 % CI |
|---|---|---|---|---|
| **`generic` (frozen)** | **pre-decision** | **0.306** (n 72) [0.21, 0.42] | **0.479** (n 140) [0.40, 0.56] | **[−0.303, −0.028]** |
| `soft` | pre-decision | 0.262 (n 65) | 0.489 (n 94) | [−0.382, −0.081] |
| `predecision` (frozen, not cross-fit) | pre-decision | 0.333 (n 66) | 0.457 (n 127) | [−0.263, **+0.023**] |
| cross-fit `predecision` recipe (§1) | pre-decision | 0.277 (n 65) | 0.487 (n 156) | — |
| `generic` | **full dialogs (leaky)** | 0.503 (n 169) | 0.462 (n 104) | [−0.076, +0.157] |
| `soft` | full dialogs (leaky) | 0.453 (n 150) | 0.457 (n 70) | [−0.139, +0.142] |

- **Under `generic`, the frozen table, it holds**, and the difference CI excludes 0.
- **It exists only pre-cut.** On whole dialogs it disappears under every table. Donors turn positive
  after agreeing and so are relabelled "recovering", which cancels the effect. The leakage cut is what
  exposes the finding. Without the cut the finding cannot be seen, so this is not an artefact of the cut.
- **One weakness to state:** under the frozen `predecision` table the CI reaches +0.023. The finding is
  robust to 2 of the 3 tables that could plausibly define "negative", not all 3.
