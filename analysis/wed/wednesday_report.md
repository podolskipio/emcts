# Wednesday report (brief dated Wed Sep 16; executed Tue 2026-09-15)

> ⚠ **Partly superseded by the Thursday freeze (2026-09-17).** What changed:
> **τ** — `--aff_pool_tau 0.35` is superseded by **0.263** (median ν under the frozen `generic` table on
> `episode` trees; 0.35 would put 93 % of steps in one bucket under `predecision`, 59 % under `soft`).
> **Valence table** — the grid runs `generic`, not `soft` (`thu/remine.md`).
> **Horizon** — the grid runs `episode` (`thu/p1_pilot.md`).
> **Momentum** — `--emo_signal delta` is the per-edge *average local affective change*, not β·Δν, and the
> selection arms are dose-matched on flip rate, not on β (`thu/momentum.md`, `thu/tau_dose.md`).
> **AffPool's affective key** — the §7.4 finding (the key buys no homogeneity on the task return, at half
> the evidence) is **confirmed** on episode trees; ActPool is built and is the control (`thu/actpool.md`).
> **Constrain** — not in the grid. The measurements below stand as the legacy/`soft` record.

Everything below is in `analysis/wed/`. Every statistic is in `wednesday.json`. CIs are cluster bootstrap over
dialogues (1000 replicates, percentile) unless stated. D1, D2 and D3 are never averaged. No default was flipped and no gate applied.

---

## 1. §5.1: the AffPool gate on the task return (read first) — **passes; no escalation**

`Q_pool` pools the **task return**; it is fed the same `v` as `Q` (confirmed in code, `FREEZE_NOTES.md` §8.2).
Yesterday's 0.31 was measured on z. On the right quantity:

| | z (yesterday, reproduced exactly) | **task return** |
|---|---|---|
| D1 ratio / noise-corrected | 0.785 / 0.312 | **0.501** [0.445, 0.576] / **0.046** [0.005, 0.116] |
| D2 ratio / noise-corrected | 0.783 / 0.286 | **0.646** [0.571, 0.775] / **0.139** [0.049, 0.278] |

Nowhere near 1.0. Two caveats the freeze should see:

- **Fresh steps only** (causal pairs): 0.86 (D1) and 0.92 [0.78, **1.11**] (D2). The D2 upper bound crosses 1.0.
- **⚠ The affective key is not what makes pooling safe.** The *unkeyed* act-only pool is as homogeneous on task
  return with about twice the evidence: D1 nc 0.073 vs 0.046, multiplier 6.67 vs 3.66; D2 0.113 vs 0.139, 4.25 vs
  2.33 (`key_geometry.md` §7.4). The key helps slightly on z, which AffPool does not pool. This is not a §11
  trigger, but it bears directly on AffPool's "affective" claim. Details: `affpool_gate_recheck.md`.

## 2. Build status: all §1 builds landed, flags default off

| flag | default | status |
|---|---|---|
| `--aff_pool`, `--aff_pool_bias` (0.1), `--aff_pool_tau` (0.35) | off | built, tested, swept |
| `--emo_centre` | off | built, tested, one-step replay on D1 |
| `--emo_constraint_tau` (unset), `--emo_constraint_m_warm` (3) | off | built, tested, replay on D2; **τ proposed 0.12, not hard-coded** |
| `--emo_valence_table generic` | `soft` stays | built, tested |
| `--terminal_on_failure` | off | alias for `--search_horizon episode` (same `dest`, one code path) |
| `backup_value` in simlog | always logged | **landed before D3; backfills D1/D2 exactly** (simlog `v`, replay-verified 62,032/62,032 steps) |

**Bit-identity with flags off:** `tests/test_wed_arms.py` (19 new, 69 total, all pass); whole-search sha256
fingerprints equal to the pre-change tree at β 0 and 0.7; the §3.2 stub regression (10 dialogues × 50 sims,
β 0 / 0.7) **IDENTICAL** down to the full episode records. GPU smoke test of every flag on the real stack passed.
Everything is written up in `FREEZE_NOTES.md` §8.

**What the build measurements say:**

- **AffPool sweep** (dialogues 131–140, β_emo 0; **7 of 10 dialogues per arm**, see §8):
  - mean RAVE β at the selected edge is 0.65 / 0.60 / 0.46 for b = 0.05 / 0.1 / 0.25;
  - the pool term flips 31–35 % of selections;
  - at depth ≥ 3 the pool dominates (β > 0.67) at every bias, and only b = 0.25 leaves the edge's own Q dominant at depth 1 (β 0.26);
  - median 3 prefixes per cell, 60 % of cells with ≥ 3.
  
  `s1_affpool_sweep.json`.
- **CenteredBias** (one-step replay on D1's β 0.7 selections):
  - flips 25.5 % [22.6, 28.2] of selections, and 99.9 % of those flips go **to an unexpanded edge**;
  - μ averages 0.31;
  - inert (0 flips) at the 8.4 % of points where all K siblings are expanded;
  - undoes Bias's narrowing: 51 % of D1 points have exactly one expanded sibling, against 38 % in D2.
- **Constrain** (exact replay on D2 at τ ≈ 0.12):
  - mask present at 11 % of selection points, fallback 0.1 %, selection flips 6 %;
  - **root decision differs from unrestricted argmax N on 19 % of turns** (36 % at τ = 0.2).

## 3. Momentum verdict: **the "neither" cell**

- **Corpus** (`momentum_corpus.md`):
  - act × Δν interaction p = 0.22 on the held-out 100 (w(e) re-mined on the other 197);
  - DiD −0.04 [−0.42, +0.35];
  - the all-297 generic-valence sensitivity has p = 0.023, but the same interaction appears with affect *level* (p = 0.017) and the DiD CI spans 0.
- **Simulator** (`momentum_simulator.md`):
  - in D1, D2 and D3, on task return and z, all and fresh steps, **12 of 12 cells fail** the pre-registered rule;
  - every within-depth permutation "effect" disappears under the within-tree × depth permutation, i.e. it is tree-level structure;
  - increments are 0.0002–0.0017 R²;
  - one directional hint (D3 task return, DiD +0.034 [+0.003, +0.082]) does not survive the controls.

**H is wrong as stated.** K3 keeps only its depth-invariance motivation. One reportable residue: in the human corpus,
emotion appeals precede donation more often after falling affect (0.67 [0.56, 0.77] vs 0.43 [0.32, 0.55] after
rising, n = 69–73). This is a *level* effect, not momentum. The §11 "falling < 10 %" trigger did not fire: 25–32 % of turns
and steps fall by more than 0.1.

## 4. Bucket key: **keep K0**; scope: **per-search**

- **No alternative key passes the §5.2 rule in both runs.**
  - K1/K2 (EWMA) each fail on evidence (−14 to −17 %) or on depth correlation in every run.
  - K3 (Δν) is depth-free (|r| 0.11 / 0.07) but no more homogeneous.
  - **K4 (depth-adjusted level) is the near-miss:** homogeneity tied, phase confound gone (r −0.03 / −0.02 vs
    −0.29 / −0.25), multiplier −11.5 % in D1 (tolerance 10 %). It is the defensible alternative if the depth-proxy
    objection matters more than evidence. It would need an online per-depth median, which is not built.
- **Redundancy (§5.3):** affect history adds +0.055 R² (+54 %) over the current state in predicting the next simulated
  reaction with the affect channel on (D1), and +0.016 (+21 %) with it off (D2). Depth adds < 0.01.
- **Drift (§5.4):** between-turn heterogeneity of cell task return is 0.24–0.26 (noise-corrected), 2–5× the
  within-search figure, for a 3–4× evidence gain. K3 drifts no less. Cross-turn scope is not defensible.

Details: `bucket_selection.md`.

## 5. Key geometry: **keep the scalar and the hard partition; §9 not built**

- **7.1:** the 7-vector adds +0.009 (D1) / +0.005 (D2) R² over ν on z, a real but small increment. The fitted weights
  do not recover w (r 0.25–0.32). On task return ν explains almost nothing (R² 0.001).
- **7.2:** effective rank 1.47–1.51 (PC1 81 %). Within-tree neighbourhoods are populated (median 17–31 at cosine ≤ 0.2),
  so the §11 trigger does not fire, but only because d is nearly one-dimensional.
- **7.3:** Euclidean, cosine and |Δν| distances predict |Δz| equally (r 0.10–0.12). The scalar was the right object.
- **7.4:** the RBF kernel is viable on its own terms (h = 0.2: 89–90 % of hard's evidence, better homogeneity).
  **But the unkeyed pool matches it on task return.**
- **7.5:** the 2-d (ν, Δν) kernel halves sharing for no gain. Closed.
- **§9 preconditions not all met** (7.1 is not clean), so `--aff_pool_kernel` was not built and no soft-vs-hard
  pilot was added to Thursday.

Details: `key_geometry.md`.

## 6. D3 status: **landed**

- **Run:** 30/30 dialogues, 0 failures, 3,579 s; integrity check PASS; `backup_value` native.
- **No happiness drift on Qwen.** Median ν stays at 0.06–0.10 from depth 1 to 10+, against Vicuna's 0.24 → 0.53.
  Yesterday's depth confound is a Vicuna property.
- **Lower level, same range.** Paired over the 30 dialogues, Qwen is less positive (parent ν −0.135 [−0.161, −0.104];
  real turns −0.073) with the same spread and negative-label rate (+1.2 pp [−4.4, +6.5]). **Not "far more
  expressive".**
- **σ_emo/|Q_emo| rises from 0.48 to 1.13** because |Q_emo| shrank.
- **Search-horizon contamination is worse:** 29 % of D3 steps are unreachable (D1 20 %), relevant to Thursday's P1.

Details: `qwen_comparison.md`.

## 7. GPU hours: **3.27 GPU-h** (`runs/gpu_hours.json`)

- smoke tests, 2 attempts: 0.38 h
- AffPool sweep: 1.81 h
- D3: 0.99 h
- server loads: 0.08 h

Everything else ran on CPU.

## 8. Not done, or done differently, and why

1. **§4 on the annotated 300, not the 917:** the non-annotated dialogues have no per-turn acts. Split 100 test / 197
   re-mine, decided before the analysis. Consequence: most test-set cells are under 30.
2. **AffPool sweep has 7 of 10 dialogues per arm.** Vicuna's SGLang scheduler hung twice today with its 15k-token KV pool
   nearly full, and the watchdog killed the server both times. The first time coincided with a WSL `dxgkrnl` GPU fault.
   The lost dialogues were the longest ones, so the sample is length-biased, and SR is not comparable across arms. The
   sweep ran at β_emo 0 only. Runs are now served under `scripts/serve_supervised.sh`.
3. **CenteredBias and Constrain were measured by replay on logged trees, not by live runs.** "Expanded-sibling
   distribution against Bias on the same seed" needs a CenteredBias run; that belongs in Thursday's pilots.
4. **§9 soft-ν not built:** a precondition failed (§5).
5. **All tree analyses use `search_horizon legacy` trees** (D1/D2 as frozen, D3 matched to D1). Key numbers
   include reachable-only sensitivities; the Phase-1 caveat stands until P1 (Thursday).
6. **Calibration fixes applied mid-day.** The clustered Wald test over-rejects with sparse act cells: 7–8 % size in
   simulation, and p ≈ 1e-14 / 1e-21 on single-digit acts. Models now drop acts with < 30 rows or < 10 per tercile, and verdicts
   rest on permutations and DiD. The §7 per-query homogeneity estimator was corrected to reproduce the cell-level
   statistic exactly before its numbers were used.
7. **`figures/` is empty:** no figures were produced. All tables are in the markdown and JSON.
8. **Nothing is committed.** The working tree carries the builds, tests and `analysis/wed/`.

## Escalation checklist (§11)

| trigger | status |
|---|---|
| §5.1 task-return homogeneity > ~1.0 | no (0.50 / 0.65); ⚠ unkeyed pool equally homogeneous |
| `backup_value` cannot be logged or backfilled | no: logged, and backfilled exactly |
| corpus per-turn acts unavailable for the 917 | **yes**; resolved 2026-09-15 by the 100/197 annotated split |
| bit-identity test fails with a flag off | no |
| falling occupancy < ~10 % in corpus and simulator | no (25–32 %) |
| within-tree neighbourhood < ~5 at cosine 0.2 | no (median 17–31) |
| D3 cannot be served | no |
