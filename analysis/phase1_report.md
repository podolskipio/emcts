# Phase 1 report — P-VAR and P-RELABEL

> ⚠ **Status update (2026-09-17).** Pilot P1 has run: `--search_horizon episode` is the recommended
> and now frozen grid default (`thu/p1_pilot.md`, `PREREG.md` Entry 5). The gate readings below were
> computed on **legacy** trees under the shipped `soft` valence table, and both of those changed at the
> freeze. Under `episode`: NodeKey ω² 0.151 → 0.035, the AffPool ratio on z 0.71 → 0.98, τ_med
> 0.40 → 0.28 (and 0.263 under the frozen `generic` table). **On the task return — the quantity AffPool
> actually pools — the gate still passes under `episode`: 0.64 [0.53, 0.75] keyed, 0.57 unkeyed**
> (`thu/actpool.md` §4). Treat the numbers below as the legacy-tree record.

**Scope.** Two gate readings (NodeKey, AffPool) and one sensitivity result (P-RELABEL). **The gates
are not applied here.** Each component sits beside its proposed threshold, and the decision is yours.

> ⚠ **Correction, 2026-09-14 (later the same day).** Tree search does not stop at the episode
> horizon: 19.7 % (D1) and 12.7 % (D2) of simulation steps, and 75 % / 50 % of trees, involve dialogue
> states past Tmax that no episode can reach. An earlier version of this document said "under 1 % of
> visits"; that was wrong. Removing those rows **halves D1's pooled ω² (0.084 → 0.041, below the 0.05
> proposal)**, moves the AffPool evidence components (multiplier 2.6 → 2.0), and weakens the
> within-prefix effect (std −0.275 → −0.215). The gate numbers below were computed on contaminated
> trees. See `phase1/SEARCH_HORIZON_BUG.md` (fix behind `--search_horizon episode`; pilot Thursday
> 2026-09-17). **Do not decide the gates on these numbers before that pilot.**

**Detail.** `analysis/phase1/` holds `p_var.md` (full narrative, D1 and D2), `p_var.json`
(machine-readable), `p_relabel.md`, `definitions.md`, `instrumentation_check.md` and `scripts/`
(everything reproducible end to end).

**Data.**
- **D1:** β 0.7. **D2:** β 0 control.
- Both: 30 dialogues, positions 101–130 of the non-annotated p4g pool (disjoint from the eval set).
- Config: vicuna-13B-AWQ on SGLang, sampled value and prior (logit scoring off), R = 4, n_sims 50,
  K 5, Tmax 10, persona, HF classifier, level signal, soft table.
- **D1 and D2 are reported side by side, never averaged.**

---

## 1. The gates

### NodeKey (median split, depth ≥ 2)

| component | threshold | **D1** (β 0.7) | **D2** (β 0) |
|---|---|---|---|
| discordance rate | ≥ 0.30 | **0.308** [0.284, 0.338] ⚠ CI straddles | **0.249** [0.220, 0.282] |
| within-prefix mean Δ | CI excludes 0 and \|Δ\|/sd(z) ≥ 0.2 | **−0.055** [−0.067, −0.043]; std **−0.275** | **−0.055** [−0.069, −0.038]; std **−0.268** |
| pooled ω² | ≥ 0.05, CI lower bound > 0 | **0.084** [0.055, 0.127] | **0.083** [0.063, 0.102] |
| boundary fraction (±0.1) | ≤ 0.35 | **0.221** | **0.214** |
| bucket–depth r | \|r\| ≤ 0.3 | **−0.290** [−0.334, −0.245] ⚠ CI straddles | **−0.240** [−0.282, −0.190] |
| *discordant edges (stop if < 200)* | | 1,447 | 1,137 |

Diagnostics that change how the gate reads:

| | D1 | D2 |
|---|---|---|
| **edge-demeaned ω²**: the part not attributable to prefix identity | **0.0066** [0.005, 0.011] | **0.0077** [0.004, 0.015] |
| fresh steps¹: Δ / std / ω² / edge-demeaned ω² | −0.079 / −0.395 / 0.123 / 0.017 | −0.073 / −0.359 / 0.106 / 0.017 |
| post-split (edge, bucket) children at depth ≥ 2 with ≥ m = 3 visits | 43 % (median 2) | 36 % (median 2) |
| label split: discordance / Δ std / ω² | 0.061 / −0.21 / 0.007 | 0.054 / −0.17 / 0.012 |

What these say:

1. **The premise holds, but the effect is small.** A less positive sampled parent gives a less
   positive child reaction on the same edge. The effect is the same with and without the affect
   channel, so it is **not circular**, and it grows with depth.
2. **About 90 % of the pooled ω² is prefix identity.** Within an edge, the bucket explains under 2 %
   of z variance.
3. **About 70 % (D1) to 75 % (D2) of depth-≥2 edges never see both buckets.** A split relabels
   those edges rather than refining them.
4. **Split children would mostly sit below the m = 3 guard.**
5. **The median split is happy-vs-neutral, not negative-vs-rest.** On the label split, which
   carries the negative-affect hypothesis, every component is far weaker.

¹ With R = 4 and the prefix-keyed realization cache, a visit made after an edge's cache fills
reuses a random cached child, whatever the current parent realization. Only fresh generations are
causal parent → child observations (`definitions.md`).

### AffPool (median split, per tree, key = bucket × act)

| component | threshold | **D1** | **D2** |
|---|---|---|---|
| median n_prefixes per cell | ≥ 4 | **3** [3, 3] | **3** [3, 3] |
| median evidence multiplier | ≥ 3 | **2.59** [2.0, 3.0] (in-cell²: 4.0) | **2.50** [2.0, 3.0] (in-cell: 3.9) |
| median top-prefix share | ≤ 0.6 | **0.625** [0.60, 0.67] | **0.643** [0.60, 0.67] |
| between / within variance ratio | ≤ 1.0 | **0.785** [0.69, 0.86] (noise-corrected 0.31) | **0.783** [0.73, 0.86] (0.29) |

- **Depth-keyed variant** (bucket × act × depth bin): multiplier 1.33 / 1.25, ratio 0.58 / 0.62.
  Heterogeneity improves about 25 %, but evidence falls to about 1.3, far below 3. **Not
  recommended.**
- **Label split:** n_prefixes 2–3, multiplier 2.0. Negative-label cells have a median of 1 prefix.
- **β has no effect on AffPool.** Pooled prefixes are fairly homogeneous in z; the weak side is
  evidence gained. The typical cell pools 3 prefixes, one of which supplies most of the visits.

² The multiplier's denominator can be the edge's total N(s,a) (primary, the node statistic the pool
supplements) or its in-bucket visits. The brief's wording allows both.

## 2. Depth visit profile

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7–10 | 11+ |
|---|---|---|---|---|---|---|---|---|
| D1 share of 33,903 visits | 25.3 % | 21.5 % | 16.6 % | 11.5 % | 7.7 % | 5.2 % | 8.9 % | 3.4 % |
| D1 median visits per parent node | 49 | **9.5** | 5 | 4 | 4 | 3 | 3 | ≤ 3 |
| D2 share of 28,129 visits | 30.8 % | 26.0 % | 18.4 % | 10.7 % | 5.7 % | 3.0 % | 4.1 % | 1.3 % |
| D2 median visits per parent node | 49 | 7.5 | 4 | 3 | 3 | 3 | 2–3 | ≤ 3 |

- **Not concentrated at depth ≤ 1:** 75 % (D1) and 69 % (D2) of visits are at depth ≥ 2. Depth-2
  parent nodes get about 8–10 visits, as expected.
- β = 0.7 makes trees narrower and deeper.
- ⚠ **Search-horizon bug** (`phase1/SEARCH_HORIZON_BUG.md`): only a donation terminated a simulated
  branch, so search expanded states past the turn limit. Depth is counted from each turn's root, so
  the right measure is root turn + depth > Tmax: **19.7 % of D1 steps, 12.7 % of D2 steps, and 76 %
  of the search at turn 9.** (An earlier version of this line said "under 1 % of visits"; that counted
  only tree depth > 16 and was wrong.) Fixed behind `--search_horizon episode`; pilot on Thursday.

## 3. Confound checks

### Bucket vs depth (§7.5)

- ν rises steeply with depth in both runs. Median parent ν is about 0.24 at depths 1–2 and 0.53 by
  depth 10, as simulated users converge to happiness.
- Correlations: r(bucket, depth) = −0.29 / −0.24; r(bucket, turn) = −0.13 / −0.14.
- **The effect survives stratification.** With a depth-specific median split, which cannot encode
  depth: Δ −0.046 / −0.043, std −0.23 / −0.21, ω² 0.069 / 0.070. With ω² groups also keyed on
  absolute dialogue position: 0.086 / 0.077.
- **Not the depth confound.** The within-prefix contrast is immune by construction.

### Channel dominance (§7.6)

| | D1 (actual β 0.7) | D2 (counterfactual β 0.7) |
|---|---|---|
| median \|β·ΔQ_emo\| / \|ΔQ\|, top-2 siblings | **0.27** [0.23, 0.32] | 0.25 [0.21, 0.28] |
| argmax flip, full PUCT | **31.6 %** [28.2, 34.8] | **15.5 %** [12.6, 18.8] |
| argmax flip among expanded siblings, exploitation terms | 11.5 % | 11.7 % |
| flips picking a visited act over an unvisited one | **83 %** | 43 % |
| selection points with ≥ 2 expanded siblings | 38 % | 52 % |

- **The affect term does not dominate the task signal.** It is about a quarter of the Q gap, and
  among expanded siblings it flips about 11.5 % of choices regardless of which tree it runs on.
- **Its main behavioural effect at β 0.7 is less exploration.** Q_emo averages +0.26 on visited
  edges, so β·Q_emo is a flat ~+0.18 bonus for anything already tried. D1 has fewer expanded
  siblings, which produces more visited-over-unvisited flips.
- An "emotion helps" result therefore partly measures an exploration-schedule change.

## 4. P-RELABEL verdict: **weakens**

- **Reproduction.** The paper's §IV-D / Figure B reproduces exactly with the original pipeline, and
  a fresh DistilRoBERTa run agrees 100 % with the logged labels.
- **Direction holds under all three classifiers.** After a negative user emotion, EmoMCTS plays more
  trust-building (+0.13 / +0.20 / +0.17 vs GDP-Zero) and less "other".
- **Never statistically supported.** Under the original classifier: n = 15 per cell, CI
  [−0.26, +0.53]. Under GoEmotions and EmoBERTa the "after negative" cells shrink to **4–6** system
  turns, and no CI excludes 0.
- **The labellers flag largely different utterances as negative.** 9 of the original's 41
  negatives are shared with each alternative. Negative base rate: 8.1 % / 1.8 % / 2.6 %.
  κ = 0.25 / 0.10 / 0.04, in or below the C4 audit's 0.11–0.21.
- **Paper implication.** Report the pattern as a direction consistent across labellers, with n and
  CI, not as evidence of emotion-conditioned repair.

## 5. GPU hours

| run | wall clock | GPU-h |
|---|---|---|
| D1 (β 0.7) | 10,707 s | **2.97** |
| D2 (β 0) | 10,018 s | **2.78** |
| **total** | | **5.75** |
| instrumentation check, P-RELABEL, all analysis | CPU | 0 |

## 6. σ_emo (supersedes 1.43 / 9.10, which must not be cited)

- **Ratio σ_emo / |Q_emo|, median:** 0.45 (D1) and 0.50 (D2).
- **p90:** 1.45 and 1.46.
- **Median N:** 6 and 5. About 22 % of edges have N = 2; those show deflated σ, so excluding them
  would raise the ratio slightly.

## 7. Not done, or done differently, and why

- **Bit-identity used the repo's deterministic stub backbone, not SGLang.** SGLang sampling is
  unseeded, so identical decisions from a real-LLM run can't be verified even for unchanged code.
  Verified at β 0.7 and 0, R 3 and 4. The real HF classifier was used.
- **Config deviates from the brief in two places, by your decision:**
  - R = 4 (frozen calib GRID) instead of 3;
  - logit scoring off (brief) although the calib GRID uses `--logit_scoring both`. **If the grid keeps
    logit scoring, this run's value/prior path differs from the grid's**, which is the transfer risk
    the brief warned about for §7.6 and the variance numbers. Settle this before citing these numbers
    for the grid.
- **Timing was measured in flight** rather than on a separate single-dialogue run: one serial
  dialogue does not predict 10-worker throughput. The projection stayed under 4.5 h. D2 was run
  because D1 landed at 14:36 and D2 finished the same day.
- **The §IV-D source runs were not in the repo.** They were recovered from an earlier session's
  scratchpad (three md5-identical copies) and are now under `analysis/phase1/relabel_src/`.
- **P-RELABEL covers the paper's runs, not D1/D2.** D1/D2 have no GDP-Zero arm, so they can't
  reproduce the §IV-D comparison.
- **Reproducibility:** with 10 workers the numpy RNG is shared across threads and SGLang is
  unseeded, so the seed is recorded but runs will not reproduce draw for draw. Analysis is
  deterministic given the logs.
- **Diagnostics added beyond the brief**, because the specified components alone would have
  misled: fresh-step estimators, edge-demeaned ω², post-split visit counts, flip decomposition,
  noise-corrected homogeneity ratio, the depth-specific median split. None of these changes a
  specified component.
- **Search-horizon bug found after the first version of this report.** The D1/D2 gate numbers include
  impossible states. The sensitivity with those rows removed is in `phase1/SEARCH_HORIZON_BUG.md` §4,
  the fix is `--search_horizon episode` (default `legacy`, bit-identical), and pilot P1 (10
  dialogues) runs on Thursday 2026-09-17. Disclosed whichever way P1 comes out.
- **Not committed.** The instrumentation (3 files under `src/`) and `analysis/phase1/` are in the
  working tree on `paper-fixes`, uncommitted.
