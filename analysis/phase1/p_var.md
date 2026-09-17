# P-VAR — NodeKey and AffPool premises (§7–§9)

> ⚠ **Correction, 2026-09-14 (later the same day).** Tree search does not stop at the episode
> horizon: 19.7 % (D1) and 12.7 % (D2) of simulation steps, and 75 % / 50 % of trees, involve dialogue
> states past Tmax that no episode can reach. An earlier version of this document said "under 1 % of
> visits"; that was wrong. Removing those rows **halves D1's pooled ω² (0.084 → 0.041, below the 0.05
> proposal)**, moves the AffPool evidence components (multiplier 2.6 → 2.0), and weakens the
> within-prefix effect (std −0.275 → −0.215). The gate numbers below were computed on contaminated
> trees. See `SEARCH_HORIZON_BUG.md` (fix behind `--search_horizon episode`; pilot Thursday
> 2026-09-17). **Do not decide the gates on these numbers before that pilot.**

Primary data: **D1** (β = 0.7). The D2 control (β = 0) is reported separately in §D2 and never
averaged with D1. Every number is in `D1/p_var_D1.json`, and the §11 summary is in `p_var.json`.

**Confidence intervals.** 95 % percentile, 1000 replicates, dialogue resampled with replacement.
**Effect size.** ω² throughout, never η².
**Two step populations.** Every NodeKey statistic is reported on *all* steps and on *fresh* steps
(`child_from_cache == False`). Once an edge's R = 4 children are cached, later visits reuse a random
cached child regardless of the current parent realization, so only fresh steps are causal
parent → child observations (`definitions.md`).

## Data (§5)

| | |
|---|---|
| dialogues | 30. Positions 101–130 of the non-annotated p4g pool (ids in `runs/dialogue_ids.txt`); asserted disjoint from eval set 1–100 and from the annotated 300 |
| config | vicuna-13B-v1.5-AWQ on one SGLang server (all roles) · `--logit_scoring off` (10-sample value estimator + top-K ranking-call prior) · R = `max_realizations` = 4 · n_sims 50 · K 5 · Tmax = `max_turns` = 10 · `--p4g_persona` · cpuct 1.0 · Q_0 0.0 · `--emo_signal level` · soft table · `--emotion_classifier hf` · 10 workers · seed 0. Exact command: `runs/D1/command.txt` |
| config vs brief | R = 4 (frozen calib GRID) rather than the brief's 3; sampled scoring as the brief says, although the calib GRID uses `--logit_scoring both`. Both were user decisions made before the run |
| outcome | SR 0.600, 6.83 turns on average (3–10) |
| wall clock / GPU | 10,707 s = **2.97 GPU-h** (1× RTX 4090) |
| log integrity | `scripts/check_instrumentation.py - runs/D1/D1`: 30/30 dialogues PASS (z tape = frozen `per_step_valences`, every parent realization traceable, z = child ν). Output in `runs/D1/alignment_check.txt` |
| `steps.parquet` | **33,903** rows · 30 dialogues · **175** trees · 3,509 prefixes · **5,355** edges · 17,208 fresh rows · 0 unresolved generating parents |
| τ_med | **+0.350**, the median of `parent_nu` over all rows. ν's observed range is [−0.135, +0.537], IQR 0.388 |

Seeding caveat: with 10 workers the global numpy RNG is shared across threads, and SGLang sampling
is unseeded. The seed is recorded, but a re-run will not reproduce D1 draw for draw.

---

## §7.1 Visit profile by depth (read this first)

Depth = edge depth. Depth 1 is the root's outgoing edges, and "median node visits" is the median
visit count of the parent nodes at that depth.

| depth | visits | share | edges | visits / edge | fresh share | median visits per parent node |
|---|---|---|---|---|---|---|
| 1 | 8,575 | 25.3 % | 650 | 13.2 | 0.30 | 49 |
| 2 | 7,284 | 21.5 % | 1,005 | 7.2 | 0.43 | **9.5** |
| 3 | 5,632 | 16.6 % | 987 | 5.7 | 0.55 | 5 |
| 4 | 3,882 | 11.5 % | 770 | 5.0 | 0.61 | 4 |
| 5 | 2,597 | 7.7 % | 568 | 4.6 | 0.64 | 4 |
| 6 | 1,761 | 5.2 % | 399 | 4.4 | 0.67 | 3 |
| 7–10 | 3,017 | 8.9 % | 711 | 4.2 | 0.71 | 3 |
| 11–16 | 877 | 2.6 % | 221 | 4.0 | 0.88 | 3 |
| 17–33 | 278 | 0.8 % | 44 | 6.3 | 1.00 | — |

- **Depth ≤ 1 holds 25 % of visits; 75 % sit at depth ≥ 2.** The "almost all visits at depth ≤ 1"
  stop condition does not apply.
- **Depth-2 parent nodes get a median of 9.5 visits**, matching the brief's expectation of about 10.
  Depth 3 gets 5, not the expected 2.
- **Edges are thin.** The median depth-2 edge has 2 visits. After a two-way NodeKey split, the median
  (edge, bucket) child at depth ≥ 2 has **2 visits**, and only **43 %** (fresh: 35 %) reach the planned
  m = 3 guard. By depth: 47 % at depth 2, 43 % at 3, 38 % at 6.
- **The depth cap does not bound search** (`SEARCH_HORIZON_BUG.md`). This table counts tree depth
  from each turn's root, so it understates the problem. The relevant quantity is root turn + depth.
  **19.7 % of D1 step rows have a child state past Tmax = 10**, in 75 % of trees, rising to 76 % of
  rows at root turn 9. (The rows past depth 16, 0.8 %, are only the extreme tail. An earlier version
  called the problem negligible on that basis; that was wrong.)

## §7.2 Primary estimator: within-prefix bucket contrast (depth ≥ 2)

Bucket 1 = `parent_nu < τ_med` (the less positive parent). Δ = mean z (bucket 1) − mean z (bucket 0),
computed on discordant edges only.

| | all steps | fresh steps |
|---|---|---|
| edges at depth ≥ 2 | 4,705 (2,956 with N ≥ 2) | 4,705 |
| **discordant edges** | **1,447** | 1,313 |
| **discordance rate** | **0.308** [0.284, 0.338] | 0.279 [0.259, 0.306] |
| discordance rate among edges with N ≥ 2 | 0.490 | 0.444 |
| **mean Δ** | **−0.055** [−0.067, −0.043] | **−0.079** [−0.095, −0.062] |
| median Δ | −0.034 | −0.070 |
| sd(z), depth ≥ 2 | 0.201 | 0.200 |
| **standardized effect** | **−0.275** [−0.329, −0.219] | **−0.395** [−0.471, −0.316] |

**Discordance by depth (all steps).** Rate falls with depth:

| depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11–16 |
|---|---|---|---|---|---|---|---|---|---|---|
| rate | 0.35 | 0.33 | 0.36 | 0.31 | 0.29 | 0.27 | 0.27 | 0.22 | 0.18 | 0.10–0.22 |
| n discordant | 348 | 323 | 276 | 173 | 117 | 74 | 51 | 31 | 19 | 35 total |
| mean Δ | −0.036 | −0.026 | −0.049 | −0.058 | −0.087 | −0.104 | −0.110 | −0.154 | −0.133 | |

What this shows:

- The effect has the sign the premise needs: a less positive sampled parent yields a less positive
  child reaction on the same edge.
- It is 1.4× larger on fresh steps, where the pairing is causal. The cache dilutes it.
- It grows with depth, while the number of discordant edges shrinks.
- Only about 31 % of depth-≥2 edges ever see both buckets. On the other 69 %, a NodeKey split would
  relabel the prefix, not refine it.
- There are well over 200 discordant edges, so the contrast is not too thin to carry a gate.

## §7.3 Secondary estimator: pooled decomposition

One-way ANOVA of z on bucket within each (depth, action) group at depth ≥ 2 (44 groups),
visit-weighted across groups:

| | all steps | fresh steps |
|---|---|---|
| **ω²** (unclipped) | **0.084** [0.055, 0.127] | 0.123 [0.084, 0.178] |
| ω² clipped at 0 | 0.085 [0.056, 0.128] | 0.124 [0.084, 0.178] |
| group ω² p10 / median / p90 | −0.053 / 0.049 / 0.614 | −0.046 / 0.067 / 0.609 |
| share of groups with ω² > 0 | 0.77 | 0.77 |
| ω² by depth bin 0–2 / 3–5 / 6+ | 0.039 / 0.055 / 0.205 | 0.049 / 0.077 / 0.259 |

### ⚠ Most of the pooled ω² is prefix identity

The brief's proxy test ("ω² meaningful, within-prefix contrast near zero") is not triggered in its
literal form, because the within-prefix standardized effect is −0.28. But the two estimators do
not agree in size. To check whether the bucket stands in for prefix identity, each z was demeaned
by its own edge's mean before the same ANOVA, which removes all between-prefix variation:

| | all steps | fresh steps |
|---|---|---|
| ω², raw | 0.084 | 0.123 |
| **ω², edge-demeaned** | **0.0066** [0.0054, 0.0113] | **0.0165** [0.0137, 0.0257] |
| edge-demeaned, by depth bin 0–2 / 3–5 / 6+ | 0.001 / 0.003 / 0.022 | 0.005 / 0.011 / 0.035 |

**About 90 % of the pooled ω² comes from which prefix, and which dialogue or persona, an edge
belongs to, not from affect variation within a prefix.**

Edge-demeaning with unbalanced buckets is somewhat conservative: the edge mean leans toward the
majority bucket. Even so, the within-edge share of variance explained by bucket is under 2 %. The
real within-prefix effect (§7.2) is small against the noise in z. The pooled ω² gate component, taken
alone, overstates what NodeKey's split would resolve.

## §7.4 Boundary fraction

| | all rows | depth ≥ 2 |
|---|---|---|
| \|parent_nu − τ_med\| < 0.10 | **0.221** | 0.225 |
| \|parent_nu − τ_med\| < 0.05 | 0.102 | 0.097 |

ν's IQR is 0.39, so ±0.1 is a wide band. The fraction is still low because ν is bimodal
(happiness-dominated realizations near +0.53, neutral-dominated near 0), and τ_med = 0.35 falls
between the modes (figure `nu_by_depth_D1.png`).

## §7.5 Confound: is the bucket a proxy for depth or turn?

**ν drifts strongly upward with depth.** Simulated users become uniformly happy deeper in the
tree:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 8,575 | 7,284 | 5,632 | 3,882 | 2,597 | 1,761 | 1,192 | 831 | 584 | 410 | 203 | 77 |
| mean ν | 0.237 | 0.245 | 0.282 | 0.331 | 0.348 | 0.389 | 0.403 | 0.413 | 0.439 | 0.452 | 0.477 | 0.495 |
| median | 0.235 | 0.236 | 0.317 | 0.407 | 0.435 | 0.470 | 0.487 | 0.501 | 0.523 | 0.529 | 0.531 | 0.532 |
| q25–q75 | 0.02–0.44 | 0.04–0.45 | 0.08–0.48 | 0.16–0.51 | 0.19–0.52 | 0.30–0.53 | 0.34–0.53 | 0.35–0.53 | 0.41–0.53 | 0.46–0.53 | 0.50–0.53 | 0.52–0.53 |
| bucket-1 occupancy | 0.65 | 0.61 | 0.52 | 0.42 | 0.37 | 0.29 | 0.27 | 0.25 | 0.18 | 0.17 | 0.14 | 0.09 |

Point-biserial correlation of `bucket_med`:

| with | r | 95 % CI |
|---|---|---|
| **depth** | **−0.290** | [−0.334, −0.245] |
| turn index | −0.131 | [−0.226, −0.029] |
| absolute position (turn + depth) | −0.298 | [−0.363, −0.221] |
| depth, rows at depth ≥ 2 only | −0.273 | — |

**The point estimate sits just inside the proposed |r| ≤ 0.3, and the CI straddles it.** A global
median does make shallow nodes mostly bucket 1 and deep nodes mostly bucket 0.

**Does the effect survive stratification?**

| re-run | Δ mean [CI] | std. effect | discordance | pooled ω² [CI] |
|---|---|---|---|---|
| global τ_med (reference) | −0.055 [−0.067, −0.043] | −0.275 | 0.308 | 0.084 [0.055, 0.127] |
| **depth-specific median split** (τ_d = median ν at depth d; bucket cannot encode depth) | **−0.046** [−0.059, −0.033] | −0.227 | 0.326 | **0.069** [0.047, 0.102] |
| same, fresh steps | −0.070 [−0.087, −0.053] | −0.349 | 0.298 | 0.103 [0.075, 0.140] |
| ω² with groups (absolute position, depth, action), global τ | — | — | — | 0.086 [0.064, 0.124] (253 groups) |

**The effect survives within depth, so it is not the depth confound.** The within-prefix contrast is
also immune by construction, since an edge fixes depth, turn and dialogue. Figures:
`figures/nu_by_depth_D1.png`, `figures/bucket_occupancy_by_depth_D1.png`.

## §7.6 Confound: is the affect channel already dominant?

Computed at 33,903 selection points, 12,956 of them with ≥ 2 expanded siblings. For the top two
siblings by full PUCT score:

| | median | p25 | p75 | p90 |
|---|---|---|---|---|
| \|ΔQ\| | 0.321 | 0.130 | 0.630 | 0.912 |
| \|β·ΔQ_emo\| | 0.088 | 0.039 | 0.157 | 0.219 |
| ratio \|β·ΔQ_emo\| / \|ΔQ\| | **0.271** [0.231, 0.317] | — | — | 1.64 (finite values) |

The ratio exceeds 1 in 18.7 % of cases, and ΔQ is exactly 0 in 3.3 %.

**Argmax flips** (the selection with β = 0.7 differs from the selection without the affect term):

| | all points | ≥ 2 expanded siblings, exploitation terms only (Q + βQ_emo vs Q) |
|---|---|---|
| flip fraction | **0.316** [0.282, 0.348] | **0.115** [0.088, 0.142] |

**What the flips are.** Of the full-PUCT flips:

- **82.8 %** swap an **unvisited** sibling (Q = Q_emo = 0) for an **already-visited** one;
- 17.1 % choose between two visited siblings;
- 0.1 % go the other way.

The mean Q_emo on visited edges is **+0.255** (ν is mostly positive, with τ_med = 0.35). So
β·Q_emo ≈ +0.18 acts as a flat bonus for anything already tried.

**The affect channel is not dominant as a task tiebreak-breaker:** the median ratio is 0.27, and
only about 5 % of all selections are the affect term choosing between two visited acts. **Its
main behavioural effect is suppressing exploration of unvisited acts.** An "emotion helps" claim
should be read as partly an exploration-schedule effect.

Flip fraction by depth: 0.39 (d1), 0.36 (d2), 0.32 (d3), 0.28 (d4), 0.26 (d5), 0.24 (d6), 0.21 (d8),
0.18 (d10). Figure: `figures/delta_q_vs_beta_delta_qemo_D1.png`. The counterfactual version of this
table for D2 is in §D2.

Note on the brief's premise: Q ∈ [−1, +1] here, not [0, 1], and with HF + soft weights
ν ∈ [−0.14, +0.54]. So the maximum possible |β·ΔQ_emo| is 0.48, and the observed p90 is 0.22.

## §7.7 NodeKey gate components (D1, median split). Not applied.

| component | proposed threshold | all steps | fresh steps |
|---|---|---|---|
| discordance rate (depth ≥ 2) | ≥ 0.30 | **0.308** [0.284, 0.338], at the threshold with CI straddling it | 0.279 [0.259, 0.306] |
| within-prefix mean Δ | CI excludes 0 and \|Δ\|/sd(z) ≥ 0.2 | **−0.055** [−0.067, −0.043]; std −0.275 | −0.079 [−0.095, −0.062]; std −0.395 |
| pooled ω² | ≥ 0.05, CI lower bound > 0 | **0.084** [0.055, 0.127] | 0.123 [0.084, 0.178] |
| *(diagnostic)* pooled ω², edge-demeaned | *(no threshold proposed)* | *0.0066* [0.0054, 0.0113] | *0.0165* [0.0137, 0.0257] |
| boundary fraction (±0.1) | ≤ 0.35 | **0.221** | — |
| bucket–depth correlation | \|r\| ≤ 0.3 | **−0.290** [−0.334, −0.245], at the threshold with CI straddling it | — |
| *(diagnostic)* post-split (edge, bucket) visits ≥ m = 3 | *(guard m = 3)* | *43 % of cells; median 2* | *35 %* |
| *(stop check)* discordant edges | ≥ ~200 | 1,447 | 1,313 |

Label split, same components: discordance **0.061**, Δ −0.042 [−0.069, −0.017] (std −0.210),
ω² **0.007** [0.005, 0.013], edge-demeaned ω² 0.002, |r| with depth 0.106. See §9.1.

---

## §8 AffPool (per tree)

**Scope.** Every cell is computed inside one `(dialogue_id, turn_index)` tree (175 trees), then
aggregated. `Q_pool` is per-search by construction (`definitions.md`).

**Definitions.**
- A cell is (bucket, action) within a tree, or (bucket, action, depth bin) in §8.3.
- Contributing prefixes are the distinct parent prefixes whose steps fall in the cell.
- `median_N_node` is the median total visit count N(s,a) of the contributing edges, i.e. the
  node's own statistic that the pool would supplement.
- An in-cell variant (edge visits inside that bucket) is also reported, because the brief's
  wording allows either reading.

### §8.1 Evidence gained (median split, cells keyed on bucket × action)

1,274 cells. 709 have ≥ 3 prefixes. 28.6 % have a single prefix (every depth-1 root edge is
single-prefix by construction).

| | p10 | p25 | **median** [CI] | p75 | p90 |
|---|---|---|---|---|---|
| n_prefixes | 1 | 1 | **3** [3, 3] | 7 | 14 |
| N_pool | 1 | 4 | 14 | 36 | 64 |
| **evidence multiplier** N_pool / median N(s,a) | 0.72 | 1.0 | **2.59** [2.0, 3.0] | 8.7 | 20.1 |
| evidence multiplier, in-cell variant | 1.0 | 1.0 | 4.0 | 13.9 | 28 |
| top-prefix share | 0.25 | 0.40 | **0.625** [0.60, 0.67] | 1.0 | 1.0 |
| Herfindahl | 0.15 | 0.27 | 0.50 | 1.0 | 1.0 |

**Within-tree distribution** (per-tree medians across the 175 trees):
- n_prefixes: p10 1.7, median 3, p90 5.
- evidence multiplier: p10 1.0, median 2.9, p90 6.5.

**By bucket:** bucket 0 has 596 cells (median multiplier 2.0); bucket 1 has 678 cells (3.0).

**Cells containing no depth-1 steps** (624 cells): median n_prefixes 2, multiplier 1.6.

The distribution is heavy-tailed. A minority of cells, around the popular acts, pool many prefixes;
the typical cell pools 3 prefixes, one of which supplies most of the visits. Figure:
`figures/prefix_sharing_D1.png`.

### §8.2 Homogeneity: pooling bias

Computed on the 709 cells with ≥ 3 prefixes: between-prefix variance of per-prefix mean z, over
pooled within-prefix variance of z, visit-weighted median across cells.

| | value [CI] |
|---|---|
| **between / within ratio** | **0.785** [0.694, 0.861] |
| noise-corrected (subtracting the within/nᵢ sampling floor from the variance of means) | 0.312 [0.248, 0.349] |

The raw ratio has a floor of about 1/n̄ even under perfect homogeneity, and with 2–3 visits per
prefix that floor is large. The noise-corrected figure is the variance-component estimate: between-prefix
variance in true means is about 30 % of within-prefix noise. Prefixes that share a bucket and an act
are **not strongly heterogeneous** in z. On this criterion pooling would not average across
genuinely different situations, although this measures homogeneity of *z* only, not of task value Q.

### §8.3 Does the pool key need depth?

Cells keyed (bucket, action, depth bin ∈ {0–2, 3–5, 6+}): 2,375 cells.

| | plain | depth-keyed |
|---|---|---|
| median n_prefixes | 3 | 2 |
| **median evidence multiplier** | 2.59 [2.0, 3.0] | **1.33** [1.2, 1.5] |
| median top-prefix share | 0.625 | 0.80 |
| **between / within ratio** | 0.785 [0.69, 0.86] | **0.583** [0.51, 0.65] |
| noise-corrected ratio | 0.312 | 0.204 |
| by depth bin 0–2 / 3–5 / 6+: multiplier | — | 1.0 / 1.75 / 2.0 |
| by depth bin 0–2 / 3–5 / 6+: ratio | — | 0.54 / 0.66 / 0.45 |

Adding depth reduces heterogeneity by about 25 %. It also halves the evidence multiplier to
1.3, well below the ~3 that §8.3 requires. **The depth-keyed variant is not recommended.** The
plain key is itself below 3.

### §8.4 AffPool gate components (D1, median split, plain key). Not applied.

| component | proposed threshold | measured |
|---|---|---|
| median n_prefixes per cell | ≥ 4 | **3** [3, 3] |
| median evidence multiplier | ≥ 3 | **2.59** [2.0, 3.0] (in-cell variant 4.0) |
| top-prefix share | ≤ 0.6 | **0.625** [0.60, 0.67] |
| between / within variance ratio | ≤ 1.0 | **0.785** [0.69, 0.86] (noise-corrected 0.31) |

Label split: median n_prefixes 2, multiplier 2.0, top share 0.71, ratio 0.77. In label-bucket-1
cells the median n_prefixes is 1 and the multiplier 0.73.

---

## §9.1 Median vs label split

| | median split | label split |
|---|---|---|
| bucket-1 occupancy, overall | **0.500** | **0.056** |
| bucket-1 occupancy, depth ≥ 2 | 0.449 | 0.048 |
| bucket-1 occupancy by depth 1 / 2 / 3 / 5 / 8 / 10 | 0.65 / 0.61 / 0.52 / 0.37 / 0.25 / 0.17 | 0.080 / 0.094 / 0.050 / 0.023 / 0.008 / 0.005 |
| discordance rate (depth ≥ 2) | 0.308 | 0.061 |
| discordant edges | 1,447 | 287 |
| within-prefix Δ [CI], std | −0.055 [−0.067, −0.043], −0.275 | −0.042 [−0.069, −0.017], −0.210 |
| pooled ω² [CI] | 0.084 [0.055, 0.127] | 0.007 [0.005, 0.013] |
| edge-demeaned ω² | 0.0066 | 0.0018 |
| bucket–depth r | −0.290 | −0.106 |
| AffPool median multiplier | 2.59 | 2.0 |

**Agreement between the two definitions** (all rows): κ = **0.112** (depth ≥ 2: 0.117).

| | label = not negative | label = negative |
|---|---|---|
| median bucket 0 (ν ≥ τ) | 16,953 | **0** |
| median bucket 1 (ν < τ) | 15,046 | 1,904 |

Every label-negative realization is below the median, but 89 % of below-median realizations are
not label-negative. They are mostly neutral-dominated: the valence weight for neutral is −0.07,
against happiness +0.54.

**The occupancy gap is the quantitative form of the "affectively flat simulator" criticism:**

- Only 5.6 % of sampled parent realizations carry a negative argmax label (fear, sadness, anger or
  disgust).
- Among the 205 real (observed) user turns, it is 8.8 % (fear 11, sadness 5, anger 2).
- The median split puts half the visits in bucket 1 only by treating "neutral rather than happy" as
  the affective contrast.

A NodeKey or AffPool split on the median is therefore a **happy-vs-neutral** split, not a
**negative-vs-non-negative** one. The label split, which carries the negative-affect hypothesis,
has negative-bucket cells too sparse to carry either feature (discordance 6 %, a median of one
prefix per negative cell).

## §9.2 σ_emo re-measurement (HF classifier, D1)

Computed over 3,432 edges with N ≥ 2 at the end of their tree, from the per-edge z tape. This is
identical to `sqrt(M2_emo/(N−1))`, as verified in the alignment check.

| | value |
|---|---|
| **ratio σ_emo / \|Q_emo\|, median** | **0.449** |
| p25 / p75 / **p90** | 0.122 / 0.870 / **1.447** |
| σ_emo median | 0.131 |
| \|Q_emo\| median | 0.342 |
| edges excluded for \|Q_emo\| = 0 | 0 |

Reliability of these σ estimates:

- **N behind them:** median N = **6** (p25 3, p75 12, p90 23). 22.0 % of edges have N = 2, and
  34.9 % have N ≤ 3.
- **By N:**

  | N | edges | σ median | ratio median |
  |---|---|---|---|
  | 2 | 754 | 0.077 | 0.357 |
  | 3 | 443 | 0.124 | 0.469 |
  | 4–5 | 515 | 0.133 | 0.426 |
  | 6–10 | 730 | 0.131 | 0.403 |
  | > 10 | 990 | 0.153 | 0.521 |

  The N = 2 edges show deflated σ (0.077 against about 0.13 at N ≥ 3), as expected from 2-sample
  estimates. Excluding them would raise the median ratio slightly, not lower it.

**The earlier 1.43 median / 9.10 p90 figures are superseded.** They came from a deterministic
stub classifier with evidence wiring, not from affect, and must not be cited. On real HF-labelled
affect the within-edge spread is about 0.45× the edge mean, not about 1.4×.

---

## D2 (β = 0 control)

**Same config, same 30 dialogues, β_emo = 0.** `Q_emo` and `M2_emo` still accumulate (the backup is
unconditional) but do not enter selection.

- Run: SR 0.733, 6.90 turns on average. 10,018 s = **2.78 GPU-h**.
- Alignment check: 30/30 PASS (`runs/D2/alignment_check.txt`).
- `D2/steps.parquet`: 28,129 rows · 177 trees · 2,950 prefixes · 5,312 edges · 15,244 fresh.
- τ_med (D2's own median) = **+0.313**.

**Reported beside D1, not averaged.**

### Is the bucket effect an artifact of the affect channel steering the tree?

**No.** The within-prefix effect, pooled ω² and its prefix-identity share all reproduce almost
exactly without the affect term:

| median split, depth ≥ 2 | D1 (β 0.7) | D2 (β 0) |
|---|---|---|
| discordance rate | 0.308 [0.284, 0.338] | **0.249** [0.220, 0.282] |
| discordant edges | 1,447 / 4,705 | 1,137 / 4,574 |
| within-prefix Δ | −0.055 [−0.067, −0.043] | **−0.055** [−0.069, −0.038] |
| standardized effect | −0.275 [−0.33, −0.22] | **−0.268** [−0.34, −0.19] |
| Δ, fresh steps | −0.079 [−0.095, −0.062]; std −0.395 | −0.073 [−0.091, −0.053]; std −0.359 |
| pooled ω² | 0.084 [0.055, 0.127] | **0.083** [0.063, 0.102] |
| pooled ω², fresh | 0.123 [0.084, 0.178] | 0.106 [0.081, 0.128] |
| **edge-demeaned ω²** | 0.0066 [0.0054, 0.0113] | **0.0077** [0.0042, 0.0145] |
| edge-demeaned ω², fresh | 0.0165 [0.0137, 0.0257] | 0.0169 [0.0097, 0.0293] |
| ω² by depth bin 0–2 / 3–5 / 6+ | 0.039 / 0.055 / 0.205 | 0.050 / 0.074 / 0.228 |
| post-split (edge, bucket) cells ≥ 3 visits | 43 % (median 2) | 36 % (median 2) |
| boundary fraction ±0.1 / ±0.05 | 0.221 / 0.102 | 0.214 / 0.095 |
| r(bucket, depth) | −0.290 [−0.334, −0.245] | −0.240 [−0.282, −0.190] |
| r(bucket, turn) | −0.131 [−0.226, −0.029] | −0.143 [−0.225, −0.057] |
| r(bucket, absolute position) | −0.298 [−0.363, −0.221] | −0.258 [−0.331, −0.169] |
| depth-specific median split: Δ / std / ω² | −0.046 / −0.227 / 0.069 | −0.043 / −0.210 / 0.070 [0.055, 0.085] |
| ω² with absolute-position strata | 0.086 | 0.077 [0.065, 0.105] |
| median parent ν at depth 1 / 2 / 3 / 5 / 8 / 10 | 0.24 / 0.24 / 0.32 / 0.44 / 0.50 / 0.53 | 0.25 / 0.23 / 0.33 / 0.48 / 0.53 / 0.53 |

**Where D2 differs is discordance: 0.249, below the 0.30 proposal.** D2's trees are broader and
shallower:

- 30.8 % of visits at depth 1 (D1: 25.3 %);
- median parent-node visits at depth 2 / 3 are 7.5 / 4 (D1: 9.5 / 5);
- maximum depth 24 (D1: 33).

So edges get fewer visits and fewer of them see both buckets. D1's discordance of 0.31 is partly a
product of β = 0.7 concentrating visits (§7.6). The premise that the emotional outcome at an edge
varies with the parent's bucket holds equally with and without the channel. How often an edge
gets to show it depends on β.

### Visit profile (D2)

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| visits | 8,673 | 7,304 | 5,175 | 3,011 | 1,597 | 855 | 502 | 305 |
| edges | 738 | 1,309 | 1,234 | 831 | 498 | 282 | 146 | 92 |
| median visits per parent node | 49 | 7.5 | 4 | 3 | 3 | 3 | 2 | 3 |

### AffPool (D2, per tree)

| component | threshold | D1 | D2 |
|---|---|---|---|
| median n_prefixes | ≥ 4 | 3 [3, 3] | **3** [3, 3] |
| median evidence multiplier | ≥ 3 | 2.59 [2.0, 3.0] | **2.50** [2.0, 3.0] (in-cell 3.9) |
| top-prefix share | ≤ 0.6 | 0.625 [0.60, 0.67] | **0.643** [0.60, 0.67] |
| between / within ratio | ≤ 1.0 | 0.785 [0.69, 0.86] | **0.783** [0.73, 0.86] (noise-corrected 0.29) |
| depth-keyed: multiplier / ratio | — | 1.33 / 0.58 | 1.25 / 0.62 |

The AffPool numbers are insensitive to β.

### Splits and σ (D2)

- **Occupancy:** median split 0.499 (depth ≥ 2: 0.457); label split **0.067** (depth ≥ 2: 0.057).
  κ between them 0.135.
- **Label split:** discordance 0.054, Δ −0.035 [−0.059, −0.010] (std −0.17), ω² 0.012 [0.007, 0.024].
- **Negative share among the 207 real user turns:** 9.7 % (D1: 8.8 %).
- **σ_emo / |Q_emo|:** median **0.501**, p25 0.161, p75 0.921, p90 **1.464**. Median N = 5, 21.5 %
  of edges at N = 2, 35.1 % at N ≤ 3, over 2,892 edges. By N: N = 2 gives 0.40, N = 3 0.50,
  N 4–5 0.56, N 6–10 0.45, N > 10 0.55.

### §7.6 counterfactual: how much would β = 0.7 move D2's selections?

Computed on D2's logged sibling states, as if β were 0.7. The logged `uct` has no affect term
because β = 0.

| | D1 (actual, β 0.7) | D2 (counterfactual β 0.7) |
|---|---|---|
| points with ≥ 2 expanded siblings | 12,956 / 33,903 (**38 %**) | 14,491 / 28,129 (**52 %**) |
| median \|ΔQ\| / \|β·ΔQ_emo\| | 0.321 / 0.088 | 0.340 / 0.079 |
| median ratio | 0.271 [0.231, 0.317] | 0.246 [0.210, 0.281] |
| ratio > 1 | 18.7 % | 16.4 % |
| **argmax flip, full PUCT** | **0.316** [0.282, 0.348] | **0.155** [0.126, 0.188] |
| argmax flip, expanded siblings, exploitation terms | 0.115 [0.088, 0.142] | 0.117 [0.086, 0.151] |
| flips choosing visited over unvisited | 82.8 % | 42.5 % |
| flips choosing between two visited siblings | 17.1 % | 57.4 % |
| flip by depth 1 / 2 / 3 / 5 / 8 / 10 | 0.39 / 0.36 / 0.32 / 0.26 / 0.21 / 0.18 | 0.23 / 0.17 / 0.13 / 0.07 / 0.03 / 0.01 |

On trees the affect term did not shape, it would flip 15.5 % of selections, and most of those
flips are between two visited acts. The flip rate among expanded siblings is the same in both runs
(11.5–11.7 %). That is the affect term's discriminating influence, and it does not depend on the
tree it runs on.

The extra flips in D1 (31.6 %) are self-reinforcing. The visited-edge bonus leaves fewer siblings
expanded: 38 % of D1 selection points have ≥ 2 expanded siblings, against 52 % in D2. That creates
more occasions where a visited edge beats an unvisited one. **This confirms the §7.6 reading: at
β = 0.7 the channel's dominant behavioural effect is narrower search, not affect-driven choice among
alternatives.**

Figures: `figures/*_D2.png`. `delta_q_vs_beta_delta_qemo_D2.png` is the counterfactual β = 0.7
version.
