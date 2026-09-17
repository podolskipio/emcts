# §7 Key geometry, Part 1

**Recommendation: keep the scalar ν and the hard partition; do not build §9 (soft-ν) now.**

- **The 7-vector key (B) and the transition keys (C) are not warranted.**
  - The emotion distribution is effectively 1.5-dimensional.
  - It carries a small but nonzero amount of information about the next reaction beyond ν.
  - No distance on it predicts outcome differences better than |Δν|.
- **The scalar RBF kernel does pass §7.4's sharing and homogeneity checks.** But §9's first precondition
  ("7.1 shows the scalar is sufficient") is not met cleanly. And the unkeyed-pool reference shows the
  *affective* key itself is not what makes pooling homogeneous on task return. A better-shaped affective key
  is solving a problem the data do not show.

95 % CIs: cluster bootstrap over dialogues, 1000 replicates. D1 and D2 are reported separately.

## §7.1 Does the distribution carry information beyond the scalar? (decisive)

OLS of the outcome on `parent_nu`, then on the 7-vector (6 columns, neutral as reference), each ± depth.

| run · outcome | R²(ν) | R²(d) | **increment d over ν** | increment with depth | corr(fitted d coefs, w) |
|---|---|---|---|---|---|
| D1 · z | 0.101 [0.067, 0.141] | 0.110 | **+0.009** [0.005, 0.015] | +0.006 [0.004, 0.013] | 0.25 |
| D2 · z | 0.076 [0.052, 0.096] | 0.081 | **+0.005** [0.002, 0.013] | +0.003 [0.001, 0.010] | 0.32 |
| D1 · z, fresh | 0.145 | 0.150 | +0.005 [0.003, 0.012] | +0.004 | 0.44 |
| D1 · task return | **0.001** [0.000, 0.010] | 0.010 | +0.009 [0.007, 0.016] | +0.009 | 0.11 |
| D2 · task return | **0.001** [0.000, 0.004] | 0.015 | +0.014 [0.008, 0.027] | +0.015 | 0.09 |

Fitted coefficients against neutral, D1 · z: happiness +0.18, surprise +0.15, sadness +0.05, fear +0.02,
anger −0.10, **disgust −0.35**. Compare w − w(neutral): happiness +0.61, fear +0.31, disgust +0.20,
anger +0.18, surprise +0.16, sadness −0.07. The regression is **not** recovering w (r = 0.25–0.32). Its
largest weight is on disgust with the opposite sign from the mined table.

**Reading.** The increment on z is real (the CI excludes 0) but small: under 1 point of R², 5–9 % of what ν explains. It
is not "near zero" in the brief's statistical sense, and not large enough to carry a key. **On task return, the
scalar ν explains essentially nothing (R² 0.001)**, and the vector explains ~1–1.5 %. For the paper: *the
task-calibrated projection ν carries most of the distribution's information about the next simulated reaction
(R² 0.10 vs 0.11), but almost none about the backed-up task value.*

## §7.2 Effective dimensionality and neighbourhood density

Distinct parent realizations per tree (D1 8,560; D2 6,930).

| | D1 | D2 |
|---|---|---|
| variance explained, PC1 / PC2 / PC3 | 0.816 / 0.108 / 0.041 | 0.804 / 0.121 / 0.043 |
| participation ratio (effective rank) | **1.47** | **1.51** |
| max_e d(e): median / p25 / p10 | 0.86 / 0.68 / 0.54 | 0.86 / 0.69 / 0.54 |
| within-tree neighbours, cosine ≤ 0.1: median (share < 5) | 24 (4.0 %) | 17 (8.0 %) |
| within-tree neighbours, cosine ≤ 0.2: median (share < 5) | 31 (2.5 %) | 20.5 (5.5 %) |

200 random parent vectors; neighbours are other distinct realizations in the same tree. **The §11
trigger (median < 5 at 0.2) does not fire:** d-space neighbourhoods are populated. They are populated
*because* the distribution is ~1.5-dimensional, a happiness–neutral axis (PC1 82 %). A vector key would
mostly reproduce the scalar one.

## §7.3 Distance-metric sanity

Run because 7.1's z increment has a CI lower bound above 0.005 in D1, computed for D2 too. Correlation between pairwise key
distance and pairwise |Δz|, ~199k within-tree pairs per run:

| metric | D1 | D2 |
|---|---|---|
| Euclidean on d | 0.119 | 0.105 |
| cosine on d | 0.107 | 0.095 |
| w-weighted \|ν_i − ν_j\| | 0.111 | 0.096 |

**No metric wins.** The full-vector distances are no better similarity functions than the scalar's. The
honest conclusion is the brief's: the scalar was the right object.

## §7.4 Kernel viability on the scalar

Statistics per query step within its (tree, act) group. The hard K0 partition goes through the same
code with weights 1[same bucket], and reproduces the cell-level homogeneity exactly (0.501 / 0.046 task
return; 0.785 / 0.312 z). Per-query multipliers aggregate differently from yesterday's cell-level 2.59, so
compare kernel against hard in each row, not against 2.59. The pool-depth column is r(query depth, the
pool's weighted mean depth): higher means the pool stays within its own phase, i.e. the key acts more as a
phase proxy (the concern behind yesterday's −0.29). Lower is better on that criterion.

### D1

| pool | N_pool | multiplier | effective prefixes | homogeneity nc (task return) | homogeneity nc (z) | r(depth, pool depth) |
|---|---|---|---|---|---|---|
| hard K0 | 53 | 3.66 [2.87, 4.69] | 4.3 | 0.046 [0.005, 0.116] | 0.312 [0.235, 0.349] | 0.681 |
| RBF h = 0.05 | 26 | 1.43 | 2.7 | 0.010 [−0.026, 0.035] | 0.200 | 0.740 |
| RBF h = 0.10 | 37 | 2.09 | 3.5 | 0.025 [−0.003, 0.042] | 0.216 | 0.714 |
| RBF h = 0.20 | 47 | 3.27 [2.51, 4.34] | 4.4 | 0.039 [0.010, 0.069] | 0.243 [0.212, 0.283] | 0.690 |
| RBF h = 0.40 | 68 | 4.80 [3.67, 6.38] | 5.5 | 0.045 [0.018, 0.080] | 0.254 [0.213, 0.299] | 0.662 |
| **unkeyed (act only)** | **97** | **6.67** [5.13, 8.84] | 6.2 | **0.073** [0.004, 0.115] | 0.323 [0.277, 0.393] | 0.624 |

### D2

| pool | N_pool | multiplier | effective prefixes | homogeneity nc (task return) | homogeneity nc (z) | r(depth, pool depth) |
|---|---|---|---|---|---|---|
| hard K0 | 38 | 2.33 [1.93, 2.78] | 2.9 | 0.139 [0.049, 0.278] | 0.286 [0.221, 0.382] | 0.680 |
| RBF h = 0.10 | 25 | 1.49 | 2.4 | 0.022 | 0.170 | 0.724 |
| RBF h = 0.20 | 34 | 2.11 [1.75, 2.51] | 3.0 | 0.041 [−0.013, 0.089] | 0.178 [0.131, 0.231] | 0.692 |
| RBF h = 0.40 | 45 | 3.01 [2.54, 3.66] | 3.7 | 0.045 [−0.003, 0.090] | 0.194 [0.150, 0.284] | 0.655 |
| **unkeyed (act only)** | **60** | **4.25** [3.33, 5.48] | 4.0 | **0.113** [0.041, 0.269] | 0.326 [0.226, 0.440] | 0.601 |

Boundary motivation (22 % of realizations within ±0.1 of τ) holds up. At h = 0.2 the kernel keeps
89–90 % of hard's evidence with better homogeneity on both quantities; at h = 0.4 it gives more evidence
than hard and is no less homogeneous. **By the §7.4 criteria the kernel is viable.**

**But the unkeyed row is the reference that decides the recommendation.** On task return it is as
homogeneous as any keyed pool, within overlapping CIs, with 1.8–2.6× hard's evidence. The affective
key, hard or soft, measurably improves homogeneity only on z, which AffPool does not pool. A softer affective key
optimizes the part of the design that the data say is not load-bearing.

## §7.5 Two-dimensional kernel on (ν, Δν)

h_Δ = h · sd(Δν) / sd(ν). Rows with Δν.

| run | pool | multiplier | homogeneity nc (task return) | nc (z) | r(depth, pool depth) |
|---|---|---|---|---|---|
| D1 | hard K0 (same rows) | 3.78 | 0.046 | 0.288 | 0.673 |
| D1 | 2-d h = 0.2 | **1.84** | 0.026 | 0.227 | 0.740 |
| D1 | 2-d h = 0.4 | 3.66 | 0.036 | 0.231 | 0.680 |
| D2 | hard K0 (same rows) | 2.43 | 0.126 | 0.258 | 0.676 |
| D2 | 2-d h = 0.2 | **1.39** | 0.012 | 0.164 | 0.762 |
| D2 | 2-d h = 0.4 | 2.37 | 0.033 | 0.169 | 0.688 |

Adding Δν as a second dimension roughly halves sharing at a given h and buys no homogeneity over the 1-d kernel
at matched evidence. **Closed.**

## §9 preconditions (soft-ν build): not all met, so not built

| precondition | status |
|---|---|
| 7.1 shows the scalar is sufficient | **Not cleanly.** The d-over-ν increment on z has a CI excluding 0 (small), and on task return ν carries almost nothing |
| 7.4 gives an h with N_pool within ~20 % of hard and homogeneity no worse than 0.31 | met (h = 0.2 or 0.4) |
| builds done | met |

Per the brief's rule ("if any fails, stop and leave the hard partition"), `--aff_pool_kernel` was not
built and no soft-vs-hard pilot was added to Thursday.

Machine-readable: `s7_key_geometry.json`, `s7_unkeyed_reference.json`. Scripts: `scripts/s7_key_geometry.py`, `scripts/s7_unkeyed_reference.py`.
