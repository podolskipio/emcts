# §5.2–5.4 Bucket selection, redundancy, cross-turn drift

**Recommendations:** keep **K0**; keep AffPool **per-search** scope. No alternative key clears the
switching rule in both runs; the closest is K4 (depth-adjusted level). Cross-turn pooling would add 2–5×
more heterogeneity than within-search pooling, and the direction key does not drift less.

All five keys are compared on the same rows, the steps where Δν exists (D1 32,433 of 33,903; D2 26,659 of
28,129; the dropped rows are depth-1 steps of the first planned turn). Δν is the parent realization's
own previous user turn (`constants.json` definitions). 7.4 % (D1) / 3.4 % (D2) of Δν are exactly 0
(verbatim repeated user lines); K3 splits strictly (`Δν < median`), the same convention as K0.
Homogeneity is on **backup_value**, the quantity AffPool pools (§5.1), with z alongside for continuity
with 0.31. 95 % CIs: cluster bootstrap over dialogues, 1000 replicates.

## §5.2 Five keys, two buckets each

### D1 (β 0.7)

| key | homogeneity raw (task return) | noise-corrected | same on z (nc) | multiplier | r(bucket, depth) | r(bucket, turn) | κ with K0 |
|---|---|---|---|---|---|---|---|
| **K0** ν < τ_med | 0.502 [0.445, 0.569] | 0.046 [0.005, 0.116] | 0.288 | 2.40 [2.0, 3.0] | −0.289 [−0.335, −0.244] | −0.119 | 1 |
| K1 φ₀.₈ | 0.480 [0.430, 0.571] | 0.039 [0.013, 0.100] | 0.274 | **2.00** | −0.325 [−0.399, −0.250] | −0.104 | 0.62 |
| K2 φ₀.₅ | 0.497 [0.433, 0.576] | 0.039 [−0.006, 0.100] | 0.244 | **2.00** | −0.323 [−0.377, −0.267] | −0.121 | 0.79 |
| K3 Δν | 0.504 [0.458, 0.587] | 0.052 [−0.005, 0.115] | 0.330 | 2.40 | **−0.108** [−0.165, −0.039] | −0.057 | 0.41 |
| K4 ν − med(ν \| depth) < 0 | 0.486 [0.434, 0.570] | 0.046 [−0.002, 0.116] | 0.229 | 2.13 | **−0.028** [−0.081, +0.051] | −0.133 | 0.76 |

### D2 (β 0)

| key | homogeneity raw (task return) | noise-corrected | same on z (nc) | multiplier | r(bucket, depth) | r(bucket, turn) | κ with K0 |
|---|---|---|---|---|---|---|---|
| **K0** | 0.646 [0.569, 0.756] | 0.126 [0.045, 0.264] | 0.258 | 2.33 [2.0, 3.0] | −0.251 [−0.290, −0.203] | −0.153 | 1 |
| K1 | 0.620 [0.562, 0.760] | 0.114 [0.035, 0.196] | 0.288 | **2.00** | −0.250 | −0.131 | 0.56 |
| K2 | 0.650 [0.562, 0.820] | 0.113 [0.018, 0.244] | 0.256 | 2.25 | −0.272 | −0.121 | 0.78 |
| K3 | 0.670 [0.621, 0.752] | 0.144 [0.081, 0.251] | 0.263 | 2.50 | **+0.065** [−0.005, +0.128] | +0.034 | 0.36 |
| K4 | 0.648 [0.584, 0.716] | 0.114 [0.057, 0.198] | 0.212 | 2.40 | **−0.020** [−0.064, +0.039] | −0.142 | 0.80 |

Median contributing prefixes per cell is 3 for every key in both runs; top-prefix share 0.63–0.67.
On all rows K0 reproduces yesterday exactly: multiplier 2.59, z 0.785 / 0.312.

**Occupancy of bucket 1, by depth 1 / 2 / 3 / 4–5 / 6+ (D1):** K0 0.65 / 0.61 / 0.52 / 0.40 / 0.23 ·
K1 0.63 / 0.64 / 0.58 / 0.45 / 0.17 · K2 0.67 / 0.62 / 0.57 / 0.42 / 0.19 · **K3 0.48 / 0.47 / 0.38 / 0.41 / 0.36** ·
**K4 0.54 / 0.50 / 0.50 / 0.50 / 0.49**. D2 has the same shape (K3 0.50 / 0.52 / 0.43 / 0.47 / 0.63; K4 flat at 0.49–0.53).

### Against the rule

Switch only if a key beats K0 on homogeneity (raw and noise-corrected) **and** keeps its multiplier within
10 % **and** has |r(bucket, depth)| no worse.

| key | D1 | D2 |
|---|---|---|
| K1, K2 (EWMA) | fail: multiplier −17 %, depth r worse | fail: multiplier (K1), depth r (K2) |
| K3 (Δν) | fail: homogeneity slightly worse | fail: homogeneity worse |
| K4 (depth-adjusted level) | fail: multiplier −11.5 % (tolerance 10 %); homogeneity tied | fail on raw homogeneity by 0.002; passes nc, multiplier, depth |

**Recommend K0.** Every homogeneity difference sits well inside overlapping CIs; nothing here is a
material gain. **K4 is the one worth naming:** it matches K0's homogeneity and all but removes the phase
confound (r(bucket, depth) −0.03 / −0.02 against −0.29 / −0.25), at a 0–12 % multiplier cost. If the
depth-proxy objection matters more for the paper than a few percent of evidence, K4 is the defensible
alternative. It needs an online per-depth median, which does not exist in code and would be a build.
**K3 does what its motivation claims on phase** (|r| 0.11 / 0.07) but pools no more homogeneously, and §4 gives it no
momentum motivation.

## §5.3 Redundancy: does affect history predict the next reaction beyond the current state?

OLS of z on the parent's affect, rows with two prior user turns (D1 n = 29,603; D2 n = 23,842).

| model | D1 R² | D2 R² |
|---|---|---|
| ν_t | 0.101 [0.063, 0.147] | 0.077 [0.054, 0.094] |
| ν_t, ν_{t−1}, ν_{t−2} | 0.156 [0.104, 0.217] | 0.093 [0.065, 0.114] |
| ν_t, φ₀.₈ | 0.157 [0.102, 0.221] | 0.085 [0.056, 0.111] |
| + depth (on the history model) | 0.165 | 0.101 |
| depth alone | 0.064 | 0.035 |

| increment | D1 | D2 |
|---|---|---|
| **history over current** | **+0.055** [0.034, 0.075] (+54 % relative) | **+0.016** [0.008, 0.027] (+21 %) |
| φ₀.₈ over current | +0.056 [0.033, 0.079] | +0.008 [0.001, 0.023] |
| depth over history | +0.009 [0.005, 0.014] | +0.009 [0.003, 0.017] |

Fresh steps only (causal pairs): history adds +0.054 (D1) / +0.017 (D2) over R² 0.142 / 0.097.

**For §5 of the paper:** "In LLM user simulators, the user's emotional history adds 5.5 points of R²
(+54 %) over the current state in predicting the next reaction with the affect channel on, and 1.6 points
(+21 %) with it off; dialogue depth adds under 1 point beyond history." The β dependence is itself a
finding: with the affect channel steering the tree (D1), affect is more autocorrelated along the paths
it selects.

## §5.4 Cross-turn drift

Per (dialogue, bucket, act) cell pooled across that dialogue's turns (≥ 3 turns), the decomposition of
§5.2 with **turn** as the grouping factor. Median over cells, visit-weighted.

| key | quantity | D1 between/within | D1 noise-corr. | D2 between/within | D2 noise-corr. | median range of turn means |
|---|---|---|---|---|---|---|
| K0 | task return | 0.309 [0.232, 0.358] | **0.256** [0.203, 0.300] | 0.313 [0.206, 0.398] | **0.241** [0.151, 0.328] | 0.68 / 0.72 |
| K3 | task return | 0.268 [0.231, 0.342] | **0.229** [0.175, 0.328] | 0.310 [0.227, 0.439] | **0.234** [0.166, 0.336] | 0.69 / 0.78 |
| K0 | z | 0.214 | 0.140 [0.077, 0.207] | 0.215 | 0.141 [0.102, 0.169] | 0.23 / 0.23 |
| K3 | z | 0.217 | 0.157 [0.109, 0.236] | 0.290 | 0.173 [0.131, 0.281] | 0.25 / 0.25 |

- **Drift is large where it matters.** On task return the between-turn component (noise-corrected ≈
  0.24–0.26) is **5.6× (D1) and 1.9× (D2)** the within-search between-prefix component under the same key
  (0.046 / 0.126). Within one dialogue, a cell's mean task return moves by a median 0.68–0.78 across turns.
  `Q(propose)` at turn 1 is not `Q(propose)` at turn 6.
- **K3 does not drift materially less** (0.229 vs 0.256; 0.234 vs 0.241; CIs overlap almost entirely),
  and on z it drifts slightly more. Cross-turn scope is not defensible for the direction key either.
- **Evidence gain is 3.2× (D1) and 3.9× (D2), not ~6×:** median N_pool per (bucket, act) cell rises from 13 / 11
  per search to 41.5 / 43 per dialogue.

**Recommend per-search scope.** The number that justifies it for §3 of the paper: cross-turn pooling
would add between-turn heterogeneity of 0.24–0.26 (noise-corrected, task return), against 0.05–0.13
within a search, for a 3–4× evidence gain.

Machine-readable: `s5_bucket_selection.json`. Script: `scripts/s5_bucket_selection.py`.
