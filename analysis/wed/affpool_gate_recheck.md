# §5.1 AffPool homogeneity gate, recomputed on the task return — read this first

**Result: no escalation.** On the quantity AffPool actually pools, the task return `backup_value`, the
between/within ratio is **lower** than yesterday's z-based number, not above 1.0.

| run | pooled quantity | ratio (visit-weighted median) | noise-corrected | evidence multiplier |
|---|---|---|---|---|
| D1 (β 0.7) | z (yesterday) | 0.785 [0.694, 0.861] | 0.312 [0.235, 0.349] | 2.59 [2.0, 3.0] |
| D1 | **backup_value** | **0.501** [0.445, 0.576] | **0.046** [0.005, 0.116] | 2.59 |
| D2 (β 0) | z | 0.783 [0.733, 0.862] | 0.286 [0.221, 0.382] | 2.50 |
| D2 | **backup_value** | **0.646** [0.571, 0.775] | **0.139** [0.049, 0.278] | 2.50 |

95 % CIs: cluster bootstrap over dialogues, 1000 replicates. Cells are (tree, K0 bucket, act) with ≥ 3
contributing prefixes (D1 709, D2 798). The z rows reproduce yesterday's `p_var.md` §8.2 to the third
decimal, and the script asserts it, so the only thing that differs between the rows is the pooled column.

## What was confirmed in code

`Q_pool` carries the **task return**: the same `v` backed up into `Q` (`FREEZE_NOTES.md` §8.2). It
must, since `Q_eff` mixes it with `Q`. D1 and D2 predate the `backup_value` name, but their simlog `v` is
the same variable at the same call site. Replaying it reproduces every logged `(N, Q)`, with 0 mismatches over
62,032 steps. **So the gate was measured on the wrong quantity, and on the right one it holds.**

## Read it with these caveats

1. **Homogeneous relative to noise, not in absolute terms.** Per visit, the task return is ~7× noisier
   than z: median within-prefix variance 0.187 against 0.026. Its between-prefix variance is ~5×
   larger: 0.107 against 0.021, an SD of ≈ 0.33 on a [−1, +1] scale. Prefixes sharing a (bucket, act)
   cell do differ in task value. The difference is small next to the variation between single visits,
   which is what the RAVE blend trades against. 32 % (D1) / 36 % (D2) of cells still have a raw ratio > 1.
2. **Fresh steps only** (child generated from this parent, not served from the cache): raw ratio
   0.859 [0.742, 0.967] (D1) and 0.915 [0.776, **1.114**] (D2); noise-corrected 0.27 / 0.30. The D2
   upper bound crosses 1.0. All steps is the operative population, because it is what the pool averages under the
   current cache. On causal pairs alone the margin is thin.
3. **Reachable rows only** (turn + depth ≤ Tmax; `SEARCH_HORIZON_BUG.md`): 0.498 / 0.083 (D1),
   0.607 / 0.114 (D2); multiplier drops to 2.0. Direction unchanged.

## A stronger caveat, from §7.4: the affective key buys no homogeneity on task return

Run through the same per-query estimator, the **unkeyed** pool (every step under the same act in the
tree, no bucket) is as homogeneous on `backup_value` as the K0-keyed pool, with ~2× the evidence
(`key_geometry.md` §7.4, `s7_unkeyed_reference.json`):

| | K0-keyed: noise-corr. / multiplier | unkeyed: noise-corr. / multiplier |
|---|---|---|
| D1 | 0.046 [0.005, 0.116] / 3.66 | 0.073 [0.004, 0.115] / 6.67 |
| D2 | 0.139 [0.049, 0.278] / 2.33 | 0.113 [0.041, 0.269] / 4.25 |

On z the key helps slightly: D1 0.312 vs 0.323, D2 0.286 vs 0.326. The gate passes, but pooling is
safe *regardless* of the affective key, and the key halves the evidence. That bears on the "affective"
claim of AffPool, not on whether it runs. **Not escalated under §11's trigger (the gate did not exceed 1.0),
but the freeze decision should see it.**

Machine-readable: `s51_affpool_gate_recheck.json`. Script: `scripts/s51_affpool_gate_recheck.py`.
