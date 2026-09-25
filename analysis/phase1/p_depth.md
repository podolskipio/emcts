# P-DEPTH — affect statistics split by depth, and what the realization cache does to them

> Same logs as P-VAR, no GPU. Script `scripts/p_depth.py`, numbers `D1/p_depth_D1.json`,
> `D2/p_depth_D2.json`. Estimators are imported from `p_var.py` unchanged, so every figure is
> directly comparable to `p_var.md`.
>
> ⚠ **Inherits P-VAR's banner.** These are **legacy-horizon** trees under the **soft** valence table
> with τ_med = 0.350 — not the frozen `generic` table at τ 0.263. Read this as evidence about the
> *premise*, not about the shipped AffPool arm. Every headline is reported twice: on all rows and on
> the **reachable** window (`turn_index + depth ≤ Tmax`), which is the correction
> `SEARCH_HORIZON_BUG.md` requires.

## Verdict

**The depth split cannot run the test, for a structural reason — and the test it was a proxy for
does run, on the same logs, and answers the question.**

Depth-1 rows have zero cache mismatch because the root realization pool has size 1. But a size-1
pool is also why `parent_nu == root_nu` on every depth-1 row, so the affect bucket is **constant
within a depth-1 edge**. The two properties are the same fact. Depth 1 is clean because it is
constant, and constant is exactly what makes the within-edge contrast and AffPool's homogeneity
inestimable there. Splitting by depth trades the confound for the estimand.

Mismatch does not need depth as a proxy: it is a **row-level** variable, and `generating_parent_nu`
resolves on 100 % of rows, so the correct (parent → child) pairing can be restored on all 25,328
depth-≥2 rows instead of discarding the 31 % that are broken. Doing that:

- **On mismatched rows the affective signal vanishes** — D1 Δ +0.008 [−0.003, +0.020], standardized
  +0.040, edge-demeaned ω² 0.0001 [−0.000, +0.007]. The placebo behaves. (In D2's all-rows window it
  is a small positive whose CI just excludes zero, §5 — opposite in sign to the real effect, and
  gone under the reachable window.)
- **Restoring the pairing makes the signal 1.45× larger** on the within-edge contrast (standardized
  −0.275 → −0.398) and **2.3× larger** on edge-demeaned ω² (0.0066 → 0.0150), with non-overlapping
  CIs. **So: yes, the cache is suppressing it, and this is a measurement rather than an argument.**
  D2 (β = 0) replicates at 1.32× and 2.2×, so it is not feedback from `Q_emo` onto the search.
- **And it is still small.** Corrected and horizon-cleaned, bucket explains **1.3 %** of within-edge
  variance in z (0.0126 [0.010, 0.022]). Both halves of the question are true: the cache suppresses
  the signal, *and* the signal is weak independent of the cache. Correcting it does not turn the
  NodeKey premise into a strong one.

The ν spread inside a node's realization pool says why a cache fix is not simply available:
**for the 757 edges whose 4-slot pool was fully exercised, the four cached children's ν spans a
median 0.357 — 96 % of the corpus-wide IQR of ν and 53 % of its entire observed range.** An
open-loop node has no single parent affect to key on.

---

## 1. Mismatch, made explicit

`mismatch` = the parent realization sampled on this visit is **not** the one the child was generated
from. Fresh rows are matched by construction; a cached row is matched only if the pool happened to
return this parent's own child. Built by exact join on `(edge_key, child_realization_id)`;
0 unresolved rows.

| | rows | cached | mismatched | rate, all rows | rate among cached |
|---|---|---|---|---|---|
| **depth 1** | 8,575 | **6,019** | **0** | **0.000** | **0.000** |
| depth ≥ 2 | 25,328 | 10,676 | 7,865 | 0.311 | **0.737** |
| depth 2 / 3 / 4 | | | | 0.422 / 0.332 / 0.286 | 0.743 / 0.737 / 0.739 |

The rate among cached rows is flat at **0.74 across every depth ≥ 2**, which is 1 − 1/R at R = 4.
The pool is drawn uniformly, so this is the expected value, not a symptom.

**Why depth 1 is at zero.** `_init_node` seeds the root pool with one state and
`_add_new_realizations`' `__eq__` test keeps it at one (`definitions.md`; `src/mcts/mcts.py:202`).
One realization means every cached depth-1 child was generated from the only parent there is.

## 2. The three statistics at depth 1: two of them do not exist

| statistic | depth 1 | depth ≥ 2 |
|---|---|---|
| edges in stratum | 650 | 4,705 |
| **edges seeing both buckets** | **0 of 650** | 1,447 of 4,705 |
| **within-edge bucket contrast** | **inestimable** | Δ −0.055, std −0.275 |
| distinct action prefixes in stratum | **1** (the empty prefix) | 798 |
| AffPool cells with ≥ 3 prefixes | **0 of 650** | 636 of 1,128 |
| **AffPool homogeneity** | **inestimable** | ratio 0.767, noise-corrected 0.303 |
| pooled ω² | **0.041** [0.017, 0.078] | **0.084** [0.055, 0.127] |
| pooled ω², edge-demeaned | **−0.0007** (0 by construction) | 0.0066 [0.005, 0.011] |

Both "inestimable" rows are counts, not judgements. `parent_nu == root_nu` holds on all 8,575
depth-1 rows, so an edge at depth 1 fixes the bucket; and a depth-1 edge has the empty action
prefix, so every depth-1 AffPool cell is single-prefix and homogeneity has nothing to compare.

**The ω² that survives is the one that cannot answer the question.** Its groups are
(depth, action) pooled across trees and dialogues, so at depth 1 **100 %** of its bucket variation
is between-tree — precisely the component `p_var.md` §7.3 showed to be ~90 % prefix/dialogue
identity. Its edge-demeaned counterpart is 0 by construction. And it is **smaller** at depth 1
(0.041) than at depth ≥ 2 (0.084), so even taken at face value it does not show a stronger signal
where the cache is absent.

## 3. The test that does run: restore the pairing instead of stratifying by depth

Same depth-≥2 rows, five treatments of the cache. `causal_label` re-labels every row with
`generating_bucket_med` — the bucket of the parent that actually produced the child.

**D1 (β = 0.7), all rows**

| population | rows | discordance | Δ mean | std. effect | pooled ω² | ω² edge-demeaned |
|---|---|---|---|---|---|---|
| `as_planner_sees` | 25,328 | 0.308 [0.284, 0.338] | −0.055 [−0.067, −0.043] | −0.275 [−0.329, −0.219] | 0.084 [0.055, 0.127] | 0.0066 [0.005, 0.011] |
| **`causal_label`** | 25,328 | **0.279** [0.259, 0.306] | **−0.080** [−0.096, −0.063] | **−0.398** [−0.473, −0.318] | **0.108** [0.075, 0.158] | **0.0150** [0.014, 0.023] |
| `matched_only` | 17,463 | 0.279 | −0.079 | −0.396 | 0.119 | 0.0151 |
| **`mismatched_only`** | 7,865 | 0.424 | **+0.008** [−0.003, +0.020] | **+0.040** [−0.013, +0.099] | 0.022 | **0.0001** [−0.000, +0.007] |
| `fresh_only` | 14,652 | 0.279 | −0.079 | −0.395 | 0.123 | 0.0165 |

**Reachable window** (18,653 rows; the correction `p_var.md`'s banner requires):

| population | Δ mean | std. effect | pooled ω² | ω² edge-demeaned |
|---|---|---|---|---|
| `as_planner_sees` | −0.044 [−0.058, −0.031] | −0.217 [−0.285, −0.153] | 0.044 [0.033, 0.060] | 0.0035 [0.003, 0.007] |
| **`causal_label`** | **−0.071** [−0.090, −0.052] | **−0.350** [−0.442, −0.259] | **0.068** [0.053, 0.088] | **0.0126** [0.010, 0.022] |
| `mismatched_only` | +0.011 [−0.001, +0.024] | +0.052 [−0.007, +0.114] | 0.017 | −0.0005 [−0.000, +0.006] |

The `as_planner_sees` reachable row reproduces `p_var.md`'s correction note (std −0.215, ω² 0.041)
to within rounding, which validates the pipeline against the documented figures.

**Four things this establishes.**

1. **The broken rows carry no signal at all.** Every mismatched-only estimate is indistinguishable
   from zero in both windows. That is what a random re-pairing should do, and it is the cleanest
   available placebo for the whole affect premise — the same z, the same edges, only the parent
   link severed.
2. **The dilution is quantitative.** 11.5 % of depth-≥2 rows carry a bucket that belongs to a parent
   which did not produce the child. The observed attenuation, Δ ratio 0.055/0.080 = **0.69**, is the
   same order as the **0.77** a binary mislabel at that rate predicts (1 − 2f), slightly stronger.
3. **It is not a sample-selection artifact.** `causal_label` (all 25,328 rows, corrected label) and
   `fresh_only` (the 14,652 de-duplicated causal pairs) agree to 0.001 on Δ. The conclusion does not
   depend on repeat-weighting cached children or on discarding two-fifths of the data.
4. **It replicates without the affect term in selection** — see §5 for D2 (β = 0), so it is not
   feedback from `Q_emo` onto what the search visits.

### ⚠ One gate component moves the *wrong* way

Correcting the pairing does not uniformly improve the NodeKey reading. Against `p_var.md` §7.7's
proposed thresholds:

| component | bar | as planner sees | causal label | effect |
|---|---|---|---|---|
| within-prefix Δ | \|Δ\|/sd ≥ 0.2, CI excludes 0 | −0.275 ✅ | **−0.398** ✅ | stronger |
| pooled ω² | ≥ 0.05, lower CI > 0 | 0.084 ✅ (reachable 0.044 ❌) | **0.108** ✅ (reachable 0.068 ✅) | stronger |
| **discordance rate** | **≥ 0.30** | **0.308** ✅ *(marginal)* | **0.279** ❌ | **fails** |

**The discordance component looked like a marginal pass because of the cache, not despite it.**
Shuffling parents across a node's pool manufactures apparent within-edge bucket variation:
mismatched rows have a discordance rate of 0.424 against 0.279 for correctly paired ones. The
quantity the gate wanted — how often an edge genuinely sees both affective contexts — is 0.279, below
its own bar. Reported, not decided: this is a pre-existing gate reading that should be re-derived on
the causal label before it is cited again.

## 4. ν across a node's R cached utterances — the truncation argument, quantified

Per edge, over the **distinct realizations the pool actually served**. Corpus-wide ν has
sd 0.207, IQR 0.373, range 0.672, τ_med 0.350.

**The 757 D1 edges whose 4-slot pool was filled and fully exercised:**

| | median | p25 | p75 |
|---|---|---|---|
| ν range across the 4 cached children | **0.357** | 0.191 | 0.495 |
| ν sd across the 4 | **0.168** | 0.088 | 0.233 |

- The median pool's range is **96 % of the corpus-wide IQR** of ν and **53 % of its entire observed
  range**; its sd is **81 % of the global sd**. **48 %** of pools have a range exceeding the whole
  corpus IQR.
- **61.6 % of the ν variance across these pools' realizations is *within* a pool**, not between
  nodes (59.5 % over all 1,412 edges with ≥ 2 served realizations).
- **69 % of these pools straddle τ_med.** Two independent draws from the same pool land in different
  affect buckets with mean probability **0.29** (0.26 over all edges with ≥ 2 served).

So R = 4 is not sampling a tight local distribution. It aliases something nearly as wide as the
marginal down to four points, and the affect bucket attached to an open-loop node is close to a coin
flip over which of its realizations was drawn. This is the mechanism behind §3: the 0.74 mismatch
rate turns into a 0.37 label change on mismatched rows precisely because the pools straddle τ — a
mismatched row's parent ν moves by a median 0.135, against an IQR of 0.373.

### The pool is not even a random sample of what the generator produced

On **1,189** D1 edges the generator produced more distinct realizations than the pool ever served
(median 5 generated against 3 served). The served pool's ν range is a median **67 %** of the
generated range, and covers **less than half** of it on 43 % of those edges.

The exclusion is systematic. On the 261 edges where the pool filled to 4 *and* more was generated:

| | n | became a parent | mean v | mean ν |
|---|---|---|---|---|
| served by the pool | 1,044 | 97.9 % | 0.570 | 0.306 |
| **never served** | **817** | **0.0 %** | **1.000** | **0.377** |

**Every one of the 817 never-served realizations has v = 1.0 and none was ever descended through.**
That is the retention rule, not chance: `_add_new_realizations` is reached only when the search
re-enters a node non-terminally (`src/mcts/mcts.py:310`, after the terminal/leaf returns at
:296–306), so a child whose visit returned immediately can never join the pool. On this subset those
children are also the more positive ones (ν 0.377 vs 0.306).

**Scope limit, stated:** across *all* fresh children the served/never-served ν means are 0.322 and
0.323 — no gap. The ν asymmetry above is a property of the 261 full-pool edges, not a global claim.
What is global is the structural exclusion of never-descended children from the pool.

## 5. D2 (β = 0) — replication

The control arm carries no affect term in selection, so anything that replicates here is not
feedback from `Q_emo` onto what the search visits. `D2/p_depth_D2.json`.

**Depth 1 is inestimable for the same structural reason:** 8,673 rows, 5,944 cached, **0
mismatched**; **0 of 738 edges** see both buckets; **1** distinct prefix, so 0 of 738 AffPool cells
reach 3 prefixes. Mismatch among cached depth-≥2 rows is **0.749**, again 1 − 1/R.

Pooled ω² is likewise *smaller* at depth 1 — **0.015** [0.005, 0.052] against **0.083**
[0.063, 0.102] at depth ≥ 2 — so on the one estimator that survives the split, both arms point the
same way: no stronger affective signal where the cache is absent.

| population | Δ mean | std. effect | pooled ω² | ω² edge-demeaned |
|---|---|---|---|---|
| `as_planner_sees` | −0.055 [−0.069, −0.038] | −0.268 [−0.342, −0.189] | 0.083 | 0.0077 [0.004, 0.015] |
| **`causal_label`** | **−0.072** [−0.089, −0.052] | **−0.353** [−0.442, −0.252] | 0.106 | **0.0170** [0.010, 0.030] |
| `mismatched_only` | +0.020 [+0.000, +0.036] | +0.098 [+0.001, +0.174] | 0.034 | 0.0032 [0.002, 0.011] |

Reachable window: −0.228 → **−0.299**, edge-demeaned ω² 0.0031 → **0.0086**. The gains are
**1.32×** on the contrast and **2.2×** on edge-demeaned ω², against D1's 1.45× and 2.3×.

**⚠ The placebo is not quite zero in D2's all-rows window.** `mismatched_only` reads +0.020
[+0.000, +0.036], a CI that just excludes zero. It returns to zero in the reachable window
(+0.013 [−0.007, +0.028]), so it is plausibly the horizon contamination rather than residual signal.
Either way the sign is **opposite** to the real effect, so it works against the naive estimate
rather than inflating it — but "the broken rows carry exactly zero" is a D1 statement, not a
universal one.

Pool spread replicates: for the 563 edges with a fully exercised 4-slot pool, the median ν range is
**0.383 = 97 % of the corpus IQR**, median sd 82 % of the global sd, **67.5 %** straddle τ_med. On
942 edges the generator produced more than the pool served (median 5 against 2), the served range a
median **54 %** of the generated one. On the 173 full-pool edges with excess generation, served ν
averages 0.316 against 0.374 for never-served — the same direction as D1.

---

## 6. What this changes

- **The depth-1 test is closed, and not by its own numbers.** It cannot be run as specified; the
  reason is `p_var.md`-documented behaviour of the root pool, not a data shortage.
- **The cache dilution is now measured, not argued:** 1.45× on the within-edge contrast, 2.3× on
  edge-demeaned ω², with a zero-valued placebo on the broken rows.
- **It does not rescue the premise.** The corrected, reachable within-edge signal is 1.3 % of
  within-edge variance in z. The affective signal is weak independent of the cache *as well as*
  suppressed by it.
- **`p_var.md` §7.7's discordance reading should be re-derived** on `generating_bucket_med`. It reads
  0.279, not 0.308, and the difference is cache-manufactured.
- **A NodeKey that splits on the sampled parent's affect is keying on a coin flip.** 69 % of full
  pools straddle τ_med. If the affective key is to be salvaged, the quantity to key on is a property
  of the node's realization *distribution*, not of whichever realization was drawn — or R must rise
  enough to make the pool mean stable, which is a generation-cost decision, not an analysis one.
