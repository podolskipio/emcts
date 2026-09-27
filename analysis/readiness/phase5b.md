# Readiness 5b — AffPool vs ActPool on the fixed open-loop planner

The direct test of whether affect works once the open-loop cache no longer scrambles moods. It replaces
the brief's "limited" Phase 5 by human decision. Pre-registered in `PREREG.md` Entry 13 (`956135a`).
Script `scripts/phase5b.py`, numbers `phase5b.json`.

**Answer.**
- **The fixes produced a small, real change in how the tree pools.** The mood key now yields slightly
  more homogeneous task returns than the act key.
- **That change does not translate into better decisions.** AffPool does not beat ActPool. Its success
  rate is 0.026 lower, not detected at a resolution of ~0.06, and lower in all 5 seeds.

## Setup

Both arms carry every fix: #6 retention, #5 fresh depth-1, and the #1 + #2 bucket_kernel draw (τ 0.263,
h 0.2). They are coupled on one store (`coupling/p4.sqlite`). Shared settings: T 1.1, persona ON, episode
horizon, R 4, top-K 5, s20, the 100 eval dialogues, seeds 1–5.

**They differ only in the pooling key:**
- **AffPool:** (tree, parent-mood bucket, act).
- **ActPool:** (tree, act).

Seed 1's ActPool is `P4_ActPool_seed1`, whose arguments are identical. All 10 runs finished 100/100.

## Success rate (primary)

| Seed | ActPool SR | AffPool SR | Diff | Coupled corr | ActPool AvgT | AffPool AvgT | Never diverge |
|---|---|---|---|---|---|---|---|
| 1 | 0.70 | 0.69 | −0.01 | 0.41 | 6.64 | 6.54 | 38 % |
| 2 | 0.77 | 0.72 | −0.05 | 0.40 | 6.35 | 6.41 | 39 % |
| 3 | 0.72 | 0.71 | −0.01 | 0.39 | 6.42 | 6.54 | 47 % |
| 4 | 0.78 | 0.74 | −0.04 | 0.35 | 6.19 | 6.60 | 39 % |
| 5 | 0.75 | 0.73 | −0.02 | 0.38 | 6.45 | 6.48 | 50 % |
| **pooled** | | | **−0.026** [−0.068, +0.018] | 0.38 | | | |

| | Value |
|---|---|
| AvgT difference (AffPool − ActPool) | +0.10 turns [−0.09, +0.30] |
| Detectable SR difference at this design (5 coupled seeds, corr 0.38) | **≈ 0.062** |

**Not detected at this resolution** (Rule 9).
- **Direction:** every seed points the same way, toward ActPool. A sign test on 5 of 5 gives p ≈ 0.06,
  so that is not significant either.
- **What the data allows:** the CI rules out AffPool being better by more than +0.018. It also rules out
  it being worse by more than 0.068.
- **Prediction:** the pre-registered prediction (not detected, |diff| < 0.06) held.

## Pooling homogeneity (decision level)

Both keyings are applied to the same AffPool steps. The measure is the noise-corrected between/within
ratio of the task return, taken as a visit-weighted median over cells with ≥ 3 prefixes; lower means more
homogeneous. Gap = ratio(act key) − ratio(mood key); a positive gap means the mood key buys homogeneity.

| | Mood key: ratio | Evidence ×| Act key: ratio | Evidence × | **Gap** | Dialogues |
|---|---|---|---|---|---|---|
| **before** (frozen AffPool, unfixed cache) | −0.033 [−0.083, +0.036] | 1.43 | −0.029 [−0.066, +0.017] | 3.00 | **+0.004** [−0.041, +0.042] | 100 |
| **after** (5 fixed AffPool runs) | −0.020 [−0.048, +0.004] | 1.33 | +0.001 [−0.026, +0.034] | 2.75 | **+0.021** [+0.003, +0.043] | 499 |

**Reading.**
- **Before the fixes, the mood key bought nothing.** The gap was 0.004, centred on zero. The brief's
  0.046 vs 0.073 contrast came from the older D1 runs (legacy horizon, soft table), and it is not present
  in the current environment.
- **After the fixes, the mood key is slightly more homogeneous than the act key.** The gap is 0.021, and
  its CI excludes zero. That is the direction the marginalization account predicts: with moods no longer
  scrambled at cache hits, grouping returns by mood separates them a little better.
- **The pre-registered "gap closes" criterion is not met.** It required the after-CI's lower bound
  (0.003) to exceed the before point estimate (0.004), and it misses by 0.001. The before and after CIs
  overlap, so a *change* from before is not established. One reason is that "before" is a single run with
  a wide interval.
- **The advantage is small, and it costs half the evidence.** Mood cells pool 1.3× a node's own visits;
  act cells pool 2.75×. The success rates reflect that trade: the slightly cleaner mood cells do not
  compensate for having half as much data in each.

## What this answers

**Do the fixes make affect work in open-loop MCTS?**
- **Mechanically, yes.** The draw matches moods (Phase 2/4: bucket mismatch 0.24 → 0.09). Mood-keyed
  pooling now carries a small, statistically non-zero homogeneity advantage that was absent before.
- **In outcomes, no.** At a resolution of ~0.06, affect-keyed pooling does not beat act-keyed pooling,
  and it trends slightly worse.

This fits the four earlier signal measurements (C1a, C1b, the 1.3 % ω² ceiling, node spread). The
plumbing now carries the affective signal, but at about 1 % of variance, splitting the evidence by mood
costs more than the signal returns.
