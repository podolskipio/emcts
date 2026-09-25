# Readiness Phase 2 — mechanical verification of the cache fixes

**Status: STOPPED FOR A HUMAN CALL.** Five of seven pre-registered checks pass. Two fail:
- **Exact-parent mismatch** is outside the brief's band. The band turns out to be **unreachable
  by any draw rule once fix #5 is on**; see §3.
- **Bucket-hit rate** falls outside its band on the *good* side.

Fixes #6 (retention) and #5 (fresh depth-1) are verified. Fix #1 + #2 (the bucket_kernel draw)
does what it was built to do **on the affect dimension**, but not on exact-parent identity.

Pre-registered in `PREREG.md` Entry 9 (`61d8f58`) before either run. Script
`scripts/phase2_verify.py`, numbers `phase2_verify.json`. Runs are in `runs/`:

| Run | Flags beyond the frozen config | Commit | n | Wall |
|---|---|---|---|---|
| `P2_NoEmo_fixes_n25` | `--cache_ended_children --cache_fresh_depth1` | `61d8f58` | 25 | 0.77 h |
| `P2_AffPool_fixes_n25` | the same + `--cache_draw bucket_kernel --cache_bucket_tau 0.263 --cache_kernel_h 0.2` | `f69b7f4`¹ | 25 | 0.75 h |

¹ Differs from `61d8f58` only in `analysis/` (`git diff 61d8f58 f69b7f4 -- src` is empty): same planner code.

Other shared settings: T 1.1, persona ON, episode horizon, R = 4, top-K 5, `num_mcts_sims 20`,
seed 1, and the first 25 eval dialogues.

**Reference ("before")**: the same measures on frozen `A_NoEmo_s20_seed1` / `A_AffPool_s20_seed1`,
restricted to the same 25 dialogues. The reference reproduces the brief's parent mismatch: 0.738
against 0.737. It is a mechanical reference, not a paired baseline (Rule 10). No success rate is
read here.

## 1. The checks

| Fix | Measure | Before (frozen, same 25) | After | Pass criterion | Verdict |
|---|---|---|---|---|---|
| **#6** retention | never-served realizations (p_depth definition) | 59 (NoEmo) / 34 (AffPool) | **0 / 0** | ≤ 10 % of before | ✅ |
| **#6** retention | ended-reply draw share − its share of the pool at draw time (NoEmo, uniform draw) | ended replies never drawable | **+0.005** [−0.009, +0.021] (0.377 vs 0.372) | CI contains 0 or \|Δ\| ≤ 0.02 | ✅ |
| **#5** depth-1 | cache hits on depth-1 edges | 1,059 / 940 | **0 / 0** | 0 | ✅ |
| **#5** depth-1 | generations per depth-1 edge | 3.2 / 3.5 | **5.9 / 5.7** (= visits per edge) | > 4 | ✅ |
| all | searches started from an ended state | not observable before | **0 / 0** | 0 | ✅ |
| **#1** bucket | draws with no bucket miss, share on the parent's side of τ | — | **1.000** | 1.000 | ✅ |
| **#1 + #2** | exact-parent mismatch, cached depth ≥ 2 (AffPool) | 0.744 | **0.744** [0.711, 0.782] | 0.20–0.35 | ❌ |
| **#1** bucket | bucket-hit rate (AffPool) | — | **0.916** [0.893, 0.938] | 0.55–0.85 | ❌ (high side) |

About "~10 generations per depth-1 edge": at `num_mcts_sims 20` with top-5 there are only ~5.5 visits
per depth-1 edge. The brief's 9.80 came from the D1 diagnostic runs. With #5, generations now equal
visits, and that is the most the fix can deliver at this budget.

## 2. What #6 and #5 do, in numbers

- **Retention (#6) works.** 1,019 (NoEmo) and 1,032 (AffPool) ended replies were filed. The cache
  served an ended reply 451 and 200 times, which could not happen before.
  - **Draw rate:** the draw rate matches the pool (above).
  - **Never-served ended replies:** among ended replies that had later cache hits at their edge, 83
    were never served, against 75.7 expected under a uniform draw (NoEmo). Missing a reply is now
    chance, not structure.
  - **The frozen pattern is gone.** The 59 / 34 never-served realizations in the frozen reference sit
    on 28 / 17 edges, and have higher ν than the served ones (0.30 vs 0.22). That is the pattern
    p_depth found at scale, and none of it remains.
- **No search starts from an ended state** in either run: the ended replies are served but never
  expanded.
- **Fresh depth-1 (#5) works.** Every depth-1 visit generates, and there are no depth-1 cache hits.

## 3. Why exact-parent mismatch did not fall — the fixes interact

Exact-parent mismatch among cached rows at depth ≥ 2, split by depth. "Floor" = the share of draws
where the **current parent generated none of the replies in the pool**, so no draw rule could match
it:

| Run | Depth | Rows | Exact mismatch | Uniform draw, same pools | Bucket mismatch | Distinct parents behind the pool | **Floor** |
|---|---|---|---|---|---|---|---|
| frozen NoEmo | 2 | 521 | 0.731 | 0.749 | 0.217 | 2.8 | 0.290 |
| frozen NoEmo | 3+ | 259 | 0.753 | 0.740 | 0.367 | 3.1 | 0.212 |
| fixed NoEmo | 2 | 766 | 0.918 | 0.902 | 0.347 | 2.8 | **0.732** |
| fixed NoEmo | 3+ | 430 | 0.665 | 0.700 | 0.247 | 2.5 | 0.265 |
| frozen AffPool | 2 | 319 | 0.737 | 0.758 | 0.248 | 2.8 | 0.266 |
| frozen AffPool | 3+ | 44 | 0.795 | 0.761 | 0.432 | 2.9 | 0.273 |
| fixed AffPool | 2 | 456 | 0.829 | 0.895 | **0.094** | 2.8 | **0.715** |
| fixed AffPool | 3+ | 127 | **0.441** | 0.665 | **0.047** | 2.5 | 0.252 |

Overall floor: frozen 0.26–0.27. **Fixed NoEmo 0.56 [0.53, 0.60]. Fixed AffPool 0.61 [0.57, 0.66].**

**Mechanism.**
1. Fix #5 uncaps the depth-1 pools (~5.9 realizations instead of ≤ 4).
2. The depth-2 parent is drawn from that larger pool.
3. The depth-2 child pool is still capped at R = 4, and was filled by ~2.8 of those parents.
4. So 72 % of depth-2 draws come from a parent that contributed nothing to the pool.

A draw that picks among existing replies cannot undo that. **With #5 on, the brief's 0.20–0.35 band is
below the floor, so no draw rule can reach it.** It sits at the frozen cache's floor, 0.27, which
suggests the band was set without #5 in mind.

**Within what is reachable, the draw works:**

- **Depth ≥ 3** (floor 0.25): exact mismatch is 0.441, against 0.665 for a uniform draw on the same
  pools.
- **Draws where a match was possible:** mismatch ≈ (0.744 − 0.614) / (1 − 0.614) ≈ 0.34.
- **On the dimension the draw targets, affect,** bucket mismatch falls from 0.27 (frozen) to **0.084**.
  The median |ν_generating − ν_current| falls from 0.119 to **0.047**.

This contradicts my pre-registered prediction (0.40–0.60). I predicted from the kernel weights and
missed the #5 interaction.

**Bucket-hit rate 0.92 vs the 0.55–0.85 band.** The band was anchored on the brief's "~69 %". That
figure is the share of pools that *straddle* τ, which is a different quantity from a hit rate. A hit
only needs one reply from the parent's side, and that is far more common. The high value is favourable
(fewer fallbacks to a uniform draw), but it is outside the pre-registered band, and it is reported as
such.

## 4. The decision needed before Phase 3

Brief v2 §4: "Any Phase 2 fix fails its check → stop and ask." Options:

1. **Accept #1 + #2 as verified on the affect dimension**, and restate the mismatch check as bucket
   mismatch (0.27 → 0.08), or as exact mismatch above the floor. #5 and #1 + #2 then run together, as
   the brief's Phase 5 AffPool/ActPool arms specify. Exact-parent mismatch at depth 2 stays high
   (0.83). That matters if the analysis needs exact pairing rather than same-mood pairing.
2. **Add a generation rule:** at depth ≥ 2, generate instead of serving from the cache when the
   current parent has no reply in the pool. That removes the floor, at extra LLM cost. Up to 61 % of
   today's depth-≥2 cache hits would become generations. It is a new fix behind its own flag, needs a
   bit-identity test, and would need a re-verification run (~1.5 h).
3. **Drop #5 from the affect arms.** The floor returns to ~0.27, and depth-1 edges replay the same R
   replies again. That trades the depth-1 fix for exact pairing.

Recommendation: **option 1** for the readiness question. The affect methods key on the parent's mood,
not its identity, so bucket mismatch is the relevant measure, and it fell as designed. Option 2 is the
cleaner long-term fix if a later analysis needs causal parent→child pairs.

The bucket-hit band needs an explicit acceptance too. Recommendation: accept it; it fails on the
favourable side and was mis-anchored.
