# TASK 5 — Momentum (`--emo_signal delta`) and dose-matching the selection arms

**Status: done.** Code question answered (§1), property verified by replay (§2) and on live trees (§6),
doses matched (§3). Live pilots ran under `episode` at Bias 0.7 / Momentum 1.2 / CenteredBias 1.1,
matched on episode NoEmo trees (§6). Legacy-tree doses (0.7 / 1.3 / 1.1) apply if `legacy` is kept.

## 1. What the arm's score actually is

`--emo_signal` is read in exactly one place, `_emotion_signal` (`src/mcts/emotion_mcts.py`). It sets the
per-visit z that `_update_emo_channel` folds into `Q_emo[s][a]` as a running mean. `_calculate_uct` scores
`Q + β·Q_emo + U` and does not know which signal filled `Q_emo`. So the arm's score is

    score(s, a) = Q(s, a) + β · (1/N(s,a)) · Σ_{visits i of (s,a)} (ν(child_i) − ν(parent_i)) / 2 + U(s, a)

where parent_i is **the parent realization sampled on visit i** (R = 4, so it can differ between visits).
An unvisited edge scores 0.

**Name for §3: "average local affective change" (ALC).** It is the mean change in user valence that
taking action *a* from state *s* has produced over the search so far. **It is not β·Δν, and not the
user's current momentum.** "Momentum" should not appear in §3 as the description of what the score is.
It can survive as the arm's label only if §3 defines it as above.

**On the brief's second option ("have the selection term read the current Δν").** As stated it cannot
work, and not for cost reasons. The current Δν at state *s* — ν(s) − ν(parent of s) — is **the same number
for every sibling action**. Adding β·Δν(s) to all K scores shifts them equally and **never changes the
argmax**. Any selection signal has to be action-dependent, and the edge-level mean above is the
action-dependent version of Δν. The variants that would differ are:

- **last-visit Δν** instead of the mean: noisier, not closer to "momentum", about 20 min to build;
- **Δν × action interaction** (e.g. a mined "which acts work when valence is falling" table): a new
  learned component, not a flag change, and not buildable or validatable today.

So there is no sub-hour change that makes the arm "read the current Δν". **Run the flag and describe it
accurately.** Whether it keeps the structural property is the empirical question in §2, and the replay
answers yes.

## 2. The structural property holds empirically

The Momentum argument needs `Q_emo` to have **mean ≈ 0 on visited edges**, so that β·`Q_emo` is not a flat
bonus for having been visited. By construction a mean of differences need not be zero. **Measured on
logged trees, it is:**

| visited edges | D2 level | **D2 ALC** | D1 level | **D1 ALC** |
|---|---|---|---|---|
| mean `Q_emo` | +0.263 | **+0.001** | +0.255 | **−0.005** |
| sd `Q_emo` | 0.174 | 0.109 | 0.174 | 0.100 |
| corr(`Q_emo`, depth) | +0.292 | **+0.054** | +0.388 | **+0.041** |
| per-visit z, mean / sd | 0.309 / 0.210 | 0.012 / 0.127 | 0.317 / 0.207 | 0.010 / 0.121 |

The visited-edge bonus is gone (+0.001 against +0.263), and ALC is nearly depth-free (r 0.05 against
0.29). **The replay reconstruction is exact:** it reproduces every logged N on 140,645 (D2) / 169,515 (D1)
sibling rows, and the logged level `Q_emo` to 2.2 × 10⁻¹⁶. The live pilot re-checks this on trees grown
under the arm (§6), because ALC-driven search could in principle select into edges with non-zero mean.

## 3. Dose-matching (human decision, 2026-09-16)

**Dose** = argmax flip rate against NoEmo, measured by one-step replay **on the same NoEmo-grown trees**
for every arm (`scripts/t5_dose_replay.py`, D2, 28,129 selection points). This is the only way to put all
three arms on a common scale. Each arm is scored on identical trees and identical selection points.

**⚠ The brief's "CenteredBias 25.5 % vs Bias 31.6 %" is not a dose comparison.** 31.6 % is Bias against
NoEmo on Bias's own D1 trees. 25.5 % (`s1_centre_replay.py`) is **CenteredBias against Bias**, also on
D1's Bias-grown trees. The two numbers have different reference arms and neither is on neutral trees.

### Flip rate against NoEmo, D2 trees

| β | 0.35 | 0.7 | 1.0 | 1.4 | 2.0 | 2.8 |
|---|---|---|---|---|---|---|
| Bias | 11.4 % | **15.5 %** | 18.2 % | 21.0 % | 24.0 % | 26.6 % |
| CenteredBias | 7.3 % | 11.3 % | 14.0 % | 16.8 % | 19.7 % | 22.2 % |
| Momentum (ALC) | 6.1 % | 9.9 % | 12.3 % | 15.1 % | 18.7 % | 22.7 % |

### Matched doses (reference Bias β 0.7 = 15.5 %)

| arm | matched β | flip rate [95 % CI] | flips to visited over unvisited | flips to unexpanded | flips between visited |
|---|---|---|---|---|---|
| Bias | 0.7 | 15.5 % [12.7, 18.9] | 42.5 % | 0.1 % | 57.4 % |
| **CenteredBias** | **1.21** | 15.5 % [13.1, 18.6] | 14.9 % | 5.7 % | 79.4 % |
| **Momentum (ALC)** | **1.47** | 15.5 % [13.2, 18.2] | 21.1 % | 23.0 % | 55.9 % |

**Ablation note for the results table:** at β 0.7, Momentum flips 9.9 % (about ⅔ of Bias's 15.5 %) and
CenteredBias flips 11.3 % (about ¾). **Both are under-dosed at a shared β. CenteredBias needs adjusting
too**, to β ≈ 1.2.

### Calibration on the pilot's own NoEmo trees (`P1_legacy`, dialogues 131–140) — the doses used

| arm | matched β (exact) | **β used** | flip rate at β used, P1 trees [95 % CI] | same β on D2 trees |
|---|---|---|---|---|
| Bias | — | **0.7** | 16.5 % [10.3, 25.8] | 15.5 % |
| CenteredBias | 1.114 | **1.1** | 16.3 % [10.4, 24.7] | 14.7 % |
| Momentum (ALC) | 1.312 | **1.3** | 16.4 % [10.9, 24.6] | 14.5 % |

Flip composition at the matched dose on P1 trees (to visited over unvisited / to unexpanded / between
visited): Bias 37.9 / 0.0 / 62.1 %, CenteredBias 11.8 / 6.9 / 81.3 %, Momentum 16.2 / 25.7 / 58.1 %.
Replay validation on P1 trees: 46,860 sibling rows, 0 N mismatches, max `Q_emo` deviation 2.2 × 10⁻¹⁶.

**Matched on the pilot, as instructed.** The pilot calibration uses 10 dialogues (9,372 selection
points, CI about ±7 points). D2 uses 30 dialogues (CI about ±3) and gives slightly higher matched β
(1.21 / 1.47). At the β used here, both arms sit within 1 point of Bias on D2 as well, so the choice
between the two calibrations is inside the noise. **These are the β values for FREEZE_NOTES:** Bias 0.7,
CenteredBias 1.1, Momentum 1.3, all dose-matched at ≈16.5 % flips against NoEmo on the pilot trees.

## 4. What Bias's flips actually do — correcting the brief's re-reading

The brief's latest reading is: "Bias sends 0.1 % of flips to unexpanded edges and 83 % to visited ones,
so it is re-ranking among already-expanded actions rather than suppressing exploration." **That inverts
the categories.** Each flip is classified by where NoEmo's choice was and where the arm's choice is:

| D1, Bias β 0.7 (own trees) | share of flips |
|---|---|
| NoEmo would pick an **unvisited** edge → Bias picks a **visited** one | **82.8 %** |
| both picks visited (re-ranking among expanded) | 17.1 % |
| Bias picks an unexpanded edge | 0.1 % |

**83 % of Bias's flips replace an exploratory choice with an already-visited edge.** That *is*
suppressed exploration, and it is what the original exploration-artefact story said. Re-ranking among
expanded actions is only the 17 %. The 0.1 % to unexpanded is structural: level `Q_emo` ≥ 0 exists only
on visited edges, so Bias can lift visited edges and never an unvisited one. **The exploration-artefact
reading stands, and §0.3 and §5 of the plan should keep it.** On neutral D2 trees the effect is weaker
(42.5 % visited-over-unvisited, 57.4 % between visited). The 82.8 % on D1 is self-reinforcing narrowing
(`p_var.md` §7.6).

**CenteredBias's "99.9 % to unexpanded" (Wednesday)** is measured against Bias on Bias's trees. It says
that removing the bonus sends choices back to the edges Bias had crowded out, which is consistent with
the story above. **Against NoEmo on neutral trees, CenteredBias flips mostly between visited edges
(79 %)**, and only 5.7 % of its flips go to unexpanded edges. That is the dose-matched picture, and the
one to report.

Plan §0.3 and §5 are not in this repository, so I cannot edit them. Corrections to carry there:

1. Keep the exploration-artefact reading of Bias: 83 % of its flips on its own trees are
   visited-over-unvisited.
2. Report CenteredBias's 99.9 % as "relative to Bias". Against NoEmo it re-ranks among visited edges.
3. Momentum's score is ALC (§1), not β·Δν.

## 5. What the replay cannot tell you

A one-step replay changes one selection on a tree the arm did not grow. It cannot show the whole-search
narrowing (38 % of D1 points with ≥ 2 expanded siblings against 52 % in D2), or whether ALC keeps mean ≈ 0
on trees it grows itself. The live pilots answer both.

## 6. Live pilots — run under `episode` (dialogues 131–140)

**Horizon switch.** P1 (`p1_pilot.md`) showed `episode` changes the planner materially, and it is the
recommended grid default. So the live pilots ran under `episode`, with doses matched on the **episode**
NoEmo trees (`dose_replay_P1_episode.json`): Bias β 0.7 = 11.9 % flips, CenteredBias matched at 1.06
(run at **1.1**), Momentum matched at 1.19 (run at **1.2**). The legacy-tree doses of §3 (1.1 / 1.3) stay
valid if the human keeps `legacy`. NoEmo reference: `P1_episode`. All runs completed 10/10 dialogues on
one server with no restarts.

| | NoEmo | Bias β 0.7 | **Momentum β 1.2** | CenteredBias β 1.1 |
|---|---|---|---|---|
| SR / AvgT | 1.00 / 5.4 | 0.90 / 6.8 | 0.80 / 6.2 | 0.80 / 6.4 |
| **mean `Q_emo` on visited edges** (own trees) | +0.267 (level, not used) | +0.220 | **+0.001** | +0.260 (centred at use) |
| dose: flip vs NoEmo on NoEmo trees (replay, at the run β) | — | 11.9 % | 12.0 % | 12.4 % |
| **flip vs NoEmo on own trees** (live) [95 % CI] | 0 % (validation) | **23.4 %** [19.9, 27.2] | **14.4 %** [10.0, 18.1] | **22.6 %** [17.5, 26.2] |
| own-tree flips: visited over unvisited / to unexpanded / between visited | — | **76.5** / 0.1 / 23.4 % | 42.5 / 8.7 / 48.8 % | 15.3 / 5.6 / 79.1 % |
| **selection points with ≥ 2 expanded siblings** | **49.2 %** | 45.2 % | **52.5 %** | 59.9 % |
| mean expanded siblings | 1.83 | 1.92 | 2.11 | 2.27 |
| root Q spread < 0.01 / median | 4.5 % / 0.63 | 3.4 % / 0.56 | 9.6 % / 0.63 | 7.4 % / 0.60 |
| wall clock / dialogue | 231 s | 294 s | 241 s | 279 s |

Doses at the run β computed directly on the episode NoEmo trees (exact matches 1.19 / 1.06 give 11.9 %).

**Validation.** The NoEmo score, rebuilt from logged Q, prior, N and parent Ns, reproduces the logged UCT
of the NoEmo run to 2.2 × 10⁻¹⁶ (0 flips on 6,360 points). The own-tree flip rates rest on that
reconstruction (`scripts/t5_own_tree_flips.py`, `own_tree_flips.json`).

The brief's checks, answered on live trees:

| check | expected | result |
|---|---|---|
| mean `Q_emo` on visited edges | ≈ 0, unlike level | ✅ **+0.001** on Momentum's own trees (Bias +0.220). **The property survives search driven by the signal** |
| flip rate | report; compare with Bias / CenteredBias | Dose-matched on neutral trees (≈ 12 %). **On their own trees, Bias and CenteredBias roughly double (23.4 %, 22.6 %) while Momentum rises only to 14.4 %.** Momentum is the least self-reinforcing |
| flips that suppress exploration (visited over unvisited) | far below Bias | ✅ **42.5 %** against Bias's **76.5 %** |
| expanded-sibling count | resembles NoEmo, not Bias | ✅ **52.5 %** of points with ≥ 2 expanded, against NoEmo 49.2 % and Bias 45.2 %. Momentum is slightly *wider* than NoEmo, and Bias is the only arm that narrows |

**What this says about the three arms at matched dose.**

- **Bias** reproduces its known signature on live trees. 76.5 % of its flips pick a visited edge over an
  unvisited one, and it is the only arm that narrows search. This is D1's pattern (82.8 %) under the fixed
  horizon.
- **CenteredBias** removes the narrowing (59.9 % of points with ≥ 2 expanded siblings, the widest of the
  three). Its flips are overwhelmingly re-ranking between visited edges (79 %). It moves selections as
  often as Bias does, but in a different way.
- **Momentum (ALC)** has no visited-edge bonus in practice, stays close to NoEmo's search width, and moves
  the fewest selections on its own trees. Of the three it is the lightest touch on the planner.

**Not interpretable at n = 10:** the SR / AvgT row. NoEmo is at 10/10 under `episode` on these
dialogues, so every arm sits at or below a ceiling. See the SR-ceiling note in `actpool.md` §3.

Runs: `runs/E_bias_b0.7`, `runs/E_momentum_b1.2`, `runs/E_centre_b1.1`. Metrics:
`episode_pilots_metrics.json`, `own_tree_flips.json`. Scripts: `scripts/t5_momentum_replay.py`,
`scripts/t5_dose_replay.py`, `scripts/t5_own_tree_flips.py`.
