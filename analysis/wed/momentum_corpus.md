# §4 The momentum hypothesis — corpus test (human P4G data)

**Verdict: H is not supported on human data.** Direction of affective change does not measurably
modulate which act precedes donation, in the primary test or in any sensitivity. One partial pattern
exists (emotion appeal after falling affect, below); it is not direction-specific and the propose row
runs against the prediction.

## Data: a deviation from the brief, decided before the analysis

The brief asks for 917 non-eval dialogues. **Per-turn system acts exist only for the 300 annotated
dialogues** (`full_dialog.csv` has no act column), so §4 runs on those. That trigger was raised and
answered on 2026-09-15: split the 297 usable dialogues (300 minus 3 content-filtered) by
`select_p4g_holdout`. The **first 100 are the test set**, and w(e) is **re-mined on the other 197**
(`corpus/w_mined_200.json`: happiness +0.41, sadness −0.39, fear −0.10, neutral −0.12), so ν on the test
dialogues is not fitted on their own outcomes. The annotated 300 are disjoint from the eval set (asserted).
Sample size is about a third of what the brief planned. The consequence is visible below: most
act × tercile cells in the test set hold fewer than 30 turns.

Table C: `corpus_turns.parquet`. One row per (system turn, distinct planner act in it); raw labels are
mapped onto the planner's 7-act inventory. ν is the HF classifier on the whole preceding user turn,
as the planner classifies a simulator reply. Outcome: dialogue-level `agree-donation`.

## Occupancy first

Terciles are cut on the sample's own Δν, so each bucket holds a third by construction. The meaningful
check is whether "falling" turns really fall:

| sample | cuts (falling < a ≤ flat ≤ b < rising) | turns | Δν < 0 | Δν < −0.1 |
|---|---|---|---|---|
| primary (test100, w200) | −0.064 / +0.061 | 837 | 51 % | 29 % |
| generic297 | −0.081 / +0.087 | 2,484 | 50 % | 32 % |

**Falling affect is well populated** (29–32 % of turns drop by more than 0.1). The §11 trigger
(falling < 10 %) does not fire on human data. Mean turn index is flat across terciles (5.6 / 5.6 / 6.0).

## Interaction table: donation rate per (Δν tercile, act), Wilson 95 %

**Primary: test100, w(e) mined on the other 197.** Cells with n < 30 are shown but not interpreted (—).

| act | falling | flat | rising |
|---|---|---|---|
| credibility appeal | 0.39 [0.28, 0.51] n=62 | 0.34 [0.25, 0.44] n=92 | 0.47 [0.35, 0.59] n=60 |
| other | 0.55 [0.47, 0.63] n=156 | 0.47 [0.39, 0.56] n=135 | 0.48 [0.40, 0.56] n=149 |
| proposition of donation | — 0.50 n=22 | — 0.33 n=18 | — 0.33 n=24 |
| emotion appeal | — 0.63 n=19 | — 0.30 n=20 | — 0.50 n=20 |
| logical appeal | — 0.57 n=23 | — 0.50 n=26 | — 0.50 n=28 |
| task related inquiry | — 0.42 n=12 | — 0.23 n=13 | — 0.61 n=18 |

**generic297:** all 297 dialogues, hand-signed valence, which is not fitted on outcomes, so there is no circularity at ~3× the sample.

| act | falling | flat | rising |
|---|---|---|---|
| credibility appeal | 0.48 [0.41, 0.54] n=212 | 0.43 [0.37, 0.49] n=271 | 0.44 [0.37, 0.52] n=165 |
| other | 0.50 [0.45, 0.55] n=399 | 0.50 [0.45, 0.55] n=368 | 0.53 [0.48, 0.58] n=413 |
| proposition of donation | 0.59 [0.46, 0.70] n=63 | 0.48 [0.38, 0.59] n=81 | 0.44 [0.34, 0.55] n=82 |
| **emotion appeal** | **0.67 [0.56, 0.77] n=73** | 0.51 [0.39, 0.63] n=65 | **0.43 [0.32, 0.55] n=69** |
| logical appeal | 0.51 [0.42, 0.61] n=105 | 0.50 [0.40, 0.60] n=98 | 0.52 [0.42, 0.62] n=96 |
| task related inquiry | 0.59 [0.44, 0.72] n=44 | 0.38 [0.24, 0.55] n=34 | 0.58 [0.45, 0.70] n=57 |

H predicts propose low → HIGH across the row and emotion appeal HIGH → low. **Propose goes the other
way** (0.59 → 0.44). **Emotion appeal does fall with Δν**, which is the falling-momentum row of H
(and §IV-D's trust-building-after-negative-affect claim).

## Formal test: logistic M0 act / M1 + Δν / M2 + act:Δν, turn index controlled

Acts enter the models only with ≥ 30 rows and ≥ 10 per tercile (greeting, n = 1, is excluded; see
"calibration"). Δν and ν are z-scored. ΔR² is McFadden pseudo-R² with cluster-bootstrap CIs (which never
include 0, since nested increments are non-negative, so the Wald p carries the inference).

| sample | M1→M2 ΔR² (Δν) | interaction p (clustered Wald, F-corr.) | naive LR p | same with **level ν**: p | DiD propose − emotion [CI] |
|---|---|---|---|---|---|
| **primary** | 0.0055 [0.0024, 0.021] | **0.22** | 0.23 | 0.036 | −0.04 [−0.42, +0.35] |
| pre-agreement turns | 0.0055 | 0.44 | 0.49 | 0.23 | −0.01 [−0.45, +0.45] |
| single-act turns | 0.0012 | 0.77 | 0.77 | 0.16 | n/a (emotion appeal < 30) |
| **generic297** | 0.0031 [0.0014, 0.0083] | **0.023** | 0.043 | **0.017** | +0.09 [−0.12, +0.32] |
| soft297 (circular) | 0.0012 | 0.43 | 0.47 | 0.42 | −0.00 [−0.21, +0.23] |

The Δν main effect (M0→M1) is ≈ 0 everywhere (primary ΔR² 0.0000 [0.0000, 0.0008]). Adding turn index
changes no interaction p by more than 0.01.

**Decision rule, fixed before the models ran:** H is supported only if (i) the interaction p < 0.05 with
turn controlled, (ii) the DiD CI is above 0, and (iii) the level-ν version does not show the same
interaction. **Primary fails (i) and (ii). generic297 passes (i) but fails (ii) and (iii).**

Per-act slopes on Δν in generic297 (log-odds per SD, cluster-bootstrap CI): emotion appeal **−0.42
[−0.71, −0.19]**, propose −0.15 [−0.44, +0.12], credibility −0.07 [−0.23, +0.08], other +0.02. On
**level** ν the emotion-appeal slope is −0.32 [−0.64, −0.04]. The one clear slope is there with level as
with direction, so the pattern is "emotion appeal works better with less positive users", not a momentum effect.

**Last system act only** (one outcome per dialogue): 1–8 turns per propose / emotion-appeal cell. Too
sparse to read; reported in the JSON, not interpreted.

## Controls, as the brief requires

- **Turn index:** in all three models; nothing moves (above).
- **Act frequency:** per-cell n is shown in every table; primary propose / emotion-appeal cells are all
  under 30 and are not interpreted.
- **Dialogue-level outcome on turn-level acts:** every turn of a donating dialogue inherits y = 1, which
  inflates association for acts common in successful dialogues. The pre-agreement restriction removes
  post-donation turns (thanks, closing); the result does not change.

## Calibration: why the clustered test is not trusted alone

A simulated clustered null (100 dialogues, dialogue-level outcome, 7 acts, two at 5 %) gave the
clustered Wald χ² test a 8.3 % size at nominal 5 %, and 7.0 % with the F(q, G−1) correction. On the real
data the uncorrected version returned p ≈ 1e-14 purely from the single `greeting` row, which got a
separated coefficient with a spuriously tiny cluster SE. Hence the act filter and the three-part rule above.

## For the paper (§6.5 item 1)

"In P4G, the effect of a persuader's act on donation does **not** depend on the direction of the
persuadee's affective change (act × Δν: p = 0.22 on held-out dialogues with outcome-free valence
weights; p = 0.023 on all 297 with a hand-signed table, where the same interaction appears with affect
*level*). Emotion appeals are followed by donation more often when the persuadee's affect is less
positive: 0.67 [0.56, 0.77] after falling affect against 0.43 [0.32, 0.55] after rising."

Machine-readable: `s4_momentum_corpus.json`. Scripts: `scripts/build_corpus_turns.py`, `scripts/s4_momentum_corpus.py`.
