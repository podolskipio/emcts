# Pre-registration

Entries are append-only and timestamped. An amendment is a new entry that names the entry it amends;
earlier entries are never edited.

---

## Entry 1 — 2026-09-16T21:49:28+02:00

Written before any grid run. Text as specified in the Thursday build spec, Task 8.

```
CO-PRIMARY OUTCOMES: SR and AvgT.
  Rationale: the plausible mechanism for affect is timing — when to ask — not
  whether success eventually arrives. AvgT is already measured.

DIFFICULTY SPLIT: eval dialogues split at the median of baseline (NoEmo) SR per
  persona. Prediction: affect arms beat NoEmo on the hard half by more than on the
  easy half.

ARM PREDICTIONS:
  Momentum > Bias, and Momentum ≥ CenteredBias. If Momentum ≈ CenteredBias the
    gain was removing the visited-edge bonus; if Momentum > CenteredBias the
    direction signal itself carries information.
  ActPool > NoEmo, and AffPool ≈ ActPool.
  ActPool's advantage is largest at n_sims=10.
  TrajValue > NoEmo; its advantage does NOT depend on n_sims, unlike the pooling
    arms, because it adjusts the value estimate rather than the sample size
    behind it.
  TrajPrompt ≥ TrajValue.
```

Undeclared subgroup analysis is not a finding. This file is what makes the difficulty split honest,
and it must exist before the first grid run.

---

## Entry 2 — 2026-09-16T22:04:14+02:00 — amends Entry 1

Written before any grid run.

```
AMENDS Entry 1, prediction "ActPool's advantage is largest at n_sims=10". Replaced by:

  ActPool's advantage is largest at low budget; measured at n_sims=20 against 50.
  n_sims=10 was dropped because 47%/29% of roots are decided by tiebreak at that
  budget, making a gain there uninterpretable.

  Budget curve (E4): n_sims ∈ {20}, with E1's 50 as the second point.
  Evidence: analysis/thu/headroom.md §4 (tiebreak rate by budget, D1/D2).

ADDS: selection arms are DOSE-MATCHED, not beta-matched.

  Bias, CenteredBias and Momentum are compared at the beta that gives each the
  same argmax flip rate against NoEmo, measured by one-step replay on the same
  NoEmo-grown trees (analysis/thu/scripts/t5_dose_replay.py). Reference: Bias at
  beta 0.7. The matched beta and the achieved flip rate of every arm are recorded
  in FREEZE_NOTES before the first grid run and reported with every result.
  Ablation reported alongside: Momentum at beta 0.7 flips about 2/3 as often as
  Bias at 0.7 (9.9% vs 15.5% on D2 trees).
```

---

## Entry 3 — 2026-09-16T22:05:49+02:00 — corrects the figures in Entry 2

Written before any grid run. No prediction changes.

```
Entry 2 quotes "47%/29% of roots are decided by tiebreak" at n_sims=10. Those were
measured at 10 root VISITS, which is n_sims=11: a 50-simulation search logs 49 root
selections, one simulation expanding the root. At exactly n_sims=10 (9 root visits),
share of roots with Q spread < 0.01, D1 (beta 0.7) / D2 (beta 0):

  n_sims=10:  49.1% [37.6, 61.9]  /  31.6% [24.9, 39.9]
  n_sims=20:  28.0% [18.0, 39.4]  /  13.0% [ 7.8, 19.6]
  n_sims=50:   2.3% [ 0.5,  4.9]  /   0.6% [ 0.0,  1.9]

95% cluster-bootstrap CIs over 30 dialogues. The decision to drop n_sims=10 stands.
Source: analysis/thu/nsims_sensitivity.json, analysis/thu/headroom.md §4.
```

---

## Entry 4 — 2026-09-17T10:25:23+02:00 — removes two arms from Entry 1, before any grid run

```
REMOVES the TrajValue predictions ("TrajValue > NoEmo; its advantage does NOT depend
  on n_sims ...") — the arm failed its pre-specified gates before any grid run:
  Gate A incremental McFadden 0.0125 [0.0039, 0.0551] < 0.02; Gate B out-of-fold Brier
  gain -0.0024 [-0.0209, +0.0167] and two cells flip sign in 2 of 5 folds.
  analysis/thu/trajvalue_gates.md. Rule: either gate fails => the value arms die.

REMOVES "TrajPrompt >= TrajValue" — TrajPrompt dropped by the human (2026-09-17) for
  time; never built, never run.

No other prediction changes.
```

---

## Entry 5 — 2026-09-17T11:27:30+02:00 — frozen environment, valence table, τ and doses; amends Entries 1–2

Written before any grid run.

```
HORIZON: --search_horizon episode for every grid run. legacy appears only as the
  disclosed ablation B2. Evidence: analysis/thu/p1_pilot.md.

VALENCE TABLE: --emo_valence_table generic (textbook signs, never fitted to outcomes).
  soft (shipped) and predecision are reported as findings, not used by grid arms,
  except predecision in the optional ablation C4.
  Evidence: analysis/thu/remine.md, including the addendum: 42 tests, 0 hits after
  multiplicity correction.

AFFPOOL TAU: --aff_pool_tau 0.263 = median parent nu under generic on the episode
  NoEmo pilot trees (50.0% occupancy). tau is not passed to ActPool.

DOSES — AMENDS Entry 2's "reference: Bias at beta 0.7":
  Arms are matched on argmax flip rate against NoEmo (one-step replay on the same
  NoEmo episode trees), NOT on beta. Beta differs per arm. Bias keeps beta 0.70;
  CenteredBias and Momentum are anchored to Bias's ACHIEVED flip rate under generic.
    Bias          --beta_emo 0.70 --emo_signal level               flip 15.20% [13.1, 17.9]
    CenteredBias  --beta_emo 1.03 --emo_signal level --emo_centre  flip 15.24% [12.6, 18.6]
    Momentum      --beta_emo 1.11 --emo_signal delta               flip 15.20% [13.4, 17.2]
  Every reported result gives each arm's beta together with its achieved flip rate.

STILL OPEN, decided before the freeze and recorded in a later entry:
  primary budget (n_sims 20 or 50) and success criterion (--p4g_success), both
  pending the n=100 NoEmo saturation run on non-eval dialogues 141-240.
```

---

## Entry 6 — 2026-09-17T17:39:01+02:00 — primary budget, success criterion; closes Entry 5's open items

Written before any grid run.

```
SATURATION CHECK (non-eval dialogs 141-240, NoEmo, episode, n_sims 50, n = 100):
  SR 0.78 [0.69, 0.85] under the environment's detector. Not saturated (threshold 0.95).
  analysis/thu/success_criterion_fix.md.

PRIMARY BUDGET: --num_mcts_sims 50. n_sims 20 is the budget contrast (plan rule: 20 would
  be primary only if 50 showed a ceiling). Budget predictions are read with 50 as the
  reference level.

SUCCESS CRITERION (environment): --p4g_success tag (the simulator's [donate] self-tag,
  as in GDP-Zero). Unchanged for every grid run.
  Why the stricter detector is NOT in the environment: success terminates the episode
  and every search branch, so a false "not a donation" corrupts every backup through
  that branch. Genuine hedged non-commitments are 3 of 78 tag-successes on the n = 100
  run (3.8%; 6.4% counting both ambiguous turns). The strict detector was fitted on
  those same logged turns and not independently validated.

SECONDARY (offline, no rerun): SR under --p4g_success committed (hedge detector v2,
  src/games/p4g_success.py), scored from the logged [donate] turns and reported for
  every arm beside the primary SR. Before it is cited as a rate, a second reader
  labels a sample of grid [donate] turns that v2 was not written against.

DROPPED: amount-based success criteria (the simulator names amounts it could not pay:
  3% of successes <= ).

CAUTION RECORDED: the cumulative SR on the n = 100 run went 1.00 (after 10) -> 0.78
  (after 100). The P1 horizon pilot (10 dialogs) is therefore not reported as a result;
  the legacy-vs-episode contrast is measured at n = 100 in B2.
```
