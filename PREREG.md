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

---

## Entry 7 — 2026-09-25T12:26:34+02:00 — Readiness Phase 1 gates (C1a, C1b, C3, T*)

Written before any Phase 1 measurement. Program: analysis/readiness (readiness brief). Environment for
every call below is the frozen grid's (Vicuna-13B-AWQ on SGLang, top_p 0.9 / top_k 40, --p4g_persona
ON, hf classifier, generic valence, --logit_scoring off) except where a temperature is named.
Persona stays ON everywhere in this program (human decision, 2026-09-25): the frozen grid ran with
it, and turning it off at T* would change two environment settings at once.

```
DATA (C1a, C1b): the 297 annotated P4G dialogues (non-eval by construction), PRE-DECISION cut as in
  Entry 4 / trajvalue_gates.md: the prefix is every turn strictly before the first user decision act,
  ending on a persuadee turn. Dialogues with no pre-decision turn are dropped (expected ~285).
  One prefix per dialogue. Outcome: `donated` (analysis/thu/user_turns.parquet).
v: the value estimator exactly as today -- P4GChatSystemPlanner.heuristic, 10 samples at T 1.1,
  with the dialogue's persona. Also recorded, as robustness only: v_logit (score_value_labels,
  deterministic).
r: a new judge prompt (observer framing, no persona): "how likely is it that this conversation is
  lost for emotional reasons -- the Persuadee annoyed, shutting down, or disengaging", answered as
  Rating: [0..10]; r = E[rating]/10 read off the logits (score_labels over "0".."10"),
  deterministic. Prompt text is fixed in analysis/readiness/scripts/phase1_c1.py at the commit
  that precedes the run.

C1a: corr(r, v) (Pearson) and incremental McFadden R2 of r over v in logit(donated).
  C1b: incremental McFadden R2 of r over n_turns (persuadee turns in the prefix).
  Both: 1000-replicate bootstrap over dialogue_id (one prefix per dialogue, so the cluster is the row).
  PASS (each): 95% CI excludes 0 AND point >= 0.02.
  Also reported, not gating: 5-fold out-of-fold log-loss gain (a nested in-sample increment
  cannot be negative), and the same with v_logit in place of v.
  "r ~ 1 - v" reading: corr(r, v) <= -0.7 AND C1a fails.
  C1 PASSES (licenses 3C) iff C1a OR C1b passes.
PREDICTION: r is mostly a restatement of v: corr(r, v) <= -0.5, and both C1a and C1b FAIL.

C3 (value-estimator noise): 20 states = persuadee-ending prefixes of dialogues played in the frozen
  A_NoEmo_s20_seed1 run (20 dialogues, one random turn each, seed 20260925). heuristic() called 20
  times per state, current configuration. Report per-state sd, its distribution, pooled within-state
  sd, between-state sd of the state means, and noise share = within var / (within + between var).
  LARGE iff noise share >= 0.10 OR median per-state sd >= 0.10.  LARGE => Phase 3B is built.
PREDICTION: LARGE (10 samples of a 5-label reward at T 1.1).

T* (1D): 30 prefixes, one per dialogue, from the annotated 297 excluding P4G_BAD_DIALOGS (seed
  20260925), each cut after a persuader turn at a uniformly random turn t in [1, n-1], so the real
  human's next reply exists. Simulator = PersuadeeChatModel as built by build_agents (persona ON),
  10 replies per prefix per T in {0.3, 0.5, 0.7, 0.9, 1.1}; nothing else changes.
  Per prefix x T: sd of nu over the 10; modal-act share (share of the 10 replies carrying the most
  common user act); coverage = the human's nu inside [min, max] of the 10 simulated nu.
  Per T: donation rate (share of replies tagged [donate]); negative-affect rate (argmax of the hf
  distribution in {sadness, fear, anger, disgust}); mean reply length (words); distinct-2 over the
  30 replies with the same sample index, averaged over the 10 indices.
  Human targets: donation rate, negative-affect rate and mean length over ALL annotated persuadee
  turns (no decision cut: the simulator is asked at every turn, before and after a decision); distinct-2 over the 30 real next replies;
  coverage target 9/11 = 0.818 (the probability an exchangeable 11th draw lands inside the range
  of 10).
SELECTION RULE for T*: for each of the 5 criteria (donation rate, negative-affect rate, length,
  distinct-2, coverage) rank the temperatures by |value - human target|; T* = lowest mean rank.
  Tie -> smaller coverage gap; still tied -> the higher T (less risk of collapse).
  Whether any method separates at T plays no part.
MODE-COLLAPSE CHECK (reported for every T, decisive below 0.5): share of prefixes whose 10 replies
  contain <= 2 distinct strings; share of prefixes whose 10 replies all carry the same act.  If T* < 0.5 the program stops for a human call (brief, Phase 1 stop rule).
PREDICTION: T* in {0.7, 0.9}; T 1.1 over-disperses (coverage above target, distinct-2 above human).
```

---

## Entry 8 — 2026-09-25T16:24:43+02:00 — DEVIATION from Entry 7's T* rule: the readiness program runs at T = 1.1

Written after Phase 1D was measured (analysis/readiness/phase1.md) and before any Phase 2-4 run.
Decision by the human, 2026-09-25.

```
RULE OUTCOME (Entry 7): T* = 0.9, mean rank 2.0 vs 2.4 for 1.1.
WHY IT IS NOT USED:
  1. Tie. Over 1,000 bootstrap resamples of the 30 prefixes, 0.9 wins 39.8 % and 1.1 wins 39.9 %.
     0.9's lead rests on the two marginal rates (15 vs 17 donation replies, 34 vs 37 negative
     replies out of 300). 1.1 is closer on both dispersion criteria, which are measured more tightly.
  2. The rule's intent -- the temperature that best matches humans -- is met equally by 1.1.
  3. The simulator is UNDER-dispersed at every temperature, 1.1 included (coverage of the real
     reply 0.73 vs target 0.82; distinct-2 0.78 vs 0.85). Lowering T moves it further from humans
     on dispersion. Entry 7's prediction "1.1 over-disperses" was wrong.
  4. 1.1 keeps full comparability with the ~89 GPU-hours of the frozen grid, and removes the
     temperature as an environment change: Phase 4 then differs from the frozen grid only in the
     cache fixes and coupled seeds.
CONSEQUENCES:
  Simulator temperature 1.1 in every readiness run. tau stays 0.263 (the T = 1.1 median), so
  Phase 4A's re-derivation of tau is not needed and the brief's "never use 0.263 at another T"
  rule is not engaged. The brief's A1 conclusion is reported as: temperature is not the lever.
NOTED, NOT ACTED ON: an under-dispersed simulator is a reason the persona prompt could add
  realism (more varied replies), the opposite of the rationale it was introduced with. This
  changes what A2 would test; it is not part of this program.
```
