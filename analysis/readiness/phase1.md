# Readiness Phase 1 — cheap gates (C1a, C1b, C3, T\*)

Pre-registered in `PREREG.md` Entry 7 (2026-09-25T12:26, commit `5ddf79c`) before any measurement.
Frozen environment throughout (Vicuna-13B-AWQ / SGLang, top_p 0.9, top_k 40, persona ON, hf classifier,
generic valence); only 1D changes the simulator temperature. Scripts: `scripts/phase1_c1.py`,
`scripts/phase1_c3.py`, `scripts/phase1_temp.py`; numbers: `phase1_c1.json`, `phase1_c3_current.json`,
`phase1_temp.json`; raw rows: `data/`.

| Gate | Result | Verdict |
|---|---|---|
| **C1a** incremental R² of r over v | **0.017** [0.001, 0.052] | **FAIL** (below 0.02) |
| **C1b** incremental R² of r over n_turns | **0.0097** [0.0002, 0.040] | **FAIL** (below 0.02) |
| **C3** value-estimator noise share | **0.014**, median per-state sd **0.00** | **not LARGE** → 3B is not built |
| **T\*** | **0.9** by the rule; bootstrap winner 0.9 in 39.8 %, 1.1 in 39.9 % | chosen, **but indistinguishable from 1.1** |

**C1 fails → 3C (emotion-aware value) is not built. C3 not large → 3B is not built.**
T\* ≥ 0.5, so no stop.

---

## 1A / 1B — does an affective-risk judgement add anything? (C1a, C1b)

285 pre-decision prefixes (one per annotated dialogue with at least one pre-decision turn; donation rate 0.477;
persona found for 284). `v` = the value estimator as the planner runs it (10 samples, T 1.1, persona);
`r` = E[rating]/10 from a fixed observer prompt, read off the logits (label-set mass median 0.986).

| | value | 95 % CI |
|---|---|---|
| corr(r, v) | **−0.41** | [−0.51, −0.30] |
| corr(r, v_logit) | −0.45 | [−0.55, −0.34] |
| McFadden R², v alone | **0.002** | |
| McFadden R², v + r | 0.019 | |
| **C1a increment** (r over v) | **0.017** | [0.001, 0.052] |
| out-of-fold log-loss gain, r over v | +0.009 | [−0.010, +0.027] |
| AUC, v → v + r (out of fold) | 0.50 → 0.58 | |
| C1a with v_logit in place of v | 0.015 | [0.001, 0.050] |
| McFadden R², n_turns alone | 0.097 | |
| **C1b increment** (r over n_turns) | **0.0097** | [0.0002, 0.040] |
| out-of-fold log-loss gain, r over n_turns | +0.001 | [−0.013, +0.016] |
| reference: v over n_turns | 0.00005 | [0.000, 0.013] |

What it means:

- **r is not a restatement of v.** They correlate at only −0.41, well short of the −0.7 that
  pre-registration took to mean "r ≈ 1 − v". The PREREG prediction (corr ≤ −0.5) was wrong on this half.
- **But r does not clear the bar either.** Both increments sit under 0.02 and neither holds up out of
  fold. C1b's 0.0097 is **below** the mined trajectory features' 0.0125 on the same corpus. So an LLM
  asked directly about emotional risk finds no more than the classifier features did.
- **The finding that matters more: today's value estimator does not predict human donation at all.**
  On real pre-decision prefixes, `v` alone has McFadden R² 0.002 and out-of-fold AUC 0.50 (0.46 for
  v_logit). Turn count alone reaches 0.097. The value estimator scores how the *simulated* persuadee
  would answer "would you donate?" at this moment. That does not track what the human later did. Any
  affect term added to `v` would be refining a quantity that carries no outcome signal on human
  data. This bounds C2 more tightly than the gate does.

## 1C — where is the value estimator's noise? (C3)

20 persuadee-ending states from dialogues played in the frozen `A_NoEmo_s20_seed1` run, `heuristic()`
called 20× each, as configured for the grid.

| | value |
|---|---|
| per-state sd: median / mean / max | **0.00** / 0.039 / 0.177 |
| per-state sd: q25 / q75 | 0.00 / 0.072 |
| pooled within-state sd | 0.072 |
| between-state sd of state means | 0.602 |
| **noise share** (within / total variance) | **0.014** |

13 of 20 states return the same value on all 20 calls. In 9 of them the simulated persuadee had
already agreed (v = 1.0 every time). The rest have sd ≤ 0.18. **The estimator is not where the noise
is**: 1.4 % of the variance in `v` across states is re-query noise. The per-visit noise the planner
sees (within-pool ν sd 0.168, return variance ~7× that of z) comes from the **simulator's** reply draws.
The logit-scoring / low-T / state-cache fixes (3B) would remove a noise source that barely exists, so
they are not built (pre-registered rule).

Caveat: 20 states drawn from played dialogues over-represent late, already-decided states. The
non-degenerate states alone have sd 0.05–0.18, so a within-search mid-dialogue sample would be noisier than
0.014. It would still be well under the 0.10 threshold unless most search states are undecided.

## 1D — simulator temperature (A1)

30 human prefixes × 5 temperatures × 10 replies (persona ON; only the simulator's temperature changes).
Human targets: donation rate, negative-affect rate and length over all 2,994 annotated persuadee turns.
Distinct-2 over the 30 real next replies. Coverage target 9/11.

| T | donation rate | neg-affect rate | length (words) | distinct-2 | coverage | ν sd / prefix | modal-act share | ≤ 2 distinct strings | single act |
|---|---|---|---|---|---|---|---|---|---|
| **human** | **0.044** | **0.113** | **13.5** | **0.852** | **0.818** | — | — | — | — |
| 0.3 | 0.037 | 0.087 | 16.2 | 0.635 | 0.367 | 0.132 | 0.93 | 3 % | **73 %** |
| 0.5 | 0.060 | 0.130 | 17.0 | 0.660 | 0.400 | 0.169 | 0.85 | 3 % | 57 % |
| 0.7 | 0.067 | 0.150 | 16.6 | 0.697 | 0.533 | 0.200 | 0.78 | 0 % | 37 % |
| **0.9** | 0.050 | 0.113 | 17.9 | 0.724 | 0.600 | 0.231 | 0.79 | 0 % | 33 % |
| 1.1 (frozen) | 0.057 | 0.123 | 20.2 | 0.781 | 0.733 | 0.219 | 0.76 | 0 % | 27 % |

Rank on |gap to human| (1 = closest) and the pre-registered mean rank:

| T | donation | neg-affect | length | distinct-2 | coverage | **mean rank** | bootstrap wins |
|---|---|---|---|---|---|---|---|
| 0.3 | 2 | 4 | 1 | 5 | 5 | 3.4 | 2 % |
| 0.5 | 4 | 3 | 3 | 4 | 4 | 3.6 | 3 % |
| 0.7 | 5 | 5 | 2 | 3 | 3 | 3.6 | 15 % |
| **0.9** | **1** | **1** | 4 | 2 | 2 | **2.0** | **39.8 %** |
| 1.1 | 3 | 2 | 5 | 1 | 1 | 2.4 | 39.9 % |

**T\* = 0.9** under the pre-registered rule. Read it with three facts beside it:

1. **0.9 and 1.1 cannot be told apart on this sample.** Each wins about 40 % of 1,000 bootstrap resamples of
   the 30 prefixes. 0.9's lead comes from the two marginal rates, and those rest on 15 vs 17 donation
   replies and 34 vs 37 negative replies out of 300. 1.1 wins the two dispersion criteria, which are measured
   far more tightly.
2. **The simulator is under-dispersed at every temperature, including 1.1.** Coverage of the real reply
   is 0.73 against 0.82. Distinct-2 is 0.78 against 0.85. And 27 % of prefixes get the same act on all 10
   replies. The PREREG prediction "1.1 over-disperses" is **wrong**. Lowering the temperature moves the
   simulator further from humans on dispersion, and closer only on two noisy rates.
3. **Collapse below 0.5 is real.** At 0.3, 73 % of prefixes get a single act on all 10 replies and
   coverage is 0.37. The brief's warning holds.

Rule 6 is followed: T\* is chosen by realism alone, and whether methods separate played no part. But
the evidence for moving off the frozen 1.1 is thin. Phase 4 runs at T\* = 0.9 as the brief requires, and
the report treats "T = 0.9 vs 1.1" as a change whose realism gain is not established.

## Consequences for Phase 3

| Build | Condition | Built? |
|---|---|---|
| 3A coupled seeds | always | **yes** |
| 3B value-estimator variance fixes | C3 LARGE | **no** — noise share 0.014 |
| 3C emotion-aware value `v' = v − λr` | C1a or C1b passes | **no** — both fail |
| 3D partial pooling | always | **yes** |
