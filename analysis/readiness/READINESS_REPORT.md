# Readiness report — can affect-aware methods run on a fixed open-loop planner, and would it mean anything?

Branch `paper-fixes-fix-openloop`, one change per commit, nothing pushed. Pre-registration: `PREREG.md`
Entries 7–13. Phase write-ups are in this folder: `phase0.md` … `phase5b.md` and `value_fork.md`.

## 1. Verdict — 🔴 RED

**The open-loop fixes work, and there is almost no affective signal for them to deliver.**

The go/no-go rule is applied mechanically:
- **RED is triggered by Phase 3B.** Node-level emotional spread cannot tell actions apart. With C1a, C1b
  and the 1.3 % ω² ceiling, that makes four independent measurements finding little affective information.
- **The direct test agrees.** On the fully fixed planner, affect-keyed pooling does not beat act-keyed
  pooling: −0.026 [−0.068, +0.018] over 5 coupled seeds, all five leaning the other way.

What RED licenses: do not run the affect-method grid for success-rate claims. The result to report is the
infrastructure. The retention bug biased the value of reachable successes low by 0.12–0.15; the draw now
matches moods; coupled seeds more than double the correlation between arms; and self-play success
measures planner–simulator agreement more than persuasion. **These are results about open-loop MCTS for
dialogue and about self-play evaluation, independent of affect.**

## 2. The sim-to-real gap — qualifies everything below

The planner's value estimator `v` is the quantity every Q is built from (`value_fork.md`).

| Root `v` predicting… | AUC | 95 % CI |
|---|---|---|
| simulated success, pooled (1,800 grid dialogues, 18 runs) | **0.65** | [0.63, 0.67] |
| simulated success, fixed turns 3–6 | 0.67–0.72 | |
| human donation, the same fixed turns | 0.47–0.59 | |
| human donation, pooled | **0.53** | [0.48, 0.58] |

- **Inside the benchmark, `v` tracks the outcome:** all 18 runs fall between 0.59 and 0.77, adding
  0.057 [0.046, 0.070] McFadden R² over turn position.
- **For real people it is at most weakly informative.** One LLM plays value estimator and persuadee with
  the same persona, and their agreement barely carries over to human outcomes.
- **So self-play success rate measures planner–simulator agreement more than persuasion.** Even a GREEN
  verdict would have licensed claims about that agreement, not about persuading people.

**Caveats:**
- **The outcomes differ:** a `[donate]` tag within 10 turns vs an actual donation.
- **The dialogues differ:** planner-generated vs human-written.
- **A clean test does not exist here:** it would need human outcomes on planner-generated dialogues.
- **Turn count is withdrawn as a baseline:** at the decision cut it encodes when the person decided (AUC 0.71).

## 3. What was fixed

| Fix | Flag | Verified? | Evidence |
|---|---|---|---|
| **#6** retention: replies that end the dialogue stay drawable | `--cache_ended_children` | ✅ | Draw rate equals the rate in the pool (+0.005 [−0.009, +0.021]). Never-served replies drop to 0. 0 searches start from an ended state. Unit test 0.4037 vs 0.40. |
| **#5** depth 1 always generates | `--cache_fresh_depth1` | ✅ | 0 depth-1 cache hits (was 4,484 per 100 dialogues); generations = visits. |
| **#1** draw from the parent's mood bucket | `--cache_draw bucket --cache_bucket_tau 0.263` | ✅ on mood | The bucket invariant holds 100 %; 91 % of hits find a same-bucket reply. |
| **#2** mood-proximity weighting within the bucket | `--cache_draw bucket_kernel --cache_kernel_h 0.2` | ✅ on mood | Median mood gap between current and generating parent halves. |
| #1 + #2 on *exact* parent identity | — | ❌ (accepted) | With #5 on, 61 % of depth-≥2 draws have a parent that generated nothing in the pool. That is a floor no draw rule can beat. Accepted by the human as verified on mood (Entry 10). |
| **Coupled seeds** (common random numbers) | `--coupled_seeds --coupling_store` | ✅ | Same arm re-run: identical in 5/5, all 3,192 calls from the store. Two arms: identical up to the first differing act in 5/5. Built as a client-side reply store, because SGLang's seeded mode does not start on the 4090. |
| Step-log field telling ended replies apart | `child_ends_search` | ✅ | Write-only, draw-free; golden fingerprints unchanged. |

Every flag defaults off. With all flags off, the planner is bit-identical to the frozen one: 150 tests
pass and the golden fingerprints are unchanged.

## 4. What helped, and how much

"Before" is the frozen grid's run on the same dialogues and environment, unless marked otherwise. That is
a mechanical reference, not a paired baseline (Rule 10).

| Measure | Before | After | Δ | 95 % CI | Helped? |
|---|---|---|---|---|---|
| never-served realizations (NoEmo, 100 dlg) | 284 (D1 brief: 817) | **0** | −284 | — | ✅ |
| **Q on success-reachable edges** (#6, NoEmo s1) | 0.545 (old draw rule, same logs) | 0.697 | **+0.151** | [0.131, 0.173] | ✅ bias removed; +0.12–0.15 in all 5 runs |
| mood-bucket mismatch at cache hits (ActPool) | 0.239 | **0.094** | −0.145 | — | ✅ |
| median \|ν_generating − ν_current\| (ActPool) | 0.097 | **0.047** | −0.050 | — | ✅ |
| exact-parent mismatch, depth ≥ 2 (ActPool) | 0.725 | 0.745 | +0.02 | — | ❌ bounded by #5's floor (0.27 → 0.61) |
| depth-1 generations per edge (NoEmo) | 3.3 | **5.7** (= visits) | +2.4 | — | ✅ (the brief's "~10" was a different budget) |
| within-pool ν sd, served pools (NoEmo / ActPool) | 0.191 / 0.204 | 0.198 / 0.184 | ≈ 0 | — | not targeted: the fixes change the draw, not the pool |
| pools straddling τ (NoEmo / ActPool) | 0.57 / 0.58 | 0.63 / 0.55 | ≈ 0 | — | not targeted |
| **correlation between arms**, NoEmo vs ActPool, same dialogues | 0.17 [−0.03, 0.38] uncoupled | **0.39** [0.20, 0.59] coupled | +0.22 | — | ✅ partly (GREEN needed ≥ 0.6) |
| dialogues where the two arms never diverge | 3 % | 35 % | +32 pts | — | ✅ |
| detectable SR difference, one comparison, n = 100 | 0.18 | 0.14 | −0.04 | 0.12–0.16 | ✅ partly |
| detectable SR difference, 5 coupled seeds | — | **0.062** | — | — | ✅ |
| NoEmo seed spread, s20 | 0.65 / 0.68 / 0.76 (uncoupled) | 0.68 / 0.75 (2 coupled seeds) | — | diff −0.07 [−0.20, +0.05] | no change expected (different seeds share nothing) |
| value-estimator per-state sd | — | median **0.00**; noise share 0.014 | — | — | the estimator was never the noise source |
| **AffPool – ActPool homogeneity gap** | +0.004 [−0.041, +0.042] | **+0.021** [+0.003, +0.043] | +0.017 | CIs overlap | ✅ small; "gap closes" test missed by 0.001 |
| **AffPool – ActPool SR** (all fixes, 5 coupled seeds) | — | **−0.026** | — | [−0.068, +0.018] | not detected |
| `w` by depth (partial pooling) | — | not run | — | — | 6B not built (3B failed) |

## 5. What was not detected — and at what resolution

- **AffPool vs ActPool, success rate:** −0.026 [−0.068, +0.018], with the design resolving ~0.062.
  An AffPool advantage larger than +0.018 is excluded. All 5 seeds lean toward ActPool (sign test p ≈ 0.06).
- **AffPool vs ActPool, AvgT:** +0.10 turns [−0.09, +0.30].
- **Homogeneity gap, before vs after:** a change is not established, because the CIs overlap. The
  after-gap alone does exclude zero.
- **Retention (#6) on success rate:** **not measured.** The NoEmo ± #6 and GDP-Zero ± #6 comparisons were
  stopped after 5 dialogues, by human decision, in favour of the direct affect test. #6's effect is
  established at decision level only (Q +0.12–0.15), not on success rate, and not on GDP-Zero's planner.
- **Temperature:** 0.9 and 1.1 are indistinguishable on 30 prefixes (each wins ~40 % of resamples).

None of these is "no effect". Each is bounded by its interval.

## 6. Affect-signal evidence, all in one place

| Measurement | What it asks | Result | Bar | Verdict |
|---|---|---|---|---|
| **C1a** | does an LLM judgement of emotional risk `r` add to `v` in predicting human donation? | 0.017 [0.001, 0.052] McFadden | 0.02 | fail: non-zero, practically small |
| **C1b** | does `r` add over turn count? | 0.0097 [0.0002, 0.040] | 0.02 | fail (and the turn-count baseline leaks) |
| **ω² ceiling** | how much of within-edge return variance does mood explain, pairing corrected? | 0.0126 [0.010, 0.022] | — | ~1.3 % |
| **3B node spread** | does the spread of mood across a node's replies separate actions? | ΔR² 0.008; sibling ratio 1.04 (p 0.050) | ratio 1.2 | fail: noise-level |
| **5b direct** | does affect-keyed pooling beat act-keyed pooling on the fixed planner? | SR −0.026 [−0.068, +0.018]; homogeneity gap +0.021 [0.003, 0.043] | — | not detected in outcomes; tiny in structure |

Five measurements from four directions (a classifier, an LLM judge, the tree's own statistics, a
controlled comparison) agree. **Affect carries about 1 % of the relevant variance here.** At that size,
splitting evidence by mood costs more than it returns.

## 7. The recommended next step

1. **Do not run the affect-method grid for success-rate claims.** Its best case, AffPool on the fully fixed
   planner, was run directly and did not separate. The methods that depend on edge-averaged Q_emo (Bias,
   CenteredBias) are unaffected by the draw fix and remain marginal over parent moods. Momentum's mood
   change is now mood-consistent but rarely from the same parent.
2. **Report the infrastructure findings**, which stand regardless:
   - **(a) The retention bug** in GDP-Zero's reference open-loop cache biased the value of reachable
     successes by −0.12 to −0.15.
   - **(b) Mood-matched drawing** removes most mood scrambling at cache hits.
   - **(c) Coupled seeds** as a reply store: correlation between arms 0.17 → 0.39.
   - **(d) The sim-to-real gap** in self-play evaluation.
3. **If a success-rate claim about #6 is wanted:** resume Entry 12 — NoEmo ± #6 and GDP-Zero ± #6, s20,
   5 coupled seeds, ~58 GPU-h. It is the one comparison where a real change is plausible, since `v` is
   informative inside the benchmark and the bias is measured.
4. **If affect is pursued at all**, the signal must come from somewhere else first:
   - a value estimator that predicts human outcomes, or
   - a simulator whose emotional reactions carry more than 1 % of return variance.

   Checking the persona prompt as a realism lever (A2) fits here, since the simulator is under-dispersed.

## 8. Caveats

- **Single backbone** (Vicuna-13B-AWQ) and a single value prompt.
- **Annotation:** author-only for the corpus acts. The hedge detector was fitted on the turns it was
  tested on.
- **T\* came from 30 prefixes** and tied with 1.1. The program runs at 1.1 by declared deviation
  (Entry 8), and the simulator is less varied than humans at every temperature.
- **The sim-to-real gap (§2)** bounds every success-rate number here.
- **Coupling is a client-side reply store, not a server seed.** It is valid as common random numbers, but
  its reach ends at the first divergent act: 29 % of dialogues diverge at turn 1.
- **#6's Q shift is a first-order counterfactual:** it re-weights draws and does not replay the search.
- **The 5b "before" homogeneity comes from one frozen run;** "after" pools five. Their CIs differ in width.
- **Phase 2 verification used 25 dialogues;** its conclusions were re-measured at 100 in `report_pools.json`.

## 9. Pilot–population gaps observed

| Pilot | Pilot result | Population result |
|---|---|---|
| episode-horizon pilot, SR (earlier) | 10/10 | 0.78 at n = 100 |
| episode vs legacy horizon (earlier) | 10/10 vs 6/10 | no difference at n = 100 |
| episode-horizon wall time (earlier) | 42 % faster | 4 % |
| 3A coupling acceptance (this program) | arms diverge by turn 2 in 5/5 dialogues | 35 % never diverge at n = 100 |
| Phase 4 prediction built on that pilot | corr < 0.3 | 0.39 |
| Phase 2 bucket-hit expectation ("~69 %") | — | 0.92 (the 69 % was a straddle rate, not a hit rate) |
| brief's "0.02–0.09" correlation between arms | — | 0.17 for the NoEmo/ActPool pair, uncoupled |
| brief's AffPool/ActPool homogeneity gap (D1, legacy) | 0.046 vs 0.073 | 0.004 in the current environment, before the fixes |
