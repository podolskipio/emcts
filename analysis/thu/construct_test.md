# TASK 2 — Construct test: resistance vs emotion

**Answer: resistance does not beat emotion. No escalation; keep ν.** On the valid, leakage-free test
the lexical resistance signal is **worse than emotion on every metric** (point estimates), and no
difference is distinguishable from zero. It wins in only 8–37 % of bootstrap resamples.

| outcome row (brief §2.3) | verdict |
|---|---|
| M2 > M1 clearly | **no** |
| M2 ≈ M1 | CIs straddle zero on every metric — consistent |
| M2 < M1 | every point estimate favours emotion — consistent |

The honest reading sits between the last two rows: **emotion is at least as good, probably slightly
better, and the sample cannot separate them.** The "positive result" the brief hoped for is not here.

**Two findings matter more than the headline, and one of them would have produced a false positive:**

1. **⚠ Without a leakage cut, tuned resistance "wins".** Scored on whole dialogues, tuned resistance
   beats emotion (ΔAUC +0.032, ΔR² +0.042). That is the persuadee's decision leaking into the
   predictor: the commitment family matches "I'll donate". Cut at the decision turn, the advantage
   disappears and reverses. **A naive version of this test reports the positive result, and it is
   spurious.**
2. **The persuasion-literature theory mostly fails on this corpus.** Of seven marker families, three
   have the **opposite** sign to the theory on the tuning split, two carry no signal, and two match.
   Question-asking — theorised as receptivity — is the strongest single predictor of *not* donating.

Sources: 297 annotated P4G dialogues (`data/p4g/300_dialog_turn_based.pkl`, minus the 3 content-filtered
ids). Machine-readable: `construct_test.json`, `marker_lists.json`, `user_turns.parquet`. Scripts:
`scripts/t2_build_turns.py`, `scripts/t2_markers.py`, `scripts/t2_construct_test.py`.

---

## 1. Design, fixed before the test fold was read

| | |
|---|---|
| **tune** | `mine200` — 197 dialogues. Marker lists were inspected **here only**. `w(e)`, and so ν, was also mined here. |
| **test** | `test100` — the held-out fold from `build_corpus_turns.py`, the same one the momentum test uses. On it, **both constructs are out of sample.** |
| **eval 100** | never touched. The annotated 300 are disjoint from it (asserted in `build_corpus_turns.py`). |
| **leakage cut** | features use only user turns **strictly before** the dialogue's first explicit decision act: `agree-donation`, `disagree-donation`, `disagree-donation-more`, `provide-donation-amount`, `confirm-donation`. Symmetric: donors lose "yes", refusers lose "no". The act labels locate the cut and are never a predictor. |
| **models** | logit(`donated`), predictors standardized. M0 `n_turns`; M1 +`nu_final`,`nu_mean`; M2 +`r_final`,`r_mean`; M3 all four. **M1 and M2 have equal degrees of freedom (k = 4).** |
| **metrics** | McFadden pseudo-R² (full test fit); out-of-fold Brier and AUC (stratified 5-fold, repeated 20×); 95 % CIs from 1000 bootstrap resamples over dialogues, **paired** across models. |

**About `TrajValue Gate A`'s split.** The brief says to use "the same held-out split used for
TrajValue's Gate A". `agent_task_trajvalue.md`, which would define it, **is not in the repository.**
I used `test100` / `mine200`, the only held-out split the corpus pipeline defines. If Gate A's spec
names a different split, this test should be re-run on it.

**ν is verified identical to the project's.** The user-turn table re-classifies with the same
DistilRoBERTa and reproduces `corpus_turns.user_nu_w200` on all **2,780** matched rows, max
|deviation| **1.4 × 10⁻⁶** (float32 rounding).

**Two versions of `r`**, so that a win cannot be credited to theory when it came from tuning:

- **`r_prior`** — the brief's lists at the brief's signs (receptivity +1, resistance −1). No tuning.
- **`r_tuned`** — each family re-signed by one mechanical rule on `mine200`: weight = sign(mean hits
  per turn in donors − non-donors), set to 0 if that gap is below 0.005. No hand-picking.

The leakage cut keeps **1,746 of 2,998** user turns. 264 of 297 dialogues have a decision act.
**5 of the 100 test dialogues** open with a decision and have no earlier turn, so the valid test runs
on **n = 95** (donation rate 0.432).

## 2. Result — pre-decision turns (the valid test)

| model | k | McFadden R² | OOF Brier ↓ | OOF AUC |
|---|---|---|---|---|
| M0 `n_turns` | 2 | 0.129 [0.040, 0.265] | 0.210 | 0.706 [0.596, 0.806] |
| **M1 emotion** | 4 | **0.175** [0.080, 0.366] | **0.201** | **0.750** [0.646, 0.849] |
| M2 resistance (prior) | 4 | 0.144 [0.070, 0.313] | 0.220 | 0.702 [0.599, 0.803] |
| M2 resistance (tuned) | 4 | 0.155 [0.072, 0.323] | 0.210 | 0.720 [0.616, 0.822] |
| M3 both (prior) | 6 | 0.187 [0.107, 0.411] | 0.209 | 0.740 [0.641, 0.841] |
| M3 both (tuned) | 6 | 0.210 [0.128, 0.427] | 0.207 | 0.740 [0.639, 0.841] |

**The comparison: M2 − M1, equal df, paired bootstrap.**

| | ΔR² | ΔAUC | ΔBrier (+ = worse) | resamples where resistance wins (R² / Brier / AUC) |
|---|---|---|---|---|
| prior − emotion | −0.031 [−0.159, +0.086] | −0.048 [−0.129, +0.023] | +0.018 [−0.007, +0.047] | 29 % / 8 % / 9 % |
| tuned − emotion | −0.020 [−0.155, +0.105] | −0.030 [−0.123, +0.055] | +0.009 [−0.022, +0.043] | 37 % / 28 % / 24 % |

Every point estimate favours emotion. Every interval contains zero.

**M3 — complementary or redundant?** Adding tuned resistance to emotion raises in-sample R² by
**+0.035** but moves out-of-fold AUC by **−0.009** and Brier by **+0.005**. An in-sample gain that
disappears out of fold, from 2 extra parameters on 95 dialogues, is overfitting, not complementarity.
**The two are not usefully complementary at this sample size.**

**Neither construct adds much beyond turn count.** `n_turns` alone reaches AUC 0.706. Emotion adds
**+0.043 AUC** on top of it. Tuned resistance adds **+0.013**, and prior resistance adds **−0.004**.

## 3. The leaky version — why the cut is not optional

The same models on **whole dialogues**, including the decision turn and everything after it:

| model | McFadden R² | OOF AUC |
|---|---|---|
| M0 `n_turns` | 0.042 | **0.405** |
| M1 emotion | 0.101 | 0.664 |
| M2 resistance (prior) | 0.051 | 0.529 |
| **M2 resistance (tuned)** | **0.142** | **0.696** |

| | ΔR² | ΔAUC |
|---|---|---|
| **tuned − emotion, LEAKY** | **+0.042** [−0.093, +0.175] | **+0.032** [−0.114, +0.169] |
| tuned − emotion, valid | −0.020 | −0.030 |

**On whole dialogues tuned resistance comes out ahead, and that is the version a quick test would
report.** The advantage comes from post-decision text: the `commitment` family matches "I'll donate",
"sounds good", "definitely". Those words report the outcome; they do not predict it. Emotion leaks
too, but less, because a classifier does not key on the literal agreement.

`n_turns` shows the same contamination. On whole dialogues its AUC is **0.405**, below chance: donors
agree and wrap up, refusers keep talking. Cut at the decision, it is **0.706**. Any per-dialogue
trajectory feature in TrajValue that is computed over the full dialogue carries this, so **Gate A
should apply the same cut.**

## 3b. Leakage, three windows side by side (added 2026-09-16, human-specified cut)

Every feature, the `r_tuned` sign rule and the `w(e)` re-mine are computed on the **same window**, for
both constructs. Script `scripts/t2_leakage.py`, output `leakage.json`.

| window | definition |
|---|---|
| **full** | every user turn. This is leaky, and it is what `w(e)` was mined on. |
| **commit cut** (human-specified) | turns strictly before the first turn with commitment language (a `commitment`-family hit). If there is none, turns strictly before the final user turn. **107 of 297 dialogues have no commitment turn** and fall to the final-turn rule. |
| **decision cut** (§1) | turns strictly before the first annotated decision act. If there is none, every turn. |

### What each window leaves in — the audit

Share of dialogues whose **kept** turns still contain an annotated decision act:

| | full | **commit cut** | decision cut |
|---|---|---|---|
| donors keeping `agree` / `amount` / `confirm` | 100 % | **52 %** | 0 % |
| non-donors keeping `amount` / `confirm` | 58 % | 24 % | 0 % |
| non-donors keeping `disagree` | 27 % | **15 %** | 0 % |
| kept turns with any decision act, donors / non-donors | 22.0 % / 11.7 % | **15.0 % / 7.0 %** | 0 / 0 |
| kept turns per dialogue, donors / non-donors | 10.1 / 10.1 | 6.4 / 7.1 | **5.1 / 7.1** |

**⚠ The commit cut removes commitment *words*, not the decision.** Half of donors still keep the turn
where they agree, because many agreements carry no commitment marker. "Sure, $1" and "yes, take it
from my bonus" do not match. "Put me down for 50 cents" does. 15 % of refusers keep an
explicit refusal. The cut is symmetric in rule but not in effect, since it removes more outcome text
from donors (22 % → 15 %) than refusers keep. **The decision cut is the only window with zero decision
acts on both sides.** It has its own asymmetry: donors decide earlier, so they keep fewer turns (5.1
against 7.1), and `n_turns` absorbs decision timing. That is why M0 alone reaches AUC 0.706 there. What
matters in that window is the **increment over M0**, not absolute AUC.

### Models on test100 (ν from the shipped `w200`)

| | full (n 100) | commit cut (n 96) | decision cut (n 95) |
|---|---|---|---|
| M0 `n_turns` — AUC | 0.405 | 0.551 | 0.706 |
| **M1 emotion** — AUC / R² | **0.664** / 0.101 | **0.654** / 0.075 | **0.750** / 0.175 |
| M2 resistance, prior — AUC | 0.529 | 0.603 | 0.702 |
| **M2 resistance, tuned** — AUC / R² | **0.733** / 0.174 | **0.500** / 0.041 | **0.720** / 0.155 |
| M1 − M0, ΔR² | +0.059 | +0.034 | +0.046 |
| **tuned M2 − M1, ΔAUC** | **+0.069** [−0.067, +0.205] | **−0.154** [−0.254, −0.059] | −0.030 [−0.123, +0.055] |
| tuned M2 − M1, ΔR² | +0.073 | −0.034 | −0.020 |
| resamples where tuned resistance wins (AUC) | **85 %** | **0 %** | 24 % |

(The full-window tuned-resistance numbers differ from §3's because `r_tuned` is now tuned on the same
window it is scored on, not on pre-decision turns. That is the identical-treatment requirement.)

**The finding: outcome leakage inflates the lexical signal, not the affective one.**

- **Tuned resistance collapses without outcome text:** AUC 0.733 → **0.500** under the commit cut. The
  "win" over emotion on full dialogues (resistance ahead in 85 % of resamples) becomes a significant
  **loss** (ΔAUC −0.154, CI excludes 0). Under the decision cut it is a tie.
- **Emotion is nearly leakage-invariant:** AUC 0.664 full, 0.654 commit cut, 0.750 decision cut (the
  last on a stronger M0 baseline). Emotion's increment over turn count stays in a narrow band, ΔR²
  +0.034 to +0.059, in every window. **"Outcome leakage inflates apparent affective prediction by X" —
  on this corpus X ≈ 0 for ν.** The inflation is in lexical prediction: +0.23 AUC for tuned resistance.
- So the headline of §2 **holds under every window**: resistance never beats emotion once outcome text
  is removed, and under the human-specified cut emotion wins outright.

### Is `w(e)` fitted to outcome language? Partly — for happiness.

`w(e)` re-mined on `mine200` in each window, with `mine_emotion_donation_p4g.py`'s formula
(w = 10 · n/(n+50) · lift). **Check against the shipped table:** my full-window re-implementation
reproduces happiness (0.44 against 0.41) and neutral (−0.08 against −0.12). The thin cells differ: fear
−0.30 / −0.10, anger −0.05 / +0.11, surprise −0.20 / +0.04, disgust +0.15 / +0.04. That is the known
instability of those cells (`mine_emotion_donation_p4g.py`: "w(fear) from +1.07 to +0.01" under
resampling). My turn segmentation also differs: 1,986 user turns against the miner's 3,175 units. **The
reproduction is approximate, not exact**, so compare windows within my implementation, below.

**One correction is needed to compare windows.** The miner measures lift against the **dialogue-level**
donation rate. Once a cut removes more donor turns than refuser turns, the turn-weighted rate drops
(0.508 → 0.429 under the decision cut) and **every** weight goes negative. That is a base-rate artefact,
not an affect effect. The weights below use the turn-weighted base rate and are shown relative to
neutral, which is what ν's ranking depends on:

| w(e) − w(neutral), mine200 | happiness | sadness | fear | anger | surprise | disgust |
|---|---|---|---|---|---|---|
| full | **+0.52** | −0.23 | −0.22 | +0.02 | −0.12 | +0.23 |
| commit cut | **+0.63** | −0.01 | −0.27 | −0.02 | −0.05 | +0.16 |
| **decision cut** | **+0.17** | **−0.65** | −0.15 | +0.01 | −0.31 | +0.23 |

P(donate | happiness-weighted turn) is 0.557 on full turns and **0.460** before the decision, against
0.500 and 0.437 for neutral.

- **About two-thirds of happiness's valence weight comes from turns at or after the decision**
  (+0.52 → +0.17 relative to neutral). Donors are happy *when they agree*. Before deciding, happy users
  are only slightly more likely to donate.
- **Sadness goes the other way:** −0.23 → **−0.65**. Before the decision, sadness is the strongest
  negative predictor, and post-decision turns dilute it. The shipped table has sadness at −0.39.
- **The commit cut leaves the happiness weight intact** (+0.63), consistent with the audit: it keeps
  half of donors' agreement turns.
- **Re-mining on clean turns does not improve prediction.** M1 with each window's own `w(e)` scores AUC
  0.650 / 0.588 / 0.723 (full / commit / decision), below the shipped `w200`'s 0.664 / 0.654 / 0.750.
  The shipped weights are partly outcome-fitted, but that does not make ν a worse predictor of donation
  on held-out dialogues. What it changes is **what ν means**: a planner steering towards the shipped
  `w(happiness)` is partly steering towards "the user sounds like someone who has just agreed".

**Reported, not decided:** whether to re-mine the deployed table on pre-decision turns is a construct
question, and the prediction numbers do not settle it. If ν is meant to be "affect that precedes
donation", the decision-cut table is the defensible one, and it would change the planner (happiness
weight down by two-thirds, sadness up by half). It would also invalidate the frozen w200-based
constants (`τ_med`, AffPool's τ 0.35), so it is a pre-freeze call.

## 4. Are `r` and `ν` the same construct measured twice? No.

Utterance-level correlation, pre-decision turns:

| | n | Pearson(ν, r_prior) | Spearman | Pearson(ν, r_tuned) | Spearman |
|---|---|---|---|---|---|
| test100 | 593 | 0.037 | 0.051 | 0.016 | −0.030 |
| mine200 | 1,153 | −0.019 | 0.003 | 0.076 | 0.049 |

|r| < 0.08 everywhere. **The two signals are essentially orthogonal.** So the tie is not two readings of
one quantity: they measure different things and predict donation about equally weakly. One caveat is
that `r` is zero on ~70 % of utterances (non-zero share 0.28–0.30), so the correlation is computed
mostly against a sparse signal.

## 5. The theory, tested on the tuning split

Donors minus non-donors, mean hits per pre-decision turn, `mine200`:

| family | theory | observed gap | tuned sign | vs theory |
|---|---|---|---|---|
| counterargument ("but", "however") | resistance − | **+0.0355** | **+** | ❌ **flipped** |
| deflection ("maybe later") | resistance − | +0.0033 | 0 | dropped (below 0.005) |
| source doubt ("scam", "overhead") | resistance − | −0.0182 | − | ✅ |
| hedge ("I guess", "probably") | resistance − | +0.0011 | 0 | dropped |
| **cause question** | receptivity + | **−0.0578** | **−** | ❌ **flipped, strongest** |
| self-disclosure ("my family") | receptivity + | −0.0075 | − | ❌ flipped |
| commitment ("I will") | receptivity + | +0.0166 | + | ✅ |

**Question-asking is the largest single effect, and it runs opposite to the theory.** Before deciding,
non-donors ask more questions about the cause — "how much actually goes to the kids?" is scrutiny,
not receptivity. "But" goes with donors, probably because engaged people qualify their statements
while disengaged people give short replies. **On P4G, the literature's resistance/receptivity split
does not map onto donation with the signs it proposes.** This is itself reportable.

Deflection fires on only ~1 % of turns. The P4G persuadees are crowdworkers paid to talk, and they
rarely put the persuader off. The construct has little to grip on in this corpus.

## 6. What this means — reported, not decided

- **Keep ν.** No swap. §2.4's "one day to swap" is not triggered.
- **Put the leakage cut into TrajValue Gate A before its table is frozen** (§3). This is the actionable
  finding. A Gate A over full dialogues can pass on post-decision text.
- **The brief's "ν explains 0.001 of task return" and this result are not in conflict.** That figure
  is about the planner's search-time task return; this is about corpus donation. Here ν adds **+0.043
  AUC** over turn count on held-out dialogues. That is small but not nothing, and it is **a ceiling on
  the task, not on the label set** (brief §2.3, row 3). A cruder signal did no better.
- Per brief Task 6: resistance did **not** win, so Gate A does **not** need re-running with `r`.

---

## Appendix — marker lists, verbatim

Regexes, case-insensitive, counted as number of matches (`scripts/t2_markers.py`):

    r(utterance) = Σ_f  w_f · hits_f  /  (1 + tokens/10)

w_f = +1 receptivity / −1 resistance for `r_prior`; the tuned column in §5 for `r_tuned`.

**Resistance**

- **counterargument:** `\bbut\b` `\bhowever\b` `\balthough\b` `\bthough\b` `\bnevertheless\b`
  `\bon the other hand\b` `\bthat said\b` `\beven so\b`
- **deflection:** `\bmaybe later\b` `\bi'?ll think about it\b` `\bnot (?:right )?now\b`
  `\bsome other time\b` `\banother time\b` `\bi'?ll (?:have to )?(?:consider|look into)\b`
  `\bnot (?:really )?(?:interested|sure i)\b` `\bi (?:can'?t|cannot) (?:afford|right now)\b`
  `\bi'?m (?:a bit )?(?:tight|broke)\b` `\bdon'?t have (?:the )?(?:money|funds|cash)\b`
  `\bi'?ll pass\b` `\bno thank(?:s| you)\b`
- **source_doubt:** `\bhow do i know\b` `\bare you sure\b` `\bscam\b` `\bfraud\b` `\blegit(?:imate)?\b`
  `\btrust(?:worthy)?\b` `\bskeptic(?:al)?\b` `\bsceptic(?:al)?\b`
  `\bhow much (?:of it |actually )?goes\b` `\bwhere does the money go\b` `\boverhead\b`
  `\badministrat(?:ive|ion) costs?\b` `\bproof\b` `\bverify\b`
- **hedge:** `\bi guess\b` `\bsort of\b` `\bkind of\b` `\bkinda\b` `\bi suppose\b` `\bprobably\b`
  `\bperhaps\b` `\bmight\b` `\bnot sure\b` `\bi dunno\b` `\bi don'?t know\b`

**Receptivity**

- **cause_question:**
  `\b(?:how|where|what|who|which)\b[^?]{0,80}\b(?:charity|children|donate|donation|fund|money|organi[sz]ation|save the children|help|support|cause)\b[^?]{0,40}\?`
  `\bhow (?:can|do) i (?:help|donate|give|contribute)\b` `\btell me more\b` `\bi'?d like to (?:know|hear) more\b`
- **self_disclosure:** `\bi once\b`
  `\bmy (?:family|mother|father|mom|dad|son|daughter|kids?|children|wife|husband|brother|sister|friend)\b`
  `\bi (?:have|had) (?:a |an )?(?:child|kids?|son|daughter)\b` `\bi work (?:with|for|at)\b`
  `\bi volunteer\b` `\bwhen i was\b` `\bin my (?:own )?experience\b`
  `\bi (?:also )?(?:donate|give|support)(?:d|s)? (?:to|regularly|monthly|every)\b` `\bgrew up\b` `\bi'?ve been\b`
- **commitment:** `\bi will\b` `\bi'?ll donate\b` `\bi'?d like to\b` `\bsign me up\b` `\bcount me in\b`
  `\bi'?m in\b` `\bi want to (?:help|donate|give|contribute)\b` `\bi can (?:donate|give|do)\b`
  `\bput me down\b` `\bhappy to (?:help|donate|give)\b` `\bi'?ll give\b` `\bsounds good\b`
  `\bdefinitely\b` `\babsolutely\b` `\ball of it\b` `\bthe (?:full|whole) (?:amount|two dollars|\$2)\b`
