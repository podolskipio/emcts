# C4 — Classifier robustness: do other emotion classifiers agree with DistilRoBERTa?

Re-labelling only. No text was generated, no MCTS was run, nothing under `src/` was touched,
nothing was committed. All code and data are under `c4/`.

Input: `spike/generations.jsonl` — **8,820** records (not 9,000; `spike/RESULTS.md` §11.1 records
that 2 of 100 dialogues had no persuadee survey row, giving 98 personas × 5 utterances × 3 seeds
× 2 arms × 3 models).

---

## 1. Verdict: ⚠️ **NOT ROBUST** — do not start the grid

The decision table's third row is triggered, twice and in opposite directions.

| Gate | Threshold | Observed | Met? |
|---|---|---|---|
| No classifier reverses a verdict | — | **alt2 reverses two**: Llama (WEAK) → 1.17× [1.07, 1.28]; Qwen (PASS) → 1.00× [0.90, 1.10] | ❌ |
| Cohen's κ | ≥ ~0.6 | **0.21**, **0.21**, **0.11** | ❌ |
| MICA cosine of score-change vectors | ≥ ~0.78 | **0.27**, **0.31**, **−0.02** (pooled) | ❌ |
| Qwen > 1.0 and Vicuna/Llama < 1.0 under all | — | holds under alt1 only | ❌ |

Recomputing the spike's own verdict rule end-to-end (permutation gate + `n_sig` + raw-ratio
threshold, `spike/phase567_analysis.py` logic, only `emo_dist` varied):

| Classifier | vicuna-13b | Llama-3.1-8B | Qwen2.5-7B | "models passing" |
|---|---|---|---|---|
| **DistilRoBERTa** (original) | WEAK | WEAK | **PASS** | 1 of 3 |
| **alt1** GoEmotions | WEAK | WEAK | **WEAK** | **0 of 3** |
| **alt2** EmoBERTa | WEAK | **PASS** | **PASS** | **2 of 3 — and a different set** |

**The spike's headline claim — "Qwen2.5-7B is the one model that responds to personas
emotionally, so use it as the simulator" — is a property of `j-hartmann/emotion-english-distilroberta-base`,
not a property of the simulators.** One alternative demotes Qwen to WEAK. The other promotes
Llama, the model the spike ranked *lowest* (0.68×), to PASS.

### What survives

- **The manipulation check is fully classifier-invariant.** Permutation p = 0.0000 for every
  model under every classifier. That personas change the simulator's output *at all* is solid.
- **H2 (agreeableness → happiness) is robust**: significant in 8 of 9 classifier × model cells,
  always with the predicted sign. It was already the spike's strongest hypothesis.
- **`P(neg)`, the quantity `Q_emo` actually consumes, agrees far better than top-1 labels do**:
  Pearson r = 0.66–0.79 where κ is only 0.11–0.21. This is a real partial defence of the method
  and is the one place the news is better than the headline. But r ≈ 0.7 is ~50% shared variance,
  and empirically that is not enough to stabilise the ranking — as the table above shows.

### What this means for the grid

Do not spend ~230 GPU-hours on a simulator chosen by a criterion that does not survive changing
the labeller. Section 9 sets out the options; the cheapest defensible one is to **re-run the
directional analysis under a committee of classifiers and pick the simulator on cross-classifier
agreement**, which costs no GPU generation because the text already exists.

---

## 2. The two classifiers (Phase 1)

Selection was written to `c4/SELECTION.md` **before** Phase 2 was run, with the rejection reasons
for the candidates not used. **Exactly two alternatives were run. No third classifier was run and
discarded.** The one variation reported beyond those two is a *mapping* sensitivity on alt1
(§2.1), computed from the same forward pass, not a different model.

| | reference | alt1 | alt2 |
|---|---|---|---|
| Model ID | `j-hartmann/emotion-english-distilroberta-base` | `SamLowe/roberta-base-go_emotions` | `tae898/emoberta-large` |
| Architecture | DistilRoBERTa-base (82M) | RoBERTa-base (125M) | **RoBERTa-large (355M)** |
| Head | single-label softmax | **multi-label, 28 sigmoids** | single-label softmax |
| Training data | 6 mixed English emotion sets | **GoEmotions** (58k Reddit, 27+neutral) | **MELD + IEMOCAP** (dialogue) |
| Labels | Ekman-7 | 28 → mapped | **native Ekman-7** |
| Differentiation axis | — | training corpus | capacity + dialogue domain |

alt2 was chosen deliberately as the dialogue-domain check, since the spike's text is dialogue
turns; alt1 as the "different corpus entirely" check. Both emit full score vectors, never top-1.
`joy → happiness` rename only, matching spike convention.

### GoEmotions → Ekman-7 mapping (alt1)

**Not invented here.** Taken verbatim from Google Research's own published grouping,
`google-research/goemotions/data/ekman_mapping.json`, plus `neutral → neutral`. Every one of the
27 emotions is assigned exactly once; none dropped.

| Ekman-7 | GoEmotions labels folded in | n |
|---|---|---|
| anger | anger, annoyance, disapproval | 3 |
| disgust | disgust | 1 |
| fear | fear, **nervousness** | 2 |
| happiness | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring | 12 |
| sadness | sadness, disappointment, embarrassment, grief, remorse | 5 |
| surprise | surprise, realization, confusion, curiosity | 4 |
| neutral | neutral | 1 |

The judgement calls the task anticipated (*annoyance* → anger, *nervousness* → fear) are Google's,
not mine, which is why this mapping was preferred over a hand-built one.

**Sigmoid → distribution.** alt1's 28 sigmoid scores do not sum to 1. Primary procedure: sum
within each Ekman group, renormalise the 7 sums to 1.

### 2.1 Mapping sensitivity — the group sizes are unequal, so this was checked

The groups are badly unbalanced (happiness 12 labels, neutral 1), so a group-**sum** collapse is
size-biased toward happiness and against neutral. To rule out that the disagreement was
manufactured by my own aggregation, the raw 28-dim scores were stored (`c4/alt1_raw_goemotions.jsonl`)
and a size-neutral group-**max** collapse (`alt1b`) evaluated from the same forward pass:

| | neutral mass | happiness mass | Qwen std. ratio | κ vs DistilRoBERTa |
|---|---|---|---|---|
| alt1 (sum) | 0.086 | 0.524 | 1.33× [1.19, 1.50] | 0.212 |
| alt1b (max) | 0.113 | 0.475 | 1.40× [1.24, 1.58] | 0.226 |

The collapse choice moves nothing material. **GoEmotions genuinely almost never fires `neutral`
on this text** — that is the model, not the aggregation. alt1 (sum) is used throughout below;
alt1b is reported alongside in the Phase 4 table.

---

## 3. Agreement statistics (Phase 3)

### 3.1 Overall (all 8,820 records)

| Pair | top-1 acc | **Cohen's κ** | mean JSD | `P(neg)` Pearson r | `P(neg)` Spearman ρ |
|---|---|---|---|---|---|
| DistilRoBERTa ↔ alt1 | 0.369 | **0.212** | 0.367 | **+0.793** | +0.698 |
| DistilRoBERTa ↔ alt2 | 0.533 | **0.208** | 0.285 | **+0.661** | +0.571 |
| alt1 ↔ alt2 | 0.188 | **0.108** | 0.529 | **+0.769** | +0.627 |
| DistilRoBERTa ↔ alt1b *(mapping check)* | 0.379 | 0.226 | 0.348 | +0.788 | +0.706 |

κ = 0.11–0.23 is "slight to fair" agreement — nowhere near the ~0.6 gate. Note how far raw
accuracy overstates it: DistilRoBERTa ↔ alt2 look like they agree on 53% of records, but 50% of
records are neutral under both, so chance-correction collapses it to κ = 0.21. **This is exactly
why κ was requested alongside accuracy.**

The `P(neg)` correlations are the important exception and are discussed in §1.

### 3.2 By arm

| Arm | Pair | top-1 acc | κ | mean JSD | `P(neg)` r |
|---|---|---|---|---|---|
| persona | Distil ↔ alt1 | 0.340 | 0.182 | 0.392 | +0.689 |
| persona | Distil ↔ alt2 | 0.533 | **0.122** | 0.285 | **+0.487** |
| persona | alt1 ↔ alt2 | 0.109 | **0.048** | 0.587 | +0.583 |
| baseline | Distil ↔ alt1 | 0.398 | 0.240 | 0.342 | +0.861 |
| baseline | Distil ↔ alt2 | 0.534 | 0.270 | 0.284 | +0.779 |
| baseline | alt1 ↔ alt2 | 0.266 | 0.170 | 0.471 | +0.883 |

**Agreement is systematically worse in the persona arm than the baseline arm on every pair and
every metric.** `P(neg)` correlation drops from 0.86 → 0.69 (alt1) and 0.78 → 0.49 (alt2). This is
the most damaging single finding in §3: the classifiers agree about the unconditioned simulator
and disagree about exactly the conditioned text whose difference the spike is measuring. The
persona-minus-baseline contrast is therefore computed in the regime where the labellers are least
interchangeable.

### 3.3 By model

| Model | Pair | top-1 acc | κ | mean JSD | `P(neg)` r |
|---|---|---|---|---|---|
| vicuna-13b-v1.5 | Distil ↔ alt1 | 0.517 | 0.322 | 0.304 | +0.853 |
| vicuna-13b-v1.5 | Distil ↔ alt2 | 0.522 | 0.273 | 0.285 | +0.779 |
| vicuna-13b-v1.5 | alt1 ↔ alt2 | 0.279 | 0.149 | 0.479 | +0.841 |
| Llama-3.1-8B | Distil ↔ alt1 | 0.265 | 0.152 | 0.408 | +0.771 |
| Llama-3.1-8B | Distil ↔ alt2 | 0.544 | 0.181 | 0.316 | +0.510 |
| Llama-3.1-8B | alt1 ↔ alt2 | 0.126 | 0.086 | 0.576 | +0.659 |
| Qwen2.5-7B | Distil ↔ alt1 | 0.325 | 0.143 | 0.389 | +0.735 |
| Qwen2.5-7B | Distil ↔ alt2 | 0.534 | 0.134 | 0.252 | +0.688 |
| Qwen2.5-7B | alt1 ↔ alt2 | 0.157 | 0.090 | 0.533 | +0.464 |

Agreement is *highest* on Vicuna (κ = 0.32) and *lowest* on Qwen (κ = 0.13–0.14) — i.e. lowest on
precisely the model the spike selected. The classifiers are least interchangeable on the winner.

### 3.4 MICA protocol — cosine of score-change vectors

Per model, the 7-dim persona-mean minus baseline-mean emotion vector, cosine-compared across
classifiers. MICA report ≥ 0.78 with downstream performance largely unaffected.

| | Distil ↔ alt1 | Distil ↔ alt2 | alt1 ↔ alt2 |
|---|---|---|---|
| **pooled (21-dim, all 3 models)** | **0.273** | **0.311** | **−0.015** |
| vicuna-13b-v1.5 | −0.108 | 0.933 | 0.119 |
| Llama-3.1-8B-Instruct | 0.796 | −0.239 | 0.164 |
| Qwen2.5-7B-Instruct | **0.785** | **−0.367** | −0.170 |

The pooled headline is **0.27 / 0.31 / −0.02**, far below MICA's 0.78 bar. The per-model row for
Qwen is the clearest statement of the problem: DistilRoBERTa and alt1 agree on the *shape* of what
personas do to Qwen (0.785, just above the bar), while DistilRoBERTa and alt2 point in
**opposite directions** (−0.367). No two classifiers agree on the direction of persona-induced
affective change across all three models — every pair has at least one negative cell.

---

## 4. Confusion matrices vs DistilRoBERTa

Top-1 labels, DistilRoBERTa in rows, alternative in columns, all 8,820 records.

### 4.1 DistilRoBERTa (rows) × alt1 GoEmotions (cols)

| | anger | disgust | fear | happiness | neutral | sadness | surprise | **row n** |
|---|---|---|---|---|---|---|---|---|
| **anger** | 0 | 0 | 0 | **49** | 1 | 0 | 4 | 54 |
| **disgust** | 2 | 0 | 3 | 14 | 0 | 10 | 11 | 40 |
| **fear** | 5 | 0 | 5 | **61** | 1 | 25 | **63** | 160 |
| **happiness** | 0 | 0 | 0 | 1946 | 1 | 2 | 41 | 1990 |
| **neutral** | 27 | 0 | 0 | **2226** | 286 | 183 | **1648** | 4370 |
| **sadness** | 3 | 0 | 0 | 245 | 3 | **752** | 90 | 1093 |
| **surprise** | 2 | 0 | 0 | 813 | 33 | 1 | 264 | 1113 |
| **col n** | 39 | **0** | 8 | 5354 | 325 | 973 | 2121 | |

### 4.2 DistilRoBERTa (rows) × alt2 EmoBERTa (cols)

| | anger | disgust | fear | happiness | neutral | sadness | surprise | **row n** |
|---|---|---|---|---|---|---|---|---|
| **anger** | 0 | 0 | 0 | 0 | **53** | 0 | 1 | 54 |
| **disgust** | 5 | 0 | 0 | 0 | 20 | 15 | 0 | 40 |
| **fear** | 10 | 0 | 0 | 1 | **89** | 60 | 0 | 160 |
| **happiness** | 15 | 0 | 0 | 371 | **1595** | 9 | 0 | 1990 |
| **neutral** | 179 | 0 | 0 | 153 | 3645 | 393 | 0 | 4370 |
| **sadness** | 15 | 0 | 0 | 6 | **406** | 666 | 0 | 1093 |
| **surprise** | 47 | 0 | 0 | 130 | **906** | 7 | 23 | 1113 |
| **col n** | 271 | **0** | **0** | 661 | 6714 | 1150 | 24 | |

### 4.3 Where the negative labels go

Row-normalised, restricted to records DistilRoBERTa calls negative:

| DistilRoBERTa says | n | → alt1 | → alt2 |
|---|---|---|---|
| **anger** | 54 | **happiness 0.91**, surprise 0.07 | **neutral 0.98** |
| **disgust** | 40 | happiness 0.35, surprise 0.28, sadness 0.25 | neutral 0.50, sadness 0.38, anger 0.12 |
| **fear** | 160 | surprise 0.39, **happiness 0.38**, sadness 0.16 | **neutral 0.56**, sadness 0.38 |
| **sadness** | 1093 | **sadness 0.69**, happiness 0.22 | **sadness 0.61**, neutral 0.37 |

Commentary:

- **Sadness is the only negative category with real cross-classifier substance.** ~60–70% of
  DistilRoBERTa's sadness survives as sadness under both alternatives. It is also 81% of
  DistilRoBERTa's negative top-1 mass (1093 / 1347), so `Q_emo`'s negative channel is, in
  practice, mostly a sadness detector — and that part is the robust part.
- **DistilRoBERTa's `anger` and `fear` are not corroborated by anything.** All 54 anger records go
  to happiness (alt1) or neutral (alt2). Of 160 fear records, alt1 says surprise/happiness 77% of
  the time and alt2 says neutral 56%. §8 shows what these records actually are.
- **The task's stated worry — "if one classifier routes sadness to neutral, that suppresses
  `Q_emo`" — is confirmed for alt2**: 37% of DistilRoBERTa's sadness becomes alt2-neutral, and
  30% of alt2's neutral column is drawn from DistilRoBERTa's non-neutral rows.
- **Neither alternative ever emits `disgust` as top-1** (col n = 0 for both), and alt2 never emits
  `fear`. Two of the four components of `P(neg)` are effectively unavailable outside DistilRoBERTa.
- Overall negative top-1 *rates* are deceptively similar — 15.3% (Distil), 11.6% (alt1), 16.1%
  (alt2) — but they disagree on *which* records: only 58–60% of DistilRoBERTa's negatives are
  negative under the alternatives.

---

## 5. Neutral fraction per classifier per arm

Mean neutral probability mass (top-1 neutral rate in parentheses).

| Classifier | overall | persona | baseline |
|---|---|---|---|
| DistilRoBERTa | 0.424 (0.495) | 0.471 | 0.377 |
| **alt1 GoEmotions** | **0.086 (0.037)** | 0.090 | 0.082 |
| alt1b (max collapse) | 0.113 (0.050) | — | — |
| **alt2 EmoBERTa** | **0.656 (0.761)** | 0.704 | 0.609 |

By model:

| Classifier | vicuna p/b | Llama p/b | Qwen p/b |
|---|---|---|---|
| DistilRoBERTa | 0.461 / 0.261 | 0.487 / 0.394 | 0.465 / 0.477 |
| alt1 | 0.100 / 0.106 | 0.076 / 0.064 | 0.094 / 0.077 |
| alt2 | 0.719 / 0.545 | 0.608 / 0.685 | **0.785 / 0.597** |

**The estimated neutrality of the simulator spans 9% → 66%, a 7.6× range, on identical text.**
This is decisive for the "affectively flat simulator" framing in the spike: under alt1 the
simulator is not flat at all (9% neutral, 52% happiness), under alt2 it is almost entirely flat
(66% neutral, 76% of records neutral at top-1). **Flatness is substantially a property of the
labeller, not the simulator** — the INSENSITIVE LABELLER row of the decision table also fires, for
alt2 specifically, which is neutral-heavy in *both* arms (0.704 / 0.609).

One thing is consistent: **the persona arm is more neutral than baseline under all three
classifiers** (Distil +0.094, alt1 +0.008, alt2 +0.095). Persona conditioning does not make the
simulator less neutral on any labeller's reading.

---

## 6. Phase 4 — spike verdicts recomputed under each classifier

Same binning procedure as `spike/RESULTS.md` §8b (pooled token deciles, arm mean per bin, direct
standardisation to the pooled length distribution). Bootstrap: 2,000 resamples within arm, bin
edges held at the observed deciles. **Only `emo_dist` was varied; every other analysis choice is
`spike/phase7b_robustness.py` unchanged.**

**Reproduction check:** the DistilRoBERTa row below reproduces the published spike numbers exactly
(0.85× / 0.68× / 1.65× [1.48, 1.85] vs published [1.49, 1.86]), so differences in the other rows
are attributable to the classifier and not to my re-implementation.

### Length-standardised ratios (persona / baseline)

| Classifier | Qwen ratio | Vicuna ratio | Llama ratio | Qwen verdict |
|---|---|---|---|---|
| **DistilRoBERTa** (original) | **1.65×** [1.48, 1.85] | 0.85× [0.77, 0.95] | 0.68× [0.62, 0.74] | **PASS** |
| **alt1** GoEmotions | **1.33×** [1.19, 1.50] | 0.47× [0.41, 0.55] | 0.67× [0.60, 0.75] | **WEAK** |
| alt1b (max collapse) | 1.40× [1.24, 1.58] | 0.44× [0.38, 0.52] | 0.66× [0.58, 0.74] | WEAK |
| **alt2** EmoBERTa | **1.00×** [0.90, 1.10] | 0.62× [0.56, 0.68] | **1.17× [1.07, 1.28]** | PASS |

### Raw (unadjusted) ratios and underlying rates

| Classifier | Model | raw persona | raw baseline | **raw ratio** | std persona | std baseline | **std ratio** |
|---|---|---|---|---|---|---|---|
| DistilRoBERTa | vicuna | 0.2125 | 0.2395 | 0.89× | 0.2036 | 0.2382 | 0.85× |
| DistilRoBERTa | Llama | 0.2408 | 0.3174 | 0.76× | 0.2350 | 0.3460 | 0.68× |
| DistilRoBERTa | Qwen | 0.2211 | 0.1693 | **1.31×** | 0.2524 | 0.1527 | **1.65×** |
| alt1 | vicuna | 0.0938 | 0.1934 | 0.48× | 0.0878 | 0.1859 | 0.47× |
| alt1 | Llama | 0.1423 | 0.1800 | 0.79× | 0.1346 | 0.2006 | 0.67× |
| alt1 | Qwen | 0.0836 | 0.1168 | **0.72×** | 0.1166 | 0.0876 | **1.33×** |
| alt2 | vicuna | 0.1683 | 0.2511 | 0.67× | 0.1504 | 0.2428 | 0.62× |
| alt2 | Llama | 0.3147 | 0.2322 | **1.36×** | 0.3037 | 0.2605 | **1.17×** |
| alt2 | Qwen | 0.1483 | 0.2004 | **0.74×** | 0.1729 | 0.1733 | **1.00×** |

### Three findings from this table

**(a) alt2 reverses two verdicts, in opposite directions.** Llama, the spike's *worst* model at
0.68×, becomes 1.17× with a CI excluding 1.0. Qwen, the spike's *only* pass at 1.65×, becomes
1.00× with a CI straddling 1.0 — the persona effect on negative affect disappears entirely. The
model ranking is not merely rescaled, it is re-ordered.

**(b) Under both alternatives, Qwen's raw ratio is below 1.0** (0.72×, 0.74×) — the persona arm is
*less* negative than baseline before any adjustment. Under DistilRoBERTa the raw ratio was already
1.31×, so the length standardisation only amplified an effect that was there. Under the
alternatives the length standardisation has to **reverse the sign of the effect** to get above 1.0
(0.72× → 1.33×). The spike's §8b argument — that length suppresses rather than inflates the gap —
is defensible, but it is a much heavier load-bearing assumption when it is doing all the work
rather than some of it. Any claim resting on alt1's 1.33× is a claim about the length adjustment
at least as much as about the persona effect.

**(c) The one thing that is robust is the manipulation check.** Permutation p = 0.0000 for all 12
classifier × model cells (`c4/verdicts.json`). Personas demonstrably change the simulator's output
under every labeller. What is not robust is *which* model changes in the affectively-correct
direction.

---

## 7. Directional correlations under each classifier

Spearman ρ, one-sided in the predicted direction, persona arm, per-persona means over 98 personas
— identical to `spike/phase567_analysis.py` Phase 7. `*` = significant at p < 0.05.

| Classifier | Model | H1 neurotic→neg | H2 agreeable→happy | H3 open→surprise | n_sig |
|---|---|---|---|---|---|
| **DistilRoBERTa** | vicuna | +0.006 (p=.478) | **+0.571*** | +0.041 (p=.344) | 1 |
| | Llama | −0.022 (p=.583) | **+0.500*** | −0.023 (p=.587) | 1 |
| | Qwen | **+0.183*** (p=.036) | **+0.373*** | **+0.221*** (p=.014) | **3** |
| **alt1** GoEmotions | vicuna | −0.105 (p=.849) | **+0.337*** | **−0.376** (p=1.00) | 1 |
| | Llama | −0.158 (p=.940) | **+0.370*** | **−0.211** (p=.982) | 1 |
| | Qwen | +0.050 (p=.313) | **+0.230*** | +0.005 (p=.482) | **1** |
| alt1b (max) | Qwen | +0.029 (p=.389) | **+0.252*** | −0.009 (p=.534) | 1 |
| **alt2** EmoBERTa | vicuna | +0.045 (p=.329) | +0.164 (p=.053) | −0.064 (p=.735) | **0** |
| | Llama | **+0.241*** (p=.008) | **+0.426*** | −0.086 (p=.801) | **2** |
| | Qwen | **−0.020** (p=.578) | **+0.461*** | **+0.248*** (p=.007) | 2 |

**H1 — the hypothesis that most directly underwrites `Q_emo` — does not survive.** Qwen's H1
(ρ = +0.183, p = .036) was already the spike's weakest result, with a bootstrap CI crossing zero
(§8 caveat in `spike/RESULTS.md`). Under alt1 it falls to ρ = +0.050 (n.s.); under alt2 it turns
**negative**, ρ = −0.020. It reappears under alt2 on a different model (Llama, ρ = +0.241,
p = .008). H1 is significant in exactly 2 of 9 cells, and never in the same cell twice.

**H2 — agreeableness → happiness — is the robust result.** Significant in 8 of 9 cells (the miss
is vicuna/alt2 at p = .053, a near-miss with the correct sign), always positively signed,
ρ = +0.16 to +0.57. This is the one directional claim the paper can make without a classifier
caveat.

**H3 — openness → surprise — is worse than not-robust; it flips sign.** Under alt1 it is
significantly *negative* for vicuna (ρ = −0.376) and Llama (ρ = −0.211). Qwen's H3 holds under
DistilRoBERTa (+0.221) and alt2 (+0.248) but is null under alt1 (+0.005). Note that alt1's
"surprise" folds in *realization*, *confusion* and *curiosity* (Google's grouping), so it is
measuring a materially different construct — a real limitation of collapsing GoEmotions, and one
reason to weight H3 lowest.

**Net: Qwen's 3/3 sweep — the entire basis for selecting it as the simulator — is 1/3 under alt1
and 2/3 under alt2 (with a different hypothesis failing).**

---

## 8. Ten highest-disagreement responses

Ranked by maximum pairwise JSD across the three classifiers. Full distributions in
`c4/disagreement_examples.json`; probabilities ≥ 0.10 shown.

| # | JSD | model / arm | response | DistilRoBERTa | alt1 | alt2 |
|---|---|---|---|---|---|---|
| 1 | 0.956 | Qwen / baseline | "That's really impactful! I'm considering donating, but I'm a bit hesitant about how the money will be used…" | **fear 0.98** | surprise 0.80 | happiness 0.93 |
| 2 | 0.955 | vicuna / persona | "As a person who values tradition and religious custom, I would want to know more about the charity…" | **disgust 0.99** | surprise 0.63, happiness 0.18, neutral 0.15 | anger 0.42, neutral 0.36, disgust 0.13 |
| 3 | 0.945 | Llama / baseline | "I'm always a bit hesitant to donate to charities, I'd like to know a bit more about what Save the Children does…" | **fear 0.99** | surprise 0.67, happiness 0.16 | neutral 0.85 |
| 4 | 0.939 | Llama / baseline | "I'm always a bit hesitant to donate to charities, I like to do my research first…" | **fear 0.99** | happiness 0.42, surprise 0.18, neutral 0.16 | neutral 0.89 |
| 5 | 0.939 | Llama / persona | "I can imagine it's tough, but I try not to get too caught up in feeling overwhelmed by every cause…" | fear 0.31, surprise 0.67 | happiness 0.49, neutral 0.25, sadness 0.14 | **neutral 0.95** |
| 6 | 0.936 | vicuna / baseline | "Sure, I'm happy to donate. What does Save the Children do?" | **happiness 0.99** | happiness 0.69, surprise 0.23 | **neutral 0.97** |
| 7 | 0.934 | Qwen / baseline | "I'm a bit hesitant right now, but maybe if I hear more about how my donation will make a difference…" | **fear 0.99** | surprise 0.40, happiness 0.31 | neutral 0.80 |
| 8 | 0.927 | vicuna / baseline | "Sure, I'm happy to help out. What does Save the Children do?" | **happiness 0.98** | happiness 0.67, surprise 0.25 | **neutral 0.97** |
| 9 | 0.927 | vicuna / baseline | "Sure, I'm happy to help out. What does Save the Children do?" | **happiness 0.98** | happiness 0.67, surprise 0.25 | **neutral 0.97** |
| 10 | 0.926 | Llama / baseline | "That sounds amazing, but I'm not sure if I can commit to donating right now, I've been a bit tight on my budget…" | **surprise 0.97** | happiness 0.49, surprise 0.39 | happiness 0.87, neutral 0.11 |

**What disagreement looks like, concretely.** Four of the ten (#1, #3, #4, #7) are the same
construction: *polite hedged deliberation about donating*, which DistilRoBERTa assigns
**fear ~ 0.99**. The converse failure is alt2's, visible at #6/#8/#9: "Sure, I'm happy to donate"
is scored **neutral 0.97**. EmoBERTa was trained on MELD/IEMOCAP, where the neutral class
dominates and politeness is not emotion; it under-reads mild affect as clearly as DistilRoBERTa
over-reads hedging. Neither is obviously right, which is why "which classifier is authoritative"
is a real question and not a formality - the two error modes push `P(neg)` in opposite directions.

> **These ten examples are selected on maximum JSD, i.e. chosen precisely for being the most
> extreme disagreements in the corpus. They are not a basis for any quantitative claim about how
> often DistilRoBERTa mistakes hedging for fear.** An earlier draft of this report drew exactly
> that inference from them. Section 8A tests it at corpus scale and finds it does not hold.


---

## 8A. Is DistilRoBERTa's `fear` label a hedging detector? (follow-up, no GPU)

Motivated by the §8 examples. Every utterance DistilRoBERTa labels `fear` (top-1, n = 160) was
scored for hedging markers. Marker families were fixed before looking at results
(`c4/phase9_hedging.py`); the four requested — *hesitant*, *not sure*, *maybe*, *would need to
know more* — plus their morphological variants and the same epistemic-hedge family.

**Result: the hypothesis is not supported. `fear` is mildly hedge-enriched, but hedging neither
explains the label nor inflates `Q_emo` — and removing `fear` entirely makes the spike's Qwen
effect stronger, not weaker.**

### 8A.1 The raw fraction is high, but so is the base rate

| | n | contains a hedge marker |
|---|---|---|
| DistilRoBERTa `fear` | 160 | **59.4%** |
| all other labels | 8,660 | 45.7% |
| whole corpus | 8,820 | 46.0% |

Odds ratio 1.73, Fisher p = 7.3e-4. So "most of them hedge" is literally true — but **46% of the
entire corpus hedges**, because it is a donation-refusal corpus and hedging is its dominant
register. The enrichment is real but modest.

### 8A.2 The control that settles it: hedge rate by label

| DistilRoBERTa label | n | hedged | mean # marker families | strict fear vocab |
|---|---|---|---|---|
| anger | 54 | 3.7% | 0.04 | 0.0% |
| disgust | 40 | 25.0% | 0.33 | 0.0% |
| **fear** | **160** | **59.4%** | **1.11** | 2.5% |
| happiness | 1990 | 27.3% | 0.34 | 0.0% |
| neutral | 4370 | 52.5% | 0.80 | 0.0% |
| sadness | 1093 | 35.7% | 0.42 | 0.0% |
| **surprise** | 1113 | **64.9%** | 0.78 | 0.0% |

**`surprise` is hedged more often than `fear` (64.9% vs 59.4%), and `neutral` nearly as often
(52.5%).** Hedging is not what distinguishes the `fear` class. `fear` does carry the most marker
families per utterance (1.11), so it is the most *densely* hedged label — but it is not
categorically a hedging detector.

### 8A.3 Narrow lexical triggering is real

| marker | in `fear` | in corpus | lift |
|---|---|---|---|
| **hesitant / hesitate** | 8.8% | 0.2% | **55.1×** |
| **reluctant / cautious / wary / skeptical** | 26.2% | 0.6% | **47.3×** |
| a bit / a little | 23.8% | 11.6% | 2.1× |
| know more / more about | 17.5% | 12.2% | 1.4× |
| not sure / unsure | 20.6% | 19.0% | 1.1× |
| maybe / perhaps | 1.9% | 1.5% | 1.2× |
| would need to | 3.8% | 5.4% | 0.7× |
| think about it | 4.4% | 7.1% | 0.6× |
| might / guess / probably | 2.5% | 5.6% | 0.4× |

The specific words in the §8 examples do trigger `fear` near-deterministically — *hesitant* at 55×
lift, the *reluctant/cautious/wary* family at 47×. Separately, **31 of the 32 corpus utterances
containing "worry" are top-1 `fear`.** But the generic hedges (*not sure*, *maybe*, *would need
to*, *think about it*) are at or **below** base rate. It is a narrow lexicon, not hedging as such.

### 8A.4 The label is not tracking genuine fear either

Fear vocabulary (*afraid, scared, anxious, terrified, nervous, worried*…) appears in 21.2% of
`fear` utterances vs a 0.4% corpus base rate — but **auditing those 34 hits, 28 are the
concern-idiom** "I've got my own family / expenses / bills **to worry about**", which is a refusal
and a statement of competing priorities, not fear. Excluding that idiom, **strict fear vocabulary
appears in 2.5% of `fear`-labelled utterances (4 / 160)**, against a 0.1% base rate.

So DistilRoBERTa's `fear` class on this corpus is neither mostly-hedging nor mostly-fear: it is a
narrow lexical class centred on *hesitant / reluctant / cautious / wary / worry-about*.

### 8A.5 …but it barely moves the continuous quantity, and it is only 16% of `P(neg)`

`Q_emo` consumes `P(neg)` mass, not top-1 labels. At that level the coupling nearly vanishes:

| | value |
|---|---|
| Spearman ρ(# hedge markers, `P(fear)`) | **+0.084** (p = 2.9e-15) |
| mean `P(fear)`, hedged vs unhedged | 0.0392 vs 0.0361 = **1.08×** |
| `fear` share of mean `P(neg)` | **16.1%** |
| **mean `P(neg)`, hedged vs unhedged** | **0.71× — hedged text is LESS negative** |

The correlation is significant only because n = 8,820; the effect size is 8% on a component worth
16% of the total. And decisively, **hedged utterances carry *less* total negative mass than
unhedged ones (0.71×)** — under all three classifiers (alt1 0.58×, alt2 0.77×). Hedging *deflates*
`Q_emo`; it does not inflate it.

### 8A.6 The decisive test: drop `fear` from `P(neg)` entirely

Length-standardised persona/baseline ratio, DistilRoBERTa, recomputed with `P(neg)` redefined as
sadness + anger + disgust:

| Model | `P(neg)` full | **`P(neg)` without fear** | unhedged only | unhedged & no fear |
|---|---|---|---|---|
| vicuna-13b-v1.5 | 0.85× | 0.80× | 0.64× | 0.62× |
| Llama-3.1-8B | 0.68× | 0.59× | 0.82× | 0.71× |
| **Qwen2.5-7B** | **1.65×** | **1.88×** | 0.95× | 1.12× |

**Removing `fear` makes the Qwen effect larger (1.65× → 1.88×), not smaller.** The same holds
under both alternatives (alt1 1.33× → 1.35×, alt2 1.00× → 1.00×). Whatever is wrong with
DistilRoBERTa's `fear` class, it is *diluting* the spike's headline result, not manufacturing it.
The hedging-artifact explanation is refuted.

### 8A.7 A different confound did surface, and it is real

The arms differ in how much they hedge:

| Model | hedged, persona | hedged, baseline |
|---|---|---|
| vicuna-13b-v1.5 | 40.1% | 35.0% |
| Llama-3.1-8B | 53.4% | 62.0% |
| **Qwen2.5-7B** | **34.4%** | **51.0%** |

Qwen's persona arm hedges **17 points less** than its baseline arm. Combined with 8A.5 — unhedged
text is more negative — this is a compositional route to the effect: persona conditioning makes
Qwen more direct and less hedging, and direct refusals score more negative. Restricting to
unhedged utterances drops Qwen from 1.65× to **0.95×**.

That restriction is *not* a clean test — it discards 34% of the persona arm and 51% of the
baseline arm, and it removes the corpus's dominant speech act, so the surviving subsets are not
comparable. It should not be read as "the effect disappears". But it does mean the Qwen effect is
entangled with a **register shift** (hedging → directness) rather than being purely an affective
shift. Whether that counts as the persona "emotional dynamics" the spike claimed is a question
about construct validity, and it is a fair thing for a reviewer to press on. It is also
independent of the classifier-robustness problem, so it needs handling either way.

### 8A.8 Correction to this report

An earlier draft of §8 argued from the ten max-JSD examples that DistilRoBERTa "systematically
scores ordinary donation hesitancy as negative affect" and that this was "a plausible mechanism
for the spike's Qwen effect". **That inference was wrong**, and it was drawn from a sample
selected for extremity. At corpus scale: hedging is not specific to `fear`, the dose-response on
`P(fear)` is 1.08×, hedged text is *less* negative overall, and deleting `fear` *raises* the Qwen
ratio. §8 has been marked accordingly. The C4 verdict is unaffected — it rests on
cross-classifier disagreement (§3, §6, §7), which stands unchanged.


---

## 9. What to do

**The grid should not start on the current simulator choice.** The decision that ~230 GPU-hours
rests on — Qwen as user simulator, because it alone showed persona-driven emotional dynamics — is
the specific finding that does not survive changing the labeller.

Options, cheapest first. All of (1)–(2) reuse existing text and cost **zero GPU generation**:

1. **Committee re-analysis (recommended).** Define `P(neg)` as the mean over a 3-classifier
   committee and re-run Phase 4 + the directional tests. Pre-register the committee before
   looking. This is a few hours and gives a simulator choice that is robust by construction. It
   also directly answers ICDM Reviewer 1: a committee-based affect measure is a stronger response
   than any single-classifier result.
2. **Re-select on cross-classifier agreement.** Choose the simulator whose persona effect has the
   same sign under all three labellers. On the current numbers no model qualifies on H1; Qwen
   qualifies on the `P(neg)` ratio under DistilRoBERTa + alt1 + alt1b but not alt2.
3. **Declare DistilRoBERTa authoritative and justify it on evidence, not convention.** Legitimate,
   and 8A removes the main objection to it: its `fear` label is lexically triggered, but `fear` is
   only 16% of `P(neg)` and removing it *strengthens* the Qwen result. This route still needs a
   human-annotated validation sample on P4G text - a few hundred turns, triple-annotated - to show
   DistilRoBERTa is the *better* labeller here, not just the incumbent.
4. **Human validation sample** — the only thing that actually resolves which classifier is right.
   Worth doing regardless of (1)–(3), and it is the strongest single artefact against Reviewer 1.

### Consequences for the paper as currently drafted

- **`spike/RESULTS.md` §1** ("1 of 3 models pass", Qwen as simulator) must be qualified: it holds
  under DistilRoBERTa and is not reproduced under either alternative.
- **§5.4b claims** about the magnitude of the persona effect cannot be stated as a point estimate.
  The defensible range across labellers is **1.00×–1.65×**, with the lower end's CI including 1.0.
- **§6/§7 "affective flatness"** must be reported as labeller-relative (neutral mass 9%–66%).
- **H1** should not be presented as established for Qwen under any reading.
- **H2 can stand** — it is robust across all classifiers and all models.
- **The manipulation check can stand** — p = 0.0000 everywhere.

---

## 10. Integrity notes

1. **8,820 records, not the 9,000 stated in the task** — this is the spike's own count
   (`spike/RESULTS.md` §11.1: 98 of 100 dialogues had complete personas). Not a shortfall here.
2. **Classifier selection was pre-committed** in `c4/SELECTION.md` before Phase 2 ran, with
   rejection reasons for candidates not used. **Exactly two alternatives were run; no third
   classifier was run and its results discarded.** `alt1b` is the *same* alt1 forward pass under a
   different documented Ekman collapse, added specifically to test whether my own aggregation was
   manufacturing disagreement (§2.1). It was not.
3. **The GoEmotions→Ekman mapping is Google's published one**, used verbatim, not hand-tuned.
4. **The DistilRoBERTa re-implementation reproduces the published spike numbers exactly**
   (§6), which is what licenses attributing the other rows to the classifier.
5. **Analysis choices were frozen to the spike's scripts.** Bin edges, standardisation target,
   bootstrap structure, correlation form and one-sided tests are `phase7b_robustness.py` /
   `phase567_analysis.py`; only the `emo_dist` field varies. The only quantity not specified in
   those scripts is the bootstrap on the standardised ratio (`spike/RESULTS.md` §8b quotes
   [1.49, 1.86] but no code was found for it); the procedure used here — resample within arm,
   edges fixed — reproduces it to [1.48, 1.85] and is documented in `c4/phase34_agreement.py`.
6. **Nothing was committed to git; nothing under `src/` was modified.** `scipy` was installed into
   a throwaway scratchpad venv, not the project environment.
7. **This result undermines the spike**, and is reported as found. No classifier was dropped,
   re-run, or re-mapped after seeing its Phase 4 numbers.

## 11. Files

| File | Contents |
|---|---|
| `c4/SELECTION.md` | Phase 1, pre-committed before re-labelling |
| `c4/phase2_relabel.py` | Re-labelling with both alternatives |
| `c4/phase2b_alt1_raw.py` | Raw 28-dim GoEmotions dump for the mapping check |
| `c4/relabelled.jsonl` | 8,820 records × 4 distributions |
| `c4/alt1_raw_goemotions.jsonl` | Raw GoEmotions sigmoid scores |
| `c4/phase34_agreement.py` | Phases 3 + 4 |
| `c4/phase4b_verdict.py` | Permutation gate + full spike verdict rule per classifier |
| `c4/agreement.json` | All Phase 3 + 4 statistics |
| `c4/verdicts.json` | Per-classifier verdicts |
| `c4/disagreement_examples.json` | Top-10 disagreements, full distributions |
| `c4/phase9_hedging.py` | Section 8A hedging-marker analysis |
| `c4/hedging.json` | Hedging statistics |
