# P-RELABEL — is the emotion-conditioned action finding an artifact of one classifier?

Script: `scripts/p_relabel.py` (CPU, no generation). Numbers: `p_relabel.json`.

## Verdict: **weakens**

The direction holds under all three classifiers: after a negative user emotion, EmoMCTS plays more
trust-building and fewer "other" acts than GDP-Zero, and after a non-negative emotion the two planners
match. So the pattern does not reverse. But it was never statistically supported, even under the
original classifier: n = 15 per negative cell, trust-building difference +0.13 with a
dialogue-clustered 95 % CI of [−0.26, +0.53]. Under either alternative labeller the conditioning cell
shrinks from 15 to 4–6 system turns, and the negative set is a mostly *different* set of
utterances: only 9 of DistilRoBERTa's 41 negative user turns are negative under GoEmotions, and 9
under EmoBERTa. No panel's CI excludes zero.

The finding can be reported as a direction consistent across three labellers, not as evidence of
emotion-conditioned repair. The paper's "0.73 vs 0.60" should carry its n and CI.

---

## What §IV-D computes, and reproduction

The paper's Figure B / `analysis/article_action_analysis.md` is produced by
`scripts/plot_emotion_conditioned_actions.py`. Reproduction imports that script's own `tally()`
rather than re-implementing it.

- **Runs:** `rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims` (EmoMCTS, β = 0.7, 40 sims, 50
  dialogues, Ollama `vicuna:13b`, `--emotion_classifier hf`, top-K 5, no persona) and
  `rollout_p4g_gdpzero_vicuna_50d_40s` (GDP-Zero, same backbone and budget). These pickles are not in
  the repo. They survive only in an earlier session's scratchpad (three md5-identical copies) and are
  now copied to `analysis/phase1/relabel_src/`, together with the GDP-Zero utterance → label cache.
- **"Last user emotion":** the label of the most recent Persuadee turn before each Persuader turn.
  The opening greeting has none and is skipped. EmoMCTS uses the label logged during the run;
  GDP-Zero's user turns are labelled post hoc with the same HF classifier. An empty utterance maps to
  neutral.
- **Negative:** {fear, sadness, anger, disgust, contempt}. No classifier here emits contempt.
- **Groups:** trust-building = {emotion appeal, credibility appeal}; proposition = {proposition of
  donation}; other = everything else.
- **Turns included:** every system turn after the greeting, in all 50 dialogues of each run.

**Reproduced exactly:** all four cells, n and proportions to 4 d.p., against
`src/outputs/old/emotion_conditioned_actions_40s.csv`. A fresh DistilRoBERTa run over the raw text
agrees 100 % with both the EmoMCTS logged labels and the GDP-Zero cache, so the pipeline is
reproducible from text, not only from cached labels.

## Classifiers (from the C4 audit, `analysis/c4/SELECTION.md`)

| panel | model | revision (HF snapshot) | head → Ekman-7 |
|---|---|---|---|
| original | `j-hartmann/emotion-english-distilroberta-base` | `0e1cd914e3d46199ed785853e12b57304e04178b` | native softmax |
| alt1 | `SamLowe/roberta-base-go_emotions` | `d75048347613a25d77de8cf6412eaae9fa7b26be` | 28 sigmoids summed per Google's Ekman mapping, renormalised |
| alt2 | `tae898/emoberta-large` | `8934b68e8b0d9fc3cd961cc7e7605533c7081e59` | native softmax |

Mappings are verbatim from `analysis/c4/phase2_relabel.py`. 308 non-empty user-turn occurrences
(245 unique utterances) across both runs.

## Three panels

Proportions of the next system act. CI = 95 % percentile, 1000 replicates, dialogues resampled with
replacement within each planner's run. Replicates where a negative cell came out empty are dropped
(original 0, GoEmotions 18, EmoBERTa 6).

### Original — DistilRoBERTa
Negative base rate over user turns: **0.081** (EmoMCTS 0.082, GDP-Zero 0.080).

| planner | after | n | trust-building | proposition | other |
|---|---|---|---|---|---|
| GDP-Zero | negative | 15 | 0.600 | 0.200 | 0.200 |
| EmoMCTS | negative | 15 | 0.733 | 0.200 | 0.067 |
| GDP-Zero | non-negative | 199 | 0.628 | 0.181 | 0.191 |
| EmoMCTS | non-negative | 178 | 0.590 | 0.208 | 0.202 |

| EmoMCTS − GDP-Zero | estimate | 95 % CI |
|---|---|---|
| trust-building, after negative | +0.133 | [−0.257, +0.525] |
| other, after negative | −0.133 | [−0.429, +0.111] |
| trust-building, after non-negative | −0.038 | [−0.129, +0.061] |
| trust-building, difference-in-differences | +0.172 | [−0.258, +0.588] |

### alt1 — GoEmotions (RoBERTa-base)
Negative base rate: **0.018** (EmoMCTS 0.016, GDP-Zero 0.019).

| planner | after | n | trust-building | proposition | other |
|---|---|---|---|---|---|
| GDP-Zero | negative | **5** | 0.800 | 0.000 | 0.200 |
| EmoMCTS | negative | **4** | 1.000 | 0.000 | 0.000 |
| GDP-Zero | non-negative | 209 | 0.622 | 0.187 | 0.191 |
| EmoMCTS | non-negative | 189 | 0.593 | 0.212 | 0.196 |

| EmoMCTS − GDP-Zero | estimate | 95 % CI |
|---|---|---|
| trust-building, after negative | +0.200 | [0.000, +0.667] |
| other, after negative | −0.200 | [−0.667, 0.000] |
| trust-building, after non-negative | −0.029 | [−0.119, +0.052] |
| trust-building, difference-in-differences | +0.229 | [−0.036, +0.714] |

The CI bound sits *at* 0.000 because of discreteness at n = 4–5 (1.000 − 0.800 is the smallest
possible positive gap). It does not exclude zero.

### alt2 — EmoBERTa (RoBERTa-large, MELD + IEMOCAP)
Negative base rate: **0.026** (EmoMCTS 0.025, GDP-Zero 0.027).

| planner | after | n | trust-building | proposition | other |
|---|---|---|---|---|---|
| GDP-Zero | negative | **6** | 0.500 | 0.167 | 0.333 |
| EmoMCTS | negative | **6** | 0.667 | 0.167 | 0.167 |
| GDP-Zero | non-negative | 208 | 0.630 | 0.183 | 0.188 |
| EmoMCTS | non-negative | 187 | 0.599 | 0.209 | 0.193 |

| EmoMCTS − GDP-Zero | estimate | 95 % CI |
|---|---|---|
| trust-building, after negative | +0.167 | [−0.509, +0.750] |
| other, after negative | −0.167 | [−0.625, +0.376] |
| trust-building, after non-negative | −0.031 | [−0.117, +0.051] |
| trust-building, difference-in-differences | +0.198 | [−0.485, +0.789] |

## Agreement on this exact utterance set

Per occurrence, non-empty user turns (n = 308).

| pair | Cohen's κ (7-class) | top-1 agreement | κ (negative vs not) | P(neg) Pearson r | C4 audit κ (spike text) |
|---|---|---|---|---|---|
| DistilRoBERTa ↔ GoEmotions | **0.248** | 0.526 | 0.328 | 0.582 | 0.212 |
| DistilRoBERTa ↔ EmoBERTa | **0.102** | 0.396 | 0.288 | 0.508 | 0.208 |
| GoEmotions ↔ EmoBERTa | **0.036** | 0.140 | 0.341 | 0.500 | 0.108 |

On dialogue text from actual rollouts, agreement is in the audit's range for the first pair and
*worse* for the two pairs involving EmoBERTa. P(neg) correlation, the audit's partial defence
(r 0.66–0.79 there), is also weaker here (0.50–0.58).

Negative-vs-not confusion, rows = first classifier (neg, non):

- DistilRoBERTa ↔ GoEmotions `[[9, 32], [0, 267]]`: GoEmotions calls 9 of DistilRoBERTa's 41 negatives negative, and adds none of its own.
- DistilRoBERTa ↔ EmoBERTa `[[9, 32], [4, 263]]`
- GoEmotions ↔ EmoBERTa `[[4, 5], [9, 290]]`

Where the labels go (7-class matrices in `p_relabel.json`, label order anger, disgust, fear,
happiness, neutral, sadness, surprise):

- **DistilRoBERTa** negatives are fear 28 and sadness 13.
- **GoEmotions** maps 17 of the 28 fears to happiness and all 13 sadnesses to happiness. It is
  dominated by happiness (228/308).
- **EmoBERTa** maps 20 of the 28 fears to neutral and 7 to anger. It is dominated by neutral
  (264/308).

So the three labellers do not disagree only at the margin. They split the rollout text into largely
disjoint negative sets.

## What this does and doesn't answer

- The reviewer's literal claim (the planner learns to avoid its own model family's trigger words) is
  answered by construction: the labeller is an independent encoder. The residual worry (the pattern
  belongs to one labeller) is **not confirmed and not refuted**. The point estimates survive the
  swap, but the data cannot distinguish a real effect from noise under any labeller.
- The occupancy gap is itself a result: 8.1 % negative under the original, 1.8–2.6 % under the
  alternatives. The rollout text is affectively flat under every classifier tried, and the "after
  negative" condition is a small-sample cell whichever one is used.
- Scope: this is the paper's 40-sim, 50-dialogue, no-persona, Ollama-backbone pair of runs. It is not
  the Phase-1 D1/D2 runs, which use a different backbone server, persona conditioning and 50 sims.
