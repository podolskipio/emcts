# Pre-registration — Persona Spike (P4G user simulator)

**Written before any generation, classification, or inspection of persona *values*.**
Prior to writing this file the only data touched was the *column header* of
`data/p4g_personas/full_info.csv` (to fix dimension names) and the dialogue-ID list of
`data/p4g/300_dialog_turn_based.pkl`. No trait values, no responses, no emotion scores were
examined. Creation time is recorded in `RESULTS.md`.

## Research question

Does conditioning the P4G user simulator on the persuadee's 23-dimension psychological
profile measurably change their emotional responses — beyond sampling noise, in the
*direction* the profile predicts?

## Models (tested independently, run sequentially)

| Role | Checkpoint | Quantization |
|---|---|---|
| incumbent | `TheBloke/vicuna-13B-v1.5-AWQ` | AWQ 4-bit |
| modern 8B | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ 4-bit |
| other family | `Qwen/Qwen2.5-7B-Instruct-AWQ` | AWQ 4-bit |

Sampling parameters are **identical across all three**: `temperature=0.7`, `top_p=0.9`,
`max_new_tokens=64`, seeds `{0,1,2}`. Any deviation invalidates the cross-model comparison.

## The 23 dimensions

Big Five (5): `extrovert`, `agreeable`, `conscientious`, `neurotic`, `open`
Moral Foundations (6): `care`, `fairness`, `loyalty`, `authority`, `purity`, `freedom`
Schwartz Portrait Values (10): `conform`, `tradition`, `benevolence`, `universalism`,
`self_direction`, `stimulation`, `hedonism`, `achievement`, `power`, `security`
Decision-Making style (2): `rational`, `intuitive`

Demographics used in rendering: `age`, `sex`, `race`, `edu`, `marital`, `employment`.

## The 5 fixed system utterances (verbatim, byte-identical across every generation)

1. **greeting** — `Hello! How are you doing today?`
2. **credibility appeal** — `Save the Children has operated for over 100 years and publishes annual impact reports.`
3. **logical appeal** — `A $2 donation provides a week of school meals for one child.`
4. **emotion appeal** — `Some of these children have lost everything. It's hard not to feel for them.`
5. **proposition of donation** — `Would you consider donating a small amount today?`

These are the *only* stimulus. They never vary by persona, arm, seed, or model. Holding the
stimulus fixed is what makes persona and seed the only varying factors, and therefore what
makes the noise-floor comparison in Phase 5 interpretable.

## Persona rendering rules

- Tertile boundaries computed **across the full corpus of persuadee profiles** (~1,017
  dialogues), not across the 100 evaluation dialogues.
- A trait is rendered only if it falls in the top tertile (`>= hi`) or bottom tertile
  (`<= lo`). Mid-tertile traits carry no information and are dropped.
- Traits sorted by distance beyond the tertile boundary, most extreme first.
- **At most 6 facets**, then one demographic sentence.
- Hard cap **80 tokens**; truncate facets (never the demographic sentence) if exceeded.
- Rendered persona is inserted **after** the static preamble and **after** the in-context
  example, immediately **before** the dialogue history, to preserve RadixAttention prefix
  sharing.
- **Baseline arm** is the identical code path with the persona string empty — same seeds,
  same ordering, same sampling parameters.

## Hypotheses (exactly three; fixed here, tested in Phase 7)

- **H1**: `neurotic` ↑ → P(fear + sadness) ↑ — Spearman ρ > 0
- **H2**: `agreeable` ↑ → P(happiness) ↑ — Spearman ρ > 0
- **H3**: `open` ↑ → P(surprise) ↑ — Spearman ρ > 0

Tested one-sided in the predicted direction, α = 0.05, with bootstrap 95% CIs (1,000
resamples). Any correlation computed for any other dimension is **exploratory** and will be
labelled as such. With 23 dimensions available, testing all of them would guarantee a false
positive; the hypothesis set is frozen at three.

## Emotion measurement

`j-hartmann/emotion-english-distilroberta-base`, identical for all models. The **full 7-dim
softmax** is stored for every generation; argmax is never stored. The classifier's native
label `joy` is recorded as `happiness` for consistency with the reporting schema — a rename
only, no change to the distribution.

`P(negative) = fear + sadness + anger + disgust`.

## Decision thresholds (applied per model)

| Condition | Verdict |
|---|---|
| permutation `p > 0.05` | **FAIL** |
| `p < 0.05`, no pre-registered ρ significant | **WEAK** |
| `p < 0.05` and ≥2 pre-registered ρ significant with predicted sign | **PASS** |
| PASS, and negative fraction materially above baseline | **STRONG PASS** |

Phase 5 gates per model: a model with `p > 0.05` is recorded FAIL and skips Phases 6–7.

## Stopping / integrity rules

- **STOP and report** if fewer than 80 of the 100 evaluation dialogues have a complete profile.
- `FACET_TEXT` will **not** be revised after seeing results. If it ever is, that fact will be
  stated explicitly and everything after is labelled exploratory.
- Null results are reported as findings, not as failures to be tuned away.
- Neutral fraction is reported per arm, so classifier insensitivity remains distinguishable
  from a genuine null.
