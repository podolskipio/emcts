# bf16 control — do the persona spike verdicts survive dequantization?

Re-runs the persona spike (`spike/RESULTS.md`) on **unquantized bf16** checkpoints for the two
models that fit on a 24 GB RTX 4090. Vicuna-13B (~26 GB at bf16) does not fit and is out of
scope. Everything except checkpoint precision is held fixed: same 98 personas, same 5 system
utterances, same 3 replicates, same prompt construction, same sampling, same emotion
classifier, same analysis code.

## Verdict

**The spike's pre-registered verdicts survive dequantization intact.** Llama-3.1-8B is still
WEAK, Qwen2.5-7B is still PASS, and the significant/non-significant pattern across H1–H3 is
unchanged for both models.

**The consequential question — did Llama's WEAK verdict come from AWQ damage? — answers no.**
Llama's length-standardised negative-affect ratio moves 0.68× → **0.78×**, CI [0.71, 0.85].
It rises, but nowhere near the 1.0× that would have overturned the finding. Llama's persuadee
responses are *less* negative with a persona than without one at full precision too. The
spike's central claim does not depend on quantization.

**One post-hoc number does not survive: Qwen's length-adjusted ratio falls 1.65× → 1.11×,**
CI [0.98, 1.26], which now includes 1.0. Qwen's *raw* ratio is unchanged (1.31× → 1.33×), so
this is a failure of the length-standardisation step, not of the affect measurement. §5.4b of
the spike should be shortened to a blunt limitation: the length-adjusted figure is not stable
across precision and should not be leaned on.

---

## ⚠ Deviation from the task spec — Phase 1 gate

**The specified Phase 1 check cannot pass, for a reason that is not drift, and I proceeded
anyway.** This is the one place I did not follow the instructions literally; flagging it first
because everything downstream depends on it.

The spec: regenerate 20 records with the AWQ Llama checkpoint and confirm they *match* the
records in `spike/generations.jsonl` for the same `(persona_id, utterance_id, seed)`.

`spike/gen_lib.py:one()` sends only `{model, messages, temperature, top_p, max_tokens}`.
**`job["seed"]` is a replicate index and never reaches the sampler.** At `temperature=0.7`
with no per-request seed and 10 concurrent workers over SGLang's continuous batching, two runs
of the same cell are independent draws. Byte-identical reproduction was never achievable —
`spike/phase8_report.py` note 4 documents exactly this. Adding a per-request seed to force
determinism would itself violate "everything else byte-identical".

So I ran the specified check, and then the check that can actually discriminate drift:

| Gate | Result |
| --- | --- |
| **A — exact text match, 20 cells (as specified)** | **0 / 20** — expected by construction, not evidence of drift |
| **B — distributional agreement, 318 cells, fresh AWQ replicate vs recorded** | mean tokens 28.73 vs 28.77 (Mann–Whitney p=0.93, KS p=0.56); neg-affect 0.2683 vs 0.2718 (p=0.70, KS p=0.62) |

Gate B is the harness-drift signal that exists, and it passes cleanly. A second, exact check
also passes: `bf16/gen_lib_bf16.py` **imports** `spike/gen_lib` rather than copying it, so
`PREAMBLE`, `build_messages`, `build_jobs` and `SAMPLING` are the same objects the spike ran
with — there is no copy that can drift. Verified: 2,940 jobs, 1,470 per arm, `SAMPLING =
{temperature: 0.7, top_p: 0.9, max_tokens: 64}`.

Stopping here would have spent the GPU budget establishing only that a stochastic harness is
stochastic. Raw data for the gate: `bf16/phase1_verify.json`.

---

## Headline table

Raw ratio = mean P(negative) persona ÷ baseline. Length-std = direct standardisation to the
pooled token-decile distribution. CI = 2,000 within-arm bootstrap resamples, bin edges fixed
at observed deciles. H1–H3 are the **pre-registered** (non-partial) one-sided Spearman tests.

| Model | precision | raw ratio | length-std ratio | CI | H1 | H2 | H3 | verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | AWQ 4-bit | 0.76× | 0.68× | [0.62, 0.74] | ✗ | ✓ | ✗ | WEAK |
| Llama-3.1-8B | **bf16** | **0.83×** | **0.78×** | **[0.71, 0.85]** | ✗ | ✓ | ✗ | **WEAK** |
| Qwen2.5-7B | AWQ 4-bit | 1.31× | 1.65× | [1.48, 1.85] | ✓ | ✓ | ✓ | PASS |
| Qwen2.5-7B | **bf16** | **1.33×** | **1.11×** | **[0.98, 1.26]** | ✓ | ✓ | ✓ | **PASS** |

The AWQ rows are **recomputed from `spike/generations.jsonl` with the same script that
produced the bf16 rows**, so the two precisions are compared by identical code. The
recomputation reproduces the spike: Llama 0.76/0.68, Qwen 1.31/1.65, and Qwen's CI comes back
[1.48, 1.85] against the [1.49, 1.86] in `spike/RESULTS.md` — a rounding-level difference from
a different bootstrap RNG, which also validates the CI implementation added here.

Pre-registered correlations, for completeness — **every one of Qwen's three hypotheses is
stronger at bf16**, and Llama's two null hypotheses flip from slightly negative to slightly
positive without approaching significance:

| Model | precision | H1 ρ (neurotic→neg) | H2 ρ (agreeable→happy) | H3 ρ (open→surprise) |
|---|---|---|---|---|
| Llama-3.1-8B | AWQ | −0.022 (p=.583) | +0.500 (p<.001) | −0.023 (p=.587) |
| Llama-3.1-8B | bf16 | +0.042 (p=.341) | +0.503 (p<.001) | +0.093 (p=.181) |
| Qwen2.5-7B | AWQ | +0.183 (p=.036) | +0.373 (p<.001) | +0.221 (p=.014) |
| Qwen2.5-7B | bf16 | +0.230 (p=.011) | +0.384 (p<.001) | +0.264 (p=.004) |

Permutation gate: p < 0.001 for both models at both precisions.

---

## Negative-affect fraction per arm

Is bf16 simply less neutral overall? No — and the direction is model-specific.

| Model | precision | persona neg | baseline neg | persona neutral | baseline neutral |
|---|---|---|---|---|---|
| Llama-3.1-8B | AWQ | 0.2408 | 0.3174 | 0.4870 | 0.3941 |
| Llama-3.1-8B | bf16 | 0.2649 | 0.3176 | 0.4925 | 0.4226 |
| Qwen2.5-7B | AWQ | 0.2211 | 0.1693 | 0.4646 | 0.4774 |
| Qwen2.5-7B | bf16 | 0.1895 | 0.1423 | 0.4444 | 0.3409 |

Llama's persona arm gains negative affect at bf16 (0.241 → 0.265) while its baseline is
static (0.3174 → 0.3176) — that is the whole of its ratio improvement. Qwen loses negative
affect in *both* arms (persona −0.032, baseline −0.027), which is why its raw ratio barely
moves. bf16 is not uniformly less neutral: Qwen's baseline neutral drops sharply (0.477 →
0.341) while Llama's rises.

## Mean token count per arm

| Model | precision | persona | baseline | persona − baseline |
|---|---|---|---|---|
| Llama-3.1-8B | AWQ | 31.25 | 25.98 | +20.3% |
| Llama-3.1-8B | bf16 | 32.39 | 28.59 | +13.3% |
| Qwen2.5-7B | AWQ | 26.13 | 20.46 | +27.7% |
| Qwen2.5-7B | bf16 | 25.54 | 18.88 | +35.3% |

The persona arm is longer at both precisions in both models, as the spike found. The *size* of
the gap moves in opposite directions (Llama narrows, Qwen widens), which matters for the
length-standardised ratio below.

## Emotion-distribution entropy — testing "4-bit flattens tail behaviours"

Aggregate distribution entropy, bits, max 2.807. If 4-bit suppressed tail emotions we would
expect AWQ entropy **below** bf16.

| Model | arm | AWQ | bf16 | Δ (bf16 − AWQ) |
|---|---|---|---|---|
| Llama-3.1-8B | persona | 2.1943 | 2.1832 | −0.0111 |
| Llama-3.1-8B | baseline | 2.1749 | 2.1702 | −0.0047 |
| Qwen2.5-7B | persona | 2.1334 | 2.1236 | −0.0098 |
| Qwen2.5-7B | baseline | 2.1431 | 2.1919 | **+0.0488** |

**The flattening hypothesis is not supported.** In 3 of 4 arms AWQ entropy is *higher*, the
opposite of the prediction, and every difference is under 0.05 bits against a 2.807-bit
ceiling. Whatever AWQ did to these two models, measurably narrowing the emotion distribution
is not it. This weakens the motivating premise for the whole control — which is worth stating
plainly, since it was the reason the run was commissioned.

## Why Qwen's length-standardised ratio moved

The raw ratio is stable (1.31× → 1.33×); only the length adjustment moves. The cause is a
length–affect coupling that itself changed with precision:

| Model | precision | ρ(tokens, neg) pooled | within persona | within baseline |
|---|---|---|---|---|
| Llama-3.1-8B | AWQ | +0.029 | −0.010 | +0.141 |
| Llama-3.1-8B | bf16 | +0.078 | −0.003 | +0.212 |
| Qwen2.5-7B | AWQ | +0.006 | −0.075 | −0.105 |
| Qwen2.5-7B | bf16 | **+0.340** | +0.021 | **+0.483** |

At bf16, Qwen's baseline responses are both shorter (18.9 tokens) and strongly length-coupled
to negative affect (ρ = +0.483, against −0.105 at AWQ). Standardising to the pooled length
distribution therefore reweights the baseline arm toward long bins where it *is* negative,
inflating the denominator and collapsing the ratio. Per-bin detail:

| Model | precision | persona > baseline |
|---|---|---|
| Qwen2.5-7B | AWQ | **10 / 10** token bins |
| Qwen2.5-7B | bf16 | **7 / 10** token bins |
| Llama-3.1-8B | AWQ | 1 / 10 token bins |
| Llama-3.1-8B | bf16 | 2 / 10 token bins |

The spike's "persona exceeds baseline in 10 of 10 token bins" claim for Qwen — offered as
evidence the length-adjusted figure was robust — **does not replicate at bf16 (7 of 10)**. The
AWQ shortest-token bin also carried an extreme value (persona 0.88 vs baseline 0.63 negative
mass) that is absent at bf16 (0.04 vs 0.07), so the AWQ estimate leaned on a bin that full
precision does not reproduce.

## Cost telemetry — for the Gate 1 budget recalculation

| Model | precision | workers | KV cache | load | 2,940 gens | throughput | peak VRAM |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | bf16 | **10 (stable)** | 27,219 tok | 36 s | 3.6 min | 13.5 gen/s · 411 tok/s | **22,463 MiB** |
| Qwen2.5-7B | bf16 | **10 (stable)** | 75,746 tok | 26 s | 2.6 min | 18.5 gen/s · 411 tok/s | **22,515 MiB** |
| Llama-3.1-8B | AWQ | 10 | — | — | 1.8 min | 26.8 gen/s | — |
| Qwen2.5-7B | AWQ | 10 | — | — | 1.5 min | 32.2 gen/s | — |

**The concurrency ceiling did not drop: 10 workers held for both models at the first attempt**,
with zero request failures and no reduction needed. `--mem-fraction-static 0.82` (unchanged
from the spike) left ample KV headroom — 27 k tokens for Llama at ~450 tokens/request is ~60
concurrent sequences, well above 10. Peak VRAM ~22.5 GB of 24 GB includes ~2.2 GB held by the
desktop compositor. **bf16 costs ~2× wall-clock per generation, not more concurrency
headroom** — budget on throughput (≈0.50× AWQ), not on a lower worker count.

## Example pairs — AWQ vs bf16, same persona / utterance / replicate

Note these are independent draws, not a controlled per-sample comparison (see the Phase 1
deviation). Full set in `bf16/example_pairs.json`.

**1. Llama · persona 0 · greeting · s0 · persona arm**
- AWQ (32t, neg=0.017): *"I'm doing great, thanks for asking. Just got back from a long study session, but I'm trying to reward myself with a nice dinner tonight."*
- bf16 (36t, neg=0.004): *"I'm doing great, thanks for asking. I just got back from a fantastic weekend getaway with friends, and I'm feeling pretty energized and ready to tackle the week."*

**2. Llama · persona 7 · emotion_appeal · s1 · persona arm**
- AWQ (36t, neg=0.662): *"I can understand that, but I'm not sure I'm comfortable with giving to a charity that might be perpetuating the very systems that created this problem in the first place."*
- bf16 (25t, neg=0.075): *"I'm sure it is, but what makes you think donating to Save the Children is the best way to help them?"*

**3. Llama · persona 7 · emotion_appeal · s1 · baseline arm**
- AWQ (24t, neg=0.982): *"It is heartbreaking to think about kids losing their homes and families, but how exactly does Save the Children help them?"*
- bf16 (45t, neg=0.056): *"I can imagine how tough it must be for them, but I've been donating to different charities in the past and it feels like there are so many good causes out there, it's hard to know where to start."*

**4. Qwen · persona 12 · proposition · s0 · persona arm**
- AWQ (30t, neg=0.072): *"I would definitely consider it. Given my values, supporting a cause that helps people and the environment aligns well with how I live my life."*
- bf16 (32t, neg=0.032): *"Sure, I would. Supporting causes that help people and the environment is important to me, and it sounds like Save the Children aligns with those values."*

**5. Qwen · persona 12 · proposition · s0 · baseline arm**
- AWQ (23t, neg=0.035): *"I'm a bit tight on funds right now, but I'll definitely keep them in mind for the future."*
- bf16 (17t, neg=0.059): *"Sure, I might consider it. What kind of impact would my donation have?"*

**6. Qwen · persona 30 · logical_appeal · s2 · persona arm**
- AWQ (24t, neg=0.078): *"I understand the importance of helping children, but with my financial situation, I think I'll pass on this one."*
- bf16 (25t, neg=0.021): *"That sounds like a great way to help, especially since it's such a small amount that can make a big impact."*

Pair 3 illustrates the classifier sensitivity behind the aggregate numbers: two fluent,
on-task baseline responses to the same appeal score 0.982 and 0.056 negative. Much of the
per-cell variance is the emotion classifier reacting to surface wording, not a precision
effect.

## Generation quality

Zero format artifacts in 5,880 bf16 generations across both models: no role prefixes, no
AI-disclosure refusals, no stray chat-template tags, no empty responses. One Llama response
(of 2,940) hit the 64-token cap; Qwen none. Cell coverage is exact — 5,880 unique
`(model, persona_id, utterance_id, seed, arm)` cells, no duplicates, no missing.

## Deviations from the spec, and changes to the analysis code

1. **Phase 1 exact-match gate** — could not pass by construction; ran it, reported 0/20, and
   proceeded on distributional + import-identity evidence. Detailed above.
2. **`phase3b_robustness.py` adds a bootstrap CI that did not exist.**
   `spike/phase7b_robustness.py` computes the point ratio only; the [1.49, 1.86] in
   `spike/RESULTS.md` came from an ad-hoc snippet not preserved in any script. Implemented as
   the task specifies — 2,000 within-arm resamples, bin edges fixed at the observed pooled
   deciles, standardisation weights also held fixed at the observed pooled distribution so
   only sampling error in the per-bin arm means is propagated. Validated by reproducing the
   spike's Qwen CI to rounding.
3. **Input/output paths and the model list are argv-parameterised** in the robustness script,
   so the identical script runs on both `spike/generations.jsonl` and `bf16/generations.jsonl`.
   `phase3_analysis.py` and `phase2b_classify.py` differ from their spike originals **only** in
   path and model-list lines — `diff` against the originals is in the command log and shows no
   other change. No analysis choice was re-tuned.
4. **`--log-level info`** instead of `warning`, to read KV-cache capacity from the server.
   Serving-only; does not touch generation.
5. **Not done:** vicuna-13B at bf16 (out of scope, does not fit in 24 GB).

## Checkpoint provenance

| Role | Repo | Revision | Precision |
|---|---|---|---|
| Llama bf16 | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | bfloat16 |
| Qwen bf16 | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | bfloat16 |
| Llama AWQ | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | `db1f81ad4b8c7e39777509fac66c652eb0a52f91` | 4-bit, group 128, GEMM, zero-point; fp16 compute |
| Qwen AWQ | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `b25037543e9394b818fdfca67ab2a00ecc7dd641` | 4-bit, group 128, GEMM, zero-point; fp16 compute |
| Vicuna AWQ (reference) | `TheBloke/vicuna-13B-v1.5-AWQ` | `ba797e56e4d65473c7fb01cded6bdb79a85dfa68` | 4-bit, group 128, GEMM, zero-point; fp16 compute |
| Emotion classifier | `j-hartmann/emotion-english-distilroberta-base` | as cached (unchanged from spike) | fp32 |

**Calibration corpora are not documented** in any of the three cached AWQ model cards, so the
uncontrolled-calibration concern that motivated this run cannot be resolved from the
checkpoints themselves — only bounded by the result, which shows little precision effect on
the pre-registered metrics.

**Secondary precision difference:** all three AWQ checkpoints declare `torch_dtype: float16`,
so their non-quantized paths computed in **fp16** while the bf16 arm computes in **bfloat16**.
The contrast is therefore "AWQ-int4/fp16 vs bf16", not weight precision in isolation.

Serving: SGLang 0.5.9, torch 2.9.1+cu130, one RTX 4090 (24 GB), `--context-length 2048`,
`--mem-fraction-static 0.82`, `--random-seed 0`, one model resident at a time.

## Files

| File | Contents |
|---|---|
| `gen_lib_bf16.py` | imports `spike/gen_lib`; adds only the bf16 model table |
| `serve_lib.py` | SGLang launch / identity assertion / teardown / VRAM sampling |
| `phase1_verify.py` → `phase1_verify.json` | the drift gate (A and B) |
| `phase2_generate.py` → `raw_*_bf16.jsonl`, `throughput_*_bf16.json`, `spotcheck_*_bf16.json` | generation + telemetry |
| `phase2b_classify.py` → `generations.jsonl` | 5,880 records, 7-dim softmax, `precision: "bf16"` |
| `phase3_analysis.py` → `analysis.json`, `null_*_bf16.png` | pre-registered analysis |
| `phase3b_robustness.py` → `robustness.json`, `robustness_awq_recomputed.json` | length standardisation + bootstrap CI, both precisions |
| `example_pairs.json` | the 6 AWQ/bf16 pairs above |
| `server_*.log` | SGLang server logs |
