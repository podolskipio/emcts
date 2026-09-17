# Plan §4c — Run Table (episode costs, Thu Sep 17 revision)

> Copied verbatim from the human's attachment (2026-09-17). Corrections and confirmations are in
> `plan_4c_review.md`. Configs are generated from `scripts/gen_grid_configs.py`, never from this text.

Supersedes the earlier §4c. Rebuilt on measured `episode` per-dialogue costs, which
are roughly 40 % below the legacy estimates the old table assumed.

---

## Measured costs (episode, Vicuna-13B, persona, n_sims=50, per dialogue)

| Arm | s/dialogue | h per 100 |
|---|---|---|
| NoEmo | 231 | 6.4 |
| Momentum | 241 | 6.7 |
| CenteredBias | 279 | 7.8 |
| ActPool | 280 | 7.8 |
| AffPool | ~280 (assume) | 7.8 |
| Bias | 294 | 8.2 |

`n_sims=20` runs scale roughly ×0.4. **Remeasure AffPool rather than assuming.**

---

## Shared arguments — every grid run, explicit, no inheritance

### Why every one is written out

Runner defaults differ from the grid config in five places. A run that inherits any
of them does not crash and does not look wrong in the log — it silently produces a
cell measured under different conditions, discovered only after the GPU hours are
spent.

| Argument | Runner default | Grid value | What inheriting it would cost |
|---|---|---|---|
| `--emotion_classifier` | `llm` | **`hf`** | Labels from a different model than D1/D2/D3, the C4 audit, the mined tables and τ_med. Also adds a generation call per user turn and sampling non-determinism to the label stream; HF is one deterministic forward pass and stays off the LLM budget, which is a line in the cost table. |
| `--n_sims` | `20` | **20 or 50, per block** | Silently runs the wrong budget arm. |
| `--max_realizations` | `3` | **4** | Changes the cache mismatch rate (1 − 1/R) and every pooling statistic. |
| `--llm_prior_topk` | none | **5** (0 only for B1) | No top-K pruning; a different action space. |
| `--seed` | unset | **1, 2 or 3** | Unreproducible, and seed-paired comparisons break. |

**The `--frozen_config` check must fail the run** if any resolved value differs
from the frozen template. Do not rely on the template alone — the assertion is what
catches a hand-edited config.

### The block, verbatim

```
--search_horizon episode          decided; legacy is B2, a disclosed ablation
--max_turns 10
--max_realizations 4              NOT the runner default of 3
--K 5
--llm_prior_topk 5                0 only for B1, plain GDP-Zero
--emotion_classifier hf           NOT the runner default of llm
--emo_valence_table generic       final; soft and predecision reported in §5
--cpuct 1.0
--Q0 0.0
--n_sims 20                       or 50 — see Block A
--seed 1                          or 2, 3
⚠ persona flag                    confirm spelling
⚠ eval-set flag                   first 100 non-annotated
```

⚠ Arm flags are declared **per runner** (`rollout.py`, `emomcts.py`), not in
`add_common_args`. Confirm each spelling before generating configs and report
corrections.

### Doses — measured under `generic`, final

```
Bias          --beta_emo 0.70   --emo_signal level
CenteredBias  --beta_emo 1.03   --emo_signal level  --emo_centre
Momentum      --beta_emo 1.11   --emo_signal delta
```

Bias keeps **β 0.7** rather than being re-anchored. Under `generic` it changes
NoEmo's pick in **15.2 %** of selections — *above* the 11.9 % the pilots validated
rather than below it, so the arm is not under-dosed, the three arms sit within
~1.1× of each other, and 0.7 is the published value. One less post-hoc choice to
declare.

CenteredBias and Momentum are anchored to **Bias's achieved 15.2 %**, not to a
target picked in advance.

**Declare in PREREG:** arms are matched on flip rate rather than on β; β differs
per arm; each achieved flip rate is reported alongside its β.

Record in FREEZE_NOTES the **visited-edge mean of `Q_emo` under `generic`** — it is
how much of Bias remains an exploration term, and CenteredBias and Momentum exist
to separate it. For reference, Bias's pick-change rate is 15.2 % under `generic`,
11.9 % under the shipped `soft` table, 7.5 % under `predecision`.

## Block A — the full budget factorial (primary)

Both budgets, every arm. The budget curve stops being a separate block and becomes
the design — which is what the cost drop buys.

Every run also carries the full shared block above. Arm-specific arguments are
**only** what is listed here — nothing else differs between arms.

| Arm | Seeds | Arm-specific arguments | n_sims=20 | n_sims=50 |
|---|---|---|---|---|
| **NoEmo** | 3 | `--beta_emo 0` | 7.7 h | 19.2 h |
| **Bias** | 3 | `--beta_emo 0.70 --emo_signal level` | 9.8 h | 24.6 h |
| **Momentum** | 2 | `--beta_emo 1.11 --emo_signal delta` | 5.4 h | 13.4 h |
| **CenteredBias** | 2 | `--beta_emo 1.03 --emo_signal level --emo_centre` | 6.2 h | 15.5 h |
| **AffPool** | 2 | `--beta_emo 0 --aff_pool --aff_pool_bias 0.25 --aff_pool_tau 0.263 ⚠--aff_pool_key affect` | 6.2 h | 15.6 h |
| **ActPool** | 2 | `--beta_emo 0 --aff_pool --aff_pool_bias 0.25 ⚠--aff_pool_key act` | 6.2 h | 15.6 h |
| | **14 runs each** | | **41.5 h** | **103.9 h** |

`--aff_pool_tau` is not passed to ActPool — the key has no bucket, so τ is unused.
Pass it and the config check should flag the contradiction.

Doses and τ are **final**, measured under `generic` on the `episode` pilot trees.
Re-derive τ only if the grid config differs from those pilots.

**Block A total: 145 h.**

`n_sims=20` is **primary** if the n=100 saturation run confirms a ceiling at 50;
otherwise 50 is primary and 20 is the budget contrast. Decide when the run lands,
before the freeze, and record which in PREREG.

**τ_med = 0.263** — the median under `generic` on the `episode` pilot trees, and it
splits steps 50/50. Superseded values, neither of which may be used: 0.35 (frozen
under `soft`/`legacy`, puts 93 % in one bucket) and 0.078 (`predecision` under
`episode`). τ is the median of the observed ν distribution, so it moves with both
the table and the horizon — re-derive if either changes again.

---

## Block B — baselines and replication

| # | Run | Args | h | Why |
|---|---|---|---|---|
| B1 | **GDP-Zero, plain** | `--beta_emo 0 --llm_prior_topk 0` — the only run where top-K is off | 8 | The actual published baseline. NoEmo is GDP-Zero **+ top-K**, not GDP-Zero. If the paper's origin is "affect beat GDP-Zero," that row has to be in the table. |
| B2 | **Legacy horizon ablation** | `--beta_emo 0 --search_horizon legacy` — the only run not on `episode` | 11 | Discloses the P1 change with a number. Compare on the common 2669-root subset. |
| B3–B5 | **Second backbone — Qwen2.5-7B-Instruct**, 100 dialogues, `n_sims` as primary | shared block with the Qwen model flag; arm args as Block A for NoEmo · Bias · ActPool | ~12 | Restored to full scale — R1 called single-backbone a "huge experimental flaw." |

**Block B total: ~31 h.**

### Why Qwen is the second backbone, and Llama the third

**Qwen is already characterized.** D3 gives 30 dialogues, integrity-checked,
`backup_value` native, paired on the same dialogues as D1 — a calibrated starting
point, a known cost, and a known failure mode. Llama would be run cold.

**Qwen answers a question Llama cannot.** D3 measured that Vicuna's happiness drift
is *not* universal: median ν flat at 0.06–0.10 across depths, against Vicuna's
0.24 → 0.53. The median split, τ_med and the momentum controls were all designed
around that drift. Running the second backbone on a model **shown** to lack it
tests whether those choices generalize; Llama is an unknown on that axis.

State this in §4 as the reason for the choice, rather than leaving it to look
arbitrary.

**Llama is the third backbone, optional — C5 in Block C.** Two backbones answer
R1's "huge experimental flaw"; a third is strengthening, not load-bearing, and it
is the first thing to drop if anything slips.

---

## Block C — slack, in priority order

~198 h capacity − 176 h (A + B) = **~22 h**. C1–C4 fit; **C5 does not** without
taking the slack from C2 and C3. Order below is the priority order.

| # | Run | h | Why |
|---|---|---|---|
| C1 | **P2 tagged-cache profiling** — implement behind a flag, 2 dialogues with `--profile_roles` **under episode** | 1 | Gives the real cost multiplier instead of the unresolved +11 % to +88 %. |
| C2 | **P2 pilot** — NoEmo ± tagged cache, 30 dialogues, only if C1's multiplier ≤ ~20 % | 8 | The single number deciding whether the next paper is about caching or affect. |
| C3 | **Third seeds** — Momentum, ActPool, AffPool at n_sims=20 | ~9 | Power on the arms carrying the claim. |
| C4 | **`predecision` valence ablation** — Bias, one seed, `--emo_valence_table predecision` with τ and dose re-derived for that table | 8 | Makes the leakage finding an experiment rather than a corpus analysis. |
| **C5** | **Third backbone — Llama-3.1-8B**, 100 dialogues, `n_sims` as primary, NoEmo · Bias · ActPool. **Pilot 5 dialogues first** — no cost measurement exists for Llama, and it may saturate or fail differently. Re-derive τ_med from its own ν distribution; do not inherit Vicuna's 0.078. | ~14 | Makes the leakage finding an experiment rather than a corpus analysis. |

---

## Order

Any prefix is submittable; the first block unlocks the human study.

1. **n_sims=20, seed 1, all six arms** (~21 h) — the full arm set at the primary
   budget. Human study starts as soon as the first three land.
2. **n_sims=50, seed 1, all six arms** (~52 h) — the ceiling contrast, and the pair
   AffPool/ActPool at the budget where pooling should matter least.
3. **B1, B2** (~19 h) — baselines, so the comparison anchor exists early.
4. **Seed 2, both budgets** (~52 h).
5. **B3–B5** Qwen, the second backbone (~12 h).
6. **Seed 3 on NoEmo and Bias, both budgets** (~20 h).
7. **Block C** in priority order: C1 → C2 → C3 → C4 → **C5 (Llama) last**.

**C5 is the third backbone and is explicitly optional.** Two backbones answer the
single-backbone complaint; the third strengthens and does not carry anything. It
is the first item dropped if any earlier block overruns, and it needs a 5-dialogue
pilot before its 14 h are committed — no Llama cost measurement exists, and τ_med
must be re-derived from its own ν distribution rather than inheriting Vicuna's
0.078.

**Hard stop Mon Sep 28, 23:59.**

---

## Not in the grid

| | Reason |
|---|---|
| **TrajValue, TrajPrompt** | Both gates failed: increment 0.0125 [0.004, 0.055] below the 0.02 bar, out-of-fold AUC 0.698 → 0.697, Brier −0.002 [−0.021, +0.017], two cells flipping sign in 2 of 5 folds. PREREG Entry 4. |
| **Constrain** | Root decision disagrees with unrestricted `argmax N` on 19 % of turns. §3 as a specified formulation. |
| **Risk** | σ̂ not estimable at median N=6 with 22 % at N=2. |
| **NodeKey** | Cache mismatch, 1 − 1/R. §3 and §5 as a result. |
| **`predecision` as default** | No pre-decision emotion signal is detectable under uniform, last-turn or recency weighting — 42 tests, 0 hits after multiplicity correction, with a control confirming the method finds the post-decision signal. Shipping it would steer the planner toward fearful users on the thinnest cell. Reported in §5; C4 if slack allows. |
