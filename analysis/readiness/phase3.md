# Readiness Phase 3 — coupled seeds (3A) and the node-level spread test (3B)

Pre-registered in `PREREG.md` Entry 10 (`64adafa`).

| | Result | Verdict |
|---|---|---|
| **3B** node-level emotional spread | predicts Q slightly beyond mean ν (ΔR² 0.008), but does not differ between sibling actions beyond sampling noise (ratio 1.04) | **FAIL** (as predicted) |
| **3A** coupled seeds | built (`--coupled_seeds`), both acceptance checks pass; arms diverge at turn 1–2 | **PASS** |

**3B failed.** Under brief v2 §3 that makes the verdict **RED** whatever Phase 4 shows. Phase 6
(loop-closing) is skipped. Phases 4–5 still run for the infrastructure results.

---

## 1. 3B — does a node's emotional spread carry value information?

Script `scripts/phase3_spread.py`, numbers `phase3_spread.json`. Zero GPU.

**Setup.**
- **Node:** a node is an edge (tree, prefix, action).
- **Spread:** the sd of ν across the node's distinct generated replies; mean ν is their mean.
- **Outcome:** Q, the mean backed-up task value over the node's visits.
- **Kept nodes:** ≥ 2 replies and ≥ 2 visits.
- **Primary data:** the frozen `A_NoEmo_s20_seed{1,2,3}`, the current environment with β = 0.
- **Secondary data:** phase1 D1 and D2, legacy horizon and soft table.

| | Primary (NoEmo s20 ×3) | D1 (β 0.7) | D2 (β 0) |
|---|---|---|---|
| nodes kept | 10,650 | 3,342 | 2,871 |
| spread: median [p10, p90] | 0.159 [0.011, 0.328] | 0.145 [0.008, 0.261] | 0.153 [0.010, 0.261] |
| spread CV across nodes | 0.77 | 0.67 | 0.65 |
| median spread / global ν sd | 0.62 | 0.73 | 0.75 |
| corr(spread, mean ν) | −0.58 | −0.45 | −0.41 |
| **Step 2**: ΔR² of spread over mean ν (depth FE, n_real) | **0.0079** [0.0046, 0.0122] ✅ | 0.0042 [0.0006, 0.0120] | 0.0073 [0.0019, 0.0133] |
| standardized coefficient of spread | +0.11 | +0.07 | +0.10 |
| Step 2 within parent (siblings demeaned) | 0.0018 [0.0001, 0.0051] | 0.0027 [0.0000, 0.0205] | 0.0073 [0.0018, 0.0187] |
| **Step 3**: sibling variance of spread / permutation null | **1.04** (p 0.050) ❌ | 0.94 (p 0.97) | 1.16 (p 0.005) |
| parents with ≥ 2 kept siblings | 2,063 | 570 | 591 |
| **3B verdict** | **FAIL** (step 3) | — | — |

What it says:

- **Spread varies across nodes, but almost all of that variation is sampling noise.** Its CV is 0.77,
  and its p10–p90 runs from near zero to 1.3× the global ν sd. Among sibling actions it varies about as
  much as it does overall: variance 0.015 against 0.016. But shuffling the replies across a parent's
  siblings produces 0.0145, because an sd estimated from 2–4 replies swings on its own. Real differences
  between actions add 4 % on top of that noise, against a pre-registered bar of 20 %. **Spread cannot tell
  one action from another at the same node**, and that is what a term in the value would have to do.
- **Spread does carry a sliver of value information** (ΔR² 0.008, CI above zero, positive sign). Within
  a parent it shrinks to 0.002. The positive sign is at least partly mechanical: a pool that mixes a
  success reply (high ν, return +1) with ordinary ones has both high spread and high Q. The frozen logs
  cannot separate ended replies, so this is not removed.
- **D2 comes closest** (ratio 1.16, p 0.005), and it is still under the bar. D1 is at the null.
- The PREREG prediction (FAIL; ratio near 1; small positive step-2 increment) held on all three counts.

**Four independent measurements now find little affective information here:**
- C1a, r over v: 0.017, below the 0.02 bar.
- C1b, r over turn count: 0.0097, and that baseline has since been shown to leak.
- The corrected ω² ceiling: 1.3 %.
- Node-level spread: 0.8 % of variance, not discriminative between actions.

Per Rule 9, this is "not detected at this resolution". It is not "no effect".

## 2. 3A — coupled seeds

### What was built, and why it is not a server seed

The brief asked for per-call seeding by (dialogue, turn, node). **SGLang cannot do that on this
machine.**
- **Seeds are ignored by default.** SGLang 0.5.9 ignores a request's `seed` unless the server runs with
  `--enable-deterministic-inference`. Measured: identical requests with one seed gave different text
  back to back, and 0 of 8 matched under load.
- **Deterministic mode does not start on this GPU.** It needs batch-invariant Triton matmuls, which
  request 106,496 B of shared memory against the RTX 4090's 101,376 B limit.
- **Deterministic mode would also change the sampler.** It seeds by (seed, position), so the value
  estimator's `n = 10` samples in one request would all come back identical.

**Coupling is therefore done on the client, as common random numbers** (`src/utils/coupling.py`,
commits `1353a7a`, `9a954ba`, `284d94c`, `ebbd7eb`).

- **Reply store.** Every SGLang call inside a coupled dialogue is keyed by
  (seed, dialogue, scope, exact request, occurrence *k*). The keys live in a shared sqlite store: the
  first run to make a call records the reply, and any run making the same call gets it back.
- **Scope.** The scope is ("search", *t*) or ("episode", *t*). Without it, the executed turn's request
  would inherit an occurrence index from however often that arm's search sent the same prompt. Two arms
  choosing the same act would then still get different utterances. The first attempt had this bug; it
  was caught before any result was read, and those runs are archived under `runs/_interrupted/`.
- **Tree draws.** Realization sampling and cache draws come from a `RandomState` per
  (seed, dialogue, turn). They no longer use the numpy stream the 10 workers interleave on.
- **Distributions are unchanged.** Each distinct key is one ordinary draw from the model; only the draws
  are shared. It also removes GPU numeric nondeterminism, which a server seed would not.
- **Off by default.** With the flag off, every function is a pass-through, the golden fingerprints are
  unchanged, and there are 150 tests (9 new for coupling).

### Acceptance (PREREG Entry 10)

Three runs, all on one store and seed 1: `P3A_NoEmo_a`, `P3A_NoEmo_b` and `P3A_ActPool`. Each ran 5
dialogues at s20 with `--num_workers 10`. NoEmo has #6 + #5; ActPool adds the #1 + #2 draw. Script
`scripts/phase3_coupling_check.py`, numbers `phase3_coupling.json`.

| Check | Result | Verdict |
|---|---|---|
| (0) flag off: fingerprints and all tests unchanged | 150 pass | ✅ |
| (i) same arm, same seed, twice: every turn's act and utterance identical | 5 / 5 dialogues; the re-run was served entirely from the store (3,192 hits, 0 misses, 0.0 h) | ✅ |
| (ii) NoEmo vs ActPool: identical up to the first differing act | 5 / 5 dialogues | ✅ |

### The limit it shows

**The two arms diverge immediately.** The first differing act comes at turn 1 in 3 of 5 dialogues and at
turn 2 in the other 2. Only **11.9 %** of ActPool's LLM calls (274 of 2,298) were served from replies
NoEmo had already drawn. Coupling removes shared noise only up to the first divergence, and here that
point comes almost at once. So Phase 4 may find that coupling buys little correlation between these two
arms. The brief anticipated this ("it may not reach 0.7").
