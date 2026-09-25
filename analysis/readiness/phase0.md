# Readiness Phase 0 — commit, verify, pin the retention condition

**Status: PASSED.** Tag `readiness-start` = `6c26862`.

## Code state

The four cache fixes were already committed before this program started, in one commit
(`e6feda7 "fixes"`), not uncommitted as the brief assumed. It was not rewritten; the remaining
pieces went in as separate commits on top:

| Commit | What |
|---|---|
| `e6feda7` | the four cache fixes (`--cache_ended_children`, `--cache_fresh_depth1`, `--cache_draw {bucket,kernel,bucket_kernel}` + `--cache_bucket_tau/--cache_kernel_h`), runner flags, inert-flag refusal |
| `05816aa` | the P-DEPTH analysis those fixes cite (`analysis/phase1/p_depth.md`) |
| `9986db0` | `tests/test_cache_fixes.py` — the fixes' own tests (were untracked) |
| `6c26862` | `tests/test_ended_draw_rate.py` — the Phase 0 condition below |

| Fix | Flag | gdpzero | emomcts |
|---|---|---|---|
| #6 retention | `--cache_ended_children` | yes | yes |
| #5 depth-1 keeps every reply | `--cache_fresh_depth1` | yes | yes |
| #1 bucket by generating parent | `--cache_draw bucket --cache_bucket_tau T` | — | yes |
| #2 freshness-weighted draw | `--cache_draw kernel --cache_kernel_h H` | — | yes |
| #1 + #2 | `--cache_draw bucket_kernel` | — | yes |

## Test suite

`python3 -m pytest tests -q` → **139 passed** (the 136 of the brief + the 3 draw-rate tests).
Golden-tree fingerprints unchanged with every new flag off
(`test_cache_fixes.py::test_explicit_defaults_reproduce_the_frozen_tree[beta0.0|beta0.7]`,
plus the pre-existing `test_wed_arms.py::test_default_search_reproduces_the_prechange_tree`).

## Ended-reply draw rate

Condition: `P(draw an ended reply at an edge) = (# ended generated) / (# total generated)`.

Test (`tests/test_ended_draw_rate.py`): one edge filled through the planner's own
generate-then-enter path with 3 non-terminal and 2 ended (donate) replies, `max_realizations = 5`
so the edge caches exactly when all five are in; then 10,000 cache hits.

| Path | Flag | Pool (live + ended) | Ended fraction drawn | Target |
|---|---|---|---|---|
| `OpenLoopMCTS._get_next_state` (gdpzero) | on | 3 + 2 | **0.4037** | 0.40 ± 0.02 ✔ |
| `EmotionAwareMultiObjectiveQ._draw_cached_child`, uniform (emomcts) | on | 3 + 2 | **0.4037** | 0.40 ± 0.02 ✔ |
| frozen cache | off | 5 + 0 (edge keeps generating until 5 live) | **0.000** | — the bias |

No extra generation: all 10,000 draws were cache hits (`cache_hits[edge] == [10000, 5]`).

Caveat on the equality in a real tree: the pool deduplicates identical replies (both the live
and the ended list), so two byte-identical generations count once in the pool and twice in the
generation tally. With T = 1.1 sampling, exact duplicates are rare; Phase 2 measures the
draw vs generation fraction on the real logs.
