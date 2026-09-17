# Recomputed grid budget — replaces the plan's estimates

Computed by `calib/budget.py` from `calib/throughput.json`. Every cell cost below is a
measurement, not an estimate. Cell size N = 100 dialogues (the README results table).

## Verdict

| | hours |
|---|---:|
| factorial (measured 13B/no-persona anchor) | **77.5** |
| factorial (plan's 10 h anchor, for comparison) | 89.3 |
| A1 extra seeds 20 · A2 18 · A3 1 · B1 28 · B2 30 | 97 |
| **recomputed total** | **174.5 h** |
| plan estimate | 227 h |
| available, Weeks 2–3 | 280 h |
| **slack** | **105.5 h** |

**Band: < 230 h → Proceed. Spend the slack on a 2nd seed for B2 first.** The band is the same
whichever 13B/no-persona anchor is used, so the decision does not turn on that substitution.
A 2nd B2 seed (~30 h) still leaves ~75 h.

## Measured cell costs (100 dialogues each, 10 workers)

| cell | source | s/turn | turns/dlg | **h / cell** | count | subtotal |
|---|---|---:|---:|---:|---:|---:|
| 8B, no persona | `run1_d24_w10`, 24 dlg scaled ×100/24 | 12.78 | 4.4 | **1.55** | 6 | 9.3 |
| 8B, persona | `run2_d24_w10`, 24 dlg scaled ×100/24 | 20.08 | 6.8 | **3.76** | 6 | 22.6 |
| 13B, no persona | `confirm_13b_nopersona_d100`, **measured at 100** | 36.00 | 6.1 | **6.08** | 3 | 18.2 |
| 13B, persona | `run3_d24_w10`, 24 dlg scaled ×100/24 | 49.54 | 6.6 | **9.12** | 3 | 27.4 |

## The plan's 10 h anchor is 1.6× too high

The 13B/no-persona cell was run at full size rather than assumed: **6.08 h**, not 10 h. The
factorial charges three of them, so the plan's arithmetic carried ~11.8 h of phantom cost.
Both figures are shown above; the measured one is used for the headline.

## Scaling assumption, and which way it is wrong

Three of the four cells are scaled from 24-dialogue runs by ×100/24, which assumes constant
cost per dialogue. It is not constant, and the error has a known sign:

| 8B / no persona | per dialogue | s/turn |
|---|---:|---:|
| 10 dialogues (`run1_w10`) | 78.9 s | 17.15 |
| 24 dialogues (`run1_d24_w10`) | 55.9 s | 12.78 |

Larger runs are **cheaper** per dialogue, because a worker that finishes a short dialogue
immediately picks up the next one and the pool stays saturated, where a 10-dialogue run at 10
workers idles as it drains. Extrapolating from 24 to 100 therefore **over**-estimates the
three scaled cells. The budget is conservative in the safe direction, and the one cell
measured at full size needed no extrapolation. No correction factor is applied, because two
points do not fix the shape of that curve.

## Fixed configuration these costs are valid for

n_sims 50 · K 5 · R 4 · Tmax 10 · depth cap 10 (`max_conv_turns`, wired to `--max_turns`) ·
β 0.7 · `--emo_signal level` · `--emo_risk_lambda 0.0` · soft valence table · hf emotion
classifier · `--logit_scoring both` · `--explicit_value_labels` OFF · AWQ 4-bit both
checkpoints · temperature 0.7/1.1, top_p 0.9, top_k 40, max_new_tokens 128 · **10 workers** ·
SGLang ctx 4096, mem-fraction-static 0.82, chunked-prefill 4096, cuda-graph-max-bs 64,
schedule-policy lpm. Full snapshot in `run_calib.py:GRID` and in each row's `grid` key.

Changing any of these invalidates the table. The two most cost-sensitive are the depth cap
and R.

> **Erratum (2026-09-14).** `max_conv_turns` / `--max_turns` is **not** a search depth cap. Tree search treats only donation as terminal, so the −1.0 the game returns at the turn limit is ignored and search expands states past Tmax (19.7 % of simulation steps in the Phase-1 D1 run). Costs and behaviour in this document were measured under that behaviour. Fixed behind `--search_horizon episode`; see `analysis/phase1/SEARCH_HORIZON_BUG.md`.


## Three findings that affect what to buy, not what it costs

1. **The 8B/no-persona cells cannot show an effect.** SR = 1.000 at n=10 and n=24 — every
   dialogue succeeds, in 4.4 turns. A cell pinned at ceiling has no variance for β or the
   emotion channel to move. Six such cells cost 9.3 h and are incapable of producing a result.
   See `calib/SR_ANOMALY.md`.
2. **The no-persona cells have no scenario variance at all.** p4g self-play carries
   `scenario = ()`; the dialogue id only labels the episode and looks up a persona. Without
   `--p4g_persona` all 717 dialogues are the same starting condition, so a no-persona cell's
   error bars are Bernoulli noise over one scenario, not variation across 100 situations.
3. **More concurrency cannot buy time back.** Throughput degrades above 10 workers and the
   ceiling is not KV capacity (peak 16.7% of the pool on the 8B). See `calib/SWEEP.md`; the
   untaken lever is fewer, larger requests, not more workers.
