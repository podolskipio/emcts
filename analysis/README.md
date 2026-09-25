# `analysis/` — what is current, what is the record

Last updated **2026-09-18**. The project froze on **2026-09-17** (`git tag thu-freeze`).

**If you read one thing:** [`../PREREG.md`](../PREREG.md) for what the grid commits to, and
[`FREEZE_NOTES.md`](FREEZE_NOTES.md) §10 for every frozen value.

## Current — the frozen configuration and the evidence behind it

| file | what it settles |
|---|---|
| [`../PREREG.md`](../PREREG.md) | Entries 1–6: outcomes, difficulty split, arm predictions, budget, doses, valence table, τ, success criterion, and the two arms that died at their gate |
| [`FREEZE_NOTES.md`](FREEZE_NOTES.md) §10 | every frozen value, the code that landed, bit-identity evidence, config checksums |
| [`thu/`](thu/) | the Thursday work: see [`thu/README.md`](thu/README.md) |
| [`../analysis/grid/configs/`](grid/configs/) | the 37 generated grid configs, each validated through the real parser |

Frozen in one line: `--search_horizon episode`, `--emo_valence_table generic`,
`--aff_pool_tau 0.263`, doses Bias 0.70 / CenteredBias 1.03 / Momentum 1.11 (matched on flip rate,
≈15.2 %), `--num_mcts_sims 50` primary with 20 as the contrast, `--p4g_success tag`,
`--emotion_classifier hf`, R 4, K 5, Tmax 10, vicuna-13B-AWQ on SGLang 0.5.9.

## The record — earlier phases, still valid on their own terms

| file | scope | read with |
|---|---|---|
| [`phase1_report.md`](phase1_report.md), [`phase1/`](phase1/) | P-VAR / P-RELABEL gate readings, the search-horizon bug and its pilot | gate numbers are on **legacy** trees under the `soft` table; banner at the top |
| [`phase1/p_depth.md`](phase1/p_depth.md) | why the depth split cannot test the cache; the cache's measured effect on every affect statistic; ν spread inside a node's R-realization pool | same banner as P-VAR; **revises `p_var.md` §7.7's discordance reading** (0.308 → 0.279 on the causal label) |
| [`wed/wednesday_report.md`](wed/wednesday_report.md), [`wed/`](wed/) | the Wednesday arms: AffPool gate, bucket selection, key geometry, momentum, Qwen | τ and the valence table moved at the freeze; banner at the top |
| [`W5_COST_TABLE.md`](W5_COST_TABLE.md) | per-role cost accounting and the SGLang migration | costs measured under **legacy**; episode costs are in `thu/p1_pilot.md` §D |
| [`calib/`](calib/) | server/worker calibration, persona penalty, SR anomaly | |
| [`c4/`](c4/), [`bf16/`](bf16/), [`spike/`](spike/) | classifier audit, bf16-vs-AWQ check, the early spike | |
| [`phase1/SEARCH_HORIZON_BUG.md`](phase1/SEARCH_HORIZON_BUG.md) | the bug, the fix, and P1's result in §6 | current |

## Historical — superseded design and results notes

All carry a dated banner. Kept because they record *why* choices were made, not what is true now:
[`CONTRIBUTION_EMOMCTS.md`](CONTRIBUTION_EMOMCTS.md) (the "+14 pp" headline predates the freeze),
[`EMOMCTS_ANALYSIS.md`](EMOMCTS_ANALYSIS.md), [`EMOMCTS_RESEARCH_DIRECTIONS.md`](EMOMCTS_RESEARCH_DIRECTIONS.md),
[`EmotionMCTSDoubleQ.md`](EmotionMCTSDoubleQ.md), [`EMOMCTS_ALGORITHM.md`](EMOMCTS_ALGORITHM.md),
[`historical_emotions_prior_mcts.md`](historical_emotions_prior_mcts.md),
[`negative_emotions_donate_happy_dont.md`](negative_emotions_donate_happy_dont.md) (its "fear predicts
donation" headline does **not** survive the leakage cut),
[`models_personality_p4g.md`](models_personality_p4g.md), [`article_action_analysis.md`](article_action_analysis.md),
[`article_examples.md`](article_examples.md), [`debug.md`](debug.md).

## Three results that changed what the paper can claim

1. **Outcome leakage in `w(e)`.** The mined valence table was fitted partly on turns at or after the
   donation decision: about two-thirds of `w(happiness)` comes from them. Cut at the decision, **no
   emotion's interval excludes the base rate** under any turn weighting (42 tests, 0 hits after
   correction), while the same estimator still finds the post-decision effect. The grid therefore runs
   the never-fitted `generic` table. `thu/remine.md`, `thu/construct_test.md` §3b.
2. **The search horizon was wrong, and it mattered.** Search ignored the episode horizon, so it
   expanded states no episode can reach and never charged for stalling. `thu/p1_pilot.md`.
3. **The success detector is lenient.** The simulator's `[donate]` self-tag accepts hedged
   non-commitments ("I will consider donating and will be in touch"); the genuine hedge rate is ~4 % of
   successes. Success also ends search branches, so a strict-but-imperfect detector inside the tree was
   judged worse than the leniency. `thu/success_criterion_fix.md`.

## Not regenerable from this repo

Per-run `*_emotions.json` (utterance → emotion dumps, up to 191 MB) and `*.log` are git-ignored; re-run
the command in each run's `metadata.json` to regenerate them. Everything the analyses read — parquet
tables, simlogs, run metadata, episode pickles — is committed.
