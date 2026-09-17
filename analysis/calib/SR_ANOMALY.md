# SR check — does the pipeline still reproduce the published 0.700?

Raised because run 1 (8B, no persona) returned SR 1.000 / AT 4.6, against a published
0.700 / 6.72. Resolved by running the anchor cell at full size.

| config | dialogues | SR | AT |
|---|---:|---:|---:|
| published README row (Vicuna-13B, **pre-fix Ollama Q4-GGUF**, no persona) | 100 | 0.700 | 6.72 |
| **Vicuna-13B AWQ, no persona, current pipeline** | **100** | **0.770** | **6.26** |
| Vicuna-13B AWQ, persona | 24 | 0.708 | 6.62 |
| Llama-3.1-8B AWQ, no persona | 24 | **1.000** | 4.38 |
| Llama-3.1-8B AWQ, persona | 24 | 0.708 | 6.75 |

**The pipeline reproduces the anchor.** 0.770 against 0.700 at n=100 is 1.7 standard errors
(SE = 0.042); not a significant difference. There is no pipeline defect.

**The eager-persuadee behaviour is specific to Llama-3.1-8B**, not a property of the
no-persona arm. Vicuna-13B without a persona sits at 0.770, close to published; the 8B
without a persona is pinned at 1.000. An earlier reading of this — that no-persona was the
artifact and the persona arm explained the gap — was wrong, and the 0.708 agreement between
the two persona runs is coincidence rather than mechanism.

Candidates checked and cleared along the way:

* **Success criterion** — unchanged: `U_Donate` in any user DA (`p4g_game.py:88`). Transcripts
  are coherent (greeting → neutral → positive reaction → "I'll donate $5" at turn 4).
* **Which dialogues** — cannot matter in the no-persona arm: `scenario = ()`, so all 717 are
  the same starting condition and n=10 vs n=100 differ only by sampling.
* **Logit scoring / the ≥0.95 check** — it was run (`scripts/validate_logit_scoring.py`, 200
  states). **The value prompt as shipped scores r = 0.9447 and misses the bar**; it clears
  only with `--explicit_value_labels` (0.9504 / 0.9544 / 0.9570), which is off by default and
  off in these runs, by decision, to keep continuity with the existing W5 rows. Recorded here
  because the grid ships a value estimator that does not meet the spec's stated criterion.
* **Re-mined soft w(·)** — not isolated. Both arms use it, so it cannot explain a difference
  between them, and the 13B reproduction bounds any effect it has at the anchor.

**Note on the README.** `README.md`'s results table is a pre-fix Ollama measurement, produced
before W5 fixes 1–17 — including fix 7, under which the top-K prior put ~96.5% of its mass on
a single dialogue act. `analysis/W5_COST_TABLE.md:16` already states that every measurement
predating those fixes is superseded. The README table has not been regenerated and is the
only place that number is still presented as current.
