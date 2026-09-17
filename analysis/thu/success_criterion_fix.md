# Success criterion: saturation check, the hedge-detector fix, and why `committed` stays offline

## 1. Saturation: not saturated

NoEmo, `--search_horizon episode`, n_sims 50, vicuna-13B, dialogs **141–240** of the non-annotated
pool (not the eval set, so no evaluation design choice was tuned on eval). 100/100 dialogs completed,
wall clock 24,230 s (**242 s/dialog**), one server start.

| criterion | SR (n = 100) | 95 % Wilson |
|---|---|---|
| **tag** (environment) | **0.78** | [0.69, 0.85] |
| committed (v2 detector, offline) | 0.75 | [0.66, 0.82] |
| amount named | 0.49 | [0.39, 0.59] |
| amount ≤ $2 | 0.03 | [0.01, 0.08] |

The upper bound 0.85 is below the 0.95 saturation threshold. **Primary budget n_sims = 50; 20 is the
contrast** (the plan's pre-set rule). Amount criteria are dropped: they measure whether the simulator
states a number, and it does not know about the $2 task payment.

**⚠ Ten-dialog pilots in this environment can be off by more than 0.2.** The runner's cumulative SR on
this set was 1.00 after 10 dialogs, 0.83 after 60, 0.81 after 90 and **0.78 after 100**. The P1 horizon
contrast (episode 10/10 against legacy 6/10) rests on 10 dialogs and **must be re-measured at n = 100 in
B2** before any number from it enters the paper.

## 2. The detector fix (v1 → v2)

**v1** matched a hedge word anywhere in the `[donate]` turn. It flagged "After **considering** the
information provided …, I have decided to donate $5", where the hedge does not govern the donation.

**v2** (`src/games/p4g_success.py`, shared by the environment and the offline scorer) calls a turn
hedged only if it contains

- a hedge word **governing a donation verb** within 3 words ("consider making a donation", "possibly
  donate", "might give"), **or** a deferral phrase ("look into it", "see how I can", "check my finances",
  "see what I can afford", "be in touch", "get back to you", "think about it"),
- **and no firm first-person commitment** anywhere in the turn ("I have decided to donate", "I would
  like to donate", "I'll donate", "I am ready to donate"). Only intensifiers may sit between the modal
  and the verb, so "I will consider donating" is not firm and "I will definitely donate" is.

### Hand labels — all 134 lenient successes in the logged runs (n = 100 run + 7 pilot arms)

Every turn either version flags, plus a scan of the 126 unflagged turns for deferral/intent language (45
candidates read in full). H = hedged non-commitment, C = committed, A = ambiguous.

| run / dialog | turn (abridged) | label | v1 | v2 |
|---|---|---|---|---|
| n100 / …225 | "I will definitely look into it and see how I can donate" | H | ✓ flag | ✓ flag |
| n100 / …603 | "Yes, I will consider making a donation" | H | ✓ | ✓ |
| n100 / …898 | "definitely interested in donating. Let me check my finances and see what I can afford" | H | ✗ **missed** | ✓ |
| legacy / …206 | "I will definitely consider donating" | H | ✓ | ✓ |
| NoEmo pilot / …705 | "I will consider donating and will be in touch" | H | ✓ | ✓ |
| Momentum pilot / …442 | "check out their website and consider making a donation" | H | ✓ | ✓ |
| n100 / …134 | "After considering the information …, I have decided to donate $5" | C | ✗ **false positive** | ✓ not flagged |
| AffPool pilot / …442 | "Yes, I would like to donate … I'll look into the options for online and mail donations" | C | ✗ **false positive** | ✓ not flagged |
| n100 / …458 | "I would like to donate … will check out their website and possibly donate online" | A | flagged | not flagged |
| n100 / …253 | "I am also ready to donate, but I want to make sure that … money is being used effectively" | A | not flagged | not flagged |

| | v1 | **v2** |
|---|---|---|
| hedges caught (of 6) | 5 | **6** |
| false positives (of 2 committed turns flagged by either) | **2** | 0 |
| on the n = 100 run's 4 v1 flags | **2 genuine, 1 spurious, 1 ambiguous** | — |
| flags outside the labelled set | 0 | 0 |

**Honest limits of v2.** It was written while reading these same 134 turns, and the labels are mine
(one reader, no second annotator). The table is a fit, not a validation. Two turns are genuinely
ambiguous, and v2 resolves both as committed because a firm "I would like / am ready to donate" is
present. Before `committed` is cited as a rate, it needs labels from a second reader on a set v2 was not
written against. The grid logs every `[donate]` turn, so that set will exist.

### Genuine hedge rate

On the n = 100 run, **3 of 78 tag-successes (3.8 %)** are hedged non-commitments under v2 (…225, …603,
…898). Counting both ambiguous turns as hedges gives 5 of 78 (6.4 %). Across all 134 logged successes:
6 of 134 (4.5 %), and 8 of 134 (6.0 %) with the ambiguous turns.

## 3. Decision (human, 2026-09-17): `--p4g_success tag` in the environment

`committed` is **not** put in the environment. Success ends the episode **and every search branch**
(`get_dialog_ended`). A detector that turns a real donation into a non-donation inside the tree corrupts
every backup through that branch. At a 4–6 % genuine hedge rate, and with a detector that is fitted
rather than validated, that costs more than the leniency it removes. `committed` is reported as an
**offline sensitivity row** for every arm, from the logged `[donate]` turns with no rerun
(`scripts/t10_success_criteria.py`). `--p4g_success committed` remains available (default `tag`,
tested). The grid does not use it.

**The benchmark finding stands on its own (§5).** The standard P4G success detector, the simulator's
self-tag, accepts hedged non-commitments such as "I will consider donating and will be in touch".
Success terminates the search, and every published P4G number was measured under this detector. We
measured the genuine hedge rate at roughly 4–6 % of successes, and judged an imperfect strict detector
inside the search to be worse than the leniency.

Files: `success_criteria_sat.{json,tsv}`, `success_criteria_pilots.{json,tsv}` (both re-scored with v2).
v1 is preserved at `runs/_p4g_success_v1.py`, and v1's n = 100 scores at `runs/_sc_sat_v1.json`.
