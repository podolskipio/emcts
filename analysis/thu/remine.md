# w(e) re-mined on pre-decision turns

**Human decision 2026-09-17:** re-mine the valence table on pre-decision turns, reset τ_med, report max |Δw|.

**Result: max |Δw| = 0.41 (happiness 0.54 → 0.13).** With post-decision turns removed, **no emotion's
Wilson CI excludes the base rate.** In the shipped table, happiness was the one cell that did. The
largest weight in the new table belongs to **fear (+0.46)**, the thinnest cell. Once outcome language is
gone, the corpus carries no emotion→donation association distinguishable from chance at the utterance
level, and the new table's ordering is driven by noise-level cells.

Shipped as `--emo_valence_table predecision`. The default `soft` is untouched (bit-identity tests pass,
106 total).

## Method

The deployed recipe, `mine_emotion_donation_p4g.py --soft` (all 300 dialogs, soft per-utterance
assignment, α = 50 shrinkage), **reproduces the shipped `EMOTION_VALENCE_MINED` exactly** — all 8
weights (`remine/shipped_repro_soft_all300.json`). Two default-off flags were added to the miner. With
neither flag set, its output is identical, and the reproduction was re-checked after the patch.

- `--pre_decision_only`: keep only persuadee utterances from turns strictly before the dialog's first
  decision act (`agree-donation`, `disagree-donation`, `disagree-donation-more`,
  `provide-donation-amount`, `confirm-donation`). The outcome label is still whole-dialog. This is the
  same cut as Task 2's decision cut.
- `--base_rate {dialog,unit}`: what lift is measured against.

**Why `unit`.** The miner computes P(donate | e) over utterances but measured lift against the share of
*dialogs* that donate. On full dialogs the two rates agree (0.493). The cut removes far more donor
utterances than refuser utterances: donors keep 47 % (7.8 of 16.5 per dialog), refusers keep 70 %
(11.1 of 15.8). The utterance-weighted donation rate falls to 0.407 while the dialog rate stays 0.479,
so every lift measured against the dialog rate goes negative by roughly the same amount. The `dialog`
table has **all seven weights ≤ 0**, happiness at −0.53, and max |Δw| 1.07. That is a base-rate artefact:
it shifts ν's level (which Bias's visited-edge bonus and τ both depend on) without reflecting affect.
The `unit` base rate measures each lift against the population actually tallied.

**Dropped:** 12 dialogs have their decision in the first turn and so have no pre-decision utterance. 10
of them are donors (base rate over the 288 kept: 0.479 by dialog).

## Tables

| w(e) | happiness | sadness | fear | anger | surprise | disgust | neutral |
|---|---|---|---|---|---|---|---|
| **shipped `soft`** (all turns) | **+0.54** | −0.14 | +0.24 | +0.11 | +0.09 | +0.13 | −0.07 |
| **`predecision`** (unit base) | **+0.13** | **−0.29** | **+0.46** | +0.12 | +0.11 | +0.13 | −0.10 |
| Δ | **−0.41** | −0.15 | +0.22 | +0.01 | +0.02 | 0.00 | −0.03 |
| relative to neutral, shipped | +0.61 | −0.07 | +0.31 | +0.18 | +0.16 | +0.20 | 0 |
| relative to neutral, predecision | +0.23 | −0.19 | +0.56 | +0.22 | +0.21 | +0.23 | 0 |
| (predecision, dialog base — artefact) | −0.53 | −0.87 | 0.00 | −0.39 | −0.53 | −0.39 | −0.80 |

Pre-decision cells, unit base 0.407:

| emotion | n_eff | P(donate) | Wilson 95 % | excludes base? |
|---|---|---|---|---|
| happiness | 495 | 0.421 | [0.378, 0.464] | no |
| sadness | 201 | 0.370 | [0.306, 0.439] | no |
| fear | **86** | **0.480** | [0.377, 0.584] | no |
| anger | 113 | 0.423 | [0.336, 0.516] | no |
| surprise | 401 | 0.419 | [0.372, 0.468] | no |
| disgust | 131 | 0.425 | [0.344, 0.511] | no |
| neutral | 1418 | 0.396 | [0.371, 0.422] | no |

## What it means — reported, not decided

1. **The shipped table is substantially an outcome detector.** Happiness drops by three-quarters in
   absolute weight (0.54 → 0.13) and by nearly two-thirds relative to neutral (0.61 → 0.23). That agrees
   with Task 2's independent turn-level implementation (+0.52 → +0.17, `construct_test.md` §3b).
2. **The replacement is not an affect model either.** It is a table of non-significant lifts. Its
   largest weight, fear +0.46, rests on 86 effective utterances. A planner using it with β > 0 steers
   hardest towards user turns classified as fearful. That is a consequence of noise, not a finding. Before
   this table goes into a grid arm, choose among:
   - **ship `predecision` as is** — honest about the data, weakest signal;
   - **use `generic`** (textbook valence, not fitted to outcomes; already built and tested);
   - **raise α** so thin cells shrink harder. At α = 50 fear keeps 63 % of its raw lift. This is a
     tuning choice made after seeing the result and would need to be declared.
3. **The unit is utterance (sentence); the planner classifies whole turns.** The miner tallies each
   persuadee sentence, while the planner computes ν on a whole simulated reply. This mismatch predates
   today and applies to every table. Noted, not fixed.
4. **Everything computed on the shipped ν is now in question:** τ_med, AffPool τ, Bias's visited-edge
   bonus (+0.26 mean), the dose calibration, TrajValue's signature terciles. Task 7 re-derives what the
   logs allow under the new table.

Files: `remine/shipped_repro_soft_all300.json`, `remine/predecision_soft_all300_{unit,dialog}.json`.

---

## Addendum (2026-09-17): does a different turn weighting reveal a pre-decision signal?

**Human question:** with the same mining code, are there emotions whose CI excludes the base rate
under **last-pre-decision-only** or **recency-weighted** (γ ≈ 0.7) selection? If one does, uniform
averaging was hiding a real signal and that table ships. If none does at any weighting, `generic`
stands, and the null is established rather than assumed.

**Answer: no emotion clears it robustly under any weighting. `generic` stands.**
- **Under the miner's own Wilson intervals, nothing excludes the base rate in any pre-decision variant.**
- **With a dialog-clustered bootstrap, nothing survives correction for the 7 emotions tested per
  variant.**
- **At the pre-specified γ 0.7, nothing excludes zero even uncorrected.**
- The few nominal hits land on a **different emotion in each variant** (fear under uniform, surprise
  under last-only and under γ 0.5). That is the pattern noise produces.

Built into the miner as `--turn_weighting {uniform,last,recency}` and `--recency_gamma`. The default is
uniform, and the shipped and `predecision` tables still reproduce exactly. The script is
`scripts/t_remine_variants.py` and the output is `remine/variants.json`.

### Method

- **Corpus and recipe:** the annotated 300 dialogs, soft assignment, α 50, unit-weighted base rate. Each
  variant's base rate is recomputed over its own weighted units.
- **Turn weighting:** k = persuadee turns before the last pre-decision turn. `last` gives weight 1 at
  k = 0 and 0 otherwise. `recency` gives γᵏ. All sentences of a turn share its weight.
- **"~900 corpus-wide" is not available.** Locating the decision turn needs the annotated acts, which only
  the 300 have. The non-annotated pool also contains the eval set.
- **Effective n:** reported as **effective dialogs**, (Σ_d T_d)² / Σ_d T_d², over per-dialog weighted mass.
  Dialogs are the independent units, because every utterance in a dialog shares its outcome. A Kish n over
  soft units is not reported: with masses below 1 it comes out larger than the weighted trials, which is
  meaningless.
- **Intervals:**
  - the miner's Wilson interval on P(donate | e);
  - **a 95 % cluster bootstrap over dialogs** (2,000 resamples, base rate recomputed in each) on the lift;
  - the same bootstrap at 1 − 0.05/7 (**Bonferroni** over the 7 emotions per variant).

### Tables

Columns: **w** = weight; **lift** = P(donate | e) − base; **trials** = weighted mass; **n_eff dialogs** =
effective dialogs; **Wilson ex** = the miner's Wilson interval excludes the base rate; **cluster 95 %** =
bootstrap interval on the lift; **Bonf/7 ex** = Bonferroni interval excludes 0.

**Uniform pre-decision** (= `predecision`), 288 dialogs, base 0.407

| emotion | w | lift | trials | n_eff dialogs | Wilson ex | cluster 95 % | Bonf/7 ex |
|---|---|---|---|---|---|---|---|
| happiness | +0.13 | +0.014 | 494.9 | 170.5 | · | [−0.020, +0.050] | · |
| sadness | −0.29 | −0.037 | 201.3 | 130.3 | · | [−0.083, +0.012] | · |
| fear | +0.46 | +0.073 | 85.8 | 106.6 | · | **[+0.002, +0.138]** | · |
| anger | +0.12 | +0.017 | 112.7 | 156.8 | · | [−0.028, +0.062] | · |
| surprise | +0.11 | +0.013 | 401.3 | 184.8 | · | [−0.022, +0.049] | · |
| disgust | +0.13 | +0.019 | 130.9 | 122.4 | · | [−0.031, +0.068] | · |
| neutral | −0.10 | −0.011 | 1418.1 | 193.7 | · | [−0.028, +0.006] | · |

**Last pre-decision turn only**, 288 dialogs, base 0.478

| emotion | w | lift | trials | n_eff dialogs | Wilson ex | cluster 95 % | Bonf/7 ex |
|---|---|---|---|---|---|---|---|
| happiness | **−0.47** | −0.073 | 91.1 | 96.8 | · | [−0.146, +0.002] | · |
| sadness | −0.30 | −0.072 | 35.4 | 60.1 | · | [−0.173, +0.038] | · |
| fear | +0.21 | +0.100 | 13.3 | **38.3** | · | [−0.055, +0.224] | · |
| anger | +0.05 | +0.017 | 18.7 | 66.3 | · | [−0.086, +0.112] | · |
| surprise | +0.60 | +0.110 | 59.8 | 89.8 | · | **[+0.027, +0.185]** | · |
| disgust | −0.00 | −0.002 | 21.0 | 51.3 | · | [−0.113, +0.111] | · |
| neutral | +0.04 | +0.005 | 222.6 | 178.6 | · | [−0.029, +0.037] | · |

**Recency, γ 0.7** (pre-specified), 288 dialogs, base 0.450

| emotion | w | lift | trials | n_eff dialogs | Wilson ex | cluster 95 % | Bonf/7 ex |
|---|---|---|---|---|---|---|---|
| happiness | +0.01 | +0.002 | 229.2 | 165.6 | · | [−0.039, +0.046] | · |
| sadness | −0.25 | −0.038 | 94.2 | 126.3 | · | [−0.096, +0.020] | · |
| fear | +0.32 | +0.074 | 38.9 | 93.8 | · | [−0.008, +0.148] | · |
| anger | +0.10 | +0.020 | 52.2 | 142.7 | · | [−0.031, +0.074] | · |
| surprise | +0.31 | +0.039 | 175.0 | 178.1 | · | [−0.004, +0.083] | · |
| disgust | +0.05 | +0.009 | 61.9 | 120.6 | · | [−0.049, +0.066] | · |
| neutral | −0.12 | −0.013 | 636.7 | 215.6 | · | [−0.033, +0.006] | · |

**Sensitivity, γ 0.5 and γ 0.85.** One nominal cluster hit: surprise at γ 0.5, [+0.005, +0.108].
Nothing survives Bonferroni, and nothing excludes the base rate under Wilson (`variants.json`).

**Reference, shipped recipe on all turns** (unit base 0.504). **Happiness** [+0.018, +0.072] and
**neutral** [−0.032, −0.004] **exclude 0 under the cluster bootstrap and after Bonferroni.** So the
bootstrap is sensitive enough to detect a lift of about 0.05 on this corpus. The shipped signal is
real, but it lives in the post-decision turns.

### Reading

1. **No robust pre-decision signal.** Across 6 variants × 7 emotions (42 tests):
   - miner's Wilson: 0 hits;
   - cluster bootstrap at 95 %: 3 hits (fear-uniform, surprise-last, surprise-γ 0.5), against about 2
     expected by chance at 5 %, and the variants are not independent;
   - Bonferroni: 0 hits.

   The hits fall on different emotions depending on the weighting, and none appears at the pre-specified
   γ 0.7.
2. **Last-only is mostly untestable, not null.** Its cells have 38–97 effective dialogs, except neutral
   at 179. Cluster intervals span 0.15–0.28 in lift, so any effect smaller than about ±0.10 cannot be
   seen. Read "no hit under last-only" as **"a moderate effect is not ruled out"**, not "no effect".
3. **Happiness flips sign as weighting moves towards the decision turn.** It goes from +0.046 lift
   (shipped, all turns) to +0.014 (uniform pre-decision) to −0.073 (last turn only, cluster
   [−0.146, +0.002]). On this corpus, a *happy* last turn before the decision leans towards **not**
   donating: pleasant disengagement rather than engagement. It is not significant, but it points the
   opposite way to the shipped table's premise.
4. **Consequence (the rule as stated):** no weighting yields a table whose top weights are distinguishable
   from the base rate, so **`generic` stands.** That comes with an established finding: in P4G, persuadee
   emotion before the donation decision does not detectably predict donation at the utterance level,
   under uniform, last-turn or recency weighting. The detection limit is about ±0.05 lift for common
   emotions and ±0.10–0.15 for rare ones. The shipped table's significant happiness effect comes from
   post-decision turns.

**Freeze implication.** `--emo_valence_table generic`. τ_med and doses on the episode pilot trees under
`generic` are already computed (`tau_dose.md`): τ_med **0.263** (50 % occupancy). Bias β 0.7 flips 15.2 %
against NoEmo, with CenteredBias matched at 1.03 and Momentum at 1.11. The dose anchor then matters much
less. Under `generic`, Bias β 0.7 (15.2 %) sits close to the 11.9 % the pilots validated, and the
11.9 % anchor would be Bias 0.49 / CenteredBias 0.67 / Momentum 0.80.
