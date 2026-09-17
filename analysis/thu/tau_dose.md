# τ and doses under `episode`, re-derived per valence table

**Human decisions (2026-09-17):** `episode` horizon; re-derive AffPool τ from episode logs with the
achieved occupancy recorded; re-mine w(e) on pre-decision turns (`remine.md`).

Every number below comes from the **P1 NoEmo `episode` trees** (dialogues 131–140, 6,360 steps). ν is
recomputed exactly from the logged per-step emotion distributions under each table. Under `soft` the
recomputation reproduces the logged `parent_nu` / `child_nu` to 1e-9 (asserted), so the other tables are
the same states re-scored, not new runs. Script `scripts/t5_dose_replay.py --table`; numbers in
`tau_dose_by_table.json` and `dose_replay_P1_episode_{predecision,generic}.json`.

## τ (AffPool K0 threshold = median parent ν)

| table | **τ_med, episode** | occupancy of low bucket at τ_med | occupancy at the old 0.35 | τ_med, legacy (for reference) |
|---|---|---|---|---|
| `soft` (shipped) | 0.279 | 50.0 % | 59 % | 0.404 |
| **`predecision`** (re-mined) | **0.078** | **50.0 %** | **93 %** | 0.103 |
| `generic` | 0.263 | 50.0 % | 60 % | 0.402 |

**To freeze: `--aff_pool_tau 0.078`, with `--emo_valence_table predecision` and `--search_horizon
episode`.** The achieved occupancy is 50.0 % on the pilot trees by construction. Under the frozen
table, the old 0.35 would put 93 % of steps in one bucket. ActPool has no τ.

**Caveat:** the τ comes from 10 dialogues of NoEmo trees. AffPool's own trees may drift, since pooling
widens search. The occupancy should be logged in the grid (`aff_bucket` is already per step), not assumed.

## The visited-edge bonus under each table

| table | mean ν (parent), episode | mean level `Q_emo` on visited edges (Bias's bonus) | mean delta `Q_emo` on visited edges |
|---|---|---|---|
| `soft` | 0.264 | **+0.267** | +0.013 |
| **`predecision`** | 0.071 | **+0.070** | +0.000 |
| `generic` | 0.184 | +0.193 | +0.023 |

**The re-mine removes most of Bias's flat bonus.** It was +0.27 because the simulator's replies are
mostly happy or neutral and the shipped table weights happiness at +0.54. At +0.13, the level signal
sits near zero on this simulator. Bias and Momentum become less different under `predecision` than
under `soft`. Momentum's structural argument (no visited-edge bonus) is now shared, approximately, by
Bias itself. **This weakens the Momentum > Bias prediction's mechanism. It does not falsify it.**

## Doses under `predecision` — the anchor needs a decision

PREREG Entry 2 fixes the reference as "Bias at β 0.7". Changing the table changes what β 0.7 does:

| anchor | Bias | CenteredBias | Momentum | achieved flip rate vs NoEmo |
|---|---|---|---|---|
| `soft`, Bias β 0.7 (the pilots) | 0.70 | 1.06 | 1.19 | 11.9 % |
| **`predecision`, Bias β 0.7** (PREREG literal) | 0.70 | **0.99** | **0.98** | **7.5 %** |
| **`predecision`, flip rate held at 11.9 %** | **1.19** | **1.57** | **1.74** | 11.9 % |
| `generic`, Bias β 0.7 | 0.70 | 1.03 | 1.11 | 15.2 % |

**Recommendation: hold the flip rate at 11.9 %** (Bias 1.19, CenteredBias 1.57, Momentum 1.74). The
PREREG's dose-matching rule exists so that no arm reads as a null through under-dosing. Keeping β 0.7
on a table with a third of the signal under-doses all three arms together: 7.5 % against the 11.9 % the
live pilots validated. This needs a one-line PREREG amendment ("reference: Bias at the flip rate β 0.7
achieved under the shipped table, 11.9 % on the episode pilot trees"). **Not applied.**

**Under either anchor, CenteredBias and Momentum match each other closely** (0.99 / 0.98, or
1.57 / 1.74). With the new table the three arms need more similar doses than before.
