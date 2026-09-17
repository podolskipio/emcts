# §6 The momentum hypothesis — simulator test, and the verdict across all four sources

## Verdict (§6.4): **the "neither" cell**

H does not hold in the P4G corpus (`momentum_corpus.md`: primary interaction p = 0.22, DiD CI spans 0;
the one significant sensitivity is equally significant for affect *level*). It does not hold in the
simulator either. In D1 (Vicuna, β 0.7), D2 (Vicuna, β 0) and D3 (Qwen, β 0.7), on the task return and on z, on
all steps and on fresh steps only, **no act × Δν interaction survives the §6.3 controls** (12 of 12 cells).
Every apparent interaction clears the within-depth permutation and fails the within-tree × depth permutation.
The association is between trees, not within them: a construction-level effect. Interaction increments are
0.0002–0.0017 R². The single directional hint is D3's task return, where propose's value rises with Δν more than
emotion appeal's does (DiD +0.034 [+0.003, +0.082]; fresh steps +0.051 [+0.012, +0.094]). It fails the
tree × depth permutation (p 0.29 / 0.095) and the depth-residualized Wald test (p 0.06 / 0.34).
Consistent with the pre-registered rule, it is reported as absent. With the corpus also null, this cannot be
read as "the simulator reproduces a human dynamic".

**Consequences.** H gives K3 no second motivation, and K3 keeps only its depth-invariance argument. §5.2 already found that argument insufficient to
switch keys. §IV-D's trust-building-after-negative-affect claim now has a properly powered test behind it,
and it is not supported as a *momentum* claim. The corpus shows it only as a *level* effect for emotion appeals.

## Data and definitions

Table A; D1/D2 backfilled `backup_value` (= simlog `v`, replay-verified), D3 native. Δν at the parent =
parent ν minus the previous user turn in the parent realization's own history; rows without one (depth 1
of the first planned turn, 1,470 per run) are dropped. Terciles are Table A's, per run (D1 −0.035 / +0.085,
D2 −0.045 / +0.087, D3 −0.053 / +0.077), never mixed with the corpus's. Acts with < 30 rows or < 10 in a
tercile are dropped from the models (D3 `other`: 15 rows; see calibration note in `momentum_corpus.md`).
Runs are never averaged. "Fresh" = child generated from this parent realization (not served from the
R = 4 cache), i.e. the causal pairs.

**Occupancy:** Δν < −0.1 on 25.8 % (D1), 26.6 % (D2), 25.8 % (D3) of steps. Human corpus 29–32 % of turns. The
§11 trigger (falling < 10 % in both) does not fire.

## Interaction tables: mean backed-up task return per (Δν tercile, act), cluster-bootstrap 95 % CI

| act | D1 falling | D1 flat | D1 rising | D2 falling | D2 flat | D2 rising | D3 falling | D3 flat | D3 rising |
|---|---|---|---|---|---|---|---|---|---|
| emotion appeal | 0.39 [0.32, 0.47] | 0.38 | 0.41 [0.35, 0.48] | 0.39 [0.31, 0.47] | 0.45 | 0.49 [0.44, 0.55] | 0.82 [0.77, 0.87] | 0.79 | 0.82 [0.75, 0.88] |
| proposition of donation | 0.43 [0.35, 0.52] | 0.37 | 0.48 [0.41, 0.57] | 0.41 [0.31, 0.53] | 0.43 | 0.53 [0.45, 0.59] | 0.82 [0.73, 0.88] | 0.79 | 0.85 [0.78, 0.90] |
| logical appeal | 0.42 | 0.34 | 0.39 | 0.44 | 0.46 | 0.49 | 0.51 | 0.47 | 0.51 |
| credibility appeal | 0.68 | 0.51 | 0.53 | 0.64 | 0.59 | 0.67 | 0.45 | 0.46 | 0.38 |
| task related inquiry | 0.41 | 0.18 | 0.37 | 0.42 | 0.42 | 0.43 | 0.52 | −0.06 (n=25, —) | 0.29 |

n per cell: emotion appeal 4,400–8,500; propose 1,300–3,200; logical 370–1,260; credibility 125–454;
task inquiry 25–850. **The planner's act mix is extremely skewed:** emotion appeal is 64 % (D1) and 69 % (D3) of
all simulated steps, against 7 % of human corpus rows. Where the rows run parallel (propose and emotion appeal
both rise with Δν in D2), that is a shared trend, not the interaction H predicts.

## Models and §6.3 controls, task return (primary) and z

OLS, depth dummies as controls. ΔR² with cluster-bootstrap CIs. Wald = F-corrected clustered test on
act:Δν. "Resid." = Δν and outcome both residualized on depth. Permutation p = share of 1000 within-stratum
shuffles of Δν with M1→M2 ΔR² ≥ observed.

| run · outcome · sample | M1→M2 ΔR² [CI] | Wald p | resid. Wald p | perm p (depth) | **perm p (tree × depth)** | DiD propose − emotion [CI] | level-ν interaction p |
|---|---|---|---|---|---|---|---|
| D1 · task · all | 0.00096 [0.00045, 0.0037] | 0.005 | 0.003 | 0.001 | **0.236** | +0.031 [−0.034, +0.091] | 0.053 |
| D1 · task · fresh | 0.00066 | 0.29 | 0.21 | 0.078 | **0.460** | +0.008 [−0.084, +0.102] | 0.41 |
| D1 · z · all | 0.00109 | 0.34 | 0.29 | 0.001 | **0.340** | −0.016 [−0.039, +0.006] | 0.0003 |
| D1 · z · fresh | 0.00035 | 0.27 | 0.22 | 0.45 | **0.794** | +0.003 [−0.023, +0.030] | 7e-7 |
| D2 · task · all | 0.00167 [0.00096, 0.0053] | 0.028 | 0.037 | 0.001 | **0.189** | +0.009 [−0.069, +0.104] | 0.003 |
| D2 · task · fresh | 0.00125 | <0.001 | <0.001 | 0.010 | **0.073** | +0.050 [−0.025, +0.132] | 0.002 |
| D2 · z · all | 0.00080 | 0.45 | 0.49 | 0.001 | 0.005 | +0.004 [−0.025, +0.038] | 2e-9 |
| D2 · z · fresh | 0.00063 | 0.10 | 0.12 | 0.13 | **0.454** | −0.011 [−0.032, +0.010] | 3e-12 |
| D3 · task · all | 0.00122 [0.00042, 0.0068] | 0.083 | 0.061 | 0.001 | **0.291** | **+0.034 [+0.003, +0.082]** | 0.11 |
| D3 · task · fresh | 0.00113 | 0.23 | 0.34 | 0.035 | **0.095** | **+0.051 [+0.012, +0.094]** | 0.031 |
| D3 · z · all | 0.00018 | 0.34 | 0.40 | 0.060 | **0.586** | +0.006 [−0.011, +0.027] | 0.21 |
| D3 · z · fresh | 0.00036 | 0.055 | 0.067 | 0.11 | **0.197** | +0.009 [−0.012, +0.031] | 0.003 |

Pre-registered rule: an interaction is present only if the residualized Wald p < 0.05, **both** permutation
p < 0.05, and the DiD CI excludes 0. **No cell passes.** The only cell with a tree × depth p < 0.05
(D2 · z · all, 0.005) has a residualized p of 0.49 and a DiD CI spanning 0.

**Reading the controls.** The within-depth permutation rejects in most "all steps" cells (p 0.001), while
the within-tree × depth permutation almost never does. Δν and the outcome share tree-level structure
(dialogue, persona, turn), which the depth-only null cannot remove. That is the brief's "mechanical
association guaranteed by construction", located at the tree rather than at depth. The **level** of affect,
by contrast, interacts with act on z strongly in Vicuna (p to 3e-12): the current affect predicts the next
reaction differently per act. That is a level property, and §5.3 shows it is mostly carried by ν_t and the
recent history.

## What goes in the paper regardless (§6.5)

1. **Corpus interaction table** (`momentum_corpus.md`): in P4G the act → donation relationship does not depend on the
   direction of affective change.
2. **Simulator–corpus gap:** both are null for H, so there is no gap on this property to report. The gap that
   *does* appear is structural. Vicuna users drift toward happiness with search depth and Qwen users do not
   (`qwen_comparison.md`), and simulated act mixes bear no resemblance to human ones (emotion appeal 64–69 % vs 7 %).
3. **Vicuna vs Qwen affective range, paired:** Qwen is lower in level (−0.135 [−0.161, −0.104] parent ν)
   with the same spread and negative-label rate, and no depth drift.
4. **AffPool homogeneity on task return:** 0.50 / 0.05 (D1), 0.65 / 0.14 (D2), gate passes, but the
   unkeyed pool is as homogeneous (`affpool_gate_recheck.md`).

Machine-readable: `s6_momentum_simulator.json`. Script: `scripts/s6_momentum_simulator.py`.
