# The value-estimator fork — does `v` predict simulated donation, human donation, or neither?

Phase 1A found that the planner's value estimator `v` does not predict human donation (AUC 0.50,
McFadden 0.002). Every Q in the tree is a backed-up `v`, so this had to be resolved before anything
else. The fork:

- **YES, v predicts simulated donation.** Then v and the simulator agree with each other but not with
  humans: a sim-to-real gap.
- **NO.** Then v predicts nothing, and the search is guided by noise.

Scripts: `scripts/fork_v_sim.py` (frozen grid logs, zero GPU) and `scripts/fork_v_human.py` (the three
checks on the human result). Numbers: `fork_v_sim.json`, `fork_v_human.json`. All intervals are 95 %
bootstrap, 1,000 replicates, clustered by dialogue (by run × dialogue for the grid).

## Verdict

**YES branch: a sim-to-real gap, not a value estimator that predicts nothing.**

- **On simulated outcomes, `v` carries real signal.** Pooled over 9,844 planned turns of 1,800 played
  dialogues (18 frozen grid runs), AUC is **0.65 [0.63, 0.67]**. At fixed turns 3–6 it is **0.67–0.72**.
  Its increment over turn position is 0.057 [0.046, 0.070] McFadden. Every one of the 18 runs lands
  between 0.59 and 0.77.
- **On human outcomes, `v` is weak.** At the same fixed turns it is 0.47–0.59, and pooled 0.53
  [0.48, 0.58]. The "no information" claim overstates it. At turn 4 the interval [0.51, 0.66] just
  excludes 0.5, so the right phrasing is "at most weakly informative".
- **At matched turns the gap is about 0.1–0.2 AUC.** The intervals don't overlap from turn 2 onward.

| turn k | sim: n | sim: AUC of v | sim: McFadden | human: n | human: AUC of v | human: AUC of v_logit | human: McFadden |
|---|---|---|---|---|---|---|---|
| 1 | 1,800 | 0.53 [0.52, 0.55] | 0.008 | 285 | 0.47 [0.43, 0.52] | 0.50 [0.43, 0.57] | 0.002 |
| 2 | 1,800 | 0.60 [0.57, 0.63] | 0.024 | 272 | 0.48 [0.41, 0.55] | 0.49 [0.42, 0.57] | 0.001 |
| 3 | 1,679 | 0.67 [0.64, 0.70] | 0.057 | 254 | 0.56 [0.48, 0.64] | 0.55 [0.48, 0.61] | 0.007 |
| 4 | 1,275 | 0.68 [0.65, 0.71] | 0.071 | 223 | 0.59 [0.51, 0.66] | 0.59 [0.51, 0.67] | 0.015 |
| 5 | 925 | 0.72 [0.69, 0.75] | 0.113 | 196 | 0.54 [0.46, 0.62] | 0.55 [0.47, 0.63] | 0.003 |
| 6 | 718 | 0.69 [0.65, 0.73] | 0.079 | 166 | 0.58 [0.49, 0.67] | 0.59 [0.49, 0.67] | 0.011 |
| pooled | 9,844 | **0.65 [0.63, 0.67]** | 0.052 | 1,396 | **0.53 [0.48, 0.58]** | | |

Row definitions:
- **Simulated rows:** the root state of planned turn k, meaning the real played dialogue the planner
  searched from. The outcome is whether that episode ended in a `[donate]` tag.
- **Human rows:** the first k exchanges of an annotated dialogue, only when its first decision act
  comes after turn k. The outcome is the recorded donation.
- **Both sides:** `v` is the estimator exactly as the grid ran it (10 samples, T 1.1, persona).

## What this does and does not explain

**It does not explain the earlier puzzles attributed to it.** Those are all measured against *simulated*
success, and there `v` is informative (AUC 0.65–0.72 mid-dialogue). "Searching harder on an
uninformative value gains nothing" is not what is going on inside the benchmark. The search maximizes
a value that does track the benchmark's outcome. The flat budget curve, the wasted-budget fix that
changed nothing, and the six null affect methods still need their own explanations. The strongest
candidate is still the one the rest of the program measures: SR noise that swamps effects (NoEmo seed
spread 0.11, correlation between arms ≈ 0).

**It does say what the benchmark measures.** The same LLM plays the value estimator and the simulated
persuadee, with the same persona. `v` asks the persuadee role "would you donate?", and success is that
same role later tagging `[donate]`. Agreement between the two is partly by construction. That
agreement barely transfers to what real persuadees did. **The self-play SR measures planner–simulator
agreement more than persuasion.** That is a statement about the GDP-Zero evaluation line, not only
about this planner, and it can lead the paper. It should be stated as a gap, with the caveats below,
not as "v is noise".

## The three checks on the human 0.50

**1. Interval on the AUC.** On the C1 prefixes: v 0.47 [0.41, 0.54], v_logit 0.49 [0.42, 0.56]. The
upper bounds sit near 0.55, so an AUC-0.6 estimator is excluded there. At fixed turns the upper bounds
reach 0.66–0.67. Across all fixed-turn rows it is 0.53 [0.48, 0.58]. **"At most weakly informative" is
supported. "No information" is too strong.**

**2. Turn count was leaking.** C1 cut every dialogue at its decision point, so `n_turns` there *is* the
turn at which the human decided. Its AUC is 0.71 [0.64, 0.76], because early deciders donate more.
That 0.097 McFadden is not a value-function baseline. It is the outcome's timing. At fixed turn
positions turn count is constant by construction, and `v` still reaches only 0.47–0.59. So the
human-side conclusion survives the leak. Only the "turn count reaches 0.097" comparison should be
dropped from any claim.

**3. Domain shift.** The fork separates it. `v` works on its own simulator's states (0.65–0.72) and
fails on human states (0.47–0.59). This is a sim-to-real finding, not proof the estimator is broken.

## Caveats that bound the claim

- **The outcomes differ.** Simulated: a `[donate]` self-tag within 10 turns. Human: an actual donation
  after the task. Part of the gap could be outcome definition, not the estimator.
- **The states differ.** Simulated states are planner-chosen and simulator-written. Human states are
  written by human persuaders and persuadees. The fair transfer test would need human outcomes on
  planner-generated dialogues, and those do not exist here.
- **Both sides condition on survival.** Sim rows at turn k are dialogues not yet succeeded; human rows
  are dialogues not yet decided. The conditioning is similar but not identical.
- **Single backbone** (Vicuna-13B-AWQ), 285 human dialogues, one prompt for `v`.
- **In-simulator curiosity, not pursued.** 22–29 % of live root states at turns 3–5 get `v = 1.0` (all
  10 samples "donate") although the simulator has not yet tagged `[donate]` there. The estimator is
  more willing to say yes than the simulator is to end the episode.

## C1a, restated

C1a's increment is 0.017 [0.001, 0.052]. The interval excludes zero, so `r` carries **statistically
non-zero** information over `v` on human outcomes, **practically small**: under the pre-registered
0.02, and not significant out of fold (log-loss gain +0.009 [−0.010, +0.027]). Much of that increment is
`r` supplying signal `v` lacks on human data (r's own AUC 0.56 [0.49, 0.63]), not affect refining a
good value.
