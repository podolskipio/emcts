# B1 diagnosis — is NoEmo (`emomcts --beta_emo 0`) GDP-Zero's search?

**Question.** On the stub backbone, `--algo gdpzero` and `emomcts β 0` gave different action sequences on
10/10 dialogues. NoEmo is the reference for every arm. If it is not GDP-Zero's search, every arm
comparison inherits the difference.

## Answer

1. **At the grid configuration (top-K 5, HF classifier), NoEmo *is* GDP-Zero, bit for bit.** It is
   identical end to end, and on real trees the only code-level difference never changes a decision
   (0 of 120,085). **No arm is invalidated.**
2. **With top-K off (B1), they diverge** through one cause, found and confirmed: the two planners bracket
   the PUCT exploration term differently, which changes the last bit. With the 15-sample histogram
   prior this breaks exact ties differently at **0.27 %** of selections, and trajectories then diverge.
   The difference is floating-point tie-breaking, not a difference in the algorithm. It is still a
   divergence from the published implementation, so **B1 is generated as `--algo gdpzero --game p4g`**,
   GDP-Zero's own `OpenLoopMCTS`.
3. **The original 10/10 divergence was the LLM emotion classifier**, the runner default, spending
   calls on the deterministic stub backbone. It was not the planner. The grid uses HF, which makes no
   backbone calls.

## Evidence

### A. Component comparison (code)

| component | `OpenLoopMCTS` | `EmotionAwareMultiObjectiveQ`, β 0 |
|---|---|---|
| terminal test (`_ends_search`), node key (DA prefix) | same | same |
| expansion, prior renormalisation, top-K hard prune | same | same (also initialises `Q_emo`, never read at β 0) |
| realization pool: add / sample / cache | same | same. It samples *before* selection, but selection draws no randomness, so the RNG stream is identical |
| backup: Q running mean, Ns, Nsa, realization values | same | same |
| root decision `get_action_prob` | argmax N | same (Constrain off) |
| **PUCT** | `Q + c·P·√Ns / (1+N)`, evaluated left to right | `Q + β·Q_emo + c·P·(√Ns / (1+N))` |
| game | `PersuasionGame` | `EmotionAwarePersuasionGame`: same system/user generation; classifies the user turn; its session renders **no** emotion into any prompt (`to_string_rep` drops it; players read role/da/utt only) |

### B. Planner on the same game, fake text stack

`scripts/b1_planner_equivalence.py`. Both planners run on the emotion-aware game, `OpenLoopMCTS` through a
one-line adapter, with the fake players of `tests/test_wed_arms.py`. Grid: 20 seeds × n_sims {20, 50} ×
top-K {off, 5} × root depth {0, 2} = **160 searches**. Ns / Nsa / Q / P tables, realization pools and
root decision are **bit-identical in 160/160**. The fake prior has continuous values, so no rational ties
arise.

### C. End to end, stub backbone, 10 dialogs × 50 sims, episode, R 4

| setting | gdpzero on `p4g` vs emomcts β 0 on `emo_p4g` | stub backbone calls |
|---|---|---|
| **HF classifier, top-K 5** (grid) | **10/10 identical** full text (role, act, utterance), actions and outcomes | 8,616 / 8,616 |
| HF classifier, **top-K off** (B1) | **2/10** identical text, 4/10 identical outcomes | 8,220 / 6,548 |
| top-K off, emomcts with `OpenLoopMCTS`'s PUCT bracketing patched in | **10/10 identical** | 8,220 / 8,220 |
| (original) LLM classifier (runner default) | 0/10 | differs (classifier calls on the stub) |

Patching only the bracketing restores identity, so **the bracketing is the sole cause**.

### D. How often does bracketing flip a decision?

- **Continuous priors** (random nodes, K = 5): **0 flips in 200,000** argmax decisions. The last bit
  differs on 13 % of sibling scores, but no decision changes.
- **Histogram priors** (multiples of 1/15, 13 acts): **272 flips in 100,000 (0.27 %).** Probabilities in
  exact rational ratio (e.g. P = 2/15 at N = 1 against P = 1/15 at N = 0) make mathematical ties, which
  the two bracketings round differently.
- **Real logged trees, top-K 5 ranking prior** (logged Q, prior, N, parent Ns at every selection point,
  both bracketings recomputed): **0 flips in 120,085 selection points**. That covers P1 legacy and
  episode, all five episode arm pilots, and the n = 100 run. The ranking prior's 0.5 % floor breaks exact
  ratios.

## Consequences

- **Block A, B2, B3–B5, C3, C4 unchanged.** All run top-K 5, where NoEmo equals GDP-Zero's search.
- **B1 = `--algo gdpzero --game p4g --llm_prior_topk 0`**, GDP-Zero's `OpenLoopMCTS` on the plain p4g
  game. It is shown equivalent to the emotion-aware game on the text the LLM sees (C, row 1). B1 writes no
  per-step simlog (`sim_steps` exist only on the emotion-aware planner). It needs only SR, AvgT and the
  offline `committed` score.
- **Not changed:** `EmotionAwareMultiObjectiveQ`'s bracketing. Aligning it would alter the default path's
  last bit and so every GOLDEN fingerprint and logged run, for a difference that never binds at the grid
  prior. If a later run turns top-K off on an emotion arm, it would bind at about 0.27 %, and that is
  where to revisit it.
