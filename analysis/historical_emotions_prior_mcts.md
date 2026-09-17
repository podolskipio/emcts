# EmotionHistoryPriorMCTS — design summary

A new MCTS variant that conditions the action prior on the *whole-conversation*
cumulative emotion distribution and softens or sharpens the prior at every node
based on (negativity × dialog progress). Lives in three new files; no existing
code modified.

## Files

| Path | Role |
| --- | --- |
| `src/mcts/emotion_prior_mcts.py` | `EmotionHistoryPriorMCTS` class + helpers |
| `src/runners/emopriormcts.py`    | Offline-eval runner |
| `scripts/train_emotion_history_prior_p4g.py` | Trainer + persona spec |

## What the algorithm does

At every node expansion (`_init_node`):

1. **Aggregate** every prior user-turn's emotion distribution into a single 8-dim
   vector `cum_dist` via EMA over `state.history`. With `alpha=1.0` (default),
   this is the arithmetic mean — a single early-anger turn permanently contributes
   `1/N` of the cum vector for the rest of the dialog. With `alpha<1.0` older
   turns decay exponentially (recency bias).

2. **Compute breadth**
   ```
   neg        = sum(cum_dist[e] for e in NEGATIVE_EMOTIONS)   # [0, 1]
   positivity = 1 - neg                                       # [0, 1]
   progress   = min(1, turn_idx / horizon_turns)              # [0, 1]
   breadth    = positivity * progress                         # [0, 1]
   ```
   - Early + negative → breadth ≈ 0 (clamp action set).
   - Late + positive  → breadth ≈ 1 (full exploration).

3. **Compute soft-prune temperature**
   ```
   T = T_min + (T_max - T_min) * breadth        # default [0.2, 1.0]
   ```

4. **Query learned prior**: tokenizer pair encoding
   ```
   text_a = recent dialog history (truncated to last K turns)
   text_b = "cum_emotion: happiness=0.30 sadness=0.10 fear=0.00 anger=0.20 ..."
   ```
   feed into a fine-tuned DistilRoBERTa → 13 DA logits → divide by `T` → softmax
   → `P[s][a]`.

5. **No hard top-K cut.** `valid_moves` keeps all allowed actions; pruning is
   entirely in the temperature shape. At `T=0.2` the top action holds ~80% of
   mass, so at `N=10` sims PUCT visits the top 1–2 actions ~8/10 times and the
   rest get 0–1 visits each — no round-robin. At `T=1.0` mass spreads across all
   13 DAs and standard exploration resumes.

## How it satisfies each requirement

| # | Requirement | Mechanism |
| --- | --- | --- |
| 1 | Emotion distribution → action prior | Learned DistilRoBERTa over `(history, cum_emotion_text)` |
| 2 | Works at `num_sims=10` | `T_min=0.2` makes the prior peaky enough that PUCT visits top 1–2 actions ~8/10 sims |
| 3 | Accumulates across conversation, not turn-bound | EMA aggregator over every user-turn distribution; `alpha=1.0` keeps early emotions weighted forever |
| 4 | Action breadth grows with depth + positivity | `breadth = positivity × min(1, turn_idx / horizon)`. Recomputed at *every* `_init_node`, so MCTS rollouts that descend into simulated future turns naturally widen the action set as `state.history` grows |
| 5 | Learnable from p4g (+ personas) | `scripts/train_emotion_history_prior_p4g.py` trains on the 250-dialog split; HF classifier labels distributions; weighted CE on dialog outcome. 5 P4G persona archetypes documented in the script's docstring for the parallel user-simulator research direction |
| 6 | New subclass in new file + new runner | `EmotionHistoryPriorMCTS(EmotionAwareOpenLoopMCTS)` + `runners/emopriormcts.py` |

## Inheritance

```
OpenLoopMCTS
└── EmotionAwareOpenLoopMCTS                        (already has emotion history)
    └── EmotionHistoryPriorMCTS                     (new — overrides _init_node only)
```

`search()`, `_calculate_uct()`, Q updates, realization tracking — all inherited
unchanged. Single overridden method: `_init_node()`. Helper methods:
`_cumulative_user_emotion`, `_predict_prior`, `_build_history_text`, `_turn_idx`.

## Train + run

```bash
python scripts/train_emotion_history_prior_p4g.py
# outputs/emotion_history_prior_bert/

PYTHONPATH=src python src/runners/emopriormcts.py \
    --game emo_p4g --num_mcts_sims 10 --num_dialogs 20 \
    --emotion_history_prior_model outputs/emotion_history_prior_bert \
    --output outputs/emopriormcts_p4g_10sims.pkl

python scripts/paired_test.py \
    outputs/rollout_p4g_gdpzero_chatgpt_20d_50s.pkl \
    outputs/emopriormcts_p4g_10sims.pkl
```

## Key CLI knobs (`emopriormcts.py`)

| Flag | Default | Effect |
| --- | --- | --- |
| `--num_mcts_sims` | 10 | The whole point — soft sharp prior makes low-sim budget viable |
| `--ema_alpha` | 1.0 | 1.0 = mean (early anger persists); <1 = recency bias |
| `--T_min` | 0.2 | Sharpness when fully negative + turn 0; lower = more aggressive clamp |
| `--T_max` | 1.0 | Flatness when fully positive + late; 1.0 = train-time T |
| `--horizon_turns` | 6 | Turn index where `progress` saturates at 1 |
| `--emotion_classifier` | hf | Match the classifier used at training time |

## Suggested P4G personas (for parallel user-simulator research)

Documented in the training-script docstring. Use to test whether the soft-prune
generalises across emotionally divergent users:

| Persona | Trajectory | What it tests |
| --- | --- | --- |
| Hostile-Defender | Anger/Disgust → softens under credibility + foot-in-door | Soft-prune correctly clamps to 1–2 actions early |
| Anxious-Sympathetic | Fear → engages with emotion appeal + personal story | Trainer learns the rare Fear→{emotion appeal} cell |
| Skeptical-Sad-Drifter | Neutral drifts to Sadness under weak appeals | Cum-EMA correctly tracks drift; `alpha<1` should help |
| Cheerful-Engager | Happiness throughout; accepts proposition early | Soft-prune correctly *widens* (T→T_max) early |
| Neutral-Pragmatist | Neutral throughout | Baseline — temperature sits mid-range |

## What this does *not* compose with

`EmotionHistoryPriorMCTS` subclasses `EmotionAwareOpenLoopMCTS` directly, not
the multi-objective/bonus/penalty/rerank ladder in `emotion_mcts.py`. To combine
with `EmotionAwareMultiObjectiveQ` (or any of the others), write a further
subclass — do not extend this one in place.

## Why this is structurally different from `EmotionConditionedPriorMCTS`

The existing class conditions on a single `last_user_emotion` argmax and uses a
fixed inference-time temperature. Two problems it had:

1. A user who was furious at turn 0 but calm by turn 3 looks identical to a
   never-angry user at turn 3 — loses the persuasion-relevant history signal.
2. With 13 valid DAs and `N=10` sims, PUCT round-robins (~1 visit per action,
   Q is noise) regardless of how peaky the prior is at decision time.

`EmotionHistoryPriorMCTS` fixes both: full cum-distribution feature + dynamic
temperature that goes much sharper than the existing one's fixed `T=0.7`.

---

## Known gaps / risks

Self-critique. Ranked by potential impact on whether this actually beats GDPZero.

### 1. `alpha=1.0` default *dilutes* the early-anger signal
Arithmetic mean: 1 angry turn early + 9 neutral turns later → `cum_anger = 0.1`.
The whole point ("being angry once or twice ... might affect entire conversation")
is forgotten within ~5 turns at the default. Either:
- Lower default `--ema_alpha` to ~0.7, OR
- Replace mean with **max-pooled-over-window** of negativity (truly persistent), OR
- Add a second feature `max_neg_ever` alongside the EMA.

### 2. Turn 0 has zero learnable signal from the cum vector
At t=0, `history=""` and `cum_dist=uniform` for **every** training example
(~250 identical inputs, all labeled "greeting"). The soft-prune is also dead at
t=0 (`progress=0 → T=T_min` regardless of emotion), so the entire emotion-prior
story only activates from t=1 onward.

### 3. Outcome weighting actively suppresses the hostile-user signal
Failing dialogs get weight 0.3. Failing dialogs over-represent cum-anger turns
(anger correlates with failure in p4g — see `EMOTION_DA_BONUS` notes on Sadness).
The trainer down-weights exactly the turns where the prior should be smartest.
Consider `--negative_weight 1.0` for this specific trainer (imitation-only),
diverging from the sibling trainer's default.

### 4. Train/inference covariate shift in the cum vector
HF classifier never produces Contempt → `cum_dist[contempt]=0.0` in 100% of
training examples → model has no signal for that dim. If anyone runs
`emopriormcts --emotion_classifier llm`, the LLM classifier *does* emit
Contempt and the distribution shifts. Mitigation: assert in `__init__` that
inference classifier matches `meta["label_classifier"]`.

### 5. No verification that T=0.2 actually concentrates mass
DistilRoBERTa on ~2k weighted-CE examples may produce nearly-flat logits on
some inputs (especially the empty-history / uniform-cum t=0 input). Dividing
nearly-equal logits by T=0.2 does NOT sharpen them — soft pruning only works
if the trained model already produces differentiable logits. The trainer has
per-bucket DA-distribution diagnostics but no entropy / top-1-mass at T_min.
Add one — if mean top-1 mass at T_min on val is < 0.4, the soft-prune
assumption fails and `--num_mcts_sims 10` will still round-robin.

### 6. `breadth = positivity × progress` is non-monotonic over the dialog
Action breadth can *shrink* mid-conversation if the user turns hostile,
contradicting "actions accumulate over turns." Two valid readings:
- Conjunctive (current implementation): both terms must grow.
- Monotone accumulator: `breadth = max(breadth_so_far, current_value)` →
  never shrinks. Requires threading state across nodes (currently impossible —
  `_init_node` is stateless).

### 7. Multiplicative form keeps positive-early conversations narrow
`positivity=0.9, progress=0` → breadth=0 → T=T_min. A cheerful user at turn 0
also gets the maximally-sharp prior. Fine in practice (greeting IS what we
want at turn 0), but the runner can never use the wide action set early even
when the user signals receptiveness. Alternative: additive form
`breadth = w_p·positivity + w_d·progress` with bias toward progress.

### 8. Simulated MCTS rollouts have classifier drift from human training data
Inside `search()`, simulated user turns come from the LLM persuadee — their
distributions reflect LLM-persuadee bias, not human-persuadee bias. The cum
vector at deep nodes is computed from a mix of (real human turns) +
(LLM-simulated turns). The model never saw this mix at training time. At
N=10 the effect is modest (depth ≤ 2), but it grows with sims.

### 9. Hostile-user training data is genuinely sparse
Across 250 training dialogs, "user hostile at turn 1" is maybe 20–40 instances.
Not enough for DistilRoBERTa to robustly learn `cum_anger=0.8 → credibility
appeal`. The persona-generation idea in the trainer docstring is the right
long-term answer — but it's not training data yet.

### 10. Soft pruning saves search quality, not LLM compute
Any visited action still generates a realization. If T=0.2 actually drives 8/10
visits to top-1, only ~3 distinct actions get LLM-called per turn → genuine
compute savings. If T=0.2 fails (#5), you pay for all of them.

## Pre-training todo

- Lower default `--ema_alpha` to 0.7 or add `max_neg_ever` as a second feature.
- Add entropy + top-1-mass diagnostic at `T_min=0.2` on val set in the trainer.
  Fail loud if mean top-1 mass < 0.4.
- Default `--negative_weight 1.0` in this specific trainer (imitation-only).
- Add `--emotion_classifier` assertion at runner startup: must match
  `meta["label_classifier"]`.

## Pre-publication todo

- Decide conjunctive vs accumulator semantics for breadth and document which
  the experiments use.
- Either keep `--num_mcts_sims 10` working *honestly* (verify #5 empirically)
  or admit the prior needs N≥20 to differentiate from
  `EmotionConditionedPriorMCTS` and reposition the contribution.
