# EMOMCTS vs GDPZERO — Why EMOMCTS is losing (40% wins)

> ⚠ **Historical debugging note.** It analyses `EmotionAwareOpenLoopMCTS` (the earlier single-channel
> planner), not the shipped `EmotionAwareMultiObjectiveQ`, and predates the Thursday freeze. Kept for
> the record of why the single-channel design was abandoned. Current design: `FREEZE_NOTES.md` §10.

Both runners are evaluated on the same vicuna:13b backbone via `run_judge.py` (`chat_gpt_3-5`).
The only meaningful difference is `EmotionAwareOpenLoopMCTS` (in `src/mcts/emotion_mcts.py`)
vs `OpenLoopMCTS` (in `src/mcts/mcts.py`). The runners themselves are near-identical clones.

Findings ranked by likelihood of being the cause.

---

## 1. Smoking gun: emotion penalty compounds through the recursion

In `src/mcts/emotion_mcts.py:182-192`:

```python
blended_v = v + emotion_penalty
self.Q[hashable_state][best_action] = (...Q + blended_v) / (... + 1)
...
self._update_realizations_Vs(next_state, blended_v)
return blended_v          # <-- propagates the penalty up the tree
```

`search()` returns `blended_v` instead of `v`. The parent caller then does
`v_parent = blended_v_child`, then adds its OWN penalty: `blended_v_parent =
blended_v_child + penalty_parent`. Penalties stack down the entire backup chain.

### Empirical confirmation (from the user's own pickles)

| Stat            | EMOMCTS    | GDPZERO    |
| --------------- | ---------- | ---------- |
| Q min           | **-1.700** | -1.000     |
| Q mean          | -0.026     | -0.013     |
| Q % below -1    | **0.4%**   | 0.0%       |
| realVs min      | **-1.750** | -1.000     |
| realVs median   | **-0.100** | 0.000      |
| realVs mean     | +0.055     | **+0.143** |
| realVs % < 0    | 54.3%      | 48.4%      |
| realVs % < -1   | **2.6%**   | 0.0%       |

Original `v` from `player.predict` is bounded in `[0, 1]`, so GDPZERO's Q/realVs
respect `[-1, +1]` (the floor comes from `get_dialog_ended = -1`). EMOMCTS values
extend *below* -1, which is only possible through compounded `blended_v` in nested
`search()` calls.

---

## 2. Penalty magnitude is mismatched to the value range

`_get_emotion_penalty` returns `-1.0` (Anger) down to `-0.4` (Sadness), but
`v ∈ [0, 1]`. A single Sadness label flips the entire task signal. This is
saying "user feels sad once" is worth as much as "persuasion fully fails."

Quick fix candidates:
```python
blended_v = (1 - λ) * v + λ * penalty       # with λ ≈ 0.1–0.2
# or
penalty *= 0.2
```

---

## 3. Penalty poisons utterance selection, not just policy

`_update_realizations_Vs(next_state, blended_v)` writes the penalized value into
`realizations_Vs`, which `get_best_realization` argmaxes over to pick the
utterance to actually emit (`emomcts.py:111`). Median realization V is **-0.1**
in EMOMCTS vs **0.0** in GDPZERO. Even when MCTS chooses the same DA, the
utterance it emits is biased toward whichever sample happened *not* to trigger a
Sadness label downstream — which is mostly noise, not quality.

---

## 4. Conceptually wrong reward shape for persuasion

The emotion at `next_state` is the user's reaction to a persuasion attempt. The
acts that actually persuade — `proposition of donation`, `emotion appeal`,
`foot in the door` — are the ones that most often elicit Sadness/Fear from a
persuadee (guilt, sympathy, urgency). Inquiry/logical appeal mostly elicit
Neutral. So `_get_emotion_penalty` is structurally biased toward the *least*
persuasive acts.

### DA distribution confirms it

| DA chosen                | EMOMCTS | GDPZERO | Δ      |
| ------------------------ | ------- | ------- | ------ |
| proposition of donation  | 19      | 23      | **-4** |
| credibility appeal       | 18      | 20      | -2     |
| task related inquiry     | 19      | 19      | 0      |
| emotion appeal           | 8       | 8       | 0      |
| logical appeal           | **11**  | 6       | **+5** |
| greeting                 | 4       | 2       | +2     |
| other                    | 73      | 74      | -1     |

EMOMCTS shifted *away* from donation-proposition and *toward* logical-appeal /
greeting. That's exactly the wrong direction for a p4g judge.

### Emotion distribution (from `_emotions.json`)

```
happiness:  74.9%
neutral  :  15.3%
sadness  :   6.8%
surprise :   1.4%
fear     :   1.1%
anger    :   0.4%
contempt :   0.1%
disgust  :   0.1%
```

Penalty fires ~10% of turns; when it does, it's a big -0.4 to -1.0 hit, which
the compounding bug then amplifies.

---

## 5. Smaller issues (not currently behavioral, but worth fixing)

### `_get_next_state_emotions` is dead code

`update_emotions` stores at key `state_hash + "__" + da`:
```python
next_state_hash = self._get_hash_for_next_action(self._to_string_rep(current_state), next_action)
self.emotions_count[next_state_hash][emotion] += 1
```
but `_get_next_state_emotions(action)` looks up by bare `action` (int):
```python
if action in self.emotions_count:    # always False
```
Always returns `{}`. The emotion-count machinery isn't actually plumbed in.

### Cached-realization branch skips `update_emotions`

`_get_next_state:102-104` returns early from the cached path without recording
an emotion. Inconsistent with the non-cached path.

---

## Suggested fix order

1. **Stop propagating the penalty.** Apply it locally only:
   ```python
   self.Q[s][a] = (Nsa*Q + (v + penalty)) / (Nsa + 1)
   self._update_realizations_Vs(next_state, v + penalty)
   return v   # leaf value, unchanged
   ```
2. **Shrink the magnitude** to `~0.1–0.2 × v` scale.
3. **Reverse the sign for "engaged" negatives** in p4g specifically. For
   persuasion, Sadness/Fear in the persuadee is often a *good* sign (emotional
   engagement). Penalize only Anger/Disgust/Contempt; give 0 (or a small bonus)
   to Sadness/Fear.
4. **Ablate the penalty entirely** (`penalty = 0` for all emotions) and re-run.
   If EMOMCTS then ties GDPZERO at ~50%, the only remaining gap is the
   emotion-aware string rep / classifier-induced realization caching, and
   you've cleanly isolated the problem.

Start with **(1) + (4)** in parallel: the local-only patch vs the zero-penalty
ablation will tell you which of "compounding bug" vs "wrong reward shape" is
dominant.
