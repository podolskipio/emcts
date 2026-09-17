# Action-distribution analysis: GDP-Zero vs EmoMCTS

Two policy-behaviour figures for the Experiments section, both on the 40-sim / 50-dialogue
PersuasionForGood runs (Vicuna-13B). Reproduce with the scripts noted under each figure.

---

## Figure A — System dialogue-act distribution by turn

![DA by turn](src/outputs/da_hist_gdpzero_vs_emomcts_40s.png)

*Per-turn fraction of dialogues in which each system act was played, GDP-Zero (left) vs EmoMCTS
(right). Reproduce:* `python scripts/plot_da_histogram.py` *(figure + `da_hist_*.csv`).*

**Fractions** (rows = system turn, columns = act):

| Turn | greeting | task-inq. | logical | credibility | **emotion** | **proposition** |
|---|---|---|---|---|---|---|
| **GDP-Zero** ||||||
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.06 | 0.59 | 0.18 | 0.16 |
| 2 | 0.08 | 0.04 | 0.10 | 0.23 | 0.44 | 0.10 |
| 3 | 0.16 | 0.02 | 0.07 | 0.14 | 0.50 | 0.11 |
| 4 | 0.00 | 0.15 | 0.12 | 0.06 | 0.29 | 0.38 |
| 5 | 0.00 | 0.00 | 0.07 | 0.00 | 0.57 | 0.29 |
| **EmoMCTS** ||||||
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.06 | 0.14 | 0.49 | 0.16 | 0.14 |
| 2 | 0.02 | 0.07 | 0.05 | 0.17 | 0.48 | 0.21 |
| 3 | 0.05 | 0.08 | 0.11 | 0.08 | 0.58 | 0.11 |
| 4 | 0.03 | 0.03 | 0.10 | 0.07 | 0.43 | 0.33 |
| 5 | 0.00 | 0.06 | 0.12 | 0.06 | 0.44 | 0.31 |

**Findings.**
- **EmoMCTS sustains emotion appeals deeper into the dialogue.** Emotion appeal stays at
  0.43–0.58 from turn 2 through turn 5; GDP-Zero is more erratic (0.29 at turn 4) and reverts to
  credibility/greeting.
- **EmoMCTS proposes earlier and more steadily.** Donation propositions appear already at turn 2
  (0.21) and hold at turns 4–5 (0.33, 0.31); GDP-Zero defers them (≈0.10 until turn 4, then a
  late spike). This matches EmoMCTS's lower AvgT (6.62 vs 7.12): build rapport, then close
  sooner.
- **GDP-Zero front-loads credibility appeal** (0.59 at turn 1) and shows more "stuck" behaviour
  (it replays greeting at turns 2–3).

---

## Figure B — Action choice conditioned on the user's last emotion

![Emotion-conditioned actions](src/outputs/emotion_conditioned_actions_40s.png)

*Given the user's most recent emotion (negative vs non-negative), the distribution of the next
system action over three groups: trust-building = {emotion, credibility appeal}; proposition;
other = {greeting, task inquiry, logical appeal}. EmoMCTS logs emotions directly; GDP-Zero's
user turns are labelled with the same deterministic HF classifier. Reproduce:*
`python scripts/plot_emotion_conditioned_actions.py` *(figure + `emotion_conditioned_*.csv`).*

**Fractions:**

| Condition | $n$ | trust-building | proposition | other |
|---|---|---|---|---|
| GDP-Zero — after **negative** | 15 | 0.600 | 0.200 | 0.200 |
| **EmoMCTS — after negative** | 15 | **0.733** | 0.200 | **0.067** |
| GDP-Zero — after non-negative | 199 | 0.628 | 0.181 | 0.191 |
| EmoMCTS — after non-negative | 178 | 0.590 | 0.208 | 0.202 |

**Findings.**
- **After a negative user emotion, EmoMCTS shifts toward trust-building** (0.73 vs 0.60) and
  almost eliminates off-topic "other" acts (0.07 vs 0.20) — it does *not* push the proposition
  harder (0.20 in both), it *repairs* first. This is the $\beta Q_{\text{emo}}$ mechanism made
  visible.
- **After a non-negative emotion the two planners are nearly identical** — EmoMCTS's behavioural
  difference is *concentrated in the emotionally-charged states*, exactly where the emotion
  channel should act, rather than being a uniform shift.

**Caveat.** The HF classifier labels few p4g user turns as negative, so the negative-emotion
buckets are small ($n=15$ per planner); the direction is consistent with the mechanism but the
negative-condition cells should be read as indicative, not conclusive. A larger evaluation set
(or pooling the 100-dialogue sweep runs) would tighten them.

---

### Reproduction

```bash
cd src
python ../scripts/plot_da_histogram.py \
    --gdpzero rollout_p4g_gdpzero_vicuna_50d_40s \
    --emomcts rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims \
    --out outputs/da_hist_gdpzero_vs_emomcts_40s.png \
    --csv outputs/da_hist_gdpzero_vs_emomcts_40s.csv

python ../scripts/plot_emotion_conditioned_actions.py \
    --gdpzero rollout_p4g_gdpzero_vicuna_50d_40s \
    --emomcts rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims \
    --out outputs/emotion_conditioned_actions_40s.png \
    --csv outputs/emotion_conditioned_actions_40s.csv
```

Both scripts accept either a run-id stem (searched under the cwd) or an explicit `.pkl` path,
and a `--max_turn` knob (Figure A). The HF user-emotion labels for GDP-Zero are cached in
`outputs/gdpzero_user_emocache.json` so reruns are instant.
