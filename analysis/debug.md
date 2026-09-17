# Debug log — EmoMCTS configuration sanity & interpretation

Running notes from interpreting unexpected results. Keep entries dated; append at the
bottom. Cross-references:
- `EMOMCTS_ANALYSIS.md` — historical analysis (2026-05 regression, 58.55% baseline)
- `EMOMCTS_ALGORITHM.md` — algorithm spec + per-proposal evaluation
- `EMOMCTS_RESEARCH_DIRECTIONS.md` — proposed extensions (Items 1-5, Investigations,
  Phase plan, Self-play RL)

---

## 2026-05-30 — `EmotionGuidedDiscountQOpenLoopMCTS` with `--c_emo_bonus 1.0` gave 50.00%

### Reported result

- Task: p4g; judge: gpt-3.5-turbo; vs: h2h (GDPZero baseline)
- Win rate: **50.00%** (76 win / 1 draw / 75 lose)
- Config (from `metadata.json`):
  - `mcts_class = EmotionGuidedDiscountQOpenLoopMCTS`
  - `lambda_emo = 0.0`
  - `c_emo_bonus = 1.0`
  - `emotion_classifier = hf`
- Run dir: `src/runners/outputs/emomcts_p4g_20d_discount_v_vicuna__hf_emo_cls__emobonus_1/`

### Important reinterpretation — `λ=0` is NOT "no penalty"

The user described this run as "with no penalty and with bonus action." That is *not*
what `lambda_emo=0.0` means in the current code path.

```python
# src/mcts/emotion_mcts.py — EmotionAwareDiscountQOpenLoopMCTS.search()
if self.lambda_emo > 0.0:
    blended_v = (1.0 - self.lambda_emo) * v + self.lambda_emo * emotion_penalty   # convex
else:
    blended_v = v + emotion_penalty                                                # additive
```

So `λ=0` routes through the **additive** branch — the π penalty is *fully active*. It
just isn't being convex-blended with v. There is currently **no flag for "disable
penalty entirely"** in the runner. The closest workaround is to manually zero out the
entries in `EmotionAwareDiscountQOpenLoopMCTS._get_emotion_penalty`, which isn't an
ablation knob you can flip without code changes.

This conflation has caused multiple confused interpretations across recent runs. Fix:
add a `--disable_penalty` CLI flag that short-circuits to `blended_v = v` regardless of
`λ`. Until then, every "no penalty" claim should be checked against the metadata.

### So what was actually compared

Prior HF + additive run (no bonus) was 45.39%. Today's run added the bonus on top of
the *same* config:

| Run | Classifier | Penalty | Bonus | Win rate |
| --- | --- | --- | --- | --- |
| `..._discount_v_vicuna_no_lambda` (2026-05-30 14:36) | HF | additive | none | **45.39%** |
| `..._hf_emo_cls__emobonus_1` (2026-05-30 21:11) | HF | additive | 1.0 | **50.00%** |

The bonus moved win rate **+4.6pp** in the expected direction. Caveats:

- Single-shot Δ on `n=152`: combined σ ≈ √(2 × 4.0²) ≈ 5.7pp → +4.6pp is ~0.8σ. Right
  direction, noise-floor magnitude. Need 2-3 more reruns per config before this is
  reportable.
- A sister run exists at 21:08: same config but `emotion_classifier=llm`
  (`..._lambda_05_llmemocls_emobonus`, despite the directory name `λ=0.5` the metadata
  shows `lambda_emo=0.0`). Its win rate is the critical comparison — if it's
  meaningfully higher than 50%, HF is the ceiling regardless of the bonus.

### Diagnostics — run before any further A/B

The three questions to answer:

#### 1. Did the bonus actually shift the DA mix?

```bash
python -c "
import json
hist = json.load(open('src/runners/outputs/emomcts_p4g_20d_discount_v_vicuna__hf_emo_cls__emobonus_1/emomcts_p4g_20d_discount_v_vicuna__hf_emo_cls__emobonus_1_da_emotions.json'))
print(json.dumps(hist, indent=2))
"
```

Compare against `..._no_lambda/*_da_emotions.json`. If the bonus is firing meaningfully,
expect:

- `(Happiness, proposition of donation)` cell count rises (was a `+0.4` bonus)
- `(Sadness, emotion appeal)` rises (was `+0.4`)
- `(Anger, proposition of donation)` falls (was `-0.3`)

If histograms look identical, `c_emo_bonus·bonus·explore` is being washed out by
`cpuct·P(a|s)·explore` and the 50% is "bonus invisible," not "bonus shifted to neutral."

#### 2. Top-line DA distribution

```bash
python -c "
import pickle
from collections import Counter
for name in ['emomcts_p4g_20d_discount_v_vicuna_no_lambda',
             'emomcts_p4g_20d_discount_v_vicuna__hf_emo_cls__emobonus_1']:
    with open(f'src/runners/outputs/{name}/{name}.pkl', 'rb') as f:
        out = pickle.load(f)
    print(name)
    print(Counter(r['new_da'] for r in out).most_common())
    print()
"
```

If the bonus is meaningful, `proposition of donation` should rise (it's positive-bonus
on Happiness/Sadness — the two most common emotions in the HF distribution). If it
doesn't, the bonus is firing too weakly.

#### 3. Sister LLM-classifier run

What was the win rate of the 21:08 `..._llmemocls_emobonus` run? Its delta vs the HF
run tells you whether HF is the bottleneck (LLM is meaningfully higher) or whether the
bonus calibration is the bottleneck (both ~50%).

### Decision tree from the diagnostics

**If (1)/(2) show the DA mix didn't shift:** the bonus magnitude is dominated by the
prior. Bump `--c_emo_bonus 2.0` or `3.0` and rerun. If DA mix then shifts but win rate
stays flat, the *direction* of the hand-seeded matrix doesn't match what gpt-3.5
rewards — switch to the data-mined matrix (Item 5b method #1).

**If DA mix shifted as expected but win rate is flat:** the judge doesn't care about
strategic DA selection at this granularity. Future gains live at the *utterance* level
(tone, phrasing) — Phase 1 (toned actions) becomes the next move.

**If the LLM-classifier sister run is meaningfully higher:** HF distribution-softness
dominates everything else. Recalibrate π for HF first (magnitude rescale by ~0.3×, see
`EMOMCTS_RESEARCH_DIRECTIONS.md` Option A under the calibration discussion), then re-run
with bonus.

**Variance check before any of the above:** rerun the *exact same* config 2 more times
with different random seeds. If you see {45, 50, 52} the "current state" is ~49% and
the 45.39% was a low outlier. If you see {50, 50, 50} something is suspiciously
deterministic — judge endpoint caching or OpenAI returning identical logits across
calls.

### The ablation grid we can't currently run

The paper-quality figure needs four cells:

| Penalty | Bonus | What it isolates | Currently runnable? |
| --- | --- | --- | --- |
| off     | off   | GDPZero-equivalent baseline                | yes (run vanilla GDPZero) |
| on      | off   | π penalty contribution alone               | yes (`--c_emo_bonus 0`) |
| off     | on    | Emotion-DA bonus contribution alone        | **no** (no `--disable_penalty` flag) |
| on      | on    | Full EmoMCTS                               | yes (`--c_emo_bonus > 0`) |

Row 3 (penalty-off, bonus-on) is exactly what was *claimed* in today's run but isn't
what was *actually* run. Until we add the flag, every "bonus alone" claim is
contaminated by an active π penalty.

### Suggested code patch

Add to `src/mcts/emotion_mcts.py:EmotionAwareDiscountQOpenLoopMCTS.__init__`:

```python
def __init__(self, game, player, configs, emotion_classifier,
             lambda_emo: float = 0.0, disable_penalty: bool = False) -> None:
    super().__init__(game, player, configs, emotion_classifier)
    self.lambda_emo = lambda_emo
    # NEW: when True, the penalty path is fully short-circuited regardless of λ.
    # Lets the ablation grid in debug.md actually run row 3 (bonus-only).
    self.disable_penalty = disable_penalty or float(getattr(configs, "disable_penalty", 0)) > 0
```

And in `search()`:

```python
if self.disable_penalty:
    blended_v = v
elif self.lambda_emo > 0.0:
    blended_v = (1.0 - self.lambda_emo) * v + self.lambda_emo * emotion_penalty
else:
    blended_v = v + emotion_penalty
```

CLI flag in `src/runners/emomcts.py`:

```python
parser.add_argument('--disable_penalty', action='store_true',
    help='short-circuit the π penalty entirely (blended_v = v). Use with --c_emo_bonus '
         'to isolate the bonus contribution. Without this flag, lambda_emo=0 routes '
         'through the additive branch v + emotion_penalty, NOT no-penalty.')
```

And forward through the planner construction. With this in place the ablation grid is:

| Penalty | Bonus | Flags |
| --- | --- | --- |
| off | off | `--c_emo_bonus 0 --disable_penalty` |
| on  | off | `--c_emo_bonus 0`                  |
| off | on  | `--c_emo_bonus 1 --disable_penalty` |
| on  | on  | `--c_emo_bonus 1`                  |

### Next-action shortlist

1. Run diagnostic (1) and (2) to verify the bonus actually shifted DA selection.
2. Pull win rate of the LLM-classifier sister run (21:08).
3. Land the `--disable_penalty` patch above.
4. Re-run the four-cell ablation grid with the patch in place. That is the figure for
   the paper; everything before it is exploration.
5. Per-config rerun ×2 for variance bounds before reporting any deltas as real.

---

## 2026-05-31 — DA distribution + bonus-matrix recalibration analysis

### Top-line: EmoMCTS chooses *strategically worse* DAs than GDPZero

Compared `emomcts_p4g_20d_discount_v_vicuna__hf_emo_cls__emobonus_1.pkl` (HF + additive
penalty + bonus 1.0) against `gdpzero_vicuna13b.pkl` (vanilla GDPZero baseline). 152
turns each, same 20 dialogs, same SYS-DA prefix order.

| DA | GDPZero | EmoMCTS | Δ |
| --- | --- | --- | --- |
| `other` | 84 | 62 | **−22** |
| `task related inquiry` | 16 | 31 | **+15** |
| `emotion appeal` | 5 | 22 | **+17** |
| `proposition of donation` | 21 | **13** | **−8** |
| `credibility appeal` | 12 | 12 | 0 |
| `logical appeal` | 12 | 11 | −1 |
| `greeting` | 2 | 1 | −1 |

DA-level agreement: 40.8%. The bonus shifted the search **38% fewer propositions, 4.4×
more emotion appeals, 2× more task inquiries** than GDPZero. The top disagreement
patterns are EmoMCTS substituting `task related inquiry` / `emotion appeal` /
`credibility appeal` for GDPZero's `proposition of donation` in 12+ cases.

### Utterance quality — nearly identical, register differs

- Length: mean 26.1 vs 24.6 tokens; medians 23 vs 22 (within noise).
- Only 1/152 utterances are textually identical → genuinely different content.
- Reading examples side-by-side: EmoMCTS utterances aren't *worse* in isolation, just
  **softer and less imperative**. GDPZero closes with concrete impact ("a safe place to
  sleep"); EmoMCTS pivots to inquiries even when the user has already committed
  (`"how about .50 cents"` → GDPZero reinforces, EmoMCTS says "thank you for your
  generosity" before the donation is sealed).

### Diagnosis — Neutral row miscalibrated

HF classifier produces ~40% Happiness, ~35% Neutral, ~8% Sadness, ~17% negatives. The
original `(Neutral, proposition of donation) = -0.10` discouraged closing on 35% of
turns; meanwhile `(Neutral, emotion appeal) = +0.20` boosted three competing DAs above
proposition. PUCT routed to those instead.

**Patch applied** to `EMOTION_DA_BONUS` (`src/mcts/emotion_mcts.py`, 2026-05-31):

```python
Emotions.Neutral: {
    "emotion appeal":           +0.10,    # was +0.20 — soften
    "personal story":           +0.10,    # was +0.20 — soften
    "task related inquiry":     +0.10,    # was +0.20 — soften
    "credibility appeal":       +0.10,    # was 0     — slight boost
    "proposition of donation":  +0.10,    # was -0.10 — flip: judge likes direct asks
},
```

Single-cell change in the most-fired row; rerun to verify direction.

### Data-mined matrix — `src/emotion_mining/mine_emotion_da_bonus_p4g.py`

Ran per-cell donation-lift mining on the 300 annotated p4g dialogs (HF classifier for
emotion labels; Laplace smoothing α=1; min observations per cell = 5; restricted to
strategic DAs only — filter applied to drop reverse-causality response DAs like
`confirm-donation`, `thank`, `praise user` that appear because the user agreed, not
because they caused agreement).

```
loaded 300 dialogs; donate rate 78.7%
cells observed: 147; kept ≥5 obs + strategic: 50
```

Mined bonuses (positive = appears more in donating dialogs):

| Emotion | Top boosters (mined) | Top penalties (mined) |
| --- | --- | --- |
| Happiness | task inquiry +0.10, self modeling +0.10, personal story +0.07 | foot in door −0.10, neutral-inquiry −0.13, source inquiry −0.04 |
| Neutral | self modeling +0.08 | source inquiry −0.21, foot in door −0.16, greeting −0.14 |
| Sadness | (nothing positive) | **personal-rel-inquiry −0.40, emotion appeal −0.24, foot in door −0.21** |
| Surprise | greeting +0.08, self modeling +0.07 | **proposition −0.26, personal-rel-inquiry −0.26** |
| Fear | credibility +0.09 | emotion appeal −0.08 |
| Anger / Disgust / Contempt | (only 1-2 cells survive; uninformative) | — |

### Three findings that contradict the hand-seeded matrix

1. **`(Sadness, emotion appeal)` is strongly negative in the data**, not positive. The
   hand-seeded matrix has it at **+0.40**; the mined matrix has **−0.244**. The
   communication-theory intuition "sad user → emotion appeal" is empirically wrong on
   p4g. Likely interpretation: dialogs where the user was sad late were already
   failing — sadness is endogenous to failure, not a precursor to success.

2. **`(Happiness, proposition of donation)` is essentially neutral, not boost.** The
   hand-seeded matrix has **+0.40**; mined has **−0.028**. Happy users don't want to
   be closed immediately; the data says continued engagement (task inquiry,
   self-modeling, personal story) does better.

3. **`proposition of donation` is mildly negative or neutral in every observed
   emotion.** No emotional state strongly rewards direct asking. The most-positive
   cells across the matrix are `task related inquiry` and `self modeling` at Happy
   states — the data suggests the win comes from *getting the user to volunteer*,
   not from asking better.

### Caveats on the mining

- **78.7% donation rate** → tiny "without_donate" denominators → noisy lift estimates.
- **Reverse causality** affects even filtered cells (a DA can appear in a dialog where
  donation happens for other reasons).
- **Length confound**: longer dialogs contain more DAs AND donate more often, so any
  common DA can look spuriously positive.
- **Negative-emotion samples are tiny**: Anger / Disgust / Contempt have only 1–2
  surviving cells each.

The mined matrix is *not* a drop-in replacement — too noisy to ship as-is — but it's
a strong corrective signal on the most-fired cells.

### Recommended hybrid matrix (apply selectively)

Use mined signal to correct the most-wrong cells of the hand-seeded matrix; keep
hand-seeded values where data is missing (low-occurrence emotions):

| Cell | Hand-seeded | Mined | **Recommended** | Reason |
| --- | --- | --- | --- | --- |
| (Happiness, proposition) | +0.40 | −0.03 | **+0.10** | Halve; judge does like some asks, data says not too aggressive |
| (Happiness, foot-in-door) | +0.30 | −0.10 | **0.00** | Mining strongly negative; drop the boost |
| (Happiness, task inquiry) | −0.10 | +0.10 | **+0.10** | Flip to mined direction |
| (Happiness, personal story) | (not set) | +0.07 | **+0.05** | New positive cell from data |
| (Happiness, self modeling) | (not set) | +0.10 | **+0.10** | New positive cell from data |
| (Sadness, emotion appeal) | +0.40 | −0.24 | **0.00** | Drop the boost entirely; data refutes |
| (Sadness, personal story) | +0.30 | n/a | **0.00** | Drop; data unavailable |
| (Sadness, proposition) | +0.20 | −0.18 | **−0.10** | Flip to mild penalty |
| (Neutral, all engagement) | +0.10 (patched) | ~0 | **+0.05** | Soften further |
| (Neutral, proposition) | +0.10 (patched) | −0.04 | **0.00** | Drop the slight boost |
| (Surprise, proposition) | 0.00 | −0.26 | **−0.15** | Add penalty (new finding) |
| (Anger / Disgust / Contempt) | keep | n/a | keep | Insufficient data |

### What to do next, in order

1. **Apply the hybrid matrix** (the recommended column above) to `EMOTION_DA_BONUS`.
   Re-run with the same `--c_emo_bonus 1.0 --emotion_classifier hf` config.
2. **Compare DA distribution** to the current `..._emobonus_1` run. Hypothesis:
   proposition count rises back toward GDPZero's 21; emotion-appeal count falls from
   22 toward GDPZero's 5; task-inquiry count drops from 31 toward something closer
   to GDPZero's 16.
3. **If DA mix corrects but win rate stays flat**: the GPT-3.5 judge isn't sensitive
   to DA-level differences at all; future improvements have to be at the utterance
   level (Phase 1 toned actions).
4. **If win rate moves positive (+5pp or more)**: ship the hybrid matrix as the new
   default; consider a `LearnedBonus` provider plumbing for the pure-mined variant
   (Item 5b method 3 — imitation lift from successful vs failed dialogs is the
   paper-worthy follow-up that addresses the reverse-causality confound).

### Open methodology question

The mining computes "appears in donating dialog" vs "appears in non-donating dialog"
per cell. A stronger signal would be **conditional on the same dialog prefix** —
"given the user just expressed Happiness at turn t, which DA at turn t+1 maximises
P(donate by turn T)?". That's a per-turn (not per-dialog) framing and requires
sequence-aware modelling. Method 3 in Item 5b is the closest cheap option:
`P_human(da | emotion, dialog donated) − P_human(da | emotion, didn't)`.

The current per-dialog lift is the fast first cut; method 3 would be the next refinement.
