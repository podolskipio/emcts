# Do negative-emotion users donate? — testing the "happy users don't donate"
hypothesis on P4G

## TL;DR

**The persuasion-theory intuition does not survive contact with the P4G data.**
The widely-cited "make donors feel guilty/sad and they'll donate to relieve it"
mechanism shows up only weakly. Happy users actually donate MORE, not less.
Fear is the one negative emotion with genuine donation lift. Sadness slightly
*hurts* donation rates.

The hand-coded `EMOTION_VALENCE` table in
`src/mcts/emotion_mcts.py:538` is closer to right than my earlier "flip the
signs" suggestion. But it still mis-weights Fear (currently -0.3, should be
positive) and treats Neutral as 0 (which is approximately correct on this data).

## The original hypothesis

From classic fundraising literature (Small & Loewenstein 2003 on identifiable
victims; Slovic et al. 2007 on psychic numbing; Dickert et al. 2011 on
sympathy and giving):

> Donations are an emotional resolution. Build sympathy / sadness / guilt about
> the cause, then channel that affect into the donation request. Happy users
> are "polite-disengagement" — they're brushing you off with politeness.

If true, this would imply for Q_emo:

```python
EMOTION_VALENCE_HYPOTHETICAL = {
    Sadness:   +0.5,   # sympathy = engine
    Fear:      +0.2,   # fear appeals work
    Happiness: -0.3,   # politeness ≠ persuasion
    Neutral:   +0.1,   # open listening
    Surprise:  +0.4,
    Anger:     -1.0,   # real hostility
    Disgust:   -0.7,
    Contempt:  -0.6,
}
```

I asked the data whether that's right.

## The analysis

Script: `src/emotion_mining/mine_emotion_donation_p4g.py`. Reproducible with:

```bash
python src/emotion_mining/mine_emotion_donation_p4g.py
```

- 300 P4G dialogs.
- HF classifier (`j-hartmann/emotion-english-distilroberta-base`) — deterministic,
  cached on disk.
- Dialog "successful" = any user turn carries the `agree-donation` label.
- Base donation rate: **49.3%** (148/300).

### Q1) Which user emotion is associated with most donations?

**(a) Per user turn** — emotion of THIS turn → outcome of its parent dialog:

| Emotion | n_turns | P(donate) | lift vs base | 95% CI |
| --- | ---: | ---: | ---: | --- |
| **Fear** | 70 | **0.600** | **+0.107** | [0.483, 0.707] |
| Happiness | 1145 | 0.552 | +0.059 | [0.523, 0.581] |
| Anger | 88 | 0.534 | +0.041 | [0.431, 0.635] |
| Disgust | 92 | 0.533 | +0.039 | [0.431, 0.631] |
| Surprise | 469 | 0.510 | +0.016 | [0.464, 0.555] |
| Neutral | 2660 | 0.484 | −0.009 | [0.465, 0.503] |
| **Sadness** | 323 | **0.458** | **−0.035** | [0.405, 0.513] |
| Contempt | 0 | — | — | (HF model never emits Contempt) |

**(b) Per dialog** — dialog's dominant user emotion → its outcome:

| Dominant | n_dialogs | P(donate) | lift |
| --- | ---: | ---: | ---: |
| Sadness | 1 | 1.000 | +0.507 (n=1, ignore) |
| Surprise | 6 | 0.667 | +0.173 |
| **Happiness** | 37 | **0.622** | **+0.128** |
| Neutral | 256 | 0.469 | −0.025 |

**Read:**
- The user emotion with the highest *per-turn* donation rate is **Fear** (+11pp lift).
  Fear-appeal literature is at least partly vindicated.
- The emotion that dominates *successful* dialogs is **Happiness** (+13pp lift).
  Happiness is associated with conversion, not disengagement.
- **Sadness is the only emotion with a negative lift.** The classical "sympathy
  drives donation" mechanism does not show up in this data — Sadness here looks
  like an *endogenous failure signal*, not a productive state. Matches the
  existing `EMOTION_DA_BONUS` comment: "sadness in p4g is endogenous to failing
  dialogs, not an opportunity."

### Q2) Does changing emotion (happy → unhappy, etc.) boost donations?

8×8 transition table (rows = prev user emotion, cols = next user emotion,
cells = P(donate); n ≥ 5 only):

```
              happin  sadnes    fear   anger  surpri  disgus  neutra
   happiness    0.61    0.37    0.67    0.62    0.51    0.81    0.53
     sadness    0.54    0.50    0.67    0.57    0.40    0.40    0.41
        fear    0.53    0.29       .       .    0.50       .    0.73
       anger    0.19    0.80       .       .    0.83       .    0.57
    surprise    0.58    0.60    0.33    0.75    0.53    0.20    0.47
     disgust    0.75    0.41       .       .    0.88       .    0.44
     neutral    0.51    0.48    0.56    0.47    0.50    0.47    0.47
```

Aggregated tests of the hypothesis:

| Transition class | n | P(donate) | lift |
| --- | ---: | ---: | ---: |
| **happy → happy** | 309 | **0.612** | **+0.118** |
| happy → negative | 109 | 0.505 | +0.011 |
| neutral → happy | 538 | 0.515 | +0.022 |
| neutral → negative | 277 | 0.487 | −0.006 |
| negative → happy | 108 | 0.509 | +0.016 |
| negative → negative | 110 | 0.536 | +0.043 |

**Verdict on "happy → unhappy boosts donation": NO, the opposite is true.**

- `happy → happy` (61.2%) beats `happy → negative` (50.5%) by **+11pp**.
- The hypothesis would have predicted the reverse.
- The single best transition with a meaningful sample is `fear → neutral`
  (72.7%, n=33): fear-then-relief looks productive, but is rare. Most of the
  big-lift transitions in the table have n < 20 and are likely noise.
- `anger → happiness` is the *worst* common transition (18.8%, n=16). Once
  hostility shows up, surface-level reconciliation does not predict donation.

### What this means for `EMOTION_VALENCE`

The current table (Happiness=+1.0, Neutral=0, Sadness=-0.2) is *qualitatively
correct in direction* but mis-weighted in magnitude. A data-grounded replacement:

```python
# Calibrated from outputs/emotion_donation_analysis.json (per-turn lift × 10).
# Magnitudes intentionally modest — these are correlational lifts, not causal
# coefficients, and the largest cell (Fear) has n=70.
EMOTION_VALENCE_MINED = {
    Emotions.Fear:      +1.07,   # was -0.30 — flip sign
    Emotions.Happiness: +0.59,   # was +1.00 — halve
    Emotions.Anger:     +0.41,   # was -1.00 — flip; n is small (88), keep skeptical
    Emotions.Disgust:   +0.39,   # was -0.70 — flip; small n
    Emotions.Surprise:  +0.16,   # was +0.40 — halve
    Emotions.Neutral:   -0.09,   # was  0.00 — slight tax on apathy
    Emotions.Sadness:   -0.35,   # was -0.20 — moderate, data agrees on direction
    Emotions.Contempt:  -0.60,   # HF never emits — kept as hand value for LLM-classifier fallback
}
```

Caveats:
- Anger and Disgust having positive per-turn lift is *suspicious* — likely
  small-sample artefact or "user is engaged enough to be irritated by something."
  Trust direction less than magnitude on those rows.
- These are correlations: "turns where the user was X happened in dialogs that
  ended in donation." Not causal — does NOT mean inducing X drives donation.

## Two design moves this analysis enables

### Move 1 — Drop the hand-coded `EMOTION_VALENCE`

Quickest patch: replace `EMOTION_VALENCE` at `src/mcts/emotion_mcts.py:538`
with `EMOTION_VALENCE_MINED` above. `EmotionAwareMultiObjectiveQ` automatically
picks it up via `_emotion_quality`. No new class.

But: the table is still a single-scalar mapping. The transition table shows
that *context matters* (e.g., `anger → happiness` is terrible but
`disgust → happiness` is great). One scalar per emotion loses that.

### Move 2 — Target a *trajectory*, not a state

Instead of asking "is the current user emotion good or bad?" ask "are we on a
trajectory of (prev_emotion → next_emotion) that historically donates?" The
mined transition table gives that directly.

That is the new class proposed below.

## Code snippet 1 — new MCTS subclass that scores transitions

`src/mcts/emotion_transition_q_mcts.py` — sketch (not yet written; this is the
proposed shape):

```python
from emotion_classifiers.llm_emotion import Emotions
from mcts.emotion_mcts import EmotionAwareOpenLoopMCTS
import json, math

# Loaded once at import from outputs/emotion_donation_analysis.json's
# transition_table_p_donate. Keys are "prev->next" strings; values are
# P(donate | this transition).
with open("outputs/emotion_donation_analysis.json") as f:
    _MINED = json.load(f)
TRANSITION_LIFT: dict = {
    k: v - _MINED["base_donation_rate"]
    for k, v in _MINED["transition_table_p_donate"].items()
    if v is not None
}

def _prev_user_emotion(history) -> str | None:
    """Most-recent persuadee emotion BEFORE the latest persuadee turn."""
    persuadee = [r for r in history if r.role == "Persuadee"]
    return str(persuadee[-2].emotion) if len(persuadee) >= 2 else None


class EmotionTransitionQMCTS(EmotionAwareOpenLoopMCTS):
    """Parallel Q channel scored by the MINED transition lift
    P(donation | prev_user_emotion -> next_user_emotion) - base_rate.

    At backup time, after the simulated user has replied, we know
    (prev_user_emo, next_user_emo). Look up its mined lift, back up via the
    same running-mean as Q_emo in EmotionAwareMultiObjectiveQ. Selection
    PUCT adds beta_trans * Q_trans.

    Differs from EmotionAwareMultiObjectiveQ in WHAT is backed up:
    valence(next_state) (a state property) vs. lift(prev->next) (an edge
    property). The edge formulation captures context-dependent effects the
    scalar valence cannot — e.g. that `disgust -> happiness` is +25pp and
    `anger -> happiness` is -31pp, even though they end in the same state.
    """

    def __init__(self, game, player, configs, emotion_classifier,
                 beta_trans: float = 0.5) -> None:
        super().__init__(game, player, configs, emotion_classifier)
        self.beta_trans = float(beta_trans)
        self.Q_trans: dict = {}

    def _transition_value(self, state, next_state) -> float:
        prev = _prev_user_emotion(state.history)
        nxt = str(next_state.predicted_emotion()) if next_state.history else None
        if prev is None or nxt is None:
            return 0.0
        return TRANSITION_LIFT.get(f"{prev}->{nxt}", 0.0)

    def _init_node(self, state):
        v = super()._init_node(state)
        h = self._to_string_rep(state)
        self.Q_trans[h] = {a: 0.0 for a in self.valid_moves[h]}
        return v

    def _calculate_uct(self, h, a) -> float:
        Ns = self.Ns[h] or 1e-8
        explore = math.sqrt(Ns) / (1 + self.Nsa[h][a])
        return (
            self.Q[h][a]
            + self.beta_trans * self.Q_trans.get(h, {}).get(a, 0.0)
            + self.configs.cpuct * self.P[h][a] * explore
        )

    def search(self, state):
        # parent's loop, with a Q_trans backup alongside the donation backup
        h = self._to_string_rep(state)
        if self.game.get_dialog_ended(state) == 1.0:
            return 1.0
        if h not in self.P:
            return self._init_node(state)
        self._add_new_realizations(state)

        best_uct, best_a = -float("inf"), -1
        for a in self.valid_moves[h]:
            u = self._calculate_uct(h, a)
            if u > best_uct:
                best_uct, best_a = u, a
        sampled = self._sample_realization(h)
        next_state = self._get_next_state(sampled, best_a)
        v = self.search(next_state)

        nsa_old = self.Nsa[h][best_a]
        self.Q[h][best_a] = (nsa_old * self.Q[h][best_a] + v) / (nsa_old + 1)

        trans_v = self._transition_value(sampled, next_state)
        self.Q_trans[h][best_a] = (
            nsa_old * self.Q_trans[h][best_a] + trans_v
        ) / (nsa_old + 1)

        self.Ns[h] += 1
        self.Nsa[h][best_a] += 1
        self._update_realizations_Vs(next_state, v)
        return v
```

**Why this is a real improvement over `EmotionAwareMultiObjectiveQ`:**
- That class scores by `valence(next_emotion)` — a state property. It can't
  distinguish "we just rescued the user from disgust → happy" (+25pp) from "we
  pissed off the user, who is now fake-happy" (-31pp).
- This class scores by the *edge* (prev_emo, next_emo). The transition table
  IS the context.
- Same compute envelope as `EmotionAwareMultiObjectiveQ` (one extra dict
  lookup per backup).

## Code snippet 2 — utterance-tone steering (not just DA selection)

Currently MCTS chooses a *strategic dialog act* (e.g. `proposition of donation`)
and the realization layer samples a few utterances for that DA. None of those
layers ask "what *tone* should the persuader use to elicit the desired user
emotion?"

Two ways to add tone control. The first is cheaper (prompt-only), the second
is the full RL-style version.

### 2a — Tone-conditioned realization sampling (prompt-only)

Add a tone tag to the persuader system prompt at realization time. The MCTS
decides BOTH the DA and the target tone, scored by which (DA, tone) pair
historically produces the user emotion the transition table predicts will be
productive.

```python
# Five tones the persuader can deliberately project. Could be expanded.
PERSUADER_TONES = [
    "sympathetic",       # warm, validating — induces user happiness/relief
    "urgent",            # immediate, time-pressured — induces fear/engagement
    "factual",           # neutral, source-cited — induces neutral attention
    "personal-vulnerable",  # self-disclosing — induces user happiness/sympathy
    "challenging",       # confronts ambivalence — induces surprise/anger
]


class TonedPersuaderChatModel(PersuaderChatModel):
    """Persuader that accepts a `tone` keyword whose effect is to inject a
    short tone-instruction sentence at the head of its system prompt.

    Same backbone model. The tone is just a prompt-side conditioning variable.
    """

    TONE_INSTRUCTIONS = {
        "sympathetic":          "Speak warmly and validate the persuadee's feelings.",
        "urgent":               "Convey time-pressure — children are suffering RIGHT NOW.",
        "factual":              "Be sober and source-cite Save the Children's track record.",
        "personal-vulnerable":  "Share a brief first-person reason this cause matters to you.",
        "challenging":          "Gently confront the persuadee's hesitation directly.",
    }

    def generate(self, state, tone: str | None = None, **kwargs):
        sys_prompt = self.base_system_prompt
        if tone is not None and tone in self.TONE_INSTRUCTIONS:
            sys_prompt = (
                f"TONE INSTRUCTION (this turn only): {self.TONE_INSTRUCTIONS[tone]}\n\n"
                + sys_prompt
            )
        return super().generate(state, system_prompt=sys_prompt, **kwargs)
```

Then in the MCTS, treat `(da, tone)` as the action space:

```python
class TonedEmotionTransitionQMCTS(EmotionTransitionQMCTS):
    """Action space = (dialog_act, persuader_tone). The persuader is asked to
    realize the chosen DA with the chosen tone. Per-action Q is now indexed by
    the pair, so the planner can learn that 'proposition of donation +
    sympathetic' converts better than 'proposition of donation + urgent' on
    this particular user.
    """

    def _init_node(self, state):
        # Expand the action set from len(dialog_acts) to len(dialog_acts) * len(tones).
        v = super()._init_node(state)
        h = self._to_string_rep(state)
        # rebuild Q / Nsa / Q_trans keyed by composite (da_idx, tone_idx)
        composite_actions = [
            (a, t) for a in self.valid_moves[h] for t in range(len(PERSUADER_TONES))
        ]
        self.Q[h] = {ca: self.configs.Q_0 for ca in composite_actions}
        self.Nsa[h] = {ca: 0 for ca in composite_actions}
        self.Q_trans[h] = {ca: 0.0 for ca in composite_actions}
        # tone prior = uniform; the DA prior is replicated across tones
        self.P[h] = {
            (a, t): self.P_da[h][a] / len(PERSUADER_TONES)
            for a in self.valid_moves[h] for t in range(len(PERSUADER_TONES))
        }
        self.valid_moves[h] = composite_actions
        return v

    def _get_next_state(self, state, composite_action):
        a, t = composite_action
        # forward tone into the persuader's generation call via the game
        return self.game.get_next_state(state, a, tone=PERSUADER_TONES[t])[0]
```

**Tradeoff:** action space grows by `len(tones)` = 5x. At N=10 sims this
under-samples badly. Needs N≥40, OR a learned tone prior (per-emotion mining of
which tone-DA pairs converted) to keep search tractable.

### 2b — Train a tone policy from the mined transition table

Use the analysis output to *learn* the tone choice rather than search over it.
Train a small classifier:

```
P(tone | state, cum_user_emotion) → optimised to maximise
E[P(donate | next_user_emotion)] under the mined transition table.
```

This collapses the 5x action-space blow-up back to 1x at search time — the
tone is chosen by the learned policy, not enumerated.

Pipeline sketch:
1. For each (state, DA, persuader_utterance) tuple in the 300 dialogs, label
   the persuader utterance with the tone classifier (hand-defined rules or a
   second HF model: `emotion-english-distilroberta-base` on the persuader text,
   re-mapped to tone categories).
2. Roll the dialog forward one user turn → observe `next_user_emotion`.
3. Look up `TRANSITION_LIFT[prev_user_emo -> next_user_emo]`.
4. Train classifier to predict tone weighted by transition lift.

Compute cost: trivial (one DistilBERT pass per persuader turn). Pay-off: the
persuader can be steered to *trigger* the productive transitions the mined
table identifies — `fear → neutral` (+23pp) etc. — instead of just reacting to
whatever emotion the user happens to be in.

## What to do next

1. **Replace `EMOTION_VALENCE` with `EMOTION_VALENCE_MINED`** — single dict
   swap, no class needed, costs nothing. Re-run Block A of the experiment
   script. If +14pp improvement persists, the original result was robust to
   the table choice (search-budget effect dominating). If it changes
   meaningfully, the table choice was load-bearing.

2. **Decide whether the transition-Q class is worth implementing.** The
   transition table has signal but most cells have n<20. Worth doing only if
   you want the *paper claim* "we use mined transition dynamics" rather than
   "we use mined per-emotion valence."

3. **Tone control is the bigger algorithmic move** and the more publishable.
   Currently no GDPZero-derivative reasons about tone. But it requires the
   persuader model to actually respond to tone tags — quick check: prompt
   `gpt-3.5-turbo` with a tone tag and verify the generated utterance shifts
   noticeably. If the model ignores the tag, this whole line of work needs a
   fine-tuned persuader (much bigger project).
