# Research Directions: Boosting Persuasiveness in EmotionAwareDiscountQOpenLoopMCTS

> ⚠ **Historical design space.** Which directions were built, gated or dropped is settled in
> `FREEZE_NOTES.md` §10 and `PREREG.md` Entries 1–6 (TrajValue and TrajPrompt are dead: `thu/trajvalue_gates.md`).

Working notes on where to push next on the emotion-aware planner. Grouped by pipeline
location; starred items are the highest-expected-ROI starting points.

## 1. Reward / value shaping (cheap, high leverage)

- **⭐ Convex blend with tunable λ** — revive `v~ = (1−λ)v + λ·Σp·π(e)` (the `lambda_emo`
  arg that was dropped). Today the penalty is added raw, so its scale can dominate
  `v ∈ [−1, 1]`.
- **⭐ Backup the emotion penalty up the tree** — currently local-only (Q update at the
  expanded node only). Persuasion is multi-turn: a DA that produces brief discomfort but
  unlocks a positive trajectory should be rewarded; local-only shaping cannot see that.
- **Trajectory-based reward** — bonus for negative → positive emotional shifts, penalty for
  the reverse. Persuasion theory says the *delta* matters more than the snapshot.
- **Learn `π(e)` from data** — regress per-emotion penalty values against actual donation
  outcomes in the p4g dataset instead of the hand-tuned table in
  `src/mcts/emotion_mcts.py:_get_emotion_penalty`.
- **Entropy-as-exploration** — high-uncertainty user emotion ⇒ more rollouts at that node
  (extra PUCT bonus proportional to `H(p(e))`).

## 2. Emotion modeling

- **⭐ Anticipatory `P(emotion | state, DA)`** — train a small model that predicts the
  expected user emotion *before* running the user-LLM rollout. Lets the planner score DAs
  without paying the rollout cost, and acts as a prior in PUCT.
- **Calibrate the LLM classifier on labeled p4g data** — current path is zero-shot + few-shot.
  Classifier biases enter both rollouts and reward shaping (a confound the HF variant only
  partially mitigates).
- **Coarse valence + fine-grained ensemble** — two-stage (valence ⇒ fine class). Reduces
  noise on the rare classes (Contempt, Disgust) that carry the biggest penalties.
- **Average HF + LLM distributions** — removes the shared-backbone confound between policy
  and reward without losing context-awareness.

## 3. Search / policy improvements

- **⭐ Better prior** — train `π_θ(a | s, last_emotion)` on successful p4g dialogs and use it
  as the PUCT prior (currently the zero-shot planner heuristic).
- **Progressive widening on productive DAs** — narrow to DAs that historically combine high
  donation rate *and* non-negative emotion. Most of the 13 DAs are rarely productive.
- **Condition emotion classification on the system DA** — e.g. `foot-in-the-door` after
  `personal-story` produces different emotions than after `credibility-appeal`. Cheap
  context win.
- **Multi-realization at the DA *and* utterance level** — today realizations vary the user
  response, but the DA→utterance mapping is single-shot. Best-of-N system utterances
  re-ranked by a persuasiveness critic would tighten realization quality without changing
  the search algorithm.

## 4. User simulator (the rollout's silent assumption)

- **⭐ Persona-conditioned user simulator** — skeptical / sympathetic / altruistic personae
  produce wildly different optimal strategies. Currently we average over an undefined
  population.
- **Emotion-consistent user agent** — have the user-LLM condition on its own predicted
  emotion trajectory (today it is unaware of the emotion it just expressed).
- **Retrieval-augmented user simulator** — ground rollouts in real p4g user lines so the
  simulator cannot drift into LLM-flavored "agreeable user" mode.

## 5. Reward signal richness

- **Graded donation amount** — where the p4g data has it (not just binary donate / no-donate).
- **Interim reward signals** — questions, positive reactions, agreement markers as small
  mid-dialog rewards. Smooths the credit-assignment problem the current sparse terminal
  reward causes.
- **⭐ Learned critic** — train a value head on full dialogs predicting `P(donate | state)`
  and use it for `Q_0` / leaf evaluation instead of the heuristic.

## 6. Inference-time cheap wins

- **Self-critique on selected DA** — after argmax, ask the LLM to predict the user's emotional
  reaction; veto if it predicts strong negative emotion.
- **Best-of-N realizations re-ranked by a separate persuasiveness / empathy reward model**
  (must be a different model than the classifier to avoid the confound).

## Suggested starting point (highest ROI, lowest engineering)

1. Fix the missing λ blend + propagate emotion penalty up the backup chain
   (small MCTS change, addresses a known modeling gap).
2. Train an anticipatory `P(emotion | state, DA)` — kills two birds: prior for PUCT *and*
   removes most LLM-classifier calls inside rollouts.
3. Persona-conditioned user simulator — currently the most under-specified component in the
   system; everything downstream is averaging over an unknown distribution.

---

# Detailed investigations

Deeper write-ups + concrete code snippets for four of the ideas above. Cross-references to
prior work in this repo are flagged as `EMOMCTS_ANALYSIS.md §N` / `EMOMCTS_ALGORITHM.md §N`.

## Investigation 1 — Backing the emotion penalty up the tree

### Prior result in this repo

`EMOMCTS_ANALYSIS.md §1` and `EMOMCTS_ALGORITHM.md "Why we can't recursively return the
discounted value"` already establish that the original `search()` returned `blended_v =
v + π(e)` to its caller. That is *not* discounting — it's an **additive shaping sum** with
no contraction factor. Five concrete failures were documented:

1. PUCT magnitude contract violated (Q escapes `[-1, +1]`; one branch's accumulated
   penalties get arbitrarily large, exploration term stays fixed → search collapses to
   "avoid every visited branch").
2. Same emotion is credited to every ancestor (double-counting).
3. Subtree depth becomes a hidden penalty multiplier; algorithm implicitly prefers
   actions that lead to *early termination*, which in p4g means `no-donation` paths get
   implicit favor.
4. Variance in `Q` grows with visits (more rollouts → noisier `Q`, the opposite of MCTS).
5. `v_leaf` from `player.predict` is already a value-function estimate (expected
   long-run return). Adding per-step shaping rewards to it without a Bellman structure is
   incoherent.

Empirically, the current shipped fix (local-only application; `search()` returns leaf `v`
unchanged) is the **+8.5pp winner** over GDPZero (58.55% vs ~50%, see
`EMOMCTS_ALGORITHM.md §Empirical results`).

### So what's actually worth investigating next?

The naive "return `v + π(e)` recursively" idea is dead. The salvageable direction is:

> **Treat π(e) as a real per-step shaping reward inside a Bellman backup with an explicit
> discount γ.** That gives multi-turn credit assignment (the original motivation) while
> keeping returns bounded, eliminating double-counting, and restoring PUCT's magnitude
> contract.

Concretely, declare the per-step reward `r_t = π(e_t)` (bounded by `max|π|`), declare
`v_leaf` a terminal value, and define the return on a rollout as

```
G(s, a) = r_t + γ · G(s_{t+1}, a_{t+1})         γ ∈ [0, 1)
        = Σ_{k≥0} γ^k · π(e_{t+1+k})  +  γ^T · v_leaf
```

Now `|G| ≤ max|π| / (1 − γ) + |v_leaf|`, finite and tunable via γ. PUCT's `c_p` can be
recalibrated against this known range exactly once.

### Tradeoffs vs the current local-only winner

- **Cost**: a Bellman-shaped Q is a real algorithmic change (a new MCTS subclass with a
  modified `search()`), versus the current 1-line additive local shaping.
- **Risk**: the +8.5pp result is *under a GPT-3.5 judge*, not measured donation. The
  hypothesis "multi-turn shaping should beat local shaping for actual persuasion" can only
  be tested against a behavioral metric (simulated `U_Donate` rate). If you stay on the
  judge metric, you are likely re-measuring "judge alignment," and local + conservative π
  may continue to win because the judge is local too.
- **Recommendation**: pair this with the behavioral evaluator the algorithm doc already
  flags as missing. Otherwise this experiment is dominated by the cheaper local variant.

### Code snippet — drop-in subclass

Add to `src/mcts/emotion_mcts.py`. Inherits from `EmotionAwareOpenLoopMCTS` so it shares
the realization + emotion-history machinery; only overrides `search()` and the penalty
table.

```python
class EmotionBellmanShapingMCTS(EmotionAwareOpenLoopMCTS):
    """Bellman-shaped emotion MCTS: π(e) is treated as a per-step reward and a real
    discount γ is applied, so returns stay bounded and credit assignment is per-edge.

    Replaces the local-only shaping in EmotionAwareDiscountQOpenLoopMCTS. See
    EMOMCTS_RESEARCH_DIRECTIONS.md "Investigation 1" for the rationale and
    EMOMCTS_ALGORITHM.md "Why we can't recursively return the discounted value" for the
    five failure modes this fixes.
    """

    def __init__(self, game, player, configs, emotion_classifier, gamma: float = 0.8) -> None:
        super().__init__(game, player, configs, emotion_classifier)
        # γ ∈ [0, 1). γ → 0 collapses to local-only shaping (current behaviour);
        # γ → 1 distributes emotion credit far up the tree. 0.7–0.9 is the standard band.
        self.gamma = gamma

    def _get_emotion_penalty(self, emotion: Emotions) -> float:
        penalties = {
            Emotions.Anger: -1.0, Emotions.Fear: -0.9, Emotions.Disgust: -0.8,
            Emotions.Contempt: -0.6, Emotions.Sadness: -0.4,
            Emotions.Surprise: 0.0, Emotions.Neutral: 0.0, Emotions.Happiness: 0.0,
        }
        return penalties.get(emotion, 0.0)

    def search(self, state: EmotionAwareDialogSession):
        hashable_state = self._to_string_rep(state)

        terminated_v = self.game.get_dialog_ended(state)
        if terminated_v == 1.0:
            return terminated_v

        if hashable_state not in self.P:
            v = self._init_node(state)
            return v
        else:
            self._add_new_realizations(state)

        best_uct, best_action = -float("inf"), -1
        for a in self.valid_moves[hashable_state]:
            uct = self._calculate_uct(hashable_state, a)
            if uct > best_uct:
                best_uct, best_action = uct, a

        state = self._sample_realization(hashable_state)
        next_state = self._get_next_state(state, best_action)

        # Bellman return: r_t + γ · G_{t+1}. r_t is the distribution-weighted emotion
        # penalty at the *child* state; G_{t+1} is the recursive Bellman return from there.
        dist = next_state.predicted_distribution() or {}
        r_t = sum(p * self._get_emotion_penalty(e) for e, p in dist.items())
        g_child = self.search(next_state)
        g_t = r_t + self.gamma * g_child   # bounded: |g_t| ≤ max|π|/(1-γ) + |v_leaf|

        # Q update on the bounded Bellman return. PUCT's c_p must be recalibrated for the
        # new Q range — quick estimate: max|π|/(1-γ) ≈ 1.0/0.2 = 5 for γ=0.8, so c_p
        # should also scale by ~5 to keep the explore term competitive.
        Q, Nsa = self.Q[hashable_state][best_action], self.Nsa[hashable_state][best_action]
        self.Q[hashable_state][best_action] = (Nsa * Q + g_t) / (Nsa + 1)
        self.Ns[hashable_state] += 1
        self.Nsa[hashable_state][best_action] += 1

        self._update_realizations_Vs(next_state, g_t)
        return g_t   # propagate the Bellman return, not the unshaped leaf v
```

### A/B plan

1. Sweep `γ ∈ {0.0, 0.3, 0.6, 0.8, 0.95}`. `γ = 0` should reproduce the local-only
   baseline ± noise (sanity check).
2. Rescale `cpuct` linearly with `1 / (1 - γ)` so the explore term stays comparable to Q.
3. Evaluate **under both** the GPT-3.5 judge and a behavioral donation-rate metric. The
   value of multi-turn shaping should show up in the behavioral metric first.

---

## Investigation 2 — Trajectory-based emotion reward (Δ shaping)

### Why this is theoretically the right move

Static per-emotion penalties have a known structural problem (`EMOMCTS_ANALYSIS.md §4`):
the most persuasive DAs (`proposition of donation`, `emotion appeal`,
`foot in the door`) are exactly the ones that elicit Sadness / Fear in the persuadee
(guilt, sympathy, urgency). A snapshot penalty therefore biases the search *away from*
persuasion. Persuasion theory (Cialdini; foot-in-the-door literature) says the value
sits in the *delta* — moving the user from Anger → Sadness, or Neutral → Surprise →
Sadness → commitment, is the success pattern, not the absolute emotional state at any
single turn.

`EMOMCTS_ALGORITHM.md Proposal 5` already sketches this as a leaf-level shaping reward.
The reason "leaf-level" matters (instead of per-edge during backup): it sidesteps the
five Bellman failures from Investigation 1. The trajectory bonus is a single bounded
number, computed once at the leaf from the history, then ridden up through GDPZero's
standard backup unchanged.

### Design — a valence-weighted Δ over consecutive user turns

For each persuadee turn `t`, define a scalar valence `ν(e_t) ∈ [-1, +1]`. The trajectory
bonus is the mean per-step Δ:

```
v_arc(h^tr) = (1 / (T - 1)) · Σ_{t=2..T} ( ν(e_t) − ν(e_{t-1}) )
```

This rewards *de-escalation* (Anger → Sadness → Neutral has positive Δ at each step) and
*engagement spikes* (Neutral → Surprise → Sadness), and penalizes the reverse
(Neutral → Anger). It collapses to 0 on an emotionally flat dialog — which is the
intended behaviour: a dialog with no emotional arc gets no shaping signal, just the raw
`v_leaf`.

Bonus is bounded by `2 · max|ν| = 2.0`, regardless of dialog length — exactly the
property that fixes the depth-as-multiplier failure mode.

### Code snippet — add to `EmotionAwareOpenLoopMCTS`

Adds one helper + a leaf blend in `_init_node`. Both safe extensions, no Bellman rewrite.

```python
# Class constant: valence in [-1, +1]. Hostile / failure = negative; engagement = mild
# negative (Sad/Fear are *good* in persuasion when reached *from* a worse state); positive
# affect = positive. These are starting weights; learn them with the same script that
# learns π (Investigation 3) once that works.
EMOTION_VALENCE = {
    Emotions.Happiness: +1.0,
    Emotions.Surprise:  +0.3,
    Emotions.Neutral:    0.0,
    Emotions.Sadness:   -0.3,
    Emotions.Fear:      -0.5,
    Emotions.Disgust:   -0.7,
    Emotions.Contempt:  -0.8,
    Emotions.Anger:     -1.0,
}

def _trajectory_bonus(self, state: EmotionAwareDialogSession) -> float:
    """Mean per-step Δ valence across persuadee turns.

    Bounded in [-2, +2]; 0 on dialogs with <2 user turns or no valence change.
    Computed from the emotions already attached to ``state.history`` — no classifier
    call.
    """
    user_valences = [
        self.EMOTION_VALENCE.get(rec.emotion, 0.0)
        for rec in state.history if rec.role == state.USR
    ]
    if len(user_valences) < 2:
        return 0.0
    deltas = [user_valences[i] - user_valences[i - 1] for i in range(1, len(user_valences))]
    return sum(deltas) / len(deltas)
```

Wire it into the leaf evaluation in `_init_node`:

```python
def _init_node(self, state: EmotionAwareDialogSession):
    hashable_state: str = self._to_string_rep(state)
    allowed_actions = self.player.get_valid_moves(state)
    self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]
    self.Ns[hashable_state] = 0
    self.Nsa[hashable_state] = {a: 0 for a in self.valid_moves[hashable_state]}
    self.Q[hashable_state] = {a: self.configs.Q_0 for a in self.valid_moves[hashable_state]}
    self.realizations[hashable_state] = [state.copy()]

    prior, v = self.player.predict(state)
    # Leaf-level shaping: convex blend with the trajectory bonus. γ_arc small (0.1–0.3)
    # keeps v_leaf dominant; the bonus is a tiebreaker that rewards good emotional arcs.
    gamma_arc = getattr(self.configs, "gamma_arc", 0.2)
    v_arc = self._trajectory_bonus(state)
    v_blended = (1 - gamma_arc) * v + gamma_arc * v_arc

    self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v_blended
    self.P[hashable_state] = prior * allowed_actions
    if np.sum(self.P[hashable_state]) == 0:
        self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
    else:
        self.P[hashable_state] /= np.sum(self.P[hashable_state])
    return v_blended
```

CLI knob to add in `runners/emomcts.py`:

```python
parser.add_argument("--gamma_arc", type=float, default=0.2,
    help="weight of the trajectory-bonus blend at the leaf. 0 = no shaping; "
         "0.2 is a safe starting point.")
# ... in the args dict:
"gamma_arc": cmd_args.gamma_arc,
```

### Why this composes cleanly with the local-only π winner

- Local-only π already provides the *immediate* per-edge shaping that the GPT-3.5 judge
  rewards (the empirical +8.5pp).
- `v_arc` adds an *orthogonal* multi-turn shaping signal at the leaf. Their
  contributions are independent: π fires on the immediate user reaction to the chosen DA;
  `v_arc` fires on the longer-horizon emotional path.
- Both are bounded; neither breaks PUCT.

A/B: turn on `--gamma_arc 0.2` on top of the current winning configuration and re-run.
If win rate doesn't move (no behavioral evaluator yet), check whether the *DA mix*
shifts toward propositions / emotion appeals — that would confirm the bonus is biasing
toward arc-positive strategies even if the judge can't see the difference.

---

## Investigation 3 — Learned `π(e)` from p4g donation outcomes

See `scripts/learn_emotion_penalty_p4g.py`. Summary:

- Loads `data/p4g/300_dialog_turn_based.pkl` (300 annotated dialogs).
- Per dialog, derives a binary donation outcome from the persuadee's labels
  (`agree-donation` / `confirm-donation` / `provide-donation-amount` → 1).
- Per dialog, runs an emotion classifier (HF by default — deterministic, zero LLM cost)
  on every persuadee utterance and builds an emotion-fraction feature vector.
- Train/val split (80/20 by dialog).
- Fits L2-regularized logistic regression `P(donate | emotion fractions)`. Falls back to
  a hand-rolled IRLS loop if sklearn isn't installed (no new repo dependency required).
- Rescales learned coefficients so `max |π| == --scale` (default 0.3) to keep π
  comparable to `v ∈ [0, 1]` — the magnitude band the local-only fix was tuned in.
- Dumps `outputs/learned_pi.json`: per-emotion penalty, raw coefficients, val accuracy,
  majority-class baseline.

Run:

```bash
python scripts/learn_emotion_penalty_p4g.py
# or
python scripts/learn_emotion_penalty_p4g.py --classifier hf --scale 0.3 \
    --out outputs/learned_pi.json
```

### Plugging the learned π into the MCTS

Replace `_get_emotion_penalty` in `EmotionAwareDiscountQOpenLoopMCTS`:

```python
def __init__(self, game, player, configs, emotion_classifier,
             learned_pi_path: str | None = None) -> None:
    super().__init__(game, player, configs, emotion_classifier)
    self._learned_pi = {}
    if learned_pi_path:
        import json, os
        with open(learned_pi_path) as f:
            blob = json.load(f)
        # store with string keys so the lookup works against either Emotions enum
        # values (which are lowercase strings via StrEnum) or raw strings.
        self._learned_pi = {k: float(v) for k, v in blob["pi"].items()}

def _get_emotion_penalty(self, emotion: Emotions) -> float:
    if self._learned_pi:
        return self._learned_pi.get(str(emotion), 0.0)
    # fall back to the hand-tuned table
    return {
        Emotions.Anger: -1.0, Emotions.Fear: -0.9, Emotions.Disgust: -0.8,
        Emotions.Contempt: -0.6, Emotions.Sadness: -0.4,
        Emotions.Surprise: 0.0, Emotions.Neutral: 0.0, Emotions.Happiness: 0.0,
    }.get(emotion, 0.0)
```

### Caveats to report alongside the numbers

- Logistic regression on dialog-level emotion *fractions* loses turn-order information
  (it can't distinguish "user was Sad → then donated" from "user was Sad → then
  refused"). If the val accuracy is barely above the majority baseline, that's a sign the
  fraction featurisation is too lossy and the trajectory-shaping reward (Investigation 2)
  is the more honest data-driven signal.
- The HF classifier never predicts Contempt (no label in
  `j-hartmann/emotion-english-distilroberta-base`). Coefficient for Contempt will be
  zero / undefined; either drop it from the table or run with `--classifier llm`.
- Class imbalance: most p4g dialogs end in donation. Sanity-check `val_acc` against the
  reported majority-baseline; a model that always predicts "donate" can hit ~70% and
  learn nothing useful.

---

## Investigation 4 — Entropy-as-exploration bonus in PUCT

### Motivation

When the cached user-emotion distribution is high-entropy (the classifier is unsure what
the user feels — mixed cues, near-tie at the top), the underlying user state is genuinely
ambiguous. MCTS should spend more rollouts at that node, because the realisation we
have is a noisy sample from a wide distribution and `Q(s, a)` estimates are
correspondingly noisier. When entropy is low (one clear emotion dominates), the cached
realisation is representative and the current `(Q, P, N)` triple is reliable — exploit.

The runner already computes and stores the full distribution on every
`EmotionalHistoryRecord.distribution`, so this is free at lookup time.

### Design

Add a third PUCT term, scaled by the *normalised* entropy of the user emotion
distribution at the realisation's last turn:

```
PUCT'(s, a) = Q(s, a)
            + c_p   · P(a|s) · √(Σ_a N(s,a)) / (1 + N(s, a))
            + c_h   · H_norm(p_e) · √(Σ_a N(s,a)) / (1 + N(s, a))
```

with `H_norm(p_e) = H(p_e) / log(|E|) ∈ [0, 1]`. `c_h` starts at the same order as
`c_p` (≈ 1.0). The bonus has the same `√N / (1 + N(s,a))` shape as standard exploration,
so it doesn't compete with `Q` once a branch is well-visited — it just shifts *early*
exploration toward emotionally ambiguous nodes.

This is a strict generalisation of vanilla PUCT (`c_h = 0` recovers it) and orthogonal
to the value-shaping interventions, so it can be combined with either Investigation 1 or
2 cleanly.

### Code snippet — override `_calculate_uct`

```python
import math

def _calculate_uct(self, hashable_state: str, action: int) -> float:
    Ns = self.Ns[hashable_state] or 1e-8
    explore = math.sqrt(Ns) / (1 + self.Nsa[hashable_state][action])
    uct = (
        self.Q[hashable_state][action]
        + self.configs.cpuct * self.P[hashable_state][action] * explore
    )

    # entropy bonus: high uncertainty in the user emotion → explore this node more.
    # Reads from the cached realisation's distribution — no extra classifier call.
    realisations = self.realizations.get(hashable_state, [])
    if realisations:
        dist = realisations[0].predicted_distribution()   # any realization works; same hashable_state
        if dist:
            probs = [p for p in dist.values() if p > 0]
            if probs:
                ent = -sum(p * math.log(p) for p in probs)
                ent_norm = ent / math.log(len(dist))      # ∈ [0, 1]
                c_h = getattr(self.configs, "c_entropy", 1.0)
                uct += c_h * ent_norm * explore
    return uct
```

CLI knob:

```python
parser.add_argument("--c_entropy", type=float, default=1.0,
    help="weight on the user-emotion-entropy PUCT bonus. 0 = vanilla PUCT.")
# in the args dict:
"c_entropy": cmd_args.c_entropy,
```

### What to A/B

1. **Ablation grid** `c_entropy ∈ {0, 0.5, 1.0, 2.0}` against the current winning
   configuration. Hypothesis: small-to-moderate `c_entropy` improves win rate by
   spending search budget on nodes where it matters; large values hurt by over-exploring
   noisy branches.
2. **Sanity log**: print mean entropy per turn alongside the aggregated emotion
   distribution the runner already emits. If entropy is consistently ~0 (classifier is
   always confident), this term will be near-zero too and won't help — that's a sign you
   need the LLM classifier rather than HF (the LLM, with sampling, produces softer
   distributions; the HF encoder is often near-deterministic on persuasion-domain text).
3. **Compose with Investigation 1 or 2**: this is the only one of the four that doesn't
   touch `Q`, so it can be layered on top of either value-shaping change without
   confounds.

---

## Composition matrix

Which of these four can be combined without confounding each other?

|                              | Bellman backup (1) | Trajectory leaf reward (2) | Learned π (3) | Entropy PUCT (4) |
| ---------------------------- | ------------------ | -------------------------- | ------------- | ---------------- |
| **Bellman backup (1)**       | —                  | overlaps (both shape v)    | composes      | composes         |
| **Trajectory leaf reward (2)** |                  | —                          | composes      | composes         |
| **Learned π (3)**            |                    |                            | —             | composes         |
| **Entropy PUCT (4)**         |                    |                            |               | —                |

- (1) + (2): both inject shaping into `v`. Run separately so each ablation is
  interpretable; only combine after each has a clean baseline.
- (1) + (3): natural pair — Bellman backup + a data-driven `π` is the "right" reward
  shaping story for a paper.
- (2) + (3): if you learn the *valence weights* in (2) from the same regression as (3),
  this is the cleanest single-experiment combination.
- (4): independent of all three; layer on whichever baseline wins.

---

# Top-3 highest-ROI items — concrete snippets

These three items were flagged as "where I'd start" earlier in this document. For each:
what to change, the snippet (drop-in, *not yet applied* to the codebase), and any
companion training script + how to run it.

## Item 1 — Wire the missing λ blend (and notes on "propagate up the chain")

### What

`EMOMCTS_ALGORITHM.md §Empirical results` flags two open items in the *current winning
configuration* (local-only, additive `v + π(e)`, 58.55% vs ~50%):

1. The `--lambda_emo` CLI flag is declared but not actually consumed by
   `EmotionAwareDiscountQOpenLoopMCTS`. Wiring it lets you test convex
   `v~ = (1−λ)·v + λ·Σp·π(e)` (bounded in `[−1, +1]`) against the unbounded additive
   form. The hypothesis the doc explicitly calls out as "not yet measured."
2. "Propagate emotion penalty up the backup chain" — this is the half that the analysis
   doc (`EMOMCTS_ANALYSIS.md §1` and `EMOMCTS_ALGORITHM.md §Why we can't recursively
   return the discounted value`) shows is broken in the *naive* form (additive,
   unbounded, ancestor double-counting). The salvageable variant is **Bellman-shaped
   backup with a discount γ**, fully written out in **Investigation 1** above. Do not
   ship a naive recursive return.

So the actual 3-line change here is the λ wiring; the "propagate" half is Investigation 1
and should be evaluated separately.

### Code snippet — drop-in patch for `EmotionAwareDiscountQOpenLoopMCTS`

```python
class EmotionAwareDiscountQOpenLoopMCTS(EmotionAwareOpenLoopMCTS):
    def __init__(self, game, player, configs, emotion_classifier) -> None:
        super().__init__(game, player, configs, emotion_classifier)
        # ▼ NEW — read λ from configs, default 0 = current additive behaviour.
        # λ ∈ (0, 1] enables convex blending: v~ = (1-λ)·v + λ·Σp·π(e), bounded in [-1, +1].
        # λ = 0 collapses to additive (v + Σp·π(e)), the current 58.55% winner.
        self.lambda_emo = float(getattr(configs, "lambda_emo", 0.0))

    # ... _get_emotion_penalty, search() unchanged except for the blending line:

    def search(self, state):
        # [...identical to the current implementation until the blend...]
        dist = next_state.predicted_distribution()
        emotion_penalty = sum(p * self._get_emotion_penalty(e) for e, p in dist.items())

        # ▼ NEW — convex blend if λ>0, else additive (current behaviour preserved).
        if self.lambda_emo > 0.0:
            blended_v = (1.0 - self.lambda_emo) * v + self.lambda_emo * emotion_penalty
        else:
            blended_v = v + emotion_penalty
        # [...rest of search() unchanged: Q update, realizations_Vs update, return v...]
```

Runner-side wiring (`src/runners/emomcts.py`): re-add the dropped `lambda_emo` to the
`args` dict so it reaches the MCTS class:

```python
args = dotdict({
    "cpuct": 1.0,
    "num_MCTS_sims": cmd_args.num_mcts_sims,
    "Q_0": cmd_args.Q_0,
    "max_realizations": cmd_args.max_realizations,
    "lambda_emo": cmd_args.lambda_emo,   # ▼ NEW — was removed in a recent revision.
})
```

The CLI argument `--lambda_emo` is already declared at the bottom of the runner; no
parser change needed.

### A/B plan

| Run | λ    | Blend form                                        |
| --- | ---- | ------------------------------------------------- |
| A   | 0.0  | additive `v + Σp·π(e)` (current 58.55% baseline)  |
| B   | 0.3  | convex                                            |
| C   | 0.5  | convex                                            |
| D   | 0.7  | convex                                            |
| E   | 1.0  | pure penalty (no `v`)                             |

Hypothesis from `EMOMCTS_ALGORITHM.md`: "see whether the bounded convex blend can match
the unbounded additive form, or whether the larger penalty magnitude is load-bearing."

If any λ > 0 matches A, ship the convex form — it has a defensible bounded-Q story for
the paper. If all λ > 0 underperform, the *magnitude* of the additive penalty (which
pushes Q into `[−2, +1]`) is doing real work, and that itself is a paper-worthy finding.

---

## Item 2 — Anticipatory `P(emotion | state, sys_DA)`

### What

A small classifier that predicts the persuadee's emotion distribution *before* the
rollout runs the user-LLM, given only the dialog state and the system DA the planner is
about to play. Two payoffs:

1. **Cheap PUCT-time bonus / prior** — re-rank candidate actions by their expected
   emotional impact before paying a single LLM rollout per action.
2. **Eliminates most in-rollout classifier traffic** — once the predictor is calibrated
   well enough on the HF-labelled training data, the MCTS rollout can skip the per-turn
   emotion classifier call and use the predicted distribution directly. The classifier
   itself only needs to run once per real conversation turn (replayed prefix).

### Training script

`scripts/train_anticipatory_emotion_p4g.py` (already written).

```
python scripts/train_anticipatory_emotion_p4g.py
# or with the LLM classifier for labels:
python scripts/train_anticipatory_emotion_p4g.py --classifier llm
```

- Loads `data/p4g/300_dialog_turn_based.pkl`.
- Caches per-utterance emotion labels in
  `outputs/anticipatory_pemo_emocache.json` (so re-runs are fast).
- Builds features per `(sys_turn_t, user_turn_t)` pair: one-hot of sys DA + last user DA
  + last user emotion + normalised turn index + bag-of-sys-DAs over the prefix.
- Fits multinomial logistic regression (sklearn) on (features → next user emotion).
- Reports train / val accuracy and majority-class baseline.
- Saves `outputs/anticipatory_pemo.pkl` and a human-readable `.json` companion.

Caveats are the same shape as the `learn_emotion_penalty_p4g.py` script — featurisation
is the weakest link, escalate to a transformer encoder over the prefix if val accuracy
is too close to baseline.

### Code snippet — MCTS integration (drop-in, separate subclass)

```python
class AnticipatoryEmotionMCTS(EmotionAwareDiscountQOpenLoopMCTS):
    """Uses a trained P(emotion | state, sys_DA) to:
       (a) bias the PUCT prior toward DAs that produce favourable user emotion, and
       (b) skip the per-rollout emotion classifier call by reading the predicted
           distribution directly.
    """

    def __init__(self, game, player, configs, emotion_classifier,
                 anticipatory_model_path: str, prior_weight: float = 0.5) -> None:
        super().__init__(game, player, configs, emotion_classifier)
        import pickle
        with open(anticipatory_model_path, "rb") as f:
            self._pemo = pickle.load(f)   # {coef, intercept, classes, sys_da_vocab, ...}
        self.prior_weight = prior_weight

    # --- feature builder (mirrors the training script's featurize()) ----------------
    def _featurize(self, state, sys_da_at_t: str) -> np.ndarray:
        from collections import Counter
        sys_das = [r.da for r in state.history if r.role == state.SYS]
        last_user_da = next((r.da for r in reversed(state.history) if r.role == state.USR), None)
        last_user_emo = next((r.emotion for r in reversed(state.history) if r.role == state.USR), None)
        prefix_counts = Counter(sys_das)
        feats = []
        feats += [1.0 if sys_da_at_t == d else 0.0 for d in self._pemo["sys_da_vocab"]]
        feats += [1.0 if last_user_da == d  else 0.0 for d in self._pemo["usr_da_vocab"]]
        feats += [1.0 if last_user_emo == e else 0.0 for e in self._pemo["emotion_vocab"]]
        feats += [len(sys_das) / max(self.game.max_conv_turns, 1)]
        denom = max(len(sys_das), 1)
        feats += [prefix_counts.get(d, 0) / denom for d in self._pemo["sys_da_vocab"]]
        return np.array(feats)

    def _predict_emotion_dist(self, state, sys_da_at_t: str) -> dict:
        x = self._featurize(state, sys_da_at_t)
        logits = self._pemo["coef"] @ x + self._pemo["intercept"]
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return dict(zip(self._pemo["classes"], p.tolist()))

    # --- (a) bias the prior in _init_node -------------------------------------------
    def _init_node(self, state):
        v_or_blended = super()._init_node(state)
        hashable_state = self._to_string_rep(state)
        # for each valid action, weight the prior by the expected emotion penalty under
        # the anticipatory model. negative penalty = down-weight; positive = up-weight.
        bonus = np.zeros_like(self.P[hashable_state])
        for a in self.valid_moves[hashable_state]:
            sys_da = self.player.dialog_acts[a]
            dist = self._predict_emotion_dist(state, sys_da)
            bonus[a] = sum(p * self._get_emotion_penalty(e) for e, p in dist.items())
        # softmax-blend: P' ∝ P · exp(β · bonus). β = prior_weight.
        adjusted = self.P[hashable_state] * np.exp(self.prior_weight * bonus)
        if adjusted.sum() > 0:
            self.P[hashable_state] = adjusted / adjusted.sum()
        return v_or_blended

    # --- (b) skip the classifier call in search() ------------------------------------
    # In search(), replace
    #     dist = next_state.predicted_distribution()
    # with
    #     sys_da = self.player.dialog_acts[best_action]
    #     dist = self._predict_emotion_dist(state, sys_da)
    # so the emotion penalty uses the *predicted* distribution rather than running the
    # classifier on the rollout's freshly-generated user reply. Skip game.get_next_state
    # entirely *for the penalty term* (you still need it for state transitions; only the
    # classifier call inside it can be cached out).
```

### A/B plan

1. Train with `--classifier hf` (deterministic labels) and confirm val accuracy ≫
   majority baseline. If not, escalate features before plumbing into MCTS.
2. Plug into MCTS as `AnticipatoryEmotionMCTS` with `prior_weight ∈ {0, 0.5, 1.0, 2.0}`,
   keeping the in-rollout classifier active. This measures the *prior bias* in
   isolation.
3. Then enable the "skip classifier in rollout" branch and remeasure. Drop in win rate
   = your predictor's labels are too noisy to substitute. No drop = you can run MCTS at
   a fraction of the LLM cost.

---

## Item 3 — Persona-conditioned user simulator

### What

Today the user simulator is a single LLM with one task prompt. MCTS averages over its
unconditional sampling distribution — an *unknown* distribution over implicit user
priors baked into the backbone. Two consequences:

- The "average user" the planner optimises against is undefined and possibly drifts as
  the backbone changes.
- The cross-rollout variance MCTS averages over has no semantic structure — it's pure
  sampling noise instead of "this rollout was skeptical-user, that one was sympathetic."

Fix: define a **mixture of named personas** and sample one per rollout. Mixture weights
are reportable, the planner now optimises against a *known* mixture, and you can ablate
robustness per persona ("EMOMCTS wins +12pp on skeptical, +3pp on sympathetic").

### Persona file

`data/personas/p4g_personas.json` (already written). Six personas (skeptical,
sympathetic, busy, altruistic, financially_stressed, neutral_curious) with weights that
sum to 1.0. Edit, version-control, share with collaborators — these *are* the user
distribution the planner is being trained against.

### Calibration script (sanity check, not training)

`scripts/calibrate_personas_p4g.py` (already written).

```
python scripts/calibrate_personas_p4g.py --n_per_persona 5 --classifier hf
```

For each persona (plus a no-persona baseline): replays N real p4g dialog prefixes,
generates K user continuations under that persona, classifies the resulting emotions,
and dumps per-persona DA + emotion histograms + sample utterances. Read the summary to
confirm that "skeptical" actually shifts mass toward negative reactions vs baseline, etc.
Iterate on the prompt text until the histograms match the persona names; only then plumb
into MCTS.

### Code snippet — drop-in MCTS / game integration

The cleanest place to inject the persona is at the user-agent prompt construction site,
not the MCTS. The persona is sampled once per **rollout** (a single
`game.get_next_state` call chain), patched into the user agent's prompt, and reset after.

Add a thin wrapper around `PersuadeeChatModel`:

```python
class PersonaPersuadeeWrapper:
    """Wraps any PersuadeeChatModel and rotates the persona prompt per call.
    Drop-in replacement for the user agent — same interface (get_utterance,
    get_utterance_w_da)."""

    def __init__(self, base_user_agent, personas: list[dict], seed: int = 0):
        # personas: [{"id", "prompt", "weight"}, ...]
        self.base = base_user_agent
        self.personas = personas
        self.weights = np.array([p["weight"] for p in personas])
        self.weights /= self.weights.sum()
        self._rng = np.random.default_rng(seed)
        self._original_task_prompt = base_user_agent.task_prompt
        self._original_new_task_prompt = base_user_agent.new_task_prompt
        self.dialog_acts = base_user_agent.dialog_acts

    def _sample_persona(self):
        i = int(self._rng.choice(len(self.personas), p=self.weights))
        return self.personas[i]

    def _apply_persona(self, persona):
        self.base.task_prompt = self._original_task_prompt + "\n\nPersona: " + persona["prompt"]
        self.base.new_task_prompt = self._original_new_task_prompt + " " + persona["prompt"]

    def _reset(self):
        self.base.task_prompt = self._original_task_prompt
        self.base.new_task_prompt = self._original_new_task_prompt

    def get_utterance(self, *args, **kwargs):
        persona = self._sample_persona()
        self._apply_persona(persona)
        try:
            out = self.base.get_utterance(*args, **kwargs)
        finally:
            self._reset()
        return out

    def get_utterance_w_da(self, *args, **kwargs):
        persona = self._sample_persona()
        self._apply_persona(persona)
        try:
            out = self.base.get_utterance_w_da(*args, **kwargs)
        finally:
            self._reset()
        return out
```

Then in `src/runners/_common.py` (the agent factory) — *not modifying it right now,
just showing where the wiring goes*:

```python
# after building the base user agent for emo_p4g:
if cmd_args.use_personas:
    import json
    with open("data/personas/p4g_personas.json") as f:
        personas = json.load(f)["personas"]
    user = PersonaPersuadeeWrapper(user, personas, seed=cmd_args.seed)
```

And the CLI knob in the runner:

```python
parser.add_argument("--use_personas", action="store_true",
    help="rotate user simulator through data/personas/p4g_personas.json")
```

### A/B plan

1. **Calibration first** (run `calibrate_personas_p4g.py`); only proceed if the
   persona histograms diverge clearly from baseline. If "skeptical" produces the same DA
   mix as the baseline, the LLM is ignoring the persona text and the prompt needs
   rewording.
2. **Mixture run** vs **baseline run** — same MCTS, same backbone, with `--use_personas`
   on/off. Hypothesis: mixture reduces win-rate variance across reruns (because the user
   distribution is now reproducible) even if mean win-rate doesn't move much.
3. **Per-persona breakdown** — fix the persona at evaluation time (one persona for the
   whole run) to surface where MCTS wins / loses. This is the headline plot for "EMOMCTS
   generalises across user types."
4. **Future**: weights become a hyperparameter — fit them on real p4g dialog labels
   (treat each real dialog as a draw from one persona) to make the mixture match the
   true user distribution rather than an editorial guess.

---

# Speedup items — concrete snippets

## Item 4 — Top-K action expansion (free progressive widening)

### What

Today every node expansion allocates a `Q`/`Nsa` slot for *every* valid DA (13 for p4g)
and the PUCT loop iterates all of them on every visit. Most of those branches receive
near-zero prior mass and would never be selected anyway, but they still consume rollout
budget when PUCT's exploration term forces a "try untried action" sweep.

Pruning each node's action set to the top-K prior actions concentrates rollouts on the
branches the prior believes matter. Equivalent to free progressive widening with a fixed
K instead of an annealed schedule — implementation cost is one method override.

### Where the change goes

`valid_moves[hashable_state]` is populated exactly once per node in `_init_node`.
Everything else (UCT selection, `get_action_prob`, `Q`/`Nsa` allocation) iterates that
set. Filtering it once at expansion narrows the entire search transparently — no other
edits needed.

One ordering wrinkle: today `_init_node` allocates `Nsa`/`Q` before computing the prior.
To top-K by prior we have to compute `P` first, rank, then allocate.

### Code snippet — emotion-aware subclass (drop-in, no existing code modified)

Add to `src/mcts/emotion_mcts.py`. Subclasses `EmotionAwareDiscountQOpenLoopMCTS` so it
inherits the distribution-weighted penalty in `search()` and the local-only propagation
guarantee — only `_init_node` changes.

```python
class TopKEmotionAwareDiscountQOpenLoopMCTS(EmotionAwareDiscountQOpenLoopMCTS):
    """Drop-in replacement for EmotionAwareDiscountQOpenLoopMCTS that prunes each node's
    action set to the top-K highest-prior valid actions at expansion time.

    Behaviour identical to the parent when configs.top_k_actions in {0, None}; smaller
    Nsa/Q dicts and faster PUCT loops when top_k_actions > 0.

    See EMOMCTS_RESEARCH_DIRECTIONS.md "Item 4 — Top-K action expansion" for rationale
    and A/B plan.
    """

    def _init_node(self, state: EmotionAwareDialogSession):
        hashable_state: str = self._to_string_rep(state)
        allowed_actions = self.player.get_valid_moves(state)
        full_valid = allowed_actions.nonzero()[0]

        # 1) compute prior + value FIRST so we can rank actions before allocating Q/Nsa.
        # (Parent does Q/Nsa allocation first; we reorder.)
        prior, v = self.player.predict(state)
        P = prior * allowed_actions
        if P.sum() == 0:
            P = allowed_actions / allowed_actions.sum()
            logger.warning("zero prior; falling back to uniform over valid moves")
        else:
            P = P / P.sum()

        # 2) top-K filter — keep only the K highest-prior valid actions.
        # K=0 (or None / unset) preserves parent behaviour: keep all valid moves.
        K = int(getattr(self.configs, "top_k_actions", 0)) or len(full_valid)
        if K < len(full_valid):
            ranked = sorted(full_valid, key=lambda a: -P[a])
            kept = np.array(ranked[:K], dtype=full_valid.dtype)
            mask = np.zeros_like(P)
            mask[kept] = 1.0
            P = P * mask
            P = P / P.sum()                  # renormalize so PUCT's P term stays calibrated
        else:
            kept = full_valid

        # 3) allocate against the (possibly pruned) action set.
        self.valid_moves[hashable_state] = kept
        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = {a: 0 for a in kept}
        self.Q[hashable_state] = {a: self.configs.Q_0 for a in kept}
        self.realizations[hashable_state] = [state.copy()]
        self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v
        self.P[hashable_state] = P
        return v
```

### Wiring — CLI + runner

`src/runners/emomcts.py` — add the CLI flag and forward it in the `args` dotdict:

```python
parser.add_argument("--top_k_actions", type=int, default=0,
    help="if > 0, only the top-K prior actions per node are expanded. "
         "0 keeps all (current behaviour). 5-7 is a sensible band for p4g's 13 DAs.")
# in the args dict:
"top_k_actions": cmd_args.top_k_actions,
```

And swap the planner construction line for the new class:

```python
# was: dialog_planner = EmotionAwareDiscountQOpenLoopMCTS(game, planner, args, emotion_classifier)
from mcts.emotion_mcts import TopKEmotionAwareDiscountQOpenLoopMCTS
dialog_planner = TopKEmotionAwareDiscountQOpenLoopMCTS(game, planner, args, emotion_classifier)
```

With `--top_k_actions 0` (the default) the subclass is observationally identical to the
parent, so this swap is safe to land before tuning K.

### A/B plan

| K            | DAs explored | Expected effect                                   |
| ------------ | ------------ | ------------------------------------------------- |
| 0 (baseline) | 13           | Current behaviour                                 |
| 7            | top 7        | Mild pruning, ~25% fewer rollouts on cold branches |
| 5            | top 5        | Sweet spot per the original recommendation         |
| 3            | top 3        | Aggressive; only safe if the prior is well-calibrated |

Metrics to track:

- **DA histogram** — should shift toward the top-prior DAs (sanity check).
- **`mean(Ns / Nsa)` per visited action** — should *rise* with smaller K (search budget
  is concentrated on fewer branches → tighter `Q` estimates per action).
- **Win rate** — primary metric. Hypothesis: flat through K=5, drops noticeably at K=3.
- **Wall-clock per turn** — primary speedup metric. Expected ~`K / 13` reduction in
  PUCT-loop time + proportional reduction in rollout cost on early-search iterations
  (when MCTS would otherwise round-robin every action).

### Caveats / sharper variants

1. **K is decided once at first expansion** of each node. If the prior is wrong at the
   root, you've permanently locked out a potentially good action. Two mitigations if it
   bites you:
   - **Soft top-K** — with probability `1−ε` keep top-K, else sample any valid action.
     Trades a few wasted sims for the ability to "re-discover" pruned actions.
   - **Progressive widening proper** — start with K=1 (or 3), unlock the next-best prior
     action when `Ns(s) > c · (N_active(s))^α`. More principled, loses the
     one-method-override simplicity.

2. **`get_action_prob` will return zeros for dropped actions.** That's intended, but if
   any downstream code (eval scripts, DA-histogram dumps, the `emotions_count`
   snapshotting) expects a non-zero entry per DA, double-check it tolerates zeros. The
   current runner does — `np.argmax` and the histogram dump both handle it cleanly.

3. **Composes cleanly** with every other item in this doc:
   - With Item 1 (λ blend) — orthogonal; affects `Q` magnitude, not action-set size.
   - With Item 2 (anticipatory prior) — *synergistic*: a better prior makes top-K
     pruning safer (the actions you're cutting really are the bad ones).
   - With Item 3 (personas) — orthogonal; affects the rollout, not the action set.
   - With Investigation 1 (Bellman backup) — orthogonal.
   - With Investigation 4 (entropy PUCT) — orthogonal but worth noting that the entropy
     bonus only fires on retained actions, so the effective exploration boost is
     concentrated where it matters.

---

## Item 5 — Emotion-DA selection bonus (active opportunity exploitation)

### The structural problem this fixes

The current `π(e)` table is *purely negative* — Anger/Fear/Disgust/Contempt/Sadness get
penalty; Surprise/Neutral/Happiness get 0. In the empirical HF run, ~75% of classifier
calls land on Neutral or Happiness, so **the planner gets no shaping signal at all on
3 out of 4 turns**. The 25% with negative emotion get a "don't go there" nudge, but the
75% with neutral/positive emotion get no "this is your moment, push *here*" signal.

Persuasion is about *exploiting opportunities* as much as it is about avoiding mistakes.
A user who's just expressed happiness or surprise is in a window where a well-timed
proposition lands; a user who's expressed fear is in a window where a foot-in-the-door
de-risks; a user who's expressed disgust is in a window where credibility-appeal
rebuilds trust. The planner needs a signal that **fires every turn and tells it which
action *amplifies* persuasion given the user's current emotional state**.

This is the missing positive-signal counterpart to `π(e)`. It's also the cheap, immediate
form of the "first-order navigation coordinate" idea from the Phase plan below — a
hand-table that ships in a day, replaceable by a learned matrix later.

### The bonus matrix — hand-seeded from persuasion theory

Values in `[−0.4, +0.4]` so the bonus is on the same scale as π. Cells omitted = 0.0
(no bias). Symmetric across emotions: every turn, multiple actions get non-zero signal.

```python
EMOTION_DA_BONUS = {
    # User HOSTILE — don't push, de-escalate first
    Emotions.Anger: {
        "task related inquiry":     +0.30,    # change subject, lower stakes
        "credibility appeal":       +0.20,    # address probable trust concern
        "personal related inquiry": +0.10,
        "proposition of donation":  -0.30,    # pushing while angry = lose
        "foot in the door":         -0.20,
    },
    Emotions.Disgust: {                        # "this feels like a scam"
        "credibility appeal":       +0.40,    # exactly when credibility matters most
        "source related inquiry":   +0.20,
        "logical appeal":           +0.20,
        "emotion appeal":           -0.20,    # feels manipulative on top of disgust
        "proposition of donation":  -0.20,
    },
    Emotions.Contempt: {                       # dismissive variant of disgust
        "credibility appeal":       +0.30,
        "logical appeal":           +0.30,
        "emotion appeal":           -0.30,
    },
    Emotions.Fear: {                           # anxious about commitment
        "foot in the door":         +0.40,    # small ask reduces fear
        "emotion appeal":           +0.20,    # validate the concern
        "credibility appeal":       +0.10,
        "proposition of donation":  -0.20,    # too big an ask when scared
    },
    # User EMOTIONALLY ENGAGED — capitalize on the moment
    Emotions.Sadness: {                        # feeling for the cause
        "emotion appeal":           +0.40,    # amplify the empathy
        "personal story":           +0.30,    # deepen the connection
        "proposition of donation":  +0.20,    # engaged → close
        "logical appeal":           -0.10,    # logic misses the emotional moment
    },
    Emotions.Surprise: {                       # got their attention
        "credibility appeal":       +0.30,    # capitalize on the spike
        "task related inquiry":     +0.20,
        "logical appeal":           +0.20,
    },
    # User UNCOMMITTED — need to create engagement
    Emotions.Neutral: {
        "emotion appeal":           +0.20,    # introduce affect
        "personal story":           +0.20,
        "task related inquiry":     +0.20,
        "proposition of donation":  -0.10,    # too early
    },
    # User RECEPTIVE — time to close
    Emotions.Happiness: {
        "proposition of donation":  +0.40,    # warm window for the ask
        "foot in the door":         +0.30,
        "credibility appeal":       +0.10,    # reinforce
        "task related inquiry":     -0.10,    # don't dawdle when ready
    },
}
```

### Code snippet — drop-in subclass

Crucial design choice: **add the bonus to PUCT, not to Q**. Adding to Q would pollute
the value estimate with shaping signal and get averaged in noisily. Adding to PUCT acts
as a *selection bias* that decays naturally as `Nsa` grows — like the prior term — so
the bonus shapes *which actions get visited*, not *what their values are*. Q stays
clean and composes with the existing penalty.

```python
import math

class EmotionGuidedDiscountQOpenLoopMCTS(EmotionAwareDiscountQOpenLoopMCTS):
    """Adds an emotion-conditioned action bonus to PUCT: B(last_user_emotion, da).

    The bonus fires EVERY turn (not just on negative emotions like π does), so it gives
    the planner a positive steering signal across the 75%+ of turns where the user is
    neutral/happy. Bonus is a *selection bias* on PUCT, not a Q reward — Q stays clean.

    Composes additively with the existing π penalty (which lives in Q-update) and with
    Item 4's top-K pruning (high-bonus actions are more likely to survive the cut).
    """

    def __init__(self, game, player, configs, emotion_classifier) -> None:
        super().__init__(game, player, configs, emotion_classifier)
        # Strength of the emotion-DA bonus relative to PUCT's other terms. Starts at the
        # same magnitude as cpuct; sweep {0.5, 1.0, 2.0} once wired.
        self.c_emo_bonus = float(getattr(configs, "c_emo_bonus", 1.0))

    def _last_user_emotion(self, hashable_state):
        """Read the last persuadee emotion from any cached realization at this node.
        All realizations share the same SYS-DA prefix → any one is fine for grabbing
        the user's last emotional state."""
        realisations = self.realizations.get(hashable_state, [])
        if not realisations:
            return None
        history = realisations[0].history
        return next(
            (r.emotion for r in reversed(history) if r.role == realisations[0].USR),
            None,
        )

    def _calculate_uct(self, hashable_state: str, action: int) -> float:
        Ns = self.Ns[hashable_state] or 1e-8
        explore = math.sqrt(Ns) / (1 + self.Nsa[hashable_state][action])
        uct = (
            self.Q[hashable_state][action]
            + self.configs.cpuct * self.P[hashable_state][action] * explore
        )

        # Emotion-DA bonus — selection bias, fires every turn.
        last_emo = self._last_user_emotion(hashable_state)
        if last_emo is not None:
            da_str = self.player.dialog_acts[action]
            bonus = EMOTION_DA_BONUS.get(last_emo, {}).get(da_str, 0.0)
            # Same √N/(1+Nsa) shape as the prior term → decays as the branch gets
            # well-visited, so it's an *early-search* bias that gets out of the way
            # once Q has real data.
            uct += self.c_emo_bonus * bonus * explore

        return uct
```

### CLI wiring (`src/runners/emomcts.py`)

```python
parser.add_argument("--c_emo_bonus", type=float, default=1.0,
    help="weight on the emotion-DA bonus in PUCT. 0 = vanilla PUCT (no bonus). "
         "1.0 puts it on the same magnitude as cpuct's prior term.")
# in the args dict:
"c_emo_bonus": cmd_args.c_emo_bonus,
```

And swap the planner construction:

```python
from mcts.emotion_mcts import EmotionGuidedDiscountQOpenLoopMCTS
dialog_planner = EmotionGuidedDiscountQOpenLoopMCTS(game, planner, args, emotion_classifier)
```

With `--c_emo_bonus 0` the subclass is observationally identical to the parent
(sanity-check baseline before tuning).

### Why this is better than tuning π or λ

- **Fires every turn**: the 75% neutral/happy turns now have active steering instead of
  being a value-function-only zone. That's a 4× increase in shaping coverage.
- **Symmetric**: rewards good emotion-action matches, penalizes bad ones. Today's π
  only penalises.
- **Selection bias, not Q pollution**: the existing penalty already lives in Q; adding
  another signal there competes with `v`. PUCT bonus lets both signals coexist —
  penalty shapes the value estimate, bonus shapes which actions get visited at all.
- **Decays with visits**: once an action has been tried enough, Q takes over. The bonus
  is a "where to look first" heuristic, not a "this is always right" prescription, so
  miscalibrated cells self-correct from data.
- **Asymmetric magnitude is fine**: bonus +0.4 for the right move, penalty −1.0 for the
  worst emotion. You can correct for "the right thing at the right time" being a
  smaller perturbation than "the wrong thing at the worst time" without changing π at
  all.

### A/B plan

1. **`--c_emo_bonus 0` sanity run** — should match the current planner output exactly.
   Confirms the subclass plumbing is correct before tuning.
2. **`--c_emo_bonus 1.0`** — same magnitude as the prior term. Hypothesis: shifts DA
   distribution noticeably (proposition-of-donation up on happy turns, emotion-appeal up
   on sad turns, etc.) and improves win-rate by 3–6pp.
3. **`--c_emo_bonus 2.0`** — stronger bias. Tests whether the bonus is load-bearing or
   whether 1.0 was already at the ceiling.
4. **DA-by-emotion histogram dump** — the runner already emits `_da_emotions.json`.
   After the bonus is wired, the per-emotion DA mix should shift toward the high-bonus
   cells in the matrix. If it doesn't, the bonus magnitude is too small relative to the
   prior — increase `c_emo_bonus` or scale the matrix.

### Where the matrix should eventually come from — learned variant

Hand-seeded is a starting point. The data-driven version is mineable from the same p4g
pickle: for each `(persuadee_emotion_at_turn_t, persuader_DA_at_turn_t+1)` cell, compute
the lift in donation rate vs the marginal:

```python
# src/emotion_mining/mine_emotion_da_bonus_p4g.py  (sketch)
#
# For each dialog: label every persuadee emotion + record the persuader DA played in
# response. Count (emotion, da) -> donation outcome pairs across all dialogs.
#
# Donation lift per cell:
#     lift(e, da) = P(donate | (e, da) appears in dialog) − P(donate | (e, da) never appears)
#
# Positive lift = persuasion booster; negative lift = persuasion penalty.
# Normalise so max |lift| matches the hand-seeded scale (~0.4) and write out
# outputs/learned_emotion_da_bonus.json.

import json, pickle
from collections import defaultdict

cell_counts = defaultdict(lambda: {"with_donate": 0, "with_no_donate": 0,
                                   "without_donate": 0, "without_no_donate": 0})
# ... featurise + count ...
lift = {(e, da): compute_lift(cell_counts[(e, da)]) for (e, da) in cell_counts}
# rescale max|lift| -> 0.4
scale = 0.4 / max(abs(v) for v in lift.values())
out = {e: {da: round(scale * v, 3) for (e2, da), v in lift.items() if e2 == e}
       for e in {e for (e, _) in lift}}
with open("outputs/learned_emotion_da_bonus.json", "w") as f:
    json.dump(out, f, indent=2)
```

Then load in the subclass `__init__`:

```python
import json
if learned_path := getattr(configs, "learned_emotion_da_bonus_path", None):
    with open(learned_path) as f:
        self._bonus_matrix = json.load(f)   # replaces EMOTION_DA_BONUS at runtime
else:
    self._bonus_matrix = EMOTION_DA_BONUS
```

The learned matrix is a paper-worthy contribution on its own: "we mine an emotion-DA
persuasion lift table from p4g and show it consistently improves over hand-seeded
persuasion intuition."

### Composition with the rest of the doc

- **With Item 1 (λ blend)**: orthogonal — bonus lives in PUCT, penalty lives in Q.
  Both can be on at once with no confound.
- **With Item 2 (anticipatory `P(emotion | state, DA)`)**: *synergistic*. Item 2
  answers "given a DA, what emotion will the user have?"; Item 5 answers "given the
  user's emotion, which DA?". Use them together — anticipatory model produces a
  forward-looking emotion estimate, Item 5's bonus reads either the current or the
  anticipated emotion. Stack them as: `bonus_value = B(anticipated_emotion(s, a), a)`.
- **With Item 3 (personas)**: orthogonal — bonus matrix is invariant to persona, but
  if you find empirically that the *best* matrix differs per persona, fork it later.
- **With Item 4 (top-K)**: *synergistic* — high-bonus actions are more likely to
  survive the K-cut, so the bonus and top-K reinforce each other. Practical effect:
  top-K with the bonus enabled keeps emotion-appropriate DAs in the active set
  preferentially.
- **With Phase 1 (toned actions)**: extends naturally — the matrix becomes a 3-tensor
  `B(emotion, da, tone)` for the joint action space. Same code path, larger lookup.
- **With Self-play RL**: the bonus is the bootstrap; the learned policy `π_θ` replaces
  it once trained. Hand-seeded → learned matrix → learned policy is a clean three-step
  progression.

---

## Item 5b — Generalising the bonus matrix and learning it from data

Item 5 ships with a hardcoded p4g table. Two follow-ups together turn it from a
hand-seeded one-task hack into a reusable framework:

1. **Generalise to a `BonusMatrixProvider` abstraction** so the same MCTS class works
   on esc and cb with task-specific tables.
2. **Learn the table from data** so the matrix isn't bottlenecked by hand-tuning.

### Generalisation — `BonusMatrixProvider` plug-in

Each task has a structurally different emotion role
(`EMOMCTS_ALGORITHM.md "How the proposals scale to ESConv and CraigslistBargain"`):

| Task | Success signal | Emotion role | "Good DA for this emotion" means |
| --- | --- | --- | --- |
| **P4G** | `U_Donate` | **Instrumental** (drives behaviour) | DA that maximises P(donate) given current persuadee affect |
| **ESC** | `U_Solved` | **Terminal** (emotion *is* the goal) | DA that reduces patient distress on the next turn |
| **CB** | `U_Deal` | **Diagnostic** (adversary signal) | DA that exploits buyer's revealed state without giving leverage |

Shape of the matrix is identical; contents differ. The MCTS class should read from a
provider, not a hardcoded dict:

```python
class BonusMatrixProvider:
    """Source of (emotion, action) bonuses for EmotionGuidedDiscountQOpenLoopMCTS.
    Lets the MCTS class stay task-agnostic — different tasks plug in different
    providers; data-mined tables plug in via LearnedBonus(json_path)."""
    def bonus(self, emotion: Emotions, action_str: str) -> float:
        raise NotImplementedError


class HandSeededP4GBonus(BonusMatrixProvider):
    def __init__(self):
        self.table = EMOTION_DA_BONUS               # the existing p4g table
    def bonus(self, emotion, action_str):
        return self.table.get(emotion, {}).get(action_str, 0.0)


class HandSeededESCBonus(BonusMatrixProvider):
    """ESC: bonus = DA that historically reduces patient distress on the next turn.
    NOTE: static absolute-emotion mapping is suboptimal here (see Proposal 1 critique
    in EMOMCTS_ALGORITHM.md — emotion *is* the success signal, so trajectory matters).
    Use this only as a bootstrap; ship Phase 3b trajectory value for real ESC."""
    def __init__(self):
        self.table = {
            Emotions.Anger:   {"reflection of feelings": +0.4, "self-disclosure": -0.2},
            Emotions.Sadness: {"affirmation and reassurance": +0.4,
                               "providing suggestions": -0.1},
            # ... fill in from supportive-conversation literature
        }
    def bonus(self, emotion, action_str): ...


class HandSeededCBBonus(BonusMatrixProvider):
    """CB: target is Neutral (deal-closing zone). Anger = buyer rejecting; Happiness
    = seller over-conceded → both penalised. Non-monotone in valence — a quirk worth
    reporting."""
    def __init__(self):
        self.table = {
            Emotions.Anger:    {"propose": -0.4, "vague-price": +0.2},
            Emotions.Happiness:{"propose": -0.3, "intro": -0.2},   # buyer too happy = bad
            Emotions.Neutral:  {"propose": +0.3, "agree": +0.2},
        }
    def bonus(self, emotion, action_str): ...


class LearnedBonus(BonusMatrixProvider):
    """Loads a JSON dumped by src/emotion_mining/mine_emotion_da_bonus_*.py.
    Drop-in for any of the hand-seeded providers."""
    def __init__(self, path: str):
        import json
        with open(path) as f:
            self.table = json.load(f)
    def bonus(self, emotion, action_str):
        return float(self.table.get(str(emotion), {}).get(action_str, 0.0))
```

`EmotionGuidedDiscountQOpenLoopMCTS.__init__` accepts a provider, defaulting to the
hand-seeded p4g one so existing runs are unaffected:

```python
def __init__(self, game, player, configs, emotion_classifier,
             lambda_emo: float = 0.0, c_emo_bonus: float | None = None,
             bonus_provider: BonusMatrixProvider | None = None) -> None:
    super().__init__(game, player, configs, emotion_classifier, lambda_emo=lambda_emo)
    # ... c_emo_bonus handling unchanged ...
    self._bonus = bonus_provider or HandSeededP4GBonus()
```

And `_calculate_uct` reads from it:

```python
bonus = self._bonus.bonus(last_emo, self.player.dialog_acts[action])
```

Per-task wiring lives in `runners/_common.py` — each `TASKS[...]` entry gets a
`bonus_provider_cls` field. The runner constructs the right one and passes it through.
Same MCTS code, different table per task.

### Learning the matrix from data

Six approaches, ordered by sophistication. Recommend method 1 as the first ship, method
3 or 5 as the headline learned variant for a paper.

#### Method 1 — Per-cell donation lift (1 day, simplest)

For each `(emotion_at_t, da_at_t+1)` cell:

```
lift(e, da) = P(donate | cell appears in dialog) − P(donate | cell does not appear)
```

Apply Laplace smoothing and drop cells with < 10 observations. Drop straight into
`LearnedBonus` via the JSON path. **Pros**: cheap, interpretable, easy to validate
against the hand-seeded matrix cell-by-cell. **Cons**: dialog-level aggregation loses
turn-order; confounded by DAs that human persuaders use a lot (high marginal donation
rate produces spurious positive lift).

Script sketch:

```python
# src/emotion_mining/mine_emotion_da_bonus_p4g.py  (~100 lines)
import json, pickle
from collections import defaultdict
# 1. Load p4g pickle; label every persuadee turn with the emotion classifier (HF).
# 2. Per dialog: walk through (persuadee_turn_t.emotion, persuader_turn_t+1.first_da)
#    pairs; track set of (e, da) cells that appear.
# 3. Per cell, count: appears_donate, appears_no_donate, absent_donate, absent_no_donate.
# 4. Apply Laplace smoothing + min-observation threshold (drop cells with < 10).
# 5. Rescale so max|lift| matches hand-seeded scale (~0.4).
# 6. Write outputs/learned_emotion_da_bonus.json (drop-in for LearnedBonus).
```

#### Method 2 — Logistic regression with `(emotion, DA)` interaction terms (1-2 days)

Extend `learn_emotion_penalty_p4g.py` with interaction features `i_e_d = 1 if
emotion==e AND next_da==d else 0`. The signed coefficient on `i_e_d` is the bonus.
**Pros**: L2 regularisation handles confounding more gracefully than raw lift.
**Cons**: still loses sequence order; requires careful interaction encoding.

#### Method 3 — Imitation lift from successful dialogs (paper-worthy, 1-2 days)

```
B(e, da) = P_human(da | last_user_emotion=e, dialog ended in donate)
         − P_human(da | last_user_emotion=e, dialog did NOT end in donate)
```

Compare what successful vs failed persuaders chose when the user was in the same
emotional state. Sidesteps explicit outcome estimation; uses *expert behaviour
contrast*. **Pros**: avoids confounding via the success/fail conditioning; no MDP
modelling. **Cons**: still needs enough samples per `(emotion, success/fail)` cell.

This is the cleanest single-step learned-matrix contribution for a paper.

#### Method 4 — Offline Q-learning from logged dialogs (medium, 3-5 days)

Treat each turn as `(s, a, r)`; fit `Q(s, a)` by Bellman backup over logged
trajectories. Bonus = marginal of Q:

```
B(e, da) = E_{s | last_emotion=e}[Q(s, da)] − E_a[E_{s | last_emotion=e}[Q(s, a)]]
```

**Pros**: accounts for downstream consequences (foot-in-the-door at turn 3 → donation
at turn 8). **Cons**: offline RL with small samples is unstable; needs conservative
estimation (CQL-style).

#### Method 5 — Imitation classifier on successful-dialog DAs (paper-worthy, 2-3 days)

Train `P_θ(da | last_user_emotion, state_features)` on successful dialogs only.
Argmax = highest-bonus DA per emotion; softmax output is a soft bonus.

This is the **anticipatory model from Item 2 inverted** — Item 2 predicts emotion given
DA; this predicts DA given emotion. Same training infrastructure, different head. If
the anticipatory model is already trained, this adds ~30 minutes of code.

#### Method 6 — Doubly-robust off-policy estimation (gold standard, 1+ week)

Disentangles logging policy (persuader's natural DA propensity) from conditional
outcome. **Pros**: highest claim-strength for "what would have happened if we'd chosen
differently." **Cons**: serious engineering. Only worth it if the data-driven matrix
is the paper's headline contribution.

### Recommended path

**This week**:
1. Refactor to `BonusMatrixProvider`. ~30 min, no behaviour change.
2. `src/emotion_mining/mine_emotion_da_bonus_p4g.py` with method 1 + Laplace smoothing. ~1 hour.
3. Run it; compare against the hand-seeded matrix cell-by-cell. Surprises in either
   direction are research signal — if learned says `(Happiness, foot-in-the-door)` is
   bad and the hand table says it's good, look at the data and figure out who's right.

**Next month, paper-worthy**:
4. Implement method 3 (imitation lift) as the headline learned-matrix variant.
5. Run the full stack (MCTS + Item 5 bonus + method-3 matrix) on esc and cb with
   their respective outcome signals. Headline plot: "one MCTS algorithm, three tasks,
   three learned matrices, each beats its baseline."

**Stretch**:
6. Method 5 if you want the bonus parameterised on full state (not just last emotion),
   trading interpretability for capacity.

### Composition with the rest of the doc

- **With Item 5 (hand-seeded bonus)** — this section is its data-driven and
  cross-task continuation; ship Item 5 first to validate the framework, then replace
  the matrix.
- **With Item 2 (anticipatory `P(emotion | state, DA)`)** — method 5 is literally the
  inverse direction of Item 2's model. If Item 2 ships, method 5 is a 30-minute add.
- **With Phase 1 (toned actions)** — once tone is in the action space, the matrix
  becomes 3-tensor `B(emotion, da, tone)`. Same mining script with a larger output
  table; the `BonusMatrixProvider.bonus()` signature gains a `tone` arg.
- **With Self-play RL** — the learned matrix is a stronger bootstrap than the
  hand-seeded one. Self-play replaces it with a learned policy after enough iterations.

---

# Phase plan — Emotion as a first-order navigation coordinate

Today emotion is a **passive penalty** applied after the fact: the planner picks DAs to
maximise donation likelihood, then `π(e)` quietly adjusts `Q` to discourage user
emotions we labelled bad. This section sketches the architectural shift that turns
emotion into a **first-order steering axis** — the planner explicitly navigates user
emotional state, and the system's affective delivery becomes a controllable lever
alongside its strategic content.

Three phases, ordered by engineering cost. Phase 1 is shippable alone (real
expressiveness gain); Phases 2 and 3 build on it.

## Phase 1 — Expand the action space from `DA` to `(DA, tone)`

### What

Today actions are 13 DAs. The persuasion *content* is in the DA; the *affective
delivery* is implicit in whatever the system-LLM happens to generate (and so is
uncontrolled, varying between rollouts as sampling noise). Splitting the action into a
`(DA, tone)` pair makes tone a controllable lever.

### Tone vocabulary

Start with seven affective styles known to matter in persuasion communication research
— iterate after seeing what the LLM actually produces under each:

```python
TONES = [
    "warm",         # personal, friendly, low pressure
    "urgent",       # time-sensitive, stakes-emphasising
    "factual",      # neutral, evidence-driven
    "empathetic",   # validating, acknowledging the persuadee's state
    "gentle",       # de-escalating, low-pressure
    "encouraging",  # affirming the persuadee's agency
    "confident",    # certain, authoritative
]
```

### Action encoding

```python
# action_id ∈ [0, len(SYS_DAS) * len(TONES)) = [0, 91)
def decode_action(action_id: int) -> tuple[str, str]:
    return SYS_DAS[action_id // len(TONES)], TONES[action_id % len(TONES)]

def encode_action(da: str, tone: str) -> int:
    return SYS_DAS.index(da) * len(TONES) + TONES.index(tone)
```

91 actions sounds large, but combined with **Item 4 (top-K)** the active set per node is
~7. The realization pool already absorbs open-loop variance — tone-induced variance just
folds into the same pool.

### Tone-conditioned utterance generation

The system agent's `get_utterance` accepts an action id today; rewrite it to also pass a
tone instruction into the prompt. Drop-in subclass — keeps the existing class as a
fallback:

```python
class TonedPersuaderModel(PersuaderChatModel):
    """PersuaderChatModel that accepts a (da, tone) joint action and injects the tone
    instruction into the system utterance prompt."""

    def get_utterance(self, state, action: int, mode: str = "train") -> str:
        da, tone = decode_action(action)
        # original prompt building, but with a tone directive appended.
        # The system-LLM is already instruction-tuned, so a single sentence suffices.
        tone_directive = (
            f"Speak with a {tone} tone. Use the dialogue act [{da}]. "
            f"Stay consistent with the persuadee's emotional state."
        )
        messages = self._build_messages(state)
        messages.append({"role": "system", "content": tone_directive})
        data = self.backbone_model.chat_generate(messages, **self.inference_args)
        return self.backbone_model._cleaned_chat_resp(data, messages)[0]
```

`get_valid_moves` widens to length `len(SYS_DAS) * len(TONES)`, with valid actions
exactly the Cartesian product of valid DAs × all tones:

```python
def get_valid_moves(self, state):
    base = super().get_valid_moves(state)        # length 13 mask over DAs
    mask = np.zeros(len(SYS_DAS) * len(TONES))
    for da_idx, ok in enumerate(base):
        if ok:
            for tone_idx in range(len(TONES)):
                mask[da_idx * len(TONES) + tone_idx] = 1
    return mask
```

### A/B

Compare current EmoMCTS vs `TonedPersuaderModel` + top-K=10 (so per-node action count
matches the current 13). Hypothesis: same compute budget, the toned variant wins on
both judge win-rate and persuasion realism (qualitative). If tone doesn't help, you've
ruled out a frequently-cited gap with one focused experiment.

---

## Phase 2 — Reactive tone prior `P(tone | last_user_emotion)`

### What

The planner shouldn't have to learn from scratch that an angry persuadee responds
better to gentle/empathetic tones than urgent/confident ones. Encode that as a **prior**
on the tone dimension, conditioned on the user's current emotion.

The joint prior factorises:

```
P(da, tone | state) = P(da | state) · P(tone | last_user_emotion)
```

`P(da | state)` is what `player.predict()` already returns. `P(tone | emotion)` is new
— hand-seeded to start, learned later when annotation budget allows.

### Hand-seeded tone prior

Calibrated against communication-research intuition: hostile users want de-escalation;
sad / fearful users want validation; neutral / curious users want substance; happy users
want momentum.

```python
TONE_PRIOR = {
    Emotions.Anger:    {"empathetic": 0.35, "gentle": 0.30, "factual": 0.15,
                        "warm": 0.10, "confident": 0.05, "encouraging": 0.03, "urgent": 0.02},
    Emotions.Disgust:  {"factual": 0.35, "confident": 0.25, "gentle": 0.15,
                        "empathetic": 0.10, "warm": 0.08, "encouraging": 0.04, "urgent": 0.03},
    Emotions.Contempt: {"factual": 0.35, "confident": 0.25, "gentle": 0.15,
                        "empathetic": 0.10, "warm": 0.08, "encouraging": 0.04, "urgent": 0.03},
    Emotions.Fear:     {"gentle": 0.30, "empathetic": 0.25, "warm": 0.20,
                        "factual": 0.15, "encouraging": 0.05, "confident": 0.03, "urgent": 0.02},
    Emotions.Sadness:  {"empathetic": 0.35, "warm": 0.25, "gentle": 0.20,
                        "encouraging": 0.10, "factual": 0.05, "confident": 0.03, "urgent": 0.02},
    Emotions.Surprise: {"encouraging": 0.25, "warm": 0.20, "factual": 0.20,
                        "confident": 0.15, "empathetic": 0.10, "gentle": 0.05, "urgent": 0.05},
    Emotions.Neutral:  {"confident": 0.25, "warm": 0.20, "factual": 0.20,
                        "encouraging": 0.15, "empathetic": 0.10, "gentle": 0.05, "urgent": 0.05},
    Emotions.Happiness:{"encouraging": 0.30, "warm": 0.25, "confident": 0.20,
                        "empathetic": 0.10, "factual": 0.10, "urgent": 0.03, "gentle": 0.02},
}
```

### Joint-prior helper

Drop into the same subclass that introduces the joint action space:

```python
def _joint_prior(self, P_da: np.ndarray, last_user_emotion: str) -> np.ndarray:
    """Factorise: P(da, tone | state) = P(da | state) · P(tone | emotion)."""
    tone_dict = TONE_PRIOR.get(last_user_emotion, {})
    P_tone = np.array([tone_dict.get(t, 1.0 / len(TONES)) for t in TONES])
    P_tone /= P_tone.sum()
    # outer product, then flatten row-major to match decode_action's indexing
    return (P_da[:, None] * P_tone[None, :]).flatten()
```

Wire it into `_init_node` *after* the existing `prior, v = player.predict(state)` call:

```python
def _init_node(self, state):
    allowed_actions = self.player.get_valid_moves(state)  # length 91 now
    full_valid = allowed_actions.nonzero()[0]

    # base DA prior from the planner LLM (length 13)
    P_da, v = self.player.predict(state)
    last_emo = next(
        (r.emotion for r in reversed(state.history) if r.role == state.USR),
        Emotions.Neutral,
    )
    P_joint = self._joint_prior(P_da, last_emo)            # length 91
    P_joint = P_joint * allowed_actions
    P_joint /= P_joint.sum()

    # ... top-K + Q/Nsa allocation unchanged (compose with Item 4 verbatim)
```

### Later: learn the table

Same featurisation as `train_anticipatory_emotion_p4g.py`, but the label is the *tone*
of the human persuader's next turn. Tone labelling is the bottleneck — there's no
ready-made annotation. Two paths:

- **Annotate by classifier**: prompt the backbone LLM to label each persuader utterance
  with one of `TONES`; treat as silver labels. Same pattern as how persuadee emotion is
  labelled today.
- **Annotate by acoustic / lexical proxies**: word lists per tone (warm-words,
  urgent-words, factual-hedges) and label by best-fit. Cheap, lossy, useful for warm
  start.

Either way, the resulting table is a drop-in replacement for the hand-seeded one.

---

## Phase 3 — Emotion trajectory as the primary navigation reward

### What

This is the operational meaning of *first-order navigation coordinate*: emotion stops
being a side-channel penalty and becomes a substrate of the **value function** itself.
The planner navigates `(dialog × emotion)` space to reach states with high historical
donation probability.

Two implementations, in increasing order of ambition. 3a is a one-evening change; 3b is
a real model train.

### 3a — Distance-to-target-arc (cheap, immediate)

Mine the p4g training dialogs that end in donation, run the emotion classifier on every
persuadee turn, and you get a corpus of *successful emotion trajectories*. Bin them into
a canonical arc and use distance-from-arc as a dense shaping reward.

Sketch (separate script, dumps a target arc JSON):

```python
# scripts/mine_successful_arcs_p4g.py
import pickle, json
from collections import defaultdict

# 1. Load p4g + use the emotion classifier (HF) to label every persuadee turn.
# 2. Keep dialogs ending in agree-donation / confirm-donation / provide-donation-amount.
# 3. Bin each successful dialog's emotion sequence by relative position
#    (turn_idx / total_turns -> emotion). Aggregate to a per-position emotion histogram.
# 4. Output: {position_bin: {emotion: probability}}.
# Use this as a reference trajectory for distance-from-arc reward.

successful_arcs = defaultdict(lambda: defaultdict(int))
# ... featurization + binning ...
# Write outputs/p4g_successful_arc.json
```

Reward computation, drop into `EmotionAwareOpenLoopMCTS` (composes with Investigation
2's trajectory bonus):

```python
def _arc_distance_reward(self, state) -> float:
    """+: trajectory matches successful pattern;  -: trajectory matches failure pattern."""
    user_emotions = [r.emotion for r in state.history if r.role == state.USR]
    if not user_emotions:
        return 0.0
    total = len(user_emotions)
    reward = 0.0
    for i, emo in enumerate(user_emotions):
        bin_id = min(int((i / max(total, 1)) * len(self._target_arc)), len(self._target_arc) - 1)
        # probability mass on this emotion at this position in successful dialogs
        target_p = self._target_arc[bin_id].get(str(emo), 0.0)
        reward += target_p
    return reward / total            # normalised in [0, 1]
```

Then blend at the leaf in `_init_node`:

```python
v_donation = self.player.predict(state)[1]              # existing P(donate) estimate
v_arc = self._arc_distance_reward(state)
beta = float(getattr(self.configs, "arc_weight", 0.3))
v_blended = (1 - beta) * v_donation + beta * v_arc
return v_blended
```

### 3b — Learn `P(donate | emotion_trajectory)` (the real version)

Train a small sequence model on `(emotion_trajectory → donate)`. This makes emotion the
*substrate* of the value function: every change in user emotion changes the predicted
donation probability, which feeds straight into MCTS's Q-update via the standard backup.
No shaping side-channel — emotion is *intrinsic* to the value.

Input is short (≤15 emotion labels per dialog), so the model can be tiny. Two reasonable
architectures:

- **GRU over emotion-id embeddings** (~10K params): trivially trains on 300 dialogs,
  near-zero inference latency.
- **Small Transformer over emotion sequence** (~100K params): more capacity, similar
  inference latency. Overkill for sequences of length ≤15 unless you augment with DA /
  tone tokens too.

Training script sketch (mirrors `train_anticipatory_emotion_p4g.py`):

```python
# scripts/train_emo_trajectory_value_p4g.py
#
# Input:  sequence of (persuadee_emotion_at_turn_t) for t = 1..T
# Target: binary donation outcome
# Model:  Embedding(|Emotions|, 16) → GRU(16, 32) → Linear(32, 1) → sigmoid
# Loss:   BCE
# Output: outputs/emo_trajectory_value.pt  (state_dict + class index)
#
# At inference time inside MCTS:
#     v_emo = self._emo_trajectory_model(state).item()    # scalar in [0, 1]
# and blend with the existing v_donation as in Phase 3a.

import torch, torch.nn as nn

class EmoTrajectoryValue(nn.Module):
    def __init__(self, n_emotions: int, embed_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_emotions, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, emo_seq, lengths):       # emo_seq: (B, T), lengths: (B,)
        x = self.embed(emo_seq)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                    enforce_sorted=False)
        _, h = self.gru(packed)                # h: (1, B, hidden)
        return torch.sigmoid(self.head(h.squeeze(0))).squeeze(-1)   # (B,)
```

Featurisation: same emotion-labelling cache as the BERT script. Dataset: per dialog, the
sequence of persuadee emotions; label is the binary donation outcome. ~300 sequences,
trains in seconds.

### Composition: full stack

```
state = (history, last_user_emotion, emotion_trajectory)
action = (da, tone)
prior  = P(da | state) · P(tone | last_user_emotion)              # Phase 1+2
value  = α · P(donate | history) + (1-α) · P(donate | trajectory) # Phase 3b
shaping = local-only π(e) from the current EmoMCTS winner          # untouched
```

The MCTS itself doesn't change — same PUCT, same realisation machinery. What changes is
*what it's searching over*: not "which DA maximises a value estimate I mostly don't
understand," but "which `(DA, tone)` trajectory navigates the user from their current
emotional state to one that historically commits, with my own delivery affect modulated
to match the user I'm reading."

---

## Suggested order

| Step | Engineering cost | Risk | Payoff signal |
| ---- | ---------------- | ---- | ------------- |
| Phase 1 alone (toned actions, uniform tone prior) | ~1 day | low | qualitative realism + small win-rate move |
| Phase 1 + Phase 2 (hand-seeded reactive tone prior) | +0.5 day | low | DA × tone histogram shifts in the expected directions |
| Phase 1 + 2 + Phase 3a (arc distance) | +1 day + arc mining script | low | win-rate improvement scoped to dialogs with strong arcs |
| Phase 1 + 2 + Phase 3b (trajectory value model) | +2-3 days | medium | the headline result for an "emotion as navigation" paper |

The phases are independently ablatable — every step is a publishable contrast against
the previous one. Start with Phase 1 to confirm the LLM actually responds to tone
directives (a precondition for everything that follows); if it doesn't, the whole stack
is moot and you need a stronger system-LLM before any of this lands.

---

# Self-play RL for the tone policy

A natural follow-on to the navigation plan: once tone is a controllable action axis
(Phase 1) and a persona mixture defines a realistic user distribution (Item 3), the tone
policy can be **learned by self-play** rather than hand-seeded. AlphaZero-style policy
iteration is the right pattern because the MCTS infrastructure is already there — the
visit-count distribution becomes the policy training target, and the terminal donation
outcome becomes the value training target.

## Setup

**Environment (already exists)**:
- State: `(dialog history, user emotion trajectory)`
- Action: `(DA, tone)` from Phase 1
- Transition: `game.get_next_state` (system LLM → utterance, user LLM → reply,
  classifier → emotion label)
- Reward: terminal `+1` / `−1` from `get_dialog_ended`; optionally dense per-step
  shaping (local-only π(e) or the Phase 3b trajectory value)

**What gets learned**:
- `π_θ(da, tone | state)` — policy over the joint action space (or `π_θ(tone | state,
  da)` if you trust the existing DA prior)
- `V_φ(state)` — value estimate; predicts eventual donation probability

**What stays fixed** (critical — otherwise the reward signal drifts):
- User simulator
- Emotion classifier
- MCTS algorithm itself

## The self-play loop (AlphaZero pattern)

```
initialize  π_θ ← current planner LLM prior   (zero-shot policy)
            V_φ ← current heuristic value     (zero-shot value)

for iteration in 1..N:
    # 1) self-play: generate K dialogs using MCTS guided by current π_θ, V_φ
    buffer = []
    for k in 1..K:
        state = game.init_dialog(random_scenario, random_persona)
        episode = []
        while not terminal:
            run MCTS(state, π_θ, V_φ, n_sims)
            visit_dist = Nsa(state) / sum(Nsa(state))      # MCTS's improved policy
            action = sample(visit_dist)
            episode.append((state, visit_dist))             # value target filled at terminal
            state = step(action)
        reward = +1 if donate else -1
        for (s, vd) in episode:
            buffer.append((s, vd, reward))                  # optionally discount by γ^remaining_turns

    # 2) train:
    #    π_θ ← cross-entropy(π_θ(s), visit_dist)
    #    V_φ ← MSE / BCE(V_φ(s), reward)
    train_networks(buffer)

    # 3) checkpoint + evaluate vs previous π_θ on held-out scenarios
```

**Why MCTS-distillation rather than direct policy gradient**: MCTS already produces a
*better* policy at decision time than the raw `π_θ` (that's the whole point of search).
Treating MCTS visit counts as the training target is supervised learning with a clean
signal — much lower variance than REINFORCE on the same episode count. AlphaZero showed
this is the empirically strongest path when search is cheap relative to policy-gradient
updates.

## Policy / value network choices

Match the network to the action space.

For **tone-only** (`|TONES| = 7`):

- **Small MLP / linear head over featurised state** — fastest to train, easiest to
  deploy. Features: last user emotion one-hot + sys DA one-hot + emotion-trajectory
  bag-of-counts + turn index. Same skeleton as the earlier logistic-regression script.
- **Small encoder fine-tuning** — reuse the BERT-based anticipatory classifier as the
  backbone, swap the head. Higher capacity, handles dialog text natively. ~5 min/epoch
  on a single GPU.

For the full **`(DA, tone)` joint** (91 actions):

- The encoder-based option becomes the right choice; linear/MLP starts to underfit.
- **Shared encoder, two heads** (policy + value) — standard AlphaZero choice, lets the
  model amortise feature learning across both objectives.

Suggested starting architecture (PyTorch sketch):

```python
class TonePolicyValueNet(nn.Module):
    def __init__(self, backbone: str = "distilroberta-base", n_tones: int = 7):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(backbone)
        H = self.encoder.config.hidden_size
        self.policy_head = nn.Linear(H, n_tones)           # logits over tones
        self.value_head = nn.Linear(H, 1)                  # scalar value in [-1, +1]

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0]                                       # [CLS] pooling
        return self.policy_head(cls), torch.tanh(self.value_head(cls)).squeeze(-1)
```

## Reward shaping options

In rough order of how much "ground truth" they require:

1. **Sparse terminal only** (`±1` on donate). Cleanest signal, hardest to learn from.
   Needs thousands of episodes per iteration.
2. **+ local-only π(e) per step** (current EmoMCTS reward). Adds dense feedback at the
   cost of biasing toward "emotionally pleasant" trajectories.
3. **+ Phase 3b trajectory value model** as an intermediate reward
   (`r_t = V_traj(state_t) − V_traj(state_{t−1})`). Dense, learned, no manual penalty
   table. **Recommended for the RL loop** — closes another circular dependency and
   avoids the policy collapsing into "be inoffensive."

## Self-play opponent design

Persuasion isn't symmetric — there's no obvious "winning the other side." Options for
the user agent during self-play, ordered by sophistication:

- **Static user simulator** (current). Single-agent RL in practice. Risk: policy
  overfits to one user's quirks.
- **Persona mixture (Item 3)** — sample a persona per episode; policy learns to be
  robust across the mixture. **Recommended starting point** — gives a meaningful
  environment without inventing an adversary.
- **Adversarial user**: train a user policy whose objective is to *resist* donation
  (`reward_user = −reward_system`). Pushes the system toward robust strategies; risk:
  degenerates into a user that refuses everything regardless of context.
- **Co-trained user with realism constraint**: user policy is constrained to stay close
  to real p4g data (KL penalty against the supervised baseline). Most expensive; gives
  the most realistic adversary.

## Cost realism

Self-play is *expensive* because each rollout is many LLM calls. Back-of-envelope:

> 10 iterations × 200 self-play dialogs × ~10 turns × ~30 LLM calls per turn (MCTS sims)
> ≈ **600K LLM calls per training run.**

Mitigations (in order of impact):

- Small local model for the user simulator (Item 6 in the speedup section).
- HF emotion classifier instead of LLM (already supported).
- Reduce `num_MCTS_sims` during training — visit-count distillation is robust to noisy
  MCTS.
- Cache aggressively (open-loop hash makes this trivial; warm cache across episodes).
- Use **Item 2's anticipatory model** to skip in-rollout classifier calls.
- Combine with **Item 4 (top-K)** to shrink the per-node action set.

With all of these, ~600K calls becomes ~30–50K — runnable overnight on a single
~8-call/sec budget.

## Simpler alternative: REINFORCE with baseline

If AlphaZero feels like too much machinery, the lightweight version is:

```
sample episode under π_θ                              (no MCTS at training time)
R = terminal reward + Σ shaping rewards
∇θ J = E[ Σ_t ∇log π_θ(a_t | s_t) · (R − V_φ(s_t)) ]  (REINFORCE with baseline)
update V_φ toward observed returns
```

Faster per iteration (no MCTS in training), but ~10× higher variance — needs ~10× more
episodes for the same signal. Use this if engineering time is tighter than compute time;
otherwise AlphaZero-distillation wins on sample efficiency.

## Training-script skeleton

`scripts/self_play_train_tone_p4g.py` (write later; sketch shown so the interfaces are
visible):

```python
"""Self-play RL for the tone policy via AlphaZero-style MCTS distillation.

Loop:
  1. self-play K episodes with MCTS guided by current π_θ / V_φ
  2. distill MCTS visit counts into π_θ; regress V_φ against terminal donation outcomes
  3. swap the new networks into the planner; repeat

Bootstrap: π_θ initialised with the current planner LLM's prior (zero-shot); V_φ with
the current heuristic value. After enough iterations the networks should outperform
both.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

# ... (same path setup as the other scripts)

# Network definition above; load/save helpers; persona sampling; emotion classifier; etc.

def self_play_episode(game, mcts_cls, π_θ, V_φ, persona, n_sims):
    state = game.init_dialog(persona=persona)
    episode = []
    while not (terminal := game.get_dialog_ended(state)):
        planner = mcts_cls(game, π_θ, V_φ, configs=...)        # MCTS uses π_θ as prior
        for _ in range(n_sims):
            planner.search(state)
        visit_dist = planner.get_action_prob(state)             # length |actions|
        action = int(np.random.choice(len(visit_dist), p=visit_dist))
        episode.append((state.copy(), visit_dist))
        state, _, _ = game.get_next_state(state, action)
    reward = +1 if terminal == 1.0 else -1                      # donate vs no-donate
    return [(s, vd, reward) for (s, vd) in episode]

def train_step(net, buffer, batch_size, optim):
    loader = torch.utils.data.DataLoader(buffer, batch_size=batch_size, shuffle=True)
    for batch in loader:
        states, visit_dists, rewards = batch
        logits, values = net(*encode_states(states))
        loss_p = torch.nn.functional.cross_entropy(logits, visit_dists)   # distillation
        loss_v = torch.nn.functional.mse_loss(values, rewards)            # value
        loss = loss_p + loss_v
        optim.zero_grad(); loss.backward(); optim.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes_per_iter", type=int, default=200)
    parser.add_argument("--n_sims", type=int, default=10)
    parser.add_argument("--backbone", default="distilroberta-base")
    parser.add_argument("--out_dir", default="outputs/selfplay_tone")
    # plus persona file, classifier choice, lr, batch_size, etc.
    args = parser.parse_args()

    game = build_emo_p4g_game(...)                  # Phase 1 toned variant
    personas = load_personas("data/personas/p4g_personas.json")
    net = TonePolicyValueNet(backbone=args.backbone)
    optim = torch.optim.AdamW(net.parameters(), lr=2e-5)

    for it in range(args.iterations):
        buffer = []
        for k in range(args.episodes_per_iter):
            persona = sample_persona(personas)
            buffer += self_play_episode(game, TopKEmotionAwareDiscountQOpenLoopMCTS,
                                         π_θ=net, V_φ=net, persona=persona,
                                         n_sims=args.n_sims)
        train_step(net, buffer, batch_size=32, optim=optim)
        save_checkpoint(net, Path(args.out_dir) / f"iter_{it:02d}.pt")
        evaluate(net, held_out_scenarios)            # win rate vs previous iter
```

## What I'd actually build (suggested order)

1. **Land Phase 1** of the navigation plan first. RL has nothing to optimise against
   until tone is a controllable axis.
2. **Add Item 3** (persona mixture). Provides a meaningful environment to RL against,
   instead of "one undefined average user."
3. **Train Phase 3b** trajectory value model. Gives a dense reward without manual
   shaping; closes a circular dependency.
4. **Build the one-script AlphaZero loop** above. Reuses MCTS as-is; the only new code
   is the buffer, the policy/value network train step, and a wrapper that loads the
   trained networks back as PUCT priors.
5. **Start with tone-only policy** (`π_θ(tone | state, da)`, 7-way). Smaller search
   space, faster signal, isolates the tone-learning question from joint DA-tone
   confounds.
6. **Evaluate vs hand-seeded `TONE_PRIOR` from Phase 2** as the baseline. If the
   learned tone policy beats hand-seeded by ≥5pp win rate, you've shown self-play RL
   adds value over expert intuition — a paper-worthy result on its own.

## Composition with the rest of the doc

- **With Phase 1 (toned actions)** — prerequisite. RL has no degree of freedom to learn
  over otherwise.
- **With Phase 2 (hand-seeded tone prior)** — the learned policy replaces the hand
  table; hand table becomes the bootstrap and the baseline.
- **With Phase 3 (trajectory value)** — natural reward source for the RL value head.
- **With Item 2 (anticipatory emotion)** — composable; the anticipatory model can
  provide intra-rollout features so the policy doesn't need to learn emotion dynamics
  from scratch.
- **With Item 3 (personas)** — defines the training environment.
- **With Item 4 (top-K)** — keeps MCTS tractable inside the self-play loop.
- **With Investigation 1 (Bellman backup)** — orthogonal but reward-shaping
  interactions worth checking: if you ship Bellman shaping AND a learned value, V_φ
  ends up modelling the Bellman-shaped return, which can be confusing. Pick one.


# Parallel emotional reasoning in MCTS

The variants in this doc so far treat emotion as a *modifier* on a single
donation-optimizing reward scaffold:

- `EmotionAwareDiscountQOpenLoopMCTS` — π penalty mixed into Q.
- `EmotionGuidedDiscountQOpenLoopMCTS` — emotion-DA bonus mixed into PUCT.
- `EmotionRealizationSelectorMCTS` — emotion valence reranks K cached realizations.

In all three, emotion is a multiplicative/additive tweak on top of the
donation objective. The search never gets to *reason about emotional outcome
as a thing in its own right* — it can't tell you "this branch has 60%
donation likelihood but causes emotional collapse" because everything is
already collapsed into one scalar Q.

Four directions to give emotion a parallel reasoning channel.

## Direction A — Multi-objective parallel Q

Track TWO independent backups per (s, a):

```
Q_donate[s][a]  — running mean of donation-rollout reward (current Q)
Q_emo[s][a]     — running mean of expected emotional valence at leaf
```

Both back up independently up the tree. PUCT becomes:

```
score(a) = Q_donate[s][a]
        + β · Q_emo[s][a]
        + c · P[s][a] · √N / (1 + Nsa[s][a])
```

`β` is a sweep knob (start with 0.0 / 0.3 / 0.7 / 1.0). Sweeping `β`
post-hoc on the same search tree is *almost free* — much cheaper than
re-running with different `lambda_emo`, because the per-channel statistics
are decomposable.

### Why this is structurally different from the current π penalty

The penalty is *baked into one Q* the moment the search backs it up, so
post-hoc attribution is impossible. With parallel Q's:

- Ablation is one float at decision time, not a re-search.
- You can answer "which actions had high `Q_emo` but low `Q_donate`?" by
  reading the tree directly — current code can't.
- Enables **Pareto-aware selection**: keep all actions on the
  donation/emotion frontier, pick among them by visit-weighted vote or by
  the action whose Pareto rank is best.

### Implementation sketch

Subclass `EmotionAwareOpenLoopMCTS`. Add `self.Q_emo` initialised like
`self.Q`. In `search()`:

```python
# Existing donation backup
self.Q[hashable_state][best_action] = (
    self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v
) / (self.Nsa[hashable_state][best_action] + 1)

# NEW: parallel emotion backup
emo_v = self._emotion_quality(next_state.predicted_distribution())  # uses EMOTION_VALENCE
self.Q_emo[hashable_state][best_action] = (
    self.Nsa[hashable_state][best_action] * self.Q_emo[hashable_state][best_action] + emo_v
) / (self.Nsa[hashable_state][best_action] + 1)
```

In `_calculate_uct`, add `self.beta_emo * self.Q_emo[s][a]` to the score.

Cost: one float per (s, a) entry, no extra LLM calls (the emotion
distribution is already cached on `next_state` by the game). ~50 LOC.

## Direction B — Emotion-rollout imagination side-tree

Today every MCTS sim does ONE expensive LLM-driven rollout to estimate
donation outcome. Add a SECOND, cheap rollout that simulates *only* the
user's emotional arc — no LLM, no utterances — using a trained Markov
chain `P(emotion_{t+1} | emotion_t, sys_da_t)` (the anticipatory model
specced in `scripts/train_anticipatory_emotion_p4g.py` is a strict
generalisation of this).

### Mechanics

For each (s, a) at expansion:

1. **Donation rollout** (current): 1 expensive LLM-driven branch → `v_donate`.
2. **Emotion rollout** (new): 50 cheap MC samples through the Markov
   chain, each running 10+ turns deep → empirical distribution over
   end-of-conversation emotional state → scalarized to `v_emo` via
   `EMOTION_VALENCE`.

Combine:
- As a second Q channel (Direction A), or
- As a refined prior (multiply `P[s][a]` by something monotone in
  `v_emo`), or
- As an extra term in PUCT directly.

### Why this is the unique research angle

The LLM-bound MCTS is shallow by necessity — every rollout step costs an
LLM call. The emotion-only side-tree is cost-asymmetric:

- Donation tree: depth ≤ 3 turns at 20 sims (~60 LLM calls).
- Emotion tree: depth 10+ turns at 50 sims (~500 Markov chain steps =
  negligible cost).

So for the first time MCTS reasons about LATE-GAME emotional consequences
of early moves — currently the search can't see past the second turn. This
is also genuinely novel: no published dialog-MCTS paper has a parallel
imagination side-tree on a learned dynamics model.

### Implementation prerequisite

Train `P_anticipatory(emotion_{t+1} | history, sys_da)` per Item 2
(already specced). Or, for a cheaper start, marginalise to a tabular
`P(emotion_{t+1} | emotion_t, sys_da)` — 8 emotions × 13 DAs × 8 emotions
= 832 cells, easy to estimate from the 300 dialogs (Laplace smoothing).

## Direction C — POMDP framing with a latent persuasion-state belief

Treat the user's true willingness as a hidden Markov state with 4-5
discrete classes:

```
{exploring, doubtful, considering, committed, refusing}
```

Each turn, the emotion observation refines a *belief distribution* over
these latent states (Bayes update). MCTS Q is now over `(history, belief)`
jointly. The planner reasons "this DA shifts belief mass from `doubtful`
toward `considering`" instead of "this DA correlates with positive emotion."

### Setup

- 5-state HMM.
- **Emission**: `P(emotion | persuasion_state)` — learned from 250 train
  dialogs by EM, with the persuadee emotion sequence as observations and
  the eventual donation outcome anchoring at least the terminal state.
- **Transition**: `P(state_{t+1} | state_t, sys_da)` — learned from the
  same data, conditioned on the DA the persuader played.
- **Belief update** on every user turn: standard HMM forward step.
- **Reward**: belief mass on `committed` at terminal turn (or expected
  donation under the marginalised emission).

MCTS keeps a `belief: np.ndarray` on each `EmotionAwareDialogSession`,
updated each turn alongside the emotion distribution. PUCT runs over
`(s_hash, action)` as today; the belief enters via the reward at terminal
states (the leaf value becomes `E_belief[donation_prob]`).

### Why this is the publishable story

Most clean reframing in the project:

- Recasts "emotion-aware reward shaping" as **POMDP planning where
  emotion is an observation**, which is a well-studied setting in the
  bandits / belief-MDP literature.
- Gives a principled answer to "why is emotion useful?" — it's the
  observation that disambiguates the latent persuasion state.
- The HMM transitions / emissions are interpretable: you can publish the
  fitted matrices and tell a story about persuasion dynamics.

Cost: most engineering of the four. Need the HMM + belief plumbing in
`EmotionAwareDialogSession`, plus the EM fit. But there are no new
moving parts at MCTS inference time — just a belief vector that gets
updated alongside the current emotion distribution.

### Composition with the rest of the doc

- With **Item 2 (anticipatory emotion)** — the same data + featurisation
  yields the emission matrix `P(emotion | state)`. Building one helps the
  other.
- With **Direction A** — belief mass on `committed` becomes the natural
  `Q_emo` channel.
- With **Investigation 3 (learned π)** — π becomes redundant once you
  have a belief: just penalise low belief on `committed`.

## Direction D — Risk-sensitive / quantile MCTS

Smallest theoretical novelty but addresses a real flaw the current penalty
masks: vanilla MCTS optimises *expected* donation, but a strategy with 60%
donation rate where 40% of failures end in *anger* is strictly worse than
55% with 5% anger — for downstream judge scoring AND for actual deployment
(angry users churn, complain, talk to others).

Replace `Q = mean(rewards)` with:

```
Q = CVaR_α(rewards)       # conditional value-at-risk
# or
Q = quantile_α(rewards)   # upper / lower α-quantile
```

Emotion gives you the variance signal: failure modes that end in negative
emotion get *extra* downweight in the risk tail. Concretely:

```python
# Per-(s, a) keep the empirical distribution of returns, not just the mean.
self.returns = defaultdict(lambda: defaultdict(list))
self.returns[s_hash][a].append((v_donation, emotion_quality))

# At UCT time:
returns_a = sorted(r for r, _ in self.returns[s_hash][a])
cvar = mean(returns_a[:int(alpha * len(returns_a))])  # avg of worst α-fraction
score = cvar + c_puct * P[s_hash][a] * sqrt(Ns) / (1 + Nsa[s_hash][a])
```

For a CVaR-with-emotion-weighting twist: weight each return by
`exp(-γ · max(0, -emotion_quality))` so emotional failures sit deeper in
the tail. Knob `γ` controls how much emotion shapes the risk view.

### Why include this

- ~30 LOC; cheapest of the four.
- Diagnoses a flaw vanilla GDPZero / current emoMCTS share: optimistic
  expected-reward maximisation produces flashy strategies that bomb on
  the long tail.
- The judge (GPT-3.5 / GPT-4o) sees the *median* not the mean — quantile
  MCTS may align better with how the judge scores.

## Recommended order

1. **Direction A first** — smallest change, biggest substrate value. The
   parallel Q channels are the architectural substrate for B, C, and D
   (they all plug into the second Q field A introduces).

2. **Direction D second** — independent of A, can ship in parallel. ~30
   LOC. Diagnoses a flaw the current π penalty can't address.

3. **Direction B third** — once parallel Q lives, the imagination
   rollouts become "just another source of `Q_emo` updates" with cheap
   Markov chain dynamics. Biggest research-novelty payoff in the
   short term.

4. **Direction C last** — most engineering and the riskiest
   research bet, but the cleanest theoretical reframing of the project
   if it works. Best paper material.

### Composition with existing variants

- A + π penalty: works, but two ways to express emotion bias — pick one
  per run for clean attribution.
- A + emotion-DA bonus: orthogonal (bonus shapes selection, parallel Q
  shapes value). Safe to combine.
- B + learned prior (`EmotionConditionedPriorMCTS`): high synergy —
  imagination rollouts refine `Q_emo`, learned prior shapes which
  branches get rolled out.
- C subsumes the π penalty entirely — drop π if you ship C.
- D + everything: pure value-aggregation change, fully compositional.


# Measurement improvements — is the metric hiding a win?

The current evaluation pipeline (`run_judge.py` → win-rate vs GDPZero on a
GPT-3.5/4o judge) has several structural blindspots for what EmoMCTS is
actually optimizing for. Before any more algorithm work, audit whether the
metric is masking a real signal.

## What the current metric measures

`run_judge.py` runs the judge on `(context, ori_resp_human, new_resp_planner)`
tuples turn-by-turn and asks "which next response is better?" Aggregates
across all turns of 20 dialogs into a single win-rate scalar. Current
EmoMCTS variants land at ~45–50% — within statistical noise of GDPZero.

## Why this is structurally biased AGAINST EmoMCTS

1. **Single-turn evaluation, multi-turn strategy.** The whole premise of
   the emotion-DA bonus / penalty / valence rerank is steering the
   *trajectory* — building rapport, recovering from emotional missteps,
   setting up a successful future ask. The judge sees one turn in
   isolation. A turn that quietly de-escalates anger and sets up a
   successful future proposition looks weaker than a flashy ask now, even
   when the de-escalating turn is what *causes* downstream donation. The
   bonus matrix's "play task related inquiry when the user is sad" gets
   actively *penalised* by a judge that prefers direct persuasion in the
   moment.

2. **Frozen human prefix kills the value of trajectory steering.** Both
   planners see the same ground-truth dialog up to turn t. EmoMCTS can't
   actually *reach* the trajectories it would have steered toward — its
   value compounds across turns the replay short-circuits. So you're
   measuring "given a state EmoMCTS would never have reached, what move
   would it play?" That's a degraded version of the policy.

3. **Judge proxy ≠ donation outcome.** Win rate measures *judge
   preference*, not "did the user donate." GPT-3.5 / GPT-4o have their
   own priors about what "good persuasion" looks like (typically: direct,
   confident, evidence-based). EmoMCTS plays stylistically *different*
   moves, not necessarily worse — but the judge's preference can be
   uncorrelated with donation lift.

4. **Aggregation flattens the signal.** A 50% aggregate can easily be:
   - 30% on greetings (where neither planner has any real degree of
     freedom; ground-truth prior dominates),
   - 50% on filler / information-exchange turns,
   - 70% on decisive emotion-sensitive turns.

   Without slicing you can't tell. The current pipeline already records
   the metadata needed (`counterfactual_da`, `last_user_emotion`,
   `turn_index`) but no script consumes it.

5. **Statistical noise floor.** 20 dialogs × ~7 turns = ~140 binary
   comparisons. Binomial 95% CI on a 50% rate is roughly ±8pp. So 45% vs
   50% vs 53% are all within noise — yet the project has been making
   algorithmic decisions based on this signal.

## Metric improvement 1 — Conditional win-rate slicing (cheapest, do first)

`run_judge.py` already writes the metadata sidecar with `last_user_emotion`,
`counterfactual_da`, `turn_index`. Add a slicer script that re-aggregates
win rate over interesting sub-populations of turns.

### Slices that matter

- **Bonus-flipped turns only.** Filter to turns where
  `counterfactual_da["argmax_visits"] != counterfactual_da["puct_no_bonus"]`
  — these are the ONLY turns where the emotion shaping changed planner
  behavior. If win rate here is 60% but aggregate is 50%, your method
  works and the dilution is what's hiding it. Conversely, if it's 45% on
  bonus-flipped turns, the bonus is actively *hurting* and you've
  diagnosed a bug.

- **Emotion-conditioned slices.** Win rate when `last_user_emotion ∈
  {anger, sadness, fear, disgust}` (negative emotions, where emotion
  shaping should matter most) vs `∈ {neutral, happiness}` (where the
  prior should dominate). If EmoMCTS is +10pp on negative-emotion turns
  and -5pp on neutral, that's a clean per-cell calibration story (and
  argues for raising `c_emo_bonus` only when last_user_emotion ∈ neg).

- **Decisive turns only.** Rerun with `--run_only_decisive_p4g_turns`
  (already implemented). Compare aggregate win rate vs full-turn
  aggregate. Decisive turns are where strategic differences matter; the
  ratio of (decisive winrate) / (full winrate) is the strategic-leverage
  estimate.

- **Per-turn-index slice.** Win rate vs absolute turn index, plot the
  curve. A planner that wins early turns but loses late turns has a
  different bug than one that wins late but loses early.

### Implementation

One Python script reading the pickles + `*_metadata.json` sidecar:

```python
# scripts/slice_winrates.py  (sketch)
import json, pickle, sys
from collections import defaultdict

def load_run(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_judge(judge_jsonl):
    # one JSON object per scored turn: {did, turn_index, winner: "A"|"B"|"tie", ...}
    return [json.loads(l) for l in open(judge_jsonl)]

def winrate(rows):
    if not rows: return None
    return sum(r["winner"] == "B" for r in rows) / len(rows)  # B = EmoMCTS

# slice keys
rows_by_emotion = defaultdict(list)
rows_by_flip    = defaultdict(list)
rows_by_turn    = defaultdict(list)

for row, judge in zip(load_run("outputs/emomcts.pkl"), load_judge("outputs/emomcts_judge.jsonl")):
    rows_by_emotion[row["last_user_emotion"]].append(judge)
    flipped = (row["counterfactual_da"]["argmax_visits"] !=
               row["counterfactual_da"]["puct_no_bonus"])
    rows_by_flip["flipped" if flipped else "same"].append(judge)
    rows_by_turn[row["turn_index"]].append(judge)

for k, rows in rows_by_emotion.items():
    print(f"  last_user_emotion={k}: n={len(rows):3d} winrate={winrate(rows):.1%}")
# ... same for flip / turn_index
```

Cost: ~50 LOC, runs in seconds over existing pickles, no new
experiments. Do this **first**, before any further algorithm work — the
existing data may already contain the answer.

## Metric improvement 2 — End-to-end simulated donation rate (extend `rollout.py`)

**Status: partially shipped.** `src/runners/rollout.py` already runs each
planner from scratch against the simulated user for a full dialog, and
`src/metrics/dialog_metrics.py` already computes SR (Success Rate = end-to-end
donation rate) and AT (Average Turn). This is the correct primary metric
and it exists today — the h2h judge is a per-turn proxy on top.

```
cd src
python runners/rollout.py --game emo_p4g --algo emomcts --num_mcts_sims 20
python metrics/run_metrics.py --episodes outputs/rollout.pkl --max_turns 8
```

What's actually missing — three gaps that turn `rollout.py` from "exists"
into "measures the variants we care about with the slices we need":

### Gap A — `rollout.py:pick_action` hardcodes the base MCTS class

```python
if algo == "emomcts":
    dp = EmotionAwareOpenLoopMCTS(game, planner, configs, emotion_classifier)
```

The penalty / bonus / valence-rerank / learned-prior subclasses
(`EmotionAwareDiscountQOpenLoopMCTS`, `EmotionGuidedDiscountQOpenLoopMCTS`,
`EmotionRealizationSelectorMCTS`, `EmotionConditionedPriorMCTS`) cannot
be selected, so the entire research arc currently has *no end-to-end
SR/AT numbers* — only h2h judge numbers. Mirror the variant selection
from `runners/emomcts.py:main` here:

```python
if algo == "emomcts":
    if cmd_args.emotion_prior_model:
        mcts_cls = EmotionConditionedPriorMCTS
    elif cmd_args.alpha_realization_emo > 0:
        mcts_cls = EmotionRealizationSelectorMCTS
    elif cmd_args.c_emo_bonus > 0:
        mcts_cls = EmotionGuidedDiscountQOpenLoopMCTS
    elif cmd_args.lambda_emo > 0 or cmd_args.use_penalty:
        mcts_cls = EmotionAwareDiscountQOpenLoopMCTS
    else:
        mcts_cls = EmotionAwareOpenLoopMCTS
    dp = mcts_cls(game, planner, configs, emotion_classifier, **extra_kwargs)
```

Plus forward `--c_emo_bonus`, `--lambda_emo`, `--alpha_realization_emo`,
`--emotion_prior_model`, `--emotion_prior_temperature` from the parser
into `configs`. ~40 LOC, no new file.

### Gap B — `make_episode` drops the Tier-1 metadata `emomcts.py` records

The h2h pipeline (`runners/emomcts.py:cmp_data`) records per-turn
`counterfactual_da`, `last_user_emotion`, `last_user_distribution`,
`turn_index` so the slicer in Metric Improvement 1 can attribute outcomes.
`make_episode` records only `success` / `num_turns` / `history`. Without
the metadata, you cannot slice end-to-end SR by "did the bonus flip an
action on a critical turn" or by "what was the user's emotional state at
the proposition turn."

Extend `make_episode` to also pull per-turn `last_user_emotion` and
`counterfactual_da` (where available) from the dialog_planner's last
state. This makes Metric Improvement 1's slicing apply to rollout
episodes, not just judge tuples.

### Gap C — No emotion-arc metric in `dialog_metrics.py`

Metric Improvement 3 (arc quality) reads the `*_emotions.json` sidecar
written by `dump_emotion_records`. Extend `dialog_metrics.compute_metrics`
to compute `arc_quality` directly from `episode["history"]` (where the
`EmotionalHistoryRecord` distributions live for emo_p4g rollouts) so the
arc number appears next to SR / AT in `format_metrics`. Then a single
`python metrics/run_metrics.py` call gives SR, AT, AND arc — one
scoreboard, three numbers.

### Why this matters

With Gaps A-C closed, the h2h judge becomes a *secondary* diagnostic
(useful for per-turn calibration), and rollout SR / AT / arc become the
headline numbers. Everything in the parallel-emotion-reasoning section
(Directions A/B/C/D) becomes measurable end-to-end on the actual game
objective, not on next-turn judge preference.

### Caveat

The user simulator is the same LLM MCTS searches against, so SR measures
"win against this simulator," not real human persuasion. Same caveat as
the current h2h judge. Mitigate with Item 3 (persona-conditioned user
simulator) and by spot-checking 10 dialogs by hand to confirm the
simulator isn't trivially donating to everything.

### Cost

Same back-of-envelope as before: ~300 LLM calls per dialog at 20 sims ×
5 turns × ~3 calls/sim. 50 dialogs × 2 planners = ~30k LLM calls per
nightly evaluation. Tractable on Ollama, careful on OpenAI.

## Metric improvement 3 — Emotion-trajectory quality (directly measures what we optimize)

Independent of donation outcome, EmoMCTS *should* produce systematically
better user emotional trajectories than GDPZero — that's literally what
the bonus matrix + penalty + valence rerank are designed for. If it
doesn't, the shaping isn't working. If it does but h2h ties, the judge is
ignoring the value the shaping creates.

### Metric

Per dialog, compute:

```python
arc_quality = sum(
    EMOTION_VALENCE.get(emo, 0.0) * prob
    for record in user_turns
    for emo, prob in record.distribution.items()
) / len(user_turns)
```

This is the mean expected valence of the user across the dialog, in
[-1, +1]. Report:

- **Mean arc quality** across the run (one scalar per planner).
- **Arc-quality distribution** (histogram) — a planner with high mean +
  low variance is steadier than a planner with high mean + high variance.
- **Conditional arc quality** by terminal outcome — does EmoMCTS produce
  better arcs even on dialogs that fail to donate? That would suggest
  the emotion shaping works but the donation outcome is decided elsewhere.

### Why include it

- 30 LOC reading existing `*_emotions.json` files. No new experiments.
- Directly measures what the shaping is *for*; the donation rate is
  downstream + noisy + judge-mediated.
- A failing dialog with a positive arc is a different problem than a
  failing dialog with a negative arc. Currently you can't tell the
  difference.

### Implementation

```python
# scripts/measure_arc_quality.py  (sketch)
import json
from mcts.emotion_mcts import EMOTION_VALENCE  # already exists

def arc_quality(emotion_records):
    user_records = [r for r in emotion_records if r["role"] == "Persuadee"]
    if not user_records:
        return None
    total = 0.0
    for rec in user_records:
        for emo, p in rec["distribution"].items():
            total += EMOTION_VALENCE.get(emo, 0.0) * p
    return total / len(user_records)

for run in ["gdpzero", "emomcts"]:
    records = json.load(open(f"outputs/{run}_emotions.json"))
    print(f"{run}: arc_quality={arc_quality(records):.3f}")
```

## Metric improvement 4 — Emotion-distribution delta between planners

Improvement 3 is a single scalar per planner (mean expected valence).
Improvement 4 decomposes the *difference* between two planners' emotion
distributions into three orthogonal components, so the comparison is
interpretable instead of just "A had more happiness than B."

### The decomposition

When you compare GDPZero vs EmoMCTS aggregate emotion distributions,
three confounded effects produce the difference:

1. **Strategic DA-choice shift** — EmoMCTS plays a different DA mix
   (e.g. proposition 48% vs GDPZero's ~10%). Different DAs elicit
   different emotion bases, so even with identical utterance quality
   the aggregate shifts.
2. **Per-DA utterance-realization quality** — for the SAME DA, does
   one planner's specific phrasing elicit better emotions than the
   other's? This is what `EmotionRealizationSelectorMCTS` is designed
   to improve.
3. **Trajectory / sequencing effect** — the path through DA-space
   changes the emotional state the user is in when a given DA is
   played. A proposition after rapport-building elicits different
   emotion than a cold-opener proposition.

Reporting only the aggregate confounds all three. Report all four
slices so each effect is attributable to its source.

### The four numbers

Per pair (planner_A, planner_B), using existing `*_emotions.json` and
`*_da_emotions.json` sidecars:

**Aggregate distribution delta** (combined effect):

```
Δ_p(emotion) = p_A(emotion) - p_B(emotion)        # per-emotion delta in pp
KL(A ∥ B)    = Σ p_A(e) · log(p_A(e) / p_B(e))    # asymmetric distance
EMD(A, B)    = earth_mover's_distance(p_A, p_B)   # symmetric distance
```

**DA-marginal shift** (isolates strategic-action selection):

```
Δ_da(da) = count_A(da) / total_A - count_B(da) / total_B   # per-DA pp shift
```

**DA-conditional emotion shift** (isolates utterance-realization quality):

For each DA played by both planners ≥ N times (set N=50):

```
p_A(emotion | da) = count_A(da, emotion) / count_A(da)
p_B(emotion | da) = count_B(da, emotion) / count_B(da)
Δ_p|da(emotion | da) = p_A(emotion | da) - p_B(emotion | da)
```

Aggregate across DAs into a weighted mean:

```
weighted_realization_delta = Σ_da [ weight(da) · Σ_e (Δ_p|da(e|da) · valence(e)) ]
   where weight(da) = (count_A(da) + count_B(da)) / total
```

If this is positive, EmoMCTS's utterances elicit better emotions even
controlling for which DA was played → realization-quality wins.

**Mean expected valence delta** (Improvement 3, restated):

```
Δ_valence = mean_valence(A) - mean_valence(B)
```

### Implementation

Roughly 80 LOC over existing data:

```python
# scripts/emotion_delta.py  (sketch)
import json
from collections import Counter
from mcts.emotion_mcts import EMOTION_VALENCE

def load(run_dir):
    return (
        json.load(open(f"{run_dir}/{run_dir.split('/')[-1]}_emotions.json")),
        json.load(open(f"{run_dir}/{run_dir.split('/')[-1]}_da_emotions.json")),
    )

def aggregate_dist(records):
    c = Counter(r["emotion"] for r in records)
    total = sum(c.values())
    return {e: n / total for e, n in c.items()}

def mean_valence(dist):
    return sum(p * EMOTION_VALENCE.get(emo, 0.0) for emo, p in dist.items())

def kl(p, q, eps=1e-9):
    return sum(p.get(e, eps) * (math.log(p.get(e, eps)) - math.log(q.get(e, eps)))
               for e in set(p) | set(q))

def da_marginal_shift(da_emo_A, da_emo_B):
    total_A = sum(d["total"] for d in da_emo_A.values())
    total_B = sum(d["total"] for d in da_emo_B.values())
    return {
        da: da_emo_A.get(da, {"total": 0})["total"] / total_A
            - da_emo_B.get(da, {"total": 0})["total"] / total_B
        for da in set(da_emo_A) | set(da_emo_B)
    }

def da_conditional_delta(da_emo_A, da_emo_B, min_n=50):
    out = {}
    for da in set(da_emo_A) & set(da_emo_B):
        a, b = da_emo_A[da], da_emo_B[da]
        if a["total"] < min_n or b["total"] < min_n:
            continue
        out[da] = {
            "valence_A": mean_valence(a["fractions"]),
            "valence_B": mean_valence(b["fractions"]),
            "valence_delta": mean_valence(a["fractions"]) - mean_valence(b["fractions"]),
            "count_A": a["total"], "count_B": b["total"],
        }
    return out

records_A, da_A = load("outputs/emomcts_run")
records_B, da_B = load("outputs/gdpzero_run")
agg_A, agg_B = aggregate_dist(records_A), aggregate_dist(records_B)

print(f"aggregate valence: A={mean_valence(agg_A):+.3f}  B={mean_valence(agg_B):+.3f}  Δ={mean_valence(agg_A)-mean_valence(agg_B):+.3f}")
print(f"KL(A‖B)={kl(agg_A, agg_B):.3f}")
print("per-emotion shift (pp):")
for e in sorted(set(agg_A) | set(agg_B), key=lambda x: -abs(agg_A.get(x,0) - agg_B.get(x,0))):
    print(f"  {e:>10}: {(agg_A.get(e,0) - agg_B.get(e,0))*100:+.1f}")

print("\nDA marginal shift (which DAs each planner over-plays vs the other):")
for da, d in sorted(da_marginal_shift(da_A, da_B).items(), key=lambda kv: -abs(kv[1])):
    print(f"  {da:>30}: {d*100:+.1f} pp")

print("\nDA-conditional valence delta (utterance-quality only, min n=50):")
for da, info in sorted(da_conditional_delta(da_A, da_B).items(),
                       key=lambda kv: -abs(kv[1]["valence_delta"])):
    print(f"  {da:>30}: Δvalence={info['valence_delta']:+.3f}  (n_A={info['count_A']}, n_B={info['count_B']})")
```

### How to read the output

A representative interpretation:

| Pattern in the output | What it means | Action |
| --- | --- | --- |
| Aggregate Δvalence > 0, DA-conditional Δvalence ≈ 0 | EmoMCTS wins by playing *different* DAs, not better utterances per DA. Bonus matrix is doing the work. | Focus future effort on the bonus matrix / learned prior. Realization rerank is a sideshow. |
| Aggregate Δvalence ≈ 0, DA-conditional Δvalence > 0 | EmoMCTS plays the same DAs as GDPZero but *says them better*. Realization quality is the win. | `EmotionRealizationSelectorMCTS` is the highest-leverage variant. |
| Both > 0 | Combined win (DA choice + utterance quality both improving). | Headline result — both interventions stack. |
| Aggregate Δvalence < 0 but SR also drops | Emotion shaping is making the user *more* negative. Bonus matrix is miscalibrated. | Audit the matrix against the per-emotion shifts (which emotion class dropped where?). |
| Aggregate Δvalence > 0 but SR is flat | User simulator is happier but doesn't donate more. Either the simulator is sycophantic-by-default OR the emotion classifier is rewarding the wrong thing (e.g. polite refusals classified as "happiness"). | Calibration audit on the emotion classifier; check 10 dialogs by hand. |

### Caveats

- Same emotion-classifier calibration issue flagged elsewhere in the
  doc — "sadness" in the HF classifier on p4g context means *polite
  refusal*, not engagement. The Δ table is only as good as the
  classifier's label semantics. Consider re-mapping
  `EMOTION_VALENCE` for p4g (sadness → strong negative, not mild) and
  reporting both the raw and re-mapped deltas.
- Sample-size threshold on DA-conditional (the `min_n=50` filter)
  matters: rare DAs like greeting or self modeling won't show up in
  the per-DA table. Lower `min_n` at the cost of noise.
- The aggregate distribution mixes turns of different lengths /
  positions. If EmoMCTS produces shorter dialogs (closes faster), the
  aggregate is weighted toward late-game emotions; GDPZero may be
  weighted toward mid-game. Normalize by per-position mean if this
  matters (`metric improvement 1` slicing buys this for free).

### Why this slots in as a diagnostic, not headline

Improvement 4 doesn't tell you whether the planner is winning the
*game* — only whether the emotional landscape shifted. Pair with SR
(Improvement 2) as the headline and Improvement 4 as the explanation:

- SR up + Δvalence up → algorithm works, and works via emotion (intended).
- SR up + Δvalence flat → algorithm works through a different channel
  (probably the DA mix change, independent of emotion).
- SR flat + Δvalence up → emotion-aware moves but no outcome win →
  most likely user-simulator sycophancy or classifier mis-calibration.
- SR down + Δvalence up → over-optimizing the emotion proxy at the
  expense of the donation objective. Reduce `c_emo_bonus` or
  `lambda_emo`.

## Recommended order (metrics)

1. **Improvement 1 today** — slicing existing pickles is cheap. Likely
   to surface a real signal already buried in the aggregate. Conditions
   subsequent algorithm choices on real data, not noise.
2. **Improvement 3 today** — also 30 LOC over existing data. Tells you
   whether the emotion shaping is working at all, independent of the
   donation outcome metric.
3. **Improvement 4 today** — same data source as Improvement 3 (the
   sidecars are already written by every emomcts run). Adds the
   GDPZero-vs-EmoMCTS comparison axis, decomposed into strategic /
   realization / trajectory effects. ~80 LOC.
4. **Improvement 2 this week** — close Gaps A-C in `rollout.py` to
   surface SR / AT / arc-quality numbers for the emotion-aware MCTS
   subclasses. Becomes the headline metric, the others become
   diagnostics.

### Composition with the rest of the doc

- All four parallel-emotion directions (A/B/C/D above) need a metric
  that can actually *see* their effect. End-to-end donation rate
  (Improvement 2) is the only metric in this section that gives a
  multi-turn planner room to breathe.
- The bonus-matrix recalibration work in debug.md was guided by judge
  win rates. Re-evaluate with the slicing script — it may turn out
  some of the "regression" cells were actually wins on the slices
  that matter.
- Improvement 4 directly tests whether the bonus matrix is doing what
  it's supposed to do (per-DA conditional delta), which is exactly the
  diagnostic the matrix-recalibration loop in debug.md has been
  missing. Wire it in as part of every bonus-matrix edit.
- POMDP framing (Direction C): the belief state itself becomes a
  natural metric — "did the belief mass on `committed` rise over the
  dialog" is a fifth measurement that directly probes the belief
  model's quality.


# Experimental results so far (as of 2026-06-01)

Living log of what's been measured vs GDPZero. Update with each new run. All
runs are on p4g with vicuna:13b as backbone, HF emotion classifier
(j-hartmann/emotion-english-distilroberta-base), and either the
gpt-3.5-turbo h2h judge (myopic per-turn preference, 152 comparisons across
20 dialogs) OR end-to-end SR via `runners/rollout.py` (terminal donation rate).

## Methods status

Three buckets: (a) **tested** with measured result, (b) **implemented**
and runnable but not yet measured, (c) **specced** in the doc but not yet
built. Each row links to the class / script that implements it where
applicable, and to the doc section that motivates it where applicable.

### (a) Tested against GDPZero — has a measured result

| # | Method | Class / file | Key config | Metric | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | GDPZero (h2h anchor) | `OpenLoopMCTS` | N=20, 20 dialogs | win rate vs human | 50% — calibration baseline |
| 2 | GDPZero (rollout) | `OpenLoopMCTS` | N=20, 20 dialogs | SR | 75% ±19pp — noisy reference (n too small) |
| 3 | π emotion penalty | `EmotionAwareDiscountQOpenLoopMCTS` | λ=0.3, N=20 | h2h vs GDPZero | ~45% — **tied / noise** |
| 4 | Bonus matrix (v1, aggressive) | `EmotionGuidedDiscountQOpenLoopMCTS` | c=1.0, +0.60 Happiness→proposition, +0.40 Neutral→proposition, top-K=7, N=20 | h2h vs GDPZero | **44.7%** (68W / 84L / 0D) — **lost** |
| 5 | Parallel Q (multi-objective) | `EmotionAwareMultiObjectiveQ` | β=0.7, top-K=5, N=40, **50 dialogs** | SR | **72% ±12pp** — statistically tied; matched GDPZero baseline missing |

### (b) Implemented but not yet measured

Everything in this bucket is one `--flag` away from a run. Ready to test
once compute budget allows.

| # | Method | Class / wiring | Notes |
| --- | --- | --- | --- |
| 6 | Bonus matrix v2 (calibrated) | `EmotionGuidedDiscountQOpenLoopMCTS` (same as #4) | Happiness/Neutral proposition cells cut from +0.60 / +0.40 to +0.20 / +0.10 after h2h analysis. Awaiting rerun. |
| 7 | Realization-layer rerank | `EmotionRealizationSelectorMCTS` (`--alpha_realization_emo`) | Picks utterance by `V + α · emotion_fit` instead of V alone. Built, never measured. |
| 8 | LLM top-K action prior | `P4GChatSystemPlanner._predict_topk_prior` (`--llm_prior_topk`) | Single LLM call returns top-K DAs; MCTS hard-prunes valid moves to those. Used inside #4-#7; never isolated as its own ablation. |
| 9 | Learned action prior | `EmotionConditionedPriorMCTS` (`--emotion_prior_model`) | DistilRoBERTa head; train with `scripts/train_emotion_conditioned_prior_p4g.py` (250 train / 50 test split). Class wired into both runners. Model not yet trained / no run yet. |
| 10 | Imagination side-tree (Direction B) | `EmotionImaginationMCTS` (`--c_imag` / `--imag_K` / `--imag_H`) | Tabular Markov dynamics baked in (`EMOTION_TRANSITION_TABLE`, 832 cells, fit from 250 dialogs by `src/emotion_mining/learn_emotion_transition_p4g.py`). Successful-DA marginal policy in `EMOTION_TRANSITION_DA_POLICY`. No run yet. |
| 11 | β=0 control of #5 | `EmotionAwareMultiObjectiveQ` with `--beta_emo 0` | Tells us whether Q_emo is the active ingredient or whether SR=72% is from top-K + N=40 alone. |
| 12 | Matched-config GDPZero | `OpenLoopMCTS` with `--llm_prior_topk 5 --num_mcts_sims 40 --max_conv 50` | The single biggest unblock — without it, #5's 72% cannot be ranked against any GDPZero number. |

### (c) Specced in doc but not built

Each row links back to the section that describes the design in full.

| # | Method | Doc section | Effort to build | Reason it's unbuilt |
| --- | --- | --- | --- | --- |
| 13 | Bellman backup of emotion penalty up the tree | Investigation 1 | medium | Risks compounding; current local penalty is the safer first step. |
| 14 | Trajectory-based emotion reward (Δ shaping) | Investigation 2 | medium | Needs a target arc; cheaper variants tested first. |
| 15 | Learned π(e) from outcomes | Investigation 3 | small (script provided) | Hand-tuned penalty was tested and tied; learned variant a follow-up. |
| 16 | Entropy-as-exploration PUCT bonus | Investigation 4 | small | Lower priority vs the parallel-channel directions. |
| 17 | Anticipatory P(emotion \| state, DA) | Item 2 | medium (script + small training) | Tabular fallback shipped via #10; full BERT version of the model unbuilt. |
| 18 | Persona-conditioned user simulator | Item 3 | small (`data/personas/p4g_personas.json`, `scripts/calibrate_personas_p4g.py` exist) | **Highest-leverage substrate change — directly attacks the sycophancy ceiling identified in Findings §1 below.** Not yet wired into eval. |
| 19 | Toned-DA action space (Phase 1) | "Self-play RL" → Setup | medium | Prerequisite for Phases 2-3; deferred. |
| 20 | Hand-seeded tone prior (Phase 2) | "Self-play RL" → Phase 2 | small | Builds on #19. |
| 21 | Self-play RL for tone policy (Phase 3) | "Self-play RL" → Phase 3 | large (~6 weeks) | Big-bang research arc; only worth pursuing if simpler variants show a directional win. |
| 22 | Multi-objective parallel Q (Direction A) | Direction A — Multi-objective parallel Q | **BUILT** as #5 — see (a) bucket. |
| 23 | Imagination side-tree (Direction B) | Direction B — Emotion-rollout imagination side-tree | **BUILT** as #10 — see (b) bucket. |
| 24 | POMDP belief over latent persuasion state (Direction C) | Direction C — POMDP framing | large | Biggest engineering, cleanest theoretical story. HMM emission/transition fit + belief plumbing in `EmotionAwareDialogSession`. |
| 25 | Risk-sensitive / quantile MCTS (Direction D) | Direction D — Risk-sensitive / quantile MCTS | small (~30 LOC) | Lowest theoretical novelty; addresses a real flaw (mean ≠ donation rate under heavy tails). |

## Supporting infrastructure built

Substrate that makes the methods above runnable / measurable. Not "methods"
per se but worth listing so future passes know what's reusable.

| Area | What | Where |
| --- | --- | --- |
| Data mining | Empirical bonus matrix from p4g donation outcomes | `src/emotion_mining/mine_emotion_da_bonus_p4g.py` → `outputs/learned_emotion_da_bonus.json`; lives as `MINED_EMOTION_DA_BONUS` in `src/mcts/emotion_mcts.py` |
| Data mining | Tabular Markov dynamics for imagination side-tree | `src/emotion_mining/learn_emotion_transition_p4g.py` → `outputs/emotion_transition_p4g.json`; baked into `EMOTION_TRANSITION_TABLE` |
| Training | Anticipatory P(emotion \| history, DA) BERT | `scripts/train_anticipatory_emotion_p4g.py` (specced; tabular fallback ships in #10) |
| Training | Emotion-conditioned action prior BERT | `scripts/train_emotion_conditioned_prior_p4g.py` (script ready; model not yet trained) |
| Personas | Persona definitions + mixture calibration | `data/personas/p4g_personas.json` + `scripts/calibrate_personas_p4g.py` |
| Eval — end-to-end | SR / AT computation | `runners/rollout.py` + `metrics/run_metrics.py` |
| Eval — h2h | Per-turn judge with metadata sidecar (GPT-3.5 / GPT-4o / GPT-4-turbo / GPT-4) | `evaluators/run_judge.py` — writes `*_metadata.json` with counterfactual_da, last_user_emotion, agreement rate, per-transition win rate, marginal win rate |
| Logging — Tier 1 | Per-turn shaping attribution | `runners/emomcts.py:_compute_counterfactual_das` — argmax_visits / argmax_q / puct_no_bonus / puct_vanilla per turn |
| Logging | Per-DA user-emotion histogram | `runners/_common.py:dump_da_emotion_records` |
| Logging | Per-utterance emotion classification cache | `runners/_common.py:dump_emotion_records` |
| Runner integration | All emotion-aware MCTS variants in both runners | `runners/emomcts.py` + `runners/rollout.py` — flags forwarded to subclasses via `--lambda_emo`, `--c_emo_bonus`, `--alpha_realization_emo`, `--beta_emo`, `--c_imag` / `--imag_K` / `--imag_H`, `--emotion_classifier`, `--llm_prior_topk`, `--emotion_prior_model` |

## Quick read of where the project stands

- **3 methods measured against GDPZero**: 1 tied (π penalty), 1 clearly
  lost (bonus matrix v1), 1 statistically tied (multi-objective Q) with
  the matched baseline still missing.
- **7 methods built and ready to measure** (rows 6-12): largely awaiting
  compute budget plus the matched GDPZero baseline that unblocks ranking.
- **13 methods specced but unbuilt** (rows 13-25): two big-bang efforts
  (POMDP Direction C, self-play RL Phase 3) are large; the rest are
  small-to-medium. Persona-conditioned simulator (#18) is the highest
  leverage of the unbuilt list given the sycophancy ceiling described
  next.

## Key diagnostic findings from the runs

### 1. The user simulator is sycophantic

Across every emomcts run logged, the persuadee emotion distribution is:

```
happiness  ≈ 47-48 %
neutral    ≈ 39-40 %
sadness    ≈ 5-7 %
surprise   ≈ 3-4 %
fear       ≈ 2-3 %
anger      ≈ 0.1-1 %
disgust    ≈ 0.1-0.5 %
contempt   ≈ 0 %
```

**~87-88 % of user turns are positive or neutral; negative emotions are 5-11 %; anger / disgust / contempt are nearly absent.**

Implications:
- The bonus matrix's negative-emotion rows (Anger, Disgust, Contempt, Fear)
  are essentially never exercised in rollout — they could be set to anything
  and SR would be unchanged.
- The π penalty's mass falls on a near-empty bucket.
- End-to-end SR saturates near 70-75% for any reasonable planner because
  the simulator agrees to donate this often regardless of strategy.
- **The simulator is the ceiling.** We're not measuring planner quality,
  we're measuring "did the planner do enough not to break the simulator's
  default-agreeable behaviour."

### 2. The emotion classifier is miscalibrated for p4g context

HF DistilRoBERTa emotion classifier labels:
- **"sadness"** → often polite refusals ("I'm sorry, I can't donate right
  now") — semantically a NEGATIVE engagement signal in everyday text but a
  REFUSAL in p4g context. The hand-seeded bonus matrix originally rewarded
  proposition under sadness on the theory of guilt-driven engagement; the
  data shows this is wrong.
- **"anger"** → false positives on neutral statements ("I'm fine. How are
  you?"). Noise.
- **"happiness"** → genuine engagement, but also bland agreement.
- **"surprise"** → questions about effectiveness or trust.

So even the rows we DO have samples for have the wrong label semantics
relative to what the bonus matrix was theorised against. The mining script
(`src/emotion_mining/mine_emotion_da_bonus_p4g.py`) partially corrected this by
fitting from data, but the corrected matrix still lost to GDPZero.

### 3. The judge prefers softness over strategy

In the 152-turn h2h on bonus-matrix v1:
- GDPZero played `other` 74 times (49 % of moves), EmoMCTS played it 2 times.
- EmoMCTS played `proposition of donation` 64 times, GDPZero 23 times.
- Aggregate win-rate: 44.7 % for EmoMCTS.
- Slice `A=other → B=proposition`: 38.2 % win rate (EmoMCTS loses 62 %).
- Slice `A=other → B=emotion appeal`: 66.7 % win rate (EmoMCTS *wins* 67 %).

GPT-3.5 reads `other` (typically empathetic / clarifying / soft) as
*persuasive*. The judge IS the standard, and the judge prefers what
GDPZero does. EmoMCTS playing more "strategic" DAs is registering as
*worse* persuasion by this metric.

### 4. Bonus is making things worse where it fires

```
bonus_changed_action: 85/152 turns (56%)  win-rate on those = 42.4 %
overall                                  win-rate on those = 44.7 %
```

Win rate on turns where the bonus changed action is **lower** than aggregate.
Direct attribution: the shaping matrix is a net negative. With the cells
the matrix fires on removed from the comparison, EmoMCTS lands ~47 %.

### 5. Conditional slice that DOES work (kept for future re-use)

```
A=other → B=emotion appeal: win-rate 66.7 % (12W / 6L)
```

When GDPZero plays soft `other` filler and EmoMCTS substitutes emotion
appeal, the judge prefers EmoMCTS 67 % of the time. **Emotion appeal as a
substitute for filler is a real win** even though it's currently drowned
by proposition-spam losses.

### 6. MultiObjectiveQ: statistical-tie at best, structural limits at worst

`EmotionAwareMultiObjectiveQ` with β=0.7, top-K=5, N=40 sims, 50 dialogs:
SR = 72 % ±12pp. The matched-config GDPZero baseline has not been run, so
the comparison vs the historic 75 % at N=20 / 20 dialogs is broken: that
baseline has a ±19pp CI and the intervals overlap by 24pp. **Cannot rank
these planners from current data.**

Independent reading of the MultiObjectiveQ run on its own: AT=6.62 ≈
GDPZero's AT=6.6 — both planners terminate in ~7 turns. Same emotion
distribution shape as bonus-matrix runs. Suggests the variant is
producing similar trajectories with slightly different strategic content,
not a structurally different policy.

## Structural barriers identified

These are honest predictions of *why* a null result is plausible — useful
both for paper write-up and for deciding what to fix:

1. **Simulator saturation** (87-88 % positive turns). Any signal added to
   the planner is consumed by an agreeable user. To measure planner
   differentiation, lower the ceiling: harder simulator (Item 3 personas).

2. **Classifier label miscalibration**. Sadness ≠ engagement in p4g
   context. Either re-map `EMOTION_VALENCE` for p4g, fine-tune a
   p4g-specific classifier, or distil labels from a stronger model.

3. **Search-budget arithmetic**. At N=20-40 and B=13, even the donation Q
   channel has Nsa~2-5 per action. Q_emo / v_imag / bonus all sit on
   top of an already-noisy substrate. Either shrink B (top-K), grow N
   (impractical), or use sample-efficient structures (tied parameters,
   warm-start from prior).

4. **Judge / objective mismatch**. The h2h judge is myopic per-turn
   preference; it cannot see multi-turn trajectory steering, which is
   what emotion-awareness is for. End-to-end SR is the right metric in
   principle but saturated on the current simulator.

5. **Emotion as modifier, not channel**. All variants tested so far
   (π penalty, bonus, parallel Q, imagination) treat emotion as a
   correction on a donation-optimising scaffold. Direction C (POMDP
   framing — emotion as observation, belief over latent persuasion
   state) is the genuinely different approach and remains unbuilt.

## What is *not* yet ruled out

Before declaring the approach a null result, these unmeasured angles
could change the picture:

- **Matched-config GDPZero baseline** (N=40, top-K=5, n=50). Essential
  before any ranking claim is interpretable.
- **β=0 control of MultiObjectiveQ**. Tells us whether Q_emo is doing
  anything at all vs whether the SR=72% is from top-K + N=40 alone.
- **`EmotionImaginationMCTS` (Direction B)** — unmeasured. Unique
  research angle: cost-asymmetric long-horizon planning via Markov
  side-tree. Free at inference (~5 ms / node expansion).
- **Conditional slicing on rare-emotion turns** (Metric Improvement 1).
  EmoMCTS might be +20pp on negative-emotion turns and -3pp on neutral
  — that signal is invisible in aggregate SR.
- **Persona-conditioned simulators** (Item 3). Different ceiling, may
  open algorithmic differentiation.
- **Different judge** (GPT-4o, or end-to-end donation rate instead of
  per-turn preference).
- **`EmotionConditionedPriorMCTS`** (learned prior from training data) —
  unmeasured. Different mechanism (replace prior, not value), may
  sidestep the search-budget arithmetic.

## Recommended next steps (priority order)

1. **Run matched-config GDPZero baseline.** N=40, top-K=5, n=50. ~20-40h
   compute. Single highest-priority measurement — without it nothing else
   is interpretable.
2. **Run β=0 control of MultiObjectiveQ** in parallel. Same config minus
   β. Tells us whether the parallel Q channel is the active ingredient
   or the top-K is.
3. **Run `--c_imag 0.5 --imag_H 3`** smoke test of imagination side-tree
   (n=10 first, then n=50 if not crashing). Unmeasured direction with
   real theoretical novelty.
4. **Slice the existing pickles** by `last_user_emotion`, `counterfactual_da
   ` (Metric Improvement 1, ~50 LOC). Conditional attribution over
   already-collected data.
5. **Compute emotion-trajectory delta** between planners (Metric
   Improvement 4, ~80 LOC). Decomposes any SR delta into DA-choice vs
   utterance-realisation contributions.

After these five, the project has clean numbers to either ship a positive
result, write the negative-result diagnostic paper (Option A in the
strategic memo), or pivot the substrate (Option B: personas, harder
simulator, p4g-tuned classifier).

## Reading guide for future passes

- For "did variant X beat GDPZero?" — see the table above. Apples-to-apples
  comparisons require matched configs (same top-K, same N, same n
  dialogs). Anything else is noise.
- For "why X didn't work" — see structural barriers (sycophantic
  simulator + miscalibrated classifier + judge preference for softness +
  multi-channel sample efficiency).
- For "what's next" — see recommended next steps; the matched GDPZero
  baseline is the unblock.
