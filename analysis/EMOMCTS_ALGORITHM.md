# EmoMCTS — Algorithm and PUCT extension proposals

## Current EmoMCTS (after the local-penalty fix)

### Additional notation (on top of GDPZero)

- `C(u)` — emotion classifier; returns one of `{Anger, Fear, Disgust, Contempt, Sadness, Surprise, Happiness, Neutral}`
- `e(h^tr)` — predicted user emotion of the last user utterance in `h^tr`, i.e.
  `e(h^tr) = C(u_user(h^tr))`
- `π : E → ℝ` — emotion penalty function

### Equations

**eq. E.1 (penalty values):**
```
π(Anger)    = -1.0
π(Fear)     = -0.9
π(Disgust)  = -0.8
π(Contempt) = -0.6
π(Sadness)  = -0.4
π(Surprise) = π(Happiness) = π(Neutral) = 0
```

**eq. E.2 (locally blended value used in updates):**
```
v~(s^tr, h^tr) = v(s^tr) + π(e(h^tr))
```

**eq. E.3 (Q update — penalty applied LOCALLY only):**
```
Q(s^tr, a)  ←  Q(s^tr, a) + ( v~(s^tr ◦ a, h^tr_child) − Q(s^tr, a) ) / ( N(s^tr, a) + 1 )
N(s^tr, a)  ←  N(s^tr, a) + 1
```

**eq. E.4 (utterance-level mean uses the same blended value):**
```
v_h(h^tr)  ←  ( v_h(h^tr) × N_h(h^tr) + v~(s^tr, h^tr) ) / ( N_h(h^tr) + 1 )
```

**eq. E.5 (returned value to parent — leaf v, NOT blended):**
```
search(s^tr) returns v(s^tr_leaf)        ⟵ no compounding of π up the backup chain
```

PUCT is unchanged from GDPZero:
```
PUCT(s^tr, a) = Q(s^tr, a) + c_p · P(a|s^tr) · √(Σ_a N(s^tr, a)) / (1 + N(s^tr, a))
```

### Algorithm: EmoMCTS Search (current version)

```
Require:
    Generative LLM Mθ
    Emotion classifier C, penalty function π
    Dialogue history h_i until turn i
    Dialogue action space a ∈ A
    n, k, c_p, Q_0
Output: a*, u_sys*

Repeat for n searches:
    initialize root s_i^tr,  H(s_i^tr) ← {h_i}
    s^tr ← s_i^tr

    // selection
    while s^tr is not a leaf node do:
        a′ ← argmax_a PUCT(s^tr, a; c_p)
        h^tr ← sample( H(s^tr) )
        s^tr ← s^tr ◦ a′
        if |H(s^tr)| < k then:
            generate h_new ← Mθ(h^tr ◦ a′)        // sys utt + simulated user reply
            classify e_new ← C(user reply of h_new)
            attach e_new to h_new
            H(s^tr) ← H(s^tr) ∪ {h_new}
        end if
    end while
    h^tr ← sample( H(s^tr) )

    // expansion
    generate p(a|s^tr) ← Mθ(h^tr)
    s^tr.p ← p(a|s^tr)
    s^tr.Q ← Q_0
    s^tr.N ← 0

    // evaluation
    generate v(s^tr) ← Mθ(h^tr)
    v_leaf ← v(s^tr)                              // pure task value

    // backpropagation (local emotion shaping, no compounding)
    while s^tr ≠ s_i^tr do:
        e ← e(h^tr)                               // emotion at current node's last user turn
        v~ ← v_leaf + π(e)                        // eq. E.2 (uses leaf v_leaf, not stacked)
        update v_h(h^tr) with eq. E.4
        save H(s^tr) ← H(s^tr) ∪ {h^tr}
        (s^tr_parent, a) ← parent of s^tr
        update Q(s^tr_parent, a), N(s^tr_parent, a) with eq. E.3 using v~
        s^tr ← s^tr_parent
    end while

// prediction after n simulations
a*       ← argmax_a N(s_i^tr, a)
s_*^tr   ← s_i^tr ◦ a*
u_sys*   ← argmax_{u_sys} v_h( H(s_*^tr) )
return a*, u_sys*
```

### Diff vs GDPZero

The only differences from the GDPZero algorithm are:
1. Every history in `H(·)` carries a classified user emotion.
2. Updates use `v~ = v_leaf + π(e)` instead of bare `v_leaf` — but `v_leaf` itself does
   not absorb `π` on the way back up (local-only application, no compounding).

---

## Empirical results (2026-05-29, vicuna:13b, gpt-3.5-turbo judge, p4g, 152 turns)

Local-only emotion shaping (`search()` returns leaf `v`; penalty applied to immediate
`Q(s,a)` and `realizations_Vs` only) was tested with two penalty tables:

| Configuration                                                | Win | Draw | Lose | Win rate    |
| ------------------------------------------------------------ | --- | ---- | ---- | ----------- |
| **Conservative π** (Anger −1.0, Fear −0.9, Sad −0.4, …)      |  89 |   0  |  63  | **58.55%**  |
| **Persuasion-flipped π** (Sad +0.3, Fear +0.2, Neutral −0.1) |  68 |   1  |  83  |   44.74%    |
| GDPZero baseline                                             |  —  |   —  |  —   |   ~50%      |
| Pre-fix EmoMCTS (compounding penalty)                        |  —  |   —  |  —   |   ~40%      |

**Headline finding**: conservative π with the local-only fix beats GDPZero by **+8.5pp**.
Flipping the sign of Sadness/Fear, which Proposal 1 predicted would help, loses by **−5pp**
against GDPZero and **−14pp** against the conservative version.

### Hypothesis: judge-alignment, not persuasion shaping

The evaluation metric is GPT-3.5 judging utterance quality, not measured donation. GPT-3.5
penalizes the same responses that elicit Anger/Sadness in the vicuna emotion classifier —
both are LLMs trained on similar conversational priors. The conservative π therefore acts
as a **judge-alignment regularizer**: it suppresses utterances the judge would also mark
down. The persuasion-specific signed weights gamble on a tail (Sad-as-engagement) that the
judge doesn't reward.

**Honest framing for the paper**: "local-only emotion shaping with conservative
negative-emotion penalties is a quality regularizer that biases MCTS toward strategies
eliciting positive/neutral user affect, yielding +8.5pp win-rate improvement over GDPZero
under LLM-judge evaluation." Avoid claiming "improved persuasion" without behavioral
evidence (e.g., simulated `U_Donate` rate).

### Implementation note

The 58.55% run was obtained with **additive blending** (`blended_v = v + π(e)`, Q ∈
[−2, +1]) + local-only propagation. The `--lambda_emo` CLI flag is not currently wired
into `EmotionAwareDiscountQOpenLoopMCTS` (the runner constructs the class without passing
`emo_lambda`, so `search()` takes the additive `else` branch). Convex-blend results not
yet measured.

---

## How to update PUCT to actually boost persuasion via emotion

Five proposals, ordered from "smallest change" to "biggest architectural shift".

> **2026-05-29 update**: Proposal 1 is empirically refuted under LLM-judge evaluation
> (see "Empirical results" above). The conservative `π` it argued against beats the
> persuasion-flipped `π` by +14pp. Keep Proposal 1 here as a record of the (wrong)
> hypothesis; do not implement.

The remaining proposals should be re-evaluated through the same "judge alignment vs.
behavioral persuasion" lens: each makes sense under one objective or the other but not
necessarily both.

### Proposal 1 — Reshape `π` for the persuasion objective  ⚠️ refuted empirically

> **2026-05-29**: tested with the table below at 44.74% win rate vs. 58.55% for the
> conservative π. Likely cause: GPT-3.5 judge penalizes the same responses that elicit
> Sadness/Fear, so rewarding them aligns the search *against* the judge. May still help
> under behavioral evaluation (measured donation rate); has not been tested there.

The current `π` is a generic "negative emotion = bad" mapping borrowed from emotional-
support intuition. In persuasion, Sadness/Fear in the *persuadee* are engagement signals
(guilt, sympathy, urgency → donate), not failure. Flip their sign:

```
π_persuasion(Anger)     = -1.0      // outright rejection
π_persuasion(Disgust)   = -0.7
π_persuasion(Contempt)  = -0.7
π_persuasion(Sadness)   = +0.3      // guilt / sympathy = engagement
π_persuasion(Fear)      = +0.2      // urgency / concern
π_persuasion(Surprise)  = +0.1
π_persuasion(Happiness) =  0
π_persuasion(Neutral)   = -0.1      // emotional flatness = poor persuasion
```

PUCT and all other equations unchanged. Zero implementation risk — one dict edit in
`_get_emotion_penalty`.

### Proposal 2 — Emotion-AMAF bonus in PUCT

Add a third term that boosts actions which historically elicited engagement-positive
emotions at the current node. This is the persuasion analogue of RAVE / AMAF.

Define a per-`(s^tr, a)` emotion histogram (already tracked in `emotions_count`):
```
n_e(s^tr, a)   = # rollouts where action a from s^tr produced user emotion e
N_e(s^tr, a)   = Σ_e n_e(s^tr, a)
```

and an engagement weight `ω(e)` (same shape as Proposal 1's π but in `[-1, +1]`).

**eq. P2 (Emotion-AMAF bonus):**
```
E(s^tr, a) = ( Σ_e ω(e) · n_e(s^tr, a) ) / ( 1 + N_e(s^tr, a) )
```

**eq. P2-PUCT (modified PUCT):**
```
PUCT'(s^tr, a) = Q(s^tr, a)
               + c_p   · P(a|s^tr) · √(Σ_a N(s^tr,a)) / (1 + N(s^tr,a))
               + c_emo · E(s^tr, a)              ← new
```

`c_emo` starts ~ same magnitude as `c_p` (say 0.5–1.0) and can be annealed down as `N`
grows so the term acts as an early-search bias toward emotionally engaging strategies.

### Proposal 3 — Emotion-conditioned prior

Re-prompt `Mθ` at expansion with the current user emotion to bias the prior `P` toward
emotion-appropriate strategies (e.g., a "Sad" user calls for `personal story` or
`proposition of donation`, not `task related inquiry`).

```
P'(a | s^tr) ∝ P(a | s^tr) · P_emo(a | s^tr, e(h^tr))
```

`P_emo` comes from a small prompt: *"The persuadee is currently feeling [e]. From {DAs},
which strategy is most likely to land a donation?"*. Renormalize, then use `P'` in the
standard PUCT term — no other change.

This sidesteps the additive-penalty issue entirely: emotion enters as an actor-prior, not
as a value bias.

### Proposal 4 — Emotion-dependent exploration coefficient

Make `c_p` a function of current emotion. When the user is hostile
(Anger/Disgust/Contempt) the current strategy is failing → explore more. When the user is
engaged (Sadness/Fear/Surprise) → exploit.

```
c_p(e) = c_p_base · ( 1 + α · 𝟙[e ∈ {Anger, Disgust, Contempt}]
                          − β · 𝟙[e ∈ {Sadness, Fear, Surprise}] )
```

Plug into PUCT directly. No extra LLM calls.

### Proposal 5 — Emotion-trajectory reward at leaves

Replace the leaf value with a mixture of task value and an emotional-arc score over the
trajectory `h^tr`:

```
v'(s^tr) = (1 − γ) · v_task(s^tr)
         + γ       · v_arc(h^tr)
```

where `v_arc` rewards de-escalation / engagement sequences:
- `Anger → Sadness → Neutral` → high reward (cooled off, listening)
- `Neutral → Surprise → Sadness` → high reward (got attention, then moved them)
- `Neutral → Anger` → low reward (botched it)

Operationally: a small table over consecutive emotion pairs (or a 2-gram Markov reward).
No extra LLM calls; cheap to compute from `e_1, ..., e_t` already attached to every
`h^tr`.

This shifts emotion from a *per-step penalty* (today) to a *trajectory shaping reward* —
the same conceptual move that distinguishes shaped-RL from one-step-reward MCTS, and the
only one of these proposals that lets emotion influence `v` itself (so the change rides
through the standard PUCT backup rather than being a side-channel).

---

## Suggested A/B order

> **2026-05-29 revised** after the empirical run. Old order led with Proposal 1, which is
> now refuted on the judge metric. Updated order:

1. **Lock in the 58.55% result as the baseline ablation.** Re-run once with a different
   seed to confirm it isn't a 2σ fluctuation on 152 samples. Record exact configuration:
   conservative π, additive blend (`v + π(e)`), local-only propagation
   (`search()` returns `v`).
2. **Wire `--lambda_emo` into `EmotionAwareDiscountQOpenLoopMCTS`** (one-line: pass
   `emo_lambda=args.lambda_emo` at the `emomcts.py:108` construction site) and sweep
   `λ ∈ {0.3, 0.5, 0.7, 1.0}`. Goal: see whether the bounded convex blend can match the
   unbounded additive form, or whether the larger penalty magnitude is load-bearing.
3. **Dump the DA→emotion histogram** from the winning run (`_da_emotions.json`) and use
   it as a paper figure showing which strategies elicit which emotions.
4. **Proposal 2 (Emotion-AMAF bonus)** as the next algorithmic step. It is the most
   "PUCT-native" of the remaining ideas and keeps Q clean (only changes selection). Use
   conservative `ω` weights aligned with the empirical π — *not* the persuasion-flipped
   ones the original Proposal 2 description used.
5. **Proposal 5 (trajectory reward)** if you want a value-level intervention with a
   stronger paper story. Define `v_arc` so it rewards reduction of negative-emotion
   density along the trajectory.

Proposals 3 and 4 remain stand-alone alternatives, deferred until you've squeezed what
you can from #1–#2 above.

If you want to test the Proposal-1 hypothesis fairly (rather than throw it out), add a
behavioral evaluator: run the chosen action through the user simulator and record
whether it eventually produces `U_Donate`. That metric is the only way to distinguish
"judge-pleasing" from "actually persuasive."

---

## How the proposals scale to ESConv and CraigslistBargain

The five proposals scale very differently because the three tasks have **structurally
different relationships between emotion and success.**

### The structural difference

| Task                  | Success signal | Role of emotion                             | "Good" emotional outcome                                 |
| --------------------- | -------------- | ------------------------------------------- | -------------------------------------------------------- |
| Persuasion (p4g)      | `U_Donate`     | **Instrumental** — drives behavior change   | Persuadee feels Sad/Concerned → donates                  |
| Emot. Support (esc)   | `U_Solved`     | **Terminal** — emotional change *is* the task | Patient distress decreases over the dialog               |
| Negotiation (cb)      | `U_Deal`       | **Diagnostic** — signal of negotiation state | Buyer mildly content; no hostility, no obvious giveaway |

Three implications follow:

1. In **p4g**, emotion is the *means*. Static per-emotion weights are workable.
2. In **esc**, emotion is the *end*. Only the *trajectory* (delta over turns) is meaningful
   — a Sad patient at turn 5 isn't a failure if they started at Anger; today's penalty
   would mis-credit it.
3. In **cb**, emotion is *information about an adversary*. Anger from the buyer = your
   offer is bad, push elsewhere. Happiness from the buyer = you over-conceded. Neutral
   is the target.

That structure is what makes the proposals scale differently.

### Per-proposal scaling

#### Proposal 1 — Reshape `π`: poor

The whole approach assumes a fixed, absolute sign per emotion. That's a p4g-specific
assumption.

- **ESC**: Sadness is the patient's *baseline*; penalizing it punishes the algorithm for
  the user's pre-existing state. You'd need delta-based weights (Δω), not absolute ω.
- **CB**: depends on who's modeled. As seller, buyer Anger is mostly bad but transient
  anger after a counter-offer is information you *want* to see, not avoid.

A single dict per task is technically a port, but it papers over the fact that absolute-
emotion weights don't capture the trajectory in ESC at all.

#### Proposal 2 — Emotion-AMAF bonus: mechanism scales, weights don't

The selection-bias-not-value-distortion mechanism is task-agnostic and safe across all
three. But `ω(e)` is still per-task:

- **P4G**: as proposed (Sad/Fear positive, Anger/Disgust/Contempt negative).
- **ESC**: ω should reward emotion *improvements*, not states. Either redefine the AMAF
  statistic to count *transitions* (Anger→Sadness, Sadness→Neutral) instead of *absolute*
  emotions, or run Proposal 5 instead.
- **CB**: ω rewards Neutral (target), mildly penalizes hostility (Anger/Disgust), and
  *mildly penalizes Happiness* (over-concession signal). The weights are non-monotone in
  valence — a quirk worth reporting.

The minimal port adds an `engagement_weights` field to each `TASKS[...]` entry and reads
it in `_emotion_amaf_bonus`. One dict per task. Story is "selection bias toward task-
appropriate emotional reactions" — works in the paper as a generic framework even if the
weight tables differ.

#### Proposal 3 — Emotion-conditioned prior: best zero-shot generalization

This is the strongest scaling proposal. You're handing the domain adaptation to the LLM:

- **P4G**: *"The persuadee is feeling [Sad]. From {DAs}, which is most likely to land a
  donation?"*
- **ESC**: *"The patient is feeling [Anger]. From {DAs}, which is most likely to bring
  them toward relief?"*
- **CB**: *"The buyer is feeling [Disgust]. From {DAs}, which is most likely to keep them
  at the table?"*

Same code path, same MCTS, the LLM's pretrained world model supplies the task-specific
reasoning. No hand-tuned ω tables. Cost: 1 extra LLM call per node expansion —
meaningful on vicuna:13b but free on GPT.

This is the cleanest "one method, three tasks" story for a paper, at the cost of being
the most LLM-dependent.

#### Proposal 4 — Emotion-dependent `c_p`: cheapest cross-task port

The meta-rule "hostile reaction → explore other strategies; engaged reaction → exploit"
is generic enough to land in all three:

- **P4G**: hostile = {Anger, Disgust, Contempt}; engaged = {Sad, Fear}.
- **ESC**: hostile = {Anger, Disgust}; engaged = {Sad, Fear, Surprise} (all signs the
  patient is opening up).
- **CB**: hostile = {Anger, Disgust, Contempt}; engaged = {Surprise, Neutral} (deal-
  closing zone).

Just two emotion sets per task. No `Q` modification, no extra LLM calls, no new tracking.
Easy port if the priority is "show the method generalizes without bespoke tuning."

#### Proposal 5 — Trajectory reward: scales **best** for ESC, naturally for CB

This is where the framework wins.

- **ESC**: trajectory shaping is the *natural* objective. Define `v_arc(h^tr)` as monotone
  reduction in patient distress level — e.g., a small dict over `(e_t, e_{t+1})` pairs
  with positive reward for moves toward Neutral/Happiness. ESC's `U_Solved` is rare and
  far-future; `v_arc` densifies the signal, which is exactly the standard MCTS trick when
  terminal rewards are sparse.
- **CB**: `v_arc` rewards de-escalation patterns (Anger→Neutral) *and* penalizes giveaway
  patterns (Neutral→Happiness). Captures both "don't lose the buyer" and "don't over-
  concede" in one term.
- **P4G**: as proposed (Neutral → Surprise → Sad → Donate is the canonical arc).

Trades implementation cost for explanatory power. Best paper story for a multi-task
evaluation — same `v_arc` machinery, three task-specific reward tables, single shared
MCTS code.

### Per-task recommendations

| Task                | Try first | Why                                                                                                                                |
| ------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Persuasion (p4g)    | **#2**    | Cleanest single-ablation story (`c_emo` sweep) and you already have the `emotions_count` plumbing.                                  |
| Emotional Support   | **#5**    | Trajectory shaping matches the task's defining property — distress reduction. Static per-emotion weights (#1, #2) fundamentally miss this. |
| Negotiation (cb)    | **#4**    | The exploit/explore dichotomy on buyer hostility is the natural negotiation heuristic. Cheap to add, low risk of bias.            |

If you want a single approach across all three for the paper, **#3 (emotion-conditioned
prior)** is the most defensible "one method, three tasks" pitch — the per-task adaptation
is delegated to the LLM, so the *algorithmic* contribution is uniform.

### What this implies for the codebase

Two practical notes if you go cross-task:

1. **Add `emo_esc` and `emo_cb` task variants** in `runners/_common.py`. The factory
   `_emotion_aware_variant` already exists; you just need an
   `EmotionAwareEmotionalSupportGame` and `EmotionAwareCBGame` mirroring
   `EmotionAwarePersuasionGame`. The emotion classifier base class is already task-
   pluggable via `user_role`.
2. **Push task-specific emotion config into `TASKS`**, not into
   `EmotionAwareOpenLoopMCTS`. Each entry gets an `engagement_weights` (for #2),
   `hostile_set` (for #4), or `v_arc_table` (for #5) field, and the MCTS class reads it
   from `game.engagement_weights` etc. Keeps the MCTS task-agnostic — important because
   otherwise you'd be writing `EmotionAwareOpenLoopMCTSForP4G`, `…ForESC`, `…ForCB`,
   which is the wrong axis to fork on.

---

## Why we can't recursively return the "discounted" value

Five reasons, in order of how badly each one breaks the algorithm. The short answer is
that what the pre-fix code did wasn't *discounting* in the RL sense — it was **additive
accumulation with no contraction factor**, and additive accumulation breaks several MCTS
invariants at once.

### Setting up the trace

Recall the pre-fix `search()`:

```
search(s_0):                                 # root call
    a_0 ← argmax PUCT
    next ← s_1
    v ← search(s_1)                          # recurse
    blended ← v + π(e_1)                     # e_1 = emotion at s_1
    Q(s_0, a_0) ← avg with blended
    return blended                            # ← the bug

search(s_1):
    a_1 ← argmax PUCT
    next ← s_2
    v ← search(s_2)
    blended ← v + π(e_2)
    Q(s_1, a_1) ← avg with blended
    return blended

search(s_2):                                  # leaf
    return v_leaf
```

What `Q(s_0, a_0)` receives on this rollout is:
```
v_leaf + π(e_2) + π(e_1)
```
On a deeper rollout (depth 5) it would receive
`v_leaf + π(e_5) + π(e_4) + π(e_3) + π(e_2) + π(e_1)`.

Now the five problems.

### 1. The PUCT magnitude contract is violated

PUCT
```
PUCT(s,a) = Q(s,a) + c_p · P(a|s) · √(Σ N) / (1 + N(s,a))
```
balances `Q` (a mean in `[-1, +1]` for GDPZero) against the exploration term
(≈ `c_p · P · O(√N / N)`, typically `~0.1–0.3` with `c_p = 1`). The constant `c_p` is
*chosen relative to that Q range*.

If `Q` is now a sum of `k` penalties, its lower bound is `−k · max|π|` — unbounded in
tree depth. Exploration term magnitude is unchanged. As soon as one branch accumulates a
few penalties, its `Q` is so far below `Q_0 = 0` for untried branches that PUCT just
keeps picking untried actions forever. The search is no longer "balance exploit and
explore" — it's "avoid anything that was ever visited."

This isn't fixable by retuning `c_p`, because the magnitude of "how many penalties
accumulated" *depends on which path you took* — it's per-state, not constant.

### 2. Double-counting: the same emotion is credited to every ancestor

The emotion at depth `d` is `π(e_d)`. It modifies `Q(s_{d−1}, a_{d−1})` (correct —
that's the action that produced `e_d`). But it also modifies `Q(s_{d−2}, a_{d−2})`,
`Q(s_{d−3}, a_{d−3})`, …, `Q(s_0, a_0)` via the returned `blended`. The same emotion
gets *blamed* on every ancestor decision.

For MCTS credit assignment to be coherent, the responsibility for an outcome at depth
`d` should sit at depth `d−1` (the action that caused it). Propagating `π` up is the
algorithmic equivalent of blaming a parent for what a grandchild did — you lose the
per-edge resolution that PUCT needs.

### 3. Subtree depth becomes a hidden penalty multiplier

`Q(s_0, a_0)` is the mean over rollouts of:
- length-2 rollouts that contributed `v_leaf + π(e_2)`  → 1 penalty
- length-5 rollouts that contributed `v_leaf + π(e_5) + … + π(e_1)` → 5 penalties

Actions whose subtrees tend to terminate quickly get *less* penalty regardless of
whether the terminations are good or bad. The algorithm now prefers actions that lead
to **early termination**, which in p4g means `no-donation` paths get implicit favor —
exactly backwards.

This is a particularly nasty failure mode because it can't be diagnosed by looking at
individual Q values; it only shows up across rollouts of different lengths.

### 4. Variance explodes inside `Q`

Same edge `(s_1, a_1)`, visited twice:

| Rollout | Path beyond       | Contributed return                                       |
| ------- | ----------------- | -------------------------------------------------------- |
| 1       | terminates at s_2 | `v_leaf + π(e_2)`                                        |
| 2       | continues to s_5  | `v_leaf + π(e_2) + π(e_3) + π(e_4) + π(e_5)`             |

`Q(s_1, a_1)` is averaging numbers from different return distributions. The running mean
is no longer estimating a single underlying value; it's estimating "expected sum of
remaining penalties along this rollout," which depends on rollout depth and exploration
choices below `s_1`. The whole point of MCTS is that more visits → tighter estimate.
Here, more visits add more variance.

### 5. The leaf `v` is a *value-function estimate*, not a reward

`v_leaf = Mθ(h^tr)` is meant to be `V(h)` — the LLM's estimate of "probability this
dialog state leads to success." It's already an *expected long-run return*. Backing up
`v + π(e_d) + π(e_{d−1}) + …` is mathematically incoherent: you're summing a value
estimate (long-horizon) with shaping rewards (per-step), without any Bellman structure
tying them together. There's no consistent MDP under which this is a valid return.

### So why is the word "discount" misleading?

In RL, **discounting** means
```
G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + …       (γ ∈ [0, 1))
```
The multiplicative `γ^k` is what makes the sum bounded: `|G| ≤ r_max / (1 − γ)`. The
pre-fix code does
```
G_t = v_leaf + π(e_{t+1}) + π(e_{t+2}) + π(e_{t+3}) + …
```
no `γ`. No contraction. Additive without bound. So **this isn't discounting — it's an
unbounded shaping sum**, and the standard MCTS machinery isn't set up to absorb it.

To make recursive propagation actually correct, you'd need to commit to one of two
reformulations:

- **Bellman shaping**: declare `π(e_t)` an immediate per-step reward `r_t`, declare
  `v_leaf` a leaf value `V(s_leaf)`, pick a discount `γ`, and back up
  `Q(s, a) ← r + γ · search(s')`. Now `Q` has a defined meaning and PUCT can be retuned.
  This is a different algorithm.
- **Leaf-level shaping (Proposal 5)**: compute the whole trajectory-shaping reward
  `v_arc(h^tr)` once, at the leaf, and let it ride through the *existing* GDPZero backup
  unchanged. The shaping signal still affects every ancestor, but as a single bounded
  number, not as a sum that grows with depth.

The local-penalty fix we shipped takes the cleanest middle road: treat `π(e_child)` as
**a one-edge local shaping reward** applied only to the immediate parent's `Q(s, a)` and
`v_h`, and let the standard MCTS return carry only the bounded leaf value. Each edge
gets credited with the emotion *it* caused, no double-counting, Q stays in `[-1, +1]`,
PUCT keeps its magnitude balance.
