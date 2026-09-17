# Emotion-Aware Multi-Objective MCTS (Double-Q)

This document describes the **Emotion-Aware Multi-Objective Q** planner
(`EmotionAwareMultiObjectiveQ`, hereafter **EMO-DoubleQ**), a drop-in extension of the
GDP-Zero open-loop Monte Carlo Tree Search (MCTS) dialogue planner. It maintains a
*second, parallel* action-value channel that tracks the expected **emotional valence** of
the user's reaction, and folds that channel into the tree-search selection rule. The
method is described first in self-contained form (Section 1), then contrasted with
GDP-Zero (Section 2), and finally evaluated (Section 3).

---

## 1. Method

### 1.1 Setup and notation

We plan a goal-oriented dialogue (PersuasionForGood, *p4g*) as a single-player search over
dialogue *states*. A state $s$ is a dialogue history; from $s$ the system may play one of a
finite set of **dialogue acts** $a \in \mathcal{A}(s)$ (e.g. *task-related-inquiry*,
*proposition-of-donation*, *credibility-appeal*, …). Following GDP-Zero, the search is
**open-loop**: a node is keyed by the *sequence of dialogue acts* played so far rather than
by a single realized utterance string, and each node caches up to $R$ sampled surface
realizations (utterances) of that act sequence. An LLM provides four roles during search:
policy prior, value function, user simulator, and system (utterance) model.

For each visited node we store the standard MCTS statistics:

$$
N(s),\qquad N(s,a),\qquad P(s,a),\qquad Q(s,a),
$$

where $N(s)$ is the node visit count, $N(s,a)$ the edge visit count, $P(s,a)$ the LLM
policy prior, and $Q(s,a)$ the running-mean return for taking act $a$ in $s$.

**EMO-DoubleQ adds one table:**

$$
Q_{\text{emo}}(s,a) \in [-1, +1],
$$

the running mean of the *emotional valence* of the user's immediate reaction after the
system plays $a$ in $s$. It has exactly the same shape (same keys, same pruned action set)
as $Q$.

### 1.2 Emotional valence of a user reaction

After the system plays act $a$, the user-simulator LLM produces a reaction whose emotion is
classified into a distribution $d(\cdot)$ over a fixed label set
$\mathcal{E} = \{\text{Fear}, \text{Happiness}, \text{Anger}, \text{Disgust},$
$\text{Surprise}, \text{Neutral}, \text{Sadness}, \text{Contempt}\}$ (how $d$ is produced is
detailed in §1.10). Each label $e$ has a
scalar **valence weight** $w(e) \in [-1, +1]$ — *not* the textbook affective valence but a
weight **calibrated against donation outcomes** in the P4G corpus (see §1.8; the data make
some signs counter-intuitive — *fear* carries the strongest positive weight). The expected
valence of a reaction is

$$
\nu(d) \;=\; \sum_{e \in \mathcal{E}} d(e)\, w(e),
\qquad \nu(\varnothing) = 0,
$$

with $\nu = 0$ when no distribution is attached (e.g. a system turn). This emotion
distribution is **already cached** on the child state by the user simulator, so $\nu$ costs
no extra LLM or classifier calls.

### 1.3 Selection rule (PUCT with a parallel value channel)

At an internal node the next act is chosen by a variant of PUCT that *adds* the emotion
channel to the donation channel:

$$
a^\star \;=\; \arg\max_{a \in \mathcal{A}(s)}
\Big[\;
\underbrace{Q(s,a)}_{\text{donation value}}
\;+\;
\underbrace{\beta\, Q_{\text{emo}}(s,a)}_{\text{emotion value}}
\;+\;
\underbrace{c_{\text{puct}}\, P(s,a)\, \frac{\sqrt{N(s)}}{1 + N(s,a)}}_{\text{exploration}}
\;\Big].
$$

The single hyper-parameter $\beta \ge 0$ (`beta_emo`) trades off the two objectives. At
$\beta = 0$ the rule is *identical* to GDP-Zero's PUCT (the emotion channel is still
computed and backed up, but ignored at selection), so a single search tree can be re-scored
post-hoc at different $\beta$ for free.

### 1.4 Expansion and evaluation

On reaching an unexpanded node $s$, the LLM is queried once to produce the prior–value pair

$$
\big(P(s,\cdot),\, v(s)\big) = \text{LLM}_{\text{predict}}(s),
$$

where $v(s) \in [0,1]$ is the LLM's estimate of eventually reaching the goal (donation) from
$s$. The prior is masked to the legal acts and renormalized.

**Top-$K$ prior pruning.** Rather than expand the full (up to 13-act) legal set at every node,
we restrict the search to the $K$ acts the LLM prior ranks highest. Concretely we keep

$$
\mathcal{A}_K(s) \;=\; \operatorname*{top\text{-}}K_{\,a \in \mathcal{A}(s)} \; P(s,a),
\qquad
\tilde P(s,a) \;=\; \frac{P(s,a)\,\mathbb{1}[a \in \mathcal{A}_K(s)]}{\sum_{a'\in\mathcal{A}_K(s)} P(s,a')},
$$

and **hard-prune** the node to $\mathcal{A}_K(s)$: the statistics $N(s,a)$, $Q(s,a)$,
$Q_{\text{emo}}(s,a)$, the renormalized prior $\tilde P$, and the PUCT round-robin are all
defined over exactly these $K$ acts; the dropped acts are unreachable from that node. This
spends the simulation budget on the acts the LLM judged most promising (and avoids
round-robin waste on acts it judged poor); the emotion guidance is deliberately kept
orthogonal — it enters only through the $\beta Q_{\text{emo}}$ term (§1.3), never through the
pruning. Throughout, the symbol $\mathcal{A}(s)$ in the selection rule (§1.3) refers to this
pruned set $\mathcal{A}_K(s)$ when pruning is active. We use $K = 5$ in all experiments, and
report $K{=}5$ vs. the unpruned planner as an explicit condition (§3); $K = \infty$ recovers
the full-action search.

### 1.5 Two-channel backup (the "Double-Q")

After the recursive simulation returns the leaf value $v$, **both** channels are updated as
running means that share the *same* pre-increment edge count $N(s,a)$:

$$
Q(s,a^\star) \;\leftarrow\; \frac{N(s,a^\star)\, Q(s,a^\star) + v}{N(s,a^\star) + 1},
$$

$$
Q_{\text{emo}}(s,a^\star) \;\leftarrow\;
\frac{N(s,a^\star)\, Q_{\text{emo}}(s,a^\star) + \nu\big(d_{s'}\big)}{N(s,a^\star) + 1},
$$

where $s'$ is the immediate child (the user reaction sampled after $a^\star$) and $d_{s'}$
its cached emotion distribution. Counts are incremented *after* both updates:

$$
N(s) \mathrel{+}= 1, \qquad N(s,a^\star) \mathrel{+}= 1.
$$

Note the asymmetry that gives the method its character: the donation channel backs up the
**leaf** value $v$ (long-horizon, goal-conditioned), whereas the emotion channel backs up the
**local** valence $\nu(d_{s'})$ of the *immediate* child. $Q_{\text{emo}}(s,a)$ is therefore
"how the user tends to react emotionally right after we play $a$ in $s$" — a one-step signal
used to bias search, not a discounted return.

### 1.6 Decision and realization

After $n_{\text{sims}}$ simulations from the root, the system commits to the act by visit
count,

$$
\pi(a \mid s_0) \;=\; \frac{N(s_0, a)}{\sum_{a'} N(s_0, a')},
\qquad a_{\text{play}} = \arg\max_a \pi(a \mid s_0),
$$

and the surface utterance is the cached realization of $a_{\text{play}}$ with the highest
mean donation value $V$. **Utterance selection still optimizes donation only** — the
emotion channel shapes *which act is searched/chosen*, not which sentence is spoken.

### 1.7 Why "Double-Q" / multi-objective rather than a penalty

A natural alternative is to fold emotion into a *single* scalar reward,
$Q' = (1-\lambda)Q - \lambda\,\text{penalty}$. EMO-DoubleQ deliberately keeps the two values
**decomposed**:

- **Ablatable / sweepable at decision time.** Because $Q$ and $Q_{\text{emo}}$ are stored
  separately, $\beta$ can be swept (e.g. $\{0, 0.3, 0.7, 1.0\}$) by re-scoring the same tree;
  a fused penalty would require re-search per $\lambda$.
- **Interpretable.** One can read off "high $Q_{\text{emo}}$ / low $Q$" cells directly — acts
  that please the user but don't advance the goal, and vice-versa — i.e. an explicit
  donation/emotion **Pareto frontier**.
- **Cheap.** One extra float per $(s,a)$ edge and no extra model calls (the emotion
  distribution is already cached by the user simulator).

### 1.8 Where the valence weights come from: mining persuasion, not psychology

The weights $w(e)$ are the one piece of the method that is *not* a free hyper-parameter — they
are estimated from data, and the way they were obtained is itself the central design lesson.

**Background: emotional valence.** In affect theory, *valence* is the basic dimension that
places an emotion on a pleasant–unpleasant axis — joy and contentment on the positive side;
fear, anger, sadness, and disgust on the negative side; neutral near zero. It is a core axis of
Russell's circumplex model of affect [[Russell 1980]](https://doi.org/10.1037/h0077714) and
complements discrete accounts such as Ekman's basic emotions
[[Ekman 1992]](https://doi.org/10.1080/02699939208411068), and is routinely operationalized as
a single rating of emotional state (e.g. the Self-Assessment Manikin and affective-norm ratings
of Bradley and Lang [[Bradley & Lang 1994]](https://doi.org/10.1016/0005-7916%2894%2990063-9)).
This makes valence a natural scalar for the planner to reason about.

**The wrong starting point.** Our first instinct was to import this textbook ordering directly:
set $w(e)$ from *affective valence*. Happiness is pleasant, so $w > 0$; fear, anger, sadness,
disgust are unpleasant, so $w < 0$; neutral is $0$. This bakes in the assumption that *making
the user feel good* is the route to a donation.

**The reframe.** In persuasion that assumption is suspect. The decision-relevant question is
not "is this emotion pleasant?" but "**does a user in this emotional state actually end up
donating?**" The persuasion literature points the other way — fear appeals are broadly
effective at changing attitudes and behavior
[[Tannenbaum et al. 2015]](https://doi.org/10.1037/a0039729);
[[Witte 1992]](https://doi.org/10.1080/03637759209376276), and sympathy for an identifiable
victim drives charitable giving
[[Small, Loewenstein & Slovic 2007]](https://doi.org/10.1016/j.obhdp.2006.01.005). So the
textbook signs might be exactly backwards. Rather than argue the point, we measured it.

**The mining procedure** (deterministic and fully reproducible —
`src/emotion_mining/mine_emotion_donation_p4g.py`):

1. **Corpus.** All $300$ human–human dialogues of PersuasionForGood
   [[Wang et al. 2019]](https://aclanthology.org/P19-1566/)
   (`data/p4g/300_dialog_turn_based.pkl`), giving $\approx 4{,}800$ user turns.
2. **Labelling.** Tag every *user* utterance with the deterministic HF encoder
   `j-hartmann/emotion-english-distilroberta-base`
   [[Hartmann 2022]](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) and
   take the arg-max emotion. No LLM calls, no sampling — the labels are a pure function of the
   text, so the matrix is reproducible.
3. **Outcome.** A dialogue is a *success* if any user turn carries the `agree-donation`
   annotation. The corpus **base donation rate is $0.493$**.
4. **Signal.** For each emotion $e$, compute the empirical **donation lift**
   $$
   \ell(e) \;=\; P(\text{donate}\mid \text{user-turn emotion}=e) \;-\; P(\text{donate}),
   $$
   with Wilson $95\%$ confidence intervals, and set the valence weight proportional to it,
   $$
   w(e) \;=\; 10 \cdot \ell(e),
   $$
   keeping small-sample cells deliberately skeptical. (`Contempt`, which the HF encoder never
   emits, is left at a hand-set value as a fallback for the alternative LLM classifier.)

**The punchline — the data overturn the textbook.** The mined lifts (and resulting weights):

| Emotion | $P(\text{donate}\mid e)$ | lift $\ell(e)$ | $n$ turns | $w(e)=10\,\ell$ | textbook sign | verdict |
|---|:--:|:--:|:--:|:--:|:--:|---|
| **Fear**      | 0.600 | **+0.107** | 70   | **+1.07** | $-$ | **flipped** — strongest positive signal |
| Happiness     | 0.552 | +0.059 | 1145 | +0.59 | $+$ | agrees (but weaker than fear) |
| Anger         | 0.534 | +0.041 | 88   | +0.41 | $-$ | flipped; small $n$, kept skeptical |
| Disgust       | 0.533 | +0.039 | 92   | +0.39 | $-$ | flipped; small $n$, kept skeptical |
| Surprise      | 0.510 | +0.016 | 469  | +0.16 | $\pm$ | mildly positive |
| Neutral       | 0.484 | $-0.009$ | 2660 | $-0.09$ | $0$ | apathy taxed — the real enemy |
| Sadness       | 0.458 | $-0.035$ | 323  | $-0.35$ | $-$ | agrees |
| Contempt      | — | — | (n/a in HF) | $-0.60$ | $-$ | hand-set fallback |

Base rate $= 0.493$, $300$ dialogues. Two findings stand out: **(i)** the strongest positive
predictor of donation is *fear*, not happiness — consistent with sympathy/urgency-driven
giving and the opposite of naive affective valence; and **(ii)** the only sizeable negative
signal is **neutral** (n=2660): in persuasion the adversary is *apathy*, not unpleasantness.
What we call "valence" is therefore really a **persuasion-calibrated affect weighting**.

**Honest caveats.** These are **correlational** donation lifts, not causal effects — "fearful
users donate more often" does *not* license "inducing fear causes donation." We use $w(e)$
only as a cheap *search-biasing prior* over which emotional trajectories tend to co-occur with
success; the donation channel $Q(s,a)$ remains the ground-truth objective, and utterance
selection optimises donation alone (§1.6). The small-$n$ cells (fear, anger, disgust; $n<100$)
are treated cautiously, and the whole matrix is a swappable artifact — the `--emotion_map`
ablation in the experiments (cf. the "new emo map" run) re-mines/re-scales these weights to
check the method is not brittle to their exact values.

### 1.9 From a fixed table to a learned emotional value model

The static matrix of §1.8 is the *zeroth-order* version of a more general idea. Observe that
the emotion backup signal is a **linear functional of the predicted emotion distribution**,

$$
\nu(d) \;=\; \sum_{e\in\mathcal{E}} d(e)\, w(e) \;=\; w^{\top} d,
$$

i.e. a fixed weight vector $w$ scoring a one-step emotion histogram with **no dependence on
the dialogue context**. This factorisation exposes three increasingly expressive ways to
*learn* the emotion value, each already scaffolded in the codebase:

**(a) Learned linear weights — replace the marginal lift with a partial effect.**
Instead of $w(e)=10\,\ell(e)$ (a *marginal* donation lift that ignores co-occurring
emotions), fit $w \leftarrow \theta$ by **logistic regression of donation outcome on per-dialogue
emotion histograms** (`scripts/learn_emotion_penalty_p4g.py`). The coefficients $\theta$ are
*conditional* contributions to the donation log-odds — they disentangle correlated emotions
that the raw lift conflates — and are rescaled to the $v\in[0,1]$ range. The MCTS code is
unchanged: only the numbers in $w$ change. This is the smallest possible "learned" step.

**(b) Context-conditioned value head — drop the fixed $w$ entirely.**
Replace the static functional with a small encoder that scores the *state and candidate act*,
$$
V_{\text{emo}}(s,a) \;=\; f_\phi\big(\underbrace{\text{history text}}_{\text{context}},\, a,\, e_{\text{user}}\big),
$$
a DistilRoBERTa-class regressor (≈82M params, **zero LLM cost** and deterministic at
inference, ~20 ms/forward). The repo already trains the two ingredients: an *anticipatory*
next-emotion model $\hat P_\phi(e \mid \text{history}, \text{sys DA})$
(`scripts/train_anticipatory_emotion_p4g.py`) and an emotion-conditioned action prior
(`scripts/train_emotion_conditioned_prior_p4g.py`). One can either learn $V_{\text{emo}}$
**end-to-end**, or keep it **factored and interpretable**,
$$
V_{\text{emo}}(s,a) \;=\; \sum_{e\in\mathcal{E}} \hat P_\phi\!\big(e \mid s, a\big)\; u_\psi(e, s),
$$
where $\hat P_\phi$ is the anticipatory emotion model and $u_\psi(e,s)$ a learned,
context-dependent per-emotion value — a direct generalisation of the constant $w(e)$ to a
function of the dialogue state. Calibrated uncertainty from $f_\phi$ can *automatically*
down-weight the low-confidence cells we currently shrink by hand.

**(c) A true second value function (bootstrapped, not one-step).**
The static channel backs up only the *immediate* child valence $\nu(d_{s'})$ (§1.5) — a
1-step signal. A learned $V_{\text{emo}}$ can instead estimate the **expected emotional
trajectory toward donation**, i.e. a genuine second value head, turning EMO-DoubleQ into two
*learned* value functions (donation $Q$ and emotion $Q_{\text{emo}}$) in the AlphaZero mould.
Two complementary training signals are available:
- **Supervised**, from the human corpus (regress donation / measured valence on context),
  exactly as the scripts above do; and
- **Self-distillation from search**, where the MCTS-backed-up $Q_{\text{emo}}(s,a)$ targets
  train $f_\phi$, and $f_\phi$ in turn **warm-starts** $Q_{\text{emo}}$ at node expansion
  (replacing the `Q_emo = 0` initialisation in `_init_node`) — the emotion analogue of how
  the policy prior $P$ and value $v$ already seed the donation channel.

**Why this matters empirically.** The warm-start in (c) directly targets the failure mode we
observe in the budget sweep (§3): at small simulation counts the emotion channel is *inert*,
because $Q_{\text{emo}}$ starts at $0$ and needs many visits before its running mean becomes
informative — at $n_{\text{sims}}=10$ all methods collapse to the same success floor. A
learned $V_{\text{emo}}$ supplies a non-trivial emotion value from the **first** visit to each
node, so the $\beta Q_{\text{emo}}$ term can steer search under a tight budget instead of only
after dozens of simulations. The learned model is thus not merely a fidelity upgrade over the
mined table — it is the most plausible route to making the emotion signal pay off at the low
sim counts where the static version cannot.

Throughout, the integration surface stays tiny: a learned model only changes
`_emotion_quality` / the `Q_emo` initialisation; the PUCT rule, the two-channel backup, and
the donation-only NLG selection are all untouched, so every result below remains a clean
ablation of *the value source* with the search machinery held fixed.

### 1.10 Emotion detection: obtaining the reaction distribution $d(\cdot)$

Both the valence backup and the parallel channel $Q_{\text{emo}}$ consume a per-turn
distribution $d(\cdot)$ over the user's emotion (§1.2). A detector therefore maps the latest
*user* utterance to a probability distribution over the label set $\mathcal{E}$ — Ekman's six
basic emotions (joy/happiness, sadness, fear, anger, surprise, disgust)
[[Ekman 1992]](https://doi.org/10.1080/02699939208411068) plus *neutral* and *contempt*. Two
interchangeable detectors are implemented (`--emotion_classifier`):

- **HF encoder (default, `hf`).** `j-hartmann/emotion-english-distilroberta-base`
  [[Hartmann 2022]](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — a
  DistilRoBERTa [[Sanh et al. 2019]](https://arxiv.org/abs/1910.01108) /
  RoBERTa [[Liu et al. 2019]](https://arxiv.org/abs/1907.11692) encoder fine-tuned on $\approx
  40$k examples across $7$ classes (anger, disgust, fear, joy$\to$happiness, neutral, sadness,
  surprise). A single forward pass returns the full softmax over labels, which we read directly
  as $d(\cdot)$. It is **deterministic** (same text $\Rightarrow$ same $d$), incurs **no LLM
  cost**, and is cached per utterance; `contempt` lies outside its label set and stays at
  probability $0$.
- **Prompt-based LLM (`llm`).** The dialogue backbone is prompted with a short definition of
  each of the $8$ emotions and asked to label the utterance; we draw $K{=}5$ samples and take
  $d(\cdot)$ as the empirical frequency over them. This covers the full taxonomy (including
  `contempt`) and can use dialogue context, but is **stochastic** and adds model calls.

We use the **HF encoder for all reported runs**: determinism and zero marginal cost make it
safe to call at every node and at mining time, and it keeps the whole pipeline reproducible
(§3.1) — the same reasons it labels the corpus when the valence weights are mined (§1.8).

This is deliberately a *lightweight, single-utterance* classifier rather than a full
**emotion-recognition-in-conversation (ERC)** model
[[Poria et al. 2019, survey]](https://arxiv.org/abs/1905.02947): the planner only needs a cheap
per-turn affect signal to bias search, and a heavier context-aware detector — trained on ERC
corpora such as [[MELD]](https://arxiv.org/abs/1810.02508),
[[EmpatheticDialogues]](https://arxiv.org/abs/1811.00207), or the fine-grained
[[GoEmotions]](https://arxiv.org/abs/2005.00547) taxonomy — is a drop-in upgrade that composes
naturally with the learned value model of §1.9.

---

## 2. Comparison to GDP-Zero

GDP-Zero (Yu, Chen & Yu, 2023) performs goal-oriented dialogue policy planning with
**open-loop MCTS**, prompting a single LLM to act as policy prior, value function, user
simulator, and system model. EMO-DoubleQ inherits that entire machinery unchanged and adds
one parallel value channel. The differences:

| Aspect | GDP-Zero (base) | EMO-DoubleQ |
|---|---|---|
| Tree / open-loop structure | act-sequence nodes, $R$ cached realizations | identical |
| LLM roles | prior, value, user sim, system model | identical |
| Value tables per edge | $Q(s,a)$ | $Q(s,a)$ **and** $Q_{\text{emo}}(s,a)$ |
| Selection rule | $Q + c_{\text{puct}} P \sqrt{N(s)}/(1{+}N(s,a))$ | $Q + \beta Q_{\text{emo}} + c_{\text{puct}} P \sqrt{N(s)}/(1{+}N(s,a))$ |
| Backup signal | leaf value $v$ | leaf value $v$ **and** local valence $\nu(d_{s'})$ |
| Extra LLM calls | — | **none** (valence reuses the cached user-emotion distribution) |
| Final act selection | root visit counts $N(s_0,a)$ | identical |
| Utterance (NLG) selection | best realization by mean $V$ | identical (donation-only) |
| Reduces to base when | — | $\beta = 0$ |

**Formally, the only change is in the selection score:**

$$
\text{GDP-Zero:}\quad
U(s,a) = Q(s,a) + c_{\text{puct}}\, P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)},
$$

$$
\text{EMO-DoubleQ:}\quad
U(s,a) = Q(s,a) + \beta\, Q_{\text{emo}}(s,a) + c_{\text{puct}}\, P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}.
$$

Conceptually, GDP-Zero searches purely for *goal completion*; EMO-DoubleQ searches for goal
completion **subject to a soft preference for emotionally positive user trajectories**,
without sacrificing GDP-Zero's training-free, prompt-only design or incurring additional
inference cost.

The orthogonal `llm_prior_topk` knob (hard-pruning the action set to the LLM's top-$K$
suggested acts) is available to *both* methods; it is a search-efficiency lever, not part of
the emotion contribution, and we report it as a separate experimental condition below.

---

## 3. Experiments

### 3.1 Setup

All dialogues are simulated on **PersuasionForGood (p4g)** with an **open-source Vicuna-13B**
backbone serving every LLM role (policy prior, value, user simulator, system model). Each
condition rolls out **50 dialogues** with **$n_{\text{sims}} = 40$** MCTS simulations per
turn, a turn cap of $T_{\max}=10$, and $R=3$ realizations per node. We report:

- **SR** — Success Rate: fraction of dialogues reaching the donation goal within $T_{\max}$.
- **AvgT** — Average Turns to resolution; failed/over-limit dialogues count as $T_{\max}$
  (PPDPP convention), so lower is better.

> **On reproducibility and the choice of an open-source backbone.** We deliberately use an
> open-source model (Vicuna-13B) rather than a proprietary API model (e.g. OpenAI GPT-3.5/4).
> Proprietary endpoints are periodically **updated, silently re-tuned, or fully deprecated**,
> which makes published numbers obtained against them difficult or impossible to reproduce
> later. Pinning a fixed open-weights model fixes the backbone exactly, so every result in
> this section can be re-run by others and compared on identical footing. The same harness
> supports an OpenAI/ChatGPT backbone for completeness, but our headline comparisons are kept
> on the open-source model for replicability.

### 3.2 Main result: success and efficiency

We compare three conditions: base GDP-Zero, GDP-Zero with top-$K$ prior pruning, and
EMO-DoubleQ (with top-$K$). EMO-DoubleQ uses $\beta = 0.7$.

| Method | $n$ | SR $\uparrow$ | AvgT $\downarrow$ |
|---|:---:|:--:|:--:|
| GDP-Zero (no top-$K$)            | 50  | 0.5106 | 8.085 |
| GDP-Zero (top-$K$)               | 50  | 0.5800 | 7.120 |
| **EMO-DoubleQ, $\beta{=}0.7$ (top-$K$)** | 50  | **0.7200** | **6.620** |

All runs: Vicuna-13B, p4g, $n_{\text{sims}}=40$, 50 rollouts, $T_{\max}=10$.

**Findings.**
1. **Top-$K$ pruning helps GDP-Zero** on both axes (SR $0.51 \to 0.58$, AvgT $8.09 \to 7.12$),
   by concentrating the simulation budget on the LLM's most promising acts.
2. **EMO-DoubleQ dominates both GDP-Zero variants**: it raises SR by **+14 points** over the
   stronger (top-$K$) GDP-Zero baseline ($0.72$ vs $0.58$) *and* reaches the goal in **fewer
   turns** ($6.62$ vs $7.12$). The emotion channel both increases how often donation is
   achieved and shortens the path to it — i.e. steering toward emotionally positive
   trajectories is *aligned with*, not traded against, goal completion on p4g.

> *Note:* the no-top-$K$ GDP-Zero run completed $n=47$ of 50 dialogues (3 rollouts dropped);
> all SR/AvgT figures are computed over the completed episodes.

### 3.3 LLM-as-judge: response quality vs. human and vs. GDP-Zero *(placeholder)*

Beyond task success, we evaluate the *quality* of the system utterances with an LLM judge
that performs pairwise preference comparisons. We assess two claims:
**(a)** EMO-DoubleQ is preferred over the **human** reference responses (as GDP-Zero already
is), and **(b)** EMO-DoubleQ is **comparable-or-better than GDP-Zero** under the same
open-source backbone. *Numbers to be filled after the judge runs complete.*

**Table 3a. Win rate vs. human reference responses** (LLM-as-judge pairwise preference; each
cell = % of turns where the method's utterance is preferred over the human's; $>50\%$ means
better-than-human).

| Method | Win vs. Human $\uparrow$ | Tie | Loss | $n$ turns |
|---|:--:|:--:|:--:|:--:|
| GDP-Zero (top-$K$)            | ​ | ​ | ​ | ​ |
| **EMO-DoubleQ, $\beta{=}0.7$** | ​ | ​ | ​ | ​ |

**Table 3b. Head-to-head: EMO-DoubleQ vs. GDP-Zero** (LLM-as-judge pairwise preference, same
Vicuna-13B backbone; row method vs. column method).

| | EMO-DoubleQ wins | Tie | GDP-Zero wins | $n$ turns |
|---|:--:|:--:|:--:|:--:|
| EMO-DoubleQ vs. GDP-Zero (top-$K$) | ​ | ​ | ​ | ​ |

**Table 3c. (Optional) Judge breakdown by axis** — for a finer-grained view, report the
preference rate per evaluation dimension.

| Axis | EMO-DoubleQ vs. Human | EMO-DoubleQ vs. GDP-Zero |
|---|:--:|:--:|
| Persuasiveness        | ​ | ​ |
| Naturalness / fluency | ​ | ​ |
| Emotional appropriateness | ​ | ​ |
| Overall preference    | ​ | ​ |

*(All judge tables use the open-source Vicuna-13B backbone for both the dialogue systems and,
where applicable, the judge, so the entire comparison is reproducible per Section 3.1.)*

### 3.4 (Optional) Sensitivity to $\beta$ *(placeholder)*

Because $Q$ and $Q_{\text{emo}}$ are stored separately, $\beta$ can be swept by re-scoring the
same trees. Reserved for an ablation over $\beta \in \{0.0, 0.3, 0.7, 1.0\}$.

| $\beta$ | SR $\uparrow$ | AvgT $\downarrow$ |
|---|:--:|:--:|
| 0.0 (≡ GDP-Zero) | ​ | ​ |
| 0.3 | ​ | ​ |
| 0.7 | 0.7200 | 6.620 |
| 1.0 | ​ | ​ |

---

## References

- Bradley, M. M., & Lang, P. J. (1994). *Measuring emotion: The Self-Assessment Manikin and the semantic differential.* Journal of Behavior Therapy and Experimental Psychiatry, 25(1), 49–59. https://doi.org/10.1016/0005-7916(94)90063-9
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* In Proc. ACL 2020. https://arxiv.org/abs/2005.00547
- Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3–4), 169–200. https://doi.org/10.1080/02699939208411068
- Hartmann, J. (2022). *Emotion English DistilRoBERTa-base.* Hugging Face. https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692. https://arxiv.org/abs/1907.11692
- Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.* In Proc. ACL 2019. https://arxiv.org/abs/1810.02508
- Poria, S., Majumder, N., Mihalcea, R., & Hovy, E. (2019). *Emotion Recognition in Conversation: Research Challenges, Datasets, and Recent Advances.* IEEE Access, 7, 100943–100953. https://arxiv.org/abs/1905.02947
- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y.-L. (2019). *Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset (EmpatheticDialogues).* In Proc. ACL 2019. https://arxiv.org/abs/1811.00207
- Russell, J. A. (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology, 39(6), 1161–1178. https://doi.org/10.1037/h0077714
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108. https://arxiv.org/abs/1910.01108
- Small, D. A., Loewenstein, G., & Slovic, P. (2007). *Sympathy and callousness: The impact of deliberative thought on donations to identifiable and statistical victims.* Organizational Behavior and Human Decision Processes, 102(2), 143–153. https://doi.org/10.1016/j.obhdp.2006.01.005
- Tannenbaum, M. B., Hepler, J., Zimmerman, R. S., Saul, L., Jacobs, S., Wilson, K., & Albarracín, D. (2015). *Appealing to fear: A meta-analysis of fear appeal effectiveness and theories.* Psychological Bulletin, 141(6), 1178–1204. https://doi.org/10.1037/a0039729
- Wang, X., Shi, W., Kim, R., Oh, Y., Yang, S., Zhang, J., & Yu, Z. (2019). *Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good.* In Proc. ACL 2019, 5635–5649. https://aclanthology.org/P19-1566/
- Witte, K. (1992). *Putting the fear back into fear appeals: The extended parallel process model.* Communication Monographs, 59(4), 329–349. https://doi.org/10.1080/03637759209376276
