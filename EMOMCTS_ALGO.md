# EmoMCTS — the algorithm as implemented

Read from the code on **2026-09-08**, branch `paper-fixes` (HEAD `2245e24` plus the
uncommitted working tree). Every claim below carries a `file:line` so it can be re-checked
rather than trusted.

> Supersedes `analysis/EMOMCTS_ALGORITHM.md` for the *current* design. That document
> describes the earlier local-penalty formulation (a hand-set penalty `π(e)` blended into the
> value) which the parallel `Q_emo` channel replaced; it is still useful as a record of the
> proposals that were considered, but it is not what runs.

---

## 1. Where it sits

```
OpenLoopMCTS                 (GDP-Zero, mcts/mcts.py)
  └─ EmotionAwareOpenLoopMCTS      (emotion_mcts.py:17)   emotion bookkeeping
       └─ EmotionAwareMultiObjectiveQ (emotion_mcts.py:340) the parallel Q_emo channel
```

`EmotionAwareMultiObjectiveQ` is what both runners instantiate — `runners/emomcts.py:193`
(dataset replay) and `runners/rollout.py:244` (self-play). The GDP-Zero skeleton is intact:
open-loop tree, realization caching, PUCT, one LLM call per node supplying both the policy
prior and the leaf value. Everything emotional is additive on top of it.

## 2. The open-loop tree (inherited)

A node is keyed by its **system dialog-act prefix only** (`_to_string_rep`,
`emotion_mcts.py:27-33`):

```
"greeting__credibility appeal__proposition of donation"
```

Not the utterances, not the user turns. A node is therefore an *action sequence*, and many
concrete dialogues collapse onto it — which is the point, because the user simulator is
stochastic. Each node caches up to `max_realizations` (default 3) concrete sessions;
selection draws one uniformly (`_sample_realization`, `:77-79`) before transitioning. A
child prefix that already holds `max_realizations` is reused instead of paying for another
pair of LLM turns (`_get_next_state`, `:108-119`), which is where the search gets its
cache-hit savings.

## 3. One simulation

```text
search(state):                                          # emotion_mcts.py:497
  if game.get_dialog_ended(state) == 1.0:  return 1.0    # donation reached

  if node unseen:                                        # EXPAND — _init_node, :35 / :448
      prior, v = player.predict(state)                   # ONE call: prior AND leaf value
      P, Q←Q_0, Ns, Nsa initialised
      Q_emo←0, M2_emo←0, emo_valences←[] on the same action set
      optional hard-prune to --llm_prior_topk by argsort(P)      # :57-76
      return v

  else: cache this realization                           # _add_new_realizations

  a* = argmax_a  Q[s][a]
              + beta_emo * (Q_emo[s][a] - emo_risk_lambda * sigma_emo[s][a])
              + cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])      # _calculate_uct, :483

  state      = sample_realization(s)                     # open-loop draw
  next_state = get_next_state(state, a*)                 # sys utt + user reply + emotion
  v          = search(next_state)                        # RECURSE

  Q[s][a*]     ← running mean with v                     # from the LEAF
  Q_emo[s][a*] ← Welford mean with z                     # from the IMMEDIATE CHILD
  Ns++, Nsa[s][a*]++                                     # after BOTH, sharing the old Nsa
  return v
```

`cpuct` is fixed at 1.0 (`runners/emomcts.py:181`).

### 3.1 The selection rule, term by term

The rule is AlphaZero's PUCT — GDP-Zero's own comment calls it "a variant of PUCT"
(`mcts/mcts.py:276`) — with one term inserted. Written as exploitation + bias + exploration:

```
score(a) =  Q[s][a]                                            (1) exploitation
         +  beta_emo * (Q_emo[s][a] - lambda * sigma_emo[s][a]) (2) emotion bias
         +  cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])        (3) exploration
```

**(1) `Q[s][a]`** — the running mean of the leaf values returned by every simulation that
passed through this edge. Initialised to `Q_0 = 0.0`, so an action that has never been tried
contributes nothing here.

**(3) the exploration term** — `P[s][a]` is the LLM policy prior, so an action the prior likes
is tried *early* even with no visits; `1/(1 + Nsa[s][a])` decays that pull as the edge
accumulates visits; `sqrt(Ns)` re-inflates it as the *parent* is visited more, so a node that
is being explored heavily keeps checking its unvisited children. Net effect: the prior
dominates the first few visits at a node, `Q` dominates thereafter. `cpuct` sets where the
crossover happens, and at 1.0 with `Q ∈ [-1,+1]` it is roughly "after a handful of visits".

Two consequences of that shape worth being explicit about:

- **The first action selected at any node is `argmax_a P[s][a]`.** With `Ns = 0` the code
  substitutes `1e-8` (`:484`), so `explore = 1e-4` for every action; all `Q` are equal at
  `Q_0`, all `Q_emo` are 0, and the tie is broken entirely by the prior. At
  `--num_mcts_sims 20` spread over a 5-to-7-action set, a node is visited few enough times
  that the prior is doing much of the work — which is why the prior's quality (§8) matters
  more here than it would at AlphaZero-scale simulation counts.
- **`Q_0 = 0.0` is optimistic for a reward in `[-1,+1]`.** An untried action is scored as
  average rather than bad, so exploration is already encouraged by the initialisation, on top
  of term (3).

**(2) the emotion bias** sits on the **exploitation** side, not inside the prior. That is
deliberate and it is what makes `beta_emo` interpretable: `Q_emo` is a mean of `ν` values in
`[-1,+1]`, the same scale as `Q`, so `beta_emo` reads directly as "how many units of donation
value is one unit of emotional valence worth". This is also why `VALENCE_SCALE = 10` is chosen
as it is (§5.2) — it puts `w(e)` on the scale that makes that reading true. Putting the same
signal into `P` instead would have made its influence decay with visits, which is not what a
value-like quantity should do.

### 3.2 Reading the risk term

`emo_risk_lambda * sigma_emo[s][a]` is a **penalty in valence units, subtracted from the
emotional mean**. (Why it exists at all is §9.4; this is what it *means*.)

**`sigma_emo[s][a]`** is the sample standard deviation (ddof=1) of the individual `z` values
backed up into *that one edge* — not across the tree, not across actions. It is derived at read
time from the Welford accumulator, `sqrt(M2_emo[s][a] / (Nsa[s][a] - 1))`
(`emotion_mcts.py:459-468`). Since every `z = ν(d) ∈ [-1,+1]`, `sigma_emo` carries **the same
units as `Q_emo`**, and answers: *across the simulations that played `a` from `s`, how much did
the user's emotional reaction vary?*

Two structural properties follow from that definition:

- **It is always ≥ 0**, so for `lambda > 0` the term is always a penalty, never a bonus.
- **It is exactly 0 below two visits** — `welford_variance` returns 0.0 for `n < 2`, and `n`
  here is `Nsa[s][a]` (`:459-464`). An edge is therefore never penalised before it has
  evidence, so `lambda` cannot suppress exploration of untried actions; it can only demote
  actions that have already proved erratic.

**`emo_risk_lambda`** is a unitless multiplier: how many standard deviations of spread to
charge against the mean. `Q_emo - lambda * sigma_emo` is the classic mean-variance (Markowitz)
form — in bandit terms a *lower* confidence bound, the mirror image of UCB. UCB is optimistic
about uncertainty; this is pessimistic about it. At `lambda = 1` an action is scored at roughly
its "one sigma below average" emotional outcome rather than its average one.

**Effect on selection.** The term only changes the ranking when two actions differ in `sigma`.
Given equal `Q_emo`, the more erratic action is demoted — an action that reliably produces mild
positive affect can outrank one that averages the same by alternating delight and fury.

**How it composes with `beta_emo`.** The term sits *inside* the bracket:

```
beta_emo * (Q_emo[s][a] - emo_risk_lambda * sigma_emo[s][a])
```

so `beta_emo` scales the emotion channel as a whole against `Q`, while `lambda` trades mean
against spread *within* that channel. They are nested, not competing.

**Choosing `lambda`, with a caution.** The measured distribution (`FREEZE_NOTES.md:178-195` —
stub classifier, so read this as scale, not as truth):

```
sigma_emo:      mean 0.1178   median 0.1040   p90 0.2180   max 0.4389
|Q_emo|:        mean 0.0826   max 0.2879
sigma/|Q_emo|:  median 1.43   p90 9.10
```

The spread is typically **larger than the mean it adjusts**. So `lambda = 1` would not be a
mild correction: the penalty would routinely exceed `Q_emo` and flip its sign, turning the
emotion channel into a near-pure variance penalty rather than a risk-adjusted valence.
Something in the 0.1–0.3 range is where the term modulates the mean instead of overwhelming
it. Worth fixing a value from the grid's real `sigma_emo` distribution rather than from these
stub numbers.

**What `sigma_emo` actually mixes.** Because the tree is open-loop *and* samples among up to
`max_realizations` system utterances per edge (§2), the spread at an edge combines genuine
heterogeneity in how users react to that action sequence with variance over *which utterance
happened to be realised*. Only the first is what `lambda` is meant to price. If the term is
ever swept, that decomposition is worth doing first — otherwise a high-`lambda` run partly
penalises actions whose realizations were diverse rather than actions whose outcomes were.

### 3.3 The asymmetry between the two backups is the core design choice

`Q` is a leaf value propagated up the recursion. `Q_emo` is **local** — it is the immediate
child's user reaction, never the leaf's (`emotion_mcts.py:534-540`). So `Q_emo[s][a]` reads
as *"how did the user emotionally react when we played `a` from `s`"*, which is what a
selection bias wants; it is deliberately not a discounted future emotional return.

## 4. The emotion channel

1. **Classify.** `EmotionAwarePersuasionGame.get_next_state` classifies the user's reply and
   attaches the **full softmax** over the 8 emotions — Happiness, Sadness, Fear, Anger,
   Surprise, Disgust, Contempt, Neutral (`emotion_classifiers/llm_emotion.py:11-19`). Backend
   is `hf` (`j-hartmann/emotion-english-distilroberta-base`, deterministic, no LLM cost) or
   `llm`. The distribution rides on the state, so the search never re-classifies.

2. **Score.** `ν(d) = Σ_e d(e)·w(e)` — `_emotion_quality` (`:407-413`), bounded in [−1,+1],
   and 0.0 when no distribution is attached (SYS turns, placeholders).

3. **Weights.** `EMOTION_VALENCE_MINED` (`:320-329`), the `soft` table and the deployed
   default. Mined from the 300 annotated P4G dialogues by
   `emotion_mining/mine_emotion_donation_p4g.py --soft`: `w(e) = 10 · lift(e)`, with the lift
   shrunk toward the 0.493 base rate by α=50 pseudo-observations so thin cells cannot
   dominate.

   | emotion | w(e) | n_eff | note |
   |---|---:|---:|---|
   | Happiness | +0.54 | 1064 | the only cell whose CI excludes the base rate |
   | Fear | +0.24 | 120 | straddles base — was +0.62 under argmax |
   | Disgust | +0.13 | 203 | straddles base |
   | Anger | +0.11 | 187 | straddles base |
   | Surprise | +0.09 | 589 | straddles base |
   | Neutral | −0.07 | 2294 | straddles base — a slight tax on apathy |
   | Sadness | −0.14 | 390 | straddles base |
   | Contempt | −0.60 | — | HF never emits it; hand value, for the LLM-classifier fallback |

   Soft assignment credits every emotion its posterior mass, which matches what `ν` consumes;
   mining by argmax fitted `w(e)` on a different functional of Φ than deployment reads. The
   `argmax` table is retained selectable as the ablation (`--emo_valence_table argmax`).

4. **Signal.** `_emotion_signal` (`:426-446`):
   - `--emo_signal level` (default) — `z = ν(d_child)`
   - `--emo_signal delta` — `z = (ν(d_child) − ν(d_parent)) / 2`, the /2 keeping `z` in
     [−1,+1] so `beta_emo` means the same thing in both arms. A missing parent distribution
     gives `z = 0.0`; it deliberately does *not* fall back to the level value, which would
     silently mix the two signals inside one run.

5. **Accumulate.** Welford (`welford_update`, `:221-237`) keeps the mean *and* `M2_emo`, so
   `σ_emo` is derived at read time and never stored (`sigma_emo`, `:466-468`). The mean line
   is algebraically identical to the running mean it replaced — that identity is what makes
   the change value-preserving. `emo_valences` additionally keeps every individual `z` in
   visit order; it is write-only, read by nothing in selection or backup, and exists so the
   frozen subtree log can ship `per_step_valences`.

## 5. Where `w(e)` comes from — the mining pipeline

Nothing in `src/emotion_mining/` runs during search. It runs offline, and the planner consumes
only the fitted constants, transcribed by hand into `EMOTION_VALENCE_MINED`
(`emotion_mcts.py:320`). The package holds two estimators of the same question — *which user
affect co-occurs with task success?* — and **only the first is live**:

| module | estimand | conditions on | consumed by |
|---|---|---|---|
| `mine_emotion_donation_p4g.py` | `w(e)` — per-emotion valence | the **current** turn | `EMOTION_VALENCE_MINED` — **live** |
| `mine_emotion_history_donation_p4g.py` | `ν(c)` — score of a trajectory | the **session so far** | not wired in (research) |

### 5.1 Three stages

```text
CORPUS → SEQUENCES                          mining_corpus.py:97-111
  for each dialog D in data/p4g/300_dialog_turn_based.pkl:
      utts ← D["dialog"][*]["ee"]           # persuadee side only, blanks dropped
      y    ← any tag == "agree-donation" in D["label"][*]["ee"]
      → EmotionSequence(dialog_id, emotions, succeeded=y, distributions)

LABEL                                        mining_corpus.py:138-196
  each utterance → HF classifier posterior, memoised to disk
  argmax mode  : build_emotion_cache               → one emotion per turn
  --soft mode  : build_emotion_distribution_cache  → the full 8-way softmax

TALLY → WEIGHTS                              mine_emotion_donation_p4g.py:203-270
```

### 5.2 The estimator

**The donation annotation is the entire supervision signal.** There is no emotion-side target
anywhere in the fit: `y ∈ {0,1}` is the human `agree-donation` tag on the persuadee side of a
dialogue, and `w(e)` is a function of nothing but those labels and the classifier's posteriors.
Note the level mismatch this forces — **features are per turn, the label is per dialogue**. A
turn inherits the outcome of the whole conversation it sits in, so a happy turn in a dialogue
that ended in a donation counts as a success even if it occurred before anything persuasive
happened. That is what makes every number here co-occurrence rather than effect, and it is also
why the intervals in §5.3 need reading with care.

Base rate over exactly the sessions the tallies cover, so every lift is measured against its
own population (`mine_emotion_donation_p4g.py:473`):

```
p₀ = (# sessions with y=1) / (# sessions)                       = 0.493 on the annotated 300
```

Soft per-turn accumulation — every user turn contributes its posterior mass to **every**
emotion (`tally_per_turn_soft`, `:203-218`; `Tally.add_soft`, `:123-135`):

```
for each user turn t, for each emotion e:
    T(e) += d_t(e)                 # effective sample size, a float
    S(e) += d_t(e) · y
```

Then, per emotion (`Tally.shrunk_lift` / `.weight`, `:144-164`):

```
p̂(e)        = S(e) / T(e)
lift(e)      = p̂(e) − p₀                                        # reported, not used
p_shrunk(e)  = (S(e) + α·p₀) / (T(e) + α)                        # α = 50
w(e)         = 10 · (p_shrunk(e) − p₀)                           # VALENCE_SCALE = 10, 2dp
```

`p_shrunk` is a Beta prior centred on the base rate: a cell with no evidence gets lift exactly
0, and thin cells are pulled toward "indistinguishable from the corpus average" in proportion
to how thin they are. `--alpha 0` recovers the raw lifts. `VALENCE_SCALE = 10` puts `w` in
roughly [−1,+1] — the range `Q_emo` expects — so `beta_emo` stays the only magnitude knob.

`contempt` gets no mined weight because the HF encoder never emits it; the −0.60 in the table
is `CONTEMPT_FALLBACK_WEIGHT` (`:92`), a hand value that only matters under
`--emotion_classifier llm`.

### 5.3 Wilson intervals — why they are used, and what they are used for

Every cell also carries a 95% Wilson score interval (`Tally.wilson_ci`, `:166-179`). It is
**reported, never fitted**: no interval enters `w(e)`. Shrinkage and the interval attack the
same problem — thin, unstable cells — from opposite ends. Shrinkage changes the number;
Wilson changes only how much a reader is entitled to believe it.

**Why not the textbook interval.** The Wald form `p̂ ± z·√(p̂(1−p̂)/n)` fails in exactly the
regime this corpus forces. It is symmetric around `p̂`, so it runs outside `[0,1]`; its width
is driven by the *observed* `p̂`, so it collapses to zero as `p̂` approaches 0 or 1; and its
real coverage at small `n` is well below the nominal 95%. The dominant-emotion table has a
sadness cell with one dialogue at `p̂ = 1.0`. Wald returns `[1.0, 1.0]` there, an interval of
zero width that excludes the base rate and would be flagged as the strongest finding in the
run. Wilson returns `[0.21, 1.00]`, which is the honest answer for `n = 1`.

**What Wilson does instead.** It inverts the score test: rather than assuming the standard
error at `p̂`, it solves `|p̂ − p| = z·√(p(1−p)/n)` for `p` and keeps both roots. In closed
form, with `z = 1.96`:

```
centre = (p̂ + z²/2n) / (1 + z²/n)
margin = (z / (1 + z²/n)) · √( p̂(1−p̂)/n + z²/4n² )
CI     = [centre − margin, centre + margin]        # clipped to [0,1]
```

The centre is itself a shrunk estimate — `n` pseudo-observations pulled toward 0.5 — so the
interval stays inside `[0,1]`, stays non-degenerate at `p̂ = 0` or `1`, and is asymmetric when
`p̂` sits near an edge. This is the standard small-`n` binomial choice and needs no
distributional assumption beyond the Bernoulli one.

**The one thing it decides.** `is_significant` (`:176-179`) is true when the interval excludes
the base rate `p₀`, and that flag is the star in the printed table and the
`ci_excludes_base_rate` field in the JSON. On the production run, one cell out of seven earns
it:

| emotion | `n_eff` | `P(donate)` | 95% Wilson CI | excludes `p₀ = 0.493` |
|---|---:|---:|---|:--:|
| happiness | 1064 | 0.550 | [0.520, 0.580] | **yes** |
| surprise | 589 | 0.503 | [0.463, 0.543] | no |
| neutral | 2294 | 0.486 | [0.465, 0.506] | no |
| disgust | 203 | 0.509 | [0.441, 0.577] | no |
| anger | 187 | 0.508 | [0.437, 0.579] | no |
| fear | 120 | 0.527 | [0.438, 0.614] | no |
| sadness | 390 | 0.478 | [0.429, 0.527] | no |

This is the sentence §5.7 rests on. Without the flag a reader sees `w(fear) = +0.24` and reads
an effect; with it they see a cell whose donation rate is consistent with the corpus average.

**Three caveats, all of which make the intervals look better than they are.**

1. **Under `--soft`, `n` is not a count.** `T(e)` is summed posterior mass, so `n_eff = 120`
   for fear means "120 turn-equivalents", not 120 turns. The Wilson formula treats it as a
   count of independent Bernoulli trials. The arithmetic carries over unchanged, but the
   interval is an approximation twice over.
2. **The trials are clustered and not independent.** Turns are the unit of tally, dialogues are
   the unit of the label, so every turn in one conversation shares one `y`. Roughly 4847 turns
   come from 300 dialogues, which means the effective number of independent observations is far
   closer to 300 than to 4847 and the per-turn intervals are correspondingly too narrow. The
   clustering-free cross-check is the dominant-emotion table, one observation per dialogue:
   there happiness is `37` dialogues at `p̂ = 0.62` with CI `[0.46, 0.76]`, which **contains**
   the base rate. Happiness clears the bar per turn and does not clear it per dialogue. Read
   the per-turn flag as the weakest of the two claims.
3. **No multiplicity correction.** Seven cells are tested at 95% against the same base rate, so
   under a global null the chance that at least one cell stars anyway is about 30%, not 5%.
   Nothing in the pipeline adjusts for this, which is a further reason the channel is gated
   behind `beta_emo`.

### 5.4 Two decisions worth separating

**Soft assignment** (adopted 2026-09-06) fixes a functional mismatch, not a statistical one.
Algorithm 2 line 7 labels a turn `argmax_e Φ(e|u)` and credits that one emotion; deployment
scores `ν(d) = Σ_e d(e)·w(e)` over the full softmax. Mining by argmax therefore fitted `w` on
a different functional of `Φ` than the planner reads. Total mass is unchanged (4847
turn-equivalents either way) but it redistributes from dominant cells to thin ones:

| | fear | anger | disgust | happiness | neutral |
|---|---:|---:|---:|---:|---:|
| argmax `n` | 70 | 88 | 92 | 1145 | 2660 |
| soft `n_eff` | 120 | 187 | 203 | 1064 | 2294 |

The largest weight change is fear, **+0.62 → +0.24**. The old value was an argmax artefact:
those 70 turns were the ones where fear happened to win a close posterior, and their 0.600
donation rate regresses to 0.527 once fear's partial mass across all turns is counted.

**Shrinkage** addresses instability, and is a property of the corpus rather than of any split.
Raw lifts on the thin cells swing wildly under resampling — mining on 200 of the 300 dialogs
moves `w(fear)` from **+1.07 to +0.01** — while none of those cells has a CI excluding the base
rate. Only happiness does.

### 5.5 Production invocation

```bash
python src/emotion_mining/mine_emotion_donation_p4g.py --soft \
    --exclude_ids data/p4g/mining_exclude_ids.txt --assert_no_exclusions \
    --out outputs/mining_final_soft_all300.json
```

Recorded provenance in that JSON: `n_sessions = 300`, `n_turns = 4847`, `assignment = "soft"`,
`shrinkage_alpha = 50.0`, `base_donation_rate = 0.4933`, `holdout_first = 0`,
`n_exclude_ids_supplied = 100`, `n_excluded_by_id = 0`, `n_sessions_before_exclusion = 300`,
`assert_no_exclusions = true`.

### 5.6 Leakage

`w(e)` is mined from all 300 annotated dialogues, so the evaluation must not be drawn from
them. Two ways to guarantee it, with very different costs:

| approach | mining corpus | max \|Δw\| |
|---|---|---|
| `--holdout_first 100` (positional holdout) | 200 dialogues | **0.62** (fear +0.62 → 0.00) |
| move the eval set to the non-annotated half | **all 300** | **0.00** |

The second is adopted. The rollout eval set is the first 100 of the 717 **non-annotated**
PersuasionForGood dialogues (`build_p4g_rollout_evalset.py` → `data/p4g/rollout_evalset_nonannotated.jsonl`),
disjoint from the annotated 300 by construction, so mining keeps the full corpus. This costs
the evaluation nothing: p4g self-play dialogues carry `scenario=()`, `game.init_dialog()`
starts from an empty session, and the runner never replays corpus text — it uses the dialogue
id only as an episode label and to look up that participant's persona.

`--exclude_ids` is passed on the production run as a **standing assertion**: it must remove 0
dialogues. `--assert_no_exclusions` makes that a hard failure — if the eval set is ever rebuilt
from the annotated half, mining exits before writing any output.

The JSON records both sides of it: `n_exclude_ids_supplied` is the length of the id list (100),
`n_excluded_by_id` is how many dialogues it actually removed (0), and
`n_sessions_before_exclusion` is the corpus size before the filter (300).

### 5.7 What the numbers do and don't license

Every value is a **correlational lift**. "Fearful users donate more often" does not license
"inducing fear causes donation", and the mining script says so in its own docstring. Only
happiness has a CI excluding the base rate; every other cell is consistent with chance however
large its point estimate looks. This is why the whole channel is gated behind `beta_emo` and
why the table is described as a weak search-biasing prior rather than a measured effect.

The same script also mines diagnostics that do **not** feed `w(e)` — per-dialogue dominant
emotion, the 8×8 transition matrix, first→last shift, and the "persuasion needs sympathy"
transition-class test. Those stay on argmax labels so they remain comparable across runs.
`--task esc` mines the same statistics against an internally-defined ESConv outcome, since
ESConv ships no outcome annotation at all.

## 6. Where emotion deliberately does *not* act

| channel | emotion? | where |
|---|---|---|
| PUCT selection bias | **yes — the only place** | `_calculate_uct`, `:483-495` |
| policy prior `P` | no | `_format_history_for_topk` strips emotions on purpose |
| leaf value `v` | no | `_build_value_messages` carries none |
| utterance choice at inference | no | `_update_realizations_Vs` tracks donation `v` only, `:544-546` |

This keeps `beta_emo` a single clean knob and the A/B uncontaminated. It is also the point of
divergence from DialogXpert, where the emotion history goes into the *action-proposal prompt*
and nowhere else.

## 7. Flags and defaults

| flag | default | effect at the default |
|---|---|---|
| `--beta_emo` | **0.0** | the emotion channel is **off**; reduces to GDP-Zero open-loop MCTS |
| `--emo_risk_lambda` | 0.0 | `Q_emo − 0·σ_emo` is bit-identical to `Q_emo` (no branch, no epsilon) |
| `--emo_signal` | `level` | `z = ν(d_child)` |
| `--emo_valence_table` | `soft` | the mined table in §4 |
| `--emotion_classifier` | `llm` | (calibration runs use `hf`) |
| `--num_mcts_sims` | 20 | inherited |
| `--max_realizations` | 3 | inherited |
| `--Q_0` | 0.0 | inherited |
| `cpuct` | 1.0 | fixed in the runner config |

Everything emotional is **instrumented but inert at defaults**. `Q_emo`, `M2_emo`, `σ_emo`
and the raw `per_step_valences` tape are computed and logged regardless of `beta_emo`, so the
grid yields the variance data without a second pass.

## 8. Two properties worth knowing before reading results

**The prior is the weak link.** Measured on live states (Llama-3.1-8B, 2026-09-08):

- under `--llm_prior_topk 5` the ranking call is near-constant across states — 17 of 24
  states returned the identical ordered list, top-1 was `proposition of donation` 20/24 — and
  its entropy is fixed by construction, since `topk_prior_probs` assigns harmonic weights
  `1/(i+1)` by rank (`players/p4g_players.py:547-556`);
- under `--logit_scoring prior` the prior is a behavioral clone, concentrating 55–90 % of its
  mass on `other`/`greeting` (H/H_max 0.547, top-1 0.632 over 8 states).

`P` enters only through the exploration term, so a bland or state-invariant prior puts the
burden on `cpuct = 1.0` for `Q` and `beta_emo·Q_emo` to overcome it. Worth confirming before
attributing a β sweep's outcome to the emotion channel.

**Failure is not terminal inside the tree.** `search` returns early only on
`get_dialog_ended(state) == 1.0`. A `−1.0` — turn cap, no-donation, stall — does not cut the
recursion, so the tree keeps extending past the failure. This is inherited verbatim from
GDP-Zero (`mcts/mcts.py:71` tests `> 0` as well), not introduced here, but it means `−1.0`
reaches `Q` only through the leaf value estimator and never as a backed-up terminal.

---

## 9. What changed relative to the committed code

Everything above describes the **working tree**. The last commit on `paper-fixes`
(`2245e24`) already contains `EmotionAwareMultiObjectiveQ`, the parallel `Q_emo` channel and
the top-K prune; the uncommitted delta is the FREEZE_NOTES Part 1 / 1.5 / 2 work. Confirmed by
`git show HEAD:src/mcts/emotion_mcts.py` — 328 lines there against 547 now.

### 9.1 The selection rule

```diff
  # HEAD 2245e24 — emotion_mcts.py:262-271
- score(a) = Q[s][a] + beta_emo * Q_emo[s][a]
-          + cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])

  # working tree — emotion_mcts.py:483-495
+ score(a) = Q[s][a] + beta_emo * (Q_emo[s][a] - emo_risk_lambda * sigma_emo[s][a])
+          + cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])
```

That is the only change to the formula. Terms (1) and (3) are untouched, `cpuct` is unchanged,
and `--beta_emo` already defaulted to `0.0` in the runner at HEAD, so the default configuration
is unchanged too.

**At the default `emo_risk_lambda = 0.0` the new expression is bit-identical to the old**, not
merely close: `0.0 * finite = 0.0` and `x - 0.0 == x` under IEEE-754, including the
`Q_emo = -0.0` case. That is why it is written with no `if lambda == 0` branch — a branch would
create a second code path that could drift, whereas the algebraic reduction cannot.
`tests/test_emo_channel_freeze.py` pins it with `.hex()` equality against the verbatim
pre-change formula over 50 randomised 13-action edge-statistic sets, plus a companion test
that `lambda > 0` *does* move high-`sigma` edges strictly downward, so the flag is not silently
inert.

### 9.2 The backup

```diff
  # HEAD — plain running mean, level signal, fixed table
- emo_v = self._emotion_quality(next_state.predicted_distribution())
- self.Q_emo[s][a] = (nsa_old * self.Q_emo[s][a] + emo_v) / (nsa_old + 1)

  # working tree — Welford, selectable signal
+ emo_v = self._emotion_signal(state, next_state)
+ self._update_emo_channel(s, a, emo_v, nsa_old)     # mean + M2_emo + emo_valences
```

`welford_update`'s mean line is algebraically identical to the running mean it replaced
(`emotion_mcts.py:221-237`), which is what makes the swap value-preserving; the added `M2_emo`
is what `sigma_emo` is derived from at read time.

### 9.3 Everything added, and whether it can change a result

| addition | where | changes results at defaults? |
|---|---|---|
| `emo_risk_lambda` term in PUCT | `:490-503` | **no** — bit-identical at `0.0` |
| Welford `M2_emo` + `sigma_emo` | `:221-247`, `:459-481` | **no** — mean unchanged, `sigma` read-only |
| `emo_valences` raw tape | `:401-405` | **no** — write-only, read by nothing |
| `--emo_signal level\|delta` | `:426-446` | **no** at `level` (the previous behaviour) |
| `--emo_valence_table soft\|argmax` | `:333-337`, `:412` | **no** at `soft` — the same object as the hardcoded table |
| `node_V`, `cache_hits` | `mcts/mcts.py` | **no** — logging only |

The whole delta is instrumentation plus two dormant experimental arms. This is deliberate: it
lands inside the freeze without invalidating a single measured run, and the grid then produces
the `sigma_emo` and `per_step_valences` data needed to decide whether the risk rule and the
delta signal are worth sweeping — rather than needing a separate pass later.

### 9.4 Why the risk term was added at all

Three arguments converge, recorded in three different places:

1. **Structural, and specific to open-loop search** (`emotion_mcts.py:216-219`). Because a node
   is an action *sequence* rather than a state (§2), one edge genuinely sees different user
   reactions across simulations. The spread at an edge is therefore heterogeneity in the
   action's outcome, not estimator noise about a single true value — so averaging it away
   discards something real. In a closed-loop tree the same spread *would* be noise you want
   averaged out, and the term would be much harder to justify.
2. **The "severe information smoothing" objection.** A scalar running mean was keeping less
   than it discarded. `scripts/report_sigma_emo.py` measured `sigma_emo / |Q_emo|` at median
   **1.43**, p90 **9.10** (`FREEZE_NOTES.md:195`) — but on the deterministic stub classifier,
   so it evidences wiring and headroom, not emotional variance. `FREEZE_NOTES.md:208-214` is
   emphatic that it must not be quoted as the latter.
3. **Expected value is the wrong objective here** (`analysis/EMOMCTS_RESEARCH_DIRECTIONS.md:2326-2331`,
   "Direction D"): a strategy with a 60% donation rate where 40% of failures end in anger is
   worse than 55% with 5% anger, for judge scoring and for deployment. Two actions with equal
   mean emotional payoff are not equally good if one is a coin flip.

Direction D proposed CVaR / quantile over the full per-edge return distribution. What shipped
is the cheap mean-variance surrogate — one extra float per edge. The richer version is not
foreclosed: `per_step_valences` keeps every individual `z`, so CVaR stays computable offline
from the frozen logs.

**It has never been on.** `FREEZE_NOTES.md:114` — "**Not swept.** Instrumentation for a later
Tier-C experiment." No run in `calib/` uses a non-zero value.
