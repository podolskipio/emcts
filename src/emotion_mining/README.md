# `emotion_mining` — fitting the emotion channel from dialog corpora

This package fits the quantities EmoMCTS reads at plan time. Nothing here runs during search;
it all runs offline, and the planner consumes only the fitted constants.

Two estimators answer two versions of the same question, "which user affect co-occurs with
task success?":

| module | estimand | state | consumed by |
|---|---|---|---|
| `mine_emotion_donation_p4g.py` | `w(e)` — per-emotion valence | the **current** turn | `mcts.emotion_mcts.EMOTION_VALENCE_MINED` (**live**) |
| `mine_emotion_history_donation_p4g.py` | `ν(c)` — score of a trajectory | the **session so far** | not yet wired in (research) |

Supporting modules: `mining_corpus.py` (corpus loading, emotion labelling, caching, splits —
shared so the two estimators cannot drift apart), `build_p4g_rollout_evalset.py` (the disjoint
eval set), `mine_emotion_da_bonus_p4g.py` and `learn_emotion_transition_p4g.py` (analysis only).

Notation used throughout:

| symbol | meaning |
|---|---|
| `u` | a user (persuadee) utterance |
| `Φ(·\|u)` | emotion classifier posterior over the 8-emotion set, `Φ ∈ Δ⁷` |
| `d_t` | `Φ(·\|u_t)`, the distribution at turn `t` |
| `y ∈ {0,1}` | dialog outcome (p4g: the `agree-donation` annotation) |
| `p₀` | corpus base rate `P(y=1)`, 0.493 on p4g |
| `w(e)` | mined valence weight for emotion `e` |
| `ν(·)` | the score the planner adds to its value estimate |

---

## 0. Shared pipeline

Both estimators consume the same three stages, so any split or labelling decision applies
identically to both.

```
CORPUS → SEQUENCES:
  for each dialog D in corpus:
      utts ← [user turns of D]                    # persuadee side only
      y    ← outcome(D)                           # p4g: any 'agree-donation' label
      emit EmotionSequence(id=D.id, utterances=utts, y=y)

  drop dialogs in P4G_BAD_DIALOGS                 # 3 content-filtered, matches the runners
  drop dialogs in exclude_ids                     # leakage insurance (§3)
  drop first N dialogs if --holdout_first N       # positional holdout, replay evals only

LABELLING (memoised on disk; labels are a pure function of the text, so the
cache may span the whole corpus without leaking anything into the fit):
  hard cache:  u ↦ argmax_e Φ(e|u)                # one string per utterance
  soft cache:  u ↦ Φ(·|u)                         # the full 8-vector
```

The two caches are consistency-checked against each other: on the p4g corpus the soft cache's
argmax reproduces the hard cache on **4524/4524 utterances (100%)**. That matters because it
means a hard-vs-soft comparison isolates the *assignment rule* and nothing else — same
classifier, same text, same winning labels.

---

## 1. Algorithm 1 — instantaneous valence `w(e)`

### What it estimates

For each emotion `e`, the **lift** in success probability among turns carrying that emotion:

```
lift(e) = P(y = 1 | user turn shows e) − p₀
w(e)    = VALENCE_SCALE · shrunk_lift(e),      VALENCE_SCALE = 10
```

### Pseudocode

```
ALGORITHM 1  MineValenceWeights(sequences, α = 50, soft = true)
──────────────────────────────────────────────────────────────────────────
 1  p₀ ← mean(seq.y for seq in sequences)              # base rate
 2  for each emotion e:  S(e) ← 0 ;  T(e) ← 0          # successes, trials
 3
 4  for each seq in sequences:
 5      for each user turn u in seq:
 6          d ← Φ(·|u)
 7          if soft:                                   # ← deployment-matched
 8              for each emotion e:
 9                  T(e) ← T(e) + d(e)
10                  S(e) ← S(e) + d(e) · seq.y
11          else:                                      # ← Algorithm 2 line 7, legacy
12              ê ← argmax_e d(e)
13              T(ê) ← T(ê) + 1 ;  S(ê) ← S(ê) + seq.y
14
15  for each emotion e with T(e) > 0:
16      p̂(e)  ← S(e) / T(e)                            # raw success rate
17      p̃(e)  ← (S(e) + α·p₀) / (T(e) + α)             # shrunk toward base rate
18      w(e)  ← VALENCE_SCALE · (p̃(e) − p₀)
19      CI(e) ← Wilson(p̂(e), T(e))                     # significance flag only
20
21  w(contempt) ← −0.60                                # HF never emits it; LLM fallback
22  return w, CI
```

### How the planner uses it

`EmotionAwareMultiObjectiveQ` keeps a second backup channel `Q_emo` alongside the donation
`Q`, and scores a leaf's predicted emotion distribution as an **expectation**:

```
ν(d) = Σ_e d(e) · w(e)                        # mcts.emotion_mcts._emotion_value
score(a) = Q[s][a] + β_emo · Q_emo[s][a] + exploration
```

### Why it is built this way

**Why soft assignment (line 7).** This is the load-bearing design choice. Deployment scores
`ν(d) = Σ_e d(e)·w(e)` over the *whole* posterior. If mining credits only `argmax_e Φ(e|u)`,
then `w` is fitted on a different functional of `Φ` than the planner reads — mining and
deployment disagree about what an emotion label *is*. Soft assignment makes the training
statistic and the inference statistic the same object.

It is also a variance fix. Total mass is conserved (4847 turn-equivalents either way, since
`Σ_e d(e) = 1` per turn), but it moves from dominant cells to thin ones:

| | happiness | neutral | fear | anger | disgust |
|---|---|---|---|---|---|
| argmax `n` | 1145 | 2660 | 70 | 88 | 92 |
| soft `n_eff` | 1064 | 2294 | **120** | **187** | **203** |

Fear is the diagnostic case. Under argmax its 70 turns donate at 0.600, giving `w = +0.62`.
Those are precisely the turns where fear happened to *win a close posterior*. Counting fear's
partial mass across all turns, the rate regresses to 0.527 — barely above `p₀ = 0.493` — and
`w(fear)` falls to **+0.24**. The old value was an argmax artifact of a thin cell, which is the
instability the corpus already showed under resampling (+1.07 on 300 dialogs, +0.01 on 200).

**Why shrinkage (line 17).** A Beta prior centred on `p₀`, with `α` pseudo-observations. A cell
with no evidence gets lift exactly 0; thin cells are pulled toward "indistinguishable from the
corpus average" in proportion to how thin they are; well-populated cells are essentially
untouched. Without it, the thinnest cells — the ones we trust least — would produce the largest
weights and dominate the channel. `α = 0` recovers raw lifts.

**Why Wilson intervals (line 19).** They behave at the small `n` this corpus forces, unlike the
normal approximation. They are reported as a **significance flag only** and never enter `w`;
their job is to stop the reader treating an unevidenced cell as a finding.

**Why `VALENCE_SCALE = 10`.** Puts `w` in roughly `[−1, +1]`, the magnitude `Q_emo` expects, so
`β_emo` is the only knob controlling the channel's influence.

### Current values (soft, all 300 annotated dialogs, α = 50, `p₀ = 0.493`)

| emotion | `w(e)` | `n_eff` | CI excludes `p₀`? |
|---|---|---|---|
| happiness | **+0.54** | 1063.9 | **yes** |
| fear | +0.24 | 119.8 | no |
| disgust | +0.13 | 203.4 | no |
| anger | +0.11 | 187.0 | no |
| surprise | +0.09 | 589.2 | no |
| neutral | −0.07 | 2293.9 | no |
| sadness | −0.14 | 389.8 | no |
| contempt | −0.60 | — | hand value |

**Only happiness is statistically distinguishable from chance.** Every other cell is consistent
with the base rate. These are a weak search-biasing prior, not measured effects, which is why
the channel is gated by `β_emo`.

---

## 2. Algorithm 2 — history-conditioned valence `ν(c)`

### The gap it addresses

Algorithm 1 scores `d_t` alone. A user who has been angry for six turns and is momentarily
neutral scores identically to one who has been neutral throughout. Persuasion is a trajectory;
the instantaneous score cannot see it.

### Cumulative state

Each turn contributes to a decay-weighted, renormalised history:

```
c_t = normalize( Σ_{i ≤ t} λ^(t−i) · onehot(e_i) ) ∈ Δ⁷
```

`λ` is the whole experiment, and it nests the instantaneous model exactly:

| `λ` | meaning |
|---|---|
| 0 | only the current turn survives → **exactly Algorithm 1's state** |
| (0,1) | recency-biased memory |
| 1 | running mean over the session |

### Pseudocode

```
ALGORITHM 2  MineHistoryWeights(sequences, Λ = {0, .25, .5, .75, .9, 1}, l2 = 1, K = 5)
──────────────────────────────────────────────────────────────────────────────────────
 1  p₀ ← mean(seq.y)
 2
 3  function BuildStates(sequences, λ):                # one row per TURN
 4      X, y, g ← [], [], []
 5      for dialog index j, seq in enumerate(sequences):
 6          acc ← 0 ∈ ℝ⁸
 7          for t, e_t in enumerate(seq.emotions):
 8              acc ← λ · acc + onehot(e_t)            # linear-time accumulator
 9              X.append( acc / sum(acc) )             # c_t on the simplex
10              y.append( seq.y )                      # dialog label on every prefix
11              g.append( j )                          # group = dialog, for CV
12      return X, y, g
13
14  # ---- model selection: λ chosen by dialog-grouped K-fold CV ----
15  for λ in Λ:
16      X, y, g ← BuildStates(sequences, λ)
17      folds ← partition(unique(g), K)                # split DIALOGS, never turns
18      for each fold F:
19          θ, b ← FitRidgeLogistic(X[g ∉ F], y[g ∉ F], l2)
20          p̂[g ∈ F] ← σ(X[g ∈ F] · θ + b)             # out-of-fold predictions
21      record AUC(y, p̂), LogLoss(y, p̂)
22  λ* ← argmax_λ held-out AUC
23
24  # ---- final fit on all data at λ* ----
25  X, y, g ← BuildStates(sequences, λ*)
26  θ, b ← FitRidgeLogistic(X, y, l2)
27
28  # ---- express in w(e) units, then restore the magnitude contract ----
29  raw(c)  ← VALENCE_SCALE · (σ(θᵀc + b) − p₀)
30  s       ← 1.0 / percentile₉₉(|raw| over all observed c)
31  ν(c)    ← s · VALENCE_SCALE · (σ(θᵀc + b) − p₀)
32
33  w_pure(e) ← ν(onehot(e))                           # comparable to Algorithm 1's w(e)
34  return θ, b, s, λ*, w_pure


SUBROUTINE  FitRidgeLogistic(X, y, l2)   — Newton/IRLS, numpy only
──────────────────────────────────────────────────────────────────
 1  A ← [1 | X] ;  P ← l2 · I ;  P[0,0] ← 0            # never penalise the intercept
 2  β ← 0
 3  repeat until ‖Δβ‖∞ < tol  (≤ 50 iters):
 4      p ← σ(Aβ) ;  W ← clip(p(1−p), 1e−6, ∞)
 5      g ← Aᵀ(y − p) − Pβ                             # penalised gradient
 6      H ← (AᵀW)A + P                                 # penalised Hessian
 7      β ← β + H⁻¹g
 8  return β[1:], β[0]
```

### Why it is built this way

**Why logistic regression rather than per-emotion lifts.** Algorithm 1's lifts are *marginal*:
correlated emotions steal each other's credit, because a turn labelled happy in a mostly-happy
dialog is counted once for happiness with no adjustment for the rest of the trajectory.
Regression coefficients are *partial* effects — each emotion's contribution holding the others
fixed. This generalises the one-turn histogram fix to a full history.

**Why train on every prefix (line 10), not just the final state.** The planner scores partial
histories mid-search, at every depth. Training only on completed dialogs would fit a
distribution the planner never queries.

**Why ridge is mandatory, not optional (subroutine line 1).** The features sit on the simplex —
they sum to 1 — so they are exactly collinear with the intercept and the unpenalised Hessian is
singular. The intercept is left unpenalised, as usual.

**Why folds split on dialogs, never turns (line 17).** Every turn of a dialog carries the *same*
label. Splitting by turn would put turns of one dialog on both sides of the fold boundary, and
the model could recover the label from a sibling turn instead of from the emotion trajectory.
That is label leakage and it inflates AUC. Grouping by dialog is the only honest split here.

**Why rescale by the 99th percentile (line 30).** `_emotion_quality` documents its output as
bounded in `[−1, +1]`, and Algorithm 1's weights peak at 0.62. A logistic model is far more
confident at the simplex edges than a marginal lift is: on this run the 99th percentile of
raw `|ν|` is **1.80**, versus a peak of 0.62 for Algorithm 1's weights — so the fitted scale is
`s = 1/1.80 = 0.556`. Pasting the raw scores in would silently multiply the emotion channel's
influence by ~3× without anyone touching `β_emo`. (The module docstring quotes ≈3.6 from an
earlier run; the mechanism is the same, the magnitude moves with the fit, which is exactly why
the scale is computed per run rather than hard-coded.) The rescale is a single positive constant, so it preserves the *ranking* of
states exactly and only restores the magnitude contract. p99 rather than max, so one outlier
state cannot set the scale.

### Results (p4g, 300 dialogs, 4847 turns, `l2 = 1`, 5 folds)

| `λ` | held-out AUC | log-loss |
|---|---|---|
| 0.0 *(= Algorithm 1's state)* | 0.486 | 0.6969 |
| 0.25 | 0.500 | 0.6964 |
| 0.50 | 0.513 | 0.6955 |
| 0.75 | 0.525 | **0.6948** |
| **0.90** *(selected)* | **0.526** | 0.6957 |
| 1.00 | 0.520 | 0.6976 |

> **Read this honestly.** History conditioning improves monotonically over the instantaneous
> state (AUC 0.486 → 0.526), and the ordering is clean and interpretable. But **the absolute
> AUC is ~0.53 — barely above the 0.5 chance line**, and λ=0 is *below* it. This is evidence
> that emotional trajectory carries a little more signal about donation than a single turn
> does; it is **not** a working predictor of donation, and should not be reported as one. The
> honest claim is a small, consistent ordering effect on a corpus where the outcome is close
> to a coin flip (`p₀ = 0.493`).

Pure-emotion weights at λ*=0.9 (`ν(onehot(e))`, after rescaling) differ substantially from
Algorithm 1 — expected, since these are partial rather than marginal effects, and are read off
the extreme corners of the simplex where the logistic model extrapolates hardest:

```
happiness +1.00   fear +2.01   disgust +1.64   anger +0.94
surprise  −0.22   neutral −0.26   sadness −1.30   contempt −0.60 (hand)
```

These corner values are the least trustworthy output of the model: a history that is 100% fear
barely exists in the corpus. The `example_histories` block in the JSON reports realistic
mixtures instead, which is the comparison to quote.

---

## 3. Leakage and splits

`w(e)` is mined from the 300 annotated dialogs. The rollout eval set must not be drawn from
them. Two ways to guarantee that, with very different costs:

| approach | mining corpus | max \|Δw\| |
|---|---|---|
| `--holdout_first 100` (positional holdout) | 200 dialogs | **0.62** (fear +0.62 → 0.00) |
| move eval to the non-annotated half | **all 300** | **0.00** |

The second is adopted. The rollout eval set is the first 100 of the 717 **non-annotated**
PersuasionForGood dialogs (`build_p4g_rollout_evalset.py` → `data/p4g/rollout_evalset_nonannotated.jsonl`),
disjoint from the annotated 300 by construction. Mining keeps the full corpus and the eval set
shares nothing with it. `--exclude_ids` is passed on the production run as a standing
assertion — it must remove 0 dialogs, and `--assert_no_exclusions` aborts the run if it doesn't.
The JSON records `n_exclude_ids_supplied` (list length) separately from `n_excluded_by_id`
(dialogs actually removed).

This costs the evaluation nothing: p4g self-play dialogs carry `scenario=()`, so
`game.init_dialog()` starts from an empty session and the runner never replays corpus text —
only the dialog id is used, as an episode label.

**Replay evals are different.** `runners/emomcts.py` and `gdpzero.py` *do* feed annotated dialog
text to the planner and score against the ground-truth next turn. Weights used there must be
re-mined with `--holdout_first 100`, accepting the 0.62 hit.

---

## 4. Limitations to state in the paper

1. **Correlational, not causal.** "Fearful users donate more often" does not license "inducing
   fear causes donation." Every number here is a co-occurrence on an observational corpus.
   The planner uses them as a search prior, which is a defensible use of a correlate; a causal
   reading is not supported.
2. **One significant cell.** Only happiness has a Wilson CI excluding `p₀`. Reporting the full
   table without that caveat would overstate it by six emotions.
3. **Base rate near 0.5.** `p₀ = 0.493` — the outcome is close to a coin flip, so AUCs near
   0.53 are a small effect, and lifts are easy to over-read.
4. **Classifier-bounded.** All affect comes from one 7-class encoder
   (`j-hartmann/emotion-english-distilroberta-base`); `contempt` is never emitted and carries a
   hand-set fallback. Systematic classifier error is inherited wholesale by `w`.
5. **Algorithm 2 still labels by argmax.** `build_history_states` consumes the hard cache, so
   the mining/deployment mismatch fixed in Algorithm 1 is still present in the history variant.
   Moving it to the soft cache is a drop-in change and would mostly smooth `c_t`. It does not
   currently affect any run, since `ν(c)` is not wired into the planner.
6. **Wilson under soft counts.** `T(e)` becomes an *effective* sample size, not a turn count, so
   the interval is approximate — it assumes integer counts. No significance conclusion changes.

---

## 5. Reproduce

```bash
# eval set (must report: overlap 0)
python src/emotion_mining/build_p4g_rollout_evalset.py

# Algorithm 1 — the live weights
python src/emotion_mining/mine_emotion_donation_p4g.py --soft \
    --exclude_ids data/p4g/mining_exclude_ids.txt --assert_no_exclusions \
    --out outputs/mining_final_soft_all300.json

# Algorithm 2 — history model + lambda sweep
python src/emotion_mining/mine_emotion_history_donation_p4g.py

# ablations
python src/emotion_mining/mine_emotion_donation_p4g.py                     # argmax baseline
python src/emotion_mining/mine_emotion_donation_p4g.py --alpha 0           # unshrunk lifts
python src/emotion_mining/mine_emotion_history_donation_p4g.py --lam 0     # instantaneous
python src/emotion_mining/mine_emotion_donation_p4g.py --task esc          # ESConv
```

Both estimators run on either corpus via `--task {p4g,esc}`. ESConv ships no outcome
annotation, so `mining_corpus.py` defines one internally — see its docstring for what that
costs in interpretability.

See `REMINING.md` for the 2026-09-06 re-mine (soft assignment + leakage insurance) with the
per-change `Δw` decomposition.
