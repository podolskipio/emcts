# EmoMCTS contribution summary

> ⚠ **Historical (status line says 2026-06-01); superseded as a results claim (2026-09-17).** The
> headline "+14 pp" was measured under the **legacy** search horizon, the **`soft`** valence table, 40
> sims and 50 dialogues, against a matched-config baseline. The frozen grid supersedes all of it:
> `episode` horizon, `generic` table, dose-matched arms, n_sims 50 primary, 100 eval dialogues, and a
> **plain GDP-Zero baseline (B1)** that runs GDP-Zero's own `OpenLoopMCTS` — the unblock this document
> asks for. Its caveat about the matched-config baseline was right and is now addressed by design.
> See `PREREG.md`, `FREEZE_NOTES.md` §10, `thu/plan_4c_run_table.md`, `thu/b1_diagnosis.md`.

Status: 2026-06-01

This document summarises the project's measured contribution and what remains
to confirm it. Sister doc `EMOMCTS_RESEARCH_DIRECTIONS.md` covers the broader
design space and all variants we explored; this file focuses specifically on
**what works, what doesn't, and what's needed to publish.**

---

## Headline result (preliminary)

`EmotionAwareMultiObjectiveQ` (Direction A from the research-directions doc)
improves end-to-end success rate by **+14 percentage points** over a
**matched-config** GDPZero baseline.

| Planner | SR | AT | Config |
| --- | --- | --- | --- |
| GDPZero — matched config | **58%** (29/50) | 7.12 | top-K=5, N=40 sims, 50 dialogs |
| `EmotionAwareMultiObjectiveQ` (β=0.7) | **72%** (36/50) | 6.62 | top-K=5, N=40 sims, 50 dialogs |
| Δ | **+14 pp** | **−0.5 turns** | — |

Both metrics move in the right direction simultaneously: the variant
**closes more dialogs** AND **closes them faster**. There is no SR-vs-AT
trade-off here.

Backbone: vicuna:13b via Ollama. Task: p4g (persuasion-for-good donation
to Save the Children). Emotion classifier: HF DistilRoBERTa
(j-hartmann/emotion-english-distilroberta-base). Setup: end-to-end
rollouts from `runners/rollout.py`, scored by terminal user DA.

> ⚠ **Important caveat on the baseline**: the GDPZero number above is a
> **matched-config** baseline (top-K=5, N=40), NOT GDPZero as published in
> the original paper (15-sample histogram prior, N=20, all 13 actions).
> The result currently supports *attribution* claims (the β·Q_emo channel
> adds value at matched compute) but **does not yet support a canonical
> "beats published GDPZero" claim**. See [Claim calibration](#claim-calibration)
> below and the [What needs to be run](#what-needs-to-be-run-to-defend-the-canonical-claim)
> section for the unblock.

---

## What's new — the algorithmic contribution

`EmotionAwareMultiObjectiveQ` (`src/mcts/emotion_mcts.py`) is a small but
structural extension to the open-loop PUCT search in
`EmotionAwareOpenLoopMCTS`.

### The change in one sentence

> **Track expected emotional valence as a SECOND, independent Q channel
> alongside the donation-reward Q, and let it influence PUCT selection
> via a separately-sweepable weight β.**

### What it does

Standard PUCT with one Q channel:

```
score(a) = Q_donate[s][a] + c_puct · P[s][a] · √N / (1 + Nsa[s][a])
```

becomes

```
score(a) = Q_donate[s][a]
         + β · Q_emo[s][a]
         + c_puct · P[s][a] · √N / (1 + Nsa[s][a])
```

where:

- `Q_donate[s][a]` is the existing donation-rollout running mean
  (unchanged, identical to GDPZero's Q).
- `Q_emo[s][a]` is a NEW running mean of `Σₑ p(e) · valence(e)` evaluated
  at the immediate child state's predicted emotion distribution. The
  emotion classifier output is already cached on the next-state object by
  `EmotionAwarePersuasionGame.get_next_state`, so this requires zero
  additional LLM or classifier calls.
- `β` is a single sweep knob. `β=0` is observationally identical to
  `EmotionAwareOpenLoopMCTS` (Q_emo is tracked but ignored at selection
  time). `β>0` lets emotion-quality bias action selection.

### Why this is structurally different from prior attempts

The project previously tried two ways of folding emotion into MCTS:

1. **π penalty** (`EmotionAwareDiscountQOpenLoopMCTS`): blends an
   emotion-based penalty INTO the donation Q via convex combination
   (`λ·v + (1-λ)·π`). Emotion and donation rewards become non-decomposable
   the moment they're backed up.
2. **Bonus matrix on PUCT** (`EmotionGuidedDiscountQOpenLoopMCTS`): adds a
   hand/data-tuned per-(last_emotion, DA) selection bias to PUCT.
   Operates at the action-bias level, not the value level.

`EmotionAwareMultiObjectiveQ` is a third, distinct approach:

- It **does not fuse** emotion and donation rewards.
- It **does not bias selection via a fixed table** — Q_emo is *learned
  during search* from the same rollouts that populate Q_donate.
- It enables **Pareto-aware inspection** of (Q_donate, Q_emo) per (s, a)
  after the fact — the search tree is fully decomposable.
- It enables **post-hoc β sweep** on the same tree, which neither of the
  above allow.

### Compute cost

- Per node: one extra float per (s, a) — `Q_emo[s][a]`.
- Per backup: one extra running-mean update.
- Per UCT call: one extra add and multiply.

There is **no extra LLM call, no extra classifier call, no new model to
load**. The emotion distribution is already classified by the game
transition. Cost is effectively zero compared to the existing LLM-bound
search.

---

## What did NOT work — three runs that tied or lost

For context on why MultiObjectiveQ is the headline:

### 1. π penalty (`EmotionAwareDiscountQOpenLoopMCTS`)
- **Config**: λ=0.3, N=20 sims, 20 dialogs
- **Setup A** (h2h judge): ~45% — **statistically tied to GDPZero**
- **Why it didn't work**: penalty fuses emotion into donation Q. With
  90%+ neutral/happy user turns, the penalty rarely fires; when it does,
  it makes the Q signal harder to interpret without improving outcomes.

### 2. Bonus matrix v1 (`EmotionGuidedDiscountQOpenLoopMCTS`, aggressive)
- **Config**: c_emo_bonus=1.0, hand+mined hybrid matrix with +0.60
  Happiness→proposition, +0.40 Neutral→proposition, top-K=7, N=20, 20
  dialogs
- **Setup A** (h2h judge): **44.7%** (68W/84L) — **clear loss**
- **Why it didn't work**: judge analysis showed the matrix over-promotes
  proposition. On 34 turns where GDPZero played soft "other" and EmoMCTS
  played "proposition", the judge preferred GDPZero 62% of the time.
  Aggressive close behaviour reads as pushy.

### 3. Bonus matrix v2 (`EmotionGuidedDiscountQOpenLoopMCTS`, calibrated)
- **Config**: matrix with Happiness/Neutral proposition cut to +0.20/+0.10
  after the v1 analysis. **Built but not yet re-evaluated** — pending.

These three results, taken together, established that **adding emotion as
a modifier on the existing Q channel either ties or loses** at the
configurations explored. MultiObjectiveQ's contribution is the specific
structural choice to track emotion as a *parallel* Q channel rather than
modifying the donation Q.

---

## How we compared to GDPZero — the matched-config methodology

The +14pp result is meaningful because it's the first measurement in the
project at fully matched conditions. Earlier comparisons were
confounded.

### The two experimental setups

| | Setup A (h2h judge) | Setup B (end-to-end SR) |
| --- | --- | --- |
| Runner | `runners/gdpzero.py` + `runners/emomcts.py` | `runners/rollout.py` |
| Mechanic | Replay human dialog turn-by-turn; planner picks next DA; record (human_resp, planner_resp) per turn | Play full dialog from scratch against simulated user; record terminal outcome |
| Evaluator | `evaluators/run_judge.py` — GPT-3.5/4o picks per turn | `metrics/run_metrics.py` — aggregates SR / AT |
| Metric | Per-turn judge preference (win rate vs GDPZero) | End-to-end donation rate (SR), avg dialog length (AT) |
| Captures | "Which next-move does the judge prefer, given the same context?" | "Does the planner actually reach donation when driving the dialog?" |

**The headline +14pp is Setup B.** This is the metric that matches the
project's actual goal (close more donations). Setup A (judge preference)
measures something different — it favored GDPZero's softer style in the
previous tests but does not directly measure donation outcome.

### Matched config

The previous runs were a mix:
- Bonus matrix v1: top-K=7, N=20, 20 dialogs (Setup A)
- GDPZero h2h reference: top-K not set, N=20, 20 dialogs (Setup A)
- Old GDPZero SR reference: 75% ±19pp on 20 dialogs (Setup B, but n too small)

The new matched comparison strictly controls everything except the
algorithm under test:
- **top-K = 5** for both (shrinks branching factor 13 → 5; same prior call
  used in both runs)
- **N = 40 sims** per turn for both
- **n = 50 dialogs** for both
- **Same dialog scenarios** (both pickles use `dialogs[:50]` from the same
  loader output, so they evaluate on the identical 50 scenarios — this
  enables paired statistical analysis)
- **Same backbone**, same simulator, same emotion classifier, same
  random-seed behaviour
- **Only difference**: `--algo gdpzero` vs `--algo emomcts --beta_emo 0.7`

### Reproducing the result

```bash
# GDPZero baseline
python src/runners/rollout.py \
    --game p4g --algo gdpzero \
    --llm ollama --ollama_model vicuna:13b \
    --llm_prior_topk 5 \
    --num_mcts_sims 40 \
    --max_conv 50 --max_turns 10 \
    --output outputs/rollout_p4g_gdpzero_vicuna_50d_40s.pkl

# MultiObjectiveQ variant
python src/runners/rollout.py \
    --game emo_p4g --algo emomcts \
    --llm ollama --ollama_model vicuna:13b \
    --emotion_classifier hf \
    --llm_prior_topk 5 \
    --beta_emo 0.7 \
    --num_mcts_sims 40 \
    --max_conv 50 --max_turns 10 \
    --output outputs/rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims.pkl

# Score
python src/metrics/run_metrics.py \
    --episodes outputs/rollout_p4g_gdpzero_vicuna_50d_40s/rollout_p4g_gdpzero_vicuna_50d_40s.pkl \
    --max_turns 10
python src/metrics/run_metrics.py \
    --episodes outputs/rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims/rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims.pkl \
    --max_turns 10
```

---

## Claim calibration

The +14pp result above does not mean the same thing as "MultiObjectiveQ
beats GDPZero." It means "MultiObjectiveQ beats a specific GDPZero
configuration that itself is not the published baseline." Three
legitimate baselines exist, and each supports a different claim.

### The three legitimate baselines

Verified against the GDPZero repo's example pickles in
`outputs/gdpzero_{5,10,20,50}sims_3rlz_*Q0_20dialogs.pkl`. The paper's
headline 93.51% vs-human number is the **N=50** run, NOT N=10 or N=20.

| Baseline | Config | What it is | What a positive result against it can claim |
| --- | --- | --- | --- |
| **GDPZero-as-published** | **N=50** sims, Q_0=0.0, max_realizations=3, **no `--llm_prior_topk`** (15-sample LLM histogram prior), all 13 actions, **gpt-3.5-turbo backbone**, 20 dialogs (= 154 turns) | The original paper's headline config. Reproduces the 93.51% vs-human number. | "Our method improves over GDPZero as published" — **the canonical contribution claim** |
| **GDPZero-matched** | top-K=5, N=40 sims, same emomcts-side prior pipeline | GDPZero with our prior modification at our chosen compute, matched to our variant | "The β·Q_emo channel adds value at matched compute" — **attribution claim** |
| **GDPZero best-of** | Sweep over (top-K, N, cpuct) and report the GDPZero peak | GDPZero given its best shot under our codebase | "Our method beats GDPZero's peak performance" — **strongest fairness claim**, optional for papers |

The +14pp SR number addresses **only the attribution claim** (matched
config). The canonical "vs published GDPZero" comparison requires Step 1
of [What needs to be run](#what-needs-to-be-run-to-defend-the-canonical-claim)
below.

### What the +14pp could mean — three scenarios

We currently cannot distinguish these without running GDPZero-as-published:

| Scenario | GDPZero-as-published lands at | Reading |
| --- | --- | --- |
| **A** | ≈ 58% (similar to matched) | Our top-K modification was neutral for GDPZero; the +14pp is a real algorithmic gain over published GDPZero. **Strong publishable result.** |
| **B** | ≈ 65-70% | Top-K + 2x sims modestly hurt GDPZero; MultiObjectiveQ recovers part of the loss AND adds value. Gap shrinks but stays positive — **publishable as a smaller win.** |
| **C** | ≈ 72%+ | Our top-K modification damaged GDPZero; MultiObjectiveQ only recovers what we broke. **No real improvement to claim.** |

Without the canonical baseline run, the contribution claim is bounded by
"performs better than a modified-GDPZero baseline at matched compute."
That is a real result and useful for attribution, but it is NOT what a
reader assumes from "beats GDPZero."

### Calibrated claim statements (what we can and cannot say right now)

| Claim | Currently defensible? |
| --- | --- |
| "MultiObjectiveQ improves end-to-end donation rate over GDPZero." | **Not yet** — depends on which GDPZero. |
| "MultiObjectiveQ improves SR by +14pp over a matched-config GDPZero baseline at N=40 / top-K=5." | **Yes** (single seed) |
| "The β·Q_emo channel contributes meaningfully to SR at matched compute." | **Pending β=0 control** (could be the top-K shrinking branching factor, not Q_emo) |
| "MultiObjectiveQ is competitive with GDPZero as published." | **Pending GDPZero-as-published run** |
| "MultiObjectiveQ generalises across personas / harder simulators." | **Pending persona experiment** |

---

## What needs to be run to defend the canonical claim

The canonical "vs GDPZero" experiment uses the paper's headline config
on the paper's primary metric (per-turn judge h2h vs human, NOT
end-to-end SR — the original paper does not report SR for p4g).

### The four-step canonical protocol

```bash
# Step 1 — GDPZero at the paper's headline config (replicates 93.51% vs human)
python src/runners/gdpzero.py \
    --game p4g \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 50 \
    --Q_0 0.0 \
    --max_realizations 3 \
    --num_dialogs 20 \
    --output outputs/gdpzero_paper_n50.pkl

# Step 2 — MultiObjectiveQ at the SAME config (only difference: --beta_emo)
python src/runners/emomcts.py \
    --game emo_p4g \
    --llm gpt-3.5-turbo \
    --emotion_classifier hf \
    --num_mcts_sims 50 \
    --Q_0 0.0 \
    --max_realizations 3 \
    --num_dialogs 20 \
    --beta_emo 0.7 \
    --output outputs/emomcts_multiobjq_paper_n50.pkl

# Step 3 — judge each vs human (the 93.51% comparison)
python src/evaluators/run_judge.py \
    --task p4g --judge gpt-3.5-turbo \
    -f outputs/gdpzero_paper_n50/gdpzero_paper_n50.pkl \
    --output outputs/judge_gdpzero_vs_human.pkl
# → target: lands near 93.51% (confirms we faithfully replicated the paper)

python src/evaluators/run_judge.py \
    --task p4g --judge gpt-3.5-turbo \
    -f outputs/emomcts_multiobjq_paper_n50/emomcts_multiobjq_paper_n50.pkl \
    --output outputs/judge_emomcts_vs_human.pkl
# → target: ≥ GDPZero's vs-human number (cleanly beats the paper number)

# Step 4 — h2h between the two
python src/evaluators/run_judge.py \
    --task p4g --judge gpt-3.5-turbo \
    -f outputs/emomcts_multiobjq_paper_n50/emomcts_multiobjq_paper_n50.pkl \
    --h2h outputs/gdpzero_paper_n50/gdpzero_paper_n50.pkl \
    --output outputs/judge_h2h_multiobjq_vs_gdpzero.pkl
# → target: > 50% with statistical significance
```

**No `--llm_prior_topk`** (the paper doesn't use it). Same N=50, same
Q_0=0.0, same backbone, same dataset slice.

**Our `evaluators/resp_ranker.py` already implements the paper's
5-vote-majority judge protocol** (`inference_args = {"n": 5,
"temperature": 0.7, "max_tokens": 2, ...}` plus `_majority_vote`). Matches
the GDPZero repo's `core/evaluator.py` exactly — verified.

### Statistical-significance considerations from the original paper

The GDPZero paper's headline numbers come from **n = 154 turns** (20
dialogs). Two notes worth flagging:

- **vs-human (93.51%, 144W/0D/10L)**: rock-solid significant. Z ≈ 10.8
  vs null `p = 0.5`; 95% Wilson CI ≈ [0.882, 0.967]. A reviewer will
  not push back on this.
- **h2h vs ChatGPT (59.09%, 91W/2D/61L)**: **marginally significant**.
  Z ≈ 2.26, two-tailed p ≈ 0.024; 95% Wilson CI ≈ [0.512, 0.665]. The
  lower bound just barely clears 50%. Five wins falling the other way
  would push p above 0.10.

Two implications for our own protocol:
- **Target ≥ 94% vs human** for the canonical comparison. Matching their
  rock-solid bar.
- **Any h2h claim at n=154 inherits the same marginal-significance
  regime.** If we report MultiObjectiveQ wins h2h at, say, 56%, that's
  *not* meaningfully more confident than the original
  GDPZero-vs-ChatGPT was. We should at minimum match their n=154, and
  ideally improve over it with n=40 dialogs (~300 turns) to put the
  comparison on firmer footing than the paper itself. The extra cost is
  modest (~$100-160 instead of $50-80) and the resulting numbers are
  defensible against a careful reviewer.

### Choice of backbone — vicuna:13b vs gpt-3.5-turbo

The paper uses **gpt-3.5-turbo** end-to-end (planner LLM, simulator LLM,
judge LLM). All of our existing runs use vicuna:13b via Ollama for the
planner+simulator and gpt-3.5-turbo only for the judge.

| Backbone choice | Cost (one run) | Claim defensibility |
| --- | --- | --- |
| **gpt-3.5-turbo** (matches paper) | ~$20-30 / run on OpenAI; ~$50-80 for the full canonical pair + judges | **Strongest** — directly comparable to paper number. Required for a "beats GDPZero" claim. |
| **vicuna:13b** (matches existing runs) | ~30-60 hours wall-clock per run on local Ollama | "Beats GDPZero when both are replicated with vicuna:13b backbone." Honest but weaker — reviewers will request gpt-3.5-turbo replication. |
| **Both** (recommended for paper) | gpt-3.5-turbo for headline, vicuna for ablation table | Best of both worlds: canonical claim against paper number, plus open-backbone replication for reproducibility. |

The pragmatic plan: **run the gpt-3.5-turbo pair first** (one-time
$50-80 cost is small relative to weeks of compute saved). Use that as
the headline. Keep the vicuna runs as the supplementary "different
backbone" data point.

### After Steps 1-4 — full picture

| Planner | Config | Metric | Result |
| --- | --- | --- | --- |
| GDPZero-as-published | N=50, Q_0=0.0, no top-K, gpt-3.5-turbo, 20 dialogs | vs-human win rate | ??? **pending — the canonical baseline** |
| MultiObjectiveQ (β=0.7) | N=50, Q_0=0.0, no top-K, gpt-3.5-turbo, 20 dialogs | vs-human win rate | ??? **pending** |
| MultiObjectiveQ vs GDPZero | both at paper config | h2h win rate | ??? **pending — the canonical h2h** |
| GDPZero-matched | N=40, top-K=5, vicuna:13b, 50 dialogs | rollout SR | 58% (measured) |
| MultiObjectiveQ (β=0.7) | N=40, top-K=5, vicuna:13b, 50 dialogs | rollout SR | 72% (measured) |
| MultiObjectiveQ β=0 control | N=40, top-K=5, vicuna:13b, 50 dialogs | rollout SR | ??? pending |

That set supports:
- **Canonical comparison** (Steps 1-4 above): MultiObjectiveQ vs published GDPZero on the paper's own metric.
- **Open-backbone replication** (existing rollout numbers): same result holds with vicuna:13b instead of gpt-3.5-turbo.
- **Attribution to algorithm**: MultiObjectiveQ vs GDPZero-matched at same compute.
- **Attribution to Q_emo specifically**: MultiObjectiveQ vs β=0 control.

### Recommended ablation table for the paper

| Comparison | Purpose |
| --- | --- |
| GDPZero (N=50, no top-K) vs MultiObjectiveQ (N=50, no top-K, β=0.7) | **headline** — canonical paper comparison |
| GDPZero (N=10, Q_0=0.25) vs MultiObjectiveQ (N=10, Q_0=0.25, β=0.7) | confirms gain at GDPZero's "fast" config from their N=10 ablation |
| MultiObjectiveQ at β ∈ {0, 0.3, 0.7, 1.0} | β-sweep dose-response curve — strongest paper figure |
| End-to-end SR at N=40, top-K=5 (existing data) | supplementary end-to-end measurement |
| Persona-conditioned eval (Tier 5) | robustness story across hard simulators |

---

## Statistical strength

### Unpaired test (conservative)

Treating the two runs as independent samples:
- pooled p̂ = 65/100 = 0.65
- SE on difference = √(0.65·0.35·(1/50+1/50)) ≈ 0.095
- Z = 0.14 / 0.095 ≈ 1.47, two-tailed p ≈ **0.14**

Not statistically significant at the standard p<0.05 threshold under this
test.

### Paired test (proper, since scenarios are identical)

Both runs evaluated on the same 50 dialog scenarios. The natural
statistical test is McNemar's on discordant pairs:

```python
import pickle, math
g = {ep['did']: ep['success'] for ep in pickle.load(open('outputs/rollout_p4g_gdpzero_vicuna_50d_40s/rollout_p4g_gdpzero_vicuna_50d_40s.pkl', 'rb'))}
m = {ep['did']: ep['success'] for ep in pickle.load(open('outputs/rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims/rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims.pkl', 'rb'))}
shared = set(g) & set(m)
b = sum(1 for d in shared if not g[d] and m[d])  # MOQ won, GDPZero lost
c = sum(1 for d in shared if g[d] and not m[d])  # GDPZero won, MOQ lost
chi2 = (abs(b - c) - 1)**2 / max(b + c, 1)  # with continuity correction
print(f"n={len(shared)} MOQ-only={b} GDPZero-only={c} chi2={chi2:.2f}")
```

With a +14pp aggregate gap and matched scenarios, MOQ-only dialogs should
materially exceed GDPZero-only dialogs, producing a chi-squared statistic
well above the p=0.05 critical value of 3.84. **Run this and record the
value in this doc.**

---

## What is NOT yet confirmed

Honest list of remaining concerns, in priority order:

1. **GDPZero-as-published baseline missing** (the canonical
   comparison). The 58% baseline is GDPZero with our top-K modification
   at our chosen compute — NOT the original-paper configuration. The
   paper's headline 93.51% vs-human number uses **N=50 sims, Q_0=0.0,
   max_realizations=3, no top-K, gpt-3.5-turbo backbone, 20 dialogs** (=
   154 turns). Until GDPZero AND MultiObjectiveQ are both run at that
   config and judged through the same 5-vote-majority pipeline, we
   cannot claim "beats GDPZero" in the sense a reader assumes. See
   [Claim calibration](#claim-calibration) above for the three
   legitimate baselines and [What needs to be run](#what-needs-to-be-run-to-defend-the-canonical-claim)
   for the exact commands. **This is the #1 unblock — total cost ~$50-80
   on OpenAI for the full canonical pair.**
2. **β=0 control missing**. To prove the parallel Q_emo channel is the
   active ingredient — and not just "top-K=5 + N=40 +
   EmotionAwareOpenLoopMCTS shape" — we need the β=0 control at the same
   config. If β=0 lands at ~70%, the gain was from top-K + N=40; if it
   lands at ~58%, Q_emo did it. Without this, even the attribution claim
   ("β·Q_emo channel adds value at matched compute") is unproven; we
   only know "EmotionAwareMultiObjectiveQ class as a whole beats matched
   GDPZero".
3. **Single seed**. Both runs used Python's default RNG initialisation. A
   different seed pair would either confirm (publishable) or shrink the
   gap (regression to noise). Need ≥2 more seed pairs.
4. **Simulator-bound**. The 58% / 72% absolute numbers are conditioned on
   vicuna:13b as the persuadee. A more agreeable simulator would raise
   both; a less agreeable one might lower both. The *gap* is what matters
   for the claim, but external validity requires the persona experiment
   (see "What to do next" below).
5. **Setup A (judge h2h) unconfirmed**. We have NOT yet run
   MultiObjectiveQ through the h2h judge pipeline at this matched config.
   Previous Setup A runs were a different variant (bonus matrix v1) at a
   different config. The h2h result for MultiObjectiveQ specifically
   could agree, disagree, or be neutral with the SR finding.
6. **No comparison vs other emotion-aware variants**. v_imag, imagination,
   learned prior — all built but unmeasured. Possible some of them would
   beat MultiObjectiveQ at the same config, in which case the
   contribution claim should shift to that variant.

---

## What to do next

Recommendations are ranked by *information gained per hour of compute*.

### Tier 1 — confirm the result (do this week)

Listed in priority order. The first row is the unblock for the canonical
"vs GDPZero" claim and should run before anything else.

| Action | Cost | Why |
| --- | --- | --- |
| **Canonical pair** (`runners/gdpzero.py --num_mcts_sims 50 --llm gpt-3.5-turbo` + `runners/emomcts.py --num_mcts_sims 50 --beta_emo 0.7 --llm gpt-3.5-turbo` + `run_judge.py` on both vs human and h2h) | ~$50-80 on OpenAI | **The #1 unblock.** This is the paper-credible "vs GDPZero" comparison using the paper's exact headline config (N=50, Q_0=0.0, no top-K, 20 dialogs, gpt-3.5-turbo, 5-vote-majority judge). Until this set of numbers exists, the contribution doc can only claim "vs matched-config GDPZero," not "vs GDPZero." See [What needs to be run](#what-needs-to-be-run-to-defend-the-canonical-claim) for the full 4-step protocol. |
| **Paired McNemar test** on existing pickles | 30 seconds | Currently the unpaired test says p≈0.14. Paired test on matched scenarios is the proper analysis and will almost certainly be p<0.05. Free statistical strength. (Run this once the canonical-GDPZero pickle exists so you can do the paired test against THAT too.) |
| **β=0 control** of MultiObjectiveQ at same config | ~30 hours | Tells us whether Q_emo is the active ingredient. Without this, the attribution claim ("β·Q_emo channel adds value") collapses to "MultiObjectiveQ class as a whole adds value", which is weaker. |
| **Setup A confirmation** — `gdpzero.py` + `emomcts.py --beta_emo 0.7` + `run_judge.py` at matched config | ~10 hours | Tells us whether the per-turn judge agrees with the end-to-end SR finding. If yes, two-channel validation. If no, interesting tension to investigate. |

If Tier 1 confirms (paired p<0.05, β=0 falls to ~58-65%, h2h ≥50%), the
result is solid enough for an internal write-up.

### Tier 2 — strengthen the headline (next 1-2 weeks)

| Action | Cost | Why |
| --- | --- | --- |
| **2 more seeds × 50 dialogs each** for both planners | ~120 hours | Replication. 3 seeds × 50 dialogs = 150 dialogs / planner; standard error on SR drops to ~4pp; the +14pp signal would be at >3σ if it holds. **This is the publication-grade replication.** |
| **β sweep** — run at β=0.3, 0.5, 1.0 in addition to 0.7 | ~120 hours (4 runs × 30h) | Maps the dose-response curve. Strongest paper figure: SR-vs-β with a clear monotone trend or a peak. |
| **Compute paired-difference distribution** — not just McNemar | 1 hour script | Per-dialog Δ, with bootstrapped CI. Lets us report effect size (14pp ± Δ) not just significance. |

### Tier 3 — should I run for 100 dialogues?

**Yes — but as part of the multi-seed campaign, not as a single n=100 run.**

Single n=100 run:
- ±10pp 95% CI on SR (binomial)
- Detects ~14pp gaps comfortably under unpaired test
- But single seed; vulnerable to RNG luck

Multi-seed alternative (recommended):
- 3 seeds × 50 dialogs = 150 episodes/planner
- ±8pp 95% CI on pooled SR
- Confirms robustness across seeds — much stronger for publication
- Same total compute (~120 hours per planner)

The multi-seed setup answers "is this real?" with much higher confidence
than a single n=100. For a paper, reviewers will explicitly ask for seed
variance. n=100 single-seed is necessary but not sufficient.

**Recommended setup for publication-grade comparison**:
- GDPZero at 3 seeds × 50 dialogs (150 episodes)
- MultiObjectiveQ (β=0.7) at 3 seeds × 50 dialogs (150 episodes)
- Report: mean SR ± SEM across seeds, paired Δ per scenario per seed,
  pooled McNemar across all 150 dialog-pairs.

### Tier 4 — what other metrics?

Once Tier 1-3 are stable, the following complementary metrics add
explanatory depth without requiring new runs:

| Metric | Where it lives | Cost | What it tells you |
| --- | --- | --- | --- |
| **Conditional SR by last user emotion** | Metric Improvement 1 in research-directions doc | 1 hour script over existing data | Does MultiObjectiveQ win uniformly, or specifically on negative-emotion users? Mechanism evidence. |
| **Emotion-trajectory quality** | Metric Improvement 3 | 1 hour script over existing emotion records | Mean expected valence integrated over user turns. Does MultiObjectiveQ produce smoother / more positive arcs even when SR ties? |
| **Emotion-distribution delta** between planners | Metric Improvement 4 | 1 hour script | Per-DA conditional emotion shift. Isolates "different strategies" from "different utterance quality". |
| **DA distribution diff** | Existing run_judge.py metadata | 0 hours (already in metadata) | Does MultiObjectiveQ play different DAs than GDPZero? If yes, strategic-mix explanation. If no, search-quality explanation. |
| **AT distribution histogram** | 1 hour script | 1 hour | Is the 0.5-turn AT improvement uniform, or driven by a few very-fast dialogs? |

### Tier 5 — the substrate change (research-paper depth)

Currently both planners are evaluated against vicuna:13b acting as a
fairly agreeable persuadee. Across all our runs, ~88% of user turns are
positive or neutral, and anger/disgust are ~0.1%. **This means the
bonus-matrix and π-penalty machinery is operating on a near-empty set.**

The recommended substrate change is **persona-conditioned simulators**:
sample a persona prefix per dialog (skeptical / cynical / hostile /
distracted / etc.) that lowers the SR ceiling and surfaces more negative
emotional reactions. This:

- Tests **robustness** rather than just absolute SR
- Surfaces a regime where emotion-aware planners can outperform
  GDPZero by larger margins than ±14pp
- Is the headline experiment for a research paper

Cost: ~2 days of implementation work (the infrastructure is partially
built — `data/personas/p4g_personas.json` and
`scripts/calibrate_personas_p4g.py` exist), then ~1-2 weeks of compute
for a 6 planner × 10 persona × 20 dialog grid (1200 dialogs total).

See `EMOMCTS_RESEARCH_DIRECTIONS.md` Part 2 of the "what algorithmic
change in current code" section for the full experimental design.

---

## Recommended next-step decision tree

```
1. Has the canonical pair been run?    →  No  →  RUN IT (~$50-80 on OpenAI)
   (GDPZero N=50 + MultiObjectiveQ N=50,
    both at gpt-3.5-turbo, judged
    vs human and h2h with 5-vote majority)
                                       →  Yes — what did it land at?
                                              ↓
                                  MultiObjectiveQ vs human ≥ GDPZero vs human
                                  AND h2h > 50% with significance?
                                        ↓
                                  Yes:  CANONICAL CLAIM SUPPORTED.
                                        Proceed to step 2 for replication.
                                  No (vs-human tied):
                                        Smaller but real win. Frame as
                                        "matches GDPZero on judge metric,
                                        improves end-to-end SR."
                                  No (loses outright):
                                        STOP — re-frame as a calibration paper
                                        or pivot to persona experiment.

2. Has the paired McNemar test been run?  →  No  →  RUN IT (30 sec)
   (on both the matched AND the published-config GDPZero pickles)
                                          →  Yes — p<0.05 on either?
                                              ↓
                                        Yes:  proceed to step 3
                                        No:   replicate with second seed
                                              before investing further

3. Has the β=0 control been run?       →  No  →  RUN IT (~30h)
                                       →  Yes — where did it land?
                                              ↓
                                        ≈ 58-65%: Q_emo IS doing the work
                                        ≈ 70%+:   top-K + N=40 was the work,
                                                  attribution shifts away
                                                  from the parallel Q channel

4. Tier 1 all confirms → Tier 2 (2 more seeds + β sweep)
                                       → Tier 4 (diagnostic metrics over
                                                 existing pickles)
                                       → Tier 5 (persona experiment)
                                       → Write paper
```

---

## Summary

- **First positive directional signal** in the project: +14pp SR on
  matched config (Setup B), with AT also improving by 0.5 turns.
- **The contribution is the parallel-Q channel architecture** — a
  decomposable two-objective backup that doesn't fuse emotion into the
  donation Q, doesn't add LLM cost, and exposes a single sweep knob β.
- **The canonical "vs GDPZero" claim is NOT yet supported.** The +14pp
  is vs a matched-config GDPZero (top-K=5, N=40, vicuna:13b), not vs
  GDPZero as published (N=50, Q_0=0.0, no top-K, gpt-3.5-turbo, 20
  dialogs, 5-vote-majority judge vs human). Total cost to run the
  canonical pair on OpenAI: ~$50-80. See [Claim
  calibration](#claim-calibration) and [What needs to be run](#what-needs-to-be-run-to-defend-the-canonical-claim).
- **Attribution to Q_emo specifically is also not yet proven.** β=0
  control at the same config is required to distinguish "the parallel Q
  channel did it" from "top-K + extra sims did it."
- **Tier 1 confirmations** (canonical baseline, paired test, β=0
  control, h2h replication) all need to happen before any external
  claim. None of them is more than ~30 hours of compute.
- **Recommended publication path**: 3 seeds × 50 dialogs per planner
  (not single n=100), plus a β sweep and the diagnostic metrics in Tier
  4. Persona-spectrum eval (Tier 5) is the strongest single experiment
  for a paper if you commit to it.
