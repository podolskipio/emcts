> **SUPERSEDED 2026-09-16.** P1 ran as specified in the Thursday build spec, not as designed below.
> **Arms:** NoEmo (β 0), `--search_horizon legacy` and `episode`, **both run fresh**, serially, same seed.
> **Dialogues:** 131–140 (disjoint from eval 1–100 and from D1–D3's 101–130). Runs are in
> `analysis/thu/runs/P1_{legacy,episode}` and the report is `analysis/thu/p1_pilot.md`.
> **Why:** (1) β 0 isolates the horizon change from the affect channel. Under β 0.7 a horizon effect
> would be confounded with Bias's tree narrowing (`p_var.md` §7.6). (2) Running legacy fresh, next to
> episode on the same server under the same load, gives a like-for-like wall-clock number, and the grid
> budget depends on it. D1 ran 30 dialogues concurrently, so its per-dialogue wall clock is not
> comparable. `analyze_pilot.py` still applies with `--ref P1_legacy --beta 0.0`.

# Pilot P1 — `--search_horizon episode` (Thursday 2026-09-17)

**Question.** What does bounding search at the episode horizon change? Specifically: correctness,
cost, search structure, outcomes on paired dialogues, and the Phase-1 gate components.

**Background.** `analysis/phase1/SEARCH_HORIZON_BUG.md`.

## Design

| | |
|---|---|
| arm | D1's exact config + `--search_horizon episode`. β 0.7, vicuna-13B-AWQ, sampled scoring, R 4, n_sims 50, K 5, Tmax 10, persona, HF classifier, 10 workers, seed 0 |
| dialogues | the first 10 of the Phase-1 set: positions 101–110 of the non-annotated pool, the same ids as D1's first 10. Not the eval set |
| comparator | D1 on the same 10 dialogues (legacy). Already logged, so it costs no GPU |
| cost | about 1 GPU-h. D1 took 2.97 h for 30 dialogues; bounded trees are shallower, so likely less |

**What P1 cannot do.** n = 10 paired dialogues cannot establish an SR difference, so the outcome
tables are descriptive. P1 answers "does the fix work, and how much does it move search and the gate
components". It does not answer "is the fixed planner better".

## Thursday

```bash
# 1. server (same as D1/D2)
nohup bash analysis/calib/serve.sh 13b > analysis/phase1/runs/logs/sglang_server_P1.log 2>&1 &
#    wait for "The server is fired up and ready to roll!" in that log

# 2. pilot run (~1 h)
nohup analysis/phase1/scripts/run_diag.sh P1 0.7 10 --search_horizon episode \
      > analysis/phase1/runs/logs/P1.log 2>&1 &

# 3. analysis (CPU, a few minutes). Exits non-zero if the correctness gate fails
python3 analysis/phase1/pilot_horizon/analyze_pilot.py \
    --pilot analysis/phase1/runs/P1/P1 --ref analysis/phase1/runs/D1/D1 --T 10 \
    --out analysis/phase1/pilot_horizon --B 1000
```

`analyze_pilot.py` writes `pilot_results.json` and `pilot_tables.md`:

- **A. Correctness gate (must pass):**
  - `metadata.json` records `search_horizon = episode`;
  - 0 steps with a child state past Tmax (turn + depth > 10);
  - 0 selections at a state of length ≥ Tmax;
  - the simlog alignment check passes.
- **B. Paired outcomes:** SR and turns, per-dialogue success, first divergent system act.
- **C. Search structure:**
  - steps per tree, max depth, depth-1 share, visits per depth-2 node;
  - fresh share;
  - share of simulations that end at the horizon with −1;
  - wall clock.
- **D. P-VAR components, same scripts, same-dialogue D1 reference:**
  - NodeKey: discordance, Δ, ω² and edge-demeaned ω², boundary, bucket–depth r;
  - AffPool components;
  - channel-dominance flips;
  - σ ratio.

The analysis was dry-run on the stub-backbone outputs (`analyze_pilot.py` on
`--search_horizon episode` vs legacy, 3 dialogues). All sections render and the gate passes; the
stub's numbers carry no meaning.

## Disclosure, whatever P1 shows

Add P1's result to `SEARCH_HORIZON_BUG.md` §6. Update `phase1_report.md` §7 with whether the Phase-1
gate numbers move materially under the fix. If they do, D1/D2 need re-running under `episode` before
anyone decides on NodeKey/AffPool. That decision is not made here.
