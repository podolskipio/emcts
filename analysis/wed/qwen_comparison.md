# §8 D3: Qwen2.5-7B vs Vicuna-13B, paired on the same 30 dialogues

**Status: D3 landed 2026-09-15 18:16. 30/30 dialogues, no failures, 3,579 s = 0.99 GPU-h.**
Log integrity: `check_instrumentation.py` PASS on all 30 (z tape = frozen per-step valences, every parent
realization traceable, z = child ν); `backup_value` native and replay-verified on 37,700/37,700 steps.

Config: `Qwen/Qwen2.5-7B-Instruct-AWQ` serving every role, `beta_emo 0.7`, dialogues 101–130 (same ids
and personas as D1), n_sims 50, Tmax 10, R 4, K 5, HF classifier, `--emo_signal level`, soft table,
sampled value/prior (`--logit_scoring off`), seed 0, 10 workers, `search_horizon legacy`. Exact
command: `runs/D3/command.txt`. Only the backbone differs from D1.

## Headline

1. **Qwen does not drift toward happiness with depth; Vicuna does.** Median parent ν by depth
   1 / 2 / 3 / 5 / 8 / 10+: Vicuna **0.24 / 0.24 / 0.32 / 0.44 / 0.50 / 0.53**; Qwen **0.07 / 0.09 / 0.10 /
   0.07 / 0.07 / 0.06**. The "users become uniformly happy deeper in the tree" pattern, which yesterday made
   K0 a phase proxy (r = −0.29), is a Vicuna property, not a property of LLM simulators.
2. **Qwen is less positive but not more expressive.** Paired on dialogue (D3 − D1, cluster bootstrap):
   parent-realization ν −0.135 [−0.161, −0.104]; real user turns −0.073 [−0.122, −0.020]. The spread is
   the same (parent ν SD 0.194 vs 0.209; per-dialogue SD −0.011 [−0.022, −0.001]; real-turn SD −0.001
   [−0.030, +0.028]), and so is the negative-label rate (real turns 9.3 % vs 8.8 %, paired +1.2 pp
   [−4.4, +6.5]; parent realizations 6.9 % vs 5.6 %). The shift is happiness → neutral, not → negative:
   real-turn labels Qwen neutral 52 % / happiness 36 % / fear 8 %; Vicuna neutral 39 % / happiness 52 % / fear 5 %.
   **The expectation in the brief ("Qwen far more expressive") is not borne out on this classifier and table.**
3. **σ_emo / |Q_emo| more than doubles (median 0.45 → 1.13; paired +0.65 [0.55, 0.74]), but because |Q_emo|
   shrank, not because the spread grew.** With ν centred near 0 the running mean sits near 0; σ is
   unchanged (z SD 0.196 vs 0.199). On Qwen the affect channel's signal is small against its own noise.
4. **Δν occupancy is equal:** 24.8 % (Qwen) vs 24.7 % (Vicuna) of steps fall by more than 0.1; terciles
   −0.053 / +0.077 vs −0.035 / +0.085. Exact Δν ties drop from 7.1 % to 0.7 % (Qwen repeats itself verbatim far less).

## Paired table (30 dialogues, D3 − D1, 95 % cluster-bootstrap CI)

| metric | Vicuna D1 | Qwen D3 | D3 − D1 |
|---|---|---|---|
| SR | 0.60 | 0.80 | +0.20 [−0.00, +0.43] |
| turns | 6.83 | 6.10 | −0.73 [−2.20, +0.63] |
| real user ν, mean | 0.256 | 0.183 | **−0.073** [−0.122, −0.020] |
| real user ν, SD within dialogue | 0.179 | 0.178 | −0.001 [−0.030, +0.028] |
| real-turn negative-label rate | 0.089 | 0.100 | +0.012 [−0.044, +0.065] |
| parent-realization ν, mean | 0.286 | 0.151 | **−0.135** [−0.161, −0.104] |
| parent-realization ν, SD | 0.194 | 0.183 | −0.011 [−0.022, −0.001] |
| parent negative-label rate | 0.067 | 0.086 | +0.019 [−0.010, +0.049] |
| SD of z | 0.199 | 0.196 | −0.003 [−0.012, +0.006] |
| share of steps with Δν < −0.1 | 0.267 | 0.241 | −0.026 [−0.059, +0.011] |
| σ_emo/\|Q_emo\| (edges N ≥ 2), median | 0.48 | 1.13 | **+0.65** [0.55, 0.74] |

SR is an n = 30 diagnostic, not a model comparison.

## Tree shape

Qwen trees are **deeper and narrower**: 80 % of visits at depth ≥ 2 (Vicuna 75 %), depth-2 parent nodes
get a median **18** visits (Vicuna 9.5), 3,995 edges vs 5,355. The act mix concentrates further on
emotion appeal (69 % of steps vs 64 %) and proposition (24 % vs 16 %). **The search-horizon contamination is
worse:** only 71 % of D3 steps are reachable (D1 80 %), and 10.4 % of visits sit at depth ≥ 11. That is
relevant to Thursday's P1 pilot (`SEARCH_HORIZON_BUG.md`).

## Momentum on Qwen

See `momentum_simulator.md` for D3's interaction table, M1→M2 increment and §6.3 controls.

Machine-readable: `s8_qwen_comparison.json`. Script: `scripts/s8_qwen_comparison.py`.
