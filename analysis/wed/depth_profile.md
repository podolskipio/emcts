# §3 Depth profile

From `steps.parquet`. Depth is edge depth (the root's outgoing edges are depth 1). "Median visits per parent node" is the median visit count of the parent nodes at that depth. ν is `parent_nu`. `reachable` is the share of steps whose child state has turn + depth ≤ Tmax (`analysis/phase1/SEARCH_HORIZON_BUG.md`).


## D1 — 33,903 steps, 175 trees, max depth 33

**74.7% of visits at depth ≥ 2; depth-2 parent nodes get a median of 9.5 visits.**

| depth | visits | share | edges | median visits / parent node | ν q25 | ν median | ν q75 | reachable |
|---|---|---|---|---|---|---|---|---|
| 1 | 8,575 | 25.3% | 650 | 49 | 0.02 | 0.24 | 0.44 | 100% |
| 2 | 7,284 | 21.5% | 1,005 | 9.5 | 0.04 | 0.24 | 0.45 | 94% |
| 3 | 5,632 | 16.6% | 987 | 5 | 0.08 | 0.32 | 0.48 | 87% |
| 4 | 3,882 | 11.5% | 770 | 4 | 0.16 | 0.41 | 0.51 | 79% |
| 5 | 2,597 | 7.7% | 568 | 4 | 0.19 | 0.43 | 0.52 | 71% |
| 6 | 1,761 | 5.2% | 399 | 3 | 0.30 | 0.47 | 0.53 | 61% |
| 7–10 | 3,017 | 8.9% | 711 | 3 | 0.37 | 0.51 | 0.53 | 30% |
| 11+ | 1,155 | 3.4% | 265 | 3 | 0.52 | 0.53 | 0.53 | 0% |

## D2 — 28,129 steps, 177 trees, max depth 24

**69.2% of visits at depth ≥ 2; depth-2 parent nodes get a median of 7.5 visits.**

| depth | visits | share | edges | median visits / parent node | ν q25 | ν median | ν q75 | reachable |
|---|---|---|---|---|---|---|---|---|
| 1 | 8,673 | 30.8% | 738 | 49 | 0.04 | 0.25 | 0.45 | 100% |
| 2 | 7,304 | 26.0% | 1,309 | 7.5 | 0.03 | 0.23 | 0.46 | 95% |
| 3 | 5,175 | 18.4% | 1,234 | 4 | 0.11 | 0.33 | 0.50 | 88% |
| 4 | 3,011 | 10.7% | 831 | 3 | 0.14 | 0.42 | 0.52 | 81% |
| 5 | 1,597 | 5.7% | 498 | 3 | 0.24 | 0.48 | 0.53 | 73% |
| 6 | 855 | 3.0% | 282 | 3 | 0.32 | 0.50 | 0.53 | 60% |
| 7–10 | 1,143 | 4.1% | 327 | 2 | 0.44 | 0.53 | 0.53 | 23% |
| 11+ | 371 | 1.3% | 93 | 3 | 0.50 | 0.53 | 0.54 | 0% |

## D3 — 37,700 steps, 153 trees, max depth 27

**80.1% of visits at depth ≥ 2; depth-2 parent nodes get a median of 18 visits.**

| depth | visits | share | edges | median visits / parent node | ν q25 | ν median | ν q75 | reachable |
|---|---|---|---|---|---|---|---|---|
| 1 | 7,497 | 19.9% | 445 | 49 | -0.02 | 0.07 | 0.34 | 100% |
| 2 | 6,309 | 16.7% | 588 | 18 | -0.02 | 0.09 | 0.29 | 96% |
| 3 | 4,962 | 13.2% | 519 | 10 | -0.01 | 0.10 | 0.29 | 90% |
| 4 | 3,852 | 10.2% | 432 | 8 | -0.03 | 0.09 | 0.31 | 83% |
| 5 | 2,973 | 7.9% | 364 | 6 | -0.02 | 0.07 | 0.30 | 74% |
| 6 | 2,364 | 6.3% | 281 | 7 | -0.03 | 0.07 | 0.30 | 65% |
| 7–10 | 5,815 | 15.4% | 709 | 7 | -0.03 | 0.08 | 0.33 | 33% |
| 11+ | 3,928 | 10.4% | 657 | 5 | -0.04 | 0.06 | 0.29 | 0% |

## Against yesterday

- **D1:** depth ≥ 2 share 74.7% (brief: 75%); depth-2 parent median 9.5 (P-VAR: 9.5; brief: ~8–10). Confirmed.
- **D2:** depth ≥ 2 share 69.2% (brief: 69%); depth-2 parent median 7.5 (P-VAR: 7.5; brief: ~8–10). Confirmed.
- **D3 (Qwen):** depth ≥ 2 share 80.1%; depth-2 parent median 18; max depth 27.
