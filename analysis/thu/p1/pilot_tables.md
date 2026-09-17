## A. Correctness: PASS

- metadata_search_horizon: episode
- steps: 6360
- max_child_len: 10
- steps_child_len_gt_T: 0
- steps_parent_len_ge_T: 0
- alignment_check: RESULT: PASS
- pass: True

## B. Paired outcomes (descriptive, n = 10)

| | D1 legacy, same dialogues | P1 episode |
|---|---|---|
| SR | 0.600 | 1.000 |
| mean turns | 6.900 | 5.400 |
| discordant pairs (success only in legacy / only in episode) | 0 | 4 |

| dialogue | legacy success / turns | episode success / turns | first divergent system turn |
|---|---|---|---|
| 20180828-144300_705_live | False / 10 | True / 5 | 1 |
| 20180828-145857_925_live | True / 5 | True / 4 | 2 |
| 20180828-155340_989_live | True / 4 | True / 6 | 1 |
| 20180828-161442_405_live | False / 10 | True / 6 | 1 |
| 20180828-163111_795_live | True / 8 | True / 8 | 2 |
| 20180828-164559_390_live | True / 5 | True / 5 | None |
| 20180828-185320_572_live | False / 10 | True / 5 | 2 |
| 20180828-190212_442_live | True / 3 | True / 5 | 2 |
| 20180828-190447_206_live | True / 7 | True / 4 | 1 |
| 20180828-213210_462_live | False / 7 | True / 6 | 1 |

## C. Search structure

| | D1 legacy, same dialogues | P1 episode |
|---|---|---|
| steps | 9372 | 6360 |
| trees | 59 | 44 |
| steps_per_tree | 158.847 | 144.545 |
| share_steps_child_len_gt_T | 0.123 | 0.000 |
| share_steps_child_len_eq_T | 0.070 | 0.009 |
| share_simulations_ending_at_horizon_failure | 0.003 | 0.019 |
| max_depth | 20 | 9 |
| share_visits_depth1 | 0.308 | 0.339 |
| median_visits_parent_node_depth2 | 9.000 | 9.000 |
| fresh_share | 0.560 | 0.543 |
| run_wall_clock | wall_clock_s=4003 | wall_clock_s=2306 |

## D. P-VAR components (median split; dialogue-clustered 95 % CI)

| | D1 legacy, same dialogues | P1 episode |
|---|---|---|
| tau_med | 0.404 | 0.279 |
| discordance | 0.198 [0.143, 0.273] | 0.299 [0.273, 0.329] |
| delta_mean | -0.051 [-0.073, -0.030] | -0.019 [-0.042, 0.004] |
| std_effect | -0.246 | -0.098 |
| omega2 | 0.151 [0.081, 0.190] | 0.035 [0.023, 0.064] |
| omega2_edge_demeaned | 0.010 | -0.002 |
| boundary_0.1 | 0.256 | 0.275 |
| bucket_depth_r | -0.221 [-0.284, -0.135] | -0.196 [-0.275, -0.136] |
| affpool_n_prefixes | 3.000 | 3.000 |
| affpool_multiplier | 2.382 | 2.000 |
| affpool_top_share | 0.600 | 0.706 |
| affpool_ratio | 0.714 | 0.983 |
| flip_full_puct | 0.165 | 0.119 |
| flip_exploit_expanded | 0.164 | 0.079 |
| flip_visited_over_unvisited | 0.379 | 0.435 |
| sigma_ratio_median | 0.404 | 0.442 |
