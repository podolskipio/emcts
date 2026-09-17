# Reproducing Phase 1 end to end

```
# 0. data set (dialogues 101-130 of the non-annotated pool)
python analysis/phase1/scripts/build_dialogue_set.py
# 1. §4 bit-identity (stub backbone; PRE = a copy of src/ before the instrumentation, with data/ symlinked beside it)
python tests/run_e2e_regression.py --src $PRE/src --out $OUT/r4beta07_pre.pkl --dialogs 3 --sims 50 --seed 0 --beta_emo 0.7 --extra <flags in instrumentation_check.md>
python tests/run_e2e_regression.py --src src      --out $OUT/r4beta07_post.pkl ...same...
python tests/compare_regression.py $OUT r4beta07 && python analysis/phase1/scripts/check_instrumentation.py $OUT/r4beta07_pre $OUT/r4beta07_post
# 2. runs (SGLang up via analysis/calib/serve.sh 13b)
analysis/phase1/scripts/run_diag.sh D1 0.7 30
analysis/phase1/scripts/run_diag.sh D2 0.0 30
# 3. per run: alignment check, flat table, analysis (+ figures)
python analysis/phase1/scripts/check_instrumentation.py - analysis/phase1/runs/D1/D1
python analysis/phase1/scripts/build_steps.py analysis/phase1/runs/D1/D1 analysis/phase1/D1
python analysis/phase1/scripts/p_var.py D1 analysis/phase1/D1 --beta 0.7 --B 1000 --figdir analysis/phase1/figures
python analysis/phase1/scripts/build_steps.py analysis/phase1/runs/D2/D2 analysis/phase1/D2
python analysis/phase1/scripts/p_var.py D2 analysis/phase1/D2 --beta 0.0 --beta_measure 0.7 --B 1000 --figdir analysis/phase1/figures
# 4. §11 summary JSON
python analysis/phase1/scripts/assemble_p_var.py --runs D1 D2
# 5. P-RELABEL (CPU)
CUDA_VISIBLE_DEVICES="" python analysis/phase1/scripts/p_relabel.py
```
`analysis/phase1/steps.parquet` is D1's table (the primary); D2's is `analysis/phase1/D2/steps.parquet`.

## Search-horizon bug (`SEARCH_HORIZON_BUG.md`)

```
python -m pytest tests/test_search_horizon.py -q
# sensitivity: drop steps whose child is past Tmax, same analysis
python analysis/phase1/scripts/build_steps.py analysis/phase1/runs/D1/D1 analysis/phase1/horizon_sensitivity/D1 --max_child_len 10
python analysis/phase1/scripts/p_var.py D1 analysis/phase1/horizon_sensitivity/D1 --beta 0.7 --B 1000
python analysis/phase1/scripts/build_steps.py analysis/phase1/runs/D2/D2 analysis/phase1/horizon_sensitivity/D2 --max_child_len 10
python analysis/phase1/scripts/p_var.py D2 analysis/phase1/horizon_sensitivity/D2 --beta 0.0 --beta_measure 0.7 --B 1000
python analysis/phase1/scripts/assemble_p_var.py --runs D1 D2
# pilot P1 (Thursday): see analysis/phase1/pilot_horizon/README.md
```
