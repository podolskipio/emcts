"""Readiness Phase 3B -- does a node's emotional SPREAD carry value information? (PREREG Entry 10)

    python analysis/readiness/scripts/phase3_spread.py

Unit: a node = an edge (tree, prefix, action). Its realizations are the distinct child replies generated
there; spread = sd of their nu, mean_nu = their mean. Outcome: the node's Q = mean backed-up task value
over its visits. Kept: >= 2 distinct realizations and >= 2 visits.

Step 2  OLS  Q ~ depth FE + n_real + mean_nu  vs  + spread; incremental R2, cluster bootstrap.
Step 3  spread's distribution, and whether it differs between SIBLING nodes more than a permutation null
        that shuffles the realizations' nu across each parent's siblings.
Primary: frozen grid A_NoEmo_s20_seed{1,2,3}; secondary: phase1 D1, D2. -> phase3_spread.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _env as E  # noqa: E402

sys.path.insert(0, os.path.join(E.REPO, "analysis", "phase1", "scripts"))
import build_steps  # noqa: E402

B, N_PERM, SEED = 1000, 200, 20260926
GRID = os.path.join(E.REPO, "analysis", "grid", "runs")
PRIMARY = {f"F_NoEmo_s20_seed{s}": os.path.join(GRID, f"A_NoEmo_s20_seed{s}", f"A_NoEmo_s20_seed{s}") for s in (1, 2, 3)}
SECONDARY = {"D1": os.path.join(E.REPO, "analysis", "phase1", "D1", "steps.parquet"),
			 "D2": os.path.join(E.REPO, "analysis", "phase1", "D2", "steps.parquet")}
OUT = os.path.join(E.READINESS, "phase3_spread.json")


def load_primary():
	parts = []
	for tag, run_dir in PRIMARY.items():
		out = os.path.join(E.READINESS, "steps", tag)
		if not os.path.exists(os.path.join(out, "steps.parquet")):
			build_steps.main(run_dir, out)
		parts.append(pd.read_parquet(os.path.join(out, "steps.parquet")).assign(run=tag))
	return pd.concat(parts, ignore_index=True)


def nodes(steps):
	"""One row per kept node."""
	s = steps.copy()
	if "run" not in s:
		s["run"] = "one"
	s["value"] = s["backup_value"] if "backup_value" in s else s["v"]
	s["edge"] = s["run"] + "|" + s["edge_key"]
	s["parent"] = s["run"] + "|" + s["tree_key"] + "|" + s["prefix_key"]
	s["cluster"] = s["run"] + "|" + s["dlg_id"]
	real = s.drop_duplicates(["edge", "child_realization_id"])
	g = real.groupby("edge")["child_nu"]
	n = pd.DataFrame({"n_real": g.size(), "mean_nu": g.mean(), "spread": g.std(ddof=1)})
	v = s.groupby("edge").agg(Q=("value", "mean"), visits=("value", "size"), depth=("depth", "first"),
							  parent=("parent", "first"), cluster=("cluster", "first"))
	n = n.join(v)
	n = n[(n.n_real >= 2) & (n.visits >= 2)].reset_index()
	return n, real[real.edge.isin(set(n.edge))][["edge", "parent", "child_nu"]]


def design(n, cols):
	d = pd.get_dummies(n["depth"].clip(upper=5).astype(int), prefix="d", drop_first=True, dtype=float)
	X = np.column_stack([np.ones(len(n)), d.to_numpy(), n[cols].to_numpy(float)])
	return X


def r2(y, X):
	beta, *_ = np.linalg.lstsq(X, y, rcond=None)
	res = y - X @ beta
	return 1 - res.var() / y.var(), beta


def incr(n, within_parent=False):
	m = n
	if within_parent:
		cols = ["Q", "mean_nu", "spread", "n_real"]
		m = n[n.groupby("parent")["edge"].transform("size") >= 2].copy()
		m[cols] = m[cols] - m.groupby("parent")[cols].transform("mean")
	y = m["Q"].to_numpy(float)
	base, full = design(m, ["n_real", "mean_nu"]), design(m, ["n_real", "mean_nu", "spread"])
	r_b, _ = r2(y, base)
	r_f, beta = r2(y, full)
	return m, float(r_f - r_b), float(beta[-1] * m["spread"].std() / m["Q"].std())


def boot_incr(n, within_parent=False):
	m, point, std_coef = incr(n, within_parent)
	rng = np.random.default_rng(SEED + within_parent)
	idx = list(n.groupby("cluster").indices.values())
	reps = []
	for _ in range(B):
		pick = rng.integers(0, len(idx), len(idx))
		sub = n.iloc[np.concatenate([idx[i] for i in pick])].reset_index(drop=True)
		# resampled clusters can repeat: keep parents distinct per copy for the within-parent version
		sub["parent"] = sub["parent"] + "#" + pd.Series(np.repeat(np.arange(len(pick)), [len(idx[i]) for i in pick])).astype(str)
		reps.append(incr(sub, within_parent)[1])
	return {"rows": int(len(m)), "increment": point, "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
			"std_coef_spread": std_coef, "PASS": bool(np.percentile(reps, 2.5) > 0)}


def sibling_variance(real, keep_parents):
	"""Mean over parents of the variance of spread among its sibling nodes."""
	r = real[real.parent.isin(keep_parents)]
	sp = r.groupby(["parent", "edge"])["child_nu"].std(ddof=1)
	return float(sp.groupby(level="parent").var(ddof=1).mean())


def discrimination(n, real):
	kids = n.groupby("parent")["edge"].size()
	keep = set(kids[kids >= 2].index)
	obs = sibling_variance(real, keep)
	rng = np.random.default_rng(SEED + 7)
	r = real[real.parent.isin(keep)].copy()
	null = []
	for _ in range(N_PERM):
		r["child_nu"] = r.groupby("parent")["child_nu"].transform(lambda s: rng.permutation(s.to_numpy()))
		null.append(sibling_variance(r, keep))
	null = np.array(null)
	ratio = obs / null.mean()
	p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
	return {"parents_with_ge2_kept_siblings": len(keep), "observed_sibling_var_of_spread": obs,
			"null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
			"ratio": float(ratio), "perm_p": p, "PASS": bool(ratio >= 1.2 and p < 0.05)}


def distribution(n, real):
	sp = n["spread"]
	glob_sd = float(real["child_nu"].std(ddof=1))
	return {"nodes": int(len(n)), "n_real_median": float(n.n_real.median()),
			"spread_quantiles": {q: float(sp.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
			"spread_mean": float(sp.mean()), "spread_sd": float(sp.std()), "spread_cv": float(sp.std() / sp.mean()),
			"global_nu_sd": glob_sd, "median_spread_over_global_sd": float(sp.median() / glob_sd),
			"corr_spread_mean_nu": float(np.corrcoef(sp, n.mean_nu)[0, 1])}


def analyse(steps):
	n, real = nodes(steps)
	return {"distribution": distribution(n, real), "step2_incremental_r2": boot_incr(n),
			"step2_within_parent": boot_incr(n, within_parent=True), "step3_discrimination": discrimination(n, real)}


def main():
	res = {"git_head": E.git_head()}
	res["primary_NoEmo_s20_seeds123"] = a = analyse(load_primary())
	a["PASS"] = bool(a["step2_incremental_r2"]["PASS"] and a["step3_discrimination"]["PASS"])
	for tag, path in SECONDARY.items():
		print(tag, flush=True)
		res[f"secondary_{tag}"] = analyse(pd.read_parquet(path))
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
