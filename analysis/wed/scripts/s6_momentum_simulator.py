"""§6 -- the momentum hypothesis on simulator data (Table A), with the §6.3 mechanical-association controls.

    python analysis/wed/scripts/s6_momentum_simulator.py [--runs D1 D2 D3]

Outcome: backup_value (primary: the quantity that decides which act wins) and z (secondary).
delta_nu at the parent = parent_nu minus the previous user turn in the parent realization's own history
(Table A definitions); rows where it does not exist (depth 1 of the first planned turn) are dropped.
Runs are analysed separately and never averaged.

Per run x outcome x sample {all, fresh}:
  table        mean outcome per (delta_nu tercile, act), n, cluster-bootstrap CI (Table A terciles, per run)
  models       OLS M0 act / M1 +dnu / M2 +act:dnu, depth-binned control, F-corrected cluster Wald on the
               interaction, R^2 increments with cluster-bootstrap CIs; the same with parent_nu (level)
  did          [mean(propose, rising) - mean(propose, falling)] - [same for emotion appeal], bootstrap CI
  6.3 residual both delta_nu and the outcome residualized on depth dummies; interaction re-tested
  6.3 permute  delta_nu shuffled within depth stratum (and, stricter, within tree x depth), 1000 times;
               observed M1->M2 increment against the null
Verdict per run x outcome: the interaction is reported as present only if the residualized Wald p < 0.05,
the observed increment exceeds the 95th percentile of BOTH permutation nulls, and the DiD CI excludes 0.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, ci, constants, fit_models, load_steps, update_wednesday, with_ci, jdump  # noqa: E402

PROPOSE, EMO = "proposition of donation", "emotion appeal"
BUCKETS = ["falling", "flat", "rising"]
N_PERM = 1000


def depth_dummies(d):
	dc = d["depth"].clip(upper=10)
	return pd.get_dummies(dc, prefix="dep", drop_first=True).astype(float)


def residualize(d, cols):
	X = np.column_stack([np.ones(len(d)), depth_dummies(d).to_numpy()])
	out = d.copy()
	for c in cols:
		y = d[c].to_numpy(float)
		b, *_ = np.linalg.lstsq(X, y, rcond=None)
		out[c] = y - X @ b
	return out


def did(d, y):
	r = d.groupby(["act", "bucket"])[y].mean()
	try:
		return float((r[(PROPOSE, "rising")] - r[(PROPOSE, "falling")]) - (r[(EMO, "rising")] - r[(EMO, "falling")]))
	except KeyError:
		return np.nan


def inc12(d, y, x, acts, controls=()):
	return fit_models(d, y, x, controls=controls, family="ols", acts=acts, cluster="dialogue_id")["inc_M1_M2"]


def fast_inc12(y, Xbase, xcol, Aind):
	"""R^2(M2) - R^2(M1) for permutation: Xbase = [1, act dummies, controls]; Aind = act dummies."""
	X1 = np.column_stack([Xbase, xcol])
	X2 = np.column_stack([X1, Aind * xcol[:, None]])
	tss = np.sum((y - y.mean()) ** 2)
	r = []
	for X in (X1, X2):
		b, *_ = np.linalg.lstsq(X, y, rcond=None)
		r.append(1 - np.sum((y - X @ b) ** 2) / tss)
	return r[1] - r[0]


def analyse(d, y, boot, rng, cuts):
	d = d.copy()
	d["bucket"] = np.where(d["delta_nu"] < cuts[0], "falling", np.where(d["delta_nu"] <= cuts[1], "flat", "rising"))
	d["act"] = d["action"]
	# same act filter as §4: acts with < 30 rows or < 10 in a tercile (D3 "other": 15 rows) get separated
	# coefficients with spuriously tiny cluster SEs and drive the Wald p to ~1e-21 while permutation is null
	ct = pd.crosstab(d["act"], d["bucket"]).reindex(columns=BUCKETS, fill_value=0)
	keep = [a for a in ct.index if ct.loc[a].sum() >= 30 and ct.loc[a].min() >= 10]
	dropped = {a: int(ct.loc[a].sum()) for a in ct.index if a not in keep}
	d = d[d["act"].isin(keep)].reset_index(drop=True)
	d["dnu_z"] = (d["delta_nu"] - d["delta_nu"].mean()) / d["delta_nu"].std()
	d["nu_z"] = (d["parent_nu"] - d["parent_nu"].mean()) / d["parent_nu"].std()
	acts = list(d["act"].value_counts().index)
	dd = depth_dummies(d)
	for c in dd.columns:
		d[c] = dd[c].to_numpy()
	ctrl = tuple(dd.columns)
	out = {"excluded_acts": dropped, "n": len(d), "occupancy": d["bucket"].value_counts(normalize=True).reindex(BUCKETS).to_dict(),
		   "share_dnu_negative": float((d["delta_nu"] < 0).mean()), "share_dnu_below_-0.1": float((d["delta_nu"] < -0.1).mean())}
	# table with cluster-bootstrap CIs
	cell = d.groupby(["act", "bucket"])[y].agg(["size", "mean"])
	by = {k: g for k, g in d.groupby("dialogue_id")}
	reps_cell, reps_did = [], []
	for draw in boot.draws:
		b = pd.concat([by[boot.d[i]] for i in draw])
		reps_cell.append(b.groupby(["act", "bucket"])[y].mean())
		reps_did.append(did(b, y))
	R = pd.concat(reps_cell, axis=1)
	out["table"] = [{"act": a, "bucket": bk, "n": int(r["size"]), "mean": float(r["mean"]),
					 "ci": ci(R.loc[(a, bk)].to_numpy()) if (a, bk) in R.index else [np.nan, np.nan],
					 "interpretable": int(r["size"]) >= 30} for (a, bk), r in cell.iterrows()]
	out["did_propose_minus_emotion"] = with_ci(did(d, y), reps_did)
	# models (no control, depth control), delta and level
	models = {}
	for x in ("dnu_z", "nu_z"):
		for cname, cc in (("none", ()), ("depth", ctrl)):
			m = fit_models(d, y, x, controls=cc, family="ols", acts=acts)
			m.pop("M2_coefficients", None)
			models[f"{x}|{cname}"] = m
	# bootstrap R^2 increments for the two headline models
	for key in ("dnu_z|depth", "nu_z|depth"):
		x = key.split("|")[0]
		i01, i12 = [], []
		for bdf in boot.frame(d.drop(columns=["bucket"])):
			r = fit_models(bdf, y, x, controls=ctrl, family="ols", acts=acts)
			i01.append(r["inc_M0_M1"])
			i12.append(r["inc_M1_M2"])
		models[key]["inc_M0_M1_ci"], models[key]["inc_M1_M2_ci"] = ci(i01), ci(i12)
	out["models"] = models
	# 6.3 (1) residualize on depth
	res_d = residualize(d, ["dnu_z", y])
	m = fit_models(res_d, y, "dnu_z", family="ols", acts=acts)
	m.pop("M2_coefficients", None)
	out["residualized"] = m
	out["residualized"]["did_on_residuals"] = did(res_d.assign(bucket=d["bucket"].to_numpy()), y)
	# 6.3 (2) permutation within depth stratum, and within tree x depth
	yv = d[y].to_numpy(float)
	Aind = np.column_stack([(d["act"] == a).to_numpy(float) for a in acts[1:]])
	Xbase = np.column_stack([np.ones(len(d)), Aind, dd.to_numpy()])
	xv = d["dnu_z"].to_numpy(float)
	obs = fast_inc12(yv, Xbase, xv, Aind)
	perm = {}
	for name, strata in (("depth", d["depth"].clip(upper=10).to_numpy()),
						 ("tree_x_depth", (d["tree_id"] + "|" + d["depth"].astype(str)).to_numpy())):
		codes = pd.factorize(strata)[0]
		order = np.argsort(codes, kind="stable")
		bounds = np.flatnonzero(np.diff(codes[order])) + 1
		groups = np.split(order, bounds)
		null = []
		for _ in range(N_PERM):
			xp = xv.copy()
			for g in groups:
				if len(g) > 1:
					xp[g] = xv[rng.permutation(g)]
			null.append(fast_inc12(yv, Xbase, xp, Aind))
		null = np.array(null)
		perm[name] = {"observed_inc12": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
					  "p_perm": float((1 + np.sum(null >= obs)) / (1 + N_PERM))}
	out["permutation"] = perm
	out["verdict"] = {
		"residualized_wald_p": out["residualized"]["interaction_wald_cluster"]["p"],
		"perm_depth_p": perm["depth"]["p_perm"], "perm_tree_depth_p": perm["tree_x_depth"]["p_perm"],
		"did_ci": out["did_propose_minus_emotion"]["ci"],
		"level_wald_p_depth_controlled": models["nu_z|depth"]["interaction_wald_cluster"]["p"],
	}
	did_ci = out["did_propose_minus_emotion"]["ci"]
	out["verdict"]["interaction_present"] = bool(
		out["verdict"]["residualized_wald_p"] < 0.05 and perm["depth"]["p_perm"] < 0.05 and perm["tree_x_depth"]["p_perm"] < 0.05
		and (did_ci[0] > 0 or did_ci[1] < 0))
	out["verdict"]["direction_matches_H"] = bool(did_ci[0] > 0)
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--runs", nargs="*", default=None)
	a = ap.parse_args()
	allA = load_steps()
	const = constants()
	path = os.path.join(WED, "s6_momentum_simulator.json")
	out = {}
	if os.path.exists(path):
		import json
		out = json.load(open(path))
	for run in (a.runs or sorted(allA["run_id"].unique())):
		A = allA[(allA["run_id"] == run)].dropna(subset=["delta_nu"]).reset_index(drop=True)
		cuts = const["runs"][run]["delta_terciles"]
		boot = Boot(A["dialogue_id"].unique())
		rng = np.random.default_rng(616)
		out[run] = {"cuts": cuts}
		for y in ("backup_value", "z"):
			for label, sub in (("all", A), ("fresh", A[~A["child_from_cache"]])):
				r = analyse(sub, y, boot, rng, cuts)
				out[run][f"{y}|{label}"] = r
				print(run, y, label, r["verdict"], "inc12", round(r["models"]["dnu_z|depth"]["inc_M1_M2"], 5), flush=True)
			jdump(out, path)
	update_wednesday("s6_momentum_simulator", out)


if __name__ == "__main__":
	main()
