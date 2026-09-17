"""§7 key geometry, Part 1 (Table A only).

    python analysis/wed/scripts/s7_key_geometry.py [--runs D1 D2]

7.1  does the 7-vector carry information about the next reaction beyond nu = w.d ?
7.2  effective dimensionality + within-tree neighbourhood density of parent d vectors
7.3  distance-metric sanity (computed only if 7.1 finds orthogonal information)
7.4  kernel pooling on the scalar nu, h in {0.05, 0.10, 0.20, 0.40}
7.5  2-d kernel on (nu, delta_nu)

Kernel statistics are computed PER QUERY STEP inside its (tree, act) group, and the hard K0 partition is
run through the same per-query code (kernel := 1[same bucket]) so the comparison is like-for-like.
Yesterday's cell-level numbers (2.59 / 0.31 / -0.29) are quoted beside them; they aggregate differently.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, ci, load_steps, update_wednesday, with_ci, wmedian, jdump  # noqa: E402

EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
W_SOFT = {"happiness": 0.54, "sadness": -0.14, "fear": 0.24, "anger": 0.11, "surprise": 0.09, "disgust": 0.13, "neutral": -0.07}
HS = [0.05, 0.10, 0.20, 0.40]


def r2(X, y):
	X = np.column_stack([np.ones(len(y)), X])
	b, *_ = np.linalg.lstsq(X, y, rcond=None)
	return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2), b


# ------------------------------------------------------------------------------------------ 7.1
def s71(A, boot):
	D = np.vstack(A["parent_emotion_dist"].to_numpy())
	res = {}
	for label, mask in (("all", np.ones(len(A), bool)), ("fresh", ~A["child_from_cache"].to_numpy())):
		for outcome in ("z", "backup_value"):
			sub = A[mask]
			Dm = D[mask]
			X6 = Dm[:, :6]  # neutral is the reference (the 7 columns sum to 1)

			def fit(idx):
				y = sub[outcome].to_numpy()[idx]
				nu = sub["parent_nu"].to_numpy()[idx][:, None]
				dep = sub["depth"].to_numpy(float)[idx][:, None]
				r_nu, _ = r2(nu, y)
				r_d, b = r2(X6[idx], y)
				r_nud, _ = r2(np.hstack([nu, dep]), y)
				r_dd, _ = r2(np.hstack([X6[idx], dep]), y)
				return {"nu": r_nu, "d7": r_d, "nu_depth": r_nud, "d7_depth": r_dd}, b
			all_idx = np.arange(len(sub))
			pt, b = fit(all_idx)
			pos = {d: np.where(sub["dialogue_id"].to_numpy() == d)[0] for d in sub["dialogue_id"].unique()}
			reps = []
			for draw in boot.draws:
				idx = np.concatenate([pos[boot.d[i]] for i in draw if boot.d[i] in pos])
				reps.append(fit(idx)[0])
			coefs = {e: float(b[1 + i]) for i, e in enumerate(EMO7[:6])}
			coefs["neutral"] = 0.0
			w_rel = {e: W_SOFT[e] - W_SOFT["neutral"] for e in EMO7}
			ce = np.array([coefs[e] for e in EMO7])
			we = np.array([w_rel[e] for e in EMO7])
			res[f"{label}|{outcome}"] = {
				"n": len(sub),
				"r2": {k: with_ci(v, [r[k] for r in reps]) for k, v in pt.items()},
				"increment_d7_over_nu": with_ci(pt["d7"] - pt["nu"], [r["d7"] - r["nu"] for r in reps]),
				"increment_d7_over_nu_with_depth": with_ci(pt["d7_depth"] - pt["nu_depth"], [r["d7_depth"] - r["nu_depth"] for r in reps]),
				"d_coefficients_vs_neutral": coefs, "w_soft_vs_neutral": w_rel,
				"corr_coefficients_with_w": float(np.corrcoef(ce, we)[0, 1]),
				"projection_share": float((ce @ we) ** 2 / ((ce @ ce) * (we @ we))),
			}
	return res


# ------------------------------------------------------------------------------------------ 7.2
def s72(A, rng):
	U = A.drop_duplicates(["tree_id", "parent_realization_id"])
	D = np.vstack(U["parent_emotion_dist"].to_numpy())
	C = np.cov(D.T)
	lam = np.sort(np.linalg.eigvalsh(C))[::-1]
	ev = lam / lam.sum()
	maxd = D.max(1)
	Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
	trees = U["tree_id"].to_numpy()
	pick = rng.choice(len(U), size=min(200, len(U)), replace=False)
	counts = {0.1: [], 0.2: []}
	counts_steps = {0.1: [], 0.2: []}
	steps_per_real = A.groupby(["tree_id", "parent_realization_id"]).size()
	for i in pick:
		same = np.where(trees == trees[i])[0]
		same = same[same != i]
		dist = 1 - Dn[same] @ Dn[i]
		w = steps_per_real.loc[list(zip(U["tree_id"].to_numpy()[same], U["parent_realization_id"].to_numpy()[same]))].to_numpy()
		for t in (0.1, 0.2):
			counts[t].append(int((dist <= t).sum()))
			counts_steps[t].append(int(w[dist <= t].sum()))
	return {
		"unique_parent_realizations": len(U),
		"variance_explained": ev[:3].tolist(), "cumulative_3": float(ev[:3].sum()),
		"participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
		"max_d_quantiles": {q: float(np.quantile(maxd, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
		"share_max_d_above_0.9": float((maxd > 0.9).mean()),
		"neighbours_within_tree_realizations": {str(t): {"median": float(np.median(v)), "p25": float(np.quantile(v, .25)),
														 "p75": float(np.quantile(v, .75)), "share_below_5": float(np.mean(np.array(v) < 5))}
												for t, v in counts.items()},
		"neighbours_within_tree_steps": {str(t): {"median": float(np.median(v)), "share_below_5": float(np.mean(np.array(v) < 5))}
										 for t, v in counts_steps.items()},
	}


# ------------------------------------------------------------------------------------------ 7.3
def s73(A, rng, n_pairs=200000):
	D = np.vstack(A["parent_emotion_dist"].to_numpy())
	y = A["z"].to_numpy()
	nu = A["parent_nu"].to_numpy()
	trees = A["tree_id"].to_numpy()
	out = {}
	ii, jj = [], []
	for idx in pd.Series(np.arange(len(A))).groupby(trees).apply(np.array):
		if len(idx) < 2:
			continue
		k = max(1, n_pairs // A["tree_id"].nunique())
		a, b = rng.choice(idx, k), rng.choice(idx, k)
		keep = a != b
		ii.append(a[keep]); jj.append(b[keep])
	i, j = np.concatenate(ii), np.concatenate(jj)
	Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
	dy = np.abs(y[i] - y[j])
	for name, dist in (("euclidean", np.linalg.norm(D[i] - D[j], axis=1)), ("cosine", 1 - np.sum(Dn[i] * Dn[j], 1)),
					   ("w_weighted", np.abs(nu[i] - nu[j]))):
		out[name] = float(np.corrcoef(dist, dy)[0, 1])
	out["pairs"] = int(len(i))
	return out


# ------------------------------------------------------------------------------------------ 7.4 / 7.5
def per_query(A, value, kernel_fn):
	"""kernel_fn(group) -> n x n weight matrix. Returns per-query stats frame."""
	rows = []
	for (tree, act), g in A.groupby(["tree_id", "action"], sort=False):
		n = len(g)
		W = kernel_fn(g)
		y = g[value].to_numpy()
		pref, pcode = np.unique(g["action_prefix"].to_numpy(), return_inverse=True)
		P = np.zeros((n, len(pref)))
		P[np.arange(n), pcode] = 1
		Kp = W @ P
		Kp2 = (W ** 2) @ P  # for effective counts: n_eff = Kp^2 / Kp2 (= the count when W is 0/1)
		Sy = W @ (P * y[:, None])
		Sy2 = W @ (P * (y ** 2)[:, None])
		N_pool = W.sum(1)
		eff_pref = N_pool ** 2 / np.maximum((Kp ** 2).sum(1), 1e-12)
		depth = g["depth"].to_numpy(float)
		pool_depth = (W @ depth) / np.maximum(N_pool, 1e-12)
		edge_N = g["edge_N"].to_numpy(float)
		for q in range(n):
			k = Kp[q]
			ok = k >= 1.0
			rec = {"tree_id": tree, "dialogue_id": g["dialogue_id"].iloc[0], "action": act, "depth": depth[q],
				   "N_pool": N_pool[q], "multiplier": N_pool[q] / edge_N[q], "eff_prefixes": eff_pref[q],
				   "pool_mean_depth": pool_depth[q], "n_pref_ge1": int(ok.sum())}
			if ok.sum() >= 3:
				# same estimator as lib.affpool_cells: unweighted variance of prefix means over the pooled
				# within-prefix variance; noise floor mean(within / n_eff). With 0/1 weights it IS that.
				kk, k2 = k[ok], Kp2[q, ok]
				m = Sy[q, ok] / kk
				neff = kk ** 2 / k2
				between = float(np.var(m, ddof=1))
				ssw = np.sum(Sy2[q, ok] - Sy[q, ok] ** 2 / kk)   # weighted SS about each prefix mean
				dfw = np.sum(kk - k2 / kk)                       # reliability-weight degrees of freedom
				if dfw > 0 and ssw > 0:
					within = ssw / dfw
					rec["ratio"] = between / within
					rec["ratio_nc"] = (between - float(np.mean(within / neff))) / within
			rows.append(rec)
	return pd.DataFrame(rows)


def summarise_queries(Q, boot):
	by = {k: g for k, g in Q.groupby("dialogue_id")}

	def summ(q):
		h = q.dropna(subset=["ratio"]) if "ratio" in q else q.iloc[:0]
		# every query is one visit, so a plain median over queries is already visit-weighted
		return (float(q["N_pool"].median()), float(q["multiplier"].median()), float(q["eff_prefixes"].median()),
				float(h["ratio"].median()) if len(h) else np.nan,
				float(h["ratio_nc"].median()) if len(h) else np.nan,
				float(np.corrcoef(q["depth"], q["pool_mean_depth"])[0, 1]))
	pt = summ(Q)
	reps = boot.run(lambda ids: summ(pd.concat([by[i] for i in ids if i in by])))
	names = ["N_pool_median", "multiplier_median", "effective_prefixes_median", "homogeneity_ratio_median",
			 "homogeneity_nc_median", "r_query_depth_pool_depth"]
	out = {n: with_ci(pt[i], [r[i] for r in reps]) for i, n in enumerate(names)}
	out["N_pool_quantiles"] = {q: float(Q["N_pool"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
	out["queries_with_ge3_prefixes"] = float(Q["ratio"].notna().mean()) if "ratio" in Q else 0.0
	return out


def s74_75(A, boot, value):
	A = A.copy()
	A["edge_N"] = A["edge_id"].map(A.groupby("edge_id").size())
	res = {}
	tau = A["tau_med"].iloc[0]
	hard = lambda g: (lambda b: (b[:, None] == b[None, :]).astype(float))((g["parent_nu"].to_numpy() < tau))
	res["hard_K0"] = summarise_queries(per_query(A, value, hard), boot)
	for h in HS:
		ker = lambda g, h=h: np.exp(-((g["parent_nu"].to_numpy()[:, None] - g["parent_nu"].to_numpy()[None, :]) ** 2) / h ** 2)
		res[f"rbf_h{h}"] = summarise_queries(per_query(A, value, ker), boot)
		print("  7.4", value, h, {k: round(v["value"], 3) for k, v in res[f"rbf_h{h}"].items() if isinstance(v, dict) and "value" in v})
	# 7.5: 2-d on (nu, delta_nu), h_delta = h * sd(delta_nu) / sd(nu)
	B = A.dropna(subset=["delta_nu"]).copy()
	ratio = B["delta_nu"].std() / B["parent_nu"].std()
	res["hard_K0_delta_rows"] = summarise_queries(per_query(B, value, hard), boot)
	for h in HS:
		hd = h * ratio

		def ker2(g, h=h, hd=hd):
			nu = g["parent_nu"].to_numpy()
			dn = g["delta_nu"].to_numpy()
			return np.exp(-((nu[:, None] - nu[None, :]) / h) ** 2 - ((dn[:, None] - dn[None, :]) / hd) ** 2)
		res[f"rbf2d_h{h}"] = summarise_queries(per_query(B, value, ker2), boot)
		res[f"rbf2d_h{h}"]["h_delta"] = hd
		print("  7.5", value, h, {k: round(v["value"], 3) for k, v in res[f"rbf2d_h{h}"].items() if isinstance(v, dict) and "value" in v})
	res["r_nu_depth"] = float(np.corrcoef(A["parent_nu"], A["depth"])[0, 1])
	return res


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--runs", nargs="*", default=["D1", "D2"])
	a = ap.parse_args()
	allA = load_steps()
	out = {}
	for run in a.runs:
		A = allA[allA["run_id"] == run].reset_index(drop=True)
		boot = Boot(A["dialogue_id"].unique())
		rng = np.random.default_rng(7)
		r = {"s7_1": s71(A, boot)}
		print(run, "7.1", {k: (round(v["increment_d7_over_nu"]["value"], 4), [round(x, 4) for x in v["increment_d7_over_nu"]["ci"]],
							   round(v["corr_coefficients_with_w"], 2)) for k, v in r["s7_1"].items()})
		r["s7_2"] = s72(A, rng)
		print(run, "7.2", r["s7_2"]["participation_ratio"], r["s7_2"]["neighbours_within_tree_realizations"])
		inc = r["s7_1"]["all|z"]["increment_d7_over_nu"]
		r["s7_3_run"] = bool(inc["ci"][0] > 0.005)
		r["s7_3"] = s73(A, rng)  # computed for both runs; interpreted only where s7_3_run
		r["s7_4_5"] = {v: s74_75(A, boot, v) for v in ("backup_value", "z")}
		out[run] = r
		jdump(out, os.path.join(WED, "s7_key_geometry.json"))
	update_wednesday("s7_key_geometry", out)


if __name__ == "__main__":
	main()
