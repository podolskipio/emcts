"""Shared statistics for the Wednesday analyses. Reads Table A/B/C; never re-derives their columns.

Conventions (brief §0): every CI is a cluster bootstrap over dialogue_id, 1000 replicates,
percentile; effect sizes are omega^2, never eta^2.
"""
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WED = os.path.join(REPO, "analysis", "wed")
B_DEFAULT = 1000
SEED = 20260916


def load_steps(run=None):
	A = pd.read_parquet(os.path.join(WED, "steps.parquet"))
	return A if run is None else A[A["run_id"] == run].reset_index(drop=True)


def load_selections(run=None):
	B = pd.read_parquet(os.path.join(WED, "selections.parquet"))
	return B if run is None else B[B["run_id"] == run].reset_index(drop=True)


def constants():
	return json.load(open(os.path.join(WED, "constants.json")))


def pct(a, q):
	a = np.asarray([x for x in np.asarray(a, float) if np.isfinite(x)])
	return float(np.percentile(a, q)) if len(a) else float("nan")


def ci(reps):
	return [pct(reps, 2.5), pct(reps, 97.5)]


class Boot:
	"""Cluster bootstrap over dialogues: stat(ids) receives dialogue ids WITH multiplicity."""

	def __init__(self, dialogues, B=B_DEFAULT, seed=SEED):
		self.d = sorted(dialogues)
		self.B = B
		rng = np.random.default_rng(seed)
		self.draws = [rng.integers(0, len(self.d), len(self.d)) for _ in range(B)]

	def run(self, stat):
		return [stat([self.d[i] for i in idx]) for idx in self.draws]

	def frame(self, df, col="dialogue_id"):
		"""Yield resampled frames (dialogues repeated with multiplicity, relabelled so a repeated
		dialogue counts as a distinct cluster)."""
		by = {k: g for k, g in df.groupby(col)}
		for idx in self.draws:
			parts = []
			for j, i in enumerate(idx):
				g = by.get(self.d[i])
				if g is not None:
					parts.append(g.assign(**{col: f"{self.d[i]}#{j}"}))
			yield pd.concat(parts, ignore_index=True)


def with_ci(value, reps):
	return {"value": value, "ci": ci(reps)}


def wmedian(x, w):
	x, w = np.asarray(x, float), np.asarray(w, float)
	m = np.isfinite(x)
	x, w = x[m], w[m]
	if not len(x):
		return np.nan
	o = np.argsort(x)
	c = np.cumsum(w[o])
	return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


# ---------------------------------------------------------------------------------------------
# AffPool cells: (tree, key, action), per tree then aggregated. Generalizes
# analysis/phase1/scripts/p_var.py:affpool_cells to any bucket column and any pooled quantity;
# with key=bucket_med, value=z it reproduces yesterday's numbers exactly (asserted in §5.1).
# ---------------------------------------------------------------------------------------------
def affpool_cells(df, key, value="z", extra_cell_cols=()):
	d = df[np.isfinite(df[key].astype(float))] if df[key].dtype.kind == "f" else df
	edge_N = df.groupby("edge_id").size()  # the node's own N(s,a) over ALL its visits
	d = d.assign(edge_N=d["edge_id"].map(edge_N))
	cols = ["dialogue_id", "tree_id", key, "action", *extra_cell_cols]
	g = d.groupby(cols + ["action_prefix"], sort=False)
	pp = g.agg(n=(value, "size"), mean=(value, "mean"), var=(value, lambda s: s.var(ddof=1)),
			   min_depth=("depth", "min")).reset_index()
	node = d.drop_duplicates(cols + ["edge_id"]).groupby(cols)["edge_N"].median().rename("median_N_node")
	rows = []
	for k, c in pp.groupby(cols, sort=False):
		ns = c["n"].to_numpy(float)
		N_pool = ns.sum()
		shares = ns / N_pool
		rec = dict(zip(cols, k))
		rec.update(n_prefixes=len(c), N_pool=int(N_pool), top_prefix_share=float(shares.max()),
				   herfindahl=float((shares ** 2).sum()), min_depth=int(c["min_depth"].min()))
		if len(c) >= 3:
			dfw = np.clip(ns - 1, 0, None)
			within = float((c["var"].fillna(0).to_numpy() * dfw).sum() / dfw.sum()) if dfw.sum() > 0 else np.nan
			between = float(np.var(c["mean"].to_numpy(), ddof=1))
			rec["between_var"], rec["within_var"] = between, within
			if within and within > 0:
				rec["ratio"] = between / within
				rec["ratio_noise_corrected"] = (between - float(np.mean(within / ns))) / within
			else:
				rec["ratio"] = np.inf if between > 0 else np.nan
				rec["ratio_noise_corrected"] = np.nan
		rows.append(rec)
	cells = pd.DataFrame(rows)
	cells = cells.merge(node.reset_index(), on=cols, how="left")
	cells["evidence_multiplier"] = cells["N_pool"] / cells["median_N_node"]
	for c in ("ratio", "ratio_noise_corrected", "between_var", "within_var"):
		if c not in cells:
			cells[c] = np.nan
	return cells


def affpool_summary(cells):
	h = cells[cells["n_prefixes"] >= 3]
	return {
		"n_prefixes_median": float(cells["n_prefixes"].median()),
		"evidence_multiplier_median": float(cells["evidence_multiplier"].median()),
		"top_prefix_share_median": float(cells["top_prefix_share"].median()),
		"between_within_ratio_wmedian": wmedian(h["ratio"], h["N_pool"]) if len(h) else np.nan,
		"between_within_ratio_noise_corrected_wmedian": wmedian(h["ratio_noise_corrected"], h["N_pool"]) if len(h) else np.nan,
	}


def affpool_block(df, key, value, boot, extra_cell_cols=()):
	cells = affpool_cells(df, key, value, extra_cell_cols)
	by_d = {k: g for k, g in cells.groupby("dialogue_id")}
	reps = boot.run(lambda ids: affpool_summary(pd.concat([by_d[i] for i in ids if i in by_d])))
	out = {k: with_ci(v, [r[k] for r in reps]) for k, v in affpool_summary(cells).items()}
	out["n_cells"] = int(len(cells))
	out["cells_ge3_prefixes"] = int((cells["n_prefixes"] >= 3).sum())
	return out, cells


# ---------------------------------------------------------------------------------------------
# regression helpers
# ---------------------------------------------------------------------------------------------
def ols_r2(X, y):
	X = np.column_stack([np.ones(len(y)), X])
	beta, *_ = np.linalg.lstsq(X, y, rcond=None)
	resid = y - X @ beta
	return 1.0 - resid.var() / y.var(), beta


def point_biserial(b, x):
	b, x = np.asarray(b, float), np.asarray(x, float)
	if b.std() == 0 or x.std() == 0:
		return np.nan
	return float(np.corrcoef(b, x)[0, 1])


def wilson(k, n, z=1.96):
	if n == 0:
		return [np.nan, np.nan]
	p = k / n
	den = 1 + z * z / n
	centre = (p + z * z / (2 * n)) / den
	half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
	return [float(centre - half), float(centre + half)]


def cohen_kappa(a, b):
	a, b = np.asarray(a, int), np.asarray(b, int)
	po = (a == b).mean()
	pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
	return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def jdump(obj, path):
	def conv(o):
		if isinstance(o, (np.floating,)):
			return float(o)
		if isinstance(o, (np.integer,)):
			return int(o)
		if isinstance(o, np.bool_):
			return bool(o)
		if isinstance(o, np.ndarray):
			return o.tolist()
		raise TypeError(type(o))
	with open(path, "w") as f:
		json.dump(obj, f, indent=1, default=conv, allow_nan=True)


def update_wednesday(section, payload):
	path = os.path.join(WED, "wednesday.json")
	allj = json.load(open(path)) if os.path.exists(path) else {}
	allj[section] = payload
	jdump(allj, path)


# ---------------------------------------------------------------------------------------------
# M0 / M1 / M2 interaction models (§4.4, §6.2): outcome ~ act [+ x] [+ act:x] [+ controls]
# Logistic (binary outcome) or OLS (continuous), numpy only. Inference on the interaction is a
# dialogue-clustered sandwich Wald test; the naive LR p is reported but is anti-conservative on
# clustered turns. Bootstrap CIs (clustered) on the R^2 / pseudo-R^2 increments.
# ---------------------------------------------------------------------------------------------
import math  # noqa: E402


def _gammq(a, x):
	"""Regularized upper incomplete gamma Q(a, x) (Numerical Recipes gser / gcf)."""
	if x <= 0:
		return 1.0
	gln = math.lgamma(a)
	if x < a + 1:
		ap, s, d = a, 1.0 / a, 1.0 / a
		for _ in range(1000):
			ap += 1
			d *= x / ap
			s += d
			if abs(d) < abs(s) * 1e-15:
				break
		return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
	b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
	h = d
	for i in range(1, 1000):
		an = -i * (i - a)
		b += 2
		d = an * d + b
		d = 1e-300 if abs(d) < 1e-300 else d
		c = b + an / c
		c = 1e-300 if abs(c) < 1e-300 else c
		d = 1.0 / d
		h *= d * c
		if abs(d * c - 1) < 1e-15:
			break
	return math.exp(-x + a * math.log(x) - gln) * h


def _betai(a, b, x):
	"""Regularized incomplete beta I_x(a, b) (Numerical Recipes betacf)."""
	if x <= 0:
		return 0.0
	if x >= 1:
		return 1.0
	bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))

	def cf(a, b, x):
		qab, qap, qam = a + b, a + 1, a - 1
		c, d = 1.0, 1 - qab * x / qap
		d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
		h = d
		for m in range(1, 1000):
			m2 = 2 * m
			aa = m * (b - m) * x / ((qam + m2) * (a + m2))
			d = 1 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d
			c = 1 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c
			d = 1 / d; h *= d * c
			aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
			d = 1 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d
			c = 1 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c
			d = 1 / d; de = d * c; h *= de
			if abs(de - 1) < 1e-15:
				break
		return h
	if x < (a + 1) / (a + b + 2):
		return bt * cf(a, b, x) / a
	return 1 - bt * cf(b, a, 1 - x) / b


class _stats:
	class chi2:
		@staticmethod
		def sf(x, k):
			return _gammq(k / 2.0, x / 2.0) if k > 0 else float("nan")

	class f:
		@staticmethod
		def sf(F, d1, d2):
			return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F)) if F > 0 else 1.0


def _logit_fit(X, y, ridge=1e-4, iters=50):
	beta = np.zeros(X.shape[1])
	pen = np.full(X.shape[1], ridge)
	pen[0] = 0.0
	for _ in range(iters):
		eta = np.clip(X @ beta, -30, 30)
		p = 1 / (1 + np.exp(-eta))
		W = p * (1 - p)
		H = X.T @ (X * W[:, None]) + np.diag(pen)
		g = X.T @ (y - p) - pen * beta
		step = np.linalg.solve(H, g)
		beta += step
		if np.max(np.abs(step)) < 1e-8:
			break
	eta = np.clip(X @ beta, -30, 30)
	p = 1 / (1 + np.exp(-eta))
	ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
	return beta, ll, p, H


def _ols_fit(X, y):
	beta, *_ = np.linalg.lstsq(X, y, rcond=None)
	fitted = X @ beta
	return beta, fitted, X.T @ X


def _cluster_cov(X, score_resid, bread_inv, groups):
	codes, inv = np.unique(groups, return_inverse=True)
	S = np.zeros((len(codes), X.shape[1]))
	np.add.at(S, inv, X * score_resid[:, None])
	G = len(codes)
	meat = S.T @ S * (G / max(G - 1, 1))
	return bread_inv @ meat @ bread_inv


def design(df, act_col, x_col, controls, acts, mode):
	"""mode 0: act; 1: act + x; 2: act + x + act:x. First act in `acts` is the reference."""
	cols = [np.ones(len(df))]
	names = ["const"]
	for a in acts[1:]:
		cols.append((df[act_col] == a).to_numpy(float))
		names.append(f"act[{a}]")
	for c in controls:
		cols.append(df[c].to_numpy(float))
		names.append(c)
	inter = []
	if mode >= 1:
		cols.append(df[x_col].to_numpy(float))
		names.append(x_col)
	if mode == 2:
		for a in acts[1:]:
			cols.append(((df[act_col] == a).to_numpy(float)) * df[x_col].to_numpy(float))
			names.append(f"act[{a}]:{x_col}")
			inter.append(len(names) - 1)
	return np.column_stack(cols), names, inter


def fit_models(df, outcome, x_col, act_col="act", controls=(), family="logit", cluster="dialogue_id", acts=None):
	acts = acts or list(df[act_col].value_counts().index)
	y = df[outcome].to_numpy(float)
	res = {"acts": acts, "n": int(len(df)), "n_clusters": int(df[cluster].nunique())}
	if family == "logit":
		p0 = y.mean()
		ll_null = float(len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
	for mode in (0, 1, 2):
		X, names, inter = design(df, act_col, x_col, controls, acts, mode)
		if family == "logit":
			beta, ll, p, H = _logit_fit(X, y)
			res[f"M{mode}"] = {"ll": ll, "pseudo_r2": 1 - ll / ll_null, "k": X.shape[1]}
			resid, bread = y - p, np.linalg.pinv(H)
		else:
			beta, fitted, XtX = _ols_fit(X, y)
			rss = float(np.sum((y - fitted) ** 2))
			res[f"M{mode}"] = {"r2": 1 - rss / float(np.sum((y - y.mean()) ** 2)), "k": X.shape[1], "rss": rss}
			resid, bread = y - fitted, np.linalg.pinv(XtX)
		if mode == 2:
			V = _cluster_cov(X, resid, bread, df[cluster].to_numpy())
			b = beta[inter]
			Vs = V[np.ix_(inter, inter)]
			rank = np.linalg.matrix_rank(Vs)
			wald = float(b @ np.linalg.pinv(Vs) @ b)
			G = df[cluster].nunique()
			# small-cluster correction: W/q against F(q, G-1) (chi2 over-rejects with ~100 clusters
			# and sparse act cells: 8.3% size at nominal 5% in a simulated clustered null)
			res["interaction_wald_cluster"] = {"chi2": wald, "df": int(rank), "p_chi2": float(_stats.chi2.sf(wald, rank)),
											   "p": float(_stats.f.sf(wald / max(rank, 1), rank, G - 1))}
			se = np.sqrt(np.clip(np.diag(V), 0, None))
			res["M2_coefficients"] = {n: {"b": float(beta[i]), "se_cluster": float(se[i])} for i, n in enumerate(names)}
		if mode == 1:
			X1 = X
	stat = "pseudo_r2" if family == "logit" else "r2"
	res["inc_M0_M1"] = res["M1"][stat] - res["M0"][stat]
	res["inc_M1_M2"] = res["M2"][stat] - res["M1"][stat]
	if family == "logit":
		lr = 2 * (res["M2"]["ll"] - res["M1"]["ll"])
		dfree = res["M2"]["k"] - res["M1"]["k"]
		res["lr_M1_M2"] = {"stat": lr, "df": dfree, "p_naive": float(_stats.chi2.sf(lr, dfree))}
	else:
		dfree = res["M2"]["k"] - res["M1"]["k"]
		F = ((res["M1"]["rss"] - res["M2"]["rss"]) / dfree) / (res["M2"]["rss"] / (len(y) - res["M2"]["k"]))
		res["F_M1_M2"] = {"F": float(F), "df": dfree, "p_naive": float(_stats.f.sf(F, dfree, len(y) - res["M2"]["k"]))}
	return res


def fit_models_boot(df, outcome, x_col, boot, **kw):
	point = fit_models(df, outcome, x_col, **kw)
	acts = point["acts"]
	incs01, incs12 = [], []
	for bdf in boot.frame(df):
		try:
			r = fit_models(bdf, outcome, x_col, acts=acts, **{k: v for k, v in kw.items() if k != "acts"})
			incs01.append(r["inc_M0_M1"])
			incs12.append(r["inc_M1_M2"])
		except Exception:
			continue
	point["inc_M0_M1_ci"] = ci(incs01)
	point["inc_M1_M2_ci"] = ci(incs12)
	point["boot_ok"] = len(incs12)
	return point
