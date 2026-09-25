"""Incremental McFadden R2 with the TrajValue gate's own estimators (analysis/thu/scripts/t6_trajvalue_gates.py),
so C1b is computed exactly as the 0.0125 it is compared against."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis", "thu", "scripts"))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402
import t6_trajvalue_gates as T6  # noqa: E402
import t2_construct_test as CT  # noqa: E402

B = 1000
SEED = 20260925


def incremental_r2(df, base, extra, bar=0.02):
	"""logit(donated) ~ base  vs  ~ base + extra. One row per dialogue, so the bootstrap over rows is the
	cluster bootstrap by dialogue_id."""
	full = base + extra
	r2_base, _ = T6.mcfadden(df, base)
	r2_full, beta = T6.mcfadden(df, full)
	inc = r2_full - r2_base
	rng = np.random.default_rng(SEED)
	n = len(df)
	reps = []
	for _ in range(B):
		sub = df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
		if sub["donated"].nunique() < 2:
			continue
		reps.append(T6.mcfadden(sub, full)[0] - T6.mcfadden(sub, base)[0])
	y = df["donated"].to_numpy(float)
	cv_rng = np.random.default_rng(SEED + 7)
	pb, pf = T6.oof_probs(df, base, cv_rng), T6.oof_probs(df, full, cv_rng)
	ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
	d_ll = ll(pb) - ll(pf)  # positive = the extra term helps out of fold
	cv = [float(d_ll[i].mean()) for i in (rng.integers(0, n, n) for _ in range(B))]
	return {
		"n": int(n), "donation_rate": float(y.mean()),
		"mcfadden_base": float(r2_base), "mcfadden_full": float(r2_full),
		"increment": lib.with_ci(float(inc), reps),
		"coef_std": {c: float(beta[1 + full.index(c)]) for c in full},
		"PASS": bool(lib.ci(reps)[0] > 0 and inc >= bar),
		"oof_logloss_gain": lib.with_ci(float(d_ll.mean()), cv),
		"auc_base": float(CT.auc(y, pb)), "auc_full": float(CT.auc(y, pf)),
	}


def corr_ci(x, y):
	x, y = np.asarray(x, float), np.asarray(y, float)
	rng = np.random.default_rng(SEED + 1)
	n = len(x)
	reps = [float(np.corrcoef(x[i], y[i])[0, 1]) for i in (rng.integers(0, n, n) for _ in range(B))]
	return lib.with_ci(float(np.corrcoef(x, y)[0, 1]), reps)


def mean_ci(x):
	x = np.asarray(x, float)
	rng = np.random.default_rng(SEED + 2)
	n = len(x)
	reps = [float(x[rng.integers(0, n, n)].mean()) for _ in range(B)]
	return lib.with_ci(float(x.mean()), reps)
