"""Readiness Phase 5 -- coupled retention comparisons (PREREG Entry 12).

    python analysis/readiness/scripts/phase5_compare.py

For NoEmo and GDP-Zero: retention (#6) minus base, on every seed whose two runs are finished.
SR and AvgT differences pooled as the mean of the per-seed paired differences; 95 % CI by bootstrap over
dialogues, each resampled dialogue carrying all its seed pairs (1000 replicates). Per seed: SR of both arms,
coupled correlation, first divergence. -> phase5.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402
import phase4_gate as G  # noqa: E402

B, SEED = 1000, 20260927
OUT = os.path.join(E.READINESS, "phase5.json")
COMPARISONS = {"NoEmo": ("P5_NoEmo_base_s{}", "P5_NoEmo_ret_s{}"),
			   "GDPZero": ("P5_GDPZero_base_s{}", "P5_GDPZero_ret_s{}")}


def done(tag):
	path = os.path.join(G.READY, tag, "run_record.json")
	return os.path.exists(path) and json.load(open(path))["status"] == "done"


def compare(base_fmt, ret_fmt):
	return compare_pairs([(k, base_fmt.format(k), ret_fmt.format(k)) for k in range(1, 6)])


def compare_pairs(pairs):
	"""pairs: [(seed, base tag, other tag)]; the difference is other - base. Unfinished pairs are skipped."""
	pairs = [(k, bt, rt) for k, bt, rt in pairs if done(bt) and done(rt)]
	seeds = [k for k, _, _ in pairs]
	if not seeds:
		return {"seeds_done": []}
	per_seed, S, T = {}, [], []
	for k, base_tag, ret_tag in pairs:
		b, r = G.load(G.READY, base_tag), G.load(G.READY, ret_tag)
		ids = sorted(set(b) & set(r))
		sb = np.array([b[d]["success"] for d in ids], float)
		sr = np.array([r[d]["success"] for d in ids], float)
		tb = np.array([b[d]["num_turns"] for d in ids], float)
		tr = np.array([r[d]["num_turns"] for d in ids], float)
		per_seed[k] = {"n": len(ids), "SR_base": float(sb.mean()), "SR_ret": float(sr.mean()),
					   "AvgT_base": float(tb.mean()), "AvgT_ret": float(tr.mean()),
					   "SR_diff": float(sr.mean() - sb.mean()), "phi": G.phi(sb, sr),
					   "divergence": G.divergence(b, r)}
		S.append(dict(zip(ids, sr - sb)))
		T.append(dict(zip(ids, tr - tb)))
	common = sorted(set.intersection(*(set(x) for x in S)))
	dS = np.array([[s[d] for s in S] for d in common])  # dialogue x seed
	dT = np.array([[t[d] for t in T] for d in common])
	rng = np.random.default_rng(SEED)
	reps_s, reps_t = [], []
	for _ in range(B):
		i = rng.integers(0, len(common), len(common))
		reps_s.append(float(dS[i].mean()))
		reps_t.append(float(dT[i].mean()))
	ci = lambda r: [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
	p = float(np.mean([v["SR_base"] for v in per_seed.values()] + [v["SR_ret"] for v in per_seed.values()]))
	rho = float(np.nanmean([v["phi"] for v in per_seed.values()]))
	mdd = 2.80 * math.sqrt(2 * p * (1 - p) * (1 - max(rho, 0.0)) / (len(common) * len(seeds)))
	return {"seeds_done": seeds, "dialogues": len(common), "per_seed": per_seed,
			"SR_diff_pooled": {"value": float(dS.mean()), "ci": ci(reps_s)},
			"AvgT_diff_pooled": {"value": float(dT.mean()), "ci": ci(reps_t)},
			"mean_phi": rho, "SR_mean": p, "detectable_difference_pooled": mdd,
			"detected": bool(ci(reps_s)[0] > 0 or ci(reps_s)[1] < 0)}


def main():
	res = {name: compare(*fmts) for name, fmts in COMPARISONS.items()}
	res["git_head"] = E.git_head()
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
