"""Readiness Phase 4 -- does coupling make comparisons resolvable? (PREREG Entry 11)

    python analysis/readiness/scripts/phase4_gate.py

Dialogue-level measures on the three coupled runs, plus the same correlation on the frozen uncoupled pair
over the same dialogues. Bootstrap over dialogues, 1000 replicates. -> phase4.json
"""
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402

B, SEED = 1000, 20260926
READY = os.path.join(E.READINESS, "runs")
GRID = os.path.join(E.REPO, "analysis", "grid", "runs")
OUT = os.path.join(E.READINESS, "phase4.json")


def load(root, tag):
	eps = pickle.load(open(os.path.join(root, tag, tag, tag + ".pkl"), "rb"))
	return {ep["did"]: ep for ep in eps}


def acts(ep):
	return [da for role, da, _e, _u in ep["history"] if role == "Persuader"]


def phi(x, y):
	x, y = np.asarray(x, float), np.asarray(y, float)
	if x.std() == 0 or y.std() == 0:
		return float("nan")
	return float(np.corrcoef(x, y)[0, 1])


def kappa(x, y):
	x, y = np.asarray(x, int), np.asarray(y, int)
	po = (x == y).mean()
	pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
	return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def boot(stat, n):
	rng = np.random.default_rng(SEED)
	reps = [stat(rng.integers(0, n, n)) for _ in range(B)]
	reps = [r for r in reps if np.isfinite(r)]
	return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def pair(a, b):
	ids = sorted(set(a) & set(b))
	x = np.array([a[d]["success"] for d in ids], float)
	y = np.array([b[d]["success"] for d in ids], float)
	return ids, x, y, {
		"n": len(ids), "sr_a": float(x.mean()), "sr_b": float(y.mean()),
		"phi": {"value": phi(x, y), "ci": boot(lambda i: phi(x[i], y[i]), len(ids))},
		"kappa": {"value": kappa(x, y), "ci": boot(lambda i: kappa(x[i], y[i]), len(ids))},
		"sr_diff": {"value": float(x.mean() - y.mean()), "ci": boot(lambda i: float(x[i].mean() - y[i].mean()), len(ids))},
		"agreement": float((x == y).mean()),
	}


def mdd(p, corr, n=100):
	return 2.80 * math.sqrt(2 * p * (1 - p) * (1 - max(corr, 0.0)) / n)


def wilson(k, n, z=1.96):
	p = k / n
	c = (p + z * z / (2 * n)) / (1 + z * z / n)
	h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
	return [c - h, c + h]


def divergence(a, b):
	rows = []
	for d in sorted(set(a) & set(b)):
		xa, xb = acts(a[d]), acts(b[d])
		first = next((t for t, (u, v) in enumerate(zip(xa, xb)) if u != v), None)
		rows.append(first)
	never = [r is None for r in rows]
	turns = [r for r in rows if r is not None]
	return {"n": len(rows), "share_never_diverge": float(np.mean(never)),
			"first_divergence_turn_hist": {int(t): int(c) for t, c in zip(*np.unique(turns, return_counts=True))},
			"median_first_divergence_turn": float(np.median(turns)) if turns else None}


def run_summary(ep):
	n = len(ep)
	k = sum(e["success"] for e in ep.values())
	hits = sum(e.get("coupling", {}).get("store_hits", 0) for e in ep.values())
	miss = sum(e.get("coupling", {}).get("store_misses", 0) for e in ep.values())
	return {"n": n, "SR": k / n, "SR_wilson": wilson(k, n), "AvgT": float(np.mean([e["num_turns"] for e in ep.values()])),
			"store_hits": hits, "store_misses": miss, "store_hit_share": hits / (hits + miss) if hits + miss else None}


def main():
	n1, act, n2 = (load(READY, t) for t in ("P4_NoEmo_seed1", "P4_ActPool_seed1", "P4_NoEmo_seed2"))
	f_n, f_a = load(GRID, "A_NoEmo_s20_seed1"), load(GRID, "A_ActPool_s20_seed1")
	res = {"runs": {"P4_NoEmo_seed1": run_summary(n1), "P4_ActPool_seed1": run_summary(act),
					"P4_NoEmo_seed2": run_summary(n2)}}
	_, _, _, r1 = pair(n1, act)
	res["R1_coupled_NoEmo_vs_ActPool"] = r1
	res["reference_frozen_uncoupled_NoEmo_vs_ActPool"] = pair(f_n, f_a)[3]
	res["NoEmo_seed1_vs_seed2"] = pair(n1, n2)[3]
	res["first_divergence_NoEmo_vs_ActPool"] = divergence(n1, act)
	res["first_divergence_NoEmo_seed1_vs_seed2"] = divergence(n1, n2)
	p = float(np.mean([r1["sr_a"], r1["sr_b"]]))
	corr = r1["phi"]["value"]
	res["implied_detectable_difference"] = {"p": p, "at_measured_corr": mdd(p, corr), "at_corr_0": mdd(p, 0.0),
											"at_corr_ci": [mdd(p, c) for c in r1["phi"]["ci"][::-1]]}
	res["GATE"] = ("full" if corr >= 0.6 else "limited" if corr >= 0.3 else "skip_SR_comparisons")
	res["git_head"] = E.git_head()
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
