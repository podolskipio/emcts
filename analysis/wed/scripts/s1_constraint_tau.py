"""§1.3 -- propose tau for --emo_constraint_tau from the nu distribution, with a counterfactual
replay of the mask on logged selection points (Table B).

    python analysis/wed/scripts/s1_constraint_tau.py

The replay is EXACT on D2 (beta = 0: the logged uct is Q + U, which is the Constrain score) and
approximate on D1 (its uct carries beta*Q_emo). It is a replay on a fixed tree: masking would have
changed the tree it runs on, so these rates are the first-step effect, not the equilibrium.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, with_ci, load_selections, load_steps, update_wednesday, jdump  # noqa: E402

M_WARM = 3


def replay(B, tau, m_warm=M_WARM):
	b = B.copy()
	b["warm"] = b["N"] >= m_warm
	b["violating"] = b["warm"] & (b["Q_emo"] < tau)
	keys = ["dialogue_id", "turn_index", "simulation_index", "depth"]
	g = b.groupby(keys, sort=False)
	pts = g.agg(n=("action", "size"), n_viol=("violating", "sum"), n_warm=("warm", "sum")).reset_index()
	pts["fallback"] = pts["n_viol"] == pts["n"]
	# chosen under the mask: max uct over feasible; fallback -> max Q_emo
	b["feasible"] = ~b["violating"]
	b["_score"] = np.where(b["feasible"], b["uct"], -np.inf)
	idx_masked = b.groupby(keys, sort=False)["_score"].idxmax()
	idx_free = b.groupby(keys, sort=False)["uct"].idxmax()
	idx_fb = b.groupby(keys, sort=False)["Q_emo"].idxmax()
	pts = pts.set_index(keys)
	pts["chosen_masked"] = b.loc[idx_masked.values, "action"].values
	pts["chosen_free"] = b.loc[idx_free.values, "action"].values
	pts["chosen_fb"] = b.loc[idx_fb.values, "action"].values
	pts["chosen"] = np.where(pts["fallback"], pts["chosen_fb"], pts["chosen_masked"])
	pts["flip"] = pts["chosen"] != pts["chosen_free"]
	masked_acts = b[b["violating"]].groupby("action").size()
	return pts.reset_index(), masked_acts


def root_disagreement(B, tau, m_warm=M_WARM):
	"""At the last depth-1 selection point of each tree (the root's final statistics before the last
	backup): argmax N over the feasible set vs over all actions."""
	r = B[B["depth"] == 1]
	last = r.groupby("tree_id")["simulation_index"].transform("max")
	r = r[r["simulation_index"] == last].copy()
	r["feasible"] = ~((r["N"] >= m_warm) & (r["Q_emo"] < tau))
	out = []
	for tree, g in r.groupby("tree_id"):
		free = g.loc[g["N"].idxmax(), "action"]
		f = g[g["feasible"]]
		if not len(f):
			chosen = g.loc[g["Q_emo"].idxmax(), "action"]
		elif f["N"].max() == 0:
			chosen = free
		else:
			chosen = f.loc[f["N"].idxmax(), "action"]
		out.append({"tree_id": tree, "dialogue_id": g["dialogue_id"].iloc[0], "disagree": chosen != free})
	return pd.DataFrame(out)


def main():
	A = load_steps()
	res = {}
	for run in ("D1", "D2"):
		a = A[A["run_id"] == run]
		B = load_selections(run)
		warm_q = B.loc[B["N"] >= M_WARM, "Q_emo"]
		dist = {
			"z_quantiles": {q: float(np.quantile(a["z"], q)) for q in (0.1, 0.25, 0.5)},
			"parent_nu_quantiles": {q: float(np.quantile(a["parent_nu"], q)) for q in (0.1, 0.25, 0.5)},
			"warm_edge_Q_emo_quantiles": {q: float(np.quantile(warm_q, q)) for q in (0.1, 0.25, 0.5)},
		}
		boot = Boot(a["dialogue_id"].unique(), B=1000)
		cands = sorted({round(dist["z_quantiles"][0.25], 3), round(dist["warm_edge_Q_emo_quantiles"][0.25], 3),
						0.0, 0.1, 0.15, 0.2, 0.25})
		rows = {}
		for tau in cands:
			pts, masked = replay(B, tau)
			roots = root_disagreement(B, tau)
			by_pd = {k: g for k, g in pts.groupby("dialogue_id")}
			by_rd = {k: g for k, g in roots.groupby("dialogue_id")}

			def stat(ids):
				p = pd.concat([by_pd[i] for i in ids if i in by_pd])
				q = pd.concat([by_rd[i] for i in ids if i in by_rd])
				return (float((p["n_viol"] > 0).mean()), float(p["fallback"].mean()), float(p["flip"].mean()),
						float(q["disagree"].mean()), float(p["n_viol"].sum() / max(1, p["n_warm"].sum())))
			reps = boot.run(stat)
			point = stat(list(by_pd))
			rows[str(tau)] = {
				"points_with_mask": with_ci(point[0], [r[0] for r in reps]),
				"fallback_rate": with_ci(point[1], [r[1] for r in reps]),
				"selection_flip_rate": with_ci(point[2], [r[2] for r in reps]),
				"root_disagreement_rate": with_ci(point[3], [r[3] for r in reps]),
				"violation_rate_warm_edges": with_ci(point[4], [r[4] for r in reps]),
				"masked_counts_by_act": masked.to_dict(),
			}
			print(run, tau, {k: round(v["value"], 3) for k, v in rows[str(tau)].items() if isinstance(v, dict) and "value" in v})
		res[run] = {"distribution": dist, "replay": rows, "exact_replay": run == "D2", "m_warm": M_WARM}
	update_wednesday("s1_3_constraint_tau", res)
	jdump(res, os.path.join(WED, "s1_constraint_tau.json"))


if __name__ == "__main__":
	main()
