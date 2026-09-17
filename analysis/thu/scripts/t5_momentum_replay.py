"""TASK 5 -- Momentum (--emo_signal delta) one-step replay on the logged trees.

    python analysis/thu/scripts/t5_momentum_replay.py

Code fact (src/mcts/emotion_mcts.py): --emo_signal changes ONLY the z backed up into Q_emo
(`_emotion_signal`); selection reads beta*Q_emo unchanged (`_calculate_uct`). So under `delta` the
selection term is beta * running-mean over the edge's visits of (nu(child) - nu(sampled parent))/2.
One channel: the backed-up signal IS the selection term.

Replay. For every logged selection point, rebuild each sibling's Q_emo under both signals from the
edge's EARLIER visits in steps.parquet (parent_nu, child_nu are logged per visit, the parent being
the sampled realization -- exactly what `delta` differences against), then take argmax of
  NoEmo     Q + U
  Bias      Q + U + beta*Q_emo_level
  Momentum  Q + U + beta*Q_emo_delta
with U = logged uct - logged beta*Q_emo. The level reconstruction is asserted against the logged Q_emo,
which validates the join. Primary run: D2 (beta 0, trees not shaped by any affect term) -- the
same counterfactual as p_var.md §7.6 (Bias 15.5 %). D1 reported beside it.
Exact for one selection on the logged tree; the arm would grow a different tree (the pilot's job).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402

THU = os.path.join(REPO, "analysis", "thu")
BETA_LOGGED = {"D1": 0.7, "D2": 0.0}
BETAS = (0.35, 0.7, 1.4, 2.8, 5.6)
PT = ["tree_id", "simulation_index", "depth", "pfx"]


def pfx(x):
	return "|".join(list(x)) if x is not None else ""


def replay(run):
	S = lib.load_steps(run)
	B = lib.load_selections(run)
	S["pfx"], B["pfx"] = S["action_prefix"].map(pfx), B["action_prefix"].map(pfx)
	S["z_level"] = S["child_nu"]
	S["z_delta"] = (S["child_nu"] - S["parent_nu"]) / 2.0

	# per edge, the running mean of z over visits strictly BEFORE each simulation
	S = S.sort_values(["tree_id", "pfx", "action", "simulation_index"])
	g = S.groupby(["tree_id", "pfx", "action"], sort=False)
	for z in ("z_level", "z_delta"):
		S[f"cum_{z}"] = g[z].cumsum()
	S["cnt"] = g.cumcount() + 1
	edge_hist = S[["tree_id", "pfx", "action", "simulation_index", "cum_z_level", "cum_z_delta", "cnt"]]

	B = B.sort_values("simulation_index")
	E = edge_hist.sort_values("simulation_index")
	M = pd.merge_asof(B, E, on="simulation_index", by=["tree_id", "pfx", "action"],
					  allow_exact_matches=False, direction="backward")
	M[["cum_z_level", "cum_z_delta", "cnt"]] = M[["cum_z_level", "cum_z_delta", "cnt"]].fillna(0.0)
	M["Qe_level"] = np.where(M["cnt"] > 0, M["cum_z_level"] / M["cnt"].clip(lower=1), 0.0)
	M["Qe_delta"] = np.where(M["cnt"] > 0, M["cum_z_delta"] / M["cnt"].clip(lower=1), 0.0)

	# validation: count and level mean must reproduce the logged N and Q_emo
	n_bad = int((M["cnt"] != M["N"]).sum())
	q_dev = float((M["Qe_level"] - M["Q_emo"]).abs().max())

	M["base"] = M["uct"] - BETA_LOGGED[run] * M["Q_emo"]  # Q + U
	M["expanded"] = M["N"] > 0
	return M, {"rows": int(len(M)), "N_mismatch_rows": n_bad, "max_abs_Qemo_level_deviation": q_dev}


def flips(M, beta, arm):
	col = {"bias": "Qe_level", "momentum": "Qe_delta"}[arm]
	M = M.assign(score=M["base"] + beta * M[col])
	g = M.groupby(PT, sort=False)
	i0 = g["base"].idxmax().values
	i1 = g["score"].idxmax().values
	pts = pd.DataFrame({"dialogue_id": M.loc[i0, "dialogue_id"].values, "depth": M.loc[i0, "depth"].values,
						"flip": M.loc[i0, "action"].values != M.loc[i1, "action"].values,
						"to_expanded": M.loc[i1, "expanded"].values,
						"from_expanded": M.loc[i0, "expanded"].values})
	f = pts[pts["flip"]]
	boot = lib.Boot(pts["dialogue_id"].unique())
	by = {k: v for k, v in pts.groupby("dialogue_id")}
	reps = boot.run(lambda ids: float(pd.concat([by[i] for i in ids])["flip"].mean()))
	return {
		"flip_rate": lib.with_ci(float(pts["flip"].mean()), reps),
		"flips_to_visited_over_unvisited": float((f["to_expanded"] & ~f["from_expanded"]).mean()) if len(f) else np.nan,
		"flips_to_unexpanded": float((~f["to_expanded"]).mean()) if len(f) else np.nan,
		"flips_between_visited": float((f["to_expanded"] & f["from_expanded"]).mean()) if len(f) else np.nan,
		"flip_by_depth": {str(d): float(v) for d, v in pts.groupby("depth")["flip"].mean().items() if d in (1, 2, 3, 5, 8, 10)},
		"n_points": int(len(pts)),
	}


def main():
	out = {"code_fact": "emo_signal changes only the z backed into Q_emo; selection uses beta*Q_emo, so under "
						"delta the selection term is beta*mean over edge visits of (nu_child - nu_sampled_parent)/2. "
						"The /2 (EMO_DELTA_RANGE_SCALE) halves the effective beta relative to raw delta-nu.",
		   "runs": {}}
	for run in ("D2", "D1"):
		M, val = replay(run)
		vis = M[M["expanded"]]
		r = {"validation": val,
			 "visited_edges": {
				 "Qemo_level_mean": float(vis["Qe_level"].mean()), "Qemo_level_sd": float(vis["Qe_level"].std()),
				 "Qemo_delta_mean": float(vis["Qe_delta"].mean()), "Qemo_delta_sd": float(vis["Qe_delta"].std()),
				 "corr_depth_level": float(vis["Qe_level"].corr(vis["depth"])),
				 "corr_depth_delta": float(vis["Qe_delta"].corr(vis["depth"])),
			 },
			 "per_visit_z": {},
			 "bias": {}, "momentum": {}}
		S = lib.load_steps(run)
		zd = (S["child_nu"] - S["parent_nu"]) / 2
		r["per_visit_z"] = {"level_mean": float(S["child_nu"].mean()), "level_sd": float(S["child_nu"].std()),
							"delta_mean": float(zd.mean()), "delta_sd": float(zd.std()),
							"raw_delta_nu_mean": float((2 * zd).mean()), "raw_delta_nu_sd": float((2 * zd).std())}
		for beta in BETAS:
			r["bias"][str(beta)] = flips(M, beta, "bias")
			r["momentum"][str(beta)] = flips(M, beta, "momentum")
		out["runs"][run] = r
		print(f"== {run} validation {val}")
		print("   visited edges", {k: round(v, 3) for k, v in r["visited_edges"].items()})
		print("   per-visit z", {k: round(v, 3) for k, v in r["per_visit_z"].items()})
		for beta in BETAS:
			b, m = r["bias"][str(beta)], r["momentum"][str(beta)]
			print(f"   beta {beta:4}: bias flip {b['flip_rate']['value']:.3f} (to visited-over-unvisited {b['flips_to_visited_over_unvisited']:.2f}, to unexpanded {b['flips_to_unexpanded']:.2f})"
				  f" | momentum flip {m['flip_rate']['value']:.3f} {np.round(m['flip_rate']['ci'], 3)} (v>u {m['flips_to_visited_over_unvisited']:.2f}, to unexp {m['flips_to_unexpanded']:.2f}, between visited {m['flips_between_visited']:.2f})")
	lib.jdump(out, os.path.join(THU, "momentum_replay.json"))


if __name__ == "__main__":
	main()
