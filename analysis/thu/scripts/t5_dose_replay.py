"""Dose-matching the three selection arms: Bias, CenteredBias, Momentum.

    python analysis/thu/scripts/t5_dose_replay.py                       # D2 (Wednesday tables)
    python analysis/thu/scripts/t5_dose_replay.py --simlog analysis/thu/runs/P1_legacy/P1_legacy --tag P1_legacy

DOSE = argmax flip rate against NoEmo, on the SAME NoEmo-grown trees, one selection at a time:

  NoEmo         Q + U
  Bias          Q + U + beta * Q_emo_level
  CenteredBias  Q + U + beta * (Q_emo_level - mu)   on expanded edges; unexpanded unchanged
                mu = mean Q_emo_level over expanded siblings (runner: _expanded_emo_mean)
  Momentum      Q + U + beta * Q_emo_delta           Q_emo_delta = mean over the edge's earlier
                visits of (nu(child) - nu(sampled parent)) / 2   (--emo_signal delta)

Every arm is scored on identical trees and identical selection points, so flip rates are commensurable.
Own-tree flip rates (Wednesday's 31.6 % for Bias on D1, 25.5 % for "CenteredBias" -- which is centred
vs BIAS on Bias's trees, not vs NoEmo) mix the dose with each arm's effect on tree shape and are not a
dose.

For each arm the script reports the flip rate over a beta grid and solves (bisection) for the beta whose
flip rate equals Bias's at the reference beta (default 0.7).
Q_emo_level is rebuilt from the logged per-visit child_nu and asserted against the logged Q_emo.
"""
import argparse
import glob
import gzip
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
PT = ["tree_id", "simulation_index", "depth", "pfx"]
GRID = (0.35, 0.7, 1.0, 1.4, 2.0, 2.8)


def _pfx(x):
	return "|".join(list(x)) if x is not None else ""


def load_wed(run):
	S, B = lib.load_steps(run), lib.load_selections(run)
	S["pfx"], B["pfx"] = S["action_prefix"].map(_pfx), B["action_prefix"].map(_pfx)
	beta = {"D1": 0.7, "D2": 0.0, "D3": 0.7}[run]
	return S[["dialogue_id", "tree_id", "pfx", "action", "simulation_index", "depth", "parent_nu", "child_nu"]], \
		B[["dialogue_id", "tree_id", "pfx", "action", "simulation_index", "depth", "N", "Q", "Q_emo", "uct"]], beta


def _table(name):
	sys.path.insert(0, os.path.join(REPO, "src"))
	import mcts.emotion_mcts as em
	return {str(getattr(k, "value", k)).lower(): v for k, v in em.EMOTION_VALENCE_TABLES[name].items()}


def _nu(dist, w):
	return sum(p * w.get(e, 0.0) for e, p in dist.items()) if dist else 0.0


def load_simlog(run, table="soft"):
	"""ν is recomputed from the logged emotion distributions under ``table``. Under "soft" (what the
	pilots ran) the recomputation must reproduce the logged parent_nu / child_nu -- asserted."""
	w = _table(table)
	w_soft = _table("soft")
	srows, brows = [], []
	for p in sorted(glob.glob(os.path.join(run, "simlog", "*.ndjson.gz"))):
		for line in gzip.open(p, "rt"):
			r = json.loads(line)
			if r.get("record_type") != "step":
				continue
			tid, pfx = f"{r['dlg_id']}|{r['turn_index']}", _pfx(r["action_prefix"])
			base = {"dialogue_id": r["dlg_id"], "tree_id": tid, "pfx": pfx,
					"simulation_index": r["simulation_index"], "depth": r["depth"]}
			pd_, cd_ = r["parent_emotion_dist"], r["child_emotion_dist"]
			assert abs(_nu(pd_, w_soft) - r["parent_nu"]) < 1e-9 and abs(_nu(cd_, w_soft) - r["child_nu"]) < 1e-9
			srows.append({**base, "action": r["action"], "parent_nu": _nu(pd_, w), "child_nu": _nu(cd_, w)})
			for sb in r["siblings"]:
				brows.append({**base, "action": sb["action"], "N": sb["N"], "Q": sb["Q"], "Q_emo": sb["Q_emo"],
							  "uct": sb["uct"]})
	meta = json.load(open(os.path.join(run, "metadata.json")))
	return pd.DataFrame(srows), pd.DataFrame(brows), float(meta["args"]["beta_emo"])


def build(S, B, beta_logged, check_logged_qemo=True):
	S = S.copy()
	S["z_level"], S["z_delta"] = S["child_nu"], (S["child_nu"] - S["parent_nu"]) / 2.0
	S = S.sort_values(["tree_id", "pfx", "action", "simulation_index"])
	g = S.groupby(["tree_id", "pfx", "action"], sort=False)
	S["cum_l"], S["cum_d"], S["cnt"] = g["z_level"].cumsum(), g["z_delta"].cumsum(), g.cumcount() + 1
	M = pd.merge_asof(B.sort_values("simulation_index", kind="mergesort"),
					  S[["tree_id", "pfx", "action", "simulation_index", "cum_l", "cum_d", "cnt"]].sort_values("simulation_index", kind="mergesort"),
					  on="simulation_index", by=["tree_id", "pfx", "action"], allow_exact_matches=False, direction="backward")
	M[["cum_l", "cum_d", "cnt"]] = M[["cum_l", "cum_d", "cnt"]].fillna(0.0)
	c = M["cnt"].clip(lower=1)
	M["Qe_level"] = np.where(M["cnt"] > 0, M["cum_l"] / c, 0.0)
	M["Qe_delta"] = np.where(M["cnt"] > 0, M["cum_d"] / c, 0.0)
	val = {"rows": int(len(M)), "N_mismatch_rows": int((M["cnt"] != M["N"]).sum()),
		   "max_abs_Qemo_level_deviation": float((M["Qe_level"] - M["Q_emo"]).abs().max())}
	assert val["N_mismatch_rows"] == 0, val
	if check_logged_qemo:  # only meaningful when nu is the table the run logged Q_emo under
		assert val["max_abs_Qemo_level_deviation"] < 1e-9, val
	M["expanded"] = M["N"] > 0
	M["base"] = M["uct"] - beta_logged * M["Q_emo"]
	mu = M[M["expanded"]].groupby(PT)["Qe_level"].mean().rename("mu")
	M = M.join(mu, on=PT)
	M["Qe_centre"] = np.where(M["expanded"], M["Qe_level"] - M["mu"], 0.0)
	# row index of the NoEmo choice, once
	M = M.reset_index(drop=True)
	M["_pt"] = M.groupby(PT, sort=False).ngroup()
	return M, val


COL = {"bias": "Qe_level", "centre": "Qe_centre", "momentum": "Qe_delta"}


def flip_rate(M, arm, beta, detail=False):
	score = M["base"].to_numpy() + beta * M[COL[arm]].to_numpy()
	pt = M["_pt"].to_numpy()
	order = np.lexsort((-score, pt))  # within point, highest score first (stable on ties: first logged)
	first = np.ones(len(order), bool)
	first[1:] = pt[order][1:] != pt[order][:-1]
	i1 = order[first]
	base = M["base"].to_numpy()
	order0 = np.lexsort((-base, pt))
	i0 = order0[first]  # same point order in both
	flip = M["action"].to_numpy()[i0] != M["action"].to_numpy()[i1]
	if not detail:
		return float(flip.mean())
	exp = M["expanded"].to_numpy()
	f = flip
	dlg = M["dialogue_id"].to_numpy()[i0]
	pts = pd.DataFrame({"dialogue_id": dlg, "flip": flip})
	by = {k: v["flip"].to_numpy() for k, v in pts.groupby("dialogue_id")}
	boot = lib.Boot(list(by))
	reps = boot.run(lambda ids: float(np.concatenate([by[i] for i in ids]).mean()))
	return {"flip_rate": lib.with_ci(float(flip.mean()), reps),
			"to_visited_over_unvisited": float((exp[i1] & ~exp[i0])[f].mean()) if f.any() else None,
			"to_unexpanded": float((~exp[i1])[f].mean()) if f.any() else None,
			"between_visited": float((exp[i1] & exp[i0])[f].mean()) if f.any() else None,
			"n_points": int(len(flip))}


def solve_beta(M, arm, target, lo=0.0, hi=8.0, iters=30):
	if flip_rate(M, arm, hi) < target:
		return None
	for _ in range(iters):
		mid = (lo + hi) / 2
		(lo, hi) = (mid, hi) if flip_rate(M, arm, mid) < target else (lo, mid)
	return (lo + hi) / 2


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--wed_run", default="D2")
	ap.add_argument("--simlog", default=None)
	ap.add_argument("--tag", default=None)
	ap.add_argument("--ref_beta", type=float, default=0.7)
	ap.add_argument("--table", default="soft", help="[--simlog] valence table to recompute nu under")
	a = ap.parse_args()
	if a.simlog:
		S, B, bl = load_simlog(a.simlog, a.table)
		tag = (a.tag or os.path.basename(a.simlog.rstrip("/"))) + ("" if a.table == "soft" else f"_{a.table}")
	else:
		S, B, bl = load_wed(a.wed_run)
		tag = a.wed_run
	if bl != 0.0:
		print(f"WARNING: trees were grown at beta_emo {bl}; the dose is only clean on NoEmo trees")
	M, val = build(S, B, bl, check_logged_qemo=(not a.simlog or a.table == "soft"))
	target = flip_rate(M, "bias", a.ref_beta)
	out = {"trees": tag, "table": a.table, "trees_beta_emo": bl, "validation": val, "reference": f"bias at beta {a.ref_beta}",
		   "target_flip_rate": target, "grid": {}, "matched": {}}
	for arm in COL:
		out["grid"][arm] = {str(b): flip_rate(M, arm, b) for b in GRID}
		bstar = a.ref_beta if arm == "bias" else solve_beta(M, arm, target)
		out["matched"][arm] = {"beta": bstar, **(flip_rate(M, arm, bstar, detail=True) if bstar is not None else {})}
		out["matched"][arm]["at_reference_beta"] = flip_rate(M, arm, a.ref_beta, detail=True)
	print(json.dumps({k: v for k, v in out.items()}, indent=1, default=float))
	lib.jdump(out, os.path.join(THU, f"dose_replay_{tag}.json"))


if __name__ == "__main__":
	main()
