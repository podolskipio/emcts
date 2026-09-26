"""Readiness 5b -- AffPool vs ActPool on the fixed planner (PREREG Entry 13).

    python analysis/readiness/scripts/phase5b.py

PRIMARY: SR / AvgT, AffPool - ActPool, pooled over the finished coupled seed pairs (phase5_compare.compare_pairs).
DECISION LEVEL: pooling homogeneity of backup_value under the affect key (tree, parent-nu bucket, act) and the
act key (tree, act), on the same AffPool steps, with the Wednesday estimator (lib.affpool_cells/affpool_summary).
gap = ratio(act key) - ratio(affect key), bootstrapped as one statistic over dialogues (run x dialogue).
BEFORE: frozen A_AffPool_s20_seed1; AFTER: the P5b_AffPool runs pooled. -> phase5b.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _env as E  # noqa: E402
import phase2_verify as P2  # noqa: E402
import phase5_compare as C  # noqa: E402

sys.path.insert(0, os.path.join(E.REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402

TAU, B, SEED = 0.263, 1000, 20260927
OUT = os.path.join(E.READINESS, "phase5b.json")
ACT = {1: "P4_ActPool_seed1", **{k: f"P5b_ActPool_s{k}" for k in range(2, 6)}}
AFF = {k: f"P5b_AffPool_s{k}" for k in range(1, 6)}
FROZEN = "F_AffPool_s20_seed1"


def wed_frame(tag):
	if tag not in P2.RUNS:
		P2.RUNS[tag] = os.path.join(E.READINESS, "runs", tag, tag)
	d = P2.steps(tag)
	return pd.DataFrame({
		"dialogue_id": tag + "|" + d["dlg_id"], "tree_id": tag + "|" + d["tree_key"],
		"edge_id": tag + "|" + d["edge_key"], "action_prefix": d["prefix_key"], "action": d["action"],
		"depth": d["depth"], "backup_value": d["backup_value"] if "backup_value" in d else d["v"],
		"affect": (d["parent_nu"] < TAU).astype(int), "act": 0})


def homogeneity(frame):
	cells = {k: lib.affpool_cells(frame, k, "backup_value") for k in ("affect", "act")}
	summ = {k: lib.affpool_summary(c) for k, c in cells.items()}
	key = "between_within_ratio_noise_corrected_wmedian"
	by = {k: {d: g for d, g in c.groupby("dialogue_id")} for k, c in cells.items()}
	ids = sorted(frame["dialogue_id"].unique())
	rng = np.random.default_rng(SEED)
	reps = {"affect": [], "act": [], "gap": []}
	for _ in range(B):
		pick = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
		s = {k: lib.affpool_summary(pd.concat([by[k][d] for d in pick if d in by[k]]))[key] for k in by}
		reps["affect"].append(s["affect"]); reps["act"].append(s["act"]); reps["gap"].append(s["act"] - s["affect"])
	ci = lambda r: lib.ci(r)
	return {"affect_key": {"ratio_nc": {"value": summ["affect"][key], "ci": ci(reps["affect"])},
						   "evidence_multiplier": summ["affect"]["evidence_multiplier_median"],
						   "cells_ge3": int((cells["affect"]["n_prefixes"] >= 3).sum())},
			"act_key": {"ratio_nc": {"value": summ["act"][key], "ci": ci(reps["act"])},
						"evidence_multiplier": summ["act"]["evidence_multiplier_median"],
						"cells_ge3": int((cells["act"]["n_prefixes"] >= 3).sum())},
			"gap": {"value": summ["act"][key] - summ["affect"][key], "ci": ci(reps["gap"])},
			"dialogues": len(ids)}


def main():
	res = {"git_head": E.git_head()}
	res["SR"] = C.compare_pairs([(k, ACT[k], AFF[k]) for k in range(1, 6)])
	res["homogeneity_before"] = homogeneity(wed_frame(FROZEN))
	after = [t for t in AFF.values() if C.done(t)]
	res["homogeneity_after"] = homogeneity(pd.concat([wed_frame(t) for t in after], ignore_index=True)) if after else None
	res["after_runs"] = after
	if after:
		b, a = res["homogeneity_before"]["gap"], res["homogeneity_after"]["gap"]
		res["gap_closes"] = bool(a["ci"][0] > b["value"])
	json.dump(res, open(OUT, "w"), indent=1, default=str)
	print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
	main()
