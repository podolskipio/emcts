"""AffPool / ActPool homogeneity gate on the TASK RETURN, legacy vs episode NoEmo trees (P1).

    python analysis/thu/scripts/t4_affpool_gate_episode.py

Same estimator as analysis/wed/scripts/s51_affpool_gate_recheck.py (lib.affpool_block). Keys:
  K0 bucket   parent_nu < tau_med of the run's own steps   (AffPool)
  unkeyed     one bucket                                    (ActPool)
Pooled quantity: backup_value (what Q_pool carries), with z alongside. Asserts the z / K0 numbers
reproduce analyze_pilot.py's p_var components (legacy 0.714, episode 0.983), so the task-return
column is computed on the same cells.
"""
import gzip, glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402

PVAR_Z = {"P1_legacy": 0.714, "P1_episode": 0.983}


def steps(run):
	rows = []
	for p in sorted(glob.glob(os.path.join(run, "simlog", "*.ndjson.gz"))):
		for line in gzip.open(p, "rt"):
			s = json.loads(line)
			if s.get("record_type") != "step":
				continue
			tid, pfx = f"{s['dlg_id']}|{s['turn_index']}", "|".join(s["action_prefix"])
			rows.append({"dialogue_id": s["dlg_id"], "tree_id": tid, "action_prefix": pfx, "action": s["action"],
						 "edge_id": f"{tid}#{pfx}#{s['action']}", "depth": s["depth"], "parent_nu": s["parent_nu"],
						 "z": s["z"], "backup_value": s["backup_value"]})
	d = pd.DataFrame(rows)
	d["bucket_med"] = (d["parent_nu"] < d["parent_nu"].median()).astype(int)
	d["unkeyed"] = 0
	return d


out = {}
for tag in ("P1_legacy", "P1_episode"):
	d = steps(os.path.join(REPO, "analysis", "thu", "runs", tag, tag))
	boot = lib.Boot(d["dialogue_id"].unique())
	res = {"tau_med": float(d["parent_nu"].median()), "steps": int(len(d))}
	for key in ("bucket_med", "unkeyed"):
		for value in ("z", "backup_value"):
			blk, _ = lib.affpool_block(d, key, value, boot)
			res[f"{key}|{value}"] = blk
	got = res["bucket_med|z"]["between_within_ratio_wmedian"]["value"]
	assert abs(got - PVAR_Z[tag]) < 0.002, (tag, got)
	out[tag] = res
	for k, v in res.items():
		if isinstance(v, dict):
			print(tag, k, "ratio", round(v["between_within_ratio_wmedian"]["value"], 3), np.round(v["between_within_ratio_wmedian"]["ci"], 3),
				  "nc", round(v["between_within_ratio_noise_corrected_wmedian"]["value"], 3), np.round(v["between_within_ratio_noise_corrected_wmedian"]["ci"], 3),
				  "mult", v["evidence_multiplier_median"]["value"], "cells>=3", v["cells_ge3_prefixes"])
lib.jdump(out, os.path.join(REPO, "analysis", "thu", "affpool_gate_episode.json"))
