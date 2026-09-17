"""§1.1 AffPool bias sweep diagnostics, dialogues 131-140 (non-eval, disjoint from D1-D3), beta_emo 0.

    python analysis/wed/scripts/s1_affpool_sweep.py

Per bias: mean RAVE beta at the selected edge by depth, cell counts, distinct prefixes per cell,
pool-term flip fraction (argmax with vs without the pool term), plus SR/turns for completeness (7
dialogues -- not a performance measurement). 3 of 10 dialogues per arm were lost to an SGLang
scheduler hang (FREEZE_NOTES §8.5); they were the longest-running ones, so the sample is biased short.
"""
import glob, gzip, json, os, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from lib import WED, Boot, with_ci, update_wednesday, jdump  # noqa: E402

out = {}
for b in ("0.05", "0.1", "0.25"):
	run = f"analysis/wed/runs/SW_b{b}/SW_b{b}"
	steps = []
	for p in sorted(glob.glob(f"{run}/simlog/*.ndjson.gz")):
		for line in gzip.open(p, "rt"):
			r = json.loads(line)
			if r["record_type"] == "step":
				steps.append({k: r[k] for k in ("dlg_id", "turn_index", "simulation_index", "depth", "action", "aff_bucket",
												"pool_beta_selected", "pool_flip", "child_from_cache")} |
							 {"prefix": "__".join(r["action_prefix"]),
							  "n_pool_sel": next(s["N_pool"] for s in r["siblings"] if s["action"] == r["action"]),
							  "n_edge_sel": next(s["N"] for s in r["siblings"] if s["action"] == r["action"])})
	S = pd.DataFrame(steps)
	S["tree"] = S["dlg_id"] + "|" + S["turn_index"].astype(str)
	boot = Boot(S["dlg_id"].unique())
	by = {k: g for k, g in S.groupby("dlg_id")}
	rep = boot.run(lambda ids: (lambda x: (x["pool_flip"].mean(), x["pool_beta_selected"].mean()))(pd.concat([by[i] for i in ids])))
	cells = S.groupby(["tree", "aff_bucket", "action"]).agg(n=("action", "size"), prefixes=("prefix", "nunique"))
	eps = pickle.load(open(f"{run}/SW_b{b}.pkl", "rb"))
	out[b] = {
		"dialogues": int(S["dlg_id"].nunique()), "steps": len(S), "trees": int(S["tree"].nunique()),
		"SR": float(np.mean([e["success"] for e in eps])), "avg_turns": float(np.mean([e["num_turns"] for e in eps])),
		"pool_flip_fraction": with_ci(float(S["pool_flip"].mean()), [r[0] for r in rep]),
		"mean_beta_selected": with_ci(float(S["pool_beta_selected"].mean()), [r[1] for r in rep]),
		"mean_beta_by_depth": S.groupby(S["depth"].clip(upper=8))["pool_beta_selected"].mean().round(3).to_dict(),
		"flip_by_depth": S.groupby(S["depth"].clip(upper=8))["pool_flip"].mean().round(3).to_dict(),
		"bucket1_share": float(S["aff_bucket"].mean()),
		"cells": int(len(cells)), "cell_steps_median": float(cells["n"].median()),
		"prefixes_per_cell_median": float(cells["prefixes"].median()),
		"prefixes_per_cell_quantiles": {q: float(cells["prefixes"].quantile(q)) for q in (0.25, 0.75, 0.9)},
		"share_cells_ge3_prefixes": float((cells["prefixes"] >= 3).mean()),
		"N_pool_at_selection_median": float(S["n_pool_sel"].median()),
		"flip_where_edge_unvisited": float(S.loc[S["n_edge_sel"] == 0, "pool_flip"].mean()),
		"flip_where_edge_visited": float(S.loc[S["n_edge_sel"] > 0, "pool_flip"].mean()),
		"wall_clock_s_arm": open(f"analysis/wed/runs/SW_b{b}/wall_clock.txt").read().strip(),
	}
	print(b, {k: v for k, v in out[b].items() if not isinstance(v, dict) or "value" in v})
	print("   beta by depth", out[b]["mean_beta_by_depth"], "flip by depth", out[b]["flip_by_depth"])
jdump(out, os.path.join(WED, "s1_affpool_sweep.json"))
update_wednesday("s1_1_affpool_sweep", out)
