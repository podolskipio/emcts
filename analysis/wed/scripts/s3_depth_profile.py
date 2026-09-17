"""§3 depth profile, per run, from Table A.

    python analysis/wed/scripts/s3_depth_profile.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, load_steps, update_wednesday  # noqa: E402

BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 6, "6"), (7, 10, "7–10"), (11, 99, "11+")]


def profile(A):
	rows = []
	# node visits: a parent node is (tree, prefix); its visits are the steps leaving it
	node_visits = A.groupby(["tree_id", "action_prefix", "depth"]).size().rename("visits").reset_index()
	for lo, hi, name in BINS:
		g = A[(A["depth"] >= lo) & (A["depth"] <= hi)]
		nv = node_visits[(node_visits["depth"] >= lo) & (node_visits["depth"] <= hi)]
		if not len(g):
			continue
		q = np.quantile(g["parent_nu"], [0.25, 0.5, 0.75])
		rows.append({
			"depth": name, "visits": len(g), "share": len(g) / len(A), "edges": g["edge_id"].nunique(),
			"median_visits_per_parent_node": float(nv["visits"].median()),
			"nu_q25": q[0], "nu_median": q[1], "nu_q75": q[2], "nu_mean": float(g["parent_nu"].mean()),
			"reachable_share": float(g["reachable"].mean()),
		})
	return pd.DataFrame(rows)


def main():
	A = load_steps()
	out, md = {}, ["# §3 Depth profile\n",
				   "From `steps.parquet`. Depth is edge depth (the root's outgoing edges are depth 1). "
				   "\"Median visits per parent node\" is the median visit count of the parent nodes at that depth. "
				   "ν is `parent_nu`. `reachable` is the share of steps whose child state has turn + depth ≤ Tmax "
				   "(`analysis/phase1/SEARCH_HORIZON_BUG.md`).\n"]
	for run in sorted(A["run_id"].unique()):
		g = A[A["run_id"] == run]
		p = profile(g)
		ge2 = float((g["depth"] >= 2).mean())
		d2 = p.loc[p["depth"] == "2", "median_visits_per_parent_node"].iloc[0]
		out[run] = {"table": p.to_dict(orient="records"), "share_visits_depth_ge2": ge2,
					"depth2_median_parent_visits": d2, "rows": len(g), "max_depth": int(g["depth"].max())}
		md.append(f"\n## {run} — {len(g):,} steps, {g['tree_id'].nunique()} trees, max depth {g['depth'].max()}\n")
		md.append(f"**{ge2:.1%} of visits at depth ≥ 2; depth-2 parent nodes get a median of {d2:g} visits.**\n")
		md.append("| depth | visits | share | edges | median visits / parent node | ν q25 | ν median | ν q75 | reachable |")
		md.append("|---|---|---|---|---|---|---|---|---|")
		for r in p.itertuples():
			md.append(f"| {r.depth} | {r.visits:,} | {r.share:.1%} | {r.edges:,} | {r.median_visits_per_parent_node:g} | "
					  f"{r.nu_q25:.2f} | {r.nu_median:.2f} | {r.nu_q75:.2f} | {r.reachable_share:.0%} |")
	md.append("\n## Against yesterday\n")
	exp = {"D1": (0.75, 9.5), "D2": (0.69, 7.5)}
	for run, (share, d2) in exp.items():
		if run in out:
			o = out[run]
			md.append(f"- **{run}:** depth ≥ 2 share {o['share_visits_depth_ge2']:.1%} (brief: {share:.0%}); depth-2 parent "
					  f"median {o['depth2_median_parent_visits']:g} (P-VAR: {d2:g}; brief: ~8–10). "
					  + ("Confirmed." if abs(o["share_visits_depth_ge2"] - share) < 0.01 else "**Diverges.**"))
	if "D3" in out:
		o = out["D3"]
		md.append(f"- **D3 (Qwen):** depth ≥ 2 share {o['share_visits_depth_ge2']:.1%}; depth-2 parent median "
				  f"{o['depth2_median_parent_visits']:g}; max depth {o['max_depth']}.")
	open(os.path.join(WED, "depth_profile.md"), "w").write("\n".join(md) + "\n")
	update_wednesday("s3_depth_profile", out)
	print("\n".join(md))


if __name__ == "__main__":
	main()
