"""§5.2 five two-bucket keys, §5.3 redundancy regression, §5.4 cross-turn drift (Table A only).

    python analysis/wed/scripts/s5_bucket_selection.py [--runs D1 D2 D3]

All keys are compared on the SAME rows: steps where delta_nu exists (drops depth-1 steps of the first
planned turn, whose parent has no earlier user turn). K0 is also reported on all rows, to connect to
yesterday's 2.59 / 0.785 / 0.31 / -0.29.

Homogeneity is computed on backup_value (the quantity AffPool pools -- FREEZE_NOTES §8.2) and on z
(the quantity yesterday's 0.31 was measured on). Recommendation rule (brief §5.2): switch only if a
key beats K0 on homogeneity AND its multiplier is within 10 % of K0's AND |r(bucket, depth)| is no
worse. Evaluated on backup_value; z shown alongside.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import (WED, Boot, affpool_block, affpool_cells, ci, cohen_kappa, load_steps, point_biserial,  # noqa: E402
				 update_wednesday, with_ci, wmedian, jdump)

KEYS = {
	"K0": "bucket_med (parent_nu < tau_med)",
	"K1": "phi_08 < median",
	"K2": "phi_05 < median",
	"K3": "delta_nu < median",
	"K4": "nu_depth_adj < 0",
}
DEPTH_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 99)]


def add_keys(A):
	A = A.copy()
	A["K0"] = A["bucket_med"]
	A["K1"] = (A["phi_08"] < A["phi_08"].median()).astype(int)
	A["K2"] = (A["phi_05"] < A["phi_05"].median()).astype(int)
	A["K3"] = np.where(A["delta_nu"].isna(), np.nan, (A["delta_nu"] < A["delta_nu"].median()).astype(float))
	A["K4"] = (A["nu_depth_adj"] < 0).astype(int)
	return A


def corr_block(d, key, boot):
	by = {k: g for k, g in d.groupby("dialogue_id")}
	pt = (point_biserial(d[key], d["depth"]), point_biserial(d[key], d["turn_index"]))
	reps = boot.run(lambda ids: (lambda x: (point_biserial(x[key], x["depth"]), point_biserial(x[key], x["turn_index"])))(
		pd.concat([by[i] for i in ids])))
	return {"r_depth": with_ci(pt[0], [r[0] for r in reps]), "r_turn": with_ci(pt[1], [r[1] for r in reps])}


def s52(A, boot):
	d = A.dropna(subset=["delta_nu"]).copy()
	d["K3"] = d["K3"].astype(int)
	out = {"rows_compared": len(d), "rows_all": len(A),
		   "delta_nu_exact_zero_share": float((d["delta_nu"] == 0).mean())}
	k0_all, _ = affpool_block(A, "K0", "backup_value", boot)
	k0_all_z, _ = affpool_block(A, "K0", "z", boot)
	out["K0_all_rows"] = {"backup_value": k0_all, "z": k0_all_z}
	for k in KEYS:
		rec = {"definition": KEYS[k]}
		for value in ("backup_value", "z"):
			blk, cells = affpool_block(d, k, value, boot)
			rec[value] = blk
		# in-cell sharing per bucket
		rec["occupancy"] = {
			"overall_bucket1": float(d[k].mean()),
			"by_depth_bucket1": {f"{lo}-{hi}" if hi != lo else str(lo): float(d[(d["depth"] >= lo) & (d["depth"] <= hi)][k].mean())
								 for lo, hi in DEPTH_BINS},
		}
		rec.update(corr_block(d, k, boot))
		a, b = d["K0"].to_numpy(int), d[k].to_numpy(int)
		rec["agreement_with_K0"] = {"kappa": cohen_kappa(a, b),
									"table": {"K0=0,K=0": int(((a == 0) & (b == 0)).sum()), "K0=0,K=1": int(((a == 0) & (b == 1)).sum()),
											  "K0=1,K=0": int(((a == 1) & (b == 0)).sum()), "K0=1,K=1": int(((a == 1) & (b == 1)).sum())}}
		out[k] = rec
	base = out["K0"]
	for k in KEYS:
		r = out[k]
		for value in ("backup_value", "z"):
			hom = r[value]["between_within_ratio_noise_corrected_wmedian"]["value"]
			hom0 = base[value]["between_within_ratio_noise_corrected_wmedian"]["value"]
			raw = r[value]["between_within_ratio_wmedian"]["value"]
			raw0 = base[value]["between_within_ratio_wmedian"]["value"]
			mult = r[value]["evidence_multiplier_median"]["value"]
			mult0 = base[value]["evidence_multiplier_median"]["value"]
			r[f"rule_{value}"] = {
				"beats_K0_homogeneity_noise_corrected": bool(hom < hom0),
				"beats_K0_homogeneity_raw": bool(raw < raw0),
				"multiplier_within_10pct": bool(mult >= 0.9 * mult0),
				"depth_corr_no_worse": bool(abs(r["r_depth"]["value"]) <= abs(base["r_depth"]["value"]) + 1e-12),
			}
			r[f"rule_{value}"]["switch"] = bool(k != "K0" and r[f"rule_{value}"]["beats_K0_homogeneity_noise_corrected"]
												 and r[f"rule_{value}"]["beats_K0_homogeneity_raw"]
												 and r[f"rule_{value}"]["multiplier_within_10pct"]
												 and r[f"rule_{value}"]["depth_corr_no_worse"])
	return out


def r2(X, y):
	X = np.column_stack([np.ones(len(y)), X])
	b, *_ = np.linalg.lstsq(X, y, rcond=None)
	return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)


def s53(A, boot):
	res = {}
	for label, d in (("all", A), ("fresh", A[~A["child_from_cache"]])):
		d = d.dropna(subset=["nu_prev", "nu_prev2"])
		specs = {
			"m1_nu_t": ["parent_nu"],
			"m2_nu_t_t1_t2": ["parent_nu", "nu_prev", "nu_prev2"],
			"m3_nu_t_phi08": ["parent_nu", "phi_08"],
			"m4_nu_t_t1_t2_depth": ["parent_nu", "nu_prev", "nu_prev2", "depth"],
			"m5_nu_t_phi08_depth": ["parent_nu", "phi_08", "depth"],
			"m0_depth": ["depth"],
		}

		def fit(x):
			y = x["z"].to_numpy()
			return {k: r2(x[v].to_numpy(float), y) for k, v in specs.items()}
		pt = fit(d)
		by = {k: g for k, g in d.groupby("dialogue_id")}
		reps = boot.run(lambda ids: fit(pd.concat([by[i] for i in ids])))
		incs = {
			"history_over_current (m2-m1)": ("m2_nu_t_t1_t2", "m1_nu_t"),
			"phi08_over_current (m3-m1)": ("m3_nu_t_phi08", "m1_nu_t"),
			"depth_over_history (m4-m2)": ("m4_nu_t_t1_t2_depth", "m2_nu_t_t1_t2"),
			"depth_over_phi08 (m5-m3)": ("m5_nu_t_phi08_depth", "m3_nu_t_phi08"),
		}
		res[label] = {"n": len(d), "r2": {k: with_ci(v, [r[k] for r in reps]) for k, v in pt.items()},
					  "increments": {k: with_ci(pt[a] - pt[b], [r[a] - r[b] for r in reps]) for k, (a, b) in incs.items()},
					  "relative_gain_history_over_current": (pt["m2_nu_t_t1_t2"] - pt["m1_nu_t"]) / pt["m1_nu_t"],
					  "relative_gain_phi08_over_current": (pt["m3_nu_t_phi08"] - pt["m1_nu_t"]) / pt["m1_nu_t"]}
	return res


def drift_cells(d, key, value):
	"""(dialogue, bucket, act) cells pooled across the dialogue's turns; groups = turn_index."""
	rows = []
	for (dlg, b, act), g in d.groupby(["dialogue_id", key, "action"]):
		per = g.groupby("turn_index")[value].agg(["size", "mean", "var"])
		if len(per) < 3:
			continue
		ns = per["size"].to_numpy(float)
		dfw = np.clip(ns - 1, 0, None)
		if dfw.sum() == 0:
			continue
		within = float((per["var"].fillna(0).to_numpy() * dfw).sum() / dfw.sum())
		between = float(np.var(per["mean"].to_numpy(), ddof=1))
		rows.append({"dialogue_id": dlg, "bucket": b, "action": act, "n_turns": len(per), "N": int(ns.sum()),
					 "range": float(per["mean"].max() - per["mean"].min()),
					 "ratio": between / within if within > 0 else np.nan,
					 "ratio_noise_corrected": (between - float(np.mean(within / ns))) / within if within > 0 else np.nan})
	return pd.DataFrame(rows)


def s54(A, boot):
	d = A.dropna(subset=["delta_nu"]).copy()
	d["K3"] = d["K3"].astype(int)
	res = {}
	for k in ("K0", "K3"):
		res[k] = {}
		for value in ("backup_value", "z"):
			cells = drift_cells(d, k, value)
			by = {g: c for g, c in cells.groupby("dialogue_id")}

			def summ(c):
				return (wmedian(c["ratio"], c["N"]), wmedian(c["ratio_noise_corrected"], c["N"]), float(c["range"].median()))
			pt = summ(cells)
			reps = boot.run(lambda ids: summ(pd.concat([by[i] for i in ids if i in by])))
			# corpus-level: (bucket, act) cell mean by turn index across all dialogues
			glob = d.groupby([k, "action", "turn_index"])[value].agg(["size", "mean"]).reset_index()
			glob = glob[glob["size"] >= 30]
			ranges = glob.groupby([k, "action"])["mean"].agg(lambda s: s.max() - s.min())
			res[k][value] = {
				"cells": len(cells),
				"between_within_turn_ratio_wmedian": with_ci(pt[0], [r[0] for r in reps]),
				"noise_corrected_wmedian": with_ci(pt[1], [r[1] for r in reps]),
				"median_range_of_turn_means_within_dialogue": with_ci(pt[2], [r[2] for r in reps]),
				"corpus_level_range_across_turns_median": float(ranges.median()) if len(ranges) else np.nan,
				"corpus_level_cell_turn_means": {f"{b}|{a}": g.set_index("turn_index")["mean"].round(3).to_dict()
												 for (b, a), g in glob.groupby([k, "action"])},
			}
	# evidence under the two scopes, K0
	tree_cells = affpool_cells(d, "K0", "backup_value")
	dlg = d.assign(tree_id=d["dialogue_id"])  # per-dialogue scope: one "tree" per dialogue
	dlg_cells = affpool_cells(dlg, "K0", "backup_value")
	res["scope_evidence_K0"] = {"per_search_N_pool_median": float(tree_cells["N_pool"].median()),
								"per_dialogue_N_pool_median": float(dlg_cells["N_pool"].median()),
								"ratio": float(dlg_cells["N_pool"].median() / tree_cells["N_pool"].median())}
	return res


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--runs", nargs="*", default=None)
	a = ap.parse_args()
	allA = load_steps()
	runs = a.runs or sorted(allA["run_id"].unique())
	out = {}
	for run in runs:
		A = add_keys(allA[allA["run_id"] == run].reset_index(drop=True))
		boot = Boot(A["dialogue_id"].unique())
		out[run] = {"s5_2": s52(A, boot), "s5_3": s53(A, boot), "s5_4": s54(A, boot)}
		r = out[run]["s5_2"]
		for k in KEYS:
			print(run, k, {v: (round(r[k][v]["between_within_ratio_wmedian"]["value"], 3),
							   round(r[k][v]["between_within_ratio_noise_corrected_wmedian"]["value"], 3),
							   round(r[k][v]["evidence_multiplier_median"]["value"], 2)) for v in ("backup_value", "z")},
				  "r_depth", round(r[k]["r_depth"]["value"], 3), "kappa", round(r[k]["agreement_with_K0"]["kappa"], 3),
				  "switch", r[k]["rule_backup_value"]["switch"])
		print(run, "5.3", {k: round(v["value"], 4) for k, v in out[run]["s5_3"]["all"]["increments"].items()})
		print(run, "5.4", {k: {v: round(out[run]["s5_4"][k][v]["noise_corrected_wmedian"]["value"], 3) for v in ("backup_value", "z")} for k in ("K0", "K3")})
		jdump(out, os.path.join(WED, "s5_bucket_selection.json"))
	update_wednesday("s5_bucket_selection", out)


if __name__ == "__main__":
	main()
