"""§8 D1 (Vicuna-13B) vs D3 (Qwen2.5-7B), paired by dialogue (positions 101-130, same persona, same config).

    python analysis/wed/scripts/s8_qwen_comparison.py

Per run: episode SR / turns; affective range of REAL user turns and of sampled parent realizations;
negative-label rate; nu by depth; delta_nu occupancy; sigma_emo / |Q_emo| on edges with N >= 2 (from the
per-step z tape, identical to sqrt(M2_emo/(N-1))). Paired differences D3 - D1 use a bootstrap over the
dialogue ids both runs share.
"""
import glob, gzip, json, os, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from lib import WED, Boot, ci, with_ci, load_steps, update_wednesday, jdump  # noqa: E402

RUN_DIRS = {"D1": "analysis/phase1/runs/D1/D1", "D2": "analysis/phase1/runs/D2/D2", "D3": "analysis/wed/runs/D3/D3"}
NEG = {"fear", "sadness", "anger", "disgust"}


def turns(run_dir):
	rows = []
	for p in sorted(glob.glob(os.path.join(run_dir, "simlog", "*.ndjson.gz"))):
		for line in gzip.open(p, "rt"):
			r = json.loads(line)
			if r["record_type"] == "turn":
				d = {k.lower(): v for k, v in r["user_emotion_dist"].items()}
				rows.append({"dialogue_id": r["dlg_id"], "turn_index": r["turn_index"], "user_nu": r["user_nu"],
							 "label": max(d, key=d.get) if d else "", "planned": r["planned"]})
	return pd.DataFrame(rows)


def per_dialogue(run):
	A = load_steps(run)
	T = turns(RUN_DIRS[run])
	eps = {e["did"]: e for e in pickle.load(open(glob.glob(os.path.join(RUN_DIRS[run], "*.pkl"))[0], "rb"))}
	z = A.groupby("edge_id").agg(n=("z", "size"), m=("z", "mean"), s=("z", lambda x: x.std(ddof=1)), dlg=("dialogue_id", "first"))
	z = z[(z["n"] >= 2) & (z["m"].abs() > 0)]
	z["ratio"] = z["s"] / z["m"].abs()
	rows = []
	for dlg in sorted(set(A["dialogue_id"]) | set(T["dialogue_id"])):
		a, t = A[A["dialogue_id"] == dlg], T[T["dialogue_id"] == dlg]
		e = eps.get(dlg)
		rows.append({
			"dialogue_id": dlg, "success": float(e["success"]) if e else np.nan, "turns": float(e["num_turns"]) if e else np.nan,
			"real_user_nu_mean": t["user_nu"].mean(), "real_user_nu_sd": t["user_nu"].std(), "real_neg_rate": t["label"].isin(NEG).mean(),
			"parent_nu_mean": a["parent_nu"].mean(), "parent_nu_sd": a["parent_nu"].std(),
			"parent_neg_rate": a["parent_label"].isin(NEG).mean(), "z_sd": a["z"].std(),
			"dnu_below_-0.1": (a["delta_nu"] < -0.1).mean(), "sigma_ratio_median": z.loc[z["dlg"] == dlg, "ratio"].median(),
			"steps": len(a),
		})
	return pd.DataFrame(rows), A, T, z


def main():
	out, frames = {}, {}
	for run in ("D1", "D3"):
		P, A, T, z = per_dialogue(run)
		frames[run] = P
		nu_depth = A.groupby(A["depth"].clip(upper=10))["parent_nu"].agg(["size", "median", "mean",
								lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
		nu_depth.columns = ["n", "median", "mean", "q25", "q75"]
		out[run] = {
			"dialogues": int(P["dialogue_id"].nunique()), "episodes": int(P["success"].notna().sum()),
			"SR": float(P["success"].mean()), "avg_turns": float(P["turns"].mean()),
			"steps": len(A), "trees": int(A["tree_id"].nunique()),
			"real_user_turns": len(T), "real_user_nu": {"mean": float(T["user_nu"].mean()), "sd": float(T["user_nu"].std()),
													   "q10": float(T["user_nu"].quantile(.1)), "q90": float(T["user_nu"].quantile(.9))},
			"real_label_shares": T["label"].value_counts(normalize=True).round(3).to_dict(),
			"real_negative_label_rate": float(T["label"].isin(NEG).mean()),
			"parent_negative_label_rate": float(A["parent_label"].isin(NEG).mean()),
			"parent_label_shares": A["parent_label"].value_counts(normalize=True).round(3).to_dict(),
			"parent_nu": {"sd": float(A["parent_nu"].std()), "min": float(A["parent_nu"].min()), "max": float(A["parent_nu"].max())},
			"nu_by_depth": nu_depth.round(3).to_dict(orient="index"),
			"delta_nu": {"terciles": [float(A["delta_nu"].quantile(1 / 3)), float(A["delta_nu"].quantile(2 / 3))],
						 "share_negative": float((A["delta_nu"] < 0).mean()), "share_below_-0.1": float((A["delta_nu"] < -0.1).mean()),
						 "sd": float(A["delta_nu"].std()), "exact_zero": float((A["delta_nu"] == 0).mean())},
			"sigma_emo_over_abs_Q_emo": {"edges": len(z), "median": float(z["ratio"].median()), "p25": float(z["ratio"].quantile(.25)),
										 "p75": float(z["ratio"].quantile(.75)), "p90": float(z["ratio"].quantile(.9)),
										 "N_median": float(z["n"].median())},
			"reachable_share": float(A["reachable"].mean()), "max_depth": int(A["depth"].max()),
			"depth_ge2_share": float((A["depth"] >= 2).mean()),
			"act_shares": A["action"].value_counts(normalize=True).round(3).to_dict(),
		}
		print(run, {k: v for k, v in out[run].items() if not isinstance(v, dict)})
	shared = sorted(set(frames["D1"]["dialogue_id"]) & set(frames["D3"]["dialogue_id"]))
	a = frames["D1"].set_index("dialogue_id").loc[shared]
	b = frames["D3"].set_index("dialogue_id").loc[shared]
	boot = Boot(shared)
	paired = {}
	for col in ("success", "turns", "real_user_nu_mean", "real_user_nu_sd", "real_neg_rate", "parent_nu_mean", "parent_nu_sd",
				"parent_neg_rate", "z_sd", "dnu_below_-0.1", "sigma_ratio_median"):
		diff = (b[col] - a[col])
		reps = boot.run(lambda ids: float(np.nanmean([diff[i] for i in ids])))
		paired[col] = {"D1_mean": float(a[col].mean()), "D3_mean": float(b[col].mean()), "diff_D3_minus_D1": with_ci(float(np.nanmean(diff)), reps)}
	out["paired"] = {"n_dialogues": len(shared), "metrics": paired}
	for k, v in paired.items():
		print("paired", k, round(v["D1_mean"], 3), round(v["D3_mean"], 3), round(v["diff_D3_minus_D1"]["value"], 3), [round(x, 3) for x in v["diff_D3_minus_D1"]["ci"]])
	jdump(out, os.path.join(WED, "s8_qwen_comparison.json"))
	update_wednesday("s8_qwen_comparison", out)


if __name__ == "__main__":
	main()
