"""TASK 1 -- headroom check. Can the environment reward action selection at all?

Reads the Wednesday tables (analysis/wed/selections.parquet, steps.parquet) and the Phase-1
per-turn tables (analysis/phase1/{D1,D2}/turns.parquet, which carry the authoritative final
root visit counts in root_visits_json). D1 and D2 are never averaged.

Four questions, per the Thursday brief:
  1. root tie rate          -- do the top two root actions separate in visits?
  2. root Q spread          -- is the decision made by search or by tiebreak?
  3. does the action matter -- realized planner SR against a random-act counterfactual
  4. n_sims sensitivity     -- does SR move with simulation budget?

Writes analysis/thu/headroom.{md,json}. Reports; decides nothing.
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
RUNS = ["D1", "D2"]
N_SIMS = 50


def final_root_visits(run):
	"""{tree_id: {action: visits}} after all n_sims simulations, from turns.parquet."""
	t = pd.read_parquet(os.path.join(REPO, "analysis", "phase1", run, "turns.parquet"))
	t = t[t["planned"]]
	out = {}
	for _, r in t.iterrows():
		v = r["root_visits_json"]
		v = json.loads(v) if isinstance(v, str) else dict(v or {})
		if v:
			out[f"{r['dlg_id']}|{int(r['turn_index'])}"] = {k: int(n) for k, n in v.items()}
	return out


def final_root_q(sel, steps, run):
	"""{tree_id: {action: (N_final, Q_final)}}.

	selections logs (N, Q) as they stood when that simulation selected at the root, i.e. after
	sim-1 backups. The last simulation's backup is therefore not in the table; we fold it in from
	steps.parquet (depth 1, last simulation) so the spread is the one the final decision saw.
	"""
	s = sel[sel["depth"] == 1]
	last = s.groupby("tree_id")["simulation_index"].max()
	tail = s[s["simulation_index"] == s["tree_id"].map(last)]
	st = steps[(steps["depth"] == 1)]
	st_last = st[st["simulation_index"] == st["tree_id"].map(last)]
	backup = {(r.tree_id, r.action): float(r.backup_value) for r in st_last.itertuples()}

	out = {}
	for tid, g in tail.groupby("tree_id"):
		d = {}
		for r in g.itertuples():
			N, Q = int(r.N), float(r.Q)
			v = backup.get((tid, r.action))
			if r.selected and v is not None:  # fold the final backup into this edge
				Q, N = (Q * N + v) / (N + 1), N + 1
			d[r.action] = (N, Q, float(r.prior))
		out[tid] = d
	return out


def gap_stats(visits):
	"""Top-two visit gap per root."""
	rows = []
	for tid, v in visits.items():
		n = sorted(v.values(), reverse=True)
		total = sum(n)
		top2 = n[0] - (n[1] if len(n) > 1 else 0)
		rows.append({"tree_id": tid, "n1": n[0], "n2": n[1] if len(n) > 1 else 0,
					 "gap": top2, "total": total, "gap_frac": top2 / total if total else np.nan})
	return pd.DataFrame(rows)


def q_stats(qmap):
	rows = []
	for tid, d in qmap.items():
		exp = {a: (N, Q, p) for a, (N, Q, p) in d.items() if N > 0}
		Qs_exp = [q for _, q, _ in exp.values()]
		Qs_all = [q for _, q, _ in d.values()]
		argmax_n = max(d.items(), key=lambda kv: kv[1][0])[0]
		argmax_q = max(exp.items(), key=lambda kv: kv[1][1])[0] if exp else None
		argmax_p = max(d.items(), key=lambda kv: kv[1][2])[0]
		rows.append({
			"tree_id": tid,
			"n_expanded": len(exp),
			"spread_expanded": (max(Qs_exp) - min(Qs_exp)) if len(Qs_exp) > 1 else 0.0,
			"spread_all": (max(Qs_all) - min(Qs_all)) if len(Qs_all) > 1 else 0.0,
			"q_max": max(Qs_exp) if Qs_exp else np.nan,
			"agree_n_q": int(argmax_n == argmax_q) if argmax_q else np.nan,
			"agree_n_prior": int(argmax_n == argmax_p),
		})
	return pd.DataFrame(rows)


def frac(mask, boot_ids, by_dialogue):
	"""Point estimate + cluster-bootstrap CI over dialogues for a per-root fraction."""
	p = float(np.mean(mask))
	reps = []
	for ids in boot_ids:
		vals = np.concatenate([by_dialogue[i] for i in ids if len(by_dialogue[i])])
		reps.append(float(np.mean(vals)) if len(vals) else np.nan)
	return {"value": p, "ci": lib.ci(reps), "n": int(len(mask))}


def run_one(run, sel_all, steps_all):
	sel = sel_all[sel_all["run_id"] == run]
	steps = steps_all[steps_all["run_id"] == run]
	visits = final_root_visits(run)
	qmap = final_root_q(sel, steps, run)

	g = gap_stats(visits)
	q = q_stats(qmap)
	df = g.merge(q, on="tree_id", how="inner")
	df["dialogue_id"] = df["tree_id"].str.rsplit("|", n=1).str[0]

	# cross-check: reconstruct final counts from selections and compare with turns.parquet
	recon = {}
	for tid, d in qmap.items():
		recon[tid] = {a: N for a, (N, _, _) in d.items()}
	mism = sum(1 for tid in recon if tid in visits and recon[tid] != visits[tid])

	boot = lib.Boot(sorted(df["dialogue_id"].unique()))
	ids_draws = [[boot.d[i] for i in idx] for idx in boot.draws]

	def by(col):
		return {k: v[col].to_numpy(float) for k, v in df.groupby("dialogue_id")}

	res = {
		"run": run,
		"n_roots": int(len(df)),
		"n_dialogues": int(df["dialogue_id"].nunique()),
		"n_sims": N_SIMS,
		"crosscheck_visit_mismatches": int(mism),
		"crosscheck_roots_compared": int(len(set(recon) & set(visits))),
		"tie": {
			"gap_le_1": frac(df["gap"] <= 1, ids_draws, by_bool(df, "gap", lambda x: x <= 1)),
			"gap_le_2": frac(df["gap"] <= 2, ids_draws, by_bool(df, "gap", lambda x: x <= 2)),
			"gap_le_5pct": frac(df["gap_frac"] <= 0.05, ids_draws, by_bool(df, "gap_frac", lambda x: x <= 0.05)),
			"gap_eq_0": frac(df["gap"] == 0, ids_draws, by_bool(df, "gap", lambda x: x == 0)),
			"gap_quantiles": {str(p): float(np.percentile(df["gap"], p)) for p in (5, 25, 50, 75, 95)},
			"gap_hist": {str(k): int(v) for k, v in df["gap"].value_counts().sort_index().items()},
			"gap_mean": float(df["gap"].mean()),
			"top1_share_median": float(np.median(df["n1"] / df["total"])),
		},
		"q_spread": {
			"below_0.01_expanded": frac(df["spread_expanded"] < 0.01, ids_draws,
										by_bool(df, "spread_expanded", lambda x: x < 0.01)),
			"below_0.01_all": frac(df["spread_all"] < 0.01, ids_draws,
								   by_bool(df, "spread_all", lambda x: x < 0.01)),
			"quantiles_expanded": {str(p): float(np.percentile(df["spread_expanded"], p)) for p in (5, 25, 50, 75, 95)},
			"mean_expanded": float(df["spread_expanded"].mean()),
			"one_expanded_only": frac(df["n_expanded"] <= 1, ids_draws,
									  by_bool(df, "n_expanded", lambda x: x <= 1)),
			"n_expanded_hist": {str(k): int(v) for k, v in df["n_expanded"].value_counts().sort_index().items()},
		},
		"agreement": {
			"argmaxN_eq_argmaxQ": frac(df["agree_n_q"].fillna(0) > 0, ids_draws,
									   by_bool(df, "agree_n_q", lambda x: x > 0)),
			"argmaxN_eq_argmaxPrior": frac(df["agree_n_prior"] > 0, ids_draws,
										   by_bool(df, "agree_n_prior", lambda x: x > 0)),
		},
	}
	return res, df


def by_bool(df, col, fn):
	d = {}
	for k, v in df.groupby("dialogue_id"):
		d[k] = fn(v[col].to_numpy(float)).astype(float)
	return d


def success_rates():
	"""SR per run over the 30 replay dialogues, with Wilson intervals."""
	out = {}
	paths = {"D1": "analysis/phase1/D1/turns.parquet",
			 "D2": "analysis/phase1/D2/turns.parquet",
			 "D1_horizon_sensitivity": "analysis/phase1/horizon_sensitivity/D1/turns.parquet",
			 "D2_horizon_sensitivity": "analysis/phase1/horizon_sensitivity/D2/turns.parquet"}
	for tag, p in paths.items():
		f = os.path.join(REPO, p)
		if not os.path.exists(f):
			continue
		t = pd.read_parquet(f)
		per = t.groupby("dlg_id")["outcome"].apply(lambda s: float((s == 1.0).any()))
		k, n = int(per.sum()), int(len(per))
		out[tag] = {"SR": k / n, "k": k, "n": n, "wilson95": lib.wilson(k, n),
					"AvgT": float(t[t["outcome"] != 0.0].groupby("dlg_id")["turn_index"].max().mean()),
					"per_dialogue": {str(a): float(b) for a, b in per.items()}}
	return out


def main():
	os.makedirs(THU, exist_ok=True)
	sel_all = lib.load_selections()
	steps_all = lib.load_steps()
	out = {"config": json.load(open(os.path.join(REPO, "analysis", "wed", "constants.json")))["config"],
		   "runs": {}}
	frames = {}
	for run in RUNS:
		res, df = run_one(run, sel_all, steps_all)
		out["runs"][run] = res
		frames[run] = df
		df.to_parquet(os.path.join(THU, f"roots_{run}.parquet"))
	out["success"] = success_rates()
	lib.jdump(out, os.path.join(THU, "headroom.json"))
	print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
	main()
