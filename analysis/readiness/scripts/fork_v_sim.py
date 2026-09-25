"""The fork: does the value estimator predict SIMULATED donation? Zero GPU, frozen grid logs only.

    python analysis/readiness/scripts/fork_v_sim.py

Unit: one planned turn of one played dialogue. x = root v, the value estimator's score of the actual
dialogue state the planner searched from (subtree log, the search-root record's leaf_value). y = whether
that episode ended in a simulated donation (episode["success"]). Every finished grid run is used; the
cluster for the bootstrap is (run, dialogue).

Reported:
  pooled       AUC of v over every planned turn; logit(success) ~ turn  vs  ~ turn + v (McFadden increment)
  fixed turn   at planned turn k = 1..6, among dialogues still running at k: AUC of v, McFadden R2 of v
               -- turn count is constant within k, so nothing can leak through it
  per run      pooled AUC per run, to see that no single arm drives it
Human comparison (same estimator, human outcome): phase1_c1.json / fork_v_human.py.
"""
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402,F401  (puts src/ on sys.path for the pickles)
import _stats as S  # noqa: E402

GRID = os.path.join(E.REPO, "analysis", "grid", "runs")
OUT = os.path.join(E.READINESS, "fork_v_sim.json")
B = 1000
K_MAX = 6


def load():
	rows = []
	for rec in sorted(glob.glob(os.path.join(GRID, "*", "run_record.json"))):
		r = json.load(open(rec))
		if r.get("status") != "done":
			continue
		tag = r["tag"]
		eps = pickle.load(open(os.path.join(E.REPO, r["results"]["episodes_file"]), "rb"))
		for ep in eps:
			for line in gzip.open(ep["subtree_log"].replace("/mnt/c/Users/Piotr/PycharmProjects/EMCTS", E.REPO), "rt"):
				n = json.loads(line)
				if n["parent_id"] is None:
					rows.append({"run": tag, "arm": r["arm"], "n_sims": r["num_mcts_sims"], "dlg": ep["did"],
								 "cluster": f"{tag}|{ep['did']}", "turn": int(n["turn"]), "v": float(n["leaf_value"]),
								 "success": int(bool(ep["success"])), "num_turns": int(ep["num_turns"])})
	return pd.DataFrame(rows)


def auc(y, x):
	y, x = np.asarray(y), np.asarray(x, float)
	if len(set(y)) < 2:
		return np.nan
	r = pd.Series(x).rank().to_numpy()
	n1 = y.sum()
	return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def cluster_boot(df, stat, seed):
	rng = np.random.default_rng(seed)
	groups = {c: g for c, g in df.groupby("cluster")}
	keys = np.array(list(groups))
	reps = []
	for _ in range(B):
		sub = pd.concat([groups[k] for k in rng.choice(keys, len(keys))], ignore_index=True)
		reps.append(stat(sub))
	reps = [x for x in reps if np.isfinite(x)]
	return {"value": stat(df), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def mcfadden(df, cols):
	d = df.rename(columns={"success": "donated"})
	return S.T6.mcfadden(d, cols)[0]


def main():
	df = load()
	res = {"n_turn_rows": int(len(df)), "n_dialogue_runs": int(df.cluster.nunique()),
		   "runs": sorted(df.run.unique()), "success_rate_dialogues": float(df.groupby("cluster").success.first().mean())}
	res["pooled"] = {
		"auc_v": cluster_boot(df, lambda d: auc(d.success, d.v), 1),
		"auc_turn": cluster_boot(df, lambda d: auc(d.success, -d.turn), 2),
		"mcfadden_turn": mcfadden(df, ["turn"]),
		"mcfadden_turn_plus_v": mcfadden(df, ["turn", "v"]),
		"increment_v_over_turn": cluster_boot(df, lambda d: mcfadden(d, ["turn", "v"]) - mcfadden(d, ["turn"]), 3),
		"mcfadden_v": cluster_boot(df, lambda d: mcfadden(d, ["v"]), 4),
	}
	res["fixed_turn"] = {}
	for k in range(1, K_MAX + 1):
		d = df[df.turn == k]
		res["fixed_turn"][k] = {"n": int(len(d)), "success_rate": float(d.success.mean()),
								"v_mean": float(d.v.mean()), "v_sd": float(d.v.std()),
								"auc_v": cluster_boot(d, lambda s: auc(s.success, s.v), 10 + k),
								"mcfadden_v": cluster_boot(d, lambda s: mcfadden(s, ["v"]), 20 + k)}
	res["per_run_auc_v"] = {run: auc(g.success, g.v) for run, g in df.groupby("run")}
	res["git_head"] = E.git_head()
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
