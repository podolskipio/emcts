"""Shared preparation (brief §2): Table A (steps), Table B (selections), constants.

    python analysis/wed/scripts/build_tables.py                 # D1, D2 (+ D3 if its simlog exists)
    python analysis/wed/scripts/build_tables.py --runs D3=analysis/wed/runs/D3/D3

Every later script reads these files; none recomputes a derived column.

Definitions (also in constants.json["definitions"]):
  tree_id        (dialogue_id, turn_index) as "dlg|turn"
  edge_id        tree_id | action_prefix | action        -- a prefix is unique only inside a tree
  backup_value   the task value backed up into Q on this step. D1/D2 predate the field name; their
                 simlog `v` is written at the same call site, from the same variable (verified:
                 check_backfill in this script replays it into every logged sibling Q)
  nu history     the parent realization's OWN affect history: the observed user turns before the
                 search root, then every simulated user turn on the chain of realizations that
                 generated it (each realization id joined to the fresh step that produced it).
                 The parent is re-sampled from the node pool on every visit, so "the previous
                 step of this simulation" is a different dialogue ~2/3 of the time; the
                 realization's own history is what the simulated user actually saw.
  delta_nu       parent_nu - previous nu in that history; null when the history has one entry
                 (depth 1 at the first planned turn)
  phi_08/phi_05  normalized EWMA  sum g^(t-k) nu_k / sum g^(t-k)  over the history
  nu_depth_adj   parent_nu - median(parent_nu at this depth), per run
  bucket_med     parent_nu < tau_med (median over the run's rows)
  bucket_lab     parent argmax in {fear, sadness, anger, disgust}
  reachable      turn_index + depth <= Tmax (10): the child state is one a real episode can reach
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(REPO, "analysis", "wed")
NEG = {"fear", "sadness", "anger", "disgust"}
EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
TMAX = 10
DEFAULT_RUNS = {
	"D1": "analysis/phase1/runs/D1/D1",
	"D2": "analysis/phase1/runs/D2/D2",
	"D3": "analysis/wed/runs/D3/D3",
}


def load_simlog(run_dir):
	steps, turns = [], []
	for p in sorted(glob.glob(os.path.join(run_dir, "simlog", "*.ndjson.gz"))):
		with gzip.open(p, "rt") as f:
			for line in f:
				r = json.loads(line)
				(steps if r["record_type"] == "step" else turns).append(r)
	return steps, turns


def vec7(d):
	d = {k.lower(): v for k, v in (d or {}).items()}
	return [float(d.get(e, 0.0)) for e in EMO7]


def ewma(hist, g):
	w = g ** np.arange(len(hist) - 1, -1, -1)
	return float(np.dot(w, hist) / w.sum())


def build_run(run_id, run_dir):
	steps, turns = load_simlog(run_dir)
	if not steps:
		return None, None, None
	t = pd.DataFrame(turns)
	obs = {}  # (dlg, turn) -> observed user nu history before that turn's search root
	for dlg, g in t.sort_values("turn_index").groupby("dlg_id"):
		nus = g.set_index("turn_index")["user_nu"]
		for ti in g["turn_index"]:
			obs[(dlg, int(ti))] = [float(nus[k]) for k in sorted(nus.index) if k < ti]

	rows, sel = [], []
	# realization id -> (generating parent realization id) within a tree, from fresh steps
	gen_parent = {}
	for s in steps:
		if not s["child_from_cache"]:
			gen_parent.setdefault((s["dlg_id"], s["turn_index"], s["child_realization_id"]),
								  (s["parent_realization_id"], s["child_nu"]))
	hist_memo = {}

	def history(dlg, ti, rid, root_rid):
		key = (dlg, ti, rid)
		if key in hist_memo:
			return hist_memo[key]
		if rid == root_rid:
			h = obs[(dlg, ti)]
		else:
			parent_rid, nu = gen_parent[key]
			h = history(dlg, ti, parent_rid, root_rid) + [nu]
		hist_memo[key] = h
		return h

	root_rid = {}
	for s in steps:
		if s["depth"] == 1:
			root_rid.setdefault((s["dlg_id"], s["turn_index"]), s["parent_realization_id"])

	for s in steps:
		dlg, ti = s["dlg_id"], int(s["turn_index"])
		h = history(dlg, ti, s["parent_realization_id"], root_rid[(dlg, ti)])
		assert abs(h[-1] - s["parent_nu"]) < 1e-9, (dlg, ti, s["depth"])
		tree_id = f"{dlg}|{ti}"
		prefix = "__".join(s["action_prefix"])
		pdist = s["parent_emotion_dist"]
		plabel = max(pdist, key=pdist.get).lower() if pdist else ""
		rows.append({
			"run_id": run_id, "dialogue_id": dlg, "turn_index": ti,
			"simulation_index": int(s["simulation_index"]), "depth": int(s["depth"]),
			"action_prefix": prefix, "action": s["action"],
			"parent_realization_id": s["parent_realization_id"], "child_realization_id": s["child_realization_id"],
			"child_from_cache": bool(s["child_from_cache"]),
			"parent_nu": float(s["parent_nu"]), "parent_emotion_dist": vec7(pdist),
			"child_nu": float(s["child_nu"]), "child_emotion_dist": vec7(s["child_emotion_dist"]),
			"z": float(s["z"]), "backup_value": float(s.get("backup_value", s["v"])),
			"backup_value_source": "backup_value" if "backup_value" in s else "v (backfilled)",
			"tree_id": tree_id, "edge_id": f"{tree_id}|{prefix}|{s['action']}",
			"parent_label": plabel,
			"nu_prev": h[-2] if len(h) >= 2 else np.nan,
			"nu_prev2": h[-3] if len(h) >= 3 else np.nan,
			"delta_nu": h[-1] - h[-2] if len(h) >= 2 else np.nan,
			"phi_08": ewma(h, 0.8), "phi_05": ewma(h, 0.5), "hist_len": len(h),
			"root_nu": obs[(dlg, ti)][-1] if obs[(dlg, ti)] else np.nan,
			"reachable": ti + int(s["depth"]) <= TMAX,
		})
		for sib in s["siblings"]:
			sel.append({
				"run_id": run_id, "dialogue_id": dlg, "turn_index": ti,
				"simulation_index": int(s["simulation_index"]), "depth": int(s["depth"]), "tree_id": tree_id,
				"action_prefix": prefix, "action": sib["action"], "N": int(sib["N"]), "Q": float(sib["Q"]),
				"Q_emo": float(sib["Q_emo"]), "M2_emo": float(sib["M2_emo"]), "prior": float(sib["prior"]),
				"uct": float(sib["uct"]), "selected": sib["action"] == s["action"],
				"parent_nu": float(s["parent_nu"]),
			})
	A = pd.DataFrame(rows)
	tau = float(np.median(A["parent_nu"]))
	A["tau_med"] = tau
	A["bucket_med"] = (A["parent_nu"] < tau).astype(int)
	A["bucket_lab"] = A["parent_label"].isin(NEG).astype(int)
	A["nu_depth_adj"] = A["parent_nu"] - A.groupby("depth")["parent_nu"].transform("median")
	B = pd.DataFrame(sel)
	d = A["delta_nu"].dropna()
	const = {
		"run_dir": run_dir, "rows": len(A), "selection_rows": len(B), "dialogues": int(A["dialogue_id"].nunique()),
		"trees": int(A["tree_id"].nunique()), "edges": int(A["edge_id"].nunique()),
		"tau_med": tau,
		"delta_terciles": [float(np.quantile(d, 1 / 3)), float(np.quantile(d, 2 / 3))],
		"delta_median": float(np.median(d)), "phi_08_median": float(A["phi_08"].median()),
		"phi_05_median": float(A["phi_05"].median()),
		"delta_nu_null_rows": int(A["delta_nu"].isna().sum()),
		"backup_value_source": sorted(A["backup_value_source"].unique().tolist()),
		"reachable_share": float(A["reachable"].mean()),
		"turn_records": len(t),
		"check_backfill": check_backfill(A, B),
	}
	meta_path = os.path.join(run_dir, "metadata.json")
	if os.path.exists(meta_path):
		meta = json.load(open(meta_path))
		const["metadata"] = {k: meta.get(k) for k in meta if k in (
			"mcts_args", "args", "cmd_args", "runner", "mcts_class")} or meta
	return A, B, const


def check_backfill(A, B):
	"""Replay backup_value per edge in simulation order; the running mean before each later visit
	must equal the Q that edge's sibling record logged at that selection point."""
	A = A.sort_values(["tree_id", "simulation_index", "depth"])
	running = {}
	expect = {}
	for r in A[["tree_id", "simulation_index", "depth", "action_prefix", "action", "backup_value"]].itertuples(index=False):
		key = (r.tree_id, r.action_prefix, r.action)
		n, q = running.get(key, (0, 0.0))
		expect[(r.tree_id, r.simulation_index, r.depth, r.action_prefix, r.action)] = (n, q)
		running[key] = (n + 1, (n * q + r.backup_value) / (n + 1))
	# the step record at (sim, depth) logs Q BEFORE that step's backup, and the edge's earlier
	# backups all come from earlier simulations -> compare on the selected sibling
	sel = B[B["selected"]]
	bad = checked = 0
	for r in sel[["tree_id", "simulation_index", "depth", "action_prefix", "action", "N", "Q"]].itertuples(index=False):
		e = expect.get((r.tree_id, r.simulation_index, r.depth, r.action_prefix, r.action))
		if e is None:
			continue
		checked += 1
		if e[0] != r.N or abs(e[1] - r.Q) > 1e-9:
			bad += 1
	return {"checked": checked, "mismatches": bad}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--runs", nargs="*", default=None, help="RUN=run_dir pairs (default D1, D2, D3 if present)")
	a = ap.parse_args()
	runs = dict(DEFAULT_RUNS) if a.runs is None else dict(x.split("=", 1) for x in a.runs)
	os.chdir(REPO)
	constants_path = os.path.join(OUT, "constants.json")
	constants = json.load(open(constants_path)) if os.path.exists(constants_path) else {}
	As, Bs = [], []
	for run_id, run_dir in runs.items():
		if not glob.glob(os.path.join(run_dir, "simlog", "*.ndjson.gz")):
			print(f"[skip] {run_id}: no simlog under {run_dir}")
			continue
		A, B, c = build_run(run_id, run_dir)
		print(run_id, json.dumps({k: v for k, v in c.items() if k != "metadata"}))
		As.append(A)
		Bs.append(B)
		constants.setdefault("runs", {})[run_id] = c
	# merge with existing tables: replace the rebuilt runs only
	for name, frames in (("steps", As), ("selections", Bs)):
		path = os.path.join(OUT, f"{name}.parquet")
		new = pd.concat(frames, ignore_index=True)
		if os.path.exists(path) and a.runs is not None:
			old = pd.read_parquet(path)
			new = pd.concat([old[~old["run_id"].isin(new["run_id"].unique())], new], ignore_index=True)
		new.to_parquet(path, index=False)
		print(f"wrote {path}: {len(new)} rows, runs {sorted(new['run_id'].unique())}")
	constants["definitions"] = __doc__
	constants["config"] = {
		"beta_emo": {"D1": 0.7, "D2": 0.0, "D3": 0.7}, "R": 4, "K": 5, "n_sims": 50, "Tmax": TMAX,
		"classifier": "hf (DistilRoBERTa, 7 emitted labels)", "valence_table": "soft",
		"value_prior_scoring": "sampled (--logit_scoring off): 10-sample value estimator + top-K ranking-call prior",
		"search_horizon": "legacy", "seed": 0, "dialogues": "positions 101-130 of the non-annotated pool",
		"backbone": {"D1": "TheBloke/vicuna-13B-v1.5-AWQ", "D2": "TheBloke/vicuna-13B-v1.5-AWQ",
					 "D3": "Qwen/Qwen2.5-7B-Instruct-AWQ"},
	}
	json.dump(constants, open(constants_path, "w"), indent=1)


if __name__ == "__main__":
	sys.exit(main())
