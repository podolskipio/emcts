"""Per-run pilot metrics for Tasks 3-5, straight from a run directory's simlog + episode pickle.

    python analysis/thu/scripts/t_pilot_metrics.py <run_dir>[:tag] ... [--server_log LOG] [--T 10] [--out JSON]

run_dir is the inner directory setup_output_dir makes (analysis/thu/runs/<TAG>/<TAG>).

Per run:
  outcomes     dialogues completed, SR, AvgT (turns of the episode record)
  root         per planned turn: top-two visit gap, Q spread over expanded root actions (final
               simulation's backup folded in, as in headroom.md), expanded count
  predict      LLM value/prior calls, bounded from the simlog: every simulation ends either in exactly
               one leaf expansion (a `predict`) or at a terminal state. A last-step backup of exactly
               +/-1 may be either, so the count is reported as [lower, upper] plus the root expansion.
  selection    expanded siblings at each selection point (median, share with >= 2)
  affpool      (if logged) mean RAVE beta at the selected edge by depth, cells, prefixes per cell,
               pool-flip fraction
  signal       mean / sd of Q_emo on visited edges at selection, and of the per-visit z
  cost         wall clock (runs/<TAG>/wall_clock.txt); with --server_log, SGLang prefix-cache hit
               rate and prefill sequence count inside the run's [start, start + wall] window
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))  # episode pickles reference src classes
import lib  # noqa: E402


def records(run):
	steps, turns = [], []
	for p in sorted(glob.glob(os.path.join(run, "simlog", "*.ndjson.gz"))):
		for line in gzip.open(p, "rt"):
			r = json.loads(line)
			(steps if r.get("record_type") == "step" else turns).append(r)
	return steps, turns


def outcomes(run):
	pk = [p for p in glob.glob(os.path.join(run, "*.pkl"))]
	eps = pickle.load(open(pk[0], "rb")) if pk else []
	if not eps:
		return {"dialogues": 0}
	succ = np.array([bool(e["success"]) for e in eps])
	turns = np.array([e["num_turns"] for e in eps], float)
	return {"dialogues": int(len(eps)), "SR": float(succ.mean()), "wilson95": lib.wilson(int(succ.sum()), len(eps)),
			"AvgT": float(turns.mean()), "dialogue_ids": sorted(e["did"] for e in eps),
			"per_dialogue": {e["did"]: {"success": bool(e["success"]), "turns": int(e["num_turns"])} for e in eps}}


def root_stats(steps):
	df = pd.DataFrame([{"dlg": s["dlg_id"], "turn": s["turn_index"], "sim": s["simulation_index"], "depth": s["depth"],
						"action": s["action"], "v": s["backup_value"], "siblings": s["siblings"]}
					   for s in steps if s["depth"] == 1])
	rows = []
	for (d, t), g in df.groupby(["dlg", "turn"]):
		last = g[g["sim"] == g["sim"].max()].iloc[0]
		N = {sb["action"]: sb["N"] for sb in last["siblings"]}
		Q = {sb["action"]: sb["Q"] for sb in last["siblings"]}
		a = last["action"]  # fold the final backup into the selected edge
		Q[a] = (Q[a] * N[a] + last["v"]) / (N[a] + 1)
		N[a] += 1
		n = sorted(N.values(), reverse=True)
		qe = [Q[k] for k in N if N[k] > 0]
		rows.append({"dlg": d, "turn": t, "gap": n[0] - n[1], "total": sum(n), "n_expanded": len(qe),
					 "spread": (max(qe) - min(qe)) if len(qe) > 1 else 0.0})
	return pd.DataFrame(rows)


def predict_bounds(steps, turns, horizon):
	"""LLM predict calls (value + prior), reconstructed from tree structure.

	Every simulation ends at exactly one leaf. A leaf that was EXPANDED (a predict call) becomes a
	node: if it is ever selected from later, its key appears as a parent prefix. A TERMINAL leaf is
	never expanded, so it can only be revisited as a leaf. Classification of each simulation's leaf,
	keyed (dialogue, turn, prefix + action):
	  expanded    key appears as a parent in some step                         -> predict
	  terminal    never a parent, reached as a leaf in >= 2 simulations        -> no call
	  non-terminal value (under legacy only +1 ends a branch; under episode +/-1) and never a parent
	              -> predict (its first visit expanded it and nothing selected below it)
	  ambiguous   never a parent, reached once, value is a terminal value      -> reported separately
	Plus one root expansion per search.
	"""
	term = (1.0,) if horizon in (None, "legacy") else (1.0, -1.0)
	parents, leaf_hits, last = set(), {}, {}
	for s in steps:
		parents.add((s["dlg_id"], s["turn_index"], tuple(s["action_prefix"])))
		k = (s["dlg_id"], s["turn_index"], s["simulation_index"])
		if k not in last or s["depth"] > last[k]["depth"]:
			last[k] = s
	for s in last.values():
		key = (s["dlg_id"], s["turn_index"], tuple(s["action_prefix"]) + (s["action"],))
		leaf_hits[key] = leaf_hits.get(key, 0) + 1
	expanded = terminal = amb = 0
	for s in last.values():
		key = (s["dlg_id"], s["turn_index"], tuple(s["action_prefix"]) + (s["action"],))
		if key in parents or s["backup_value"] not in term:
			expanded += 1
		elif leaf_hits[key] >= 2:
			terminal += 1
		else:
			amb += 1
	roots = len({(s["dlg_id"], s["turn_index"]) for s in steps})
	lower = roots + expanded
	return {"search_roots": roots, "simulations_with_steps": len(last), "leaf_expanded": expanded,
			"leaf_terminal_revisited": terminal, "leaf_ambiguous": amb, "terminal_values_considered": list(term),
			"predict_calls_lower": lower, "predict_calls_upper": lower + amb,
			"per_root_lower": lower / max(roots, 1), "per_root_upper": (lower + amb) / max(roots, 1)}


def selection_stats(steps, T):
	n_exp = np.array([sum(1 for sb in s["siblings"] if sb["N"] > 0) for s in steps])
	qe_vis = np.array([sb["Q_emo"] for s in steps for sb in s["siblings"] if sb["N"] > 0])
	z = np.array([s["z"] for s in steps])
	child_len = np.array([s["turn_index"] + s["depth"] for s in steps])
	out = {"selection_points": int(len(steps)),
		   "expanded_siblings_median": float(np.median(n_exp)), "expanded_siblings_mean": float(n_exp.mean()),
		   "share_ge2_expanded": float((n_exp >= 2).mean()),
		   "Qemo_visited_mean": float(qe_vis.mean()) if len(qe_vis) else None,
		   "Qemo_visited_sd": float(qe_vis.std()) if len(qe_vis) else None,
		   "z_mean": float(z.mean()), "z_sd": float(z.std()),
		   "max_depth": int(max(s["depth"] for s in steps)),
		   "share_steps_child_past_T": float((child_len > T).mean()),
		   "fresh_share": float(np.mean([not s["child_from_cache"] for s in steps]))}
	if steps and "centre_flip" in steps[0]:
		out["centre_flip"] = float(np.mean([s["centre_flip"] for s in steps]))
	return out


def affpool_stats(steps):
	if not steps or "aff_bucket" not in steps[0]:
		return None
	df = pd.DataFrame([{"tree": (s["dlg_id"], s["turn_index"]), "depth": s["depth"], "bucket": s["aff_bucket"],
						"action": s["action"], "prefix": tuple(s["action_prefix"]), "beta": s["pool_beta_selected"],
						"flip": s["pool_flip"]} for s in steps])
	cells = df.groupby(["tree", "bucket", "action"]).agg(n=("beta", "size"), prefixes=("prefix", "nunique"))
	return {"mean_beta_by_depth": {str(k): float(v) for k, v in df.groupby("depth")["beta"].mean().items() if k <= 8},
			"mean_beta_selected": float(df["beta"].mean()),
			"pool_flip_fraction": float(df["flip"].mean()),
			"cells": int(len(cells)), "cells_per_tree": float(len(cells) / df["tree"].nunique()),
			"steps_per_cell_median": float(cells["n"].median()),
			"prefixes_per_cell_median": float(cells["prefixes"].median()),
			"share_cells_ge3_prefixes": float((cells["prefixes"] >= 3).mean()),
			"buckets_seen": sorted(int(b) for b in df["bucket"].unique())}


TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def server_cost(log, start_iso, wall_s):
	t0 = dt.datetime.fromisoformat(start_iso).replace(tzinfo=None)
	t1 = t0 + dt.timedelta(seconds=wall_s)
	seqs = new = cached = 0
	for line in open(log, errors="replace"):
		m = TS.match(line)
		if not m:
			continue
		t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
		if t0 <= t <= t1:
			seqs += int(m.group(2)); new += int(m.group(3)); cached += int(m.group(4))
	return {"prefill_sequences": seqs, "new_tokens": new, "cached_tokens": cached,
			"prefix_cache_hit_rate": cached / (new + cached) if new + cached else None}


def one(run, tag, a):
	steps, turns = records(run)
	outer = os.path.dirname(os.path.abspath(run.rstrip("/")))
	wall = start = None
	if os.path.exists(os.path.join(outer, "wall_clock.txt")):
		wall = int(open(os.path.join(outer, "wall_clock.txt")).read().strip().split("=")[1])
	if os.path.exists(os.path.join(outer, "started_at.txt")):
		start = open(os.path.join(outer, "started_at.txt")).read().strip()
	meta = json.load(open(os.path.join(run, "metadata.json")))
	R = root_stats(steps)
	res = {"tag": tag, "run": os.path.relpath(run, REPO),
		   "config": {k: meta["args"].get(k) for k in ("beta_emo", "emo_signal", "search_horizon", "aff_pool",
													   "aff_pool_bias", "aff_pool_key", "num_mcts_sims", "num_workers", "data")},
		   "outcomes": outcomes(run),
		   "root": {"n_roots": int(len(R)), "gap_le2": float((R["gap"] <= 2).mean()), "gap_median": float(R["gap"].median()),
					"spread_lt_0.01": float((R["spread"] < 0.01).mean()), "spread_median": float(R["spread"].median()),
					"n_expanded_median": float(R["n_expanded"].median()),
					"visits_per_root_median": float(R["total"].median())},
		   "predict": predict_bounds(steps, turns, meta["args"].get("search_horizon")),
		   "selection": selection_stats(steps, a.T),
		   "affpool": affpool_stats(steps),
		   "cost": {"wall_clock_s": wall, "started_at": start}}
	n = res["outcomes"].get("dialogues") or 0
	if wall and n:
		res["cost"]["wall_clock_s_per_dialogue"] = wall / n
		res["cost"]["wall_clock_s_per_turn"] = wall / max(sum(v["turns"] for v in res["outcomes"]["per_dialogue"].values()), 1)
	if a.server_log and wall and start:
		res["cost"]["server"] = server_cost(a.server_log, start, wall)
	res["_roots"] = R.to_dict("records")
	return res


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("runs", nargs="+")
	ap.add_argument("--server_log", default=None)
	ap.add_argument("--T", type=int, default=10)
	ap.add_argument("--out", default=None)
	a = ap.parse_args()
	out = {}
	for spec in a.runs:
		run, _, tag = spec.partition(":")
		tag = tag or os.path.basename(run.rstrip("/"))
		out[tag] = one(run, tag, a)
		show = {k: v for k, v in out[tag].items() if k not in ("_roots",)}
		show["outcomes"] = {k: v for k, v in show["outcomes"].items() if k not in ("per_dialogue", "dialogue_ids")}
		print(json.dumps(show, indent=1, default=float))
	if a.out:
		lib.jdump(out, a.out)


if __name__ == "__main__":
	main()
