"""Readiness Phase 5, decision level -- how much does retention (#6) move Q? (brief v2 Phase 5, "Report:
decision level first").

    python analysis/readiness/scripts/phase5_decision.py [TAG ...]

Measured inside a run that has #6 on, so no second run and no pairing is needed. On every edge that served
at least one cache hit, compare
  Q_fixed    the mean backed-up value over the edge's visits, as the planner saw it (fresh + cached draws)
  Q_frozen   the same mean with the cached draws re-weighted as the frozen cache would have drawn them:
             uniformly over the NON-terminal replies in the pool only (the frozen cache never held an ended
             reply). A cached draw of a live reply r counts with weight
                 (pool_n / pool_live) if r is live,   0 if r ended,
             so the cached draws' expected value becomes their mean over the live replies.
The difference Q_fixed - Q_frozen is the bias #6 removes. "Success-reachable" edges are those where at least
one generated reply ended in success (v = +1). Ended failures (v = -1, turn limit) push the other way, so both
are reported. Cluster bootstrap over dialogues. -> phase5_decision.json
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

B, SEED = 1000, 20260926
OUT = os.path.join(E.READINESS, "phase5_decision.json")
RUN_DIRS = {t: os.path.join(E.READINESS, "runs", t, t) for t in
			("P2_NoEmo_fixes_n25", "P2_AffPool_fixes_n25", "P4_NoEmo_seed1", "P4_ActPool_seed1", "P4_NoEmo_seed2")}


def steps(tag):
	P2.RUNS.setdefault(tag, RUN_DIRS[tag])
	return P2.annotate(P2.steps(tag))


def edge_table(df):
	df = df.copy()
	value = df["backup_value"].astype(float)
	# the edge's replies: live vs ended (success / failure), from its fresh generations
	fresh = df[~df.child_from_cache]
	kind = {}
	for r in fresh.itertuples(index=False):
		kind.setdefault((r.edge_key, r.child_realization_id),
						"live" if not r.ends else ("success" if r.backup_value > 0 else "failure"))
	df["kind"] = [kind.get((e, c), "live") for e, c in zip(df.edge_key, df.child_realization_id)]
	hit = df.child_from_cache
	# pool composition at the time of each hit: live replies among those generated so far at the edge
	live_counts = {}
	pool_live = []
	for r in df.itertuples(index=False):
		d = live_counts.setdefault(r.edge_key, set())
		if not r.child_from_cache and r.kind == "live":
			d.add(r.child_realization_id)
		pool_live.append(len(d))
	df["pool_live"] = pool_live
	w = np.ones(len(df))
	cached_live = hit & (df.kind == "live") & (df.pool_live > 0)
	w[cached_live.to_numpy()] = (df.pool_n / df.pool_live)[cached_live].to_numpy()
	w[(hit & (df.kind != "live")).to_numpy()] = 0.0
	df["w_frozen"], df["value"] = w, value
	rows = []
	for e, g in df.groupby("edge_key", sort=False):
		if not g.child_from_cache.any():
			continue
		# edges where every cached draw was an ended reply have no frozen counterpart
		if g.loc[g.child_from_cache, "w_frozen"].sum() == 0:
			continue
		q_fixed = g.value.mean()
		q_frozen = float(np.average(g.value, weights=g.w_frozen))
		kinds = set(g.loc[~g.child_from_cache, "kind"])
		rows.append({"edge_key": e, "dlg_id": g.dlg_id.iloc[0], "depth": int(g.depth.iloc[0]), "visits": len(g),
					 "q_fixed": q_fixed, "q_frozen": q_frozen, "shift": q_fixed - q_frozen,
					 "success_reachable": "success" in kinds, "failure_reachable": "failure" in kinds,
					 "share_hits_ended": float((g.child_from_cache & (g.kind != "live")).sum() / g.child_from_cache.sum())})
	return pd.DataFrame(rows)


def boot_mean(t, col, seed):
	if not len(t):
		return None
	rng = np.random.default_rng(seed)
	idx = list(t.groupby("dlg_id").indices.values())
	reps = [float(t.iloc[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])][col].mean()) for _ in range(B)]
	return {"value": float(t[col].mean()), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
			"edges": int(len(t))}


def summarise(t):
	out = {"edges_with_cache_hits": int(len(t))}
	for name, sub in (("all", t), ("success_reachable", t[t.success_reachable]),
					  ("success_only_reachable", t[t.success_reachable & ~t.failure_reachable]),
					  ("failure_only_reachable", t[t.failure_reachable & ~t.success_reachable]),
					  ("neither", t[~t.success_reachable & ~t.failure_reachable])):
		out[name] = {"shift": boot_mean(sub, "shift", SEED), "q_fixed": float(sub.q_fixed.mean()) if len(sub) else None,
					 "q_frozen": float(sub.q_frozen.mean()) if len(sub) else None,
					 "share_hits_ended": float(sub.share_hits_ended.mean()) if len(sub) else None}
	return out


def main():
	tags = sys.argv[1:] or [t for t in RUN_DIRS if os.path.exists(RUN_DIRS[t])]
	res = {"git_head": E.git_head()}
	for tag in tags:
		print(tag, flush=True)
		res[tag] = summarise(edge_table(steps(tag)))
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
