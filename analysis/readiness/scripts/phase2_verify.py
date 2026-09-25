"""Readiness Phase 2 -- mechanical verification of the cache fixes (PREREG Entry 9).

    python analysis/readiness/scripts/phase2_verify.py

Flattens each run's simlog with analysis/phase1/scripts/build_steps.py (into analysis/readiness/steps/<tag>/),
then computes every Entry 9 check on the two Phase 2 runs and, as a mechanical reference, the same measures
on the frozen A_NoEmo_s20_seed1 / A_AffPool_s20_seed1 restricted to the same dialogues. -> phase2_verify.json

Row order within a tree is the order the planner took the steps (simulation, then depth), which the simlog
preserves; every "at the time of this draw" quantity below is computed by walking the rows in that order.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _env as E  # noqa: E402

sys.path.insert(0, os.path.join(E.REPO, "analysis", "phase1", "scripts"))
import build_steps  # noqa: E402
import p_depth  # noqa: E402

TAU = 0.263
R = 4
B = 1000
SEED = 20260925
RUNS = {
	"P2_NoEmo_fixes_n25": os.path.join(E.READINESS, "runs", "P2_NoEmo_fixes_n25", "P2_NoEmo_fixes_n25"),
	"P2_AffPool_fixes_n25": os.path.join(E.READINESS, "runs", "P2_AffPool_fixes_n25", "P2_AffPool_fixes_n25"),
	"F_NoEmo_s20_seed1": os.path.join(E.REPO, "analysis", "grid", "runs", "A_NoEmo_s20_seed1", "A_NoEmo_s20_seed1"),
	"F_AffPool_s20_seed1": os.path.join(E.REPO, "analysis", "grid", "runs", "A_AffPool_s20_seed1", "A_AffPool_s20_seed1"),
}
REFERENCE_OF = {"P2_NoEmo_fixes_n25": "F_NoEmo_s20_seed1", "P2_AffPool_fixes_n25": "F_AffPool_s20_seed1"}
OUT = os.path.join(E.READINESS, "phase2_verify.json")


def steps(tag):
	out = os.path.join(E.READINESS, "steps", tag)
	if not os.path.exists(os.path.join(out, "steps.parquet")):
		build_steps.main(RUNS[tag], out)
	return pd.read_parquet(os.path.join(out, "steps.parquet"))


def annotate(df):
	"""Walk each tree in step order and attach, per row: the id of the parent realization that generated the
	child, and -- for cache hits -- the edge's generated set at that moment (size, ended count, and how many
	of them this row's parent generated)."""
	df = df.reset_index(drop=True).copy()
	ends = df["child_ends_search"].astype(bool) if "child_ends_search" in df else pd.Series(False, index=df.index)
	gen_parent, pool_n, pool_ended, pool_same_parent, pool_parents = [], [], [], [], []
	edges = {}  # edge_key -> {child_id: (generating parent id, ended)}
	for i, r in enumerate(df.itertuples(index=False)):
		e = edges.setdefault(r.edge_key, {})
		if not r.child_from_cache:
			e.setdefault(r.child_realization_id, (r.parent_realization_id, bool(ends.iat[i])))
			pool_n.append(np.nan); pool_ended.append(np.nan); pool_same_parent.append(np.nan); pool_parents.append(np.nan)
		else:
			pool_n.append(len(e))
			pool_ended.append(sum(x[1] for x in e.values()))
			pool_same_parent.append(sum(x[0] == r.parent_realization_id for x in e.values()))
			pool_parents.append(len({x[0] for x in e.values()}))
		gen_parent.append(e.get(r.child_realization_id, (None, None))[0])
	df["generating_parent_id"] = gen_parent
	df["pool_n"], df["pool_ended"], df["pool_same_parent"] = pool_n, pool_ended, pool_same_parent
	df["pool_distinct_parents"] = pool_parents
	df["ends"] = ends.to_numpy()
	return df


def boot(df, stat, seed=SEED):
	rng = np.random.default_rng(seed)
	idx = list(df.groupby("dlg_id").indices.values())
	reps = []
	for _ in range(B):
		pick = rng.integers(0, len(idx), len(idx))
		reps.append(stat(df.iloc[np.concatenate([idx[i] for i in pick])]))
	reps = [x for x in reps if np.isfinite(x)]
	return {"value": float(stat(df)), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def mismatch_block(df):
	c2 = df[df.child_from_cache & (df.depth >= 2)]
	if not len(c2):
		return {"cached_depth_ge_2_rows": 0}
	return {
		"cached_depth_ge_2_rows": int(len(c2)),
		"exact_parent_mismatch": boot(c2, lambda d: float((d.generating_parent_id != d.parent_realization_id).mean())),
		"nu_based_mismatch_p_depth": float((np.abs(c2.generating_parent_nu - c2.parent_nu) > 1e-12).mean()),
		"uniform_counterfactual_mismatch": boot(c2, lambda d: float((1 - d.pool_same_parent / d.pool_n).mean())),
		"bucket_mismatch": float(((c2.generating_parent_nu < TAU) != (c2.parent_nu < TAU)).mean()),
		"median_abs_nu_gap": float(np.abs(c2.generating_parent_nu - c2.parent_nu).median()),
		"floor_parent_absent_from_pool": boot(c2, lambda d: float((d.pool_same_parent == 0).mean()), SEED + 3),
		# why exact-parent mismatch moves: it is bounded below by how many distinct parents fed the pool
		"by_depth": {("2" if k == 2 else "3+"): {
			"rows": int(len(g)),
			"exact_parent_mismatch": float((g.generating_parent_id != g.parent_realization_id).mean()),
			"uniform_counterfactual_mismatch": float((1 - g.pool_same_parent / g.pool_n).mean()),
			"bucket_mismatch": float(((g.generating_parent_nu < TAU) != (g.parent_nu < TAU)).mean()),
			"mean_distinct_generating_parents_in_pool": float(g.pool_distinct_parents.mean()),
			# no draw rule can match a parent that generated nothing in the pool: the floor on exact mismatch
			"floor_parent_absent_from_pool": float((g.pool_same_parent == 0).mean()),
			"mean_pool_size": float(g.pool_n.mean())}
			for k, g in c2.assign(dd=np.minimum(c2.depth, 3)).groupby("dd")},
	}


def never_served(df):
	d = p_depth.add_cache_state(df.assign(generating_parent_nu=df.generating_parent_nu.fillna(df.parent_nu)))
	return p_depth.pool_spread(d, TAU)["pool_selection_bias"]


def depth1(df):
	d1 = df[df.depth == 1]
	per_edge = d1.groupby("edge_key").agg(visits=("child_from_cache", "size"),
										 generations=("child_from_cache", lambda s: int((~s).sum())))
	return {"depth1_rows": int(len(d1)), "depth1_cache_hits": int(d1.child_from_cache.sum()),
			"depth1_edges": int(len(per_edge)),
			"generations_per_depth1_edge_mean": float(per_edge.generations.mean()),
			"visits_per_depth1_edge_mean": float(per_edge.visits.mean()),
			"generations_per_depth1_edge_median": float(per_edge.generations.median()),
			"visits_per_edge_by_depth": {int(k): float(v) for k, v in
										 df.groupby(["depth", "edge_key"]).size().groupby("depth").mean().items() if k <= 4}}


def ended_checks(df):
	hits = df[df.child_from_cache]
	ended_ids = set(df.loc[df.ends, "child_realization_id"])
	res = {
		"ended_replies_generated": int((df.ends & ~df.child_from_cache).sum()),
		"cache_hits": int(len(hits)),
		"cache_hits_on_ended_reply": int(hits.ends.sum()),
		"search_from_ended_state": int(df.parent_realization_id.isin(ended_ids).sum()),
	}
	if len(hits):
		res["draw_rate_observed"] = float(hits.ends.mean())
		res["draw_rate_expected"] = float((hits.pool_ended / hits.pool_n).mean())
		res["draw_rate_diff"] = boot(hits, lambda d: float(d.ends.mean() - (d.pool_ended / d.pool_n).mean()), SEED + 1)
	# ended replies never served vs the uniform expectation prod(1 - 1/m) over the hits after filing
	obs, exp_ = 0, 0.0
	for e, g in df.groupby("edge_key", sort=False):
		gen = g[~g.child_from_cache & g.ends]
		for pos, cid in zip(gen.index, gen.child_realization_id):
			later = g[(g.index > pos) & g.child_from_cache]
			if not len(later):
				continue
			obs += int(not (later.child_realization_id == cid).any())
			exp_ += float(np.prod(1 - 1 / later.pool_n.to_numpy(float)))
	res["ended_with_later_hits_never_served"] = obs
	res["ended_never_served_expected_uniform"] = exp_
	return res


def draw_block(df):
	h = df[df.child_from_cache]
	if "cache_draw_bucket_miss" not in h or not len(h):
		return {}
	miss = h.cache_draw_bucket_miss.astype(bool)
	ok = h[~miss]
	return {"cached_rows": int(len(h)), "bucket_hit_rate": boot(h, lambda d: 1 - float(d.cache_draw_bucket_miss.astype(bool).mean()), SEED + 2),
			"bucket_invariant_share": float(((ok.child_generating_parent_nu < TAU) == (ok.parent_nu < TAU)).mean()),
			"logged_vs_joined_generating_nu_max_abs_diff": float(np.abs(h.child_generating_parent_nu - h.generating_parent_nu).max())}


def main():
	res = {"tau": TAU, "git_head": E.git_head()}
	data = {t: annotate(steps(t)) for t in ("P2_NoEmo_fixes_n25", "P2_AffPool_fixes_n25")}
	for new, ref in REFERENCE_OF.items():
		ids = set(data[new].dlg_id)
		data[ref] = annotate(steps(ref).pipe(lambda d: d[d.dlg_id.isin(ids)]))
	for tag, df in data.items():
		print(tag, len(df), flush=True)
		res[tag] = {"rows": int(len(df)), "dialogues": int(df.dlg_id.nunique()),
					"unresolved_generating_parent": int(df.generating_parent_id.isna().sum()),
					"never_served_p_depth": never_served(df), "depth1": depth1(df),
					"mismatch": mismatch_block(df), "draw": draw_block(df)}
		if "child_ends_search" in df:
			res[tag]["ended"] = ended_checks(df)
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
