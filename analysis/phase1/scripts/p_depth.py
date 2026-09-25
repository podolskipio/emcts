"""P-DEPTH — every affect statistic split by depth 1 vs depth >= 2, plus the realization-pool spread.

    python analysis/phase1/scripts/p_depth.py D1 analysis/phase1/D1 [--B 1000]

Motivation. Depth-1 steps are the only rows where the realization cache cannot corrupt the
(parent -> child) pairing: the root pool stays size 1 (`definitions.md`), so a cached depth-1 child
was generated from the same parent realization that is sampled now. If the affective signal is
materially stronger there, the cache is what suppresses it; if it is equally weak, the signal is weak
independent of the cache.

Estimators are imported from `p_var.py` unchanged, so every number is comparable to `p_var.md`.
Three things this script adds:

  1. `mismatch` as an explicit row-level variable: the currently sampled parent realization differs
     from the one the child was generated from. `generating_parent_nu` (built in `build_steps.py`)
     resolves on 100 % of rows, so this is exact, not inferred from `child_from_cache`.
  2. The depth-1 / depth->=2 split of the within-edge contrast, the pooled omega^2 and AffPool's
     homogeneity -- including, where the split makes a statistic inestimable, the count that shows why.
  3. The spread of nu across each node's cached realization pool (R = max_realizations), which turns
     the truncation argument into a number.

Output: <steps_dir>/p_depth_<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_var import Boot, affpool, ci, nodekey_contrast, pct, pooled_omega2  # noqa: E402

BUCKET = "bucket_med"
R = 4     # --max_realizations, the realization-pool cap (p_var.md: R = 4 for D1/D2)
TMAX = 10  # --max_turns; a child state at turn_index + depth > TMAX is unreachable by any episode


# ----------------------------------------------------------------------------------------------
# row-level cache state
# ----------------------------------------------------------------------------------------------
def add_cache_state(df):
	"""mismatch: the child was generated from a different parent realization than the one sampled now.

	Fresh rows are matched by construction. A cached row is matched only when the pool happened to
	return the child that this very parent produced (probability ~1/R if the pool is full)."""
	d = df.copy()
	if d["generating_parent_nu"].isna().any():
		raise ValueError("unresolved generating parent; p_depth needs an exact join")
	d["mismatch"] = np.abs(d["generating_parent_nu"] - d["parent_nu"]) > 1e-12
	d["matched"] = ~d["mismatch"]
	d["abs_pos"] = d["turn_index"] + d["depth"]
	# SEARCH_HORIZON_BUG.md: 19.7 % of D1 rows sit past Tmax and no episode can reach them. The
	# p_var.md correction note reports the reachable-only figures, so every headline here is
	# reported in both windows.
	d["reachable"] = d["abs_pos"] <= TMAX
	return d


def mismatch_audit(d):
	def block(x):
		cached = x[x["child_from_cache"]]
		return {"rows": int(len(x)), "cached_rows": int(len(cached)),
				"cached_share": float(x["child_from_cache"].mean()) if len(x) else np.nan,
				"mismatched_rows": int(x["mismatch"].sum()),
				"mismatch_rate_all_rows": float(x["mismatch"].mean()) if len(x) else np.nan,
				"mismatch_rate_among_cached": float(cached["mismatch"].mean()) if len(cached) else np.nan}

	out = {"all": block(d), "depth_1": block(d[d.depth == 1]), "depth_ge_2": block(d[d.depth >= 2]),
		   "by_depth": {int(k): block(g) for k, g in d.groupby("depth") if k <= 12}}
	# the pool-size prediction: a full pool of R returns the generating child with probability 1/R
	pool = d[d["child_from_cache"]].groupby("edge_key")["child_realization_id"].nunique()
	out["predicted_mismatch_among_cached_if_uniform_over_R"] = {
		"median_served_pool_size": float(pool.median()) if len(pool) else np.nan,
		"one_minus_one_over_R_at_R4": 0.75}
	return out


# ----------------------------------------------------------------------------------------------
# the three statistics, per depth stratum
# ----------------------------------------------------------------------------------------------
def estimability(d, bucket):
	"""Why a stratum can or cannot carry each estimator. Counts, not assertions."""
	per_edge = d.groupby("edge_key")[bucket].nunique()
	per_group = d.groupby(["dlg_id", "depth", "action"])[bucket].nunique()
	prefixes_per_cell = d.groupby(["dlg_id", "tree_key", bucket, "action"])["prefix_key"].nunique()
	return {
		"rows": int(len(d)),
		"edges": int(per_edge.size),
		"edges_with_both_buckets": int((per_edge >= 2).sum()),
		"share_edges_with_both_buckets": float((per_edge >= 2).mean()) if per_edge.size else np.nan,
		"omega2_groups_dlg_depth_action": int(per_group.size),
		"omega2_groups_with_both_buckets": int((per_group >= 2).sum()),
		"affpool_cells": int(prefixes_per_cell.size),
		"affpool_cells_with_ge3_prefixes": int((prefixes_per_cell >= 3).sum()),
		"distinct_prefixes_per_cell_median": float(prefixes_per_cell.median()) if prefixes_per_cell.size else np.nan,
		"distinct_prefixes_in_stratum": int(d["prefix_key"].nunique()),
	}


def stratum_block(d, bucket, boot, min_depth):
	"""within-edge contrast + pooled omega^2 (raw and edge-demeaned) + AffPool homogeneity."""
	out = {"rows": int(len(d)), "est": estimability(d, bucket)}
	if out["est"]["edges_with_both_buckets"] > 0:
		# nodekey_contrast hardcodes depth >= 2; shift a depth-1 frame so the same code runs on it
		dd = d.assign(depth=d["depth"] + 1) if min_depth == 1 else d
		out["within_edge_contrast"] = nodekey_contrast(dd, bucket, boot)
	else:
		out["within_edge_contrast"] = {"estimable": False,
									   "reason": "no edge in this stratum sees both buckets; the "
												 "bucket is constant within the edge"}
	if out["est"]["omega2_groups_with_both_buckets"] > 0:
		out["pooled_omega2"] = pooled_omega2(d, bucket, ["depth", "action"], boot, min_depth=min_depth)
		de = d.assign(z=d["z"] - d.groupby("edge_key")["z"].transform("mean"))
		out["pooled_omega2_edge_demeaned"] = pooled_omega2(de, bucket, ["depth", "action"], boot, min_depth=min_depth)
	else:
		out["pooled_omega2"] = {"estimable": False, "reason": "no (dialogue, depth, action) group sees both buckets"}
	if out["est"]["affpool_cells_with_ge3_prefixes"] > 0:
		ap, cells = affpool(d, bucket, boot, depth_keyed=False)
		out["affpool"] = ap
	else:
		out["affpool"] = {"estimable": False,
						  "reason": f"only {out['est']['distinct_prefixes_in_stratum']} distinct prefix(es) "
									f"in this stratum; homogeneity needs >= 3 prefixes sharing a cell"}
	return out


DEPTH1_OMEGA2_NOTE = (
	"At depth 1 the root pool stays size 1, so parent_nu == root_nu and the bucket is constant "
	"within a tree. The (depth, action) groups this omega^2 is computed over therefore draw ALL of "
	"their bucket variation from between trees and between dialogues -- it is the between-tree "
	"contrast, not a within-edge one, and it is exactly the component p_var.md 7.3 showed to be "
	"~90 % prefix/dialogue identity. The edge-demeaned counterpart is 0 by construction.")


# ----------------------------------------------------------------------------------------------
# the cache contrast at depth >= 2 (what depth 1 cannot give)
# ----------------------------------------------------------------------------------------------
def cache_contrast(d, boot):
	"""Same estimators, four ways of handling the cache, on the same depth->=2 rows.

	  as_planner_sees  -- bucket_med, every row. The planner's own view: for a mismatched row the
						  bucket belongs to a parent that did not produce this child.
	  causal_label     -- generating_bucket_med, every row. Same rows, corrected label.
	  matched_only     -- rows whose sampled parent IS the generating parent.
	  mismatched_only  -- the complement. Under random re-pairing this should be attenuated toward 0.
	  fresh_only       -- p_var.md's population, for continuity."""
	sub = d[d.depth >= 2]
	pops = {
		"as_planner_sees": (sub, BUCKET),
		"causal_label": (sub, "generating_bucket_med"),
		"matched_only": (sub[sub["matched"]], BUCKET),
		"mismatched_only": (sub[sub["mismatch"]], BUCKET),
		"fresh_only": (sub[~sub["child_from_cache"]], BUCKET),
	}
	out = {}
	for name, (x, bcol) in pops.items():
		rec = {"rows": int(len(x)), "bucket_column": bcol,
			   "within_edge_contrast": nodekey_contrast(x, bcol, boot),
			   "pooled_omega2": pooled_omega2(x, bcol, ["depth", "action"], boot)}
		xe = x.assign(z=x["z"] - x.groupby("edge_key")["z"].transform("mean"))
		rec["pooled_omega2_edge_demeaned"] = pooled_omega2(xe, bcol, ["depth", "action"], boot)
		out[name] = rec
	# how much does the re-pairing actually scramble the label? If a node's pool is bucket-homogeneous,
	# a mismatch costs nothing.
	mm = sub[sub["mismatch"]]
	out["label_corruption_on_mismatched_rows"] = {
		"rows": int(len(mm)),
		"share_bucket_label_changed": float((mm[BUCKET] != mm["generating_bucket_med"]).mean()) if len(mm) else np.nan,
		"share_bucket_label_changed_all_depth_ge_2": float((sub[BUCKET] != sub["generating_bucket_med"]).mean()),
		"mean_abs_nu_shift": float(np.abs(mm["parent_nu"] - mm["generating_parent_nu"]).mean()) if len(mm) else np.nan,
		"median_abs_nu_shift": float(np.abs(mm["parent_nu"] - mm["generating_parent_nu"]).median()) if len(mm) else np.nan,
	}
	# attenuation prediction: a share p of rows carrying a randomised label shrinks a contrast by
	# roughly (1 - p * share_label_changed * 2); reported as the observed ratio instead of assumed
	return out


# ----------------------------------------------------------------------------------------------
# realization-pool spread: is R aliasing a wide nu distribution down to R points?
# ----------------------------------------------------------------------------------------------
def pool_spread(d, tau):
	"""Per edge, the nu of the distinct child realizations the pool actually served, against the
	nu of every distinct realization the generator produced at that edge."""
	served = (d[d["child_from_cache"]].drop_duplicates(["edge_key", "child_realization_id"])
			  .loc[:, ["edge_key", "dlg_id", "depth", "child_realization_id", "child_nu"]])
	generated = (d[~d["child_from_cache"]].drop_duplicates(["edge_key", "child_realization_id"])
				 .loc[:, ["edge_key", "dlg_id", "depth", "child_realization_id", "child_nu"]])

	def agg(frame, label):
		g = frame.groupby("edge_key")["child_nu"].agg(n="size", lo="min", hi="max", sd=lambda s: float(s.std(ddof=1)),
													  mean="mean")
		g["range"] = g["hi"] - g["lo"]
		g["straddles_tau"] = (g["lo"] < tau) & (g["hi"] >= tau)
		g["label"] = label
		return g

	sv, gn = agg(served, "served_pool"), agg(generated, "generated")
	nu_all = d["child_nu"]
	glob = {"n": int(len(nu_all)), "sd": float(nu_all.std(ddof=1)),
			"range": float(nu_all.max() - nu_all.min()),
			"iqr": float(nu_all.quantile(0.75) - nu_all.quantile(0.25)),
			"min": float(nu_all.min()), "max": float(nu_all.max()), "tau_med": float(tau)}

	def summarize(g, min_n, exact_n=None):
		h = g[g["n"] == exact_n] if exact_n else g[g["n"] >= min_n]
		if not len(h):
			return {"edges": 0}
		return {
			"edges": int(len(h)),
			"pool_size_distribution": {int(k): int(v) for k, v in g["n"].value_counts().sort_index().items()},
			"sd_within_pool": {"median": float(h["sd"].median()), "p25": pct(h["sd"], 25), "p75": pct(h["sd"], 75),
							   "p90": pct(h["sd"], 90), "mean": float(h["sd"].mean())},
			"range_within_pool": {"median": float(h["range"].median()), "p25": pct(h["range"], 25),
								  "p75": pct(h["range"], 75), "p90": pct(h["range"], 90)},
			"median_range_as_share_of_global_range": float(h["range"].median() / glob["range"]),
			"median_range_as_share_of_global_iqr": float(h["range"].median() / glob["iqr"]),
			"median_sd_as_share_of_global_sd": float(h["sd"].median() / glob["sd"]),
			"share_straddling_tau_med": float(h["straddles_tau"].mean()),
			"share_range_gt_half_global_iqr": float((h["range"] > 0.5 * glob["iqr"]).mean()),
			"share_range_gt_global_iqr": float((h["range"] > glob["iqr"]).mean()),
		}

	# variance decomposition: how much of total nu variance is WITHIN a node's pool?
	def icc(frame):
		g = frame.groupby("edge_key")["child_nu"].agg(["size", "mean", "var"])
		g = g[g["size"] >= 2]
		if not len(g):
			return {}
		dfw = (g["size"] - 1).to_numpy(float)
		within = float((g["var"].fillna(0).to_numpy() * dfw).sum() / dfw.sum())
		between = float(np.var(g["mean"].to_numpy(), ddof=1))
		return {"edges": int(len(g)), "within_pool_var": within, "between_pool_var": between,
				"within_share_of_total": within / (within + between) if (within + between) > 0 else np.nan}

	out = {"global_nu": glob,
		   "served_pool": summarize(sv, 2), "generated": summarize(gn, 2),
		   # the headline cut: edges whose pool of R was filled AND fully exercised, so the spread is
		   # over exactly the R utterances the planner kept reusing
		   "served_pool_exactly_R": summarize(sv, 2, exact_n=R),
		   "variance_decomposition_served": icc(served), "variance_decomposition_generated": icc(generated)}

	# truncation: edges where the generator produced MORE distinct realizations than the pool holds
	both = sv[["n", "lo", "hi", "range", "sd"]].join(gn[["n", "lo", "hi", "range", "sd"]], lsuffix="_pool",
													rsuffix="_gen", how="inner")
	wide = both[(both["n_gen"] > both["n_pool"]) & (both["n_gen"] >= 3)]
	out["truncation"] = {
		"edges_with_both": int(len(both)),
		"edges_generator_produced_more_than_pool_served": int(len(wide)),
		"median_n_generated": float(wide["n_gen"].median()) if len(wide) else np.nan,
		"median_n_served": float(wide["n_pool"].median()) if len(wide) else np.nan,
		"median_pool_range_over_generated_range": float((wide["range_pool"] / wide["range_gen"].replace(0, np.nan)).median())
		if len(wide) else np.nan,
		"share_pool_range_below_half_generated": float((wide["range_pool"] < 0.5 * wide["range_gen"]).mean())
		if len(wide) else np.nan,
	}
	# bucket coin-flip: with p of a pool below tau, two independent draws disagree with prob 2p(1-p)
	served_b = d[d["child_from_cache"]].drop_duplicates(["edge_key", "child_realization_id"]).copy()
	served_b["below"] = served_b["child_nu"] < tau
	p = served_b.groupby("edge_key")["below"].agg(["mean", "size"])
	p = p[p["size"] >= 2]
	out["bucket_coin_flip"] = {
		"edges": int(len(p)),
		"mean_disagreement_prob_two_draws": float((2 * p["mean"] * (1 - p["mean"])).mean()) if len(p) else np.nan,
		"share_pools_mixed_on_tau": float(((p["mean"] > 0) & (p["mean"] < 1)).mean()) if len(p) else np.nan,
	}
	# selection bias: among edges whose pool filled to 4 and that generated more, do the served
	# realizations differ in nu from the ones the pool never retained?
	full = sv[sv["n"] == R].index
	cand = generated[generated["edge_key"].isin(full)].copy()
	served_ids = set(zip(served["edge_key"], served["child_realization_id"]))
	cand["was_served"] = [(e, c) in served_ids for e, c in zip(cand["edge_key"], cand["child_realization_id"])]
	ex = cand.groupby("edge_key").filter(lambda g: (~g["was_served"]).any())
	out["pool_selection_bias"] = {
		"edges_pool_full_and_generator_produced_more": int(ex["edge_key"].nunique()),
		"mean_nu_served": float(ex.loc[ex["was_served"], "child_nu"].mean()) if len(ex) else np.nan,
		"mean_nu_never_served": float(ex.loc[~ex["was_served"], "child_nu"].mean()) if len(ex) else np.nan,
		"n_served": int(ex["was_served"].sum()), "n_never_served": int((~ex["was_served"]).sum()),
	}
	return out


# ----------------------------------------------------------------------------------------------
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("tag")
	ap.add_argument("steps_dir")
	ap.add_argument("--B", type=int, default=1000)
	a = ap.parse_args()

	df = add_cache_state(pd.read_parquet(os.path.join(a.steps_dir, "steps.parquet")))
	boot = Boot(sorted(df["dlg_id"].unique()), a.B)
	tau = float(df["tau_med"].iloc[0])

	res = {"tag": a.tag, "rows": int(len(df)), "dialogues": int(df["dlg_id"].nunique()),
		   "trees": int(df["tree_key"].nunique()), "edges": int(df["edge_key"].nunique()),
		   "tau_med": tau, "bucket": BUCKET,
		   "bootstrap": {"B": a.B, "cluster": "dlg_id", "interval": "percentile 95%"}}

	print(f"[{a.tag}] mismatch audit", flush=True)
	res["mismatch"] = mismatch_audit(df)

	print(f"[{a.tag}] depth-1 stratum", flush=True)
	res["depth_1"] = stratum_block(df[df.depth == 1], BUCKET, boot, min_depth=1)
	res["depth_1"]["omega2_note"] = DEPTH1_OMEGA2_NOTE

	print(f"[{a.tag}] depth>=2 stratum", flush=True)
	res["depth_ge_2"] = stratum_block(df[df.depth >= 2], BUCKET, boot, min_depth=2)

	print(f"[{a.tag}] cache contrast (all rows)", flush=True)
	res["cache_contrast_depth_ge_2"] = cache_contrast(df, boot)
	print(f"[{a.tag}] cache contrast (reachable rows)", flush=True)
	res["cache_contrast_depth_ge_2_reachable"] = cache_contrast(df[df["reachable"]], boot)
	res["reachable_share"] = float(df["reachable"].mean())

	print(f"[{a.tag}] pool spread", flush=True)
	res["pool_spread"] = pool_spread(df, tau)
	res["pool_spread_reachable"] = pool_spread(df[df["reachable"]], tau)

	path = os.path.join(a.steps_dir, f"p_depth_{a.tag}.json")
	with open(path, "w") as f:
		json.dump(res, f, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))
	print(f"wrote {path}")


if __name__ == "__main__":
	main()
