"""Before/after pool measures for the readiness report, on the 100-dialogue Phase 4 runs against their frozen
counterparts (same dialogues, same environment, no fixes; not a paired comparison -- a mechanical one).

    python analysis/readiness/scripts/report_pools.py

Per run: served-pool nu spread (p_depth.pool_spread, the brief's "within-pool nu sd" and "pools straddling
tau"), exact-parent / bucket mismatch and the no-draw floor (phase2_verify), depth-1 generations, and
visits per edge by depth. -> report_pools.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _env as E  # noqa: E402
import phase2_verify as P2  # noqa: E402

PAIRS = {"NoEmo": ("F_NoEmo_s20_seed1", "P4_NoEmo_seed1"), "ActPool": ("F_ActPool_s20_seed1", "P4_ActPool_seed1")}
P2.RUNS.update({
	"F_ActPool_s20_seed1": os.path.join(E.REPO, "analysis", "grid", "runs", "A_ActPool_s20_seed1", "A_ActPool_s20_seed1"),
	"P4_NoEmo_seed1": os.path.join(E.READINESS, "runs", "P4_NoEmo_seed1", "P4_NoEmo_seed1"),
	"P4_ActPool_seed1": os.path.join(E.READINESS, "runs", "P4_ActPool_seed1", "P4_ActPool_seed1"),
})
OUT = os.path.join(E.READINESS, "report_pools.json")


def pool_block(df):
	d = P2.p_depth.add_cache_state(df.assign(generating_parent_nu=df.generating_parent_nu.fillna(df.parent_nu)))
	ps = P2.p_depth.pool_spread(d, P2.TAU)
	sp = ps["served_pool"]
	return {"served_pool_edges": sp.get("edges"), "served_pool_sd_median": sp.get("sd_within_pool", {}).get("median"),
			"served_pool_sd_share_of_global": sp.get("median_sd_as_share_of_global_sd"),
			"served_pool_straddling_tau": sp.get("share_straddling_tau_med"),
			"global_nu_sd": ps["global_nu"]["sd"], "never_served": ps["pool_selection_bias"]["n_never_served"]}


def main():
	res = {"tau": P2.TAU, "git_head": E.git_head()}
	for arm, (before, after) in PAIRS.items():
		res[arm] = {}
		for label, tag in (("before", before), ("after", after)):
			df = P2.annotate(P2.steps(tag))
			m = P2.mismatch_block(df)
			res[arm][label] = {"tag": tag, "dialogues": int(df.dlg_id.nunique()), "pools": pool_block(df),
							   "depth1": P2.depth1(df),
							   "exact_parent_mismatch": m["exact_parent_mismatch"], "bucket_mismatch": m["bucket_mismatch"],
							   "median_abs_nu_gap": m["median_abs_nu_gap"], "floor": m["floor_parent_absent_from_pool"],
							   "draw": P2.draw_block(df)}
			print(arm, label, flush=True)
	json.dump(res, open(OUT, "w"), indent=1, default=str)
	print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
	main()
