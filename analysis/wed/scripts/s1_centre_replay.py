"""§1.2 -- CenteredBias on D1's logged selection points (beta = 0.7): one-step replay.

    python analysis/wed/scripts/s1_centre_replay.py

uct_centre(a) = uct(a) - beta*mu for expanded siblings (N > 0), unchanged for unexpanded ones,
mu = mean Q_emo over the expanded siblings. Exact for one selection on the logged tree; the arm
would have grown a different tree, which the pilot measures.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, with_ci, load_selections, update_wednesday  # noqa: E402

BETA = 0.7
KEYS = ["dialogue_id", "turn_index", "simulation_index", "depth"]


def main():
	B = load_selections("D1")
	B["expanded"] = B["N"] > 0
	mu = B[B["expanded"]].groupby(KEYS)["Q_emo"].mean().rename("mu")
	B = B.join(mu, on=KEYS)
	B["uct_c"] = np.where(B["expanded"], B["uct"] - BETA * B["mu"], B["uct"])
	g = B.groupby(KEYS, sort=False)
	pts = g.agg(n=("action", "size"), n_exp=("expanded", "sum"), mu=("mu", "first")).reset_index()
	pts["chosen_bias"] = B.loc[g["uct"].idxmax().values, "action"].values
	pts["chosen_centre"] = B.loc[g["uct_c"].idxmax().values, "action"].values
	pts["centre_choice_expanded"] = B.loc[g["uct_c"].idxmax().values, "expanded"].values
	pts["flip"] = pts["chosen_bias"] != pts["chosen_centre"]
	pts["has_unexpanded"] = pts["n_exp"] < pts["n"]
	by = {k: v for k, v in pts.groupby("dialogue_id")}
	boot = Boot(pts["dialogue_id"].unique())

	def stat(ids):
		p = pd.concat([by[i] for i in ids])
		f = p[p["flip"]]
		return (float(p["flip"].mean()), float(p["has_unexpanded"].mean()),
				float((~f["centre_choice_expanded"]).mean()) if len(f) else np.nan,
				float(p["mu"].mean()))
	reps = boot.run(stat)
	pt = stat(list(by))
	out = {
		"flip_rate": with_ci(pt[0], [r[0] for r in reps]),
		"share_points_with_unexpanded_edges": with_ci(pt[1], [r[1] for r in reps]),
		"share_flips_to_unexpanded": with_ci(pt[2], [r[2] for r in reps]),
		"mu_mean": with_ci(pt[3], [r[3] for r in reps]),
		"mu_quantiles": {q: float(pts["mu"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
		"flip_rate_where_unexpanded_remain": float(pts.loc[pts["has_unexpanded"], "flip"].mean()),
		"flip_rate_all_expanded": float(pts.loc[~pts["has_unexpanded"], "flip"].mean()),
		"expanded_count_distribution_bias_D1": pts["n_exp"].value_counts(normalize=True).sort_index().to_dict(),
		"flip_by_depth": pts.groupby(pts["depth"].clip(upper=8))["flip"].mean().to_dict(),
	}
	D2 = load_selections("D2").assign(expanded=lambda d: d["N"] > 0).groupby(KEYS)["expanded"].sum()
	out["expanded_count_distribution_D2_beta0"] = D2.value_counts(normalize=True).sort_index().to_dict()
	for k, v in out.items():
		print(k, v)
	update_wednesday("s1_2_centre_replay_D1", out)


if __name__ == "__main__":
	main()
