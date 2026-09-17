"""§5.1 -- the AffPool homogeneity gate on the TASK RETURN (backup_value) instead of z.

    python analysis/wed/scripts/s51_affpool_gate_recheck.py

Asserts first that lib.affpool_cells reproduces yesterday's D1 z numbers (0.785 / 0.312 / 2.59),
so the only thing that changes between the two columns of the table is the pooled quantity.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, affpool_block, load_steps, update_wednesday, jdump  # noqa: E402

YESTERDAY = {"D1": {"ratio": 0.785, "nc": 0.312, "mult": 2.59}, "D2": {"ratio": 0.783, "nc": 0.29, "mult": 2.50}}


def main():
	out = {}
	for run in ("D1", "D2"):
		A = load_steps(run)
		boot = Boot(A["dialogue_id"].unique())
		res = {}
		for label, df in (("all", A), ("reachable", A[A["reachable"]]), ("fresh", A[~A["child_from_cache"]])):
			for value in ("z", "backup_value"):
				blk, cells = affpool_block(df, "bucket_med", value, boot)
				res[f"{label}|{value}"] = blk
				if label == "all" and value == "backup_value":
					h = cells[cells["n_prefixes"] >= 3]
					res["all|backup_value"]["share_cells_ratio_gt1"] = float((h["ratio"] > 1).mean())
					res["all|backup_value"]["share_cells_nc_gt1"] = float((h["ratio_noise_corrected"] > 1).mean())
					res["all|backup_value"]["within_var_median"] = float(h["within_var"].median())
					res["all|backup_value"]["between_var_median"] = float(h["between_var"].median())
				if label == "all" and value == "z":
					h = cells[cells["n_prefixes"] >= 3]
					res["all|z"]["within_var_median"] = float(h["within_var"].median())
					res["all|z"]["between_var_median"] = float(h["between_var"].median())
		y = YESTERDAY[run]
		got = res["all|z"]
		assert abs(got["between_within_ratio_wmedian"]["value"] - y["ratio"]) < 0.0015, got
		assert abs(got["evidence_multiplier_median"]["value"] - y["mult"]) < 0.006, got
		assert abs(got["between_within_ratio_noise_corrected_wmedian"]["value"] - y["nc"]) < 0.006, got
		out[run] = res
		print(run, json.dumps({k: (v["between_within_ratio_wmedian"], v["between_within_ratio_noise_corrected_wmedian"])
							   for k, v in res.items()}, default=float))
	update_wednesday("s5_1_affpool_gate_recheck", out)
	jdump(out, os.path.join(WED, "s51_affpool_gate_recheck.json"))


if __name__ == "__main__":
	main()
