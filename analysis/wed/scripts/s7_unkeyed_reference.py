"""§7.4 reference: the UNKEYED pool (every step under the same act in the tree, weight 1) through the
same per-query code as hard K0 and the kernels. h -> infinity is this pool; if it is as homogeneous as
the keyed ones, the affective key is not what makes pooling safe.

    python analysis/wed/scripts/s7_unkeyed_reference.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, load_steps, update_wednesday, jdump  # noqa: E402
from s7_key_geometry import per_query, summarise_queries  # noqa: E402

out = {}
allA = load_steps()
for run in ("D1", "D2"):
	A = allA[allA["run_id"] == run].reset_index(drop=True)
	A["edge_N"] = A["edge_id"].map(A.groupby("edge_id").size())
	boot = Boot(A["dialogue_id"].unique())
	out[run] = {v: summarise_queries(per_query(A, v, lambda g: np.ones((len(g), len(g)))), boot) for v in ("backup_value", "z")}
	for v in out[run]:
		print(run, v, {k: (round(x["value"], 3), [round(c, 3) for c in x["ci"]]) for k, x in out[run][v].items() if isinstance(x, dict) and "value" in x})
jdump(out, os.path.join(WED, "s7_unkeyed_reference.json"))
update_wednesday("s7_4_unkeyed_reference", out)
