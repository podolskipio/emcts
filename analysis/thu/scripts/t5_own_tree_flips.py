"""Own-tree flip rate of each live arm against NoEmo's argmax at the same selection point.

    python analysis/thu/scripts/t5_own_tree_flips.py <run_dir>:<tag> ...

At every logged selection point the arm chose argmax(logged uct). The NoEmo choice at that point is
argmax(Q + cpuct*P*sqrt(Ns)/(1+N)), rebuilt from the logged sibling Q, prior, N and parent_Ns
(cpuct 1.0; Ns = 0 -> 1e-8, as in _calculate_uct). On a NoEmo run the two must coincide exactly: that
is the validation. Unlike t5_dose_replay.py (identical NoEmo trees for every arm, the DOSE), this
includes each arm's effect on the tree it grew.
"""
import gzip, glob, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wed", "scripts"))
import lib  # noqa: E402

out = {}
for spec in sys.argv[1:]:
	run, tag = spec.split(":")
	rows, max_uct_dev = [], 0.0
	for p in sorted(glob.glob(os.path.join(run, "simlog", "*.ndjson.gz"))):
		for line in gzip.open(p, "rt"):
			s = json.loads(line)
			if s.get("record_type") != "step":
				continue
			Ns = s["parent_Ns"] or 1e-8
			sib = s["siblings"]
			base = [sb["Q"] + sb["prior"] * math.sqrt(Ns) / (1 + sb["N"]) for sb in sib]
			uct = [sb["uct"] for sb in sib]
			i0, i1 = int(np.argmax(base)), int(np.argmax(uct))
			assert sib[i1]["action"] == s["action"] or "constraint_masked" in s, (tag, s["action"], sib[i1]["action"])
			rows.append({"dlg": s["dlg_id"], "depth": s["depth"], "flip": i0 != i1,
						 "to_exp": sib[i1]["N"] > 0, "from_exp": sib[i0]["N"] > 0,
						 "n_exp": sum(sb["N"] > 0 for sb in sib)})
			max_uct_dev = max(max_uct_dev, max(abs(a - b) for a, b in zip(base, uct)))
	d = pd.DataFrame(rows)
	f = d[d.flip]
	by = {k: v.flip.to_numpy() for k, v in d.groupby("dlg")}
	reps = lib.Boot(list(by)).run(lambda ids: float(np.concatenate([by[i] for i in ids]).mean()))
	out[tag] = {"points": int(len(d)), "flip_rate": lib.with_ci(float(d.flip.mean()), reps),
				"to_visited_over_unvisited": float((f.to_exp & ~f.from_exp).mean()) if len(f) else None,
				"to_unexpanded": float((~f.to_exp).mean()) if len(f) else None,
				"between_visited": float((f.to_exp & f.from_exp).mean()) if len(f) else None,
				"max_abs_uct_minus_noemo_score": max_uct_dev}
	print(tag, json.dumps(out[tag]))
lib.jdump(out, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "own_tree_flips.json"))
