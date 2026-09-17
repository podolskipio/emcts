"""Saturation: how much of SR = 1.0 is a lenient success detector, and do stricter criteria de-saturate?

    python analysis/thu/scripts/t10_success_criteria.py <run_dir>:<tag> ... [--out JSON] [--dump TSV]

The environment ends a p4g episode with success when the user-act classifier labels a persuadee turn
`donate`. This re-scores the LOGGED episodes (no rerun) under stricter, transparent criteria applied to
the persuadee's final turn and the one before it (the `donate` turn and its lead-in):

  lenient        the environment's label (what SR reports today)
  committed      lenient AND no deferral/hedge language in the donate turn
                 ("consider", "think about", "will be in touch", "later", "maybe", "might", "not sure")
  amount         committed AND an explicit amount is named (e.g. "$5", "5 dollars", "fifty cents")
  amount_le_2    amount AND every named amount is <= $2, the P4G task's real donation ceiling

Regexes live in src/games/p4g_success.py, shared with the environment (--p4g_success), and are copied into the output. They were written against the 10 P1_episode
dialogues only (pilot set 131-140), before the n = 100 run was read.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402

from games.p4g_success import AMOUNT, DEFERRAL, FIRM, HEDGE_GOVERNS, amounts, hedge_match  # noqa: E402  (the environment uses the same module)


def score(ep):
	user = [h for h in ep["history"] if h[0] != "Persuader"]
	donate_turn = user[-1][-1] if user else ""
	lead_in = user[-2][-1] if len(user) > 1 else ""
	lenient = bool(ep["success"])
	hm = hedge_match(donate_turn)
	hedged = hm is not None
	amt = amounts(donate_turn) or amounts(lead_in)
	committed = lenient and not hedged
	return {"did": ep["did"], "turns": ep["num_turns"], "lenient": lenient, "committed": committed,
			"amount": committed and bool(amt), "amount_le_2": committed and bool(amt) and max(amt) <= 2.0,
			"amounts": amt, "hedge_match": hm,
			"donate_turn": donate_turn}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("runs", nargs="+")
	ap.add_argument("--out", default=None)
	ap.add_argument("--dump", default=None)
	a = ap.parse_args()
	out, dump = {"regex": {"hedge_governs": HEDGE_GOVERNS.pattern, "deferral": DEFERRAL.pattern, "firm": FIRM.pattern,
					  "amount": AMOUNT.pattern}, "runs": {}}, []
	for spec in a.runs:
		run, tag = spec.split(":")
		eps = pickle.load(open(glob.glob(os.path.join(run, "*.pkl"))[0], "rb"))
		rows = [score(e) for e in eps]
		n = len(rows)
		res = {"n": n}
		for crit in ("lenient", "committed", "amount", "amount_le_2"):
			k = sum(r[crit] for r in rows)
			res[crit] = {"SR": k / n, "k": k, "wilson95": lib.wilson(k, n)}
		succ_amt = [max(r["amounts"]) for r in rows if r["lenient"] and r["amounts"]]
		res["amounts_named_among_lenient_successes"] = {
			"n": len(succ_amt), "median": float(np.median(succ_amt)) if succ_amt else None,
			"share_le_2": float(np.mean([x <= 2 for x in succ_amt])) if succ_amt else None,
			"values": sorted(succ_amt)}
		res["lenient_but_hedged"] = [{"did": r["did"], "match": r["hedge_match"], "text": r["donate_turn"][:200]}
									  for r in rows if r["lenient"] and not r["committed"]]
		out["runs"][tag] = res
		print(f"{tag:12s} n {n:3d}  " + "  ".join(
			f"{c} {res[c]['SR']:.2f} [{res[c]['wilson95'][0]:.2f},{res[c]['wilson95'][1]:.2f}]"
			for c in ("lenient", "committed", "amount", "amount_le_2"))
			+ f"  | amounts median {res['amounts_named_among_lenient_successes']['median']}")
		dump += [(tag, r["did"], r["lenient"], r["committed"], r["amount"], r["amount_le_2"], r["amounts"],
				  r["donate_turn"].replace("\t", " ").replace("\n", " ")) for r in rows]
	if a.out:
		lib.jdump(out, a.out)
	if a.dump:
		with open(a.dump, "w") as f:
			f.write("run\tdid\tlenient\tcommitted\tamount\tamount_le_2\tamounts\tdonate_turn\n")
			for row in dump:
				f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
	main()
