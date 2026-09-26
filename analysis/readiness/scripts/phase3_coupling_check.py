"""Readiness Phase 3A acceptance (PREREG Entry 10) on the played dialogues.

    python analysis/readiness/scripts/phase3_coupling_check.py

(i)  P3A_NoEmo_a vs P3A_NoEmo_b (same arm, seed, store): every dialogue identical, turn for turn
     (speaker, act, utterance); the re-run should be served entirely from the store.
(ii) P3A_NoEmo_a vs P3A_ActPool (same seed, store): identical up to the first turn whose system act
     differs. Also reported: where that first divergence falls.
-> phase3_coupling.json
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402,F401  (src/ on the path for the pickles)

RUNS = os.path.join(E.READINESS, "runs")
OUT = os.path.join(E.READINESS, "phase3_coupling.json")


def episodes(tag):
	eps = pickle.load(open(os.path.join(RUNS, tag, tag, tag + ".pkl"), "rb"))
	return {ep["did"]: ep for ep in eps}


def turns(ep):
	return [(role, da, str(emo), utt) for role, da, emo, utt in ep["history"]]


def compare_identical(a, b):
	rows = []
	for did in sorted(a):
		ta, tb = turns(a[did]), turns(b[did])
		first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
		if first is None and len(ta) != len(tb):
			first = min(len(ta), len(tb))
		rows.append({"dlg_id": did, "identical": first is None, "first_difference_at_entry": first,
					 "entries": [len(ta), len(tb)], "coupling_b": b[did].get("coupling")})
	return rows


def compare_until_divergence(a, b):
	rows = []
	for did in sorted(a):
		ta, tb = turns(a[did]), turns(b[did])
		sys_idx = [i for i, t in enumerate(ta) if i < len(tb) and t[0] == "Persuader"]
		act_div = next((i for i in sys_idx if ta[i][1] != tb[i][1]), None)
		limit = act_div if act_div is not None else min(len(ta), len(tb))
		mismatch = next((i for i in range(limit) if ta[i] != tb[i]), None)
		rows.append({"dlg_id": did, "first_act_divergence_entry": act_div,
					 "first_act_divergence_turn": None if act_div is None else act_div // 2,
					 "identical_before_divergence": mismatch is None, "first_mismatch_before_divergence": mismatch,
					 "turns": [len(ta) // 2, len(tb) // 2]})
	return rows


def main():
	a, b, act = episodes("P3A_NoEmo_a"), episodes("P3A_NoEmo_b"), episodes("P3A_ActPool")
	i, ii = compare_identical(a, b), compare_until_divergence(a, act)
	res = {"i_same_arm_rerun": i, "ii_noemo_vs_actpool": ii,
		   "i_PASS": all(r["identical"] for r in i),
		   "ii_PASS": all(r["identical_before_divergence"] for r in ii),
		   "store_a": {d: e.get("coupling") for d, e in a.items()},
		   "git_head": E.git_head()}
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
