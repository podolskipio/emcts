"""Build the p4g self-play rollout eval set from the NON-annotated half of PersuasionForGood.

Leakage insurance (P2). ``w(e)`` is mined from the 300 annotated dialogs. If the rollout eval
set is also drawn from those 300, the weights have seen the eval dialogs — even though the
p4g self-play runner replays no corpus text (p4g dialogs carry ``scenario=()``, so
``game.init_dialog()`` starts empty and only the dialog id is used as an episode label).
Rather than argue about whether that constitutes leakage, we make the two sets disjoint by
construction: mine on all 300 annotated, evaluate on the first 100 of the 717 dialogs that
carry no dialog-act annotation at all.

This costs nothing on either side. Mining keeps the full 300 (holding out 100 instead would
move w(fear) by ~1.0 -- the thin cells are unstable, see the mining module docstring), and
the eval set loses nothing it was using, because self-play never reads the turns.

Output is the JSON-lines format ``runners/_common._read_p4g_jsonl`` already accepts, so the
runners take it with ``--data``. Dialog-act fields are absent by construction (these dialogs
are unannotated); the reader maps missing acts to ``other`` / ``U_Neutral``, which is
harmless for self-play since the turns are never replayed.

Usage
-----
    python src/emotion_mining/build_p4g_rollout_evalset.py
    python src/emotion_mining/build_p4g_rollout_evalset.py --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from utils.p4g_personas import PSYCH_TRAITS, load_profiles

ANNOTATED_PKL = "data/p4g/300_dialog_turn_based.pkl"
FULL_DIALOG_CSV = "data/p4g_personas/full_dialog.csv"
FULL_INFO_CSV = "data/p4g_personas/full_info.csv"
P4G_BAD_DIALOGS = {"20180808-024552_152_live", "20180723-100140_767_live",
                   "20180825-080802_964_live"}


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n", type=int, default=100, help="dialogs in the eval set")
	ap.add_argument("--out", default="data/p4g/rollout_evalset_nonannotated.jsonl")
	ap.add_argument("--ids_out", default="data/p4g/mining_exclude_ids.txt",
	                help="ids written for --exclude_ids; empty when disjoint by construction, "
	                     "but written anyway so the mining run can assert the disjointness")
	args = ap.parse_args()
	os.chdir(REPO_ROOT)

	annotated = set(pickle.load(open(ANNOTATED_PKL, "rb")).keys())
	dialog = pd.read_csv(FULL_DIALOG_CSV)
	info = pd.read_csv(FULL_INFO_CSV)

	order = list(dict.fromkeys(dialog["B2"].tolist()))          # corpus file order
	pool = [d for d in order if d not in annotated and d not in P4G_BAD_DIALOGS]
	print(f"annotated: {len(annotated)}   full corpus: {len(order)}   "
	      f"non-annotated pool: {len(pool)}")
	chosen = pool[:args.n]
	if len(chosen) < args.n:
		sys.exit(f"FATAL: only {len(chosen)} non-annotated dialogs available, need {args.n}")

	# --- persona join (P4) -------------------------------------------------
	# --p4g_persona conditions the simulator on the participant who actually took part, so an
	# eval id with no survey row silently degrades to an unconditioned simulator for that
	# dialogue. Fail here instead: the eval set is only comparable across runs if all n
	# dialogues carry a persona. The per-trait counts say how much of the 23-dimension survey
	# each dialogue actually contributes -- a blank cell renders as no clause, so a trait far
	# below n is a trait the personas mostly do not mention.
	profiles = load_profiles()
	unresolved = [d for d in chosen if d not in profiles]
	if unresolved:
		sys.exit(f"FATAL: {len(unresolved)}/{len(chosen)} eval ids have no persuadee survey row "
		         f"in {FULL_INFO_CSV}: {unresolved[:5]}")
	print(f"persona join: {len(chosen)}/{len(chosen)} eval ids resolve to a persuadee profile")
	print(f"non-missing survey responses over the {len(PSYCH_TRAITS)} psychological dimensions "
	      f"(out of {len(chosen)}):")
	for trait in PSYCH_TRAITS:
		n_present = sum(1 for d in chosen if profiles[d].get(trait) is not None)
		print(f"  {trait:>15}: {n_present:3d}")

	# donation outcome, for reference only -- self-play does not read it
	ee = info[info["B4"] == 1].drop_duplicates(subset="B2").set_index("B2")
	by_dialog = {d: g for d, g in dialog.groupby("B2", sort=False)}

	n_turns = 0
	with open(args.out, "w") as f:
		for did in chosen:
			g = by_dialog[did].sort_values(["Turn", "B4"])
			turns = [{"speaker": "sys" if int(r.B4) == 0 else "usr",
			          "text": str(r.Unit).strip()}
			         for r in g.itertuples() if str(r.Unit).strip()]
			donated = None
			if did in ee.index and pd.notna(ee.loc[did, "B6"]):
				donated = float(ee.loc[did, "B6"]) > 0
			f.write(json.dumps({"id": did, "dialog": turns, "donated": donated}) + "\n")
			n_turns += len(turns)

	with open(args.ids_out, "w") as f:
		f.write("# p4g rollout eval-set dialog ids (non-annotated half).\n")
		f.write("# Disjoint from the 300 annotated mining dialogs by construction; passed to\n")
		f.write("# mine_emotion_donation_p4g.py --exclude_ids as an assertion, not a filter.\n")
		for did in chosen:
			f.write(did + "\n")

	overlap = set(chosen) & annotated
	print(f"wrote {len(chosen)} dialogs ({n_turns} turns) -> {args.out}")
	print(f"wrote {len(chosen)} ids -> {args.ids_out}")
	print(f"OVERLAP WITH MINING CORPUS: {len(overlap)}  "
	      f"{'OK - disjoint' if not overlap else 'FATAL: ' + str(overlap)}")
	if overlap:
		sys.exit(1)


if __name__ == "__main__":
	main()
