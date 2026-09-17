"""Saturation-check set: positions 141-240 of the NON-annotated P4G pool (1-based).

Same pool and order as build_p4g_rollout_evalset.py / analysis/wed/scripts/build_sweep_set.py. Disjoint by
construction from the eval set (1-100), D1-D3 (101-130) and the pilot set (131-140); asserted. Used to
estimate NoEmo SR under --search_horizon episode at n = 100 WITHOUT reading the eval set, because the
number decides the evaluation design (success criterion / primary budget), which must not be tuned on eval.

    python analysis/thu/scripts/build_saturation_set.py
"""
import json, os, pickle, sys
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)
from utils.p4g_personas import load_profiles

BAD = {"20180808-024552_152_live", "20180723-100140_767_live", "20180825-080802_964_live"}
annotated = set(pickle.load(open("data/p4g/300_dialog_turn_based.pkl", "rb")).keys())
dialog = pd.read_csv("data/p4g_personas/full_dialog.csv")
order = list(dict.fromkeys(dialog["B2"].tolist()))
pool = [d for d in order if d not in annotated and d not in BAD]
eval_ids = [json.loads(l)["id"] for l in open("data/p4g/rollout_evalset_nonannotated.jsonl")]
assert eval_ids == pool[:100], "eval set is not pool[:100]; ordering assumption broken"
chosen = pool[140:240]
assert len(chosen) == 100
d13 = [json.loads(l)["id"] for l in open("analysis/phase1/runs/dialogues_101_130.jsonl")]
sweep = [json.loads(l)["id"] for l in open("analysis/wed/runs/sweep_dialogues_131_140.jsonl")]
for other, name in ((eval_ids, "eval"), (annotated, "annotated"), (d13, "D1-D3"), (sweep, "pilot 131-140")):
	assert not set(chosen) & set(other), f"overlaps {name}"
profiles = load_profiles()
missing = [d for d in chosen if d not in profiles]
assert not missing, f"no persona for {missing}"

by_dialog = {d: g for d, g in dialog.groupby("B2", sort=False)}
with open("analysis/thu/runs/saturation_dialogues_141_240.jsonl", "w") as f:
	for did in chosen:
		g = by_dialog[did].sort_values(["Turn", "B4"])
		turns = [{"speaker": "sys" if int(r.B4) == 0 else "usr", "text": str(r.Unit).strip()}
				 for r in g.itertuples() if str(r.Unit).strip()]
		f.write(json.dumps({"id": did, "dialog": turns}) + "\n")
print(f"wrote {len(chosen)} dialogues (pool size {len(pool)}); first={chosen[0]} last={chosen[-1]}")
