"""Phase-1 diagnostic dialogue set: dialogues 101-130 of the NON-annotated PersuasionForGood pool.

Same pool and order as src/emotion_mining/build_p4g_rollout_evalset.py (corpus file order,
annotated 300 and the 3 bad ids removed), whose first 100 are the evaluation set. Taking
pool[100:130] makes this set disjoint from the eval set by construction; the script asserts it
against data/p4g/rollout_evalset_nonannotated.jsonl and checks every id has a persona row.

    python analysis/phase1/scripts/build_dialogue_set.py
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
chosen = pool[100:130]
assert not set(chosen) & set(eval_ids) and not set(chosen) & annotated
profiles = load_profiles()
missing = [d for d in chosen if d not in profiles]
assert not missing, f"no persona for {missing}"

out_dir = "analysis/phase1/runs"
by_dialog = {d: g for d, g in dialog.groupby("B2", sort=False)}
with open(f"{out_dir}/dialogues_101_130.jsonl", "w") as f:
    for did in chosen:
        g = by_dialog[did].sort_values(["Turn", "B4"])
        turns = [{"speaker": "sys" if int(r.B4) == 0 else "usr", "text": str(r.Unit).strip()}
                 for r in g.itertuples() if str(r.Unit).strip()]
        f.write(json.dumps({"id": did, "dialog": turns}) + "\n")
with open(f"{out_dir}/dialogue_ids.txt", "w") as f:
    f.write("# Phase-1 diagnostic set: positions 101-130 (1-based) of the non-annotated p4g pool.\n")
    f.write("# Eval set = positions 1-100 (data/p4g/rollout_evalset_nonannotated.jsonl). Disjoint, asserted.\n")
    for i, did in enumerate(chosen, start=101):
        f.write(f"{i}\t{did}\n")
print(f"wrote {len(chosen)} dialogues; first={chosen[0]} last={chosen[-1]}")
