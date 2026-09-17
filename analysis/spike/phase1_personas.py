"""Phase 1 - load, validate, and map P4G persuadee psychological profiles."""
import json, pickle, sys
import pandas as pd, numpy as np

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
BIG5   = ["extrovert","agreeable","conscientious","neurotic","open"]
MORAL  = ["care","fairness","loyalty","authority","purity","freedom"]
SCHWARTZ = ["conform","tradition","benevolence","universalism","self_direction",
            "stimulation","hedonism","achievement","power","security"]
DECISION = ["rational","intuitive"]
DIMS = BIG5 + MORAL + SCHWARTZ + DECISION
DEMO = ["age","sex","race","edu","marital","employment"]
BAD = {"20180808-024552_152_live","20180723-100140_767_live","20180825-080802_964_live"}

info = pd.read_csv(f"{ROOT}/data/p4g_personas/full_info.csv")
print(f"full_info.csv rows={len(info)}  cols={len(info.columns)}")

# persuadee rows only (B4: 0=persuader, 1=persuadee)
ee = info[info["B4"] == 1].copy()
print(f"persuadee rows={len(ee)}  unique dialogues={ee['B2'].nunique()}")

missing_cols = [c for c in DIMS + DEMO if f"{c}.x" not in ee.columns]
if missing_cols:
    sys.exit(f"FATAL: columns absent from CSV: {missing_cols}")
print(f"all {len(DIMS)} trait cols + {len(DEMO)} demo cols present  "
      f"({len(BIG5)}+{len(MORAL)}+{len(SCHWARTZ)}+{len(DECISION)}={len(DIMS)})")

ee = ee.drop_duplicates(subset="B2", keep="first").set_index("B2")

def build(row):
    p = {}
    for d in DIMS:
        v = row[f"{d}.x"]
        if pd.isna(v): return None
        p[d] = float(v)
    for d in DEMO:
        v = row[f"{d}.x"]
        p[d] = None if pd.isna(v) else (int(v) if d == "age" else str(v).strip())
    if p["age"] is None: return None
    p["user_id"] = str(row["B3"])
    return p

corpus = {did: p for did, row in ee.iterrows() if (p := build(row)) is not None}
print(f"\nCORPUS: complete profiles = {len(corpus)} / {len(ee)} persuadee dialogues")

# ---- range validation over the FULL corpus -----------------------------------
vals = np.array([[p[d] for d in DIMS] for p in corpus.values()])
lo_all, hi_all = vals.min(), vals.max()
print(f"trait value range across corpus: [{lo_all:.3f}, {hi_all:.3f}]  (expected within [1,6])")
bad = [(d, vals[:, i].min(), vals[:, i].max()) for i, d in enumerate(DIMS)
       if vals[:, i].min() < 1 or vals[:, i].max() > 6]
print("OUT-OF-RANGE dims:", bad if bad else "none - all 23 dims within 1-6")

# ---- tertile boundaries over the FULL corpus, not the 100 --------------------
bounds = {d: (float(np.quantile(vals[:, i], 1/3)), float(np.quantile(vals[:, i], 2/3)))
          for i, d in enumerate(DIMS)}
print("\ntertile bounds (lo=33rd pct, hi=67th pct) computed over "
      f"{len(corpus)} corpus profiles:")
for d in DIMS: print(f"  {d:<15} lo={bounds[d][0]:.3f}  hi={bounds[d][1]:.3f}")

# ---- map the 100 evaluation dialogues ----------------------------------------
allд = pickle.load(open(f"{ROOT}/data/p4g/300_dialog_turn_based.pkl", "rb"))
eval_ids = [d for d in allд.keys() if d not in BAD][:100]
print(f"\nevaluation dialogues selected: {len(eval_ids)} "
      f"(pkl order, minus {len(BAD)} filtered bad dialogues)")

have = [d for d in eval_ids if d in corpus]
miss = [d for d in eval_ids if d not in corpus]
print(f"WITH complete profile: {len(have)}   WITHOUT (excluded): {len(miss)}")
if miss: print("  excluded ids:", miss)

if len(have) < 80:
    sys.exit(f"\n*** STOP: only {len(have)}/100 dialogues have complete profiles (<80). ***")
print(f"\nGATE PASSED: {len(have)}/100 >= 80")

personas = [{"persona_id": i, "dialogue_id": d, **corpus[d]} for i, d in enumerate(have)]
json.dump({"dims": DIMS, "demo": DEMO, "bounds": bounds,
           "n_corpus_profiles": len(corpus), "excluded_dialogues": miss,
           "personas": personas},
          open(f"{ROOT}/spike/personas.json", "w"), indent=1)

print("\n=== 3 FULL PROFILES ===")
for p in personas[:3]:
    print(f"\n[{p['persona_id']}] {p['dialogue_id']}  user={p['user_id']}")
    print("   " + " ".join(f"{d}={p[d]:.2f}" for d in DIMS))
    print("   demo: " + ", ".join(f"{d}={p[d]}" for d in DEMO))
print(f"\nwrote spike/personas.json  ({len(personas)} personas)")
