"""Table C (brief §2) -- corpus_turns.parquet, from the 300 ANNOTATED P4G dialogues. Not from any run.

    python analysis/wed/scripts/build_corpus_turns.py

Why the annotated 300, not the brief's 917: per-turn system acts exist only for the annotated
dialogues (data/p4g_personas/full_dialog.csv has no act column). Decision taken 2026-09-15: split
the 300 (minus the 3 content-filtered ids) by src/emotion_mining/mining_corpus.select_p4g_holdout
-- the first 100 are the momentum TEST set, the remaining 197 re-mine w(e)
(analysis/wed/corpus/w_mined_200.json) so the test's nu is not fitted on its own outcomes. The
annotated 300 are disjoint from the eval set (positions 1-100 of the NON-annotated pool).

One row per (system turn, distinct planner act in that turn). Columns:
  dialogue_id, turn_index, split {test100, mine200}
  system_act        a planner-inventory act present in the persuader turn (raw label hyphens ->
                    spaces); turns with none of them -> "other"
  n_acts_in_turn    distinct inventory acts in the turn (1 = single-act turn)
  user_nu           nu of the user turn immediately before this system turn (whole turn classified,
                    as the planner classifies a simulator reply); user_nu_prev the one before that
  delta_nu          user_nu - user_nu_prev (null when there is no earlier user turn)
  *_w200 / *_soft / *_generic   the same under the 200-mined table (primary), the deployed all-300
                    table (circular on outcomes), and the generic hand-signed table (not fitted)
  user_emotion_dist 7-vector (happiness, sadness, fear, anger, surprise, disgust, neutral)
  donated           dialogue-level: any user 'agree-donation' label
  first_agree_turn, pre_outcome (turn_index <= first agree turn, or the dialogue never agrees)
  is_last_system_act  the dialogue's last persuader turn
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "emotion_mining"))
os.chdir(REPO)

from mining_corpus import P4G_BAD_DIALOGS, read_p4g_sessions, select_p4g_holdout  # noqa: E402

EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
INVENTORY = ["credibility appeal", "emotion appeal", "proposition of donation", "logical appeal",
			 "task related inquiry", "greeting", "other"]
TABLES = {
	"w200": json.load(open("analysis/wed/corpus/w_mined_200.json"))["valence_weights"],
	"soft": {"happiness": 0.54, "fear": 0.24, "disgust": 0.13, "anger": 0.11, "surprise": 0.09, "neutral": -0.07,
			 "sadness": -0.14, "contempt": -0.60},
	"generic": {"happiness": 0.54, "surprise": 0.0, "neutral": 0.0, "sadness": -0.54, "fear": -0.54, "anger": -0.54,
				"disgust": -0.54, "contempt": -0.60},
}


def main():
	# cross-check the literal tables against the planner's own constants
	from mcts.emotion_mcts import EMOTION_VALENCE_TABLES
	for name in ("soft", "generic"):
		assert {str(e).lower(): w for e, w in EMOTION_VALENCE_TABLES[name].items()} == TABLES[name], name
	path = "data/p4g/300_dialog_turn_based.pkl"
	dialogs = pickle.load(open(path, "rb"))
	sessions = read_p4g_sessions(path)
	mine, test_ids = select_p4g_holdout(sessions, 100)
	test_ids = set(test_ids)
	eval_ids = {json.loads(l)["id"] for l in open("data/p4g/rollout_evalset_nonannotated.jsonl")}
	assert not eval_ids & set(dialogs), "annotated corpus overlaps the eval set"

	texts, recs = set(), []
	for did, d in dialogs.items():
		if did in P4G_BAD_DIALOGS:
			continue
		for i, turn in enumerate(d["dialog"]):
			ee = " ".join(turn.get("ee", [])).strip()
			if ee:
				texts.add(ee)
	from transformers import pipeline
	pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None,
					device=-1, truncation=True)
	texts = sorted(texts)
	dist = {}
	for k in range(0, len(texts), 64):
		batch = texts[k:k + 64]
		for t, scores in zip(batch, pipe(batch, batch_size=64)):
			m = {("happiness" if s["label"] == "joy" else s["label"]): s["score"] for s in scores}
			tot = sum(m.values())
			dist[t] = [m.get(e, 0.0) / tot for e in EMO7]
		print(f"classified {min(k + 64, len(texts))}/{len(texts)}", flush=True)

	def nu(vec, table):
		return float(sum(p * table[e] for p, e in zip(vec, EMO7)))

	for did, d in dialogs.items():
		if did in P4G_BAD_DIALOGS:
			continue
		turns = d["dialog"]
		donated = any("agree-donation" in lab.get("ee", []) for lab in d["label"])
		agree = [i for i, lab in enumerate(d["label"]) if "agree-donation" in lab.get("ee", [])]
		first_agree = agree[0] if agree else None
		sys_turns = [i for i, t in enumerate(turns) if " ".join(t.get("er", [])).strip()]
		last_sys = sys_turns[-1] if sys_turns else None
		user_hist = []  # (turn index, dist vector) of user turns so far
		for i, turn in enumerate(turns):
			er = " ".join(turn.get("er", [])).strip()
			if er and user_hist:
				acts = sorted({lab.replace("-", " ") for lab in d["label"][i].get("er", [])} & set(INVENTORY) - {"other"})
				acts = acts or ["other"]
				cur = user_hist[-1][1]
				prev = user_hist[-2][1] if len(user_hist) >= 2 else None
				base = {
					"dialogue_id": did, "turn_index": i, "split": "test100" if did in test_ids else "mine200",
					"n_acts_in_turn": len([a for a in acts if a != "other"]),
					"user_emotion_dist": cur, "user_label": EMO7[int(np.argmax(cur))],
					"donated": int(donated), "first_agree_turn": first_agree,
					"pre_outcome": bool(first_agree is None or i <= first_agree),
					"is_last_system_act": i == last_sys, "n_user_turns_before": len(user_hist),
				}
				for name, table in TABLES.items():
					base[f"user_nu_{name}"] = nu(cur, table)
					base[f"user_nu_prev_{name}"] = nu(prev, table) if prev is not None else np.nan
					base[f"delta_nu_{name}"] = base[f"user_nu_{name}"] - base[f"user_nu_prev_{name}"]
				for a in acts:
					recs.append({**base, "system_act": a})
			ee = " ".join(turn.get("ee", [])).strip()
			if ee:
				user_hist.append((i, dist[ee]))
	C = pd.DataFrame(recs)
	C["user_nu"], C["user_nu_prev"], C["delta_nu"] = C["user_nu_w200"], C["user_nu_prev_w200"], C["delta_nu_w200"]
	C.to_parquet("analysis/wed/corpus_turns.parquet", index=False)
	# corpus terciles go to constants.json, kept apart from Table A's
	cpath = "analysis/wed/constants.json"
	const = json.load(open(cpath)) if os.path.exists(cpath) else {}
	test = C[C["split"] == "test100"].drop_duplicates(["dialogue_id", "turn_index"])
	allc = C.drop_duplicates(["dialogue_id", "turn_index"])
	const["corpus"] = {
		"source": "300 annotated P4G dialogues minus 3 content-filtered; test = first 100 by select_p4g_holdout",
		"n_dialogues": int(C["dialogue_id"].nunique()), "n_test_dialogues": int(test["dialogue_id"].nunique()),
		"rows": len(C), "system_turns": len(allc),
		"corpus_terciles": {
			"test100_w200": [float(test["delta_nu_w200"].quantile(1 / 3)), float(test["delta_nu_w200"].quantile(2 / 3))],
			"all_generic": [float(allc["delta_nu_generic"].quantile(1 / 3)), float(allc["delta_nu_generic"].quantile(2 / 3))],
			"all_soft": [float(allc["delta_nu_soft"].quantile(1 / 3)), float(allc["delta_nu_soft"].quantile(2 / 3))],
		},
		"valence_tables": TABLES,
		"classifier": "j-hartmann/emotion-english-distilroberta-base, whole user turn",
	}
	json.dump(const, open(cpath, "w"), indent=1)
	print(C.groupby("split").agg(rows=("dialogue_id", "size"), dialogues=("dialogue_id", "nunique")))
	print(C["system_act"].value_counts())
	print(json.dumps(const["corpus"]["corpus_terciles"]))


if __name__ == "__main__":
	main()
