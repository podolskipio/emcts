"""TASK 2 step 1 -- per-user-turn table with TEXT, for the construct test.

analysis/wed/corpus_turns.parquet has nu and donated but no utterance text, and it is keyed by
SYSTEM turn. The resistance signal is lexical, so it needs the persuadee's words. This rebuilds the
user side from data/p4g/300_dialog_turn_based.pkl, re-classifies with the same DistilRoBERTa the
planner uses, and ASSERTS that the resulting nu reproduces corpus_turns.user_nu_w200 exactly -- so
the two constructs are compared on identical rows under identical nu.

Writes analysis/thu/user_turns.parquet.
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
THU = os.path.join(REPO, "analysis", "thu")
CACHE = os.path.join(THU, "user_turn_emotions.json")
# j-hartmann labels -> the project's 7 names (mining_corpus convention)
LABEL_MAP = {"joy": "happiness", "sadness": "sadness", "fear": "fear", "anger": "anger",
			 "surprise": "surprise", "disgust": "disgust", "neutral": "neutral"}


def classify(texts):
	if os.path.exists(CACHE):
		d = json.load(open(CACHE))
		if all(t in d for t in texts):
			return d
	from transformers import pipeline
	pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
					top_k=None, device=-1, truncation=True)
	out = {}
	for k in range(0, len(texts), 64):
		chunk = texts[k:k + 64]
		for t, scores in zip(chunk, pipe(chunk)):
			out[t] = {LABEL_MAP[s["label"]]: float(s["score"]) for s in scores}
		print(f"  classified {min(k + 64, len(texts))}/{len(texts)}", flush=True)
	json.dump(out, open(CACHE, "w"))
	return out


def main():
	w200 = json.load(open("analysis/wed/corpus/w_mined_200.json"))["valence_weights"]
	dialogs = pickle.load(open("data/p4g/300_dialog_turn_based.pkl", "rb"))
	sessions = read_p4g_sessions("data/p4g/300_dialog_turn_based.pkl")
	_, test_ids = select_p4g_holdout(sessions, 100)
	test_ids = set(test_ids)

	rows = []
	for did, d in dialogs.items():
		if did in P4G_BAD_DIALOGS:
			continue
		for i, turn in enumerate(d["dialog"]):
			ee = " ".join(turn.get("ee", [])).strip()
			if not ee:
				continue
			rows.append({"dialogue_id": did, "user_turn": i, "text": ee,
						 "split": "test100" if did in test_ids else "mine200"})
	df = pd.DataFrame(rows)

	dist = classify(sorted(df["text"].unique()))
	D = np.array([[dist[t].get(e, 0.0) for e in EMO7] for t in df["text"]])
	df[[f"p_{e}" for e in EMO7]] = D
	df["nu"] = D @ np.array([w200.get(e, 0.0) for e in EMO7])

	# dialogue-level donation label, taken from the table the rest of the project uses
	ct = pd.read_parquet("analysis/wed/corpus_turns.parquet")
	don = ct.groupby("dialogue_id")["donated"].max()
	df["donated"] = df["dialogue_id"].map(don)
	df = df[df["donated"].notna()].reset_index(drop=True)

	# --- verification: nu must reproduce corpus_turns.user_nu_w200 -------------------------------
	# corpus_turns row (dialogue, system turn_index t) carries the user turn t-1 immediately before.
	ref = ct[["dialogue_id", "turn_index", "user_nu_w200"]].drop_duplicates()
	ref = ref.assign(user_turn=ref["turn_index"] - 1)
	m = ref.merge(df[["dialogue_id", "user_turn", "nu"]], on=["dialogue_id", "user_turn"], how="inner")
	dev = (m["user_nu_w200"] - m["nu"]).abs()
	# float32 classifier scores round differently between the two passes; 1e-6 is that noise floor,
	# not a definitional difference (max deviation is printed so a real drift would show).
	bad = m[dev > 1e-5]
	print(f"nu cross-check: {len(m)} rows compared, max |deviation| = {dev.max():.2e}, "
		  f"{len(bad)} above 1e-5")
	assert len(bad) == 0, bad.head().to_string()

	df.to_parquet(os.path.join(THU, "user_turns.parquet"))
	print(df.groupby("split").agg(turns=("text", "size"), dialogues=("dialogue_id", "nunique")).to_string())
	print("donation rate by split:")
	print(df.groupby("split").apply(lambda g: g.groupby("dialogue_id")["donated"].first().mean()).to_string())


if __name__ == "__main__":
	main()
