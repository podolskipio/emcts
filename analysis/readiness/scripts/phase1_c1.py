"""Readiness Phase 1A / 1B (C1a, C1b) -- PREREG Entry 7.

    python analysis/readiness/scripts/phase1_c1.py collect   # LLM calls -> data/c1_rows.parquet
    python analysis/readiness/scripts/phase1_c1.py analyse   # -> phase1_c1.json

One pre-decision prefix per annotated dialogue (every turn strictly before the first user decision act,
ending on a persuadee turn). Per prefix:
  v        the value estimator as today: P4GChatSystemPlanner.heuristic (10 samples, T 1.1, persona)
  v_logit  the same prompt, logit-scored (deterministic) -- robustness only
  r        the affective-risk judgement below, E[rating]/10 off the logits over "0".."10"
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402
import _stats as S  # noqa: E402

from utils.rewards import reward_dict  # noqa: E402

OUT_ROWS = os.path.join(E.READINESS, "data", "c1_rows.parquet")
OUT_JSON = os.path.join(E.READINESS, "phase1_c1.json")
RATINGS = [str(i) for i in range(11)]

R_SYSTEM = (
	"You are an expert observer of persuasion conversations. Below is a conversation in which a Persuader "
	"is trying to convince a Persuadee to donate to a charity called Save the Children. Read it and judge "
	"the Persuadee's emotional state."
)
R_QUESTION = (
	"On a scale from 0 to 10, how likely is it that this conversation is already lost for EMOTIONAL "
	"reasons -- the Persuadee is annoyed, irritated, shutting down or disengaging -- regardless of whether "
	"they have said yes or no to donating? 0 means not at all likely, 10 means certain. "
	"Answer in exactly this format: Rating: [N]"
)


def r_messages(state):
	transcript = "\n".join(f"{role}: {utt}" for role, _da, utt in state)
	return [{"role": "system", "content": R_SYSTEM},
			{"role": "user", "content": f"{transcript}\n\n{R_QUESTION}"}]


def score_r(state, top_logprobs_num=0):
	s = E.backbone().score_labels(r_messages(state), labels=RATINGS, prefill="Rating: [", close="]",
								  top_logprobs_num=top_logprobs_num)
	return float(s.expectation({k: int(k) / 10 for k in RATINGS})), s


def prefixes():
	import t2_construct_test as CT
	dec = CT.decision_turns()
	u = pd.read_parquet(os.path.join(E.REPO, "analysis", "thu", "user_turns.parquet"))
	donated = u.groupby("dialogue_id")["donated"].first().to_dict()
	rows = []
	for did, turns in E.annotated_dialogs().items():
		if did not in donated:
			continue
		cut = dec.get(did, len(turns))
		pre = turns[:cut]
		if not pre:
			continue
		rows.append({"dialogue_id": did, "turns": pre, "n_turns": len(pre), "donated": int(donated[did])})
	return rows


def collect():
	personas = E.personas()
	rows = prefixes()
	print(f"{len(rows)} pre-decision prefixes")

	def one(row):
		state = E.session(row["turns"])
		_game, _sys, _usr, planner = E.agents(personas.get(row["dialogue_id"]))
		v, das = planner.heuristic(state)
		v_logit = float(planner.score_value_labels(state).expectation(reward_dict["p4g"]))
		r, rs = score_r(state, top_logprobs_num=20)
		return {"dialogue_id": row["dialogue_id"], "n_turns": row["n_turns"], "donated": row["donated"],
				"has_persona": row["dialogue_id"] in personas, "v": v, "v_samples": json.dumps(das),
				"v_logit": v_logit, "r": r, "r_probs": json.dumps(dict(zip(RATINGS, map(float, rs.probs)))),
				"r_label_set_mass": rs.label_set_mass}

	with ThreadPoolExecutor(16) as ex:
		out = list(tqdm(ex.map(one, rows), total=len(rows)))
	os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)
	pd.DataFrame(out).to_parquet(OUT_ROWS)
	print("wrote", OUT_ROWS)


def analyse():
	df = pd.read_parquet(OUT_ROWS)
	res = {
		"n_prefixes": int(len(df)), "donation_rate": float(df.donated.mean()),
		"personas_found": int(df.has_persona.sum()),
		"r_summary": {"mean": float(df.r.mean()), "sd": float(df.r.std()), "min": float(df.r.min()),
					  "max": float(df.r.max()), "label_set_mass_median": float(df.r_label_set_mass.median())},
		"v_summary": {"mean": float(df.v.mean()), "sd": float(df.v.std())},
		"corr_r_v": S.corr_ci(df.r, df.v),
		"corr_r_v_logit": S.corr_ci(df.r, df.v_logit),
		"corr_v_v_logit": S.corr_ci(df.v, df.v_logit),
		"C1a_r_over_v": S.incremental_r2(df, ["v"], ["r"]),
		"C1a_r_over_v_logit": S.incremental_r2(df, ["v_logit"], ["r"]),
		"C1b_r_over_n_turns": S.incremental_r2(df, ["n_turns"], ["r"]),
		"reference_v_over_n_turns": S.incremental_r2(df, ["n_turns"], ["v"]),
	}
	c1a, c1b = res["C1a_r_over_v"]["PASS"], res["C1b_r_over_n_turns"]["PASS"]
	res["r_is_1_minus_v"] = bool(res["corr_r_v"]["value"] <= -0.7 and not c1a)
	res["C1_PASS"] = bool(c1a or c1b)
	res["git_head"] = E.git_head()
	with open(OUT_JSON, "w") as f:
		json.dump(res, f, indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	{"collect": collect, "analyse": analyse}[sys.argv[1]]()
