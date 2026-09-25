"""Checks on "v does not predict human donation" (phase1.md §1A), before believing it.

    python analysis/readiness/scripts/fork_v_human.py collect   # LLM calls -> data/v_fixed_turn_rows.parquet
    python analysis/readiness/scripts/fork_v_human.py analyse   # -> fork_v_human.json

1. Interval on the AUC. Direct AUC of v, v_logit, r and n_turns on the C1 prefixes (phase1_c1's rows,
   no new calls), 1000-replicate bootstrap over dialogues.
2. Turn-count leakage. C1 cut every dialogue at its decision point, so n_turns there IS the turn the
   decision came at. Here the prefix is cut at a FIXED turn k = 1..6 (persuadee turn k, 1-based), among
   dialogues whose first decision act comes after it -- no stated decision is in any prefix, and turn
   count is constant within k. v is the estimator as today (10 samples, T 1.1, persona) plus v_logit.
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
import phase1_c1 as C1  # noqa: E402
from fork_v_sim import auc  # noqa: E402

from utils.rewards import reward_dict  # noqa: E402

K_MAX = 6
OUT_ROWS = os.path.join(E.READINESS, "data", "v_fixed_turn_rows.parquet")
OUT_JSON = os.path.join(E.READINESS, "fork_v_human.json")


def boot_auc(df, x, seed, sign=1):
	rng = np.random.default_rng(seed)
	y, v = df["donated"].to_numpy(), sign * df[x].to_numpy(float)
	reps = [auc(y[i], v[i]) for i in (rng.integers(0, len(df), len(df)) for _ in range(1000))]
	return {"value": auc(y, v), "ci": [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]}


def collect():
	import t2_construct_test as CT
	dec = CT.decision_turns()
	u = pd.read_parquet(os.path.join(E.REPO, "analysis", "thu", "user_turns.parquet"))
	donated = u.groupby("dialogue_id")["donated"].first().to_dict()
	personas = E.personas()
	jobs = []
	for did, turns in E.annotated_dialogs().items():
		if did not in donated:
			continue
		pre = dec.get(did, len(turns))  # number of turns strictly before the first decision act
		for k in range(1, min(pre, K_MAX) + 1):
			jobs.append({"dialogue_id": did, "k": k, "turns": turns[:k], "donated": int(donated[did]),
						 "decision_turn": pre})

	def one(job):
		state = E.session(job["turns"])
		_g, _s, _u, planner = E.agents(personas.get(job["dialogue_id"]))
		v, _ = planner.heuristic(state)
		v_logit = float(planner.score_value_labels(state).expectation(reward_dict["p4g"]))
		return {k: job[k] for k in ("dialogue_id", "k", "donated", "decision_turn")} | {"v": v, "v_logit": v_logit}

	with ThreadPoolExecutor(16) as ex:
		rows = list(tqdm(ex.map(one, jobs), total=len(jobs)))
	pd.DataFrame(rows).to_parquet(OUT_ROWS)
	print("wrote", OUT_ROWS, len(rows))


def analyse():
	c1 = pd.read_parquet(C1.OUT_ROWS)
	res = {"c1_prefixes_auc": {
		"n": int(len(c1)),
		"v": boot_auc(c1, "v", 1), "v_logit": boot_auc(c1, "v_logit", 2),
		"r (higher risk -> less donation)": boot_auc(c1, "r", 3, sign=-1),
		"n_turns (shorter -> more donation)": boot_auc(c1, "n_turns", 4, sign=-1),
	}}
	df = pd.read_parquet(OUT_ROWS)
	res["fixed_turn"] = {}
	for k, d in df.groupby("k"):
		d = d.reset_index(drop=True)
		res["fixed_turn"][int(k)] = {
			"n": int(len(d)), "donation_rate": float(d.donated.mean()),
			"v_mean": float(d.v.mean()), "v_sd": float(d.v.std()),
			"auc_v": boot_auc(d, "v", 10 + k), "auc_v_logit": boot_auc(d, "v_logit", 20 + k),
			"mcfadden_v": float(S.T6.mcfadden(d, ["v"])[0]),
		}
	# all fixed-turn rows pooled, dialogue-clustered: several rows per dialogue here
	rng = np.random.default_rng(99)
	ids = df.dialogue_id.unique()
	by = {i: g for i, g in df.groupby("dialogue_id")}
	reps = []
	for _ in range(1000):
		s = pd.concat([by[i] for i in rng.choice(ids, len(ids))])
		reps.append(auc(s.donated.to_numpy(), s.v.to_numpy()))
	res["fixed_turn_pooled_auc_v"] = {"value": auc(df.donated.to_numpy(), df.v.to_numpy()),
									  "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
									  "n_rows": int(len(df)), "n_dialogues": int(len(ids))}
	res["git_head"] = E.git_head()
	json.dump(res, open(OUT_JSON, "w"), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == "__main__":
	{"collect": collect, "analyse": analyse}[sys.argv[1]]()
