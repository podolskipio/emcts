"""Readiness Phase 1D (A1) -- simulator temperature sweep and T* by the pre-registered rule (PREREG Entry 7).

    python analysis/readiness/scripts/phase1_temp.py collect   # LLM calls -> data/temp_rows.parquet
    python analysis/readiness/scripts/phase1_temp.py analyse   # -> phase1_temp.json

30 prefixes, one per annotated dialogue (seed 20260925), each cut after the persuader turn of a uniformly
random turn t in [1, n-1] so the real persuadee's reply to it exists. The simulator is PersuadeeChatModel
as build_agents makes it (persona ON), with only its temperature changed; 10 replies per prefix per T.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402

from games import PersuasionGame  # noqa: E402

TEMPS = [0.3, 0.5, 0.7, 0.9, 1.1]
N_PREFIX, N_SAMPLES, SEED = 30, 10, 20260925
NEG = {"sadness", "fear", "anger", "disgust"}
COVERAGE_TARGET = 9 / 11
OUT_ROWS = os.path.join(E.READINESS, "data", "temp_rows.parquet")
OUT_JSON = os.path.join(E.READINESS, "phase1_temp.json")


def prefixes():
	rng = np.random.default_rng(SEED)
	dialogs = E.annotated_dialogs()
	ids = sorted(did for did, t in dialogs.items() if len(t) >= 2)
	out = []
	for did in rng.choice(ids, N_PREFIX, replace=False):
		turns = dialogs[did]
		t = int(rng.integers(1, len(turns)))  # 1 .. n-1
		out.append({"dialogue_id": str(did), "t": t, "turns": turns[:t + 1],
					"human_utt": turns[t]["usr_utt"], "human_act": turns[t]["usr_da"]})
	return out


def collect():
	personas = E.personas()
	clf = E.hf_classifier()
	jobs = [(p, T) for p in prefixes() for T in TEMPS]

	def one(job):
		p, T = job
		state = E.session(p["turns"], end_on="sys")
		_g, _s, user, _pl = E.agents(personas.get(p["dialogue_id"]), user_temperature=T)
		with ThreadPoolExecutor(N_SAMPLES) as ex:
			replies = list(ex.map(lambda _: user.get_utterance_w_da(state), range(N_SAMPLES)))
		return [{"dialogue_id": p["dialogue_id"], "t": p["t"], "T": T, "sample": i, "act": act, "utt": utt}
				for i, (act, utt) in enumerate(replies)]

	with ThreadPoolExecutor(8) as ex:
		rows = [r for rs in tqdm(ex.map(one, jobs), total=len(jobs)) for r in rs]
	df = pd.DataFrame(rows)
	human = pd.DataFrame([{"dialogue_id": p["dialogue_id"], "t": p["t"], "T": -1.0, "sample": 0,
						   "act": p["human_act"], "utt": p["human_utt"]} for p in prefixes()])
	df = pd.concat([df, human], ignore_index=True)
	dists = [clf.predict_distribution_from_utterance(u) for u in df.utt]
	df["nu"] = [E.nu(d) for d in dists]
	df["argmax_emotion"] = [str(max(d, key=d.get)) for d in dists]
	os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)
	df.to_parquet(OUT_ROWS)
	print("wrote", OUT_ROWS, len(df))


def tokens(s):
	return re.findall(r"\w+|[^\w\s]", s.lower())


def distinct2(utts):
	bigrams = [bg for u in utts for bg in zip(tokens(u), tokens(u)[1:])]
	return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def human_targets():
	dialogs = E.annotated_dialogs()
	turns = [t for ts in dialogs.values() for t in ts]
	u = pd.read_parquet(os.path.join(E.REPO, "analysis", "thu", "user_turns.parquet"))
	pcols = [c for c in u.columns if c.startswith("p_")]
	argmax = u[pcols].to_numpy().argmax(1)
	neg = np.isin(np.array([c[2:] for c in pcols])[argmax], list(NEG))
	return {
		"donation_rate": float(np.mean([t["usr_da"] == PersuasionGame.U_Donate for t in turns])),
		"neg_affect_rate": float(neg.mean()),
		"length": float(np.mean([len(t["usr_utt"].split()) for t in turns])),
		"n_turns": len(turns),
	}


def analyse():
	df = pd.read_parquet(OUT_ROWS)
	sim, hum = df[df["T"] > 0], df[df["T"] < 0].set_index("dialogue_id")
	tgt = human_targets()
	tgt["distinct2"] = distinct2(hum.utt.tolist())
	tgt["coverage"] = COVERAGE_TARGET
	per_T, per_prefix = {}, []
	for T, g in sim.groupby("T"):
		cov, sd_nu, modal, collapse_str, collapse_act = [], [], [], [], []
		for did, gp in g.groupby("dialogue_id"):
			h = hum.loc[did, "nu"]
			lo, hi = gp.nu.min(), gp.nu.max()
			cov.append(lo <= h <= hi)
			sd_nu.append(gp.nu.std(ddof=1))
			vc = gp.act.value_counts()
			modal.append(vc.iloc[0] / len(gp))
			collapse_str.append(gp.utt.nunique() <= 2)
			collapse_act.append(len(vc) == 1)
			per_prefix.append({"T": T, "dialogue_id": did, "nu_sd": float(sd_nu[-1]), "modal_act_share": float(modal[-1]),
							   "covered": bool(cov[-1]), "distinct_strings": int(gp.utt.nunique()),
							   "donate_share": float((gp.act == PersuasionGame.U_Donate).mean())})
		d2 = np.mean([distinct2(gs.utt.tolist()) for _, gs in g.groupby("sample")])
		per_T[float(T)] = {
			"donation_rate": float((g.act == PersuasionGame.U_Donate).mean()),
			"neg_affect_rate": float(g.argmax_emotion.isin(NEG).mean()),
			"length": float(g.utt.str.split().str.len().mean()),
			"distinct2": float(d2),
			"coverage": float(np.mean(cov)),
			"nu_sd_mean": float(np.mean(sd_nu)),
			"modal_act_share_mean": float(np.mean(modal)),
			"collapse_le2_strings": float(np.mean(collapse_str)),
			"collapse_single_act": float(np.mean(collapse_act)),
		}
	crit = ["donation_rate", "neg_affect_rate", "length", "distinct2", "coverage"]
	gaps = {T: {c: abs(v[c] - tgt[c]) for c in crit} for T, v in per_T.items()}
	ranks = {c: pd.Series({T: gaps[T][c] for T in gaps}).rank(method="average") for c in crit}
	mean_rank = {T: float(np.mean([ranks[c][T] for c in crit])) for T in gaps}
	order = sorted(gaps, key=lambda T: (mean_rank[T], gaps[T]["coverage"], -T))
	res = {"human_targets": tgt, "per_T": per_T, "gaps": gaps,
		   "ranks": {c: {float(k): float(v) for k, v in r.items()} for c, r in ranks.items()},
		   "mean_rank": mean_rank, "T_star": order[0], "order": order,
		   "per_prefix": per_prefix, "git_head": E.git_head()}
	json.dump(res, open(OUT_JSON, "w"), indent=1)
	print(json.dumps({k: v for k, v in res.items() if k != "per_prefix"}, indent=1))


if __name__ == "__main__":
	{"collect": collect, "analyse": analyse}[sys.argv[1]]()
