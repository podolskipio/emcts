"""Readiness Phase 1C (C3) -- where is the value estimator's noise? PREREG Entry 7.

    python analysis/readiness/scripts/phase1_c3.py [--variant current]

20 states: one persuadee-ending prefix, at a random turn, from each of 20 dialogues played in the frozen
A_NoEmo_s20_seed1 run (seed 20260925), with that dialogue's persona. The value estimator is called
REPEATS times per state. `--variant` names the estimator configuration, so Phase 3B re-runs this script on
the same 20 states with its fixes:
  current         heuristic() as the frozen grid runs it: 10 samples at T 1.1
  logit           --logit_scoring value: the same prompt read off the logits
  lowT            10 samples at --value_temperature 0.3
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env as E  # noqa: E402

from games import PersuasionGame  # noqa: E402
from utils.sessions import DialogSession  # noqa: E402

RUN = os.path.join(E.REPO, "analysis", "grid", "runs", "A_NoEmo_s20_seed1", "A_NoEmo_s20_seed1",
				   "A_NoEmo_s20_seed1.pkl")
N_STATES, REPEATS, SEED = 20, 20, 20260925


def states():
	rng = np.random.default_rng(SEED)
	episodes = pickle.load(open(RUN, "rb"))
	picks = rng.choice(len(episodes), N_STATES, replace=False)
	out = []
	for i in sorted(picks):
		ep = episodes[i]
		usr_idx = [j for j, h in enumerate(ep["history"]) if h[0] == PersuasionGame.USR]
		end = usr_idx[rng.integers(len(usr_idx))]
		s = DialogSession(PersuasionGame.SYS, PersuasionGame.USR)
		for role, da, _emo, utt in ep["history"][:end + 1]:
			s.add_single(role, da, utt)
		out.append({"dlg_id": ep["did"], "turn": end // 2, "state": s, "persona": ep["persona"]})
	return out


def value_fn(variant, persona):
	kw = {"logit_scoring": "value"} if variant == "logit" else {}
	_g, _s, _u, planner = E.build_agents("p4g", E.backbone(), "chat", persona=persona, llm_prior_topk=5, **kw)
	if variant == "lowT":
		planner.value_temperature = 0.3
	return lambda st: float(planner.heuristic(st)[0])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--variant", default="current", choices=["current", "logit", "lowT"])
	a = ap.parse_args()
	sts = states()

	def one(item):
		f = value_fn(a.variant, item["persona"])
		return [f(item["state"]) for _ in range(REPEATS)]

	with ThreadPoolExecutor(20) as ex:
		vals = list(tqdm(ex.map(one, sts), total=len(sts)))
	V = np.array(vals)                      # states x repeats
	sd = V.std(1, ddof=1)
	within = float(np.mean(V.var(1, ddof=1)))
	between = float(V.mean(1).var(ddof=1))
	res = {
		"variant": a.variant, "n_states": N_STATES, "repeats": REPEATS,
		"per_state": [{"dlg_id": s["dlg_id"], "turn": s["turn"], "mean": float(m), "sd": float(d),
					   "values": list(map(float, v))} for s, m, d, v in zip(sts, V.mean(1), sd, V)],
		"per_state_sd": {"median": float(np.median(sd)), "mean": float(sd.mean()), "min": float(sd.min()),
						 "max": float(sd.max()), "q25": float(np.quantile(sd, .25)), "q75": float(np.quantile(sd, .75))},
		"pooled_within_sd": within ** 0.5, "between_state_sd": between ** 0.5,
		"noise_share": within / (within + between) if within + between > 0 else 0.0,
		"git_head": E.git_head(),
	}
	res["LARGE"] = bool(res["noise_share"] >= 0.10 or res["per_state_sd"]["median"] >= 0.10)
	out = os.path.join(E.READINESS, f"phase1_c3_{a.variant}.json")
	json.dump(res, open(out, "w"), indent=1)
	print(json.dumps({k: v for k, v in res.items() if k != "per_state"}, indent=1))


if __name__ == "__main__":
	main()
