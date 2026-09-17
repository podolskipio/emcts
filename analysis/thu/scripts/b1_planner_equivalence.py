"""B1 diagnosis, part 1: is EmotionAwareMultiObjectiveQ at beta 0 the same SEARCH as OpenLoopMCTS (GDP-Zero)?

Both planners run on the SAME game (the emotion-aware p4g game, through a one-line adapter so OpenLoopMCTS
can consume its (state, emotion) return), the same fake text/LLM/classifier stack as tests/test_wed_arms.py,
the same numpy seed, many root states and simulation budgets. Compared per search: every Ns / Nsa / Q / P
table, the realization pools, the sequence of selected actions, and the root argmax.
"""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests"))
import test_wed_arms as W  # noqa: E402  (puts src/ on the path; real game + planners, fake text sources)
from mcts.mcts import OpenLoopMCTS  # noqa: E402
from utils.utils import dotdict  # noqa: E402

em = W.em


class StateOnly:
	"""The emotion game with get_next_state returning the state alone, as OpenLoopMCTS expects."""
	def __init__(self, game):
		self._g = game
	def __getattr__(self, k):
		return getattr(self._g, k)
	def get_next_state(self, state, action, mode="train"):
		return self._g.get_next_state(state, action, mode)[0]


def tables(p):
	def norm(x):
		if isinstance(x, float):
			return x.hex()
		if isinstance(x, (np.floating,)):
			return float(x).hex()
		if isinstance(x, (np.integer,)):
			return int(x)
		if isinstance(x, dict):
			return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
		if isinstance(x, (list, tuple, np.ndarray)):
			return [norm(v) for v in x]
		return x
	return {"Ns": norm(p.Ns), "Nsa": norm(p.Nsa), "Q": norm(p.Q), "P": norm(p.P),
			"realizations": {k: [r.to_string_rep(keep_sys_da=True, keep_user_da=True) for r in v] for k, v in sorted(p.realizations.items())}}


def one(seed, sims, topk, root_depth):
	out = {}
	for name in ("gdpzero", "emomcts_beta0"):
		np.random.seed(seed)
		clf = W.HashClassifier()
		game = W.EmotionAwarePersuasionGame(W.FakeSystem(), W.FakeUser(), None, False, clf, max_conv_turns=W.T)
		state = game.state_of(game.get_next_state(game.init_dialog(), W.SYS_ACTS.index(W.PersuasionGame.S_Greeting)))
		for k in range(root_depth):  # deeper roots: extra real turns
			state = game.state_of(game.get_next_state(state, (seed + k) % len(W.SYS_ACTS)))
		planner = W.FakePlanner()
		planner.llm_prior_topk = topk
		cfg = dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 2, "search_horizon": "episode"})
		if name == "gdpzero":
			p = OpenLoopMCTS(StateOnly(game), planner, cfg)
		else:
			p = em.EmotionAwareMultiObjectiveQ(game, planner, cfg, clf, beta_emo=0.0)
		for _ in range(sims):
			p.search(state)
		t = tables(p)
		out[name] = {"sha": hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest(),
					 "root_argmax": int(np.argmax(p.get_action_prob(state))), "tables": t}
	return out


def main():
	res, n_same, n_root = [], 0, 0
	for seed in range(20):
		for sims in (20, 50):
			for topk in (None, 5):
				for depth in (0, 2):
					o = one(seed, sims, topk, depth)
					same = o["gdpzero"]["sha"] == o["emomcts_beta0"]["sha"]
					root = o["gdpzero"]["root_argmax"] == o["emomcts_beta0"]["root_argmax"]
					n_same += same; n_root += root
					row = {"seed": seed, "sims": sims, "topk": topk, "root_depth": depth, "tables_identical": same, "root_identical": root}
					if not same:
						a, b = o["gdpzero"]["tables"], o["emomcts_beta0"]["tables"]
						row["differs_in"] = [k for k in a if a[k] != b[k]]
					res.append(row)
	print(f"searches: {len(res)}  identical tables (bit-exact): {n_same}  identical root decision: {n_root}")
	for r in res:
		if not r["tables_identical"]:
			print("  DIFF", r)
	json.dump(res, open(os.path.join(REPO, "analysis", "thu", "b1_planner_equivalence.json"), "w"), indent=1)


if __name__ == "__main__":
	main()
