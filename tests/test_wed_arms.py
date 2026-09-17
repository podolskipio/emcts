"""Acceptance tests for the Wednesday arms: AffPool, CenteredBias, Constrain, the generic valence
table, --terminal_on_failure and the backup_value log field.

Every new behaviour sits behind a flag whose default is the shipped planner, so -- as in
tests/test_emo_channel_freeze.py -- the first job of these tests is BIT-IDENTITY with the flags off:

  * scores: `_calculate_uct` with no pool / centre term is `.hex()`-equal to the pre-change formula;
  * whole searches: a real-game search at the defaults reproduces GOLDEN, a fingerprint of every
    table and every sim_steps record produced by the PRE-change source tree (the working tree before
    these arms landed, which already carried --search_horizon and the Phase-1 simlog). Regenerate it
    only from a pre-change tree:

        EMCTS_SRC=<pre-change src> python tests/test_wed_arms.py --golden

The rest checks that each arm does what its spec says when it is on.

    python -m pytest tests/test_wed_arms.py -q
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys

import numpy as np
import pytest

SRC = os.environ.get("EMCTS_SRC") or os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402
from games import PersuasionGame, EmotionAwarePersuasionGame  # noqa: E402
import mcts.emotion_mcts as em  # noqa: E402
from utils.utils import dotdict  # noqa: E402

SYS_ACTS = PersuasionGame.get_game_ontology()["system"]["dialog_acts"]
T = 6
HF_EMOTIONS = [e for e in Emotions if e != Emotions.Contempt]

# Fingerprints of default-flag searches, produced by the PRE-change tree (see module docstring).
GOLDEN = {
	"beta0.0": "363ac6e4af2cd6a597f78242f2b03d146264383c2f3538b632cfb22f2e222837",
	"beta0.7": "8bbfb26f0377588af598219f02aae31efb6ddf5586d32c35c623ce2c54438185",
}


# ---------------------------------------------------------------------------
# fakes: text sources, classifier and LLM planner only; game and planner are real
# ---------------------------------------------------------------------------
class FakeSystem:
	dialog_acts = SYS_ACTS

	def __init__(self):
		self.n = itertools.count()

	def get_utterance(self, state, action):
		return f"system line {next(self.n)}"


class FakeUser:
	def __init__(self):
		self.n = itertools.count()

	def get_utterance_w_da(self, state, _, mode="train"):
		return PersuasionGame.U_Neutral, f"user line {next(self.n)}"


class HashClassifier:
	"""A different, deterministic distribution per utterance (sha1-seeded), so nu varies across
	realizations and the buckets / centring / masks are exercised. Takes no global random draws."""
	emotions = list(Emotions)

	def __init__(self):
		self.records = []

	def predict_distribution_from_full_history(self, state, utt):
		rng = random.Random(hashlib.sha1(utt.encode()).hexdigest())
		w = [rng.random() ** 3 for _ in HF_EMOTIONS]
		d = {e: x / sum(w) for e, x in zip(HF_EMOTIONS, w)}
		d[Emotions.Contempt] = 0.0
		return d


class FakePlanner:
	"""Prior over 4 acts, leaf value a pure function of the dialogue length and last utterance."""
	dialog_acts = SYS_ACTS
	llm_prior_topk = None

	def get_valid_moves(self, state):
		m = np.zeros(len(SYS_ACTS))
		m[:4] = 1
		return m

	def predict(self, state):
		prior = np.array([0.4, 0.3, 0.2, 0.1] + [0.0] * (len(SYS_ACTS) - 4))
		h = int(hashlib.sha1(str(state.history[-1]).encode()).hexdigest()[:8], 16)
		return prior, (h % 2001) / 1000.0 - 1.0


def run_search(sims=120, seed=0, **planner_kw):
	np.random.seed(seed)
	clf = HashClassifier()
	game = EmotionAwarePersuasionGame(FakeSystem(), FakeUser(), None, False, clf, max_conv_turns=T)
	root = game.state_of(game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting)))
	cfg = dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 2, "search_horizon": "legacy"})
	p = em.EmotionAwareMultiObjectiveQ(game, FakePlanner(), cfg, clf, **planner_kw)
	for _ in range(sims):
		p.search(root)
	return p, root


PRECHANGE_STEP_KEYS = {
	"simulation_index", "depth", "action_prefix", "action", "parent_realization_idx", "parent_realization_id",
	"parent_nu", "parent_emotion_dist", "child_realization_id", "child_from_cache", "child_nu",
	"child_emotion_dist", "parent_Ns", "siblings", "selected_action", "z", "v",
}


def fingerprint(p) -> str:
	"""sha256 over every statistics table and every pre-change sim_steps field, floats via .hex()."""
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
	blob = {
		"Ns": norm(p.Ns), "Nsa": norm(p.Nsa), "Q": norm(p.Q), "P": norm(p.P), "Q_emo": norm(p.Q_emo),
		"M2_emo": norm(p.M2_emo), "emo_valences": norm(p.emo_valences),
		"realizations": {k: [len(r) for r in v] for k, v in sorted(p.realizations.items())},
		"sim_steps": [norm({k: s[k] for k in sorted(PRECHANGE_STEP_KEYS)}) for s in p.sim_steps],
	}
	return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# bit-identity with every flag off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_default_search_reproduces_the_prechange_tree(beta):
	p, _ = run_search(beta_emo=beta)
	assert fingerprint(p) == GOLDEN[f"beta{beta}"]


def test_constructor_defaults_are_off():
	p, _ = run_search(sims=1)
	assert p.aff_pool is False and p.emo_centre is False and p.emo_constraint_tau is None
	assert p.Q_pool == {} and p.N_pool == {} and p.root_decision == {}


def prechange_uct(p, s, a):
	Ns = p.Ns[s] or 1e-8
	explore = math.sqrt(Ns) / (1 + p.Nsa[s][a])
	q_emo = p.Q_emo.get(s, {}).get(a, 0.0)
	q_emo_adj = q_emo - p.emo_risk_lambda * p.sigma_emo(s, a)
	return p.Q[s][a] + p.beta_emo * q_emo_adj + p.configs.cpuct * p.P[s][a] * explore


def random_node_planner(seed, n_actions=5, **kw):
	rng = random.Random(seed)
	s, acts = "greeting__x", list(range(n_actions))
	nsa = {a: rng.choice([0, 0, 1, 2, 3, 5, 9]) for a in acts}
	p = object.__new__(em.EmotionAwareMultiObjectiveQ)
	p.configs = dotdict({"cpuct": 1.0})
	p.beta_emo, p.emo_risk_lambda = kw.get("beta_emo", 0.7), kw.get("emo_risk_lambda", 0.0)
	p.aff_pool_bias, p.aff_pool_tau = kw.get("aff_pool_bias", 0.1), 0.35
	p.valid_moves = {s: np.array(acts)}
	p.Ns, p.Nsa = {s: sum(nsa.values())}, {s: nsa}
	p.Q = {s: {a: (rng.uniform(-1, 1) if nsa[a] else 0.0) for a in acts}}
	p.P = {s: {a: rng.uniform(0, 1) for a in acts}}
	p.Q_emo = {s: {a: (rng.uniform(-0.14, 0.54) if nsa[a] else 0.0) for a in acts}}
	p.M2_emo = {s: {a: rng.uniform(0, 0.5) for a in acts}}
	p.Q_pool = {(b, a): rng.uniform(-1, 1) for b in (0, 1) for a in acts if rng.random() < 0.7}
	p.N_pool = {k: rng.randint(1, 30) for k in p.Q_pool}
	return p, s, acts


def test_scores_with_no_arm_are_bit_identical_to_prechange():
	for seed in range(200):
		p, s, acts = random_node_planner(seed, emo_risk_lambda=0.0 if seed % 2 else 0.3)
		for a in acts:
			assert p._calculate_uct(s, a).hex() == prechange_uct(p, s, a).hex()
			assert p._calculate_uct(s, a, bucket=None, emo_mu=None).hex() == prechange_uct(p, s, a).hex()


def test_default_valence_table_is_still_the_mined_object():
	assert em.EMOTION_VALENCE_TABLES["soft"] is em.EMOTION_VALENCE_MINED


def test_backup_value_is_logged_and_equals_the_scalar_backed_up_into_q():
	p, _ = run_search(beta_emo=0.7)
	# replay every step's backup_value into a running mean per edge; it must end at Q exactly
	sums = {}
	for step in p.sim_steps:
		assert step["backup_value"] == step["v"]
		root = p._sim_root_key
		node = "__".join(([root] if root else []) + step["action_prefix"])
		sums.setdefault((node, step["action"]), []).append(step["backup_value"])
	for (node, act), vs in sums.items():
		a = SYS_ACTS.index(act)
		q = 0.0
		for n, v in enumerate(vs):
			q = (n * q + v) / (n + 1)
		assert q == p.Q[node][a]
		assert len(vs) == p.Nsa[node][a]


# ---------------------------------------------------------------------------
# AffPool
# ---------------------------------------------------------------------------
def test_rave_beta_schedule():
	assert em.rave_beta(5, 0, 0.1) == 0.0
	assert em.rave_beta(0, 7, 0.1) == 1.0
	assert em.rave_beta(4, 12, 0.25) == pytest.approx(12 / (4 + 12 + 4 * 4 * 12 * 0.0625))
	# more own visits -> less weight on the pool; larger bias -> less weight on the pool
	assert em.rave_beta(10, 12, 0.1) < em.rave_beta(2, 12, 0.1)
	assert em.rave_beta(4, 12, 0.25) < em.rave_beta(4, 12, 0.05)


def test_q_eff_replaces_q_and_nothing_else():
	for seed in range(50):
		p, s, acts = random_node_planner(seed)
		for a in acts:
			for b in (0, 1):
				beta = em.rave_beta(p.Nsa[s][a], p.N_pool.get((b, a), 0), p.aff_pool_bias)
				q_eff = p.Q[s][a] if beta == 0 else (1 - beta) * p.Q[s][a] + beta * p.Q_pool[(b, a)]
				expected = prechange_uct(p, s, a) - p.Q[s][a] + q_eff
				assert p._calculate_uct(s, a, bucket=b) == pytest.approx(expected, abs=1e-12)


def test_pool_carries_the_task_return_pooled_over_every_node():
	p, root = run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1)
	cells = {}
	for step in p.sim_steps:
		assert step["aff_bucket"] == int(step["parent_nu"] < p.aff_pool_tau)
		cells.setdefault((step["aff_bucket"], step["action"]), []).append(step["backup_value"])
	assert {(b, p.player.dialog_acts[a]) for b, a in p.N_pool} == set(cells)
	for (b, act), vs in cells.items():
		key = (b, SYS_ACTS.index(act))
		assert p.N_pool[key] == len(vs)
		assert p.Q_pool[key] == pytest.approx(sum(vs) / len(vs), abs=1e-12)
	# both buckets occur, and cells really do pool several prefixes
	assert {b for b, _ in cells} == {0, 1}
	prefixes = {}
	for step in p.sim_steps:
		prefixes.setdefault((step["aff_bucket"], step["action"]), set()).add(tuple(step["action_prefix"]))
	assert max(len(v) for v in prefixes.values()) >= 3


def test_aff_pool_changes_the_search_and_logs_its_diagnostics():
	base, _ = run_search(beta_emo=0.0)
	pool, _ = run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1)
	assert fingerprint(pool) != fingerprint(base)
	for step in pool.sim_steps:
		assert {"aff_bucket", "pool_beta_selected", "pool_flip"} <= set(step)
		assert all({"Q_pool", "N_pool", "pool_beta"} <= set(sib) for sib in step["siblings"])
		assert 0.0 <= step["pool_beta_selected"] <= 1.0
	assert any(step["pool_flip"] for step in pool.sim_steps)


# ---------------------------------------------------------------------------
# CenteredBias
# ---------------------------------------------------------------------------
def test_centering_uses_expanded_siblings_only_and_leaves_unexpanded_scores_alone():
	for seed in range(100):
		p, s, acts = random_node_planner(seed)
		expanded = [a for a in acts if p.Nsa[s][a] > 0]
		mu = p._expanded_emo_mean(s)
		if not expanded:
			assert mu is None
			continue
		assert mu == pytest.approx(sum(p.Q_emo[s][a] for a in expanded) / len(expanded))
		for a in acts:
			centred = p._calculate_uct(s, a, emo_mu=mu)
			if a in expanded:
				assert centred == pytest.approx(prechange_uct(p, s, a) - p.beta_emo * mu, abs=1e-12)
			else:
				assert centred.hex() == prechange_uct(p, s, a).hex()


def test_centering_cannot_reorder_expanded_siblings():
	"""The defining property: among the siblings mu is computed over, the ranking is Bias's."""
	for seed in range(200):
		p, s, acts = random_node_planner(seed)
		expanded = [a for a in acts if p.Nsa[s][a] > 0]
		if len(expanded) < 2:
			continue
		mu = p._expanded_emo_mean(s)
		bias_rank = sorted(expanded, key=lambda a: -prechange_uct(p, s, a))
		centre_rank = sorted(expanded, key=lambda a: -p._calculate_uct(s, a, emo_mu=mu))
		assert bias_rank == centre_rank


def test_emo_centre_search_logs_mu_and_flips():
	p, _ = run_search(beta_emo=0.7, emo_centre=True)
	assert all({"emo_mu", "n_expanded", "centre_flip", "centre_flip_to"} <= set(s) for s in p.sim_steps)
	for s in p.sim_steps:
		assert (s["centre_flip_to"] is None) == (not s["centre_flip"])
		if s["n_expanded"] == 0:
			assert s["emo_mu"] is None


# ---------------------------------------------------------------------------
# Constrain
# ---------------------------------------------------------------------------
def test_feasible_set_rule():
	acts = [0, 1, 2, 3]
	q_emo = {0: 0.5, 1: 0.1, 2: 0.3, 3: -0.1}
	assert em.feasible_actions(acts, q_emo, {0: 5, 1: 5, 2: 5, 3: 5}, 0.2, 3) == [0, 2]
	assert em.feasible_actions(acts, q_emo, {0: 5, 1: 2, 2: 5, 3: 0}, 0.2, 3) == [0, 1, 2, 3]
	# nothing feasible -> the single highest-Q_emo action
	assert em.feasible_actions(acts, q_emo, {a: 9 for a in acts}, 0.9, 3) == [0]


def test_constrain_selects_only_feasible_actions_and_decides_the_root_over_them():
	tau = 0.2
	p, root = run_search(sims=150, beta_emo=0.0, emo_constraint_tau=tau, emo_constraint_m_warm=3)
	for s in p.sim_steps:
		sib = {x["action"]: x for x in s["siblings"]}
		chosen = sib[s["action"]]
		assert chosen["Q_emo"] >= tau or chosen["N"] < 3 or s["constraint_fallback"]
		assert s["action"] not in s["constraint_masked"]
	assert any(s["constraint_masked"] for s in p.sim_steps)
	prob = p.get_action_prob(root)
	d = p.root_decision
	assert d and SYS_ACTS[int(np.argmax(prob))] == d["action"]
	assert d["action"] in d["feasible"] or d["root_fallback"]
	assert d["disagree"] == (d["action"] != d["unrestricted_action"])


def test_constraint_off_leaves_the_root_distribution_untouched():
	p, root = run_search(beta_emo=0.7)
	prob = p.get_action_prob(root)
	key = p._to_string_rep(root)
	expected = np.zeros(len(SYS_ACTS))
	for a in p.valid_moves[key]:
		expected[a] = p.Nsa[key][a]
	assert np.array_equal(prob, expected / expected.sum())
	assert p.root_decision == {}


# ---------------------------------------------------------------------------
# small items
# ---------------------------------------------------------------------------
def test_generic_table_signs_and_scale():
	g = em.EMOTION_VALENCE_TABLES["generic"]
	mined = em.EMOTION_VALENCE_MINED
	assert g[Emotions.Happiness] == max(abs(mined[e]) for e in HF_EMOTIONS)
	assert g[Emotions.Neutral] == 0.0 and g[Emotions.Surprise] == 0.0
	for e in (Emotions.Sadness, Emotions.Fear, Emotions.Anger, Emotions.Disgust):
		assert g[e] == -g[Emotions.Happiness]
	assert g[Emotions.Contempt] == mined[Emotions.Contempt]
	assert "generic" in em.EMO_VALENCE_TABLES


@pytest.mark.parametrize("runner", ["rollout", "emomcts"])
def test_runner_flags_default_to_shipped_behaviour(runner, monkeypatch):
	import runners.rollout as rollout_runner
	import runners.emomcts as emomcts_runner
	mod = {"rollout": rollout_runner, "emomcts": emomcts_runner}[runner]
	src = open(mod.__file__).read()
	block = src[src.index('if __name__ == "__main__":'):]
	block = block.replace("cmd_args = finalize_args(parser.parse_args())", "ARGS = parser")
	block = block.split("ARGS = parser")[0] + "ARGS = parser"
	ns = dict(vars(mod))
	ns["__name__"] = "__main__"
	exec(compile(block, mod.__file__, "exec"), ns)
	parser = ns["ARGS"]
	a = parser.parse_args([])
	assert a.search_horizon == "legacy"
	assert a.aff_pool is False and a.emo_centre is False and a.emo_constraint_tau is None
	assert a.aff_pool_bias == 0.1 and a.aff_pool_tau == 0.35 and a.emo_constraint_m_warm == 3
	assert a.emo_valence_table == "soft"
	assert parser.parse_args(["--terminal_on_failure"]).search_horizon == "episode"
	assert parser.parse_args(["--emo_valence_table", "generic"]).emo_valence_table == "generic"


if __name__ == "__main__":
	if "--golden" in sys.argv:
		for beta in (0.0, 0.7):
			p, _ = run_search(beta_emo=beta)
			print(f'"beta{beta}": "{fingerprint(p)}",')
		sys.exit(0)
	sys.exit(pytest.main([__file__, "-q"]))
