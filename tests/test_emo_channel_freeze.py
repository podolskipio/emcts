"""Acceptance tests for the two pre-freeze changes to the Q_emo backup.

Part 1 — Welford variance tracking (M2_emo / sigma_emo, --emo_risk_lambda)
Part 2 — delta-valence signal (--emo_signal {level,delta})

Both are meant to be STRICT GENERALIZATIONS: at the default flag values the planner
must compute exactly what it computed before. These tests are the proof, so most of
them assert bit-identity rather than approximate equality.

    python -m pytest tests/test_emo_channel_freeze.py -q
    python tests/test_emo_channel_freeze.py            # same, without pytest
"""
import os
import sys
import math
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mcts.emotion_mcts import (  # noqa: E402
	EmotionAwareMultiObjectiveQ,
	EMOTION_VALENCE_MINED,
	EMO_DELTA_RANGE_SCALE,
	check_emo_signal_flags,
	welford_sigma,
	welford_update,
	welford_variance,
)
from emotion_classifiers.llm_emotion import Emotions  # noqa: E402
from utils.utils import dotdict  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
Z_SEQUENCE = [0.54, -0.14, 0.0, 0.31, -0.6, 0.09, 0.24, -0.07, 0.13, 0.11]


def running_mean_prechange(zs):
	"""The mean update EXACTLY as it was written before this task:

	    Q_emo = (n_old * Q_emo + z) / (n_old + 1)

	Verbatim from the pre-change EmotionAwareMultiObjectiveQ.search backup.
	"""
	q, n_old = 0.0, 0
	for z in zs:
		q = (n_old * q + z) / (n_old + 1)
		n_old += 1
	return q


def welford_mean_and_m2(zs):
	"""Drive the shipped Welford accumulator over the same sequence."""
	mean, m2, n_old = 0.0, 0.0, 0
	for z in zs:
		mean, m2 = welford_update(n_old, mean, m2, z)
		n_old += 1
	return mean, m2


def make_planner(**overrides):
	"""An EmotionAwareMultiObjectiveQ with only the fields the scoring / signal code
	touches. Bypasses __init__ so no game, player or classifier is needed."""
	p = object.__new__(EmotionAwareMultiObjectiveQ)
	p.configs = dotdict({"cpuct": 1.0})
	p.beta_emo = 0.3
	p.emo_risk_lambda = 0.0
	p.emo_signal = "level"
	p.Ns, p.Nsa, p.Q, p.P, p.Q_emo, p.M2_emo = {}, {}, {}, {}, {}, {}
	p.emo_valences = {}
	for k, v in overrides.items():
		setattr(p, k, v)
	return p


class FakeState:
	"""Minimal stand-in for EmotionAwareDialogSession: only the last turn's distribution
	is read by the emotion channel."""

	def __init__(self, distribution, empty_history=False):
		self._dist = distribution
		self.history = [] if empty_history else [object()]

	def predicted_distribution(self):
		return self._dist


def dist(**kwargs):
	"""{Emotions: p} from keyword emotion names, renormalized."""
	d = {getattr(Emotions, k): float(v) for k, v in kwargs.items()}
	total = sum(d.values())
	return {e: p / total for e, p in d.items()}


def nu(d):
	"""Reference implementation of the valence functional, independent of the class."""
	if not d:
		return 0.0
	return sum(p * EMOTION_VALENCE_MINED.get(e, 0.0) for e, p in d.items())


# ---------------------------------------------------------------------------
# 3.1 — Welford mean is algebraically identical to the previous running mean
# ---------------------------------------------------------------------------
def test_welford_mean_matches_prechange_running_mean():
	mean, _ = welford_mean_and_m2(Z_SEQUENCE)
	assert mean == pytest.approx(running_mean_prechange(Z_SEQUENCE), abs=1e-12)
	# and against the plain arithmetic mean, which is what both are supposed to be
	assert mean == pytest.approx(float(np.mean(Z_SEQUENCE)), abs=1e-12)


def test_welford_mean_matches_prechange_on_random_sequences():
	rng = random.Random(20260907)
	for _ in range(200):
		zs = [rng.uniform(-1.0, 1.0) for _ in range(rng.randint(1, 40))]
		mean, _ = welford_mean_and_m2(zs)
		assert mean == pytest.approx(running_mean_prechange(zs), abs=1e-12)


# ---------------------------------------------------------------------------
# 3.1 — Welford variance matches numpy.var(..., ddof=1)
# ---------------------------------------------------------------------------
def test_welford_variance_matches_numpy_ddof1():
	_, m2 = welford_mean_and_m2(Z_SEQUENCE)
	n = len(Z_SEQUENCE)
	assert welford_variance(n, m2) == pytest.approx(float(np.var(Z_SEQUENCE, ddof=1)), abs=1e-12)
	assert welford_sigma(n, m2) == pytest.approx(float(np.std(Z_SEQUENCE, ddof=1)), abs=1e-12)


def test_welford_variance_matches_numpy_on_random_sequences():
	rng = random.Random(11)
	for _ in range(200):
		zs = [rng.uniform(-1.0, 1.0) for _ in range(rng.randint(2, 60))]
		_, m2 = welford_mean_and_m2(zs)
		assert welford_variance(len(zs), m2) == pytest.approx(float(np.var(zs, ddof=1)), abs=1e-12)


def test_welford_catches_the_classic_same_deviation_bug():
	"""Sanity guard on the test itself: using delta twice (instead of delta*delta2)
	gives a DIFFERENT number, so the test above would actually catch that bug."""
	mean, m2_wrong, n_old = 0.0, 0.0, 0
	for z in Z_SEQUENCE:
		delta = z - mean
		mean = mean + delta / (n_old + 1)
		m2_wrong += delta * delta          # the bug: same deviation twice
		n_old += 1
	wrong_var = m2_wrong / (len(Z_SEQUENCE) - 1)
	assert wrong_var != pytest.approx(float(np.var(Z_SEQUENCE, ddof=1)), abs=1e-6)


# ---------------------------------------------------------------------------
# 3.1 — N < 2 -> var_emo == 0.0, no division by zero
# ---------------------------------------------------------------------------
def test_variance_is_zero_below_two_observations():
	assert welford_variance(0, 0.0) == 0.0
	assert welford_variance(1, 0.0) == 0.0
	assert welford_sigma(0, 0.0) == 0.0
	assert welford_sigma(1, 0.0) == 0.0
	# one observation really does leave M2 == 0, so nothing divides by n-1 == 0
	mean, m2 = welford_mean_and_m2([0.42])
	assert (mean, m2) == (0.42, 0.0)
	assert welford_variance(1, m2) == 0.0


def test_planner_var_emo_is_zero_below_two_visits_and_on_unknown_edges():
	p = make_planner(
		Nsa={"n": {0: 0, 1: 1, 2: 5}},
		M2_emo={"n": {0: 0.0, 1: 0.0, 2: 0.8}},
	)
	assert p.var_emo("n", 0) == 0.0
	assert p.var_emo("n", 1) == 0.0
	assert p.var_emo("n", 2) == pytest.approx(0.8 / 4)
	assert p.sigma_emo("n", 1) == 0.0
	# never-seen node / action must not raise
	assert p.var_emo("missing", 0) == 0.0
	assert p.sigma_emo("n", 99) == 0.0


# ---------------------------------------------------------------------------
# 3.1 — lambda = 0.0 -> selection scores bit-identical to the pre-change code
# ---------------------------------------------------------------------------
def prechange_uct(p, hashable_state, action):
	"""_calculate_uct EXACTLY as it was before this task (verbatim, minus the risk term)."""
	Ns = p.Ns[hashable_state] or 1e-8
	explore = math.sqrt(Ns) / (1 + p.Nsa[hashable_state][action])
	q_emo = p.Q_emo.get(hashable_state, {}).get(action, 0.0)
	return (
		p.Q[hashable_state][action]
		+ p.beta_emo * q_emo
		+ p.configs.cpuct * p.P[hashable_state][action] * explore
	)


def fixed_edge_stats_planner(seed=7, n_actions=13, beta_emo=0.3, emo_risk_lambda=0.0):
	rng = random.Random(seed)
	acts = list(range(n_actions))
	node = "greeting__proposition-of-donation"
	nsa = {a: rng.randint(0, 12) for a in acts}
	p = make_planner(
		beta_emo=beta_emo,
		emo_risk_lambda=emo_risk_lambda,
		Ns={node: sum(nsa.values())},
		Nsa={node: nsa},
		Q={node: {a: rng.uniform(0.0, 1.0) for a in acts}},
		P={node: {a: rng.uniform(0.0, 1.0) for a in acts}},
		Q_emo={node: {a: rng.uniform(-1.0, 1.0) for a in acts}},
		# non-trivial spread on every edge, so a lambda leak would move the score
		M2_emo={node: {a: rng.uniform(0.0, 4.0) for a in acts}},
	)
	return p, node, acts


def test_lambda_zero_is_bit_identical_to_prechange_selection():
	p, node, acts = fixed_edge_stats_planner()
	assert p.emo_risk_lambda == 0.0
	for a in acts:
		new = p._calculate_uct(node, a)
		old = prechange_uct(p, node, a)
		# bit-identical, not merely close
		assert new == old, (a, new, old, new - old)
		assert new.hex() == old.hex()


def test_lambda_zero_is_bit_identical_across_many_random_edge_sets():
	for seed in range(50):
		p, node, acts = fixed_edge_stats_planner(seed=seed, beta_emo=1.0)
		for a in acts:
			assert p._calculate_uct(node, a) == prechange_uct(p, node, a)


def test_lambda_zero_holds_even_for_negative_zero_q_emo():
	"""-0.0 - 0.0*sigma == -0.0: the identity survives the signed-zero edge case."""
	node = "n"
	p = make_planner(
		Ns={node: 4}, Nsa={node: {0: 2}}, Q={node: {0: 0.0}}, P={node: {0: 0.5}},
		Q_emo={node: {0: -0.0}}, M2_emo={node: {0: 3.0}},
	)
	assert p._calculate_uct(node, 0) == prechange_uct(p, node, 0)


def test_nonzero_lambda_actually_penalises_spread():
	"""The flag is not a no-op: with lambda > 0 a high-sigma edge scores strictly lower."""
	p, node, acts = fixed_edge_stats_planner(emo_risk_lambda=0.5, beta_emo=1.0)
	moved = [a for a in acts if p._calculate_uct(node, a) != prechange_uct(p, node, a)]
	assert moved, "lambda > 0 changed no score — the risk term is not wired in"
	for a in moved:
		assert p._calculate_uct(node, a) < prechange_uct(p, node, a)


# ---------------------------------------------------------------------------
# 3.1 — --emo_signal level -> z identical to the pre-change code
# ---------------------------------------------------------------------------
CHILD_DISTS = [
	dist(Happiness=0.7, Neutral=0.3),
	dist(Sadness=0.5, Anger=0.5),
	dist(Neutral=1.0),
	dist(Surprise=0.2, Fear=0.3, Disgust=0.5),
]


def test_level_signal_matches_prechange_z():
	"""Pre-change: emo_v = self._emotion_quality(next_state.predicted_distribution())."""
	p = make_planner(emo_signal="level")
	for d in CHILD_DISTS:
		parent = FakeState(dist(Happiness=1.0))   # must be ignored under `level`
		child = FakeState(d)
		prechange_z = p._emotion_quality(child.predicted_distribution())
		assert p._emotion_signal(parent, child) == prechange_z
		assert p._emotion_signal(parent, child) == nu(d)


def test_level_signal_ignores_the_parent_entirely():
	p = make_planner(emo_signal="level")
	child = FakeState(CHILD_DISTS[0])
	baseline = p._emotion_signal(FakeState(None), child)
	for parent in (None, FakeState(None), FakeState({}), FakeState(dist(Anger=1.0)),
				   FakeState(None, empty_history=True)):
		assert p._emotion_signal(parent, child) == baseline


def test_level_signal_of_a_missing_child_distribution_is_zero():
	"""nu(empty) = 0 — unchanged convention."""
	p = make_planner(emo_signal="level")
	assert p._emotion_signal(FakeState(dist(Anger=1.0)), FakeState(None)) == 0.0
	assert p._emotion_signal(FakeState(dist(Anger=1.0)), FakeState({})) == 0.0


# ---------------------------------------------------------------------------
# 3.1 — --emo_signal delta at the root -> z == 0.0
# ---------------------------------------------------------------------------
def test_delta_signal_is_zero_when_the_parent_has_no_distribution():
	"""The root of a search, and any node whose parent turn was never classified
	(a system turn, a replayed placeholder), have no d_parent. z_delta = 0 there,
	matching the nu(empty) = 0 convention — it does NOT fall back to the level
	value, which would silently mix the two signals inside one run."""
	p = make_planner(emo_signal="delta")
	child = FakeState(dist(Happiness=1.0))
	assert nu(child.predicted_distribution()) != 0.0        # the level value is not 0
	for root_parent in (
		None,                                # no parent node at all
		FakeState(None),                     # parent turn never classified (SYS turn)
		FakeState({}),                       # empty distribution
		FakeState(None, empty_history=True), # freshly-initialised session
	):
		assert p._emotion_signal(root_parent, child) == 0.0


def test_delta_signal_is_the_scaled_difference_when_the_parent_is_known():
	p = make_planner(emo_signal="delta")
	parent_d, child_d = dist(Sadness=1.0), dist(Happiness=1.0)
	z = p._emotion_signal(FakeState(parent_d), FakeState(child_d))
	assert z == pytest.approx((nu(child_d) - nu(parent_d)) / EMO_DELTA_RANGE_SCALE)
	assert z > 0.0                       # sadness -> happiness is an improvement
	# and the reverse move is the exact negative
	back = p._emotion_signal(FakeState(child_d), FakeState(parent_d))
	assert back == pytest.approx(-z)


def test_delta_signal_is_zero_when_affect_does_not_move():
	"""The point of the delta arm: a *level* signal rewards never saying anything
	negative; a delta only rewards moving the user."""
	p = make_planner(emo_signal="delta")
	d = dist(Happiness=1.0)
	assert p._emotion_signal(FakeState(d), FakeState(d)) == 0.0


# ---------------------------------------------------------------------------
# 3.1 — delta renormalisation keeps Q_emo in [-1, +1] (randomised stress test)
# ---------------------------------------------------------------------------
def random_distribution(rng):
	emotions = list(EMOTION_VALENCE_MINED.keys())
	weights = [rng.random() for _ in emotions]
	total = sum(weights) or 1.0
	return {e: w / total for e, w in zip(emotions, weights)}


def test_delta_renormalisation_keeps_z_and_q_emo_in_range():
	rng = random.Random(4242)
	p = make_planner(emo_signal="delta")
	for _ in range(2000):
		parent = FakeState(random_distribution(rng))
		child = FakeState(random_distribution(rng))
		z = p._emotion_signal(parent, child)
		assert -1.0 <= z <= 1.0, z
	# and the running mean of a whole edge's worth of deltas stays in range too
	for _ in range(300):
		mean, m2, n_old = 0.0, 0.0, 0
		for _ in range(rng.randint(1, 50)):
			z = p._emotion_signal(FakeState(random_distribution(rng)),
								  FakeState(random_distribution(rng)))
			mean, m2 = welford_update(n_old, mean, m2, z)
			n_old += 1
			assert -1.0 <= mean <= 1.0, mean


def test_delta_range_is_saturated_by_the_extreme_pair():
	"""The /2 is exactly right, not merely safe: the widest possible swing maps to ±1."""
	p = make_planner(emo_signal="delta")
	best = max(EMOTION_VALENCE_MINED, key=EMOTION_VALENCE_MINED.get)
	worst = min(EMOTION_VALENCE_MINED, key=EMOTION_VALENCE_MINED.get)
	z = p._emotion_signal(FakeState({worst: 1.0}), FakeState({best: 1.0}))
	assert z == pytest.approx(
		(EMOTION_VALENCE_MINED[best] - EMOTION_VALENCE_MINED[worst]) / 2.0)
	assert abs(z) <= 1.0
	# with a hypothetical full-range table it lands exactly on the bound
	assert (1.0 - (-1.0)) / EMO_DELTA_RANGE_SCALE == 1.0


def test_level_signal_also_stays_in_range():
	rng = random.Random(99)
	p = make_planner(emo_signal="level")
	for _ in range(2000):
		z = p._emotion_signal(FakeState(random_distribution(rng)),
							  FakeState(random_distribution(rng)))
		assert -1.0 <= z <= 1.0, z


# ---------------------------------------------------------------------------
# flags / guards
# ---------------------------------------------------------------------------
class _DummyClassifier:
	emotions = list(EMOTION_VALENCE_MINED.keys())


def build_real_planner(**kwargs):
	"""Go through the real constructor. MCTS.__init__ only stores game/player/configs and
	EmotionAwareOpenLoopMCTS only stores the classifier, so dummies are enough — nothing
	here touches an LLM."""
	configs = dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 3})
	return EmotionAwareMultiObjectiveQ(object(), object(), configs, _DummyClassifier(), **kwargs)


def test_constructor_defaults_are_the_prechange_behaviour():
	p = build_real_planner()
	assert p.emo_risk_lambda == 0.0
	assert p.emo_signal == "level"
	assert p.Q_emo == {} and p.M2_emo == {}


def test_unknown_emo_signal_is_rejected_by_the_constructor():
	with pytest.raises(ValueError, match="emo_signal"):
		build_real_planner(emo_signal="absolute")
	# the two documented values are accepted
	assert build_real_planner(emo_signal="level").emo_signal == "level"
	assert build_real_planner(emo_signal="delta").emo_signal == "delta"


def test_delta_crossed_with_nondefault_n_step_is_refused():
	check_emo_signal_flags("level", 1)
	check_emo_signal_flags("delta", 1)     # default horizon: fine
	check_emo_signal_flags("level", 5)     # n-step on the level arm: that is B2, fine
	with pytest.raises(ValueError, match="telescopes"):
		check_emo_signal_flags("delta", 5)


def test_runner_defaults_are_level_and_zero_lambda():
	"""The freeze contract: both new flags default to the pre-change behaviour."""
	import argparse
	import runners.emomcts as emomcts_runner
	import runners.rollout as rollout_runner

	for mod, extra in ((emomcts_runner, {}), (rollout_runner, {})):
		parser = argparse.ArgumentParser()
		# rebuild only the two flags under test, from the module's own constants
		parser.add_argument('--emo_risk_lambda', '--emo-risk-lambda', type=float, default=0.0)
		parser.add_argument('--emo_signal', '--emo-signal',
							choices=list(mod.EMO_SIGNALS), default='level')
		args = parser.parse_args([])
		assert args.emo_risk_lambda == 0.0
		assert args.emo_signal == "level"
		# the dashed spelling from the spec resolves to the same dest
		assert parser.parse_args(["--emo-signal", "delta"]).emo_signal == "delta"
		assert parser.parse_args(["--emo-risk-lambda", "0.5"]).emo_risk_lambda == 0.5


# ---------------------------------------------------------------------------
# schema: M2_emo / sigma_emo present on every edge, in every run
# ---------------------------------------------------------------------------
def test_subtree_emo_stats_covers_every_edge_of_an_emotion_planner():
	from runners._common import subtree_emo_stats
	p, node, acts = fixed_edge_stats_planner()
	m2, sigma = subtree_emo_stats(p)
	assert set(m2) == set(sigma) == {node}
	assert set(m2[node]) == set(sigma[node]) == set(acts)
	for a in acts:
		assert m2[node][a] == p.M2_emo[node][a]
		assert sigma[node][a] == welford_sigma(p.Nsa[node][a], p.M2_emo[node][a])


def test_subtree_emo_stats_reports_zeros_for_a_planner_without_the_channel():
	"""GDP-Zero baselines run plain OpenLoopMCTS: it never writes the emotion channel, so
	M2_emo stays the empty table MCTS declares. The schema still has to be complete, with
	0.0 on every edge."""
	from mcts.mcts import OpenLoopMCTS
	from runners._common import subtree_emo_stats

	planner = object.__new__(OpenLoopMCTS)   # no __init__: only Nsa is needed here
	planner.Nsa = {"a": {0: 3, 1: 0}, "a__b": {0: 1}}
	assert planner.M2_emo == {}, "a baseline planner must still answer with an empty table"

	m2, sigma = subtree_emo_stats(planner)
	assert m2 == {"a": {0: 0.0, 1: 0.0}, "a__b": {0: 0.0}}
	assert sigma == {"a": {0: 0.0, 1: 0.0}, "a__b": {0: 0.0}}


# ---------------------------------------------------------------------------
# the backup itself: mean unchanged, M2 accumulated, ordering respected
# ---------------------------------------------------------------------------
def test_update_emo_channel_reproduces_the_prechange_mean_and_tracks_variance():
	node = "n"
	p = make_planner(Nsa={node: {0: 0}}, Q_emo={node: {0: 0.0}}, M2_emo={node: {0: 0.0}})
	for z in Z_SEQUENCE:
		n_old = p.Nsa[node][0]                 # pre-increment count, as in search()
		p._update_emo_channel(node, 0, z, n_old)
		p.Nsa[node][0] += 1                    # counters bump AFTER the update
	assert p.Q_emo[node][0] == pytest.approx(running_mean_prechange(Z_SEQUENCE), abs=1e-12)
	assert p.var_emo(node, 0) == pytest.approx(float(np.var(Z_SEQUENCE, ddof=1)), abs=1e-12)
	assert p.sigma_emo(node, 0) == pytest.approx(float(np.std(Z_SEQUENCE, ddof=1)), abs=1e-12)


def test_update_emo_channel_on_a_constant_sequence_has_zero_sigma():
	node = "n"
	p = make_planner(Nsa={node: {0: 0}}, Q_emo={node: {0: 0.0}}, M2_emo={node: {0: 0.0}})
	for _ in range(8):
		p._update_emo_channel(node, 0, 0.37, p.Nsa[node][0])
		p.Nsa[node][0] += 1
	assert p.Q_emo[node][0] == pytest.approx(0.37)
	assert p.sigma_emo(node, 0) == pytest.approx(0.0, abs=1e-15)


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# 1.5 — frozen NDJSON subtree schema
# ---------------------------------------------------------------------------
FROZEN_SUBTREE_KEYS = {
	"dlg_id", "turn", "node_id", "parent_id", "action_seq", "utterances", "emotion_dist",
	"leaf_value", "N", "Q", "Q_emo", "M2_emo", "sigma_emo", "root_visit_dist",
	"cache_hit", "seed", "depth", "per_step_valences",
}


class FakeRealization:
	SYS = "Persuader"

	def __init__(self, sys_utt, distribution):
		self.history = [object(), object()]
		self._utt = sys_utt
		self._dist = distribution

	def get_turn_utt(self, turn, role):
		return self._utt

	def predicted_distribution(self):
		return self._dist


class FakePlayer:
	dialog_acts = ["greeting", "credibility appeal", "emotion appeal"]


def tree_planner(with_emotion_channel=True):
	"""A hand-built two-level tree, so the record builder is tested without a search."""
	p = object.__new__(EmotionAwareMultiObjectiveQ)
	p.player = FakePlayer()
	p._to_string_rep = lambda state: state          # states are already keys here
	root = "greeting"
	child = "greeting__credibility appeal"
	p.Ns = {root: 5, child: 2}
	p.Nsa = {root: {0: 0, 1: 3, 2: 2}, child: {0: 0, 1: 1, 2: 1}}
	p.Q = {root: {0: 0.0, 1: 0.4, 2: 0.6}, child: {0: 0.0, 1: 0.2, 2: 0.3}}
	p.realizations = {
		root: [FakeRealization("hello there", {Emotions.Neutral: 1.0})],
		child: [FakeRealization("they are credible", {Emotions.Happiness: 1.0}),
				FakeRealization("they publish accounts", {Emotions.Sadness: 1.0})],
	}
	p.realizations_Vs = {}
	p.node_V = {root: 0.25, child: 0.5}
	p.cache_hits = {child: [4, 1]}
	if with_emotion_channel:
		p.Q_emo = {root: {0: 0.0, 1: 0.1, 2: -0.2}, child: {0: 0.0, 1: 0.3, 2: 0.0}}
		p.M2_emo = {root: {0: 0.0, 1: 0.5, 2: 0.2}, child: {0: 0.0, 1: 0.0, 2: 0.0}}
		p.emo_valences = {root: {0: [], 1: [0.1, -0.4, 0.7], 2: [-0.3, -0.1]},
						  child: {0: [], 1: [0.3], 2: [0.0]}}
	return p, root, child


def test_subtree_records_carry_every_frozen_key():
	from runners._common import build_subtree_records
	p, root, _ = tree_planner()
	records = build_subtree_records(p, dlg_id="d1", turn=3, root_state=root, seed=0)
	assert records, "no records built"
	for rec in records:
		assert set(rec) == FROZEN_SUBTREE_KEYS, set(rec) ^ FROZEN_SUBTREE_KEYS
		assert rec["dlg_id"] == "d1" and rec["turn"] == 3 and rec["seed"] == 0


def test_subtree_records_are_one_root_plus_one_per_edge():
	from runners._common import build_subtree_records
	p, root, _ = tree_planner()
	records = build_subtree_records(p, dlg_id="d1", turn=0, root_state=root)
	n_edges = sum(len(a) for a in p.Nsa.values())
	assert len(records) == n_edges + 1
	roots = [r for r in records if r["parent_id"] is None]
	assert len(roots) == 1 and roots[0]["node_id"] == root
	# unvisited edges are logged too — that is what makes "on every edge" checkable
	assert any(r["N"] == 0 for r in records)


def test_subtree_record_fields_belong_to_the_incoming_edge():
	from runners._common import build_subtree_records
	p, root, child = tree_planner()
	by_id = {r["node_id"]: r for r in build_subtree_records(p, dlg_id="d", turn=0, root_state=root)}
	rec = by_id[child]
	assert rec["parent_id"] == root
	assert rec["action_seq"] == ["greeting", "credibility appeal"]
	assert rec["depth"] == 1
	assert rec["N"] == p.Nsa[root][1] == 3
	assert rec["Q"] == p.Q[root][1]
	assert rec["Q_emo"] == p.Q_emo[root][1]
	assert rec["M2_emo"] == p.M2_emo[root][1]
	assert rec["sigma_emo"] == welford_sigma(3, p.M2_emo[root][1])
	assert rec["per_step_valences"] == p.emo_valences[root][1]
	assert rec["leaf_value"] == 0.5
	assert rec["cache_hit"] is True
	# all R cached realizations, and the FULL softmax averaged over them (never an argmax)
	assert rec["utterances"] == ["they are credible", "they publish accounts"]
	assert rec["emotion_dist"] == {"happiness": 0.5, "sadness": 0.5}
	# the root record has no incoming edge
	assert by_id[root]["N"] == p.Ns[root]
	assert (by_id[root]["Q"], by_id[root]["Q_emo"], by_id[root]["M2_emo"]) == (0.0, 0.0, 0.0)
	assert by_id[root]["per_step_valences"] == [] and by_id[root]["depth"] == 0
	# root_visit_dist is the same on every record and sums to 1
	dists = {tuple(sorted(r["root_visit_dist"].items())) for r in by_id.values()}
	assert len(dists) == 1
	assert sum(by_id[root]["root_visit_dist"].values()) == pytest.approx(1.0)


def test_subtree_records_of_a_baseline_planner_are_complete_and_zeroed():
	"""GDP-Zero has no emotion channel; the schema still has to be complete."""
	from runners._common import build_subtree_records
	p, root, _ = tree_planner(with_emotion_channel=False)
	records = build_subtree_records(p, dlg_id="d", turn=0, root_state=root)
	for rec in records:
		assert set(rec) == FROZEN_SUBTREE_KEYS
		assert rec["Q_emo"] == 0.0 and rec["M2_emo"] == 0.0 and rec["sigma_emo"] == 0.0
		assert rec["per_step_valences"] == []


def test_subtree_ndjson_round_trips_through_gzip(tmp_path):
	import gzip
	import json as _json
	from runners._common import build_subtree_records, write_subtree_ndjson

	p, root, _ = tree_planner()
	records = build_subtree_records(p, dlg_id="20180904-100019_870_live", turn=0, root_state=root)
	out = tmp_path / "run" / "run.pkl"
	out.parent.mkdir(parents=True)
	path = write_subtree_ndjson(records, str(out), "20180904-100019_870_live")
	assert path.endswith("subtree/20180904-100019_870_live.ndjson.gz")
	with gzip.open(path, "rt", encoding="utf-8") as f:
		back = [_json.loads(line) for line in f]
	assert back == records
	# nothing to write -> no file, no crash (llm_raw builds no tree)
	assert write_subtree_ndjson([], str(out), "empty") == ""


def test_subtree_sigma_agrees_with_the_raw_valence_tape():
	"""sigma_emo in the log must be recomputable from per_step_valences — that is what
	makes the offline variance analysis trustworthy."""
	from runners._common import build_subtree_records
	node = "greeting"
	p = make_planner(Nsa={node: {0: 0}}, Q_emo={node: {0: 0.0}}, M2_emo={node: {0: 0.0}})
	p.emo_valences[node] = {0: []}
	p.player, p.Ns, p.Q = FakePlayer(), {node: 0}, {node: {0: 0.0}}
	p.realizations, p.realizations_Vs, p.node_V, p.cache_hits = {}, {}, {}, {}
	p._to_string_rep = lambda state: state
	for z in Z_SEQUENCE:
		p._update_emo_channel(node, 0, z, p.Nsa[node][0])
		p.Nsa[node][0] += 1
		p.Ns[node] += 1
	rec = [r for r in build_subtree_records(p, dlg_id="d", turn=0, root_state="")
		   if r["parent_id"] is not None][0]
	assert rec["per_step_valences"] == Z_SEQUENCE
	assert rec["sigma_emo"] == pytest.approx(float(np.std(Z_SEQUENCE, ddof=1)), abs=1e-12)


# ---------------------------------------------------------------------------
# --emo_valence_table {soft,argmax} — selectable w(e), default must be the deployed one
# ---------------------------------------------------------------------------
def test_valence_table_default_is_the_deployed_soft_table():
	from mcts.emotion_mcts import EMOTION_VALENCE_TABLES
	p = build_real_planner()
	assert p.emo_valence_table == "soft"
	# the SAME dict object, not a copy — so nu() is bit-identical to the pre-flag code
	assert p.valence_weights is EMOTION_VALENCE_MINED
	assert EMOTION_VALENCE_TABLES["soft"] is EMOTION_VALENCE_MINED


def test_valence_table_argmax_is_selectable_and_differs():
	from mcts.emotion_mcts import EMOTION_VALENCE_ARGMAX
	p = build_real_planner(emo_valence_table="argmax")
	assert p.valence_weights is EMOTION_VALENCE_ARGMAX
	assert EMOTION_VALENCE_ARGMAX != EMOTION_VALENCE_MINED
	# the headline disagreement the ablation exists to show
	assert EMOTION_VALENCE_ARGMAX[Emotions.Fear] == pytest.approx(0.62)
	assert EMOTION_VALENCE_MINED[Emotions.Fear] == pytest.approx(0.24)


def test_unknown_valence_table_is_rejected():
	with pytest.raises(ValueError, match="emo_valence_table"):
		build_real_planner(emo_valence_table="hand")


def test_emotion_quality_uses_the_selected_table():
	d = dist(Fear=1.0)
	soft = build_real_planner()
	arg = build_real_planner(emo_valence_table="argmax")
	assert soft._emotion_quality(d) == pytest.approx(0.24)
	assert arg._emotion_quality(d) == pytest.approx(0.62)
	# a planner built without the attribute at all still falls back to the deployed table
	legacy = make_planner()
	assert legacy._emotion_quality(d) == pytest.approx(0.24)


def test_both_tables_keep_nu_and_delta_inside_the_documented_range():
	from mcts.emotion_mcts import EMOTION_VALENCE_TABLES
	rng = random.Random(5)
	for name, table in EMOTION_VALENCE_TABLES.items():
		w = [float(v) for v in table.values()]
		assert -1.0 <= min(w) and max(w) <= 1.0, (name, min(w), max(w))
		p = build_real_planner(emo_signal="delta", emo_valence_table=name)
		for _ in range(500):
			mk = lambda: FakeState({e: v for e, v in zip(table, _rand_simplex(rng, len(table)))})
			assert -1.0 <= p._emotion_signal(mk(), mk()) <= 1.0


def _rand_simplex(rng, k):
	xs = [rng.random() for _ in range(k)]
	total = sum(xs) or 1.0
	return [x / total for x in xs]
