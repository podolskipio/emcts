"""Cache fixes (analysis/phase1/p_depth.md). Same contract as tests/test_wed_arms.py: every behaviour
sits behind a flag whose default is the frozen planner, so the first job is bit-identity.

  * --cache_ended_children  a generated reply that ends the search is filed under its node at
                            generation time, so the cache can serve it. It is never sampled as a parent.
  * --cache_fresh_depth1    the root's outgoing edges always generate; their child pools are uncapped.
  * --cache_draw            bucket / kernel / bucket_kernel: a cache hit prefers replies generated under
                            a parent whose nu is close to the current parent's. No extra generation.

    python -m pytest tests/test_cache_fixes.py -q
"""
import dataclasses
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_wed_arms as W  # noqa: E402  (also puts src/ on sys.path)

from games import PersuasionGame, EmotionAwarePersuasionGame  # noqa: E402
from mcts.emotion_mcts import CACHE_DRAWS, _realization_id  # noqa: E402
from mcts.mcts import OpenLoopMCTS  # noqa: E402
from utils.utils import dotdict  # noqa: E402

em = W.em
SYS_ACTS = W.SYS_ACTS
TAU = 0.0  # HashClassifier's nu straddles 0 under the soft table


class SometimesDonateUser:
	"""Every third reply donates (ends the search under both horizons); the rest are distinct lines."""
	def __init__(self):
		self.n = itertools.count()

	def get_utterance_w_da(self, state, _, mode="train"):
		i = next(self.n)
		if i % 3 == 2:
			return PersuasionGame.U_Donate, f"Fine, I will donate. ({i})"
		return PersuasionGame.U_Neutral, f"user line {i}"


def emo_search(sims=120, seed=0, user=None, horizon="legacy", max_realizations=2, cfg=None, **planner_kw):
	"""W.run_search with the cache configs exposed. Returns (planner, game, root)."""
	np.random.seed(seed)
	clf = W.HashClassifier()
	game = EmotionAwarePersuasionGame(W.FakeSystem(), user or W.FakeUser(), None, False, clf, max_conv_turns=W.T)
	root = game.state_of(game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting)))
	c = dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": max_realizations, "search_horizon": horizon,
				 **(cfg or {})})
	p = em.EmotionAwareMultiObjectiveQ(game, W.FakePlanner(), c, clf, **planner_kw)
	for _ in range(sims):
		p.search(root)
	return p, game, root


def forbid_generation_from_ended(p, game):
	"""Wrap game.get_next_state to fail if search ever generates from a state that ended the search."""
	inner = game.get_next_state

	def wrapped(state, action, *a, **k):
		assert not p._ends_search(game.get_dialog_ended(state)), "generated past the end of an episode"
		return inner(state, action, *a, **k)
	game.get_next_state = wrapped


# ---------------------------------------------------------------------------
# bit-identity with every flag off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_explicit_defaults_reproduce_the_frozen_tree(beta):
	cfg = {"cache_ended_children": False, "cache_fresh_depth1": False}
	p, _, _ = emo_search(beta_emo=beta, cfg=cfg, cache_draw="uniform")
	assert W.fingerprint(p) == W.GOLDEN[f"beta{beta}"]
	assert p.ended_realizations == {} and p._uncapped_pools == set()
	assert not any("cache_draw_prob" in s or "child_generating_parent_nu" in s for s in p.sim_steps)


def test_constructor_and_config_defaults_are_off():
	p, _, _ = emo_search(sims=1)
	assert p.cache_ended_children is False and p.cache_fresh_depth1 is False and p.cache_draw == "uniform"
	assert OpenLoopMCTS(None, W.FakePlanner(), dotdict({"max_realizations": 2})).cache_ended_children is False


# ---------------------------------------------------------------------------
# --cache_ended_children
# ---------------------------------------------------------------------------
def test_ended_replies_never_reach_the_pool_without_the_flag():
	p, game, _ = emo_search(user=SometimesDonateUser())
	assert p.ended_realizations == {}
	for pool in p.realizations.values():
		assert all(game.get_dialog_ended(r) != 1.0 for r in pool)


@pytest.mark.parametrize("horizon", ["legacy", "episode"])
def test_ended_replies_are_filed_served_and_never_expanded(horizon):
	p, game, root = emo_search(sims=0, user=SometimesDonateUser(), horizon=horizon, cfg={"cache_ended_children": True})
	forbid_generation_from_ended(p, game)
	for _ in range(200):
		p.search(root)
	ended_ids = {_realization_id(r) for rs in p.ended_realizations.values() for r in rs}
	assert ended_ids, "no reply ended the search"
	for key, rs in p.ended_realizations.items():
		assert all(p._ends_search(game.get_dialog_ended(r)) for r in rs)
		assert len(p._cached_children(key)) <= p.max_realizations
	# the parent pools stay free of them, and the cache now serves them
	for pool in p.realizations.values():
		assert not any(_realization_id(r) in ended_ids for r in pool)
	served = [s for s in p.sim_steps if s["child_from_cache"] and s["child_realization_id"] in ended_ids]
	assert served, "an ended reply was never served from the cache"
	assert all(s["v"] == game.get_dialog_ended(r) for s in served
			   for r in [next(r for rs in p.ended_realizations.values() for r in rs
							  if _realization_id(r) == s["child_realization_id"])])


def test_ended_children_on_the_task_only_planner():
	np.random.seed(0)
	game = PersuasionGame(W.FakeSystem(), SometimesDonateUser(), None, False, max_conv_turns=W.T)
	root = game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting))
	p = OpenLoopMCTS(game, W.FakePlanner(), dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 2,
													 "search_horizon": "legacy", "cache_ended_children": True}))
	forbid_generation_from_ended(p, game)
	for _ in range(150):
		p.search(root)
	assert p.ended_realizations
	assert sum(h for h, _ in p.cache_hits.values()) > 0


# ---------------------------------------------------------------------------
# --cache_fresh_depth1
# ---------------------------------------------------------------------------
def test_depth1_always_generates_and_keeps_every_reply(caplog):
	p, _, _ = emo_search(sims=150, cfg={"cache_fresh_depth1": True})
	d1 = [s for s in p.sim_steps if s["depth"] == 1]
	deeper = [s for s in p.sim_steps if s["depth"] >= 2]
	assert d1 and not any(s["child_from_cache"] for s in d1)
	assert any(s["child_from_cache"] for s in deeper), "deeper edges must still cache"
	root = p._sim_root_key
	for a in p.valid_moves[root]:
		child = root + "__" + SYS_ACTS[a]
		if p.Nsa[root][a] > p.max_realizations + 1:
			assert child in p._uncapped_pools
			assert len(p.realizations[child]) > p.max_realizations
	assert all(len(v) <= p.max_realizations for k, v in p.realizations.items() if k not in p._uncapped_pools)
	assert "len(self.realizations" not in caplog.text


def test_depth1_counts_match_generation():
	p, _, _ = emo_search(sims=150, cfg={"cache_fresh_depth1": True})
	root = p._sim_root_key
	for a in p.valid_moves[root]:
		child = root + "__" + SYS_ACTS[a]
		if p.Nsa[root][a]:
			assert p.cache_hits[child] == [0, p.Nsa[root][a]]


# ---------------------------------------------------------------------------
# --cache_draw
# ---------------------------------------------------------------------------
def test_every_draw_is_tagged_with_its_generating_parent():
	p, _, _ = emo_search(cache_draw="bucket", cache_bucket_tau=TAU)
	for s in p.sim_steps:
		assert "child_generating_parent_nu" in s
		if not s["child_from_cache"]:
			assert s["child_generating_parent_nu"] == s["parent_nu"]
	for key, pool in p.realizations.items():
		if key != p._sim_root_key:
			assert all(_realization_id(r) in p.generating_parent_nu for r in pool)


def test_bucket_draw_matches_the_parent_bucket_unless_it_misses():
	p, _, _ = emo_search(sims=300, cache_draw="bucket", cache_bucket_tau=TAU)
	hits = [s for s in p.sim_steps if s["child_from_cache"]]
	assert hits
	matched = [s for s in hits if not s["cache_draw_bucket_miss"]]
	assert matched
	for s in matched:
		assert (s["child_generating_parent_nu"] < TAU) == (s["parent_nu"] < TAU)
	# the uniform draw on the same seed mismatches far more often
	u, _, _ = emo_search(sims=300, cache_draw="kernel", cache_kernel_h=100.0)  # flat kernel = uniform weights
	mis = lambda steps: np.mean([(s["child_generating_parent_nu"] < TAU) != (s["parent_nu"] < TAU)
								 for s in steps if s["child_from_cache"]])
	assert mis(p.sim_steps) < mis(u.sim_steps)


def test_narrow_kernel_picks_the_nearest_generating_parent():
	p, _, _ = emo_search(sims=300, cache_draw="kernel", cache_kernel_h=1e-3)
	hits = [s for s in p.sim_steps if s["child_from_cache"]]
	assert hits
	for s in hits:
		key = "__".join(W.em._node_das(p._sim_root_key) + s["action_prefix"] + [s["action"]])
		gen = [p.generating_parent_nu[_realization_id(c)] for c in p._cached_children(key)]
		assert abs(s["child_generating_parent_nu"] - s["parent_nu"]) == pytest.approx(
			min(abs(g - s["parent_nu"]) for g in gen))


def test_draw_probabilities_are_the_kernel_weights():
	p, _, root = emo_search(sims=0, cache_draw="bucket_kernel", cache_bucket_tau=TAU, cache_kernel_h=0.3)
	np.random.seed(1)
	key = "k"
	kids = []
	for i, nu in enumerate([-0.5, -0.1, 0.2, 0.4]):
		s = root.copy()
		s.history[-1] = dataclasses.replace(s.history[-1], utt=f"variant {i}")
		kids.append(s)
		p.generating_parent_nu[_realization_id(s)] = nu
	p.realizations[key] = kids
	counts = np.zeros(4)
	for _ in range(4000):
		c, info = p._draw_cached_child(key, 0.3)
		counts[kids.index(c)] += 1
	w = np.array([0.0, 0.0, np.exp(-(0.1 / 0.3) ** 2), np.exp(-(0.1 / 0.3) ** 2)])  # bucket (nu >= 0), then kernel
	assert counts[:2].sum() == 0
	assert counts / counts.sum() == pytest.approx(w / w.sum(), abs=0.03)


@pytest.mark.parametrize("kw", [
	{"cache_draw": "nearest"},
	{"cache_draw": "bucket"},
	{"cache_draw": "kernel"},
	{"cache_draw": "kernel", "cache_kernel_h": 0.0},
	{"cache_draw": "bucket_kernel", "cache_kernel_h": 0.3},
])
def test_bad_cache_draw_settings_are_rejected(kw):
	with pytest.raises(ValueError):
		emo_search(sims=1, **kw)


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("runner", ["rollout", "emomcts", "gdpzero"])
def test_runner_cache_flags_default_off(runner):
	import test_thu_arms as TA
	parser, _ = TA._runner_parser(runner)
	a = parser.parse_args([])
	assert a.cache_ended_children is False and a.cache_fresh_depth1 is False
	if runner != "gdpzero":
		assert a.cache_draw == "uniform" and a.cache_bucket_tau is None and a.cache_kernel_h is None
		assert set(CACHE_DRAWS) == {"uniform", "bucket", "kernel", "bucket_kernel"}


def test_inert_cache_flags_are_refused():
	import test_thu_arms as TA
	parser, mod = TA._runner_parser("rollout")
	ok = ["--algo", "emomcts", "--cache_draw", "bucket_kernel", "--cache_bucket_tau", "0.263", "--cache_kernel_h", "0.2"]
	mod.finalize_args(parser.parse_args(ok + ["--output", "/tmp/x/o.pkl"]), argv=ok)
	for bad, why in [
		(["--algo", "gdpzero", "--cache_draw", "bucket"], "only --algo emomcts"),
		(["--algo", "emomcts", "--cache_draw", "kernel", "--cache_bucket_tau", "0.2"], "cache_bucket_tau needs"),
		(["--algo", "emomcts", "--cache_draw", "bucket", "--cache_kernel_h", "0.2"], "cache_kernel_h needs"),
	]:
		with pytest.raises(SystemExit, match=why):
			mod.finalize_args(parser.parse_args(bad), argv=bad)


# ---------------------------------------------------------------------------
# step log: child_ends_search (write-only, readiness Phase 2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ended_children", [False, True])
def test_child_ends_search_marks_ended_replies_and_none_is_ever_a_parent(ended_children):
	p, game, _ = emo_search(sims=200, user=SometimesDonateUser(), horizon="episode",
							cfg={"cache_ended_children": ended_children})
	assert all("child_ends_search" in s for s in p.sim_steps)
	ended = {s["child_realization_id"] for s in p.sim_steps if s["child_ends_search"]}
	assert ended, "no reply ended the search"
	assert not ended & {s["parent_realization_id"] for s in p.sim_steps}
	for s in p.sim_steps:
		if s["child_ends_search"]:
			assert s["v"] in (1.0, -1.0)
	# served ended replies exist only with the fix
	served_ended = [s for s in p.sim_steps if s["child_ends_search"] and s["child_from_cache"]]
	assert bool(served_ended) is ended_children
