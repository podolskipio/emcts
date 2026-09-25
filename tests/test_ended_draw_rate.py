"""Readiness Phase 0: the condition --cache_ended_children must meet to remove the retention bias.

At an edge, a cache hit must draw a reply that ended the episode at the rate such replies were
generated there:

    P(draw an ended reply at an edge) = (# ended generated) / (# total generated)

Too low and Q stays biased low on success-reachable edges (the frozen cache draws them at rate 0);
too high and the fix overshoots. The edge here is filled through the planner's own generation path
with 3 non-terminal and 2 ended replies, then drawn 10,000 times: the ended fraction must be 0.40 +- 0.02.

    python -m pytest tests/test_ended_draw_rate.py -q
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_wed_arms as W  # noqa: E402  (also puts src/ on sys.path)

from games import PersuasionGame, EmotionAwarePersuasionGame  # noqa: E402
from mcts.mcts import OpenLoopMCTS  # noqa: E402
from utils.utils import dotdict  # noqa: E402

SYS_ACTS = W.SYS_ACTS
PATTERN = ["live", "ended", "live", "ended", "live"]  # 3 non-terminal, 2 ended
N_DRAWS = 10_000


class PatternUser:
	"""Replies follow PATTERN once, then keep talking: 'ended' donates, which ends the search."""
	def __init__(self):
		self.n = itertools.count()

	def get_utterance_w_da(self, state, _, mode="train"):
		i = next(self.n)
		if i < len(PATTERN) and PATTERN[i] == "ended":
			return PersuasionGame.U_Donate, f"Fine, I will donate. ({i})"
		return PersuasionGame.U_Neutral, f"user line {i}"


def _cfg(ended_children):
	# max_realizations = 5 so the edge caches exactly when all five generations are in
	return dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": len(PATTERN), "search_horizon": "episode",
					"cache_ended_children": ended_children})


def _fill_edge(p, root, action):
	"""Generate len(PATTERN) replies at root --action--> child the way search does: generate, then
	enter the reply (a non-terminal one joins the pool on entry, an ended one returns before it)."""
	for _ in PATTERN:
		child = p._get_next_state(root, action)
		p.search(child)


def _ended_fraction(p, game, draw):
	np.random.seed(123)
	return np.mean([game.get_dialog_ended(draw()) == 1.0 for _ in range(N_DRAWS)])


def _task_planner(ended_children):
	np.random.seed(0)
	game = PersuasionGame(W.FakeSystem(), PatternUser(), None, False, max_conv_turns=W.T)
	root = game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting))
	p = OpenLoopMCTS(game, W.FakePlanner(), _cfg(ended_children))
	p.search(root)  # expands the root and pins it as the search root
	return p, game, root


def test_gdpzero_draws_ended_replies_at_their_generation_rate():
	p, game, root = _task_planner(ended_children=True)
	a = SYS_ACTS.index(PersuasionGame.S_CredibilityAppeal)
	_fill_edge(p, root, a)
	key = p._to_string_rep(root) + "__" + SYS_ACTS[a]
	assert len(p.realizations[key]) == 3 and len(p.ended_realizations[key]) == 2
	assert p._serves_from_cache(p._to_string_rep(root), key)
	frac = _ended_fraction(p, game, lambda: p._get_next_state(root, a))
	assert frac == pytest.approx(0.40, abs=0.02)
	# every one of the draws was a cache hit: nothing extra generated
	assert p.cache_hits[key] == [N_DRAWS, len(PATTERN)]


def test_emomcts_draws_ended_replies_at_their_generation_rate():
	np.random.seed(0)
	clf = W.HashClassifier()
	game = EmotionAwarePersuasionGame(W.FakeSystem(), PatternUser(), None, False, clf, max_conv_turns=W.T)
	root = game.state_of(game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting)))
	p = W.em.EmotionAwareMultiObjectiveQ(game, W.FakePlanner(), _cfg(True), clf)
	p.search(root)
	a = SYS_ACTS.index(PersuasionGame.S_CredibilityAppeal)
	_fill_edge(p, root, a)
	key = p._to_string_rep(root) + "__" + SYS_ACTS[a]
	assert len(p.realizations[key]) == 3 and len(p.ended_realizations[key]) == 2
	# the hit path EmotionAwareMultiObjectiveQ.search takes, under the default uniform draw
	frac = _ended_fraction(p, game, lambda: p._draw_cached_child(key, 0.0)[0])
	assert frac == pytest.approx(0.40, abs=0.02)


def test_frozen_cache_never_draws_an_ended_reply():
	"""The bias the fix removes: without the flag the same edge caches only its live replies, so a
	hit draws an ended one at rate 0 against a generation rate of 0.4."""
	p, game, root = _task_planner(ended_children=False)
	a = SYS_ACTS.index(PersuasionGame.S_CredibilityAppeal)
	_fill_edge(p, root, a)
	key = p._to_string_rep(root) + "__" + SYS_ACTS[a]
	assert p.ended_realizations == {} and len(p.realizations[key]) == 3
	# 3 < max_realizations, so the frozen edge keeps generating; fill it the rest of the way
	while not p._serves_from_cache(p._to_string_rep(root), key):
		p.search(p._get_next_state(root, a))
	assert _ended_fraction(p, game, lambda: p._get_next_state(root, a)) == 0.0
