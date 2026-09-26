"""--coupled_seeds (utils/coupling.py): common random numbers across arms.

  * off by default: no dialogue context -> every call is a pass-through, the planner draws from np.random
  * the same (seed, dialogue, request, occurrence) returns the same reply, from any run sharing the store
  * each occurrence of a repeated request is its own draw; a failed request does not use one up
  * the per-turn planner RandomState makes a search reproducible whatever the global numpy stream does

    python -m pytest tests/test_coupling.py -q
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_wed_arms as W  # noqa: E402  (also puts src/ on sys.path)

from utils import coupling  # noqa: E402
from mcts.mcts import OpenLoopMCTS  # noqa: E402
from utils.utils import dotdict  # noqa: E402


def counter_generator():
	n = itertools.count()
	return lambda: [{"generated_text": f"reply {next(n)}"}]


def test_no_context_is_a_pass_through():
	gen = counter_generator()
	assert not coupling.active()
	assert coupling.call({"m": 1}, gen) == [{"generated_text": "reply 0"}]
	assert coupling.call({"m": 1}, gen) == [{"generated_text": "reply 1"}]
	assert coupling.planner_rng(3) is None


def test_default_planner_draws_from_the_numpy_module():
	assert OpenLoopMCTS(None, W.FakePlanner(), dotdict({"max_realizations": 2})).rng is np.random


def test_same_key_same_reply_across_runs(tmp_path):
	store = coupling.ReplyStore(str(tmp_path / "s.sqlite"))
	arm_a, arm_b = counter_generator(), counter_generator()
	with coupling.dialogue(store, 1, "d1"):
		a = [coupling.call({"m": "p"}, arm_a) for _ in range(3)] + [coupling.call({"m": "q"}, arm_a)]
	with coupling.dialogue(store, 1, "d1") as ctx:
		b = [coupling.call({"m": "p"}, arm_b) for _ in range(3)] + [coupling.call({"m": "q"}, arm_b)]
		assert (ctx.hits, ctx.misses) == (4, 0)
	assert a == b
	# repeated requests are separate draws, not one reply replayed
	assert len({r[0]["generated_text"] for r in a[:3]}) == 3


def test_seed_and_dialogue_are_part_of_the_key(tmp_path):
	store = coupling.ReplyStore(str(tmp_path / "s.sqlite"))
	gen = counter_generator()
	with coupling.dialogue(store, 1, "d1"):
		x = coupling.call({"m": "p"}, gen)
	with coupling.dialogue(store, 2, "d1"):
		y = coupling.call({"m": "p"}, gen)
	with coupling.dialogue(store, 1, "d2"):
		z = coupling.call({"m": "p"}, gen)
	assert len({str(x), str(y), str(z)}) == 3


def test_failed_request_does_not_consume_an_occurrence(tmp_path):
	store = coupling.ReplyStore(str(tmp_path / "s.sqlite"))
	with coupling.dialogue(store, 1, "d1"):
		first = coupling.call({"m": "p"}, lambda: ["first"])
	calls = iter([RuntimeError("boom"), ["second"]])

	def flaky():
		v = next(calls)
		if isinstance(v, Exception):
			raise v
		return v
	with coupling.dialogue(store, 1, "d1"):
		assert coupling.call({"m": "p"}, lambda: ["never"]) == first     # occurrence 0: stored
		with pytest.raises(RuntimeError):
			coupling.call({"m": "p"}, flaky)                               # occurrence 1 fails ...
		assert coupling.call({"m": "p"}, flaky) == ["second"]              # ... and is retried as 1
	with coupling.dialogue(store, 1, "d1"):
		coupling.call({"m": "p"}, lambda: ["x"])
		assert coupling.call({"m": "p"}, lambda: ["y"]) == ["second"]      # a re-run sees the same occurrence 1


def test_run_in_carries_the_context_into_worker_threads(tmp_path):
	from multiprocessing.pool import ThreadPool
	store = coupling.ReplyStore(str(tmp_path / "s.sqlite"))
	with coupling.dialogue(store, 1, "d1") as ctx:
		pool = ThreadPool(2)
		assert pool.apply(coupling.run_in, (ctx, coupling.active)) is True
		assert pool.apply(coupling.active) is False
		pool.close()


def test_planner_rng_makes_the_search_reproducible(tmp_path):
	store = coupling.ReplyStore(str(tmp_path / "s.sqlite"))

	def search(global_seed):
		np.random.seed(global_seed)  # the shared stream other workers would be advancing
		with coupling.dialogue(store, 7, "d1"):
			p, _, _ = W_search(rng=coupling.planner_rng(turn=2))
		return W.fingerprint(p)
	assert search(0) == search(12345)
	with coupling.dialogue(store, 7, "d1"):
		assert coupling.planner_rng(2).randint(10**9) == coupling.planner_rng(2).randint(10**9)
		assert coupling.planner_rng(2).randint(10**9) != coupling.planner_rng(3).randint(10**9)


def W_search(rng, sims=120):
	"""test_wed_arms.run_search with the planner's RNG swapped in before any search."""
	from games import PersuasionGame, EmotionAwarePersuasionGame
	clf = W.HashClassifier()
	game = EmotionAwarePersuasionGame(W.FakeSystem(), W.FakeUser(), None, False, clf, max_conv_turns=W.T)
	root = game.state_of(game.get_next_state(game.init_dialog(), W.SYS_ACTS.index(PersuasionGame.S_Greeting)))
	c = dotdict({"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 2, "search_horizon": "legacy"})
	p = W.em.EmotionAwareMultiObjectiveQ(game, W.FakePlanner(), c, clf)
	p.rng = rng
	for _ in range(sims):
		p.search(root)
	return p, game, root


def test_runner_flags_default_off_and_store_needs_the_flag():
	import test_thu_arms as TA
	parser, mod = TA._runner_parser("rollout")
	a = parser.parse_args([])
	assert a.coupled_seeds is False and a.coupling_store is None
	bad = ["--coupling_store", "/tmp/x.sqlite"]
	with pytest.raises(SystemExit, match="coupling_store needs --coupled_seeds"):
		mod.finalize_args(parser.parse_args(bad), argv=bad)
