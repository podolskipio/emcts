"""--search_horizon: does tree search stop where a real episode stops?

The bug (analysis/phase1/SEARCH_HORIZON_BUG.md): every search path treated only success as
terminal, so a branch that hit the turn limit (get_dialog_ended == -1.0) kept being expanded
and valued -- the tree evaluated dialogue states no episode can reach. `--search_horizon
episode` stops on any non-zero get_dialog_ended; `legacy` (default) must stay the old rule.

These tests drive the REAL games (`get_dialog_ended`, `get_next_state`, the stall detector) and
the real planners' `search`; only the text sources and the LLM planner are fakes.

    python -m pytest tests/test_search_horizon.py -q
"""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402
from games import PersuasionGame, EmotionAwarePersuasionGame  # noqa: E402
from mcts.emotion_mcts import EmotionAwareMultiObjectiveQ  # noqa: E402
from mcts.mcts import MCTS, OpenLoopMCTS, SEARCH_HORIZONS  # noqa: E402
from utils.utils import dotdict  # noqa: E402

SYS_ACTS = PersuasionGame.get_game_ontology()["system"]["dialog_acts"]
T = 4  # max_conv_turns for every test game


# ---------------------------------------------------------------------------
# fakes: text sources and planner only
# ---------------------------------------------------------------------------
class FakeSystem:
	"""repeat: the same line every time (drives the stall detector). deterministic: a pure function
	of (turn, act), so the closed-loop planner -- which keys nodes by the full text -- revisits
	nodes. Otherwise every call is a new line, giving the open-loop pools distinct realizations."""
	dialog_acts = SYS_ACTS

	def __init__(self, repeat=False, deterministic=False):
		self.repeat, self.deterministic, self.n = repeat, deterministic, itertools.count()

	def get_utterance(self, state, action):
		if self.repeat:
			return "Please donate."
		return f"system {len(state)} act {action}" if self.deterministic else f"system line {next(self.n)}"


class FakeUser:
	def __init__(self, repeat=False, donate_after=None, deterministic=False):
		self.repeat, self.donate_after, self.deterministic = repeat, donate_after, deterministic
		self.n = itertools.count()

	def get_utterance_w_da(self, state, _, mode="train"):
		i = next(self.n)
		if self.donate_after is not None and i >= self.donate_after:
			return PersuasionGame.U_Donate, "Fine, I will donate."
		if self.repeat:
			return PersuasionGame.U_Neutral, "I see."
		return PersuasionGame.U_Neutral, (f"user {len(state.history)}" if self.deterministic else f"user line {i}")


class FakeClassifier:
	emotions = list(Emotions)

	def __init__(self):
		self.records = []

	def predict_distribution_from_full_history(self, state, utt):
		return {e: (1.0 if e == Emotions.Happiness else 0.0) for e in Emotions}


class FakePlanner:
	"""Uniform prior over 3 acts, constant leaf value, no top-K pruning."""
	dialog_acts = SYS_ACTS
	llm_prior_topk = None

	def __init__(self, value=0.2):
		self.value = value

	def get_valid_moves(self, state):
		m = np.zeros(len(SYS_ACTS))
		m[:3] = 1
		return m

	def predict(self, state):
		return np.ones(len(SYS_ACTS)) / len(SYS_ACTS), self.value


class Recorder:
	"""Wraps game.get_next_state to record the length of every state search transitions FROM."""

	def __init__(self, game):
		self.lens = []
		inner = game.get_next_state

		def wrapped(state, action, *a, **k):
			self.lens.append(len(state))
			return inner(state, action, *a, **k)
		game.get_next_state = wrapped


def configs(horizon, **extra):
	c = {"cpuct": 1.0, "Q_0": 0.0, "max_realizations": 2, **extra}
	if horizon is not None:
		c["search_horizon"] = horizon
	return dotdict(c)


def emo_game(repeat=False, donate_after=None):
	return EmotionAwarePersuasionGame(FakeSystem(repeat), FakeUser(repeat, donate_after), None, False,
									  FakeClassifier(), max_conv_turns=T)


def root_after_greeting(game):
	return game.state_of(game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting)))


def run_emo(horizon, sims=80, **game_kw):
	np.random.seed(0)
	game = emo_game(**game_kw)
	root = root_after_greeting(game)
	rec = Recorder(game)
	p = EmotionAwareMultiObjectiveQ(game, FakePlanner(), configs(horizon), FakeClassifier(), beta_emo=0.7)
	for _ in range(sims):
		p.search(root)
	return p, root, rec


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------
def test_choices_and_default():
	assert SEARCH_HORIZONS == ("legacy", "episode")
	p = OpenLoopMCTS(None, FakePlanner(), configs(None))  # a configs dict without the key
	assert p.search_horizon == "legacy"


def test_unknown_horizon_is_rejected():
	with pytest.raises(ValueError, match="search_horizon"):
		OpenLoopMCTS(None, FakePlanner(), configs("tmax"))


def test_legacy_rule_is_exactly_the_prechange_comparison():
	legacy = OpenLoopMCTS(None, FakePlanner(), configs("legacy"))
	episode = OpenLoopMCTS(None, FakePlanner(), configs("episode"))
	for v in (1.0, -1.0, 0.0, -0.0, 0.5, -0.5, 1.0000001, float("inf")):
		assert legacy._ends_search(v) == (v == 1.0)
		assert episode._ends_search(v) == (v != 0.0)


# ---------------------------------------------------------------------------
# emotion-aware open loop (the planner the grid runs)
# ---------------------------------------------------------------------------
def test_legacy_search_goes_past_the_turn_limit():
	"""Documents the bug: under legacy, search transitions out of states at or beyond T."""
	_, _, rec = run_emo("legacy")
	assert max(rec.lens) >= T


def test_episode_search_never_transitions_from_a_turn_limit_state():
	p, root, rec = run_emo("episode")
	assert max(rec.lens) == T - 1          # it does reach the horizon...
	assert all(n < T for n in rec.lens)    # ...and never simulates past it
	# every node the tree expanded is a reachable, non-terminal state
	assert all(len(key.split("__")) < T for key in p.P)


def test_episode_backs_up_the_failure_value():
	"""With a constant +0.2 leaf value, a -1 only enters Q through the horizon."""
	p_ep, root, _ = run_emo("episode")
	p_leg, _, _ = run_emo("legacy")
	key = p_ep._to_string_rep(root)
	assert min(p_ep.Q[key].values()) < 0.0
	assert min(p_leg.Q[key].values()) >= 0.0


def test_episode_stops_on_a_verbatim_stall():
	# every line repeats, so the first child of the 1-turn root (len 2) is already a verbatim stall:
	# episode only ever transitions out of the root, legacy keeps simulating the loop
	p, root, rec = run_emo("episode", repeat=True)
	assert len(root) == 1 and not p.game.is_stalled(root)
	assert set(rec.lens) == {1}
	_, _, rec_leg = run_emo("legacy", repeat=True)
	assert max(rec_leg.lens) > 2


def test_donation_is_terminal_under_both_rules():
	for horizon in SEARCH_HORIZONS:
		p, root, rec = run_emo(horizon, donate_after=1)  # every simulated user reply donates
		assert all(n == len(root) for n in rec.lens)     # no transition out of a donated state
		key = p._to_string_rep(root)
		assert max(p.Q[key].values()) == pytest.approx(1.0)


def test_emo_channel_still_backs_up_the_edge_into_a_terminal_child():
	p, root, _ = run_emo("episode")
	for node, acts in p.Nsa.items():
		for a, n in acts.items():
			assert len(p.emo_valences[node][a]) == n


# ---------------------------------------------------------------------------
# plain open loop (GDP-Zero) and closed loop (gdpzero_noopenloop)
# ---------------------------------------------------------------------------
def plain_game(deterministic):
	return PersuasionGame(FakeSystem(deterministic=deterministic), FakeUser(deterministic=deterministic),
						  None, False, max_conv_turns=T)


@pytest.mark.parametrize("cls", [OpenLoopMCTS, MCTS])
def test_plain_planners_respect_the_horizon(cls):
	for horizon, past_allowed in (("legacy", True), ("episode", False)):
		np.random.seed(0)
		game = plain_game(deterministic=(cls is MCTS))
		root = game.get_next_state(game.init_dialog(), SYS_ACTS.index(PersuasionGame.S_Greeting))
		rec = Recorder(game)
		p = cls(game, FakePlanner(), configs(horizon))
		for _ in range(80):
			p.search(root)
		assert (max(rec.lens) >= T) == past_allowed, (cls.__name__, horizon, max(rec.lens))


if __name__ == "__main__":
	sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Tmax is one flag for every runner (FREEZE_NOTES §9)
# ---------------------------------------------------------------------------
def test_max_turns_is_shared_and_reaches_the_game():
	"""--max_turns lives in add_common_args, so no runner can silently inherit the game's 15."""
	import argparse
	from runners._common import add_common_args

	p = add_common_args(argparse.ArgumentParser(), default_output="x.pkl")
	assert p.parse_args([]).max_turns == 10
	assert p.parse_args(["--max_turns", "7"]).max_turns == 7
	# and every runner hands it to the game rather than leaving the default in place
	import pathlib
	src = pathlib.Path(__file__).resolve().parents[1] / "src" / "runners"
	for name in ("rollout", "gdpzero", "emomcts", "gdpzero_noRS", "gdpzero_noopenloop", "raw_prompting"):
		assert "max_conv_turns=cmd_args.max_turns" in (src / f"{name}.py").read_text(), name


def test_replay_root_past_the_horizon_is_skipped_not_valued():
	"""A corpus prefix at/past Tmax: search cannot expand it, so get_action_prob is all-NaN.
	replay_root_is_terminal is what keeps the replay runners off that state -- under `episode`
	only; `legacy` replay output must be untouched."""
	from runners._common import replay_root_is_terminal

	game = PersuasionGame(FakeSystem(), FakeUser(), None, False, max_conv_turns=T)
	state = game.init_dialog()
	for i in range(T + 2):  # a real dialogue longer than the horizon, as p4g's are
		state.add_single(game.SYS, SYS_ACTS[0], f"sys {i}")
		state.add_single(game.USR, PersuasionGame.U_Neutral, f"usr {i}")

	assert game.get_dialog_ended(state) == -1.0
	assert replay_root_is_terminal(game, state, "episode") is True
	assert replay_root_is_terminal(game, state, "legacy") is False   # legacy unchanged

	# what the guard prevents: an unexpanded root gives NaN, and argmax then "picks" act 0
	p = OpenLoopMCTS(game, FakePlanner(), configs("episode"))
	for _ in range(20):
		p.search(state)
	assert p._to_string_rep(state) not in p.Ns          # never expanded
	assert np.isnan(np.asarray(p.get_action_prob(state))).all()

	# a root inside the horizon is planned normally
	short = game.init_dialog()
	short.add_single(game.SYS, SYS_ACTS[0], "sys")
	short.add_single(game.USR, PersuasionGame.U_Neutral, "usr")
	assert replay_root_is_terminal(game, short, "episode") is False
