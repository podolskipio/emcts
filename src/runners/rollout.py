"""Self-play rollouts: produce episode records that ``metrics/run_metrics.py`` can score.

Unlike the response-comparison runners (``raw_prompting.py`` / ``gdpzero*.py``), this one plays
*full* dialogs from scratch: a **rollout policy** picks the next system dialog act, the user
simulator replies, and we repeat until the game ends or the turn limit is hit. For every dialog
it records ``{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}``
(see ``metrics/dialog_metrics.py`` for the schema).

The action-selection step is **pluggable** via ``--algo``. Built-in policies (see ``POLICIES``):

  * ``llm_raw`` — single LLM call: ``argmax(planner.predict(state))`` (mirrors ``runners/raw_prompting``)
  * ``gdpzero`` — open-loop MCTS over LLM rollouts (``--num_mcts_sims`` / ``--max_realizations`` / ``--Q_0``)
  * ``emomcts`` — emotion-aware open-loop MCTS (``EmotionAwareOpenLoopMCTS``); same MCTS flags + an emotion classifier

To add a new policy, subclass :class:`RolloutPolicy` (any class with
``pick_action(state, *, game, system, planner) -> int`` works), register it in ``POLICIES``,
and — if it needs hyper-parameters — extend ``add_policy_args`` / ``make_policy``.
``rollout_one`` and ``main`` then stay untouched.

NOTE: the MCTS classes in ``mcts/mcts.py`` currently call ``game.get_next_state(state, action)``
with two positional args and ``game.get_next_state_batched(...)``, neither of which the games
implement yet — so ``--algo gdpzero`` / ``emomcts`` are wired but not yet runnable end-to-end.
``llm_raw`` works today. ``emomcts`` additionally needs an emotion-aware game producing
``EmotionAwareDialogSession`` and a real ``emotion_classifier`` (a stub is used by default).

    cd src
    python runners/rollout.py --game cb                                       # llm_raw, all CB dialogs
    python runners/rollout.py --game p4g --algo gdpzero  --num_mcts_sims 20   # GDP-Zero MCTS planning
    python runners/rollout.py --game esc --algo emomcts --num_mcts_sims 20    # Emotion-aware MCTS
    python metrics/run_metrics.py --episodes outputs/rollout.pkl --max_turns 8
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import re
import pickle
import logging
import argparse

import numpy as np
from tqdm.auto import tqdm

from runners._common import TASKS, make_backbone_model, build_agents, add_common_args, finalize_args, resolve_data_path

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------
# pluggable rollout policies
# ---------------------------------------------------------------------------
class RolloutPolicy:
	"""A policy picks one system dialog-act index per turn.

	Subclasses receive the same per-turn context (``state``, ``game``, ``system``, ``planner``)
	and return an integer action into ``system.dialog_acts``. The actual utterance is then
	realized by ``game.get_next_state(state, action)``.
	"""
	name = "base"

	def pick_action(self, state, *, game, system, planner) -> int:
		raise NotImplementedError


class LLMRawPolicy(RolloutPolicy):
	"""One-shot LLM planner: ``argmax(planner.predict(state))`` — mirrors ``runners/raw_prompting``.

	No tree search; the chat planner's prior over dialog acts is taken at face value. Cheapest
	baseline (one LLM call per turn) and the only policy currently runnable end-to-end.
	"""
	name = "llm_raw"

	def pick_action(self, state, *, game, system, planner) -> int:
		prior, _v = planner.predict(state)
		return int(np.argmax(np.asarray(prior)))


class _MCTSConfigMixin:
	"""Builds a ``dotdict`` of the MCTS hyper-parameters shared by GDP-Zero and EmoMCTS."""

	def _build_configs(self, num_mcts_sims, max_realizations, Q_0, cpuct):
		from utils.utils import dotdict
		return dotdict({
			"cpuct": cpuct,
			"num_MCTS_sims": num_mcts_sims,
			"Q_0": Q_0,
			"max_realizations": max_realizations,
		})


class GDPZeroPolicy(_MCTSConfigMixin, RolloutPolicy):
	"""Open-loop MCTS over LLM rollouts (as in ``runners/gdpzero.py``)."""
	name = "gdpzero"

	def __init__(self, num_mcts_sims=20, max_realizations=3, Q_0=0.0, cpuct=1.0):
		self.configs = self._build_configs(num_mcts_sims, max_realizations, Q_0, cpuct)

	def pick_action(self, state, *, game, system, planner) -> int:
		from mcts.mcts import OpenLoopMCTS
		dp = OpenLoopMCTS(game, planner, self.configs)
		for _ in tqdm(range(self.configs.num_MCTS_sims), leave=False, desc="gdpzero"):
			dp.search(state)
		policy = dp.get_action_prob(state)
		return int(np.argmax(np.asarray(policy)))


class _StubEmotionClassifier:
	"""Minimal placeholder for ``EmotionAwareOpenLoopMCTS``'s ``emotion_classifier`` argument.

	Only ``.emotions`` is consulted by ``emotion_mcts.py`` today (it seeds a per-node counter).
	Swap this for a real classifier (e.g. a HF text-classification pipeline) before relying on
	the emotion signal — the stub does no actual classification.
	"""
	emotions = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")

	def classify(self, _utt):  # pragma: no cover -- not exercised by the current MCTS code
		return "neutral"


class EmoMCTSPolicy(_MCTSConfigMixin, RolloutPolicy):
	"""Emotion-aware open-loop MCTS (``mcts.emotion_mcts.EmotionAwareOpenLoopMCTS``).

	Requires the game to produce ``EmotionAwareDialogSession`` states (with the per-turn
	``emotion`` slot). The classifier is injected at construction; a stub with the standard
	Ekman emotion labels is used by default — supply ``emotion_classifier=`` for real use.
	"""
	name = "emomcts"

	def __init__(self, num_mcts_sims=20, max_realizations=3, Q_0=0.0, cpuct=1.0,
				 emotion_classifier=None):
		self.configs = self._build_configs(num_mcts_sims, max_realizations, Q_0, cpuct)
		self.emotion_classifier = emotion_classifier or _StubEmotionClassifier()

	def pick_action(self, state, *, game, system, planner) -> int:
		from mcts.emotion_mcts import EmotionAwareOpenLoopMCTS
		dp = EmotionAwareOpenLoopMCTS(game, planner, self.configs, self.emotion_classifier)
		for _ in tqdm(range(self.configs.num_MCTS_sims), leave=False, desc="emomcts"):
			dp.search(state)
		policy = dp.get_action_prob(state)
		return int(np.argmax(np.asarray(policy)))


POLICIES = {
	LLMRawPolicy.name:    LLMRawPolicy,
	GDPZeroPolicy.name:   GDPZeroPolicy,
	EmoMCTSPolicy.name:   EmoMCTSPolicy,
}


def add_policy_args(parser):
	"""Wire CLI flags shared by all policies + any policy-specific hyper-parameters."""
	parser.add_argument("--algo", choices=list(POLICIES), default=LLMRawPolicy.name,
						help="rollout policy for the system side (see POLICIES in rollout.py)")
	# MCTS hyper-parameters (shared by gdpzero + emomcts)
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="[--algo gdpzero|emomcts] MCTS simulations per turn")
	parser.add_argument("--max_realizations", type=int, default=3, help="[--algo gdpzero|emomcts] realizations sampled per state")
	parser.add_argument("--Q_0", type=float, default=0.0, help="[--algo gdpzero|emomcts] initial Q value for unvisited states")
	parser.add_argument("--cpuct", type=float, default=1.0, help="[--algo gdpzero|emomcts] UCT exploration constant")


def make_policy(algo, cmd_args) -> RolloutPolicy:
	if algo == LLMRawPolicy.name:
		return LLMRawPolicy()
	if algo == GDPZeroPolicy.name:
		return GDPZeroPolicy(
			num_mcts_sims=cmd_args.num_mcts_sims,
			max_realizations=cmd_args.max_realizations,
			Q_0=cmd_args.Q_0,
			cpuct=cmd_args.cpuct,
		)
	if algo == EmoMCTSPolicy.name:
		return EmoMCTSPolicy(
			num_mcts_sims=cmd_args.num_mcts_sims,
			max_realizations=cmd_args.max_realizations,
			Q_0=cmd_args.Q_0,
			cpuct=cmd_args.cpuct,
		)
	raise ValueError(f"unknown --algo {algo!r}; choose from {list(POLICIES)}")


# ---------------------------------------------------------------------------
# rollout loop
# ---------------------------------------------------------------------------
def _extract_cb_price(state):
	"""Best-effort: the agreed price is the last number mentioned in the negotiation."""
	last = None
	for entry in state.history:
		utt = str(entry[-1]).replace("$", "")
		for tok in _NUM_RE.findall(utt):
			try:
				last = float(tok.replace(",", ""))
			except ValueError:
				pass
	return last


def rollout_one(game, system, planner, policy, max_turns, scenario):
	"""Play one full episode with ``policy`` choosing each system action; return the final session."""
	state = game.init_dialog(*scenario)
	# turn 0: the only valid move at the start is the greeting -> realize it directly
	valid0 = np.asarray(planner.get_valid_moves(state), dtype=float)
	greeting_idx = int(np.nonzero(valid0)[0][0]) if valid0.sum() > 0 else 0
	state, _ = game.get_next_state(state, greeting_idx)
	# then let the policy plan turn by turn until the game ends or we hit the limit
	while game.get_dialog_ended(state) == 0.0 and len(state) < max_turns:
		action = policy.pick_action(state, game=game, system=system, planner=planner)
		state, _ = game.get_next_state(state, action)
	return state


def make_episode(task, did, game, state, *, algo=None):
	ended = game.get_dialog_ended(state)
	episode = {
		"did": did,
		"task": task,
		"algo": algo,
		"success": bool(ended >= 1.0),
		"num_turns": len(state),
		"history": [list(t) for t in state.history],
	}
	if task == "cb":
		episode["buyer_price"] = getattr(state, "buyer_price", None)
		episode["seller_price"] = getattr(state, "seller_price", None)
		episode["deal_price"] = _extract_cb_price(state) if episode["success"] else None
	return episode


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser(description="self-play rollouts -> episode records for SR / AT / SL")
	add_common_args(parser, default_output="outputs/rollout.pkl")
	parser.add_argument("--max_turns", type=int, default=10, help="hard cap on dialog turns per episode")
	add_policy_args(parser)
	cmd_args = finalize_args(parser.parse_args())
	print(f"algo={cmd_args.algo}  saving to {cmd_args.output}")

	cfg = TASKS[cmd_args.game]
	backbone_model, family = make_backbone_model(
		cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host,
	)
	# llm_raw: keep the model's built-in inference defaults; MCTS open-loop wants sampling on
	sys_inference_args = {} if cmd_args.algo == LLMRawPolicy.name else None
	game, system, user, planner = build_agents(
		cmd_args.game, backbone_model, family,
		sys_inference_args=sys_inference_args,
	)
	policy = make_policy(cmd_args.algo, cmd_args)

	data_path = resolve_data_path(cmd_args.data or cfg.default_data)
	dialogs = cfg.read_dialogs(data_path, set(system.dialog_acts))
	print(f"task={cmd_args.game}  loaded {len(dialogs)} scenarios from {data_path}  (max_turns={cmd_args.max_turns})")

	episodes = []
	cap = len(dialogs) if cmd_args.max_conv is None or cmd_args.max_conv < 0 else cmd_args.max_conv
	n = min(cap, len(dialogs))
	pbar = tqdm(total=n, desc=f"rollout {cmd_args.game}/{cmd_args.algo}")
	for dialog in dialogs[:n]:
		did = dialog["id"]
		try:
			state = rollout_one(game, system, planner, policy, cmd_args.max_turns, dialog["scenario"])
			episodes.append(make_episode(cmd_args.game, did, game, state, algo=cmd_args.algo))
			if cmd_args.debug:
				game.display(state)
				print(f"  -> success={episodes[-1]['success']}  turns={episodes[-1]['num_turns']}")
		except Exception as e:
			logger.exception(f"rollout {did} failed: {e}")
			if cmd_args.raise_errors:
				raise
		with open(cmd_args.output, "wb") as f:
			pickle.dump(episodes, f)
		pbar.update(1)
	pbar.close()

	# quick on-the-fly summary
	try:
		from metrics.dialog_metrics import compute_metrics, format_metrics
		print(format_metrics(compute_metrics(episodes, task=cmd_args.game, max_turns=cmd_args.max_turns)))
	except Exception:
		pass
	print(f"done: {len(episodes)} episodes -> {cmd_args.output}")


if __name__ == "__main__":
	main()
