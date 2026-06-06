"""Self-play rollouts: produce episode records that ``metrics/run_metrics.py`` can score.

Unlike the response-comparison runners (``raw_prompting.py`` / ``gdpzero*.py``), this one plays
*full* dialogs from scratch: the chosen ``--algo`` picks the next system dialog act, the user
simulator replies, and we repeat until the game ends or the turn limit is hit. For every dialog
it records ``{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}``
(see ``metrics/dialog_metrics.py`` for the schema).

``--algo`` selects how each system action is picked (see :func:`pick_action`):

  * ``llm_raw`` — single LLM call: ``argmax(planner.predict(state))`` (mirrors ``runners/raw_prompting``)
  * ``gdpzero`` — open-loop MCTS over LLM rollouts (``--num_mcts_sims`` / ``--max_realizations`` / ``--Q_0`` / ``--cpuct``)
  * ``emomcts`` — emotion-aware open-loop MCTS (``EmotionAwareOpenLoopMCTS``); needs an emotion-aware task (``--game emo_p4g``)

    cd src
    python runners/rollout.py --game cb                                          # llm_raw, all CB dialogs
    python runners/rollout.py --game p4g     --algo gdpzero --num_mcts_sims 20    # GDP-Zero MCTS planning
    python runners/rollout.py --game emo_p4g --algo emomcts --num_mcts_sims 20    # Emotion-aware MCTS
    python metrics/run_metrics.py --episodes outputs/rollout.pkl --max_turns 8

NOTE: ``llm_raw`` and ``gdpzero`` run end-to-end against the current games (their
``get_next_state(state, action[, mode])`` accepts 2-arg calls). ``emomcts`` requires a game that
produces ``EmotionAwareDialogSession`` states and has an emotion classifier attached (the
``emo_*`` tasks in ``_common.py``); pointing it at a plain task raises a clear error.
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

from utils.utils import dotdict
from mcts.mcts import OpenLoopMCTS
from mcts.emotion_mcts import EmotionAwareMultiObjectiveQ
from runners._common import TASKS, make_backbone_model, build_agents, load_dialogs, dump_emotion_records, add_common_args, finalize_args, setup_output_dir

logger = logging.getLogger(__name__)

ALGOS = ("llm_raw", "gdpzero", "emomcts")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------
# action selection (one system dialog-act index per turn)
# ---------------------------------------------------------------------------
def pick_action(algo, state, *, game, planner, configs, emotion_classifier,
				mcts_cls=None, mcts_kwargs=None) -> int:
	"""Pick the next system dialog-act index for ``state`` using ``algo``.

	For ``algo == "emomcts"`` the caller supplies the concrete subclass via
	``mcts_cls`` (one of EmotionAwareOpenLoopMCTS / EmotionAwareDiscountQOpenLoopMCTS
	/ EmotionGuidedDiscountQOpenLoopMCTS / EmotionRealizationSelectorMCTS) plus any
	subclass-specific kwargs in ``mcts_kwargs``. This lets the rollout runner mirror
	the variant-selection logic in ``runners/emomcts.py`` so SR/AT numbers reflect
	the actual emotion-aware MCTS being researched, not just the bare base class.
	"""
	if algo == "llm_raw":
		# one-shot LLM planner: take the chat planner's prior at face value (no search)
		prior, _v = planner.predict(state)
		return int(np.argmax(np.asarray(prior)))

	if algo == "gdpzero":
		# open-loop MCTS over LLM rollouts (as in runners/gdpzero.py)
		dp = OpenLoopMCTS(game, planner, configs)
		for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="gdpzero"):
			dp.search(state)
		return int(np.argmax(np.asarray(dp.get_action_prob(state))))

	if algo == "emomcts":
		cls = mcts_cls or EmotionAwareOpenLoopMCTS
		kwargs = mcts_kwargs or {}
		dp = cls(game, planner, configs, emotion_classifier, **kwargs)
		for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="emomcts"):
			dp.search(state)
		return int(np.argmax(np.asarray(dp.get_action_prob(state))))

	raise ValueError(f"unknown --algo {algo!r}; choose from {list(ALGOS)}")


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


def rollout_one(game, planner, algo, configs, emotion_classifier, max_turns, scenario,
				mcts_cls=None, mcts_kwargs=None):
	"""Play one full episode with ``algo`` choosing each system action; return the final session."""
	state = game.init_dialog(*scenario)
	# turn 0: the only valid move at the start is the greeting -> realize it directly
	valid0 = np.asarray(planner.get_valid_moves(state), dtype=float)
	greeting_idx = int(np.nonzero(valid0)[0][0]) if valid0.sum() > 0 else 0
	# EmotionAwarePersuasionGame.get_next_state returns (state, v, emotion); base returns (state, v).
	# Take only state[0] so both shapes work.
	state = game.get_next_state(state, greeting_idx)[0]
	# then plan turn by turn until the game ends or we hit the limit
	while game.get_dialog_ended(state) == 0.0 and len(state) < max_turns:
		action = pick_action(algo, state, game=game, planner=planner,
							  configs=configs, emotion_classifier=emotion_classifier,
							  mcts_cls=mcts_cls, mcts_kwargs=mcts_kwargs)
		state = game.get_next_state(state, action)[0]
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
def main(cmd_args):
	print(f"algo={cmd_args.algo}  saving to {cmd_args.output}")

	# load agents from TASKS; llm_raw keeps the model's built-in inference defaults, MCTS open-loop wants sampling on
	sys_inference_args = {} if cmd_args.algo == "llm_raw" else None
	backbone_model, family = make_backbone_model(cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host)
	game, system, user, planner = build_agents(
		cmd_args.game, backbone_model, family,
		sys_inference_args=sys_inference_args,
		llm_prior_topk=getattr(cmd_args, "llm_prior_topk", None),
	)
	all_dialogs = load_dialogs(cmd_args.game, cmd_args, system)

	configs = dotdict({
		"cpuct": cmd_args.cpuct,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
		"lambda_emo": cmd_args.lambda_emo,
		"c_emo_bonus": cmd_args.c_emo_bonus,
		"alpha_realization_emo": cmd_args.alpha_realization_emo,
		"beta_emo": cmd_args.beta_emo,
		"c_imag": cmd_args.c_imag,
		"imag_K": cmd_args.imag_K,
		"imag_H": cmd_args.imag_H,
	})
	emotion_classifier = getattr(game, "emotion_classifier", None)
	if cmd_args.algo == "emomcts" and emotion_classifier is None:
		raise ValueError(
			f"--algo emomcts needs an emotion-aware task with a classifier attached; "
			f"--game {cmd_args.game!r} has none (try --game emo_p4g)."
		)
	# #7: swap to the encoder-based HF classifier if requested (see runners/emomcts.py).
	if emotion_classifier is not None and getattr(cmd_args, "emotion_classifier", "llm") == "hf":
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		print(f"swapping classifier -> HFEmotionClassifier ({HFEmotionClassifier.DEFAULT_MODEL})")
		emotion_classifier = HFEmotionClassifier()
		game.emotion_classifier = emotion_classifier

	mcts_cls = None
	mcts_kwargs: dict = {}
	if cmd_args.algo == "emomcts":
		# Emotion-aware planner: the parallel multi-objective Q (EmotionAwareMultiObjectiveQ).
		# beta_emo weights the emotion-valence channel in PUCT; beta_emo=0 recovers the
		# task-only open-loop search.
		mcts_cls = EmotionAwareMultiObjectiveQ
		mcts_kwargs = {"beta_emo": cmd_args.beta_emo}

	_mcts_class_by_algo = {
		"llm_raw": "(none — llm_raw baseline)",
		"gdpzero": "OpenLoopMCTS",
		"emomcts": mcts_cls.__name__ if mcts_cls else "EmotionAwareMultiObjectiveQ",
	}
	setup_output_dir(cmd_args, runner_name="runners/rollout.py",
					 mcts_class=_mcts_class_by_algo.get(cmd_args.algo, cmd_args.algo),
					 mcts_args=configs)

	episodes = []
	cap = len(all_dialogs) if cmd_args.max_conv is None or cmd_args.max_conv < 0 else cmd_args.max_conv
	n = min(cap, len(all_dialogs))
	print(f"task={cmd_args.game}  {n} scenarios  (max_turns={cmd_args.max_turns})")
	pbar = tqdm(total=n, desc=f"rollout {cmd_args.game}/{cmd_args.algo}")
	for dialog in all_dialogs[:n]:
		did = dialog["id"]
		try:
			state = rollout_one(game, planner, cmd_args.algo, configs,
								 emotion_classifier, cmd_args.max_turns, dialog["scenario"],
								 mcts_cls=mcts_cls, mcts_kwargs=mcts_kwargs)
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
		# cumulative SR / AvgT over ALL episodes so far, printed every 10 completed dialogs
		if episodes and len(episodes) % 10 == 0:
			try:
				from metrics.dialog_metrics import compute_metrics, format_metrics
				m = compute_metrics(episodes, task=cmd_args.game, max_turns=cmd_args.max_turns)
				pbar.write(f"[after {len(episodes)} dialogs] {format_metrics(m)}")
			except Exception:
				pass
	pbar.close()

	# quick on-the-fly summary
	try:
		from metrics.dialog_metrics import compute_metrics, format_metrics
		print(format_metrics(compute_metrics(episodes, task=cmd_args.game, max_turns=cmd_args.max_turns)))
	except Exception:
		pass

	# emotion distribution + utterance->emotion records (only emomcts classifies; no-op otherwise)
	dump_emotion_records(emotion_classifier, cmd_args.output)
	print(f"done: {len(episodes)} episodes -> {cmd_args.output}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="self-play rollouts -> episode records for SR / AT / SL")
	add_common_args(parser, default_output="outputs/rollout.pkl")
	parser.add_argument("--max_turns", type=int, default=10, help="hard cap on dialog turns per episode")
	parser.add_argument("--max_conv", type=int, default=20, help="max scenarios to roll out (-1 for all)")
	parser.add_argument("--raise_errors", action="store_true", help="re-raise instead of skipping a failing rollout")
	parser.add_argument("--algo", choices=list(ALGOS), default="llm_raw",
						help="how to pick each system action (see pick_action in rollout.py)")
	# MCTS hyper-parameters (used by gdpzero + emomcts)
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="[--algo gdpzero|emomcts] MCTS simulations per turn")
	parser.add_argument("--max_realizations", type=int, default=3, help="[--algo gdpzero|emomcts] realizations sampled per state")
	parser.add_argument("--Q_0", type=float, default=0.0, help="[--algo gdpzero|emomcts] initial Q value for unvisited states")
	parser.add_argument("--cpuct", type=float, default=1.0, help="[--algo gdpzero|emomcts] UCT exploration constant")
	# Emotion-aware MCTS hyper-parameters (mirror runners/emomcts.py; only used when --algo emomcts).
	parser.add_argument('--emotion_classifier', choices=['llm', 'hf'], default='llm',
						help='[--algo emomcts] which emotion classifier to use. '
							 '"llm" = prompt-based, "hf" = j-hartmann/emotion-english-distilroberta-base.')
	parser.add_argument('--lambda_emo', type=float, default=0.0,
						help='[--algo emomcts] convex-blend weight on emotion penalty in Q updates. '
							 '0.0 = no penalty; >0 routes through EmotionAwareDiscountQOpenLoopMCTS (penalty path).')
	parser.add_argument('--c_emo_bonus', type=float, default=0.0,
						help='[--algo emomcts] weight on the emotion-DA selection bonus in PUCT. '
							 '0.0 = no bonus; >0 routes through EmotionGuidedDiscountQOpenLoopMCTS (uses EMOTION_DA_BONUS matrix).')
	parser.add_argument('--alpha_realization_emo', type=float, default=0.0,
						help='[--algo emomcts] utterance-layer rerank weight. '
							 '0.0 = no rerank; >0 routes through EmotionRealizationSelectorMCTS.')
	parser.add_argument('--beta_emo', type=float, default=0.0,
						help='[--algo emomcts] weight on the parallel Q_emo channel in PUCT '
							 '(Direction A — multi-objective parallel Q). '
							 '0.0 = no MultiObjectiveQ path; >0 routes through EmotionAwareMultiObjectiveQ '
							 'which scores actions by Q + β·Q_emo + cpuct·P·√N/(1+Nsa). Sweep {0.3, 0.7, 1.0}. '
							 'Does NOT compose with --c_emo_bonus / --alpha_realization_emo / --lambda_emo.')
	parser.add_argument('--c_imag', type=float, default=0.0,
						help='[--algo emomcts] weight on the imagination side-tree in PUCT '
							 '(Direction B — Markov rollouts). 0.0 = no imagination; >0 routes through '
							 'EmotionImaginationMCTS which runs K cheap Markov rollouts H turns deep at '
							 'every expansion. Sweep {0.3, 0.5, 1.0}.')
	parser.add_argument('--imag_K', type=int, default=50,
						help='[--c_imag>0] Monte Carlo samples per (state, action) at expansion.')
	parser.add_argument('--imag_H', type=int, default=10,
						help='[--c_imag>0] imagination rollout horizon in turns.')
	cmd_args = finalize_args(parser.parse_args())

	main(cmd_args)
