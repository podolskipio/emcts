"""Self-play rollouts: produce episode records that ``metrics/run_metrics.py`` can score.

Unlike the response-comparison runners (``raw_prompting.py`` / ``gdpzero*.py``), this one plays
*full* dialogs from scratch: the policy planner picks the next system dialog act, the user
simulator replies, and we repeat until the game ends or the turn limit is hit. For every dialog
it records ``{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}``
(see ``metrics/dialog_metrics.py`` for the schema).

It uses the *greedy* planner (chat-planner prior -> argmax); a tree-search rollout needs the
``mcts/mcts.py`` <-> game ``get_next_state`` reconciliation first (see ``runners/gdpzero.py``).

    cd src
    python runners/rollout.py --game cb  --data data/cb/cb-valid.txt --max_turns 8  --output outputs/rollout_cb.pkl
    python runners/rollout.py --game p4g --data data/p4g/300_dialog_turn_based.pkl --max_turns 10 --output outputs/rollout_p4g.pkl
    python metrics/run_metrics.py --episodes outputs/rollout_cb.pkl --task cb --max_turns 8
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

from runners._common import TASKS, make_backbone_model, build_agents, add_common_args, finalize_args

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


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


def rollout_one(game, system, planner, max_turns, scenario):
	"""Play one full episode; returns the final DialogSession."""
	state = game.init_dialog(*scenario)
	# turn 0: the only valid move at the start is the greeting -> realize it
	valid0 = np.asarray(planner.get_valid_moves(state), dtype=float)
	greeting_idx = int(np.nonzero(valid0)[0][0]) if valid0.sum() > 0 else 0
	state, _agent_state, _ = game.get_next_state(state, greeting_idx, agent_state=[])
	# then plan turn by turn until the game ends or we hit the limit
	while game.get_dialog_ended(state) == 0.0 and len(state) < max_turns:
		valid = np.asarray(planner.get_valid_moves(state), dtype=float)
		prior, _v = planner.predict(state)
		masked = np.asarray(prior, dtype=float) * valid
		action = int(np.argmax(masked)) if masked.sum() > 0 else int(np.argmax(valid))
		state, _agent_state, _ = game.get_next_state(state, action, agent_state=[])
	return state


def make_episode(task, did, game, state):
	ended = game.get_dialog_ended(state)
	episode = {
		"did": did,
		"task": task,
		"success": bool(ended >= 1.0),
		"num_turns": len(state),
		"history": [list(t) for t in state.history],
	}
	if task == "cb":
		episode["buyer_price"] = getattr(state, "buyer_price", None)
		episode["seller_price"] = getattr(state, "seller_price", None)
		episode["deal_price"] = _extract_cb_price(state) if episode["success"] else None
	return episode


def main():
	parser = argparse.ArgumentParser(description="self-play rollouts -> episode records for SR / AT / SL")
	add_common_args(parser, default_output="outputs/rollout.pkl")
	parser.add_argument("--max_turns", type=int, default=10, help="hard cap on dialog turns per episode")
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	cfg = TASKS[cmd_args.game]
	backbone_model, family = make_backbone_model(
		cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host,
	)
	game, system, user, planner = build_agents(
		cmd_args.game, backbone_model, family,
		zero_shot=cmd_args.zero_shot,
		sys_inference_args={},  # greedy: use the model's built-in inference defaults
	)
	data_path = cmd_args.data or cfg.default_data
	dialogs = cfg.read_dialogs(data_path, set(system.dialog_acts))
	print(f"task={cmd_args.game}  loaded {len(dialogs)} scenarios from {data_path}  (max_turns={cmd_args.max_turns})")

	episodes = []
	n = min(cmd_args.num_dialogs, len(dialogs))
	pbar = tqdm(total=n, desc=f"rollout {cmd_args.game}")
	for dialog in dialogs[:n]:
		did = dialog["id"]
		try:
			state = rollout_one(game, system, planner, cmd_args.max_turns, dialog["scenario"])
			episodes.append(make_episode(cmd_args.game, did, game, state))
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
