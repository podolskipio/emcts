"""GDP-Zero ablation: open-loop MCTS, but the final utterance is freshly generated.

Same evaluation loop as the original GDP-Zero ``runners/gdpzero_noRS.py``: build the open-loop
MCTS tree exactly like ``gdpzero.py``, but instead of returning the best-scoring realization from
the tree, re-generate the chosen dialog act once via ``game.get_next_state`` (no realization
*selection*). Agents and dataset are built from ``TASKS`` so it runs on ``--game p4g|esc|cb``.

    cd src
    python runners/gdpzero_noRS.py --game p4g --data data/p4g/300_dialog_turn_based.pkl
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import logging
import pickle
import argparse

from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from utils.gen_models import OpenAIModel
from mcts.mcts import OpenLoopMCTS
from runners._common import TASKS, make_backbone_model, build_agents, load_dialogs, add_common_args, finalize_args, setup_output_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	cfg = TASKS[cmd_args.game]

	# load agents from TASKS for the chosen dataset
	backbone_model, family = make_backbone_model(
		llm=cmd_args.llm,
		gen_sentences=cmd_args.gen_sentences,
		ollama_model=cmd_args.ollama_model,
		ollama_host=cmd_args.ollama_host,
		sglang_model=cmd_args.sglang_model,
	)
	game, system, user, planner = build_agents(cmd_args.game, backbone_model, family)

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	all_dialogs = load_dialogs(cmd_args.game, cmd_args, system)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"max_realizations": cmd_args.max_realizations,
		"Q_0": cmd_args.Q_0,
	})
	setup_output_dir(cmd_args, runner_name="runners/gdpzero_noRS.py",
					 mcts_class="OpenLoopMCTS (no realization sampling)", mcts_args=args)

	output = []  # for evaluation. [{did, context, ori_da, ori_resp, new_da, new_resp, debug}, ...]
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	# --num_workers > 1 evaluates several dialogs at once so SGLang can batch the requests: the
	# runner is otherwise one dependent chain of short requests with the GPU idle in between (see
	# the '[cache:...] N% of wall' line). Threads, not processes — the time goes on waiting for
	# HTTP, and the agents/games hold no mutable state outside __init__, so sharing them is safe.
	# Each turn still builds its own MCTS object, so no search state is shared between dialogs.
	# Per-turn records are collected per dialog and merged in dialog order on every dump, so the
	# output pickle does not depend on which dialog happens to finish first.
	finished = []            # (dialog index, per-turn records{}), guarded by results_lock
	results_lock = Lock()

	def run_one_dialog(indexed_dialog):
		nonlocal num_done
		idx, dialog = indexed_dialog
		dialog_output = []   # this dialog's records; merged into `output` under the lock
		did = dialog["id"]
		turns = dialog["turns"]
		print("evaluating dialog id: ", did)
		context = ""

		state = game.init_dialog(*dialog["scenario"])
		for t in range(len(turns) - 1):  # skip last turn: there is no next turn to evaluate against
			turn, next_turn = turns[t], turns[t + 1]
			usr_da, usr_utt = turn["usr_da"], turn["usr_utt"]
			sys_da, sys_utt = turn["sys_da"], turn["sys_utt"]

			# game ended
			if usr_da == cfg.success_user_da:
				break

			state.add_single(game.SYS, sys_da, sys_utt)
			state.add_single(game.USR, usr_da, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			{game.SYS}: {sys_utt}
			{game.USR}: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# mcts policy
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			dialog_planner = OpenLoopMCTS(game, planner, args)
			print("searching")
			for i in tqdm(range(args.num_MCTS_sims)):
				dialog_planner.search(state)

			mcts_policy = dialog_planner.get_action_prob(state)
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]

			# generate a new utterance for the chosen DA (no realization selection)
			next_best_state, _ = game.get_next_state(state, np.argmax(mcts_policy))
			mcts_pred_rep = next_best_state.history[-2][2]

			# next ground truth utterance
			human_resp = next_turn["sys_utt"]
			next_sys_da = next_turn["sys_da"]

			# logging for debug
			debug_data = {
				"probs": mcts_policy,
				"da": mcts_policy_next_da,
				"search_tree": {
					"Ns": dialog_planner.Ns,
					"Nsa": dialog_planner.Nsa,
					"Q": dialog_planner.Q,
					"P": dialog_planner.P,
					"Vs": dialog_planner.Vs,
					"realizations": dialog_planner.realizations,
				},
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				"debug": debug_data,
			}
			dialog_output.append(cmp_data)
		with results_lock:
			finished.append((idx, dialog_output))
			output[:] = [rec for entry in sorted(finished, key=lambda t: t[0]) for rec in entry[1]]
			with open(cmd_args.output, "wb") as f:
				pickle.dump(output, f)
			num_done += 1
			pbar.update(1)

	indexed_dialogs = list(enumerate(all_dialogs[:num_dialogs]))
	workers = max(1, min(cmd_args.num_workers, len(indexed_dialogs)))
	if workers == 1:
		# unchanged sequential path, so single-worker runs stay identical to before
		for indexed_dialog in indexed_dialogs:
			run_one_dialog(indexed_dialog)
	else:
		print(f"evaluating {workers} dialogs concurrently (--num_workers {cmd_args.num_workers})")
		pool = ThreadPool(processes=workers)
		pool.map(run_one_dialog, indexed_dialogs)
		pool.close()
		pool.join()
	pbar.close()
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_common_args(parser, default_output="outputs/gdpzero_noRS.pkl")
	parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
	parser.add_argument('--num_workers', type=int, default=1,
						help='evaluate this many dialogs concurrently (threads). 1 = the old '
						     'sequential behaviour. Higher values keep several requests in flight '
						     'so SGLang can batch them; the GPU is otherwise idle between calls. '
						     '4-8 suits a single local server. Records stay in dialog order.')
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	main(cmd_args)
