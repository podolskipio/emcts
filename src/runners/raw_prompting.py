"""Raw-prompting baseline: greedy one-step policy planning (no tree search).

Same evaluation loop as the original GDP-Zero ``runners/raw_prompting.py``: replay a held-out
dataset and, at every turn, ask the planner for its prior over dialog acts, pick the argmax, and
realize that act with ``game.get_next_state``. Agents and dataset are built from ``TASKS`` so it
runs on ``--game p4g|esc|cb``.

    cd src
    python runners/raw_prompting.py --game p4g --data data/p4g/300_dialog_turn_based.pkl
    python runners/raw_prompting.py --game esc --data data/esc/esc-valid.txt
    python runners/raw_prompting.py --game cb  --data data/cb/cb-valid.txt --llm ollama --ollama_model llama3.1
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import logging
import pickle
import argparse

import numpy as np
from tqdm.auto import tqdm

from utils.gen_models import OpenAIModel
from runners._common import TASKS, make_backbone_model, build_agents, load_dialogs, add_common_args, finalize_args, setup_output_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	cfg = TASKS[cmd_args.game]

	# load agents from TASKS for the chosen dataset; raw prompting uses the system model's defaults
	backbone_model, family = make_backbone_model(cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host)
	game, system, user, planner = build_agents(cmd_args.game, backbone_model, family, sys_inference_args={})

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	all_dialogs = load_dialogs(cmd_args.game, cmd_args, system)

	num_dialogs = cmd_args.num_dialogs
	setup_output_dir(cmd_args, runner_name="runners/raw_prompting.py",
					 mcts_class="(none — raw prompting baseline)")

	output = []  # for evaluation. [{did, context, ori_da, ori_resp, new_da, new_resp, debug}, ...]
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for dialog in all_dialogs:
		if num_done == num_dialogs:
			break

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

			# greedy policy from the planner's one-shot prior
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			prior, v = planner.predict(state)
			greedy_policy = system.dialog_acts[np.argmax(prior)]
			next_best_state, _ = game.get_next_state(state, np.argmax(prior))
			greedy_pred_resp = next_best_state.history[-2][2]

			# next ground truth utterance
			human_resp = next_turn["sys_utt"]
			next_sys_da = next_turn["sys_da"]

			# logging for debug
			debug_data = {
				"prior": prior,
				"da": greedy_policy,
				"v": v,
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': greedy_pred_resp,
				'new_da': greedy_policy,
				"debug": debug_data,
			}
			output.append(cmp_data)
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	pbar.close()
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_common_args(parser, default_output="outputs/raw_prompt.pkl")
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test on')
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	main(cmd_args)
