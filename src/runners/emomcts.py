"""Emotion-aware GDP-Zero: open-loop MCTS with an emotion classifier on top.

Same evaluation loop as ``runners/gdpzero.py`` but swaps ``OpenLoopMCTS`` for
``EmotionAwareOpenLoopMCTS``. Run it against an emotion-aware task (``--game emo_p4g``, the
default): those tasks are registered in ``runners/_common.py`` so ``build_agents`` returns a game
whose ``init_dialog`` yields an ``EmotionAwareDialogSession`` and attaches the task's emotion
classifier to the game (read here via ``game.emotion_classifier``). The output pickle is the same
per-turn schema as ``gdpzero.py``, so ``evaluators/run_judge.py`` can compare the two head-to-head.

    cd src
    python runners/gdpzero.py  --game p4g     --output outputs/gdpzero_p4g.pkl
    python runners/emomcts.py  --game emo_p4g --output outputs/emomcts_p4g.pkl
    python evaluators/run_judge.py --task p4g -f outputs/emomcts_p4g.pkl --h2h outputs/gdpzero_p4g.pkl
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import logging
import pickle
import argparse

import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from utils.gen_models import OpenAIModel
from mcts.emotion_mcts import EmotionAwareOpenLoopMCTS, EmotionAwareDiscountQOpenLoopMCTS
from runners._common import TASKS, make_backbone_model, build_agents, load_dialogs, dump_emotion_records, dump_da_emotion_records, add_common_args, finalize_args, setup_output_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def main(cmd_args):
	cfg = TASKS[cmd_args.game]

	# load agents from TASKS for the chosen dataset
	backbone_model, family = make_backbone_model(cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host)
	game, system, user, planner = build_agents(cmd_args.game, backbone_model, family)

	emotion_classifier = getattr(game, "emotion_classifier", None)
	if emotion_classifier is None:
		raise ValueError(
			f"--game {cmd_args.game!r} is not emotion-aware, so no emotion classifier was attached. "
			f"Run emomcts with an emotion-aware task (e.g. --game emo_p4g)."
		)

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	all_dialogs = load_dialogs(cmd_args.game, cmd_args, system)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
		# convex-blend weight on the emotion penalty in search(): v~ = (1-λ)·v + λ·π(e).
		# λ=0 collapses to GDPZero; small λ (~0.2–0.3) is a safe first try.
		"lambda_emo": cmd_args.lambda_emo,
	})
	setup_output_dir(cmd_args, runner_name="runners/emomcts.py",
					 mcts_class="EmotionAwareDiscountQOpenLoopMCTS", mcts_args=args)

	output = []  # for evaluation. [{did, context, ori_da, ori_resp, new_da, new_resp, debug}, ...]
	# per-turn snapshots of dialog_planner.emotions_count, aggregated by dump_da_emotion_records
	# into a system-DA -> user-emotion histogram for the run.
	da_emotion_counts = []
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating dialogues")
	for dialog in all_dialogs:
		if num_done == num_dialogs:
			break

		did = dialog["id"]
		turns = dialog["turns"]
		print("evaluating dialog id: ", did)
		context = ""

		state = game.init_dialog(*dialog["scenario"])
		for t in range(len(turns) - 1):  # skip last turn: there is no next turn to evaluate against
			print(f"dialogue {num_done}, turn {t}")
			turn, next_turn = turns[t], turns[t + 1]
			usr_da, usr_utt = turn["usr_da"], turn["usr_utt"]
			sys_da, sys_utt = turn["sys_da"], turn["sys_utt"]

			# game ended
			if usr_da == cfg.success_user_da:
				break

			# emotion-aware session: the replayed prefix has no labelled emotion -> neutral placeholder
			state.add_single(game.SYS, sys_da, "Neutral", sys_utt)
			user_emotion = emotion_classifier.predict_from_single_utterance(usr_utt)
			state.add_single(game.USR, usr_da, user_emotion, usr_utt)

			# update context for evaluation
			context = f"""
			{context}
			{game.SYS}: {sys_utt}
			{game.USR}: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# emotion-aware mcts policy
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			dialog_planner = EmotionAwareDiscountQOpenLoopMCTS(game, planner, args, emotion_classifier, emo_lambda=0.5)
			print("searching")
			for i in tqdm(range(args.num_MCTS_sims)):
				dialog_planner.search(state)

			mcts_policy = dialog_planner.get_action_prob(state)
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]

			# fetch the generated utterance from simulation
			mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

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
					"realizations_Vs": dialog_planner.realizations_Vs,
					"realizations_Ns": dialog_planner.realizations_Ns,
					"emotions_count": dialog_planner.emotions_count,
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
			output.append(cmp_data)
			# snapshot the per-turn DA->emotion counts (dict-copy to detach from the planner's defaultdict)
			da_emotion_counts.append({k: dict(v) for k, v in dialog_planner.emotions_count.items()})

			if cmd_args.debug:
				print(context)
				print("human resp: ", human_resp)
				print("human da: ", next_sys_da)
				print("mcts resp: ", mcts_pred_rep)
				print("mcts da: ", mcts_policy_next_da)
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	pbar.close()

	# emotion distribution + utterance->emotion records (seeding here + inside the MCTS, both go
	# through the same shared classifier instance)
	dump_emotion_records(emotion_classifier, cmd_args.output)
	# per-DA user-emotion histogram aggregated across MCTS rollouts (research-reportable stat)
	dump_da_emotion_records(da_emotion_counts, cmd_args.output)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	add_common_args(parser, default_output="outputs/emomcts.pkl")
	parser.set_defaults(game="emo_p4g")  # emomcts only makes sense on an emotion-aware task
	parser.add_argument(''
						'--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
	parser.add_argument('--lambda_emo', type=float, default=0.3,
						help='convex-blend weight on emotion penalty in Q updates: v~ = (1-λ)·v + λ·π(e). '
							 '0.0 = GDPZero-equivalent; small λ (0.2-0.3) is a safe first try.')
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	main(cmd_args)
