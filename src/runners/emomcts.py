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
import math
import pickle
import argparse

import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from utils.gen_models import OpenAIModel
from mcts.emotion_mcts import EmotionAwareMultiObjectiveQ
from runners._common import TASKS, make_backbone_model, build_agents, load_dialogs, dump_emotion_records, dump_da_emotion_records, add_common_args, finalize_args, setup_output_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def log_emotions(emotion_classifier):
	records_start = emotion_classifier.records
	agg_dist = {str(e): 0.0 for e in emotion_classifier.emotions}
	for rec in records_start:
		for e, p in rec["distribution"].items():
			agg_dist[str(e)] += p
	n_calls = len(records_start)
	if n_calls > 0:
		agg_dist = {e: p / n_calls for e, p in agg_dist.items()}
	dist_str = ", ".join(f"{e}={p:.2f}" for e, p in sorted(agg_dist.items(), key=lambda kv: -kv[1]))
	print(f"aggregated emotion distribution over {n_calls} classifier calls: {{{dist_str}}}")
	# for each emotion bucket, surface up to 3 example utterances classified into it
	# *during this turn* — interpretation hook for the aggregate above.
	examples_per_emotion: dict = {str(e): [] for e in emotion_classifier.emotions}
	for rec in records_start:
		bucket = examples_per_emotion.setdefault(rec["emotion"], [])
		if len(bucket) < 3 and rec["utterance"] not in bucket:
			bucket.append(rec["utterance"])
	for emo in emotion_classifier.emotions:
		utts = examples_per_emotion[str(emo)]
		print(f"  [{emo}]")
		if not utts:
			print(f"    - (no examples yet)")
			continue
		for u in utts:
			trimmed = u if len(u) <= 80 else u[:150] + "..."
			print(f"    - {trimmed}")


def _compute_counterfactual_das(dialog_planner, state, system) -> dict:
	"""Per-turn attribution: for each shaping ablation, what DA would PUCT pick from
	the post-search state?

	Returns a dict ``{strategy: da_string}`` with these keys:

	- ``argmax_visits``     : the DA actually played (argmax over Nsa); same as new_da.
	                          Reported here so downstream diffing scripts find all keys
	                          in one place.
	- ``argmax_q``          : argmax over Q only (no exploration term, no bonus). What
	                          the value estimate alone says.
	- ``puct_no_bonus``     : PUCT without the c_emo_bonus·B(emotion, da)·explore term.
	                          When this differs from argmax_visits the bonus was the
	                          deciding factor among the search-budget-shifting effects
	                          captured in Nsa.
	- ``puct_vanilla``      : Q + cpuct·P·explore — what GDPZero's PUCT would pick from
	                          THIS post-search state. Approximates "vanilla next move
	                          from here" (NOT a full vanilla re-search; Q itself already
	                          carries the penalty's contribution).

	Caveat: the penalty lives in Q (mixed in during search across all visits) and is
	NOT cleanly recoverable post-hoc. ``puct_vanilla`` therefore approximates "vanilla
	choice given the Q values we ended up with" rather than "vanilla choice from a
	parallel vanilla search." For the latter, run GDPZero in parallel and diff via the
	run_judge --h2h metadata. This counterfactual is the cheap in-run proxy.
	"""
	hashable_state = dialog_planner._to_string_rep(state)
	if hashable_state not in dialog_planner.Q:
		# state was never expanded — nothing to attribute. Shouldn't happen post-search,
		# but be defensive so logging never crashes the run.
		return {}

	Q = dialog_planner.Q[hashable_state]
	Nsa = dialog_planner.Nsa[hashable_state]
	Ns = dialog_planner.Ns[hashable_state] or 1e-8
	P = dialog_planner.P[hashable_state]
	valid = dialog_planner.valid_moves[hashable_state]
	cpuct = dialog_planner.configs.cpuct

	def _argmax(score_fn):
		best_a, best_s = -1, -float("inf")
		for a in valid:
			s = score_fn(a)
			if s > best_s:
				best_s, best_a = s, a
		return system.dialog_acts[best_a] if best_a >= 0 else None

	def _puct_no_bonus(a):
		explore = math.sqrt(Ns) / (1 + Nsa[a])
		return Q[a] + cpuct * P[a] * explore

	return {
		"argmax_visits":  system.dialog_acts[max(Nsa, key=Nsa.get)],
		"argmax_q":       _argmax(lambda a: Q[a]),
		"puct_no_bonus":  _argmax(_puct_no_bonus),
		# puct_vanilla is identical to puct_no_bonus by construction (Q already absorbs
		# the penalty); separate key reserved so future "true vanilla" attribution
		# (e.g., logging GDPZero's Q from a parallel search) can swap in without a
		# downstream schema change.
		"puct_vanilla":   _argmax(_puct_no_bonus),
	}


def main(cmd_args):
	cfg = TASKS[cmd_args.game]

	# load agents from TASKS for the chosen dataset
	backbone_model, family = make_backbone_model(cmd_args.llm, cmd_args.gen_sentences, cmd_args.ollama_model, cmd_args.ollama_host)
	game, system, user, planner = build_agents(
		cmd_args.game, backbone_model, family,
		llm_prior_topk=getattr(cmd_args, "llm_prior_topk", None),
	)

	emotion_classifier = getattr(game, "emotion_classifier", None)
	if emotion_classifier is None:
		raise ValueError(
			f"--game {cmd_args.game!r} is not emotion-aware, so no emotion classifier was attached. "
			f"Run emomcts with an emotion-aware task (e.g. --game emo_p4g)."
		)
	# #7: swap to the encoder-based HF classifier if requested. Drop-in interface — the rest of
	# the runner / MCTS sees the same methods. Done after build_agents so the classifier slot on
	# the game is replaced consistently (game.get_next_state reads from there too).
	if getattr(cmd_args, "emotion_classifier", "llm") == "hf":
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		print(f"swapping classifier -> HFEmotionClassifier ({HFEmotionClassifier.DEFAULT_MODEL})")
		emotion_classifier = HFEmotionClassifier()
		game.emotion_classifier = emotion_classifier

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	all_dialogs = load_dialogs(cmd_args.game, cmd_args, system)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
		"beta_emo": cmd_args.beta_emo,
	})
	# Emotion-aware planner: the parallel multi-objective Q (EmotionAwareMultiObjectiveQ),
	# which scores actions by Q + beta_emo*Q_emo + cpuct*P*sqrt(N)/(1+Nsa). beta_emo weights
	# the emotion-valence channel; beta_emo=0 recovers the task-only open-loop search.
	mcts_cls = EmotionAwareMultiObjectiveQ
	setup_output_dir(cmd_args, runner_name="runners/emomcts.py",
					 mcts_class=mcts_cls.__name__, mcts_args=args)

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
			turn, next_turn = turns[t], turns[t + 1]
			usr_da, usr_utt = turn["usr_da"], turn["usr_utt"]
			sys_da, sys_utt = turn["sys_da"], turn["sys_utt"]

			# game ended
			if usr_da == cfg.success_user_da:
				break

			# emotion-aware session: the replayed prefix has no labelled emotion -> neutral placeholder
			state.add_single(game.SYS, sys_da, "Neutral", sys_utt)
			user_dist = emotion_classifier.predict_distribution_from_full_history(state, usr_utt)
			user_emotion = max(user_dist, key=user_dist.get)
			state.add_single(game.USR, usr_da, user_emotion, usr_utt, user_dist)

			print(f"dialogue {num_done}, turn {t}")

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
			# EmotionAwareMultiObjectiveQ subclasses EmotionAwareOpenLoopMCTS directly;
			# its only extra knob is beta_emo (the weight on the parallel Q_emo channel).
			dialog_planner = mcts_cls(
				game,
				planner,
				args,
				emotion_classifier,
				beta_emo=cmd_args.beta_emo,
			)
			for _ in tqdm(range(args.num_MCTS_sims)):
				dialog_planner.search(state)

			log_emotions(emotion_classifier)

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

			# Tier-1 logging additions (see debug.md "what to log" discussion):
			# - counterfactual_da: what each shaping ablation would have picked. Lets
			#   you compute retrospective attribution after run_judge labels each turn
			#   ("the bonus flipped action on N turns, B won X / lost Y of those").
			# - last_user_emotion / last_user_distribution: condition slicing. After
			#   run_judge, ask "of turns where last_user_emotion=Happiness and the
			#   bonus flipped action, did B win?" — per-cell bonus-matrix calibration.
			# - turn_index / dialog_length: positional slicing (early vs late turn).
			counterfactual_da = _compute_counterfactual_das(dialog_planner, state, system)
			last_user_emo = state.predicted_emotion() if state.history else None
			last_user_dist = state.predicted_distribution() if state.history else None

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				'counterfactual_da': counterfactual_da,
				'last_user_emotion': str(last_user_emo) if last_user_emo is not None else None,
				'last_user_distribution': last_user_dist,
				'turn_index': t,
				'dialog_length': len(turns),
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
	parser.add_argument('--emotion_classifier', choices=['llm', 'hf'], default='llm',
						help='which emotion classifier to use. '
							 '"llm" = prompt-based (shares the system backbone; uses few-shot + low temp + cache). '
							 '"hf" = j-hartmann/emotion-english-distilroberta-base (deterministic encoder, no LLM cost).')
	parser.add_argument('--beta_emo', type=float, default=0.0,
						help='weight on the parallel Q_emo channel in PUCT (EmotionAwareMultiObjectiveQ). '
							 'Tracks the task value Q and the emotion-valence value Q_emo separately and '
							 'scores actions by Q + β·Q_emo + cpuct·P·√N/(1+Nsa). 0.0 recovers the '
							 'task-only open-loop search; sweep {0.3, 0.7, 1.0}.')
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	main(cmd_args)
