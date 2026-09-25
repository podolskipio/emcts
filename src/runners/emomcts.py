"""Emotion-aware GDP-Zero: open-loop MCTS with an emotion classifier on top.

Same evaluation loop as ``runners/gdpzero.py`` but swaps ``OpenLoopMCTS`` for
``EmotionAwareOpenLoopMCTS``. Run it against an emotion-aware task (``--game emo_p4g``, the
default): those tasks are registered in ``runners/_common.py`` so ``build_agents`` returns a game
whose ``init_dialog`` yields an ``EmotionAwareDialogSession``. The classifier that game labels user
reactions with is built once here (``make_emotion_classifier``) and handed to every dialog. The
output pickle is the same per-turn schema as ``gdpzero.py``, so ``evaluators/run_judge.py`` can
compare the two head-to-head.

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

from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from utils.gen_models import OpenAIModel
from mcts.mcts import SEARCH_HORIZONS
from mcts.emotion_mcts import (
	EmotionAwareMultiObjectiveQ, AFF_POOL_KEYS, CACHE_DRAWS, EMO_SIGNALS, EMO_VALENCE_TABLES, check_emo_signal_flags
)
from runners._common import (
	TASKS, make_backbone_model, make_emotion_classifier, build_agents, load_dialogs, replay_root_is_terminal,
	load_p4g_personas, apply_seed, dump_emotion_records, dump_da_emotion_records,
	add_common_args, finalize_args, setup_output_dir, subtree_emo_stats,
	build_subtree_records, write_subtree_ndjson,
)

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
	check_emo_signal_flags(cmd_args.emo_signal)
	apply_seed(cmd_args)

	backbone_model, family = make_backbone_model(
		llm=cmd_args.llm,
		gen_sentences=cmd_args.gen_sentences,
		ollama_model=cmd_args.ollama_model,
		ollama_host=cmd_args.ollama_host,
		sglang_model=cmd_args.sglang_model,
	)
	# One classifier for the whole run: every dialog's game gets this same instance, so the
	# run's utterance -> emotion records (runner seeding + everything the MCTS classifies)
	# land in one place. None on a task that has no emotion channel, which emomcts refuses.
	emotion_classifier = make_emotion_classifier(cmd_args.game, cmd_args.emotion_classifier, backbone_model)
	if emotion_classifier is None:
		raise ValueError(
			f"--game {cmd_args.game!r} is not emotion-aware, so there is no emotion classifier. "
			f"Run emomcts with an emotion-aware task (e.g. --game emo_p4g)."
		)
	# Every dialog builds its own agents from these (see run_one_dialog), so two concurrent
	# workers never touch the same game / planner / user simulator.
	agent_kwargs = dict(
		llm_prior_topk=cmd_args.llm_prior_topk,
		logit_scoring=cmd_args.logit_scoring,
		explicit_value_labels=cmd_args.explicit_value_labels,
		emotion_classifier=emotion_classifier,
		# the game's horizon, same as rollout.py's. Omitting it inherited the game default of
		# 15 and gave this runner a different environment from the grid's 10.
		max_conv_turns=cmd_args.max_turns,
	)

	ontology = cfg.game_cls.get_game_ontology()
	print(f"System dialog acts: {ontology['system']['dialog_acts']}")
	print(f"User dialog acts: {ontology['user']['dialog_acts']}")

	all_dialogs = load_dialogs(cmd_args.game, cmd_args)
	# {} unless --p4g_persona; see load_p4g_personas / utils.p4g_personas
	persona_texts = load_p4g_personas(cmd_args)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
		"beta_emo": cmd_args.beta_emo,
		"emo_risk_lambda": cmd_args.emo_risk_lambda,
		"emo_signal": cmd_args.emo_signal,
		"emo_valence_table": cmd_args.emo_valence_table,
		"search_horizon": cmd_args.search_horizon,
		"aff_pool": cmd_args.aff_pool,
		"aff_pool_bias": cmd_args.aff_pool_bias,
		"aff_pool_tau": cmd_args.aff_pool_tau,
		"aff_pool_key": cmd_args.aff_pool_key,
		"emo_centre": cmd_args.emo_centre,
		"emo_constraint_tau": cmd_args.emo_constraint_tau,
		"emo_constraint_m_warm": cmd_args.emo_constraint_m_warm,
		"cache_ended_children": cmd_args.cache_ended_children,
		"cache_fresh_depth1": cmd_args.cache_fresh_depth1,
		"cache_draw": cmd_args.cache_draw,
		"cache_bucket_tau": cmd_args.cache_bucket_tau,
		"cache_kernel_h": cmd_args.cache_kernel_h,
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
	# --num_workers > 1 evaluates several dialogs at once so SGLang can batch the requests: the
	# runner is otherwise one dependent chain of short requests with the GPU idle in between (see
	# the '[cache:...] N% of wall' line). Threads, not processes — the time goes on waiting for
	# HTTP. Each dialog builds its own agents and each turn its own MCTS object, so concurrent
	# workers share only the backbone model (an HTTP client) and the emotion classifier.
	# Per-turn records are collected per dialog and merged in dialog order on every dump, so the
	# output pickle does not depend on which dialog happens to finish first.
	finished = []            # (dialog index, per-turn records{}), guarded by results_lock
	results_lock = Lock()

	def run_one_dialog(indexed_dialog):
		nonlocal num_done
		idx, dialog = indexed_dialog
		dialog_output = []   # this dialog's records; merged into `output` under the lock
		dialog_subtree = []   # frozen-schema subtree records for this dialogue (task 1.5)
		dialog_da_counts = []  # per-turn DA->emotion snapshots, merged with the records
		did = dialog["id"]
		# Agents are per dialog, so each worker thread owns its own: building them is pure
		# object construction (no I/O), and --p4g_persona conditions the user simulator and the
		# planner's value estimator on THIS dialogue's persuadee. The run-wide emotion
		# classifier is passed in, so every game classifies through the same instance.
		game, system, user, planner = build_agents(
			cmd_args.game, backbone_model, family,
			persona=persona_texts.get(did), **agent_kwargs,
		)
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

			# Nothing to plan once the episode is over by the game's own rule: under
			# --search_horizon episode search would return -1.0 without expanding this root and
			# get_action_prob would hand back NaN. No-op under legacy. See replay_root_is_terminal.
			if replay_root_is_terminal(game, state, cmd_args.search_horizon):
				break

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
				emo_risk_lambda=cmd_args.emo_risk_lambda,
				emo_signal=cmd_args.emo_signal,
				emo_valence_table=cmd_args.emo_valence_table,
				aff_pool=cmd_args.aff_pool,
				aff_pool_bias=cmd_args.aff_pool_bias,
				aff_pool_tau=cmd_args.aff_pool_tau,
				aff_pool_key=cmd_args.aff_pool_key,
				emo_centre=cmd_args.emo_centre,
				emo_constraint_tau=cmd_args.emo_constraint_tau,
				emo_constraint_m_warm=cmd_args.emo_constraint_m_warm,
				cache_draw=cmd_args.cache_draw,
				cache_bucket_tau=cmd_args.cache_bucket_tau,
				cache_kernel_h=cmd_args.cache_kernel_h,
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
			_m2_emo, _sigma_emo = subtree_emo_stats(dialog_planner)
			# frozen NDJSON subtree schema (task 1.5): one record per edge, per turn.
			dialog_subtree += build_subtree_records(
				dialog_planner, dlg_id=did, turn=t, root_state=state,
				seed=cmd_args.seed)
			debug_data = {
				"probs": mcts_policy,
				"da": mcts_policy_next_da,
				"search_tree": {
					"Ns": dialog_planner.Ns,
					"Nsa": dialog_planner.Nsa,
					"Q": dialog_planner.Q,
					"P": dialog_planner.P,
					"Vs": dialog_planner.Vs,
					# Welford variance of the emotion channel, per edge. Written by every
					# runner in every run (schema freeze): planners with no emotion channel
					# report 0.0 on every edge rather than omitting the field.
					"M2_emo": _m2_emo,
					"sigma_emo": _sigma_emo,
					"realizations": dialog_planner.realizations,
					"realizations_Vs": dialog_planner.realizations_Vs,
					"realizations_Ns": dialog_planner.realizations_Ns,
					# plain dict copy: emotions_count is a defaultdict whose factory is a *bound method*
					# of the planner, so pickling it as-is drags in the whole MCTS -> game -> backbone
					# model (and, under --llm sglang, the openai client's unpicklable RLock).
					"emotions_count": {k: dict(v) for k, v in dialog_planner.emotions_count.items()},
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
			dialog_output.append(cmp_data)
			# snapshot the per-turn DA->emotion counts (dict-copy to detach from the planner's defaultdict)
			dialog_da_counts.append({k: dict(v) for k, v in dialog_planner.emotions_count.items()})

			if cmd_args.debug:
				print(context)
				print("human resp: ", human_resp)
				print("human da: ", next_sys_da)
				print("mcts resp: ", mcts_pred_rep)
				print("mcts da: ", mcts_policy_next_da)
		# one gzipped NDJSON per dialogue, written once the dialogue is done so
		# concurrent workers never share a file.
		write_subtree_ndjson(dialog_subtree, cmd_args.output, did)
		with results_lock:
			finished.append((idx, dialog_output, dialog_da_counts))
			output[:] = [rec for entry in sorted(finished, key=lambda t: t[0]) for rec in entry[1]]
			da_emotion_counts[:] = [c for entry in sorted(finished, key=lambda t: t[0]) for c in entry[2]]
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
	parser.add_argument('--search_horizon', '--search-horizon', choices=list(SEARCH_HORIZONS), default='legacy',
						help='which get_dialog_ended values end a simulated branch. "legacy" (DEFAULT, unchanged, GDP-Zero): only success is terminal, so search keeps expanding past the turn limit and after a verbatim stall -- states no real episode reaches. "episode": any non-zero get_dialog_ended is terminal and its value (+1 donate / -1 turn limit or stall) is backed up, so search stops where the episode loop stops. See analysis/phase1/SEARCH_HORIZON_BUG.md.')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
	parser.add_argument('--beta_emo', type=float, default=0.0,
						help='weight on the parallel Q_emo channel in PUCT (EmotionAwareMultiObjectiveQ). '
							 'Tracks the task value Q and the emotion-valence value Q_emo separately and '
							 'scores actions by Q + β·Q_emo + cpuct·P·√N/(1+Nsa). 0.0 recovers the '
							 'task-only open-loop search; sweep {0.3, 0.7, 1.0}.')
	parser.add_argument('--emo_risk_lambda', '--emo-risk-lambda', type=float, default=0.0,
						help='risk aversion on the emotion channel: selection uses '
							 'Q_emo - λ·σ_emo in place of Q_emo, where σ_emo is the Welford '
							 'standard deviation of the z values backed up into that edge. '
							 'DEFAULT 0.0, which reduces the expression to Q_emo exactly '
							 '(x - 0.0*σ is bit-identical to x), i.e. the pre-change rule. '
							 'Instrumentation for a later Tier-C experiment; not swept now.')
	parser.add_argument('--emo_signal', '--emo-signal', choices=list(EMO_SIGNALS), default='level',
						help='what the emotion channel backs up. '
							 '"level" (DEFAULT, unchanged behaviour): z = ν(d_s′), the absolute '
							 'valence of the user reaction. '
							 '"delta" (Tier-C): z = (ν(d_s′) - ν(d_parent))/2 -- the CHANGE in '
							 'valence, a potential-function shaping (Ng, Harada & Russell 1999) that '
							 'rewards moving the user from worse to better rather than never '
							 'triggering negative words. The /2 renormalises [-2,+2] back to '
							 '[-1,+1] so β_emo means the same thing in both arms. z = 0 where the '
							 'parent carries no emotion distribution (ν(∅) = 0 convention).')
	parser.add_argument('--emo_valence_table', '--emo-valence-table',
						choices=list(EMO_VALENCE_TABLES), default='soft',
						help='which mined w(e) table the emotion channel scores nu(d) with. '
							 '"soft" (DEFAULT, deployed): mined crediting every emotion its posterior '
							 'mass, T(e)+=d(e) -- matches what the planner consumes. '
							 '"argmax": the same corpus/base rate/shrinkage but crediting only '
							 'argmax_e Phi(e|u); retained as the ablation for that mismatch. Both are '
							 'mined on all 300 ANNOTATED dialogs, so evaluate on non-annotated data '
							 '(see REPLAY_DATA in scripts/run_paper_experiments.sh).')
	# Wednesday arms (analysis/FREEZE_NOTES.md §8). All default to the shipped behaviour.
	parser.add_argument('--terminal_on_failure', '--terminal-on-failure', dest='search_horizon',
						action='store_const', const='episode', default=argparse.SUPPRESS,
						help='end a simulated branch on -1.0 (turn limit / stall) as well as on +1.0. An alias '
							 'for --search_horizon episode, the one code path that implements it; off by default.')
	parser.add_argument('--aff_pool', '--aff-pool', action='store_true',
						help='[emomcts] AffPool: blend Q with a per-search (bucket, act) pool of task returns, '
							 'MC-RAVE weight N_pool/(N+N_pool+4*N*N_pool*bias^2). DEFAULT off.')
	parser.add_argument('--aff_pool_bias', '--aff-pool-bias', type=float, default=0.1,
						help='[emomcts --aff_pool] RAVE bias b; sweep {0.05, 0.1, 0.25} on non-eval dialogues.')
	parser.add_argument('--aff_pool_tau', '--aff-pool-tau', type=float, default=0.35,
						help='[emomcts --aff_pool] bucket threshold: parent nu < tau is bucket 1 (0.35 = D1 tau_med).')
	parser.add_argument('--aff_pool_key', '--aff-pool-key', choices=list(AFF_POOL_KEYS), default='affect',
						help='[emomcts --aff_pool] pool key: affect = (K0 bucket, act) = AffPool; act = (act) alone = ActPool.')
	parser.add_argument('--emo_centre', '--emo-centre', action='store_true',
						help='[emomcts] CenteredBias: beta*(Q_emo - mu) on expanded edges, mu = mean Q_emo over '
							 'the expanded siblings; unexpanded edges unchanged. DEFAULT off.')
	parser.add_argument('--emo_constraint_tau', '--emo-constraint-tau', type=float, default=None,
						help='[emomcts] Constrain: select (and decide at the root) only among actions with '
							 'Q_emo >= tau or N < m_warm. Unset (DEFAULT) = off. The spec formula has no beta '
							 'term: run with --beta_emo 0.0 for it; beta > 0 adds beta*Q_emo inside the mask.')
	parser.add_argument('--emo_constraint_m_warm', '--emo-constraint-m-warm', type=int, default=3,
						help='[emomcts --emo_constraint_tau] visits before an action can be masked.')
	# Cache fixes (analysis/phase1/p_depth.md). All default to the frozen cache.
	parser.add_argument('--cache_ended_children', '--cache-ended-children', action='store_true',
						help='file a generated reply that ends the search under its node at generation time, so '
							 'the cache can serve it. DEFAULT off: only a non-terminal re-entry fills the pool, so a '
							 'reply that ends the episode is never served (analysis/phase1/p_depth.md).')
	parser.add_argument('--cache_fresh_depth1', '--cache-fresh-depth1', action='store_true',
						help='never serve the search root\'s outgoing edges from the cache: every depth-1 visit '
							 'generates, and the depth-1 child keeps every reply in its pool. Deeper edges still '
							 'cache. DEFAULT off.')
	parser.add_argument('--cache_draw', '--cache-draw', choices=list(CACHE_DRAWS), default='uniform',
						help='how a cache hit picks among the cached replies. "uniform" (DEFAULT, unchanged); '
							 '"bucket": only replies generated under a parent on the current parent\'s side of '
							 '--cache_bucket_tau (uniform on a bucket miss); "kernel": weight '
							 'exp(-(nu_now - nu_gen)^2 / h^2); "bucket_kernel": the kernel within the bucket. '
							 'Generates nothing extra.')
	parser.add_argument('--cache_bucket_tau', '--cache-bucket-tau', type=float, default=None,
						help='[--cache_draw bucket*] parent nu threshold of the bucket draw. No default: required.')
	parser.add_argument('--cache_kernel_h', '--cache-kernel-h', type=float, default=None,
						help='[--cache_draw *kernel] bandwidth h of the kernel draw. No default: required.')
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)

	main(cmd_args)
