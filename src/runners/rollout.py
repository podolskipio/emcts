"""Self-play rollouts: produce episode records that ``metrics/run_metrics.py`` can score.

Unlike the response-comparison runners (``raw_prompting.py`` / ``gdpzero*.py``), this one plays
*full* dialogs from scratch: the chosen ``--algo`` picks the next system dialog act, the user
simulator replies, and we repeat until the game ends or the turn limit is hit. For every dialog
it records ``{did, task, success, num_turns, history, [deal_price, buyer_price, seller_price]}``
(see ``metrics/dialog_metrics.py`` for the schema).

``--algo`` selects how each system action is picked (see :func:`pick_action`):

  * ``llm_raw`` — single LLM call: ``argmax(planner.predict(state))`` (mirrors ``runners/raw_prompting``)
  * ``gdpzero`` — open-loop MCTS over LLM rollouts (``--num_mcts_sims`` / ``--max_realizations`` / ``--Q_0`` / ``--cpuct``)
  * ``emomcts`` — emotion-aware open-loop MCTS (``EmotionAwareMultiObjectiveQ``); needs an emotion-aware task (``--game emo_p4g``)

    cd src
    python runners/rollout.py --game cb                                          # llm_raw, all CB dialogs
    python runners/rollout.py --game p4g     --algo gdpzero --num_mcts_sims 20    # GDP-Zero MCTS planning
    python runners/rollout.py --game emo_p4g --algo emomcts --num_mcts_sims 20    # Emotion-aware MCTS
    python metrics/run_metrics.py --episodes outputs/rollout.pkl --max_turns 8

NOTE: ``llm_raw`` and ``gdpzero`` run end-to-end against the current games (their
``get_next_state(state, action[, mode])`` accepts 2-arg calls). ``emomcts`` requires a game that
produces ``EmotionAwareDialogSession`` states and an emotion classifier to label the simulated
user reactions (the ``emo_*`` tasks in ``_common.py``); pointing it at a plain task raises a
clear error.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import re
import pickle
import logging
import argparse

from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from mcts.mcts import OpenLoopMCTS, SEARCH_HORIZONS
from mcts.emotion_mcts import (
	EmotionAwareMultiObjectiveQ, AFF_POOL_KEYS, CACHE_DRAWS, EMO_SIGNALS, EMO_VALENCE_TABLES, EMOTION_VALENCE_TABLES, check_emo_signal_flags,
)
from utils import role_profiler, coupling
from runners._common import (
	make_backbone_model, make_emotion_classifier, build_agents, load_dialogs,
	load_p4g_personas, apply_seed, dump_emotion_records, add_common_args, finalize_args,
	setup_output_dir, build_subtree_records, write_subtree_ndjson,
	build_simlog_step_records, write_simlog_ndjson,
)

logger = logging.getLogger(__name__)

ALGOS = ("llm_raw", "gdpzero", "emomcts")
# p4g self-play draws its scenarios from the unannotated full corpus, not the annotated 300
# (see _read_p4g_full_csv). --data overrides it.
P4G_ROLLOUT_DATA = "data/p4g_personas/full_dialog.csv"
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------
# action selection (one system dialog-act index per turn)
# ---------------------------------------------------------------------------
def pick_action(algo, state, *, game, planner, configs, emotion_classifier) -> "Tuple[int, object]":
	"""Pick the next system dialog-act index for ``state`` using ``algo``.

	Returns ``(action, planner_used)`` where ``planner_used`` is the MCTS object that was
	searched, or ``None`` for ``llm_raw`` (no tree). The caller needs it to write the
	per-edge M2_emo / sigma_emo records into the episode -- the schema freeze says those
	fields are present on every edge of every run, and a rollout throws the tree away.

	``configs`` carries the MCTS hyper-parameters, the four emotion-channel knobs included,
	so the emomcts branch below builds the same planner ``runners/emomcts.py`` does and the
	SR/AT numbers reflect the emotion-aware MCTS actually being researched.
	"""
	if algo == "llm_raw":
		# one-shot LLM planner: take the chat planner's prior at face value (no search)
		prior, _v = planner.predict(state)
		return int(np.argmax(np.asarray(prior))), None

	if algo == "gdpzero":
		# open-loop MCTS over LLM rollouts (as in runners/gdpzero.py)
		dp = OpenLoopMCTS(game, planner, configs)
		if coupling.active():
			dp.rng = coupling.planner_rng(turn=len(state))  # --coupled_seeds: this turn's own draw stream
		for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="gdpzero"):
			dp.search(state)
		return int(np.argmax(np.asarray(dp.get_action_prob(state)))), dp

	if algo == "emomcts":
		# Emotion-aware planner: the parallel multi-objective Q, scoring actions by
		# Q + beta_emo*Q_emo + cpuct*P*sqrt(N)/(1+Nsa). beta_emo=0 recovers the task-only
		# open-loop search.
		dp = EmotionAwareMultiObjectiveQ(
			game, planner, configs, emotion_classifier,
			beta_emo=configs.beta_emo,
			emo_risk_lambda=configs.emo_risk_lambda,
			emo_signal=configs.emo_signal,
			emo_valence_table=configs.emo_valence_table,
			aff_pool=configs.aff_pool,
			aff_pool_bias=configs.aff_pool_bias,
			aff_pool_tau=configs.aff_pool_tau,
			aff_pool_key=configs.aff_pool_key,
			emo_centre=configs.emo_centre,
			emo_constraint_tau=configs.emo_constraint_tau,
			emo_constraint_m_warm=configs.emo_constraint_m_warm,
			cache_draw=configs.cache_draw,
			cache_bucket_tau=configs.cache_bucket_tau,
			cache_kernel_h=configs.cache_kernel_h,
		)
		if coupling.active():
			dp.rng = coupling.planner_rng(turn=len(state))  # --coupled_seeds: this turn's own draw stream
		for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="emomcts"):
			dp.search(state)
		return int(np.argmax(np.asarray(dp.get_action_prob(state)))), dp

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
				dlg_id=None, seed=None):
	"""Play one full episode with ``algo`` choosing each system action.

	Returns ``(final_session, subtree_records, simlog_records)``. ``subtree_records`` is the frozen
	NDJSON subtree schema (task 1.5): one record per edge of the tree, for every planned
	turn of this episode. The caller gzips it to one file per dialogue. Empty for
	``--algo llm_raw``, which builds no tree and therefore has no edges. ``simlog_records`` is
	the P-VAR simulation-step log (see SIMLOG_DIRNAME in runners/_common.py): the emomcts
	planner's per-step tape for every planned turn plus one "turn" record per realized turn.
	"""
	subtree_records = []
	simlog_records = []
	state = game.init_dialog(*scenario)
	# turn 0: the only valid move at the start is the greeting -> realize it directly
	valid0 = np.asarray(planner.get_valid_moves(state), dtype=float)
	greeting_idx = int(np.nonzero(valid0)[0][0]) if valid0.sum() > 0 else 0
	# EmotionAwarePersuasionGame.get_next_state returns (state, emotion); base returns state.
	# Normalize to the state so both shapes work.
	state = game.state_of(game.get_next_state(state, greeting_idx))
	simlog_records.append(_simlog_turn_record(game, None, state, dlg_id=dlg_id, turn=0, planned=False,
											  valence_table=configs.emo_valence_table))
	# this turn's system-utterance / user-simulator / emotion-classifier calls are already in the
	# profiler's numerator, so it has to be in the denominator too -- otherwise every calls/turn
	# figure is inflated by (turns+1)/turns (33% on a 4-turn episode).
	role_profiler.mark_turn()
	# then plan turn by turn until the game ends or we hit the limit
	while game.get_dialog_ended(state) == 0.0 and len(state) < max_turns:
		action, dp = pick_action(algo, state, game=game, planner=planner,
							  configs=configs, emotion_classifier=emotion_classifier)
		if dp is not None:
			subtree_records += build_subtree_records(
				dp, dlg_id=dlg_id, turn=len(state), root_state=state, seed=seed)
			simlog_records += build_simlog_step_records(dp, dlg_id=dlg_id, turn=len(state))
		root_state, turn = state, len(state)
		state = game.state_of(game.get_next_state(state, action))
		simlog_records.append(_simlog_turn_record(game, root_state, state, dlg_id=dlg_id, turn=turn, planned=True,
												  valence_table=configs.emo_valence_table, planner_used=dp))
		role_profiler.mark_turn()  # denominator for the per-role calls/turn (--profile_roles)
	return state, subtree_records, simlog_records


def _simlog_turn_record(game, root_state, new_state, *, dlg_id, turn, planned, valence_table, planner_used=None):
	"""The simlog "turn" record: the observed search-root affect, the act that was realized,
	and the user's real reaction to it. Reads states only -- no model calls, no random draws.

	``root_state`` is the observed dialogue the turn's tree was rooted at (None for the
	unplanned greeting); ``new_state`` is that dialogue after the realized system turn and the
	simulated user reply."""
	weights = EMOTION_VALENCE_TABLES[valence_table or "soft"]
	nu = lambda dist: float(sum(p * weights.get(e, 0.0) for e, p in dist.items())) if dist else 0.0
	dist_json = lambda dist: {str(e): float(p) for e, p in dist.items()} if dist else {}
	root_dist = root_state.predicted_distribution() if root_state is not None and root_state.history else None
	sys_rec, usr_rec = new_state.history[-2], new_state.history[-1]
	usr_dist = new_state.predicted_distribution()
	root_visits = {}
	if planner_used is not None and root_state is not None:
		root_key = planner_used._to_string_rep(root_state)
		root_visits = {planner_used.player.dialog_acts[a]: int(n) for a, n in planner_used.Nsa.get(root_key, {}).items()}
	record = {
		"record_type": "turn",
		"dlg_id": dlg_id,
		"turn_index": int(turn),
		"planned": bool(planned),
		"root_nu": nu(root_dist),
		"root_emotion_dist": dist_json(root_dist),
		"system_act": sys_rec[1],
		"system_utterance": sys_rec[-1],
		"user_act": usr_rec[1],
		"user_utterance": usr_rec[-1],
		"user_emotion_dist": dist_json(usr_dist),
		"user_nu": nu(usr_dist),
		"outcome": float(game.get_dialog_ended(new_state)),
		"root_visits": root_visits,
	}
	# --emo_constraint_tau: the feasible-set root decision beside the unrestricted argmax N
	if planner_used is not None and planner_used.root_decision:
		record["root_constraint"] = planner_used.root_decision
	return record


def make_episode(task, did, game, state, *, algo=None, subtree_log_path=None):
	ended = game.get_dialog_ended(state)
	episode = {
		"did": did,
		"task": task,
		"algo": algo,
		"success": bool(ended >= 1.0),
		"num_turns": len(state),
		"history": [list(t) for t in state.history],
		# pointer to this dialogue's frozen-schema subtree log (task 1.5). The records
		# themselves live in the gzipped NDJSON, not in this pickle. "" for llm_raw,
		# which builds no tree.
		"subtree_log": subtree_log_path or "",
	}
	if task == "cb":
		episode["buyer_price"] = state.buyer_price
		episode["seller_price"] = state.seller_price
		episode["deal_price"] = _extract_cb_price(state) if episode["success"] else None
	return episode


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def main(cmd_args):
	print(f"algo={cmd_args.algo}  saving to {cmd_args.output}")

	# See runners/emomcts.py: delta-valence must not be crossed with a non-default n-step
	# horizon, because the n-step return over deltas telescopes.
	check_emo_signal_flags(cmd_args.emo_signal)
	apply_seed(cmd_args)

	backbone_model, family = make_backbone_model(
		llm=cmd_args.llm,
		gen_sentences=cmd_args.gen_sentences,
		ollama_model=cmd_args.ollama_model,
		ollama_host=cmd_args.ollama_host,
		sglang_model=cmd_args.sglang_model,
	)
	# One classifier for the whole run (None unless --game is emotion-aware): every dialog's
	# game gets this same instance, so the run's utterance -> emotion records stay in one place.
	emotion_classifier = make_emotion_classifier(cmd_args.game, cmd_args.emotion_classifier, backbone_model)
	if cmd_args.algo == "emomcts" and emotion_classifier is None:
		raise ValueError(
			f"--algo emomcts needs an emotion-aware task with a classifier; "
			f"--game {cmd_args.game!r} has none (try --game emo_p4g)."
		)
	# Every dialog builds its own agents from these (see run_one_dialog), so two concurrent
	# workers never touch the same game / planner / user simulator.
	agent_kwargs = dict(
		# llm_raw keeps the model's built-in inference defaults; the MCTS open loop wants sampling on
		sys_inference_args={} if cmd_args.algo == "llm_raw" else None,
		llm_prior_topk=cmd_args.llm_prior_topk,
		logit_scoring=cmd_args.logit_scoring,
		explicit_value_labels=cmd_args.explicit_value_labels,
		success_criterion=cmd_args.p4g_success,
		# the rollout loop below stops at --max_turns; the game gets the same horizon. Search only
		# stops there under --search_horizon episode -- the legacy default keeps simulating past it
		# (analysis/phase1/SEARCH_HORIZON_BUG.md).
		max_conv_turns=cmd_args.max_turns,
		emotion_classifier=emotion_classifier,
	)

	# p4g rollouts are self-play: scenario is () and the corpus turns are never replayed, so the
	# dataset only supplies which participants to play against. The GDP-Zero pickle (TASKS
	# default) is the annotated 300 that w(e) is mined from; the full corpus is 717 further
	# dialogs with no dialog-act labels -- useless to the replay runners, but exactly right here,
	# and disjoint from the mining set so --p4g_persona cannot feed back a mined participant.
	if cmd_args.data is None and cmd_args.game.endswith("p4g"):
		cmd_args.data = P4G_ROLLOUT_DATA
	all_dialogs = load_dialogs(cmd_args.game, cmd_args)
	# {} unless --p4g_persona; {dialogue_id: persona text} otherwise (utils/p4g_personas.py)
	persona_texts = load_p4g_personas(cmd_args)

	configs = dotdict({
		"cpuct": cmd_args.cpuct,
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
	mcts_class_by_algo = {
		"llm_raw": "(none — llm_raw baseline)",
		"gdpzero": "OpenLoopMCTS",
		"emomcts": EmotionAwareMultiObjectiveQ.__name__,
	}
	setup_output_dir(cmd_args, runner_name="runners/rollout.py",
					 mcts_class=mcts_class_by_algo[cmd_args.algo],
					 mcts_args=configs)

	if cmd_args.profile_roles:
		role_profiler.enable()
		print("per-role cost profiling on (paper §W5 'before' row)")

	episodes = []
	cap = len(all_dialogs) if cmd_args.max_conv is None or cmd_args.max_conv < 0 else cmd_args.max_conv
	n = min(cap, len(all_dialogs))
	print(f"task={cmd_args.game}  {n} scenarios  (max_turns={cmd_args.max_turns})")
	pbar = tqdm(total=n, desc=f"rollout {cmd_args.game}/{cmd_args.algo}")

	# --num_workers > 1 plays several dialogs at once so SGLang can batch them: the runner is
	# otherwise one dependent chain of ~0.2s requests with the GPU idle in between (see the
	# '[cache:...] N% of wall' line). Threads, not processes — the time is spent waiting on HTTP.
	# Each dialog builds its own agents and each turn its own MCTS object, so concurrent workers
	# share only the backbone model (an HTTP client) and the emotion classifier.
	# Results carry their scenario index and are re-sorted on every dump, so the output pickle is
	# in scenario order regardless of which dialog finishes first.
	if cmd_args.coupled_seeds:
		missing = [need for need, ok in (("--llm sglang", cmd_args.llm == "sglang"), ("--seed", cmd_args.seed is not None),
										 ("--coupling_store", bool(cmd_args.coupling_store))) if not ok]
		if missing:
			raise SystemExit(f"--coupled_seeds needs {', '.join(missing)}")
	reply_store = coupling.ReplyStore(cmd_args.coupling_store) if cmd_args.coupled_seeds else None
	finished = []            # (scenario index, episode), guarded by results_lock
	results_lock = Lock()
	first_error = []         # first worker exception, re-raised after the pool drains (--raise_errors)

	def run_one_dialog(indexed_dialog):
		idx, dialog = indexed_dialog
		did = dialog["id"]
		# Agents are per dialog, so each worker thread owns its own: building them is pure
		# object construction (~0.04ms, no I/O), and --p4g_persona conditions the two objects
		# that role-play the persuadee (user simulator + the planner's value estimator) on THIS
		# dialogue's participant. persona_texts is {} without the flag (and misses the handful
		# of participants who skipped the survey), in which case persona is None and the prompts
		# are byte-identical to an unconditioned run.
		persona = persona_texts.get(did)
		game, _system, _user, planner = build_agents(
			cmd_args.game, backbone_model, family, persona=persona, **agent_kwargs)
		try:
			# --coupled_seeds: every LLM call and tree draw of this episode is shared with any run that uses
			# the same store and seed (utils/coupling.py); reply_store is None otherwise, a no-op
			with coupling.dialogue(reply_store, cmd_args.seed, did) as coupled:
				state, subtree_records, simlog_records = rollout_one(game, planner, cmd_args.algo, configs,
									 emotion_classifier, cmd_args.max_turns, dialog["scenario"],
									 dlg_id=did, seed=cmd_args.seed)
			# one gzipped NDJSON per dialogue, written once the episode is done so
			# concurrent workers never share a file.
			subtree_log_path = write_subtree_ndjson(subtree_records, cmd_args.output, did)
			# the P-VAR simulation-step log sits next to it (<run_dir>/simlog/<did>.ndjson.gz);
			# its path is derivable from the run dir, so the episode record is left unchanged.
			write_simlog_ndjson(simlog_records, cmd_args.output, did)
			episode = make_episode(cmd_args.game, did, game, state, algo=cmd_args.algo,
								   subtree_log_path=subtree_log_path)
			episode["persona"] = persona
			if coupled is not None:
				episode["coupling"] = {"store_hits": coupled.hits, "store_misses": coupled.misses}
		except Exception as e:
			logger.exception(f"rollout {did} failed: {e}")
			with results_lock:
				if not first_error:
					first_error.append(e)
				pbar.update(1)
			return
		with results_lock:
			finished.append((idx, episode))
			episodes[:] = [ep for _, ep in sorted(finished, key=lambda t: t[0])]
			if cmd_args.debug:
				game.display(state)
				print(f"  -> success={episode['success']}  turns={episode['num_turns']}")
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

	indexed_dialogs = list(enumerate(all_dialogs[:n]))
	workers = max(1, min(cmd_args.num_workers, n))
	if workers == 1:
		# unchanged sequential path, so single-worker runs stay identical to before
		for indexed_dialog in indexed_dialogs:
			run_one_dialog(indexed_dialog)
	else:
		print(f"rolling out {workers} dialogs concurrently (--num_workers {cmd_args.num_workers})")
		pool = ThreadPool(processes=workers)
		pool.map(run_one_dialog, indexed_dialogs)
		pool.close()
		pool.join()
	pbar.close()
	if first_error and cmd_args.raise_errors:
		raise first_error[0]

	# quick on-the-fly summary
	try:
		from metrics.dialog_metrics import compute_metrics, format_metrics
		print(format_metrics(compute_metrics(episodes, task=cmd_args.game, max_turns=cmd_args.max_turns)))
	except Exception:
		pass

	# emotion distribution + utterance->emotion records (only emomcts classifies; no-op otherwise)
	dump_emotion_records(emotion_classifier, cmd_args.output)
	print(f"done: {len(episodes)} episodes -> {cmd_args.output}")

	if cmd_args.profile_roles:
		role_profiler.report(label=f"{cmd_args.llm}/{cmd_args.algo}")
		role_profiler.dump(cmd_args.output, label=f"{cmd_args.llm}/{cmd_args.algo}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="self-play rollouts -> episode records for SR / AT / SL. On the p4g tasks "
					f"--data defaults to {P4G_ROLLOUT_DATA} (the 717 unannotated dialogs) rather "
					"than the annotated 300 the replay runners use; see _read_p4g_full_csv.")
	add_common_args(parser, default_output="outputs/rollout.pkl")  # --max_turns is defined there
	parser.add_argument("--max_conv", type=int, default=20, help="max scenarios to roll out (-1 for all)")
	parser.add_argument("--raise_errors", action="store_true", help="re-raise instead of skipping a failing rollout")
	parser.add_argument("--p4g_success", "--p4g-success", choices=["tag", "committed", "amount"], default="tag",
						help="[p4g] which [donate]-tagged persuadee turns end the episode -- and a search "
						     "branch -- in success (games/p4g_success.py). tag (DEFAULT, GDP-Zero): the tag "
						     "alone. committed: no deferral/hedge language in the turn ('I will consider "
						     "donating' is not a donation). amount: committed and an explicit amount named.")
	parser.add_argument("--profile_roles", action="store_true",
						help="record calls, latency and tokens in/out per role (policy prior, value "
						     "estimator, user simulator, system utterance model, emotion classifier) "
						     "and print/dump the cost table at the end. Use --num_workers 1 for it: "
						     "concurrent dialogs overlap, so latency per call stops being comparable.")
	parser.add_argument("--algo", choices=list(ALGOS), default="llm_raw",
						help="how to pick each system action (see pick_action in rollout.py)")
	# MCTS hyper-parameters (used by gdpzero + emomcts)
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="[--algo gdpzero|emomcts] MCTS simulations per turn")
	parser.add_argument("--max_realizations", type=int, default=3, help="[--algo gdpzero|emomcts] realizations sampled per state")
	parser.add_argument("--Q_0", type=float, default=0.0, help="[--algo gdpzero|emomcts] initial Q value for unvisited states")
	parser.add_argument("--cpuct", type=float, default=1.0, help="[--algo gdpzero|emomcts] UCT exploration constant")
	parser.add_argument('--search_horizon', '--search-horizon', choices=list(SEARCH_HORIZONS), default='legacy',
						help='which get_dialog_ended values end a simulated branch. "legacy" (DEFAULT, unchanged, GDP-Zero): only success is terminal, so search keeps expanding past the turn limit and after a verbatim stall -- states no real episode reaches. "episode": any non-zero get_dialog_ended is terminal and its value (+1 donate / -1 turn limit or stall) is backed up, so search stops where the episode loop stops. See analysis/phase1/SEARCH_HORIZON_BUG.md.')
	# Emotion-aware MCTS hyper-parameters (mirror runners/emomcts.py; only used when --algo emomcts).
	parser.add_argument('--beta_emo', type=float, default=0.0,
						help='[--algo emomcts] weight on the parallel Q_emo channel in PUCT '
							 '(Direction A — multi-objective parallel Q). '
							 '0.0 = no MultiObjectiveQ path; >0 routes through EmotionAwareMultiObjectiveQ '
							 'which scores actions by Q + β·Q_emo + cpuct·P·√N/(1+Nsa). Sweep {0.3, 0.7, 1.0}.')
	parser.add_argument('--emo_risk_lambda', '--emo-risk-lambda', type=float, default=0.0,
						help='[--algo emomcts] risk aversion on the emotion channel: selection uses '
							 'Q_emo - λ·σ_emo instead of Q_emo, σ_emo being the Welford standard '
							 'deviation of the z values on that edge. DEFAULT 0.0 reduces it to Q_emo '
							 'exactly (x - 0.0*σ is bit-identical to x). Not swept now.')
	parser.add_argument('--emo_signal', '--emo-signal', choices=list(EMO_SIGNALS), default='level',
						help='[--algo emomcts] what the emotion channel backs up. "level" (DEFAULT, '
							 'unchanged): z = ν(d_s′). "delta" (Tier-C): z = (ν(d_s′) - ν(d_parent))/2, '
							 'the change in valence -- potential-function shaping; the /2 renormalises '
							 'to [-1,+1] so β_emo is comparable across arms; z = 0 where the parent has '
							 'no emotion distribution.')
	parser.add_argument('--emo_valence_table', '--emo-valence-table',
						choices=list(EMO_VALENCE_TABLES), default='soft',
						help='[--algo emomcts] which mined w(e) table the emotion channel scores nu(d) with. '
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
	# Coupled seeds (utils/coupling.py). DEFAULT off.
	parser.add_argument('--coupled_seeds', '--coupled-seeds', action='store_true',
						help='common random numbers across arms: every LLM call is keyed by (seed, dialogue, exact '
							 'request, occurrence) in --coupling_store, so any run with the same store and seed that '
							 'sends the same request gets the same reply; tree draws use a RandomState per (seed, '
							 'dialogue, turn). Needs --llm sglang, --seed and --coupling_store. DEFAULT off.')
	parser.add_argument('--coupling_store', '--coupling-store', type=str, default=None,
						help='[--coupled_seeds] sqlite file shared by the runs being coupled.')
	# Cache fixes (analysis/phase1/p_depth.md). All default to the frozen cache.
	parser.add_argument('--cache_ended_children', '--cache-ended-children', action='store_true',
						help='[--algo gdpzero|emomcts] file a generated reply that ends the search under its node at generation time, so '
							 'the cache can serve it. DEFAULT off: only a non-terminal re-entry fills the pool, so a '
							 'reply that ends the episode is never served (analysis/phase1/p_depth.md).')
	parser.add_argument('--cache_fresh_depth1', '--cache-fresh-depth1', action='store_true',
						help='[--algo gdpzero|emomcts] never serve the search root\'s outgoing edges from the cache: every depth-1 visit '
							 'generates, and the depth-1 child keeps every reply in its pool. Deeper edges still '
							 'cache. DEFAULT off.')
	parser.add_argument('--cache_draw', '--cache-draw', choices=list(CACHE_DRAWS), default='uniform',
						help='[--algo emomcts] how a cache hit picks among the cached replies. "uniform" (DEFAULT, unchanged); '
							 '"bucket": only replies generated under a parent on the current parent\'s side of '
							 '--cache_bucket_tau (uniform on a bucket miss); "kernel": weight '
							 'exp(-(nu_now - nu_gen)^2 / h^2); "bucket_kernel": the kernel within the bucket. '
							 'Generates nothing extra.')
	parser.add_argument('--cache_bucket_tau', '--cache-bucket-tau', type=float, default=None,
						help='[--algo emomcts --cache_draw bucket*] parent nu threshold of the bucket draw. No default: required.')
	parser.add_argument('--cache_kernel_h', '--cache-kernel-h', type=float, default=None,
						help='[--algo emomcts --cache_draw *kernel] bandwidth h of the kernel draw. No default: required.')
	cmd_args = finalize_args(parser.parse_args())

	main(cmd_args)
