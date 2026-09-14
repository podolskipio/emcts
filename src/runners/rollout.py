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
from mcts.mcts import OpenLoopMCTS
from mcts.emotion_mcts import EmotionAwareMultiObjectiveQ, EMO_SIGNALS, EMO_VALENCE_TABLES, check_emo_signal_flags
from utils import role_profiler
from runners._common import (
	make_backbone_model, make_emotion_classifier, build_agents, load_dialogs,
	load_p4g_personas, apply_seed, dump_emotion_records, add_common_args, finalize_args,
	setup_output_dir, build_subtree_records, write_subtree_ndjson,
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
		)
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

	Returns ``(final_session, subtree_records)``. ``subtree_records`` is the frozen
	NDJSON subtree schema (task 1.5): one record per edge of the tree, for every planned
	turn of this episode. The caller gzips it to one file per dialogue. Empty for
	``--algo llm_raw``, which builds no tree and therefore has no edges.
	"""
	subtree_records = []
	state = game.init_dialog(*scenario)
	# turn 0: the only valid move at the start is the greeting -> realize it directly
	valid0 = np.asarray(planner.get_valid_moves(state), dtype=float)
	greeting_idx = int(np.nonzero(valid0)[0][0]) if valid0.sum() > 0 else 0
	# EmotionAwarePersuasionGame.get_next_state returns (state, emotion); base returns state.
	# Normalize to the state so both shapes work.
	state = game.state_of(game.get_next_state(state, greeting_idx))
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
		state = game.state_of(game.get_next_state(state, action))
		role_profiler.mark_turn()  # denominator for the per-role calls/turn (--profile_roles)
	return state, subtree_records


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
		# the rollout loop below stops at --max_turns; give the search the same horizon so it
		# does not simulate branches past the point the dialogue can reach.
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
			state, subtree_records = rollout_one(game, planner, cmd_args.algo, configs,
								 emotion_classifier, cmd_args.max_turns, dialog["scenario"],
								 dlg_id=did, seed=cmd_args.seed)
			# one gzipped NDJSON per dialogue, written once the episode is done so
			# concurrent workers never share a file.
			subtree_log_path = write_subtree_ndjson(subtree_records, cmd_args.output, did)
			episode = make_episode(cmd_args.game, did, game, state, algo=cmd_args.algo,
								   subtree_log_path=subtree_log_path)
			episode["persona"] = persona
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
	add_common_args(parser, default_output="outputs/rollout.pkl")
	parser.add_argument("--max_turns", type=int, default=10, help="hard cap on dialog turns per episode")
	parser.add_argument("--max_conv", type=int, default=20, help="max scenarios to roll out (-1 for all)")
	parser.add_argument("--raise_errors", action="store_true", help="re-raise instead of skipping a failing rollout")
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
	cmd_args = finalize_args(parser.parse_args())

	main(cmd_args)
