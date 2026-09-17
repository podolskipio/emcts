"""Shared corpus layer for the emotion-mining scripts: p4g and esc, one sequence shape.

Both mining scripts (``mine_emotion_donation_p4g.py``, ``mine_emotion_history_donation_p4g.py``)
need the same three things from a corpus: the user's emotions per turn, a binary session
outcome, and a split. This module produces them for either task so the scripts themselves stay
task-agnostic.

Everything downstream consumes ``EmotionSequence(dialog_id, emotions, succeeded)`` — the
per-task differences are resolved here and nowhere else.

The two tasks are not the same problem
--------------------------------------
Per EMOMCTS_ALGORITHM.md's "structural difference" table:

* **p4g** — emotion is *instrumental*. The outcome (``agree-donation``) is an annotated,
  behavioural fact that exists independently of any emotion label, so "which emotions precede
  donation" is a clean, non-circular question.
* **esc** — emotion is *terminal*: emotional change **is** the task, and ESConv ships no
  outcome annotation whatsoever (the dump has only emotion_type / problem_type / situation /
  dialog, and ``read_esc`` hardcodes every seeker turn to ``U_FeelTheSame``). There is no
  donation-analogue to predict.

So for esc the outcome is **defined internally as improvement**: the session succeeds if the
seeker's affect over the last ``--esc_window`` turns is better than over the first
``--esc_window``. To keep that from being tautological, two rules are enforced here:

1. The turns used to compute the outcome are **excluded from the features**. ``emotions`` holds
   only the turns before the closing window, so the model predicts a later, disjoint stretch of
   the conversation rather than describing itself.
2. The outcome uses ``TEXTBOOK_VALENCE`` — a fixed affect sign — and never the mined weights.
   Labelling with anything we are fitting would manufacture the result.

This makes esc weights answer "which early emotional trajectories precede improvement", which
is a weaker and more self-referential claim than p4g's. Both the predictor and the label come
from the same classifier, so read esc numbers with that in mind; they are not evidence that a
strategy *caused* improvement.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402

EMOTION_ORDER = [
	Emotions.Happiness, Emotions.Sadness, Emotions.Fear, Emotions.Anger,
	Emotions.Surprise, Emotions.Disgust, Emotions.Contempt, Emotions.Neutral,
]
EMO_KEYS = [str(e) for e in EMOTION_ORDER]

# Kept in sync with runners/_common.P4G_BAD_DIALOGS: dialogs the OpenAI content filter
# rejects, which the eval runners skip. Duplicated (not imported) because _common pulls in
# the game/player stack, and these scripts must stay runnable with no backbone configured.
P4G_BAD_DIALOGS = {"20180808-024552_152_live", "20180723-100140_767_live", "20180825-080802_964_live"}

# Plain affective valence, used ONLY to define the esc improvement label. Deliberately the
# textbook sign ordering rather than anything mined, so the label cannot inherit the very
# effect the scripts are trying to measure.
TEXTBOOK_VALENCE = {
	"happiness": +1.0, "surprise": 0.0, "neutral": 0.0,
	"sadness": -1.0, "fear": -1.0, "anger": -1.0, "disgust": -1.0, "contempt": -1.0,
}

TASK_DEFAULTS = {
	"p4g": {"data": "data/p4g/300_dialog_turn_based.pkl",
	        "cache": "outputs/emotion_donation_emocache.json"},
	"esc": {"data": "data/esc/esc-train.txt",
	        "cache": "outputs/esc_emotion_emocache.json"},
}


@dataclass
class EmotionSequence:
	"""One session reduced to what the tallies need: user emotions in order, and its outcome.

	For esc, ``emotions`` excludes the closing window that defined ``succeeded`` — see the
	module docstring.
	"""
	dialog_id: str
	emotions: list[str]
	succeeded: bool
	# Full per-turn distributions, aligned 1:1 with ``emotions``. Populated only when
	# load_sequences(..., soft=True); empty list in argmax mode so the hard path is untouched.
	distributions: list[dict] = field(default_factory=list)
	# Per-unit tally weights, aligned 1:1 with ``emotions``. Empty = every unit weighs 1 (the shipped
	# behaviour). Filled by load_sequences(..., turn_weighting="last"|"recency").
	weights: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Reading raw user utterances per session
# --------------------------------------------------------------------------------------

# Persuadee acts that state the donation decision. With pre_decision_only, features stop before
# the first turn carrying any of them, for donors and refusers alike (analysis/thu/construct_test.md
# §3b: about 2/3 of w(happiness) was fitted on turns at or after the decision).
P4G_DECISION_ACTS = {"agree-donation", "disagree-donation", "disagree-donation-more",
                     "provide-donation-amount", "confirm-donation"}


def first_decision_turn(dialog: dict) -> int | None:
	"""Index of the first turn whose persuadee labels include a decision act, or None."""
	for i, label in enumerate(dialog["label"]):
		if P4G_DECISION_ACTS & set(label.get("ee", []) if isinstance(label, dict) else []):
			return i
	return None


TURN_WEIGHTINGS = ("uniform", "last", "recency")


def p4g_unit_weights(path: str, pre_decision_only: bool, turn_weighting: str, gamma: float) -> dict[str, list[float]]:
	"""Per-utterance weights aligned with ``read_p4g_sessions(path, pre_decision_only)``'s utterance lists.

	k = number of persuadee turns between an utterance's turn and the reference turn, where the reference
	is the last turn kept: the turn just before the first decision act (pre_decision_only), or the
	dialog's last turn otherwise. Every utterance of a turn shares its turn's weight.
	  last     1 if k == 0 else 0     -- only the final kept turn: the state the planner asks about
	  recency  gamma ** k              -- EWMA-style: the final kept turn dominates, early turns fade
	"""
	with open(path, "rb") as f:
		dialogs: dict = pickle.load(f)
	out = {}
	for did, dialog in dialogs.items():
		stop = first_decision_turn(dialog) if pre_decision_only else None
		kept = [(i, [(u or "").strip() for u in turn.get("ee", []) if (u or "").strip()])
				for i, turn in enumerate(dialog["dialog"]) if stop is None or i < stop]
		kept = [(i, utts) for i, utts in kept if utts]
		ws = []
		for rank, (i, utts) in enumerate(kept):
			k = len(kept) - 1 - rank
			w = (1.0 if k == 0 else 0.0) if turn_weighting == "last" else gamma ** k
			ws += [w] * len(utts)
		out[did] = ws
	return out


def read_p4g_sessions(path: str, pre_decision_only: bool = False) -> list[tuple[str, list[str], bool]]:
	"""``(dialog_id, persuadee_utterances, donated)`` per dialog, in pickle order.

	``pre_decision_only`` keeps only utterances from turns strictly before the first decision act
	(``P4G_DECISION_ACTS``). The outcome label still comes from the whole dialog. Default False is
	the shipped behaviour.
	"""
	with open(path, "rb") as f:
		dialogs: dict = pickle.load(f)
	sessions = []
	for did, dialog in dialogs.items():
		stop = first_decision_turn(dialog) if pre_decision_only else None
		utterances = [(utt or "").strip()
		              for i, turn in enumerate(dialog["dialog"])
		              if stop is None or i < stop
		              for utt in turn.get("ee", [])
		              if (utt or "").strip()]
		donated = any(tag == "agree-donation"
		              for label in dialog["label"]
		              for tag in label.get("ee", []))
		sessions.append((did, utterances, donated))
	return sessions


def read_esc_sessions(path: str) -> list[tuple[str, list[str], None]]:
	"""``(dialog_id, seeker_utterances, None)`` per session — ESConv ships no outcome label.

	Consecutive same-speaker turns are kept as separate utterances, matching how the p4g reader
	treats the ``ee`` list, so turn counts stay comparable across tasks.
	"""
	sessions = []
	with open(path) as f:
		for i, line in enumerate(f):
			line = line.strip()
			if not line:
				continue
			dialog = json.loads(line)
			utterances = [(turn.get("text") or "").strip()
			              for turn in dialog["dialog"]
			              if turn.get("speaker") == "usr" and (turn.get("text") or "").strip()]
			sessions.append((dialog.get("id", f"esc-{i}"), utterances, None))
	return sessions


# --------------------------------------------------------------------------------------
# Emotion labelling
# --------------------------------------------------------------------------------------

def build_emotion_cache(utterances: list[str], cache_path: str) -> dict[str, str]:
	"""Map each user utterance to its argmax HF emotion, memoised on disk.

	Labels are a pure function of the text, so the cache may span the whole corpus regardless
	of the split — caching a held-out utterance leaks nothing into the mined weights.
	"""
	cache: dict[str, str] = {}
	if os.path.exists(cache_path):
		with open(cache_path) as f:
			cache = json.load(f)

	uncached = sorted({utt for utt in utterances if utt not in cache})
	if uncached:
		print(f"labelling {len(uncached)} new utterances with the HF classifier ...")
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		classifier = HFEmotionClassifier()
		for utt in uncached:
			distribution = classifier.predict_distribution_from_utterance(utt)
			cache[utt] = str(max(distribution, key=distribution.get))
		os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
		with open(cache_path, "w") as f:
			json.dump(cache, f)

	print(f"emotion cache: {len(cache)} utterances ({len(uncached)} new)")
	return cache


def build_emotion_distribution_cache(utterances: list[str], cache_path: str) -> dict[str, dict]:
	"""Map each user utterance to its FULL emotion distribution, memoised on disk.

	The argmax cache above throws away everything except the winning label. Soft mining
	(``T(e) += d(e)``, ``S(e) += d(e)*y``) needs the whole vector, because deployment scores
	the emotion channel as ``nu(d) = sum_e d(e)*w(e)`` over the full softmax — so mining by
	argmax and deploying by expectation disagree about what an emotion label *is*.

	Same caching contract as the argmax cache: labels are a pure function of the text, so the
	cache may span the whole corpus regardless of split without leaking anything into w(e).
	"""
	cache: dict[str, dict] = {}
	if os.path.exists(cache_path):
		with open(cache_path) as f:
			cache = json.load(f)

	uncached = sorted({utt for utt in utterances if utt not in cache})
	if uncached:
		print(f"labelling {len(uncached)} new utterances (full distribution) ...")
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		classifier = HFEmotionClassifier()
		for i, utt in enumerate(uncached, 1):
			dist = classifier.predict_distribution_from_utterance(utt)
			cache[utt] = {str(e): float(p) for e, p in dist.items()}
			if i % 500 == 0:
				print(f"  {i}/{len(uncached)}", flush=True)
		os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
		with open(cache_path, "w") as f:
			json.dump(cache, f)

	print(f"distribution cache: {len(cache)} utterances ({len(uncached)} new)")
	return cache


# --------------------------------------------------------------------------------------
# Outcomes and splits
# --------------------------------------------------------------------------------------

def mean_valence(emotions: list[str]) -> float:
	if not emotions:
		return 0.0
	return sum(TEXTBOOK_VALENCE.get(e, 0.0) for e in emotions) / len(emotions)


ESC_OUTCOMES = ("final_positive", "improved")


def esc_outcome(emotions: list[str], window: int, kind: str) -> bool | None:
	"""Binary session outcome for esc. Returns None if the session is too short to score.

	``final_positive`` (default) — the closing window's mean affect is above neutral, i.e. the
	seeker *ended in a good state*. This is the closer analogue of the game's ``U_Solved`` and
	of ESC's terminal objective.

	``improved`` — the closing window beats the opening window. Intuitive, but **badly
	confounded**: a seeker who opens distressed has room to improve while one who opens content
	does not, so the label is mechanically anti-correlated with the very early emotions used as
	features. On this corpus it inverts every weight (sadness +0.25, happiness -0.35) and pushes
	the base rate to 0.78 — the floor effect EMOMCTS_ALGORITHM.md warns about when it notes that
	"sadness is the patient's baseline". Kept for comparison, not recommended.

	Both leave the closing window out of the features (see ``load_sequences``), so neither model
	is scored on the turns that defined its own label.
	"""
	if len(emotions) < 2 * window + 1:
		return None
	if kind == "final_positive":
		return mean_valence(emotions[-window:]) > 0.0
	if kind == "improved":
		return mean_valence(emotions[-window:]) > mean_valence(emotions[:window])
	raise ValueError(f"unknown esc outcome {kind!r}; expected one of {ESC_OUTCOMES}")


def select_p4g_holdout(sessions: list, holdout_first: int) -> tuple[list, list[str]]:
	"""Split p4g the way the replay runners do: drop bad dialogs, THEN take the first N.

	Returns ``(sessions_to_mine, held_out_eval_ids)``. The filter ordering matters — 3 bad
	dialogs sit at raw indices 4, 7 and 19, so slicing the raw pickle order would hold out 3
	fewer eval dialogs than the runner actually visits.
	"""
	replayable = [did for did, _, _ in sessions if did not in P4G_BAD_DIALOGS]
	eval_ids = replayable[:holdout_first]
	held_out = set(eval_ids)
	return [s for s in sessions if s[0] not in held_out], eval_ids


# --------------------------------------------------------------------------------------

def load_sequences(task: str, data_path: str | None, cache_path: str | None,
                   holdout_first: int = 0, esc_window: int = 2,
                   esc_outcome_kind: str = "final_positive",
                   soft: bool = False, dist_cache_path: str | None = None,
                   exclude_ids: set | None = None, assert_no_exclusions: bool = False,
                   pre_decision_only: bool = False, turn_weighting: str = "uniform",
                   recency_gamma: float = 0.7):
	"""Load either corpus into ``[EmotionSequence]`` plus a metadata dict describing the load."""
	data_path = data_path or TASK_DEFAULTS[task]["data"]
	cache_path = cache_path or TASK_DEFAULTS[task]["cache"]

	if task == "p4g":
		sessions = read_p4g_sessions(data_path, pre_decision_only=pre_decision_only)
	elif task == "esc":
		if pre_decision_only:
			raise ValueError("pre_decision_only is defined for p4g (annotated decision acts); esc has none")
		sessions = read_esc_sessions(data_path)
	else:
		raise ValueError(f"unknown task {task!r}; expected 'p4g' or 'esc'")
	print(f"loaded {len(sessions)} {task} sessions from {data_path}")

	eval_ids: list[str] = []
	if task == "p4g":
		sessions, eval_ids = select_p4g_holdout(sessions, holdout_first)
		if holdout_first:
			print(f"split: holding out {len(eval_ids)} eval dialogs, mining on {len(sessions)}")
	elif holdout_first:
		# esc has no replay-eval convention to mirror; refuse rather than invent one.
		raise ValueError("--holdout_first is only defined for p4g (it mirrors the p4g replay "
		                 "runners' dialog ordering). Leave it at 0 for esc.")

	# Explicit exclusion set (leakage insurance): drop these dialog ids from the mining corpus
	# regardless of ordering. Independent of --holdout_first, which slices by position.
	# Record how many dialogs the id list actually removed, not how long the list was: the
	# production run passes the eval ids as a tripwire that must remove 0.
	n_before_exclusion = len(sessions)
	if exclude_ids:
		sessions = [s for s in sessions if s[0] not in exclude_ids]
		print(f"excluded {n_before_exclusion - len(sessions)} dialogs by id (leakage insurance); "
		      f"{len(sessions)} remain")
	n_excluded_by_id = n_before_exclusion - len(sessions)
	if assert_no_exclusions:
		if not exclude_ids:
			raise SystemExit("--assert_no_exclusions needs a non-empty --exclude_ids list; "
			                 "without one the assertion checks nothing")
		if n_excluded_by_id:
			raise SystemExit(f"exclude_ids removed {n_excluded_by_id} dialogs; the eval set "
			                 f"overlaps the mining corpus")

	if turn_weighting not in TURN_WEIGHTINGS:
		raise ValueError(f"turn_weighting must be one of {TURN_WEIGHTINGS}")
	unit_w = {}
	if turn_weighting != "uniform":
		if task != "p4g" or not soft:
			raise ValueError("turn_weighting is defined for p4g soft mining only")
		unit_w = p4g_unit_weights(data_path, pre_decision_only, turn_weighting, recency_gamma)

	cache = build_emotion_cache([u for _, utts, _ in sessions for u in utts], cache_path)
	dist_cache = {}
	if soft:
		dist_cache = build_emotion_distribution_cache(
			[u for _, utts, _ in sessions for u in utts],
			dist_cache_path or cache_path.replace(".json", "_dist.json"))

	sequences: list[EmotionSequence] = []
	n_dropped = 0
	for did, utterances, outcome in sessions:
		emotions = [cache[u] for u in utterances if u in cache]
		dists = [dist_cache[u] for u in utterances if u in dist_cache] if soft else []
		weights = []
		if unit_w:
			assert len(unit_w[did]) == len(utterances), did
			weights = [w for u, w in zip(utterances, unit_w[did]) if u in dist_cache]
			if not any(weights):
				n_dropped += 1
				continue
		if not emotions:
			n_dropped += 1
			continue
		if task == "esc":
			result = esc_outcome(emotions, esc_window, esc_outcome_kind)
			if result is None:              # too short for disjoint windows
				n_dropped += 1
				continue
			# features stop before the window that defined the label
			emotions, outcome = emotions[:-esc_window], result
			if soft:
				dists = dists[:len(emotions)]
		sequences.append(EmotionSequence(did, emotions, bool(outcome), dists, weights))

	if n_dropped:
		print(f"dropped {n_dropped} sessions (no labelled turns"
		      f"{', or shorter than 2*esc_window+1' if task == 'esc' else ''})")

	meta = {
		"task": task, "data": data_path, "cache": cache_path,
		"n_sessions": len(sequences), "n_turns": sum(len(s.emotions) for s in sequences),
		"holdout_first": holdout_first, "n_eval_holdout": len(eval_ids),
		"eval_dialog_ids": eval_ids,
		"assignment": "soft" if soft else "argmax",
		"n_exclude_ids_supplied": len(exclude_ids) if exclude_ids else 0,
		"n_excluded_by_id": n_excluded_by_id,
		"n_sessions_before_exclusion": n_before_exclusion,
		"assert_no_exclusions": assert_no_exclusions,
		"pre_decision_only": pre_decision_only,
		"turn_weighting": turn_weighting,
		"recency_gamma": recency_gamma if turn_weighting == "recency" else None,
	}
	if task == "esc":
		meta["esc_window"] = esc_window
		meta["esc_outcome"] = esc_outcome_kind
		meta["outcome"] = f"internally defined: {esc_outcome_kind} over the closing window"
	else:
		meta["outcome"] = "agree-donation annotation"
	return sequences, meta


def add_corpus_args(parser) -> None:
	"""Shared corpus flags, so both mining scripts expose an identical interface."""
	parser.add_argument("--task", choices=["p4g", "esc"], default="p4g")
	parser.add_argument("--data", default=None, help="defaults to the task's corpus")
	parser.add_argument("--cache", default=None, help="defaults to the task's emotion cache")
	parser.add_argument("--esc_window", type=int, default=2,
	                    help="[esc] turns at each end used to define the outcome; the closing "
	                         "window is always excluded from the features.")
	parser.add_argument("--esc_outcome", choices=ESC_OUTCOMES, default="final_positive",
	                    help="[esc] how to define session success. 'improved' is baseline-"
	                         "confounded - see esc_outcome().")
