import os
import sys
import csv
import json
import gzip
import pickle
from collections import Counter
import logging

# allow running these files directly (python src/runners/<x>.py) by putting `src/` on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# repo root is two levels up from this file (src/runners/_common.py); used to resolve relative
# --data / default_data paths so the runners load data/<task>/... regardless of CWD.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_data_path(path):
	"""Resolve a ``--data`` value to something a reader can open.

	* ``hf:<repo>...`` URIs pass through unchanged.
	* Absolute paths pass through unchanged.
	* Relative paths are resolved against the repo root (so ``data/esc/esc-valid.txt`` works
	  from any working directory). If that path doesn't exist but the CWD-relative version
	  does, fall back to it for back-compat with the previous behaviour.
	"""
	if path is None or path.startswith("hf:") or os.path.isabs(path):
		return path
	rooted = os.path.join(REPO_ROOT, path)
	if os.path.exists(rooted):
		return rooted
	if os.path.exists(path):
		return os.path.abspath(path)
	return rooted  # let the reader raise the FileNotFoundError with a useful path

import numpy as np

from utils.sessions import DialogSession
from utils.utils import dotdict
from utils.gen_models import (
	OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel, SGLangChatModel,
)
from utils.prompt_examples import EXP_DIALOG, ESConv_EXP_DIALOG, CB_EXP_DIALOG

from games import PersuasionGame, EmotionAwarePersuasionGame, EmotionalSupportGame, CBGame
from emotion_classifiers.llm_emotion import P4GLLMEmotionClassifier
from utils.hf_loaders import HF_PREFIX, read_p4g_hf, read_esc_hf, read_cb_hf
from players.p4g_players import (
	PersuaderModel, PersuaderChatModel, PersuadeeModel, PersuadeeChatModel,
	P4GSystemPlanner, P4GChatSystemPlanner,
)
from players.esc_players import (
	TherapistModel, TherapistChatModel, PatientModel, PatientChatModel,
	ESCSystemPlanner, ESCChatSystemPlanner,
)
from players.cb_players import (
	BuyerModel, BuyerChatModel, SellerModel, SellerChatModel,
	CBSystemPlanner, CBChatSystemPlanner,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dataset readers: each returns a list of normalized dialogs
#   {"id": str, "scenario": tuple (positional args for game.init_dialog), "turns": [Turn, ...]}
# where Turn = {"sys_da": str, "sys_utt": str, "usr_da": str, "usr_utt": str}
# ---------------------------------------------------------------------------
def _iter_json_objects(path):
	"""Yield top-level JSON objects from a file that is either a JSON array, JSON-lines,
	or several pretty-printed JSON objects concatenated together."""
	with open(path, "r", encoding="utf-8") as f:
		text = f.read().strip()
	# fast path: a single JSON array
	try:
		obj = json.loads(text)
		if isinstance(obj, list):
			yield from obj
			return
		yield obj
		return
	except json.JSONDecodeError:
		pass
	# fallback: many JSON values back-to-back (handles both JSON-lines and pretty-printed)
	dec = json.JSONDecoder()
	idx = 0
	n = len(text)
	while idx < n:
		while idx < n and text[idx] in " \t\r\n":
			idx += 1
		if idx >= n:
			break
		obj, end = dec.raw_decode(text, idx)
		yield obj
		idx = end


def _collapse_segments(raw_turns, sys_key="sys", usr_key="usr"):
	"""Collapse consecutive same-speaker turns into segments [(speaker, text, da_or_None), ...]."""
	segments = []
	for t in raw_turns:
		speaker = t.get("speaker")
		text = (t.get("text") or "").strip()
		da = t.get("strategy")
		if not text:
			continue
		if segments and segments[-1][0] == speaker:
			prev_speaker, prev_text, prev_da = segments[-1]
			segments[-1] = (prev_speaker, f"{prev_text} {text}".strip(), da or prev_da)
		else:
			segments.append((speaker, text, da))
	return segments


def _pair_segments(segments, sys_key="sys", usr_key="usr"):
	"""From a [(speaker, text, da), ...] list build aligned (sys_segment, usr_segment) turns,
	skipping any leading user turn / trailing unpaired system turn."""
	turns = []
	i = 0
	while i + 1 < len(segments):
		if segments[i][0] != sys_key:
			i += 1
			continue
		if segments[i + 1][0] != usr_key:
			i += 1
			continue
		turns.append((segments[i], segments[i + 1]))
		i += 2
	return turns


# --- P4G (GDP-Zero's data/p4g/300_dialog_turn_based.pkl) -------------------
_P4G_USER_DA_MAP = {
	"disagree-donation": PersuasionGame.U_NoDonation,
	"negative-reaction-to-donation": PersuasionGame.U_NegativeReaction,
	"positive-reaction-to-donation": PersuasionGame.U_PositiveReaction,
	"agree-donation": PersuasionGame.U_Donate,
}
# dialogs with content that OpenAI content filters reject (kept from GDP-Zero)
P4G_BAD_DIALOGS = {"20180808-024552_152_live", "20180723-100140_767_live", "20180825-080802_964_live"}


def _resolve_p4g_sys_da(raw_label, system_dialog_acts):
	"""Pick a system DA from a raw P4G label; fall back to "other" if unknown."""
	if raw_label and raw_label in system_dialog_acts:
		return raw_label
	return "other"


def _read_p4g_pickle(path, system_dialog_acts):
	with open(path, "rb") as f:
		all_dialogs = pickle.load(f)
	out = []
	for did, dialog in all_dialogs.items():
		if did in P4G_BAD_DIALOGS:
			continue
		turns = []
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn.get("ee", [])) == 0:
				break
			sys_utt = " ".join(turn["er"]).strip()
			usr_utt = " ".join(turn["ee"]).strip()
			# system DA: take one of the labelled DAs that we know about, else "other"
			sys_das = set(dialog["label"][t]["er"])
			hit = sys_das.intersection(system_dialog_acts)
			sys_da = list(hit)[-1] if hit else "other"
			# user DA: map dataset label -> game DA
			raw_usr_da = dialog["label"][t]["ee"][-1]
			usr_da = _P4G_USER_DA_MAP.get(raw_usr_da, PersuasionGame.U_Neutral)
			turns.append({
				"sys_da": sys_da, "sys_utt": sys_utt,
				"usr_da": usr_da, "usr_utt": usr_utt,
			})
		if turns:
			out.append({"id": did, "scenario": (), "turns": turns})
	return out


def _read_p4g_jsonl(path, system_dialog_acts):
	"""Read the ESC/CB-style JSON-lines variant produced by ``runners/convert_p4g_to_jsonl.py``."""
	out = []
	for i, dialog in enumerate(_iter_json_objects(path)):
		did = dialog.get("id", f"p4g-{i}")
		if did in P4G_BAD_DIALOGS:
			continue
		segments = _collapse_segments(dialog["dialog"])
		turns = []
		for (sp_s, txt_s, da_s), (sp_u, txt_u, da_u) in _pair_segments(segments):
			sys_da = _resolve_p4g_sys_da(da_s, system_dialog_acts)
			usr_da = _P4G_USER_DA_MAP.get(da_u or "", PersuasionGame.U_Neutral)
			turns.append({"sys_da": sys_da, "sys_utt": txt_s, "usr_da": usr_da, "usr_utt": txt_u})
		if turns:
			out.append({"id": did, "scenario": (), "turns": turns})
	return out


# --- PersuasionForGood full corpus (data/p4g_personas/full_dialog.csv: one row per utterance,
#     columns Unit / Turn / B4 (0 = persuader, 1 = persuadee) / B2 (dialogue id)) ---
ANNOTATED_P4G_PKL = "data/p4g/300_dialog_turn_based.pkl"


def _annotated_p4g_ids():
	"""Dialogue ids of the annotated 300 (the GDP-Zero pickle), for excluding them below."""
	with open(resolve_data_path(ANNOTATED_P4G_PKL), "rb") as f:
		return set(pickle.load(f).keys())


def _read_p4g_full_csv(path, system_dialog_acts):
	"""Read the full 1017-dialogue P4G corpus, minus the annotated 300.
	The annotated 300 are dropped because w(e) is mined from them (see
	``emotion_mining/build_p4g_rollout_evalset.py``); evaluating on the remaining 717 keeps the
	rollout set disjoint from the mining corpus by construction, personas included.
	"""
	annotated = _annotated_p4g_ids()
	# {did: {(turn, speaker): [unit, ...]}} -- a speaker occasionally has several rows per turn
	by_dialog = {}
	with open(path, newline="", encoding="utf-8") as f:
		for row in csv.DictReader(f):
			did = row["B2"]
			turn_key = (int(row["Turn"]), int(row["B4"]))
			by_dialog.setdefault(did, {}).setdefault(turn_key, []).append(str(row["Unit"]).strip())

	n_total = len(by_dialog)
	out = []
	for did, units in by_dialog.items():
		if did in annotated or did in P4G_BAD_DIALOGS:
			continue
		turns = []
		for t in sorted({turn for turn, _spk in units}):
			sys_utt = " ".join(u for u in units.get((t, 0), []) if u).strip()
			usr_utt = " ".join(u for u in units.get((t, 1), []) if u).strip()
			if not sys_utt or not usr_utt:
				continue  # the corpus ends on a half turn now and then; self-play ignores turns anyway
			turns.append({
				# unannotated corpus: no dialog acts to map, so the neutral fallbacks both readers use
				"sys_da": "other", "sys_utt": sys_utt,
				"usr_da": PersuasionGame.U_Neutral, "usr_utt": usr_utt,
			})
		if turns:
			out.append({"id": did, "scenario": (), "turns": turns})
	print(f"p4g full corpus: {n_total} dialogs, {n_total - len(out)} dropped "
	      f"(annotated 300 + bad ids + empty) -> {len(out)} non-annotated scenarios")
	return out


def read_p4g(path, system_dialog_acts):
	if path.startswith(HF_PREFIX):
		return read_p4g_hf(path, system_dialog_acts)
	# JSON-lines variant produced by convert_p4g_to_jsonl.py; the pickle is the GDP-Zero original
	if path.endswith(".pkl"):
		return _read_p4g_pickle(path, system_dialog_acts)
	# data/p4g_personas/full_dialog.csv -- the unannotated full corpus, for self-play rollouts
	if path.endswith(".csv"):
		return _read_p4g_full_csv(path, system_dialog_acts)
	return _read_p4g_jsonl(path, system_dialog_acts)


# --- ESConv (DPDP's esc-valid.txt: JSON-lines of {emotion_type, problem_type, situation, dialog:[{text,speaker,strategy?}]}) ---
def read_esc(path, system_dialog_acts):
	if path.startswith(HF_PREFIX):
		return read_esc_hf(path, system_dialog_acts)
	out = []
	for i, dialog in enumerate(_iter_json_objects(path)):
		segments = _collapse_segments(dialog["dialog"])
		turns = []
		for (sp_s, txt_s, da_s), (sp_u, txt_u, _da_u) in _pair_segments(segments):
			sys_da = da_s if da_s in system_dialog_acts else EmotionalSupportGame.S_Others
			turns.append({
				"sys_da": sys_da, "sys_utt": txt_s,
				# the ESConv dump doesn't label the seeker's reaction; use the neutral DA
				"usr_da": EmotionalSupportGame.U_FeelTheSame, "usr_utt": txt_u,
			})
		if not turns:
			continue
		scenario = (dialog.get("emotion_type", "anxiety"), dialog.get("problem_type", "ongoing stress"))
		out.append({"id": dialog.get("id", f"esc-{i}"), "scenario": scenario, "turns": turns})
	return out


# --- CraigslistBargain (DPDP's cb-valid.txt: JSON-lines of {item_name, buyer_*, seller_*, dialog:[{text,speaker,strategy}]}) ---
_CB_DEAL_STRATEGIES = {"agree", "affirm", "accept"}


def read_cb(path, system_dialog_acts):
	if path.startswith(HF_PREFIX):
		return read_cb_hf(path, system_dialog_acts)
	out = []
	for i, dialog in enumerate(_iter_json_objects(path)):
		segments = _collapse_segments(dialog["dialog"])
		turns = []
		for (sp_s, txt_s, da_s), (sp_u, txt_u, da_u) in _pair_segments(segments):
			sys_da = da_s if da_s in system_dialog_acts else CBGame.S_Inquire
			# CBGame only labels the seller's turn as deal / no-deal
			usr_da = CBGame.U_Deal if (da_u in _CB_DEAL_STRATEGIES) else CBGame.U_No_deal
			turns.append({"sys_da": sys_da, "sys_utt": txt_s, "usr_da": usr_da, "usr_utt": txt_u})
		if not turns:
			continue
		scenario = (
			dialog.get("item_name", ""),
			dialog.get("buyer_item_description", ""),
			dialog.get("buyer_price"),
			dialog.get("seller_item_description", ""),
			dialog.get("seller_price"),
		)
		out.append({"id": dialog.get("id", f"cb-{i}"), "scenario": scenario, "turns": turns})
	return out


# ---------------------------------------------------------------------------
# task registry
# ---------------------------------------------------------------------------
TASKS = {
	"p4g": dotdict({
		"game_cls": PersuasionGame,
		"sys_model": PersuaderModel, "sys_chat": PersuaderChatModel,
		"usr_model": PersuadeeModel, "usr_chat": PersuadeeChatModel,
		"planner": P4GSystemPlanner, "chat_planner": P4GChatSystemPlanner,
		"example": EXP_DIALOG,
		"success_user_da": PersuasionGame.U_Donate,
		"read_dialogs": read_p4g,
		"default_data": "data/p4g/300_dialog_turn_based.pkl",
		# the p4g user model and planner take a `persona`; the planner also takes
		# llm_prior_topk / logit_scoring / explicit_value_labels. See build_agents.
		"supports_persona": True,
		"supports_planner_tuning": True,
		"emotion_aware": False,
		"emotion_classifier_cls": None,
	}),
	"esc": dotdict({
		"game_cls": EmotionalSupportGame,
		"sys_model": TherapistModel, "sys_chat": TherapistChatModel,
		"usr_model": PatientModel, "usr_chat": PatientChatModel,
		"planner": ESCSystemPlanner, "chat_planner": ESCChatSystemPlanner,
		"example": ESConv_EXP_DIALOG,
		"success_user_da": EmotionalSupportGame.U_Solved,
		"read_dialogs": read_esc,
		"default_data": "data/esc/esc-valid.txt",
		"supports_persona": False,
		"supports_planner_tuning": False,
		"emotion_aware": False,
		"emotion_classifier_cls": None,
	}),
	"cb": dotdict({
		"game_cls": CBGame,
		"sys_model": BuyerModel, "sys_chat": BuyerChatModel,
		"usr_model": SellerModel, "usr_chat": SellerChatModel,
		"planner": CBSystemPlanner, "chat_planner": CBChatSystemPlanner,
		"example": CB_EXP_DIALOG,
		"success_user_da": CBGame.U_Deal,
		"read_dialogs": read_cb,
		"default_data": "data/cb/cb-valid.txt",
		"supports_persona": False,
		"supports_planner_tuning": False,
		"emotion_aware": False,
		"emotion_classifier_cls": None,
	}),
}


# ---------------------------------------------------------------------------
# emotion-aware task variants
#
# An emotion-aware task reuses everything from its base task (models, planner, reader,
# few-shot example, success DA, data) but swaps in a game whose ``init_dialog`` returns an
# ``EmotionAwareDialogSession``, and names the ``emotion_classifier_cls`` that
# ``make_emotion_classifier`` builds once per run for ``build_agents`` to hand that game — so
# registering the next dataset (esc / cb) is one ``_emotion_aware_variant`` call once its
# emotion-aware game exists.
# ---------------------------------------------------------------------------
def _emotion_aware_variant(base_task, game_cls, emotion_classifier_cls):
	cfg = dotdict(dict(base_task))
	cfg["game_cls"] = game_cls
	cfg["emotion_aware"] = True
	cfg["emotion_classifier_cls"] = emotion_classifier_cls
	return cfg


TASKS["emo_p4g"] = _emotion_aware_variant(TASKS["p4g"], EmotionAwarePersuasionGame, P4GLLMEmotionClassifier)


# ---------------------------------------------------------------------------
# model / agent construction
# ---------------------------------------------------------------------------
def make_backbone_model(llm, gen_sentences=-1, ollama_model=None, sglang_model=None, ollama_host=None):
	"""Build the LLM backend + a flag for which (chat vs completion) model family to use."""
	if llm == "ollama":
		return OllamaChatModel(ollama_model, base_url=ollama_host, gen_sentences=gen_sentences), "chat"
	if llm == "sglang":
		return SGLangChatModel(sglang_model), "chat"
	if llm in ("code-davinci-002", "text-davinci-002", "text-davinci-003"):
		return OpenAIModel(llm), "completion"
	if llm == "gpt-3.5-turbo":
		return OpenAIChatModel(llm, gen_sentences), "chat"
	if llm == "chatgpt":
		return AzureOpenAIChatModel(llm, gen_sentences), "chat"
	raise ValueError(f"unsupported --llm {llm}")


def build_agents(task_name, backbone_model, family, *, infer_user_da=False,
				 sys_inference_args=None, usr_inference_args=None,
				 llm_prior_topk: int | None = None,
				 logit_scoring: "str | bool" = "off",
				 explicit_value_labels: bool = False,
				 persona: str | None = None,
				 max_conv_turns: int | None = None,
				 success_criterion: str | None = None,
				 emotion_classifier=None):
	"""Construct (game, system, user, planner) for ``task_name``.

	``family`` is "chat" or "completion" (selects the *ChatModel / *ChatSystemPlanner
	vs the plain variants), matching how the backbone model was created.

	``max_conv_turns`` is the game's horizon: ``game.get_dialog_ended`` returns -1.0 once a state
	reaches it. It bounds how deep MCTS simulates ONLY under ``--search_horizon episode``; the default
	``legacy`` rule treats only success as terminal, so search expands past it
	(analysis/phase1/SEARCH_HORIZON_BUG.md).

	``persona`` (p4g only) conditions the two objects that role-play the persuadee -- the user
	simulator and the planner's value estimator -- on the real participant from that dialogue's
	pre-task survey (see ``utils/p4g_personas.py``).
	``None`` gives the unconditioned agents, byte-identical to GDPZero's.

	``emotion_classifier`` is required by the emotion-aware tasks and ignored by the others;
	see make_emotion_classifier.

	``infer_user_da`` says who assigns the user's dialog act. ``False`` (the default, matching
	GDPZero) has the user agent tag its own turn through ``get_utterance_w_da``; ``True`` has it
	write only the utterance and lets the planner's critic infer the act in
	``game.get_next_state``. It also selects how the esc/cb agents are prompted: their
	instruction-style prompts carry no ``[act]`` labels, which is why nothing can self-tag under
	them. MCTS is unaffected either way -- ``planner.predict`` calls ``heuristic`` itself for the
	leaf value, so the critic's value is never a transition output.
	"""
	cfg = TASKS[task_name]
	chat = (family == "chat")
	SysModel = cfg.sys_chat if chat else cfg.sys_model
	UsrModel = cfg.usr_chat if chat else cfg.usr_model
	Planner = cfg.chat_planner if chat else cfg.planner

	ontology = cfg.game_cls.get_game_ontology()
	sys_da = ontology["system"]["dialog_acts"]
	user_da = ontology["user"]["dialog_acts"]
	example = DialogSession(cfg.game_cls.SYS, cfg.game_cls.USR).from_history(cfg.example)

	# top_p/top_k are pinned rather than left to the backend. Unset, every server applies its own
	# default -- Ollama uses top_p=0.9/top_k=40, while SGLang adopts the model's generation_config
	# (vicuna-13b-v1.5 ships top_p=0.6). A 0.6 nucleus collapses the open-loop rollouts: on one
	# mid-dialogue context, 30 persuadee samples at temperature 1.1 came back byte-identical
	# 30/30 at top_p=0.6 vs 30/30 *distinct* at 0.9, which is what drove the repetition loops in
	# the SGLang §W5 row. Pinning them also makes the backend comparison a like-for-like one.
	SAMPLING = {"top_p": 0.9, "top_k": 40}
	if sys_inference_args is None:
		sys_inference_args = {"temperature": 0.7, "do_sample": True, "return_full_text": False, **SAMPLING}  # MCTS open loop
	if usr_inference_args is None:
		usr_inference_args = {
			"max_new_tokens": 128, "temperature": 1.1, "repetition_penalty": 1.0,
			"do_sample": True, "return_full_text": False, **SAMPLING,  # MCTS open loop
		}
	system = SysModel(
		sys_da, backbone_model,
		conv_examples=[example],
		inference_args=sys_inference_args,
		infer_user_da=infer_user_da,
	)
	usr_kwargs = dict(
		inference_args=usr_inference_args,
		backbone_model=backbone_model,
		conv_examples=[example],
		infer_user_da=infer_user_da,
	)
	# `persona` is p4g-only: the esc/cb user models and planners have no such parameter and
	# would raise TypeError. Which players take it is declared on the task (supports_persona /
	# supports_planner_tuning above) rather than sniffed off the constructor signature.
	if persona and cfg.supports_persona:
		usr_kwargs["persona"] = persona
	user = UsrModel(user_da, **usr_kwargs)

	planner_kwargs = dict(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[example],
	)
	if persona and cfg.supports_persona:
		planner_kwargs["persona"] = persona
	if cfg.supports_planner_tuning:
		if llm_prior_topk is not None:
			planner_kwargs["llm_prior_topk"] = llm_prior_topk
		# "off" / False / None all mean "sample the value and the prior, don't score logits"
		if logit_scoring not in (None, False, "off"):
			planner_kwargs["logit_scoring"] = logit_scoring
		if explicit_value_labels:
			planner_kwargs["explicit_value_labels"] = True
	planner = Planner(**planner_kwargs)

	game_kwargs = {} if max_conv_turns is None else {"max_conv_turns": max_conv_turns}
	# --p4g_success (rollout.py). Only passed when stricter than the tag, so every default call
	# constructs the game exactly as before; non-p4g games have no such criterion.
	if success_criterion not in (None, "tag"):
		if not task_name.endswith("p4g"):
			raise ValueError(f"success_criterion {success_criterion!r} is defined for p4g, not {task_name!r}")
		game_kwargs["success_criterion"] = success_criterion
	if not cfg.emotion_aware:
		return cfg.game_cls(system, user, planner, infer_user_da=infer_user_da, **game_kwargs), system, user, planner
	# An emotion-aware game requires a classifier: its get_next_state labels the simulated user
	# reaction with it. The caller owns the instance (make_emotion_classifier) so that one
	# classifier -- and so one `records` list -- is shared by every dialog of a run.
	if emotion_classifier is None:
		raise ValueError(
			f"task {task_name!r} is emotion-aware and needs an emotion classifier; "
			f"build one with make_emotion_classifier() and pass it in."
		)
	game = cfg.game_cls(system, user, planner, infer_user_da, emotion_classifier, **game_kwargs)
	return game, system, user, planner


def make_emotion_classifier(task_name, kind, backbone_model):
	"""The one emotion classifier a run shares, or ``None`` on a task that has no emotions.

	``kind`` is ``--emotion_classifier``: "llm" builds the task's prompt-based classifier on the
	backbone model (few-shot + low temperature + cache), "hf" the encoder-based drop-in
	(j-hartmann/emotion-english-distilroberta-base -- deterministic, no LLM cost). Both expose
	the same interface, so nothing downstream branches on which one is in use.

	Build it once per run and hand the same instance to every ``build_agents`` call: it
	accumulates the run's utterance -> emotion ``records``, which dump_emotion_records writes out.
	"""
	cfg = TASKS[task_name]
	if not cfg.emotion_aware:
		return None
	if kind == "hf":
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		print(f"emotion classifier: HFEmotionClassifier ({HFEmotionClassifier.DEFAULT_MODEL})")
		return HFEmotionClassifier()
	if kind == "llm":
		return cfg.emotion_classifier_cls(backbone_model)
	raise ValueError(f"unsupported --emotion_classifier {kind!r}; choose 'llm' or 'hf'")


# ---------------------------------------------------------------------------
# loading the dataset for a task
# ---------------------------------------------------------------------------
def load_dialogs(task_name, cmd_args):
	"""Read + normalize the task's dataset into ``[{id, scenario, turns}, ...]`` (see the readers).

	``--data`` overrides ``TASKS[task].default_data``. The system dialog acts come straight off
	the game ontology -- the same list the system agent gets -- so the reader can map dataset
	labels onto this game's ontology without a built agent to ask.
	"""
	cfg = TASKS[task_name]
	system_dialog_acts = set(cfg.game_cls.get_game_ontology()["system"]["dialog_acts"])
	data_path = resolve_data_path(cmd_args.data or cfg.default_data)
	dialogs = cfg.read_dialogs(data_path, system_dialog_acts)
	print(f"loaded {len(dialogs)} dialogs from {data_path}")
	return dialogs


# Subtree logging
#   {
#    "dlg_id","turn","node_id","parent_id","action_seq":[...],
#    "utterances":[...],          # all R cached realizations at this node
#    "emotion_dist":{...},        # FULL softmax, mean over those realizations
#    "leaf_value":0.0,
#    "N":0,"Q":0.0,"Q_emo":0.0,"M2_emo":0.0,"sigma_emo":0.0,
#    "root_visit_dist":{...},
#    "cache_hit":true,"seed":0,"depth":0,
#    "per_step_valences":[...]
#    }   # every z backed up on this edge, in visit order

SUBTREE_LOG_DIRNAME = "subtree"


def _json_safe(value):
	"""numpy scalars -> python scalars, Emotions -> str. json.dumps handles the rest."""
	if isinstance(value, (np.floating, np.integer)):
		return value.item()
	return value


def _node_das(node_id: str) -> list:
	"""The DA prefix of a node key, as a list. "" is the empty prefix, not [""]."""
	return node_id.split("__") if node_id else []


def _node_emotion_dist(planner, node_id: str) -> dict:
	"""Mean FULL emotion softmax over the node's cached realizations.

	Never an argmax, and never the raw counts: each cached realization carries the
	classifier's whole distribution over the user reaction at that node, and this
	averages them so the record summarises all R samples rather than one of them.
	"""
	dists = []
	for realization in planner.realizations.get(node_id, []):
		if not realization.history:
			continue
		dist = realization.predicted_distribution()  # None on a session with no emotions
		if dist:
			dists.append(dist)
	if not dists:
		return {}
	agg: dict = {}
	for dist in dists:
		for emotion, p in dist.items():
			agg[str(emotion)] = agg.get(str(emotion), 0.0) + float(p)
	return {e: p / len(dists) for e, p in agg.items()}


def _node_utterances(planner, node_id: str) -> list:
	"""The last system utterance of each cached realization at this node (all R samples)."""
	utterances = []
	for realization in planner.realizations.get(node_id, []):
		try:
			utterances.append(realization.get_turn_utt(turn=-1, role=realization.SYS))
		except (IndexError, AttributeError):
			continue  # the dialogue-start root has no system turn yet
	if utterances:
		return utterances
	# fall back to the distinct utterances the realization-value tracker saw
	return list(planner.realizations_Vs.get(node_id, {}).keys())


def _root_visit_dist(planner, root_id: str, dialog_acts: list) -> dict:
	"""{dialog act: share of root visits}. Identical on every record of a turn; carried
	per record so a single NDJSON line is self-contained."""
	counts = planner.Nsa.get(root_id, {})
	total = float(sum(counts.values()))
	if total <= 0:
		return {}
	out = {}
	for action, n in counts.items():
		idx = int(action)
		name = dialog_acts[idx] if idx < len(dialog_acts) else str(idx)
		out[name] = float(n) / total
	return out


def build_subtree_records(planner, *, dlg_id, turn: int, root_state, seed=None) -> list:
	"""One frozen-schema record per edge of ``planner``'s tree. See the block comment above.

	Reads only write-only bookkeeping the planner already keeps (Nsa/Q/Q_emo/M2_emo/
	emo_valences/realizations/node_V/cache_hits), so calling it has no effect on search.
	"""
	from mcts.emotion_mcts import welford_sigma

	dialog_acts = list(planner.player.dialog_acts)
	open_loop = planner.is_open_loop
	root_id = planner._to_string_rep(root_state)
	root_depth = len(_node_das(root_id))
	root_visit_dist = _root_visit_dist(planner, root_id, dialog_acts)
	seed = _json_safe(seed)

	# Emotion-channel tables. Empty on the GDP-Zero baselines, which never write them --
	# MCTS declares all of these so every planner answers with the same shape (see mcts.py).
	q_emo_all = planner.Q_emo
	m2_all = planner.M2_emo
	valences_all = planner.emo_valences
	cache_all = planner.cache_hits
	node_v = planner.node_V

	def record(node_id, parent_id, n, q, q_emo, m2, valences):
		sigma = welford_sigma(int(n), float(m2))
		return {
			"dlg_id": dlg_id,
			"turn": int(turn),
			"node_id": node_id,
			"parent_id": parent_id,
			"action_seq": _node_das(node_id) if open_loop else [],
			"utterances": _node_utterances(planner, node_id),
			"emotion_dist": _node_emotion_dist(planner, node_id),
			"leaf_value": (float(node_v[node_id]) if node_id in node_v else None),
			"N": int(n),
			"Q": float(q),
			"Q_emo": float(q_emo),
			"M2_emo": float(m2),
			"sigma_emo": float(sigma),
			"root_visit_dist": root_visit_dist,
			"cache_hit": bool(cache_all.get(node_id, [0, 0])[0] > 0),
			"seed": seed,
			"depth": (len(_node_das(node_id)) - root_depth) if open_loop else None,
			"per_step_valences": [float(z) for z in valences],
		}

	# the search root has no incoming edge: N is its total visit count, the edge-valued
	# fields are 0.0 / [] by construction.
	records = [record(root_id, None, planner.Ns.get(root_id, 0), 0.0, 0.0, 0.0, [])]

	for parent_id, actions in planner.Nsa.items():
		parent_q = planner.Q.get(parent_id, {})
		parent_q_emo = q_emo_all.get(parent_id, {})
		parent_m2 = m2_all.get(parent_id, {})
		parent_valences = valences_all.get(parent_id, {})
		for action, n in actions.items():
			idx = int(action)
			da = dialog_acts[idx] if idx < len(dialog_acts) else str(idx)
			node_id = f"{parent_id}__{da}" if parent_id else da
			records.append(record(
				node_id if open_loop else f"{parent_id}#{da}",
				parent_id,
				n,
				parent_q.get(action, 0.0),
				parent_q_emo.get(action, 0.0),
				parent_m2.get(action, 0.0),
				parent_valences.get(action, []),
			))
	return records


def _safe_filename(name) -> str:
	return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name)) or "dialog"


def write_subtree_ndjson(records: list, output_path: str, dlg_id) -> str:
	"""Write one dialogue's subtree records to <run_dir>/subtree/<dlg_id>.ndjson.gz.

	One file per dialogue, written once when the dialogue finishes, so concurrent
	workers never touch the same file. Returns the path (or "" when there is nothing
	to write, e.g. --algo llm_raw, which builds no tree).
	"""
	if not records:
		return ""
	run_dir = os.path.dirname(os.path.abspath(output_path))
	log_dir = os.path.join(run_dir, SUBTREE_LOG_DIRNAME)
	os.makedirs(log_dir, exist_ok=True)
	path = os.path.join(log_dir, f"{_safe_filename(dlg_id)}.ndjson.gz")
	with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
		for rec in records:
			f.write(json.dumps(rec, default=str) + "\n")
	return path


# Simulation-step log (P-VAR instrumentation, analysis/phase1). A second carrier NEXT TO the
# frozen subtree log, which stays byte-for-byte as specified above. One gzipped NDJSON per
# dialogue, <run_dir>/simlog/<dlg_id>.ndjson.gz, two record types:
#
#   record_type "step" -- one per selection+backup inside one turn's search:
#     dlg_id, turn_index, simulation_index, depth (edge depth: root's outgoing edges are 1),
#     action_prefix (acts from the search root to the parent), action,
#     parent_realization_idx / parent_realization_id, parent_nu, parent_emotion_dist,
#     child_realization_id, child_from_cache, child_nu, child_emotion_dist, z, v,
#     parent_Ns, siblings [{action, N, Q, Q_emo, M2_emo, prior, uct}] at the moment of choice,
#     selected_action
#   record_type "turn" -- one per realized dialogue turn (written by the runner):
#     dlg_id, turn_index, planned, root_nu, root_emotion_dist, system_act, system_utterance,
#     user_act, user_utterance, user_emotion_dist, user_nu, outcome, root_visits
SIMLOG_DIRNAME = "simlog"


def build_simlog_step_records(planner, *, dlg_id, turn: int) -> list:
	"""``planner.sim_steps`` with the dialogue coordinates attached. [] for planners that keep
	no tape (the GDP-Zero baselines, which have no emotion channel)."""
	steps = getattr(planner, "sim_steps", None) or []
	return [{"record_type": "step", "dlg_id": dlg_id, "turn_index": int(turn), **step} for step in steps]


def write_simlog_ndjson(records: list, output_path: str, dlg_id) -> str:
	"""Write one dialogue's simlog records to <run_dir>/simlog/<dlg_id>.ndjson.gz ("" if none)."""
	if not records:
		return ""
	run_dir = os.path.dirname(os.path.abspath(output_path))
	log_dir = os.path.join(run_dir, SIMLOG_DIRNAME)
	os.makedirs(log_dir, exist_ok=True)
	path = os.path.join(log_dir, f"{_safe_filename(dlg_id)}.ndjson.gz")
	with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
		for rec in records:
			f.write(json.dumps(rec, default=str) + "\n")
	return path


def subtree_emo_stats(dialog_planner):
	"""Per-edge ``(M2_emo, sigma_emo)`` for the subtree log schema.

	Returns two ``{node: {action: float}}`` dicts shaped exactly like ``planner.Nsa``.

	SCHEMA FREEZE: these two fields are written by EVERY runner, on EVERY edge, in
	EVERY run. A planner without the emotion channel (the GDP-Zero baselines, which
	use plain ``OpenLoopMCTS``) leaves ``M2_emo`` at the empty table MCTS declares, and
	reports 0.0 on every edge rather than omitting the field -- so the analysis code
	reads one schema across all arms and a missing key always means a bug, never a
	baseline.

	``sigma_emo`` is derived here at write time; the planner never stores it.
	"""
	from mcts.emotion_mcts import welford_sigma

	nsa = dialog_planner.Nsa
	stored_m2 = dialog_planner.M2_emo
	m2_out, sigma_out = {}, {}
	for node, actions in nsa.items():
		node_m2 = stored_m2.get(node, {})
		m2_out[node] = {a: float(node_m2.get(a, 0.0)) for a in actions}
		sigma_out[node] = {a: welford_sigma(actions[a], m2_out[node][a]) for a in actions}
	return m2_out, sigma_out


def dump_da_emotion_records(da_emotion_counts: list, output_path: str):
	"""Aggregate per-turn MCTS ``emotions_count`` dicts into a system-DA -> emotion histogram.

	``da_emotion_counts`` is a list (one per dialog turn evaluated) of the
	``EmotionAwareOpenLoopMCTS.emotions_count`` dict at that turn, keyed by the child state's
	DA prefix (``parent_prefix + "__" + da``). The last "__"-segment identifies which system DA
	was just attempted in the rollout; we group user-emotion counts by that DA across the run so
	you can report, per strategy: how often each user emotion followed it.

	Writes ``<output_base>_da_emotions.json`` and prints a one-line-per-DA summary. No-op when
	there is nothing to record.
	"""
	from collections import Counter, defaultdict
	agg: dict = defaultdict(Counter)
	for per_turn in da_emotion_counts:
		for state_hash, emo_counts in (per_turn or {}).items():
			# the part after the final "__" is the DA that produced these emotions
			da = state_hash.rsplit("__", 1)[-1] if state_hash else "<root>"
			for emotion, n in emo_counts.items():
				if n:
					agg[da][str(emotion)] += n
	if not agg:
		print("no DA->emotion records to save")
		return None

	output = {}
	for da, counts in sorted(agg.items()):
		total = sum(counts.values())
		output[da] = {
			"total": total,
			"counts": dict(counts.most_common()),
			"fractions": {e: round(n / total, 4) for e, n in counts.most_common()},
		}

	out_path = os.path.splitext(output_path)[0] + "_da_emotions.json"
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(output, f, indent=2, ensure_ascii=False)

	print("\nDA -> user emotion distribution (from MCTS rollouts):")
	for da, info in output.items():
		top = ", ".join(f"{e}={n}" for e, n in list(info["counts"].items())[:3])
		print(f"  {da:>30}: {info['total']:5d}  (top: {top})")
	print(f"saved DA->emotion records to {out_path}")
	return out_path


def dump_emotion_records(emotion_classifier, output_path):
	"""Print the emotion distribution and save the utterance->emotion records to JSON.

	Records come from the shared classifier instance, so this captures every classification made
	during the run (runner seeding + inside the MCTS). The JSON lands next to ``output_path`` as
	``<output_base>_emotions.json``. No-op when there is no classifier / no records.
	"""
	records = emotion_classifier.records if emotion_classifier is not None else None
	if not records:
		print("no emotion records to save")
		return None
	total = len(records)
	print(f"\nEmotion distribution over {total} classified user utterances:")
	for emotion, n in Counter(r["emotion"] for r in records).most_common():
		print(f"  {emotion:>10}: {n:4d} ({100.0 * n / total:5.1f}%)")

	emotions_path = os.path.splitext(output_path)[0] + "_emotions.json"
	with open(emotions_path, "w", encoding="utf-8") as f:
		json.dump(records, f, ensure_ascii=False, indent=2)
	print(f"saved {total} utterance->emotion records to {emotions_path}")
	return emotions_path


# ---------------------------------------------------------------------------
# shared argparse helpers
# ---------------------------------------------------------------------------
def add_common_args(parser, default_output):
	parser.add_argument("--game", type=str, default="p4g", choices=list(TASKS.keys()),
						help="which dialog game / dataset to evaluate on")
	parser.add_argument("--data", type=str, default=None,
						help="path to the dataset file (default: TASKS[game].default_data)")
	parser.add_argument("--output", type=str, default=default_output, help="output pickle path")
	parser.add_argument("--llm", type=str, default="gpt-3.5-turbo",
						choices=["code-davinci-002", "text-davinci-002", "gpt-3.5-turbo", "chatgpt", "ollama", "sglang"],
						help="backbone model ('ollama' = local Ollama server, see --ollama_model)")
	parser.add_argument("--ollama_model", type=str, default="llama3.1", help="[--llm ollama] model name served by Ollama")
	parser.add_argument("--sglang_model", type=str, default="TheBloke/vicuna-13B-v1.5-AWQ", help="[--llm sglang] model name served by SGLang")
	parser.add_argument("--ollama_host", type=str, default=None, help="[--llm ollama] server URL (default $OLLAMA_HOST or http://localhost:11434)")
	parser.add_argument("--gen_sentences", type=int, default=-1, help="truncate generations to this many sentences (-1 = no limit)")
	parser.add_argument("--debug", action="store_true", help="print each turn's context / prediction")
	parser.add_argument("--llm_prior_topk", type=int, default=None,
						help="if set to an int K, the planner replaces the default 15-sample "
						     "DA-histogram prior with a single LLM call that, given dialog "
						     "history + DAs played so far, returns the top-K most promising next "
						     "DAs. The MCTS then HARD-PRUNES the action space to exactly those K "
						     "actions for that node — Nsa/Q/valid_moves are restricted, the round-"
						     "robin only covers K actions, and dropped actions are unreachable. "
						     "Saves ~14 LLM calls per prior computation AND eliminates round-robin "
						     "waste on actions the LLM said were bad. Emotion conditioning is "
						     "intentionally separate (lives in the emotion-aware Q channel, "
						     "--beta_emo). "
						     "None (default) preserves legacy 13-action behaviour. Reasonable "
						     "values: K=5 or K=7.")
	parser.add_argument("--explicit_value_labels", action="store_true",
						help="restate the five donation labels in the value estimator's final user "
						     "message -- where the answer is read -- instead of only in the leading "
						     "system message ~1000 tokens earlier, and say what the label answers "
						     "(paper §W5). Applies to the SHARED value prompt, so the sampled and the "
						     "logit-scored value both get it. Over three 200-state samples it raises "
						     "agreement between the two estimators from 0.933/0.945/0.939 to "
						     "0.950/0.954/0.957 -- clearing the 0.95 acceptance bar, which the prompt "
						     "as shipped does not -- mostly by making the SAMPLED estimator less "
						     "noisy (test-retest 0.955 -> 0.981). Default off because it changes the "
						     "MCTS leaf value for every p4g run, so every row in W5_COST_TABLE.md "
						     "predates it.")
	parser.add_argument("--logit_scoring", nargs="?", const="both", default="off",
						choices=["off", "value", "prior", "both"],
						help="score the value estimator and the policy prior off the model's "
						     "logits instead of sampling completions and histogramming them "
						     "Requires a backbone that can score continuations (--llm sglang); ignored "
						     "with a warning on the others. Composes with --llm_prior_topk, "
						     "which still prunes MCTS to the K highest-prior actions. A bare "
					     "--logit_scoring means 'both'; 'value' and 'prior' switch one role at "
					     "a time, which is how the §W5 table attributes the saving. 'prior' is "
					     "not cost-only -- it restores the 15-sample histogram's distribution "
					     "(r = 0.97 over 200 states), which --llm_prior_topk's ranking call does "
					     "not approximate (r = -0.28).")
	parser.add_argument("--emotion_classifier", choices=["llm", "hf"], default="llm",
						help="which emotion classifier the emotion-aware games use (ignored on "
						     "the plain tasks). 'llm' = prompt-based, sharing the system backbone "
						     "(few-shot + low temperature + cache). 'hf' = "
						     "j-hartmann/emotion-english-distilroberta-base, a deterministic "
						     "encoder with no LLM cost.")
	parser.add_argument("--seed", type=int, default=None,
						help="seed random / numpy before the run. Default None = unseeded, i.e. "
						     "the previous behaviour. Set it for the pre/post-freeze regression, "
						     "where the realization sampler has to draw the same way in both runs.")
	parser.add_argument("--num_workers", type=int, default=1,
						help="evaluate this many dialogs concurrently (threads). 1 = the old "
						     "sequential behaviour. Higher values keep several requests in flight "
						     "so SGLang can batch them; the GPU is otherwise idle between calls. "
						     "4-8 suits a single local server. Records stay in dialog order.")
	parser.add_argument("--max_turns", type=int, default=10,
						help="the episode horizon Tmax, in turns. Two things at once, and they must "
						     "be the same number: rollout.py stops an episode here, and every runner "
						     "passes it to build_agents(max_conv_turns=...) so game.get_dialog_ended "
						     "returns -1.0 at it. Under --search_horizon episode that is what bounds "
						     "search depth to Tmax - t from a tree rooted at turn t; under legacy "
						     "search ignores it (analysis/phase1/SEARCH_HORIZON_BUG.md). It lives "
						     "here, shared, rather than on each runner: the replay runners used to "
						     "omit it and silently inherit the game default of 15, which made their "
						     "environment differ from the grid's 10.")
	parser.add_argument("--p4g_persona", action="store_true",
						help="condition the p4g user simulator on the REAL persuadee who took "
						     "part in each replayed dialogue, from the Persuasion for Good "
						     "pre-task survey (data/p4g_personas/full_info.csv -- Big Five, "
						     "Moral Foundations, Schwartz values, decision style, demographics). "
						     )
	parser.add_argument("--frozen_config", type=str, default=None,
						help="JSON {\"frozen\": {dest: value}}. Fail before any work if the resolved args "
						     "differ from these values (see check_frozen_config). Default None = no check. "
						     "Every grid config passes the frozen template here.")
	return parser


def replay_root_is_terminal(game, state, search_horizon) -> bool:
	"""True when the replay runners have nothing left to plan at ``state``.

	The replay runners walk a real corpus dialogue and search from each prefix. Under
	``--search_horizon episode`` a prefix at or past the game's horizon is terminal, so
	``search`` returns -1.0 without ever expanding the root: ``Ns`` stays empty, every
	``Nsa`` is 0, and ``get_action_prob`` divides 0/0 -> NaN -> ``argmax`` silently returns
	action 0. Asking "what should the system say next" there is meaningless anyway -- the
	environment has already ended the dialogue.

	This bites because the corpus is longer than the horizon: p4g dialogues run to 15 turns
	while Tmax is 10, so 28 of 2697 replay search roots (1.0 %) sit at or past it.

	Returns False under ``legacy``, where the horizon is not consulted at all, so legacy
	replay output is unchanged.
	"""
	if search_horizon != "episode":
		return False
	return game.get_dialog_ended(state) != 0.0


def apply_seed(cmd_args):
	"""Seed ``random`` and ``numpy`` from ``--seed``. No-op when the flag is unset."""
	if cmd_args.seed is None:
		return
	import random
	random.seed(cmd_args.seed)
	np.random.seed(cmd_args.seed)
	print(f"seeded random/numpy with {cmd_args.seed}")


def load_p4g_personas(cmd_args):
	"""``{dialogue_id: persona text}`` when ``--p4g_persona`` is set, else ``{}``.

	Returning a plain dict means the runners can call ``.get(did)`` unconditionally and hand
	``None`` to ``build_agents(persona=...)`` for a dialogue with no survey response.
	"""
	if not cmd_args.p4g_persona:
		return {}
	if not cmd_args.game.endswith("p4g"):
		raise ValueError(f"--p4g_persona applies to the p4g tasks, not --game {cmd_args.game!r}")
	from utils.p4g_personas import load_persona_texts
	personas = load_persona_texts()
	print(f"--p4g_persona: loaded {len(personas)} persuadee personas from the p4g survey")
	return personas


def check_frozen_config(cmd_args, path):
	"""Refuse to run when the RESOLVED args differ from the frozen grid values in ``path``.

	``path`` is JSON ``{"frozen": {dest: value, ...}}``. Every key must be an argument this runner
	actually resolved (a typo, or a flag the runner does not declare, fails too), and its value --
	after defaults, aliases and inheritance -- must equal the frozen one. The runner defaults are not
	the grid config (n_sims 20, R 3, no top-K prior, llm classifier, unseeded), so a config that leans
	on one would otherwise produce a cell that has to be thrown away.

	On success records the file's sha256 on ``cmd_args`` so metadata.json names the frozen config the
	run was checked against.
	"""
	import hashlib
	raw = open(path, "rb").read()
	frozen = json.loads(raw)["frozen"]
	resolved = vars(cmd_args)
	problems = []
	for key, want in sorted(frozen.items()):
		if key not in resolved:
			problems.append(f"  {key}: not an argument of this runner (frozen {want!r})")
		elif resolved[key] != want or type(resolved[key]) is bool and type(want) is not bool:
			problems.append(f"  {key}: resolved {resolved[key]!r} != frozen {want!r}")
	if problems:
		raise SystemExit(f"--frozen_config {path}: resolved config does not match the frozen values:\n"
						 + "\n".join(problems))
	cmd_args.frozen_config_sha256 = hashlib.sha256(raw).hexdigest()
	print(f"--frozen_config: {len(frozen)} values match {path} (sha256 {cmd_args.frozen_config_sha256[:12]})")


# (flag spellings, applies-only-when) for arm flags that are inert outside their arm. Passing one where
# it is inert is a config error, not a no-op: e.g. --aff_pool_tau on ActPool, whose key has no bucket.
_INERT_WHEN = [
	(("--aff_pool_tau", "--aff-pool-tau"), lambda a: getattr(a, "aff_pool", False) and getattr(a, "aff_pool_key", "affect") == "affect",
	 "--aff_pool_tau only applies to AffPool (--aff_pool --aff_pool_key affect); ActPool has no bucket"),
	(("--aff_pool_bias", "--aff-pool-bias", "--aff_pool_key", "--aff-pool-key"), lambda a: getattr(a, "aff_pool", False),
	 "AffPool/ActPool settings need --aff_pool"),
	(("--coupling_store", "--coupling-store"), lambda a: getattr(a, "coupled_seeds", False),
	 "--coupling_store needs --coupled_seeds"),
	(("--cache_draw", "--cache-draw"), lambda a: getattr(a, "algo", "emomcts") == "emomcts",
	 "--cache_draw reads the parent's nu, which only --algo emomcts has"),
	(("--cache_bucket_tau", "--cache-bucket-tau"), lambda a: getattr(a, "cache_draw", "uniform") in ("bucket", "bucket_kernel"),
	 "--cache_bucket_tau needs --cache_draw bucket or bucket_kernel"),
	(("--cache_kernel_h", "--cache-kernel-h"), lambda a: getattr(a, "cache_draw", "uniform") in ("kernel", "bucket_kernel"),
	 "--cache_kernel_h needs --cache_draw kernel or bucket_kernel"),
]


def check_inert_arm_flags(cmd_args, argv):
	"""Refuse arm flags passed where they have no effect -- a hand-edited config that mixes two arms."""
	given = {tok.split("=", 1)[0] for tok in argv if tok.startswith("--")}
	problems = [f"  {'/'.join(spellings)}: {why}" for spellings, applies, why in _INERT_WHEN
				if given & set(spellings) and not applies(cmd_args)]
	if problems:
		raise SystemExit("arm flags passed where they are inert:\n" + "\n".join(problems))


def finalize_args(cmd_args, argv=None):
	check_inert_arm_flags(cmd_args, sys.argv[1:] if argv is None else argv)
	if getattr(cmd_args, "frozen_config", None):
		check_frozen_config(cmd_args, cmd_args.frozen_config)
	out_dir = os.path.dirname(cmd_args.output)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	return cmd_args


def setup_output_dir(cmd_args, runner_name: str, mcts_class: str, mcts_args=None) -> str:
	"""Re-point ``cmd_args.output`` into a per-run subdirectory and write metadata.json.

	Given ``--output outputs/foo.pkl``, creates ``outputs/foo/`` and mutates
	``cmd_args.output`` to ``outputs/foo/foo.pkl``. All sibling artifacts written via paths
	derived from ``cmd_args.output`` (e.g. ``*_emotions.json``, ``*_da_emotions.json``)
	naturally land in the same directory. Writes ``outputs/foo/metadata.json`` with a
	snapshot of cmd_args, the runner identity, the MCTS class name, the MCTS hyperparams
	dict, and a UTC start timestamp. Written early so crashed runs still leave a trace.
	"""
	from datetime import datetime, timezone
	base, ext = os.path.splitext(cmd_args.output)
	if not ext:
		ext = ".pkl"
	run_id = os.path.basename(base) or "run"
	run_dir = base  # e.g. 'outputs/foo'
	os.makedirs(run_dir, exist_ok=True)
	cmd_args.output = os.path.join(run_dir, run_id + ext)

	args_snapshot = dict(vars(cmd_args))
	metadata = {
		"runner": runner_name,
		"mcts_class": mcts_class,
		"started_at": datetime.now(timezone.utc).isoformat(),
		"args": args_snapshot,
		"mcts_args": dict(mcts_args) if mcts_args else None,
	}
	meta_path = os.path.join(run_dir, "metadata.json")
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, default=str)
	print(f"run dir: {run_dir}")
	print(f"  output:   {cmd_args.output}")
	print(f"  metadata: {meta_path}")
	return run_dir
