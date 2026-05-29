"""Shared plumbing for the offline policy-planning runners (gdpzero / raw prompting / ...).

These scripts replay a held-out dataset of dialogs: for every turn they rebuild the
conversation state, ask the planner for the next system dialog act + utterance, and
record it next to the ground-truth response so it can be scored later.

Everything that differs between the tasks lives in ``TASKS`` (which game / model / planner
classes to use, the few-shot example dialog, and how to read that task's dataset into a
normalized form). A runner calls :func:`make_backbone_model` + :func:`build_agents` to construct
``(game, system, user, planner)`` for ``--game`` and :func:`load_dialogs` to read the dataset,
then writes its own evaluation loop inline (matching the original GDP-Zero ``runners/`` scripts,
just with the p4g-only agent/data construction replaced by these TASKS-driven helpers).
"""
import os
import sys
import json
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

from utils.sessions import DialogSession, EmotionAwareDialogSession
from utils.utils import dotdict
from utils.gen_models import (
	OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel,
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
			turns.append({"sys_da": sys_da, "sys_utt": sys_utt, "usr_da": usr_da, "usr_utt": usr_utt})
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


def read_p4g(path, system_dialog_acts):
	if path.startswith(HF_PREFIX):
		return read_p4g_hf(path, system_dialog_acts)
	# JSON-lines variant produced by convert_p4g_to_jsonl.py; the pickle is the GDP-Zero original
	if path.endswith(".pkl"):
		return _read_p4g_pickle(path, system_dialog_acts)
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
	}),
}


# ---------------------------------------------------------------------------
# emotion-aware task variants
#
# An emotion-aware task reuses everything from its base task (models, planner, reader,
# few-shot example, success DA, data) but swaps in a game whose ``init_dialog`` returns an
# ``EmotionAwareDialogSession`` and adds an ``emotion_classifier_cls`` that ``build_agents``
# instantiates and attaches to the game. Emotion-aware runners (``runners/emomcts.py``) then
# read ``game.emotion_classifier`` instead of hardcoding one — so registering the next dataset
# (esc / cb) is one ``_emotion_aware_variant`` call once its emotion-aware game exists.
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
def make_backbone_model(llm, gen_sentences=-1, ollama_model="llama3.1", ollama_host=None):
	"""Build the LLM backend + a flag for which (chat vs completion) model family to use."""
	if llm == "ollama":
		return OllamaChatModel(ollama_model, base_url=ollama_host, gen_sentences=gen_sentences), "chat"
	if llm in ("code-davinci-002", "text-davinci-002", "text-davinci-003"):
		return OpenAIModel(llm), "completion"
	if llm == "gpt-3.5-turbo":
		return OpenAIChatModel(llm, gen_sentences), "chat"
	if llm == "chatgpt":
		return AzureOpenAIChatModel(llm, gen_sentences), "chat"
	raise ValueError(f"unsupported --llm {llm}")


def build_agents(task_name, backbone_model, family, *, zero_shot=False,
				 sys_inference_args=None, usr_inference_args=None):
	"""Construct (game, system, user, planner) for ``task_name``.

	``family`` is "chat" or "completion" (selects the *ChatModel / *ChatSystemPlanner
	vs the plain variants), matching how the backbone model was created.

	``zero_shot`` defaults to ``False`` to match GDPZero's behaviour: the user agent emits its
	own DA via ``get_utterance_w_da`` (rather than having ``game.get_next_state`` infer it from
	the planner heuristic). MCTS still works because ``planner.predict`` calls ``heuristic``
	internally for its leaf-value signal — the ``v`` ``get_next_state`` returns under
	``zero_shot=True`` is currently discarded by ``mcts/mcts.py`` either way.
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

	if sys_inference_args is None:
		sys_inference_args = {"temperature": 0.7, "do_sample": True, "return_full_text": False}  # MCTS open loop
	if usr_inference_args is None:
		usr_inference_args = {
			"max_new_tokens": 128, "temperature": 1.1, "repetition_penalty": 1.0,
			"do_sample": True, "return_full_text": False,  # MCTS open loop
		}
	system = SysModel(
		sys_da, backbone_model,
		conv_examples=[example],
		inference_args=sys_inference_args,
		zero_shot=zero_shot,
	)
	user = UsrModel(
		user_da,
		inference_args=usr_inference_args,
		backbone_model=backbone_model,
		conv_examples=[example],
		zero_shot=zero_shot,
	)
	planner = Planner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[example],
	)
	# emotion-aware tasks build the classifier on the backbone model and pass it to the game,
	# whose __init__ requires it (and whose get_next_state classifies the user's emotion).
	if cfg.get("emotion_aware") and cfg.get("emotion_classifier_cls"):
		emotion_classifier = cfg.emotion_classifier_cls(backbone_model)
		game = cfg.game_cls(system, user, planner, zero_shot, emotion_classifier)
	else:
		game = cfg.game_cls(system, user, planner, zero_shot=zero_shot)
	return game, system, user, planner


# ---------------------------------------------------------------------------
# loading the dataset for a task
# ---------------------------------------------------------------------------
def load_dialogs(task_name, cmd_args, system):
	"""Read + normalize the task's dataset into ``[{id, scenario, turns}, ...]`` (see the readers).

	``--data`` overrides ``TASKS[task].default_data``; the system's dialog acts are passed so the
	reader can map dataset labels onto this game's ontology.
	"""
	cfg = TASKS[task_name]
	data_path = resolve_data_path(cmd_args.data or cfg.default_data)
	dialogs = cfg.read_dialogs(data_path, set(system.dialog_acts))
	print(f"loaded {len(dialogs)} dialogs from {data_path}")
	return dialogs


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
	records = getattr(emotion_classifier, "records", None)
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
						choices=["code-davinci-002", "text-davinci-002", "gpt-3.5-turbo", "chatgpt", "ollama"],
						help="backbone model ('ollama' = local Ollama server, see --ollama_model)")
	parser.add_argument("--ollama_model", type=str, default="llama3.1", help="[--llm ollama] model name served by Ollama")
	parser.add_argument("--ollama_host", type=str, default=None, help="[--llm ollama] server URL (default $OLLAMA_HOST or http://localhost:11434)")
	parser.add_argument("--gen_sentences", type=int, default=-1, help="truncate generations to this many sentences (-1 = no limit)")
	parser.add_argument("--debug", action="store_true", help="print each turn's context / prediction")
	return parser


def finalize_args(cmd_args):
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

	args_snapshot = vars(cmd_args).copy() if hasattr(cmd_args, "__dict__") else dict(cmd_args)
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
