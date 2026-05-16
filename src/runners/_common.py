"""Shared plumbing for the offline policy-planning runners (gdpzero / raw prompting / ...).

These scripts replay a held-out dataset of dialogs: for every turn they rebuild the
conversation state, ask the planner for the next system dialog act + utterance, and
record it next to the ground-truth response so it can be scored later.

Everything that differs between the three tasks lives in ``TASKS`` (which game / model /
planner classes to use, the few-shot example dialog, and how to read that task's dataset
into a normalized form). The per-algorithm differences live in the individual runner files,
which call :func:`run_eval` with their own ``plan_turn`` callback.
"""
import os
import sys
import json
import pickle
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

from tqdm.auto import tqdm

from utils.sessions import DialogSession
from utils.utils import dotdict
from utils.gen_models import (
	OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel,
)
from utils.prompt_examples import EXP_DIALOG, ESConv_EXP_DIALOG, CB_EXP_DIALOG

from games import PersuasionGame, EmotionalSupportGame, CBGame
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
	game = cfg.game_cls(system, user, planner, zero_shot=zero_shot)
	return game, system, user, planner


# ---------------------------------------------------------------------------
# the shared eval loop
# ---------------------------------------------------------------------------
def run_eval(task_name, cmd_args, plan_turn, *, sys_inference_args=None, usr_inference_args=None):
	"""Replay up to ``cmd_args.max_conv`` dialogs (or all of them when ``max_conv`` is ``None``)
	and, at every turn, call ``plan_turn`` to get the predicted next (dialog_act, utterance,
	debug_dict). Dumps a pickle of comparison records to ``cmd_args.output`` after every dialog.

	``plan_turn`` is called as ``plan_turn(game, system, planner, backbone_model, state, cmd_args)``
	and must return ``(next_da: str, next_utt: str, debug: dict)``.
	"""
	cfg = TASKS[task_name]
	backbone_model, family = make_backbone_model(
		cmd_args.llm, getattr(cmd_args, "gen_sentences", -1),
		getattr(cmd_args, "ollama_model", "llama3.1"), getattr(cmd_args, "ollama_host", None),
	)
	game, system, user, planner = build_agents(
		task_name, backbone_model, family,
		sys_inference_args=sys_inference_args, usr_inference_args=usr_inference_args,
	)
	print(f"task={task_name}  system DAs={system.dialog_acts}")
	print(f"task={task_name}  user DAs={user.dialog_acts}")

	data_path = resolve_data_path(cmd_args.data or cfg.default_data)
	dialogs = cfg.read_dialogs(data_path, set(system.dialog_acts))
	print(f"loaded {len(dialogs)} dialogs from {data_path}")

	output = []  # [{did, context, ori_da, ori_resp, new_da, new_resp, debug}, ...]
	num_dialogs = len(dialogs) if cmd_args.max_conv is None or cmd_args.max_conv < 0 else cmd_args.max_conv
	num_done = 0
	pbar = tqdm(total=min(num_dialogs, len(dialogs)), desc=f"eval {task_name}")
	for dialog in dialogs:
		if num_done >= num_dialogs:
			break
		did = dialog["id"]
		turns = dialog["turns"]
		if len(turns) < 2:
			continue
		state = game.init_dialog(*dialog["scenario"])
		context = ""
		try:
			for t in range(len(turns) - 1):
				cur, nxt = turns[t], turns[t + 1]
				state.add_single(game.SYS, cur["sys_da"], cur["sys_utt"])
				state.add_single(game.USR, cur["usr_da"], cur["usr_utt"])
				context = f"{context}\n{game.SYS}: {cur['sys_utt']}\n{game.USR}: {cur['usr_utt']}".strip()

				# user already reached the goal -> nothing left to plan
				if cur["usr_da"] == cfg.success_user_da:
					break

				if isinstance(backbone_model, OpenAIModel):
					backbone_model._cached_generate.cache_clear()
				next_da, next_utt, debug = plan_turn(game, system, planner, backbone_model, state, cmd_args)

				output.append({
					"did": did,
					"context": context,
					"ori_da": nxt["sys_da"],
					"ori_resp": nxt["sys_utt"],
					"new_da": next_da,
					"new_resp": next_utt,
					"debug": debug,
				})
				if getattr(cmd_args, "debug", False):
					print(context)
					print(f"  human:  [{nxt['sys_da']}] {nxt['sys_utt']}")
					print(f"  pred:   [{next_da}] {next_utt}")
		except Exception as e:  # keep partial results, move on
			logger.exception(f"dialog {did} failed: {e}")
			if getattr(cmd_args, "raise_errors", False):
				raise
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	pbar.close()
	print(f"done: {num_done} dialogs, {len(output)} turn-records -> {cmd_args.output}")
	return output


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
	parser.add_argument("--max_conv", type=int, default=20,
						help="max number of conversations to evaluate (default: 20; use -1 for all dialogs in the dataset)")
	parser.add_argument("--debug", action="store_true", help="print each turn's context / prediction")
	parser.add_argument("--raise_errors", action="store_true", help="re-raise instead of skipping a failing dialog")
	return parser


def finalize_args(cmd_args):
	out_dir = os.path.dirname(cmd_args.output)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	return cmd_args
