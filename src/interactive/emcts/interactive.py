import os
import sys

# allow running this file directly (python src/interactive/interactive.py) by
# putting the project's `src/` directory on the import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import logging
import argparse

from tqdm.auto import tqdm

from utils.gen_models import OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel
from utils.sessions import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG, ESConv_EXP_DIALOG, CB_EXP_DIALOG

from games import PersuasionGame, EmotionalSupportGame, CBGame
from mcts.mcts import OpenLoopMCTS

from players.p4g_players import (
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner,
)
from players.esc_players import (
	TherapistChatModel, PatientChatModel, ESCChatSystemPlanner,
)
from players.cb_players import (
	BuyerChatModel, SellerChatModel, CBChatSystemPlanner,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task wiring. Everything that differs between p4g / esc / cb lives here so
# the play_* functions below are task-agnostic.
# ---------------------------------------------------------------------------
GAME_REGISTRY = {
	"p4g": dotdict({
		"game_cls": PersuasionGame,
		"sys_chat_cls": PersuaderChatModel,
		"usr_chat_cls": PersuadeeChatModel,
		"chat_planner_cls": P4GChatSystemPlanner,
		"example": EXP_DIALOG,
		"greeting_da": PersuasionGame.S_Greeting,
		"greeting_utt": "Hello. How are you?",
		"neutral_da": PersuasionGame.U_Neutral,
		# the human plays the "user" agent of the game
		"user_role_hint": "You are now the Persuadee.",
		"init_dialog": lambda game, args: game.init_dialog(),
	}),
	"esc": dotdict({
		"game_cls": EmotionalSupportGame,
		"sys_chat_cls": TherapistChatModel,
		"usr_chat_cls": PatientChatModel,
		"chat_planner_cls": ESCChatSystemPlanner,
		"example": ESConv_EXP_DIALOG,
		"greeting_da": EmotionalSupportGame.S_Others,
		"greeting_utt": "Hello! How are you feeling today?",
		"neutral_da": EmotionalSupportGame.U_FeelTheSame,
		"user_role_hint": "You are now the Patient.",
		"init_dialog": lambda game, args: game.init_dialog(args.emotion_type, args.problem_type),
	}),
	"cb": dotdict({
		"game_cls": CBGame,
		"sys_chat_cls": BuyerChatModel,
		"usr_chat_cls": SellerChatModel,
		"chat_planner_cls": CBChatSystemPlanner,
		"example": CB_EXP_DIALOG,
		"greeting_da": CBGame.S_Greet,
		"greeting_utt": "Hi! I'm interested in the item you have for sale.",
		"neutral_da": CBGame.U_No_deal,
		"user_role_hint": "You are now the Seller.",
		"init_dialog": lambda game, args: game.init_dialog(
			args.cb_item, args.cb_buyer_desc, args.cb_buyer_price,
			args.cb_seller_desc, args.cb_seller_price,
		),
	}),
}


def build_game(backbone_model, args, *, sys_inference_args=None, usr_inference_args=None):
	"""Construct (game, system, user, planner) for the requested task."""
	cfg = GAME_REGISTRY[args.game]

	game_ontology = cfg.game_cls.get_game_ontology()
	sys_da = game_ontology["system"]["dialog_acts"]
	user_da = game_ontology["user"]["dialog_acts"]
	system_name = cfg.game_cls.SYS
	user_name = cfg.game_cls.USR

	example = DialogSession(system_name, user_name).from_history(cfg.example)

	system = cfg.sys_chat_cls(
		sys_da,
		backbone_model,
		conv_examples=[example],
		inference_args=sys_inference_args or {},
		zero_shot=args.zero_shot,
	)
	user = cfg.usr_chat_cls(
		user_da,
		inference_args=usr_inference_args or {
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model,
		conv_examples=[example],
		zero_shot=args.zero_shot,
	)
	planner = cfg.chat_planner_cls(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[example],
	)
	game = cfg.game_cls(system, user, planner, zero_shot=args.zero_shot)
	return game, system, user, planner


def fresh_state(game, args):
	"""Initial dialog session with the system's opening turn already added."""
	cfg = GAME_REGISTRY[args.game]
	state = cfg.init_dialog(game, args)
	state.add_single(game.SYS, cfg.greeting_da, cfg.greeting_utt)
	return state


def label_user_turn(user, state, your_utt, cfg):
	"""Append the human turn to `state`, labelling it with a predicted dialog act
	(falling back to the task's neutral DA if the user model can't predict one)."""
	tmp_state = state.copy()
	tmp_state.add_single(state.USR, cfg.neutral_da, your_utt)
	try:
		user_da = user.predict_da(tmp_state)
	except NotImplementedError:
		user_da = cfg.neutral_da
	if user_da not in user.dialog_acts:
		user_da = cfg.neutral_da
	logging.info(f"user_da: {user_da}")
	state.add_single(state.USR, user_da, your_utt)
	return user_da


def play_gdpzero(backbone_model, args):
	mcts_args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": args.num_mcts_sims,
		"max_realizations": args.max_realizations,
		"Q_0": args.Q_0,
	})
	cfg = GAME_REGISTRY[args.game]
	game, system, user, planner = build_game(
		backbone_model, args,
		sys_inference_args={
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		},
	)

	state = fresh_state(game, args)
	print(f"{cfg.user_role_hint} Type 'q' to quit, and 'r' to restart.")
	print(f"{game.SYS}: {cfg.greeting_utt}")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = fresh_state(game, args)
			game.display(state)
			your_utt = input("You: ")
			continue

		label_user_turn(user, state, your_utt.strip(), cfg)

		# planning
		if isinstance(backbone_model, OpenAIModel):
			backbone_model._cached_generate.cache_clear()
		dialog_planner = OpenLoopMCTS(game, planner, mcts_args)
		for _ in tqdm(range(mcts_args.num_MCTS_sims)):
			dialog_planner.search(state)

		mcts_policy = dialog_planner.get_action_prob(state)
		best_action = int(np.argmax(mcts_policy))
		mcts_policy_next_da = system.dialog_acts[best_action]
		logger.info(f"mcts_policy: {mcts_policy}")
		logger.info(f"mcts_policy_next_da: {mcts_policy_next_da}")
		logger.info(dialog_planner.Q)

		sys_utt = dialog_planner.get_best_realization(state, best_action)
		logging.info(f"sys_da: [{mcts_policy_next_da}]")
		print(f"{game.SYS}: {sys_utt}")

		state.add_single(game.SYS, mcts_policy_next_da, sys_utt)
		your_utt = input("You: ")
	return


def play_raw_prompt(backbone_model, args):
	cfg = GAME_REGISTRY[args.game]
	game, system, user, planner = build_game(backbone_model, args)

	state = fresh_state(game, args)
	print(f"{cfg.user_role_hint} Type 'q' to quit, and 'r' to restart.")
	print(f"{game.SYS}: {cfg.greeting_utt}")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = fresh_state(game, args)
			game.display(state)
			your_utt = input("You: ")
			continue

		label_user_turn(user, state, your_utt.strip(), cfg)

		# greedy one-step planning with the chat planner
		prior, _v = planner.predict(state)
		best_action = int(np.argmax(prior))
		greedy_da = system.dialog_acts[best_action]
		next_best_state, _ = game.get_next_state(state, best_action)
		# get_next_state appends [system_turn, simulated_user_turn]; we want the system utterance
		greedy_resp = next_best_state.history[-2][2]

		logging.info(f"sys_da: [{greedy_da}]")
		print(f"{game.SYS}: {greedy_resp}")

		state.add_single(game.SYS, greedy_da, greedy_resp)
		your_utt = input("You: ")
	return


def make_backbone_model(args):
	if args.llm == "ollama":
		return OllamaChatModel(args.ollama_model, base_url=args.ollama_host, gen_sentences=args.gen_sentences)
	if args.llm in ["code-davinci-002", "text-davinci-002", "text-davinci-003"]:
		return OpenAIModel(args.llm)
	if args.llm == "gpt-3.5-turbo":
		return OpenAIChatModel(args.llm, args.gen_sentences)
	if args.llm == "chatgpt":
		return AzureOpenAIChatModel(args.llm, args.gen_sentences)
	raise ValueError(f"unsupported --llm {args.llm}")


def main(args):
	backbone_model = make_backbone_model(args)

	if args.algo == "gdpzero":
		print(f"using GDPZero (MCTS) as planning algorithm on '{args.game}'")
		play_gdpzero(backbone_model, args)
	elif args.algo == "raw-prompt":
		print(f"using raw prompting as planning on '{args.game}'")
		play_raw_prompt(backbone_model, args)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", type=int, default=logging.WARNING, help="logging mode",
						choices=[logging.INFO, logging.DEBUG, logging.WARNING])
	parser.add_argument("--game", type=str, default="p4g", choices=["p4g", "esc", "cb"],
						help="which dialog game to play")
	parser.add_argument("--algo", type=str, default="gdpzero", choices=["gdpzero", "raw-prompt"],
						help="planning algorithm")
	parser.add_argument("--zero_shot", type=int, default=1, choices=[0, 1],
						help="1: simulate the user with the user LLM + planner heuristic; 0: use the user model's get_utterance_w_da")
	# backbone LLM
	parser.add_argument("--llm", type=str, default="gpt-3.5-turbo",
						choices=["code-davinci-002", "gpt-3.5-turbo", "text-davinci-002", "chatgpt", "ollama"],
						help="backbone model: an OpenAI/Azure model name, or 'ollama' for a local Ollama server (see --ollama_model)")
	parser.add_argument("--ollama_model", type=str, default="llama3.1",
						help="[--llm ollama] name of the model served by Ollama (e.g. llama3.1, qwen2.5, mistral)")
	parser.add_argument("--ollama_host", type=str, default=None,
						help="[--llm ollama] Ollama server URL (default: $OLLAMA_HOST or http://localhost:11434)")
	parser.add_argument("--gen_sentences", type=int, default=3,
						help="number of sentences to generate from the llm. Longer ones will be truncated by nltk.")
	# MCTS (gdpzero) hyper-params
	parser.add_argument("--num_mcts_sims", type=int, default=10, help="number of mcts simulations")
	parser.add_argument("--max_realizations", type=int, default=3, help="number of realizations per mcts state")
	parser.add_argument("--Q_0", type=float, default=0.25, help="initial Q value for uninitialized states")
	# esc scenario
	parser.add_argument("--emotion_type", type=str, default="anxiety", help="[esc] patient emotion type")
	parser.add_argument("--problem_type", type=str, default="job crisis", help="[esc] patient problem type")
	# cb scenario
	parser.add_argument("--cb_item", type=str, default="a used mountain bike", help="[cb] item being negotiated")
	parser.add_argument("--cb_buyer_desc", type=str, default="A used mountain bike, looks well maintained.", help="[cb] buyer-side item description")
	parser.add_argument("--cb_buyer_price", type=float, default=80.0, help="[cb] price the buyer is targeting")
	parser.add_argument("--cb_seller_desc", type=str, default="Selling my used mountain bike, great condition.", help="[cb] seller-side item description")
	parser.add_argument("--cb_seller_price", type=float, default=150.0, help="[cb] price the seller is asking")
	args = parser.parse_args()

	args.zero_shot = bool(args.zero_shot)
	logging.basicConfig(level=args.log)
	logger.setLevel(args.log)

	main(args)
