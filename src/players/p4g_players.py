import logging
import re

import numpy as np

from typing import List, Tuple

from players.planner import DialogPlanner
from utils.sessions import DialogSession
from utils.gen_models import GenerationModel, DialogModel
from games import PersuasionGame
from collections import Counter
from utils.rewards import reward_dict
from utils.role_profiler import (
	role, POLICY_PRIOR, VALUE_ESTIMATOR, USER_SIMULATOR, SYSTEM_UTTERANCE
)
from utils.p4g_personas import persona_suffix
from players.prompting import parse_das, recent_turns, split_da


logger = logging.getLogger(__name__)


def _mean_reward(sampled_das) -> float:
	"""Mean reward over the persuadee acts that carry one; 0.0 if none of them do."""
	scores = [reward_dict['p4g'][da] for da in sampled_das if da in reward_dict['p4g']]
	return float(np.mean(scores)) if scores else 0.0


class P4GSystemPlanner(DialogPlanner):
	def __init__(
			self,
			dialog_acts,
			max_hist_num_turns,
			user_dialog_acts,
			user_max_hist_num_turns,
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession]=None,
			persona: str = None,
	):
		super().__init__()
		self.persona = persona or None
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
		self.user_dialog_acts = user_dialog_acts
		self.user_max_hist_num_turns = user_max_hist_num_turns  # used in heuristic function
		self.conv_examples = conv_examples
		self.generation_model = generation_model
		self.smoothing = 1.0
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The Persuader can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()

		self.inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return

	def process_exp(self, keep_sys_da=True, keep_user_da=False):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da) + "\n"
		return prompt_exps.strip()

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def predict(self, state:DialogSession, policy=None, ent_bound=None) -> "Tuple[np.ndarray, float]":
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{state.to_string_rep(keep_sys_da=True)}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		logger.debug(prompt)
		data = self.generation_model.generate(prompt, **self.inf_args)
		sampled_das = parse_das(data, self.dialog_acts)
		logger.debug(f"sampled das: {sampled_das}")
		return self._histogram(sampled_das)

	def _histogram(self, sampled_das) -> "np.ndarray":
		"""Sampled acts as a distribution, with `self.smoothing` added so an act that happened
		not to be drawn is not assigned probability zero."""
		prob = np.zeros(len(self.dialog_acts)) + self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		return prob / prob.sum()

	def heuristic(self, state:DialogSession) -> float:
		"""Ask the persuadee simulator the donation question outright; score its answer."""
		assert(state[-1][0] == PersuasionGame.USR)
		prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is a conversation between a Persuader and	a Persuadee about a charity called Save the Children. 
		{self.process_exp(keep_sys_da=False, keep_user_da=True)}
		The following is a new conversation between another Persuader and Persuadee.{persona_suffix(self.persona, prefix=" The Persuadee: ")}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
		Persuader: Would you be interested in donating to Save the Children?
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()

		inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.generate(prompt, **inf_args)
		sampled_das = parse_das(data, self.user_dialog_acts)

		logger.debug(f"persuadee prompt: {prompt}")
		logger.debug(f"sampled das: {sampled_das}")
		return _mean_reward(sampled_das), sampled_das


class P4GChatSystemPlanner(P4GSystemPlanner):
	def __init__(
		self,
		dialog_acts,
		max_hist_num_turns,
		user_dialog_acts,
		user_max_hist_num_turns,
		generation_model:GenerationModel,
		conv_examples: List[DialogSession] = [],
		llm_prior_topk: int | None = None,
		logit_scoring: "str | bool" = False,
		explicit_value_labels: bool = False,
		persona: str = None,
	) -> None:
		super().__init__(
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns,
			generation_model, conv_examples, persona=persona,
		)
		self.task_prompt = f"""
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children. You can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and a Persuadee."
		self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

		self.inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		# Ask for the K most promising acts in one call instead of histogramming 15 samples.
		self.llm_prior_topk = llm_prior_topk
		# Read the value and/or the prior off the logits instead of sampling them. Both are
		# closed-label questions, so the logits give exactly what the samples estimate.
		# Needs a backbone that can score a continuation (SGLang); off everywhere else.
		mode = {False: "off", True: "both", None: "off"}.get(logit_scoring, logit_scoring)
		if mode not in ("off", "value", "prior", "both"):
			raise ValueError(f"logit_scoring must be off/value/prior/both, got {mode!r}")
		if mode != "off" and not generation_model.supports_label_scoring():
			logger.warning(
				f"--logit_scoring {mode} requested but {type(generation_model).__name__} cannot "
				f"score labels; falling back to the sampling paths."
			)
			mode = "off"
		self.logit_scoring = mode
		self.logit_value = mode in ("value", "both")
		self.logit_prior = mode in ("prior", "both")
		# Restate the donation labels where the answer is read rather than only in the leading
		# system message (paper §W5). Off by default: it moves the leaf value of every p4g run,
		# so turning it on breaks comparison with the rows already in W5_COST_TABLE.md.
		self.explicit_value_labels = bool(explicit_value_labels)
		# One considered answer, long enough for a numbered list -- not samples.
		self.topk_inf_args = {
			"max_new_tokens": 256,
			"temperature": 0.3,
			"return_full_text": False,
			"do_sample": False,
			"num_return_sequences": 1,
		}
		return
	
	def process_chat_exp(self, 
			new_task_prompt,
			assistant_role=PersuasionGame.SYS,
			keep_sys_da=True, keep_user_da=False):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self._process_chat_turns(exp, keep_sys_da, keep_user_da, assistant_role)
			prompt_exps.append({
				"role":"system", "content": new_task_prompt
			})
		return prompt_exps[:-1]

	def _process_chat_turns(self,
			exp:DialogSession, 
			keep_sys_da, keep_user_da,
			assistant_role=PersuasionGame.SYS,
			max_hist_num_turns: int = -1):
		"""``exp`` as chat messages, with whichever speaker ``assistant_role`` names cast as
		the assistant. Guarded on the raw history, not ``len(exp)``: that counts turns, so a
		state ending mid-turn -- exactly what the user simulator is asked about -- read as
		empty and dropped the whole conversation from the prompt."""
		if len(exp.history) == 0:
			return []
		assert(exp[0][0] == PersuasionGame.SYS)  # P4G dialogues open with the Persuader

		prompt_messages = []
		for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
			keep_da = keep_sys_da if role == PersuasionGame.SYS else keep_user_da
			content = f"{role}: [{da}] {utt}" if keep_da else f"{role}: {utt}"
			prompt_messages.append({
				"role": "assistant" if role == assistant_role else "user",
				"content": content.strip(),
			})
		return prompt_messages

	def _build_prior_messages(self, state:DialogSession) -> list:
		"""The prompt the policy prior answers: task + few-shot demo + the live dialogue, with
		the model left to continue as `Persuader: [<dialog act>] ...`. Shared by the sampling
		path (which histograms 15 completions) and the logit path (which scores the 13 acts)."""
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self._process_chat_turns(
				state, keep_sys_da=True, keep_user_da=False,
				max_hist_num_turns=self.max_hist_num_turns,
			)
		return messages

	def predict(self, state:DialogSession, policy=None, ent_bound=None) -> "Tuple[np.ndarray, float]":
		"""(prior over the persuader's acts, leaf value). The logit prior replaces the top-K
		call rather than preceding it: it already returns the full distribution, which MCTS
		then prunes to the K best."""
		if self.logit_prior:
			return self._predict_logit_prior(state)
		if self.llm_prior_topk is not None and self.llm_prior_topk > 0:
			return self._predict_topk_prior(state, self.llm_prior_topk)
		prob = self.sample_prior_probs(state)
		v, _ = self.heuristic(state)
		return prob, v

	def sample_prior_probs(self, state:DialogSession) -> "np.ndarray":
		"""The generation-based prior: 15 sampled persuader turns, histogrammed over the acts.
		Public so the logit path can be validated against it."""
		messages = self._build_prior_messages(state)
		with role(POLICY_PRIOR):
			data = self.generation_model.chat_generate(messages, **self.inf_args)

		sampled_das = parse_das(data, self.dialog_acts)
		logger.debug(f"sampled das: {sampled_das}")
		return self._histogram(sampled_das)

	# ---------------------------------------------------------------------
	# Single-call top-K prior. Emotion is deliberately absent: that signal is the PUCT
	# bonus's job, and keeping the two apart is what makes them separately ablatable.
	# ---------------------------------------------------------------------

	def _format_history_for_topk(self, state: DialogSession) -> str:
		"""The dialogue as ``Speaker [act]: text`` lines, windowed like every other role."""
		lines = []
		for _i, (role, da, utt) in recent_turns(state, self.max_hist_num_turns):
			default_da = 'other' if role == PersuasionGame.SYS else 'neutral'
			lines.append(f"{role} [{da or default_da}]: {utt}")
		return "\n".join(lines).strip()

	def _build_topk_messages(self, state: DialogSession, k: int) -> list:
		history_block = self._format_history_for_topk(state) or "(no conversation yet — this is the opening turn)"
		action_list = ", ".join(f"[{da}]" for da in self.dialog_acts)
		instruction = (
			f"You are advising the Persuader. Given the conversation so far and the "
			f"dialogue actions played at each turn, list the TOP {k} most promising "
			f"next dialogue actions for the Persuader — ordered from most to least "
			f"promising for landing a donation.\n\n"
			f"Available actions (use ONLY these names verbatim, in brackets):\n{action_list}\n\n"
			f"Conversation so far:\n{history_block}\n\n"
			f"Output a numbered list of exactly {k} actions, one per line, in the form:\n"
			f"1. [action]\n2. [action]\n... (up to {k})"
		)
		# The instruction must be a *user* message: as a trailing system message vicuna's
		# template renders it as plain text and the model carries on role-playing instead of
		# answering, which collapsed the prior onto one action.
		return [
			{"role": "system", "content": self.task_prompt},
			*self.prompt_examples,
			{"role": "user", "content": instruction},
		]

	def _parse_topk_response(self, response_text: str, k: int) -> list:
		"""Up to K acts from the model's numbered list, in the order it ranked them.

		Three passes, each more forgiving than the last: bracketed names, then bare
		numbered-list lines, then any mention of an act anywhere in the text.
		"""
		text = response_text or ""
		picked = []

		def take(candidate) -> bool:
			for da in self.dialog_acts:
				if da.lower() == candidate and da not in picked:
					picked.append(da)
					return True
			return False

		for m in re.finditer(r"\[([^\]]+)\]", text):
			take(m.group(1).strip().lower())
			if len(picked) >= k:
				return picked[:k]
		for line in text.splitlines():
			take(re.sub(r"^\s*\d+[\.\):]\s*", "", line).strip().lower())
			if len(picked) >= k:
				return picked[:k]
		# longest names first, so "proposition of donation" wins over a shorter substring
		for da in sorted(self.dialog_acts, key=len, reverse=True):
			if da.lower() in text.lower() and da not in picked:
				picked.append(da)
				if len(picked) >= k:
					break
		return picked[:k]

	def _predict_topk_prior(self, state: DialogSession, k: int) -> "Tuple[np.ndarray, float]":
		prob = self.topk_prior_probs(state, k)
		v, _ = self.heuristic(state)
		return prob, v

	def topk_prior_probs(self, state: DialogSession, k: int) -> "np.ndarray":
		"""The ranking prior on its own, without the leaf value.

		Ranked acts share the mass by harmonic decay (1/1, 1/2, ... 1/K); every act also keeps
		a 0.5% floor, small enough that the ranking dominates but non-zero so PUCT can still
		recover from a bad pick.
		"""
		messages = self._build_topk_messages(state, k)
		with role(POLICY_PRIOR):
			data = self.generation_model.chat_generate(messages, **self.topk_inf_args)
		if isinstance(data, list) and data:
			response_text = data[0].get("generated_text") or data[0].get("content") or str(data[0])
		else:
			response_text = str(data)
		picked = self._parse_topk_response(response_text, k)
		logger.debug(f"topk-prior picked: {picked} (parsed from: {response_text[:200]!r})")

		if not picked:
			logger.warning("topk-prior parsing returned no actions; using uniform fallback")
			return np.ones(len(self.dialog_acts)) / len(self.dialog_acts)

		floor = 0.005
		prob = np.zeros(len(self.dialog_acts)) + floor
		weights = np.array([1.0 / (i + 1) for i in range(len(picked))])
		weights /= weights.sum()
		mass = 1.0 - floor * len(self.dialog_acts)
		for da, w in zip(picked, weights):
			prob[self.dialog_acts.index(da)] += mass * w
		return prob / prob.sum()

	def _build_value_messages(self, state:DialogSession) -> list:
		"""The prompt the value estimator answers: the dialogue replayed from the *persuadee's*
		side, closed with a direct ask, so the model's next turn is `Persuadee: [<label>] ...`
		and the label is its answer to "would you donate". Shared by the sampling path (10
		completions, averaged) and the logit path (the 5 labels scored)."""
		assert(state[-1][0] == PersuasionGame.USR)
		user_task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee.
		""".replace("\t", "").strip()
		user_new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you)."
		# The value estimator role-plays the persuadee, so it takes the same persona as the
		# rollout simulator. It goes on the live-dialogue marker only: the few-shot demo is a
		# different persuadee.
		live_new_task_prompt = user_new_task_prompt + persona_suffix(self.persona)

		messages = [
			{'role': 'system', 'content': user_task_prompt},
			*self.process_chat_exp(new_task_prompt=user_new_task_prompt, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True),
			{'role': 'system', 'content': live_new_task_prompt}
		]
		messages += self._process_chat_turns(
			state, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True,
			max_hist_num_turns=self.user_max_hist_num_turns,
		)
		messages.append({
			'role': 'user', 'content': f'{PersuasionGame.SYS}: Would you be interested in donating to Save the Children?'
		})
		if self.explicit_value_labels:
			# The listing alone measured no better than the prompt as it shipped; naming what
			# the label answers is what carried it. See scripts/explicit_labels_ablation.py.
			listing = " ".join(f"[{da}]" for da in self.user_dialog_acts)
			messages[-1] = {**messages[-1], 'content': messages[-1]['content'] + (
				f"\nAnswer with your reaction to that question, beginning with exactly one of "
				f"these labels: {listing}. Use [{PersuasionGame.U_Donate}] only if you are "
				f"agreeing to donate, and [{PersuasionGame.U_NoDonation}] only if you are refusing."
			)}
		return messages

	def heuristic(self, state:DialogSession) -> float:
		"""Ask the persuadee simulator the donation question outright; score its answer."""
		if self.logit_value:
			return self._heuristic_logits(state)

		messages = self._build_value_messages(state)
		inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		with role(VALUE_ESTIMATOR):
			data = self.generation_model.chat_generate(messages, **inf_args)
		sampled_das = parse_das(data, self.user_dialog_acts)

		logger.debug(f"persuadee prompt: {messages}")
		logger.debug(f"sampled das: {sampled_das}")
		return _mean_reward(sampled_das), sampled_das

	# ---------------------------------------------------------------------
	# Logit-scored value and prior (paper §W5) -- see self.logit_scoring
	# ---------------------------------------------------------------------
	def score_value_labels(self, state:DialogSession, top_logprobs_num: int = 0):
		"""P(donation label | dialogue) over the 5 persuadee acts, one prompt, no generation.
		Returned whole so callers can read P(donate) or the full distribution, not just v."""
		with role(VALUE_ESTIMATOR):
			return self.generation_model.score_labels(
				self._build_value_messages(state),
				labels=list(self.user_dialog_acts),
				prefill=f"{PersuasionGame.USR}: [",
				close="]",
				top_logprobs_num=top_logprobs_num,
			)

	@staticmethod
	def _as_pseudo_samples(scores, n: int = 10) -> list:
		"""A label distribution as the sample list the sampling path returns.

		`PersuasionGame.map_user_action` reads the modal non-donate act out of the samples, so
		the logit path has to hand it the same shape. Largest-remainder apportionment of n
		slots reproduces the histogram the draws were estimating, without the sampling noise.
		"""
		exact = scores.probs * n
		counts = np.floor(exact).astype(int)
		for idx in np.argsort(-(exact - counts))[: n - counts.sum()]:
			counts[idx] += 1
		return [label for label, c in zip(scores.labels, counts) for _ in range(c)]

	def _heuristic_logits(self, state:DialogSession) -> "Tuple[float, list]":
		assert(state[-1][0] == PersuasionGame.USR)
		scores = self.score_value_labels(state)
		# The sampling path's reward table and drop rule, as an exact expectation.
		v = scores.expectation(reward_dict['p4g'])
		logger.debug(f"logit-scored value: v={v:.4f} P(donate)={scores.prob_of(PersuasionGame.U_Donate):.4f} "
					 f"probs={dict(zip(scores.labels, np.round(scores.probs, 4)))}")
		return float(v), self._as_pseudo_samples(scores)

	def score_prior_labels(self, state:DialogSession, top_logprobs_num: int = 0):
		"""P(dialog act | dialogue) over the persuader's act set, one prompt, no generation.

		No smoothing floor: a softmax over logprobs is already positive everywhere, so unlike
		the sampled histogram it cannot shut PUCT out of an action."""
		with role(POLICY_PRIOR):
			return self.generation_model.score_labels(
				self._build_prior_messages(state),
				labels=list(self.dialog_acts),
				prefill=f"{PersuasionGame.SYS}: [",
				close="]",
				top_logprobs_num=top_logprobs_num,
			)

	def _predict_logit_prior(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		scores = self.score_prior_labels(state)
		logger.debug(f"logit-scored prior: {dict(zip(scores.labels, np.round(scores.probs, 4)))}")
		v, _ = self.heuristic(state)
		return scores.probs, v


class PersuaderModel(DialogModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {},
			infer_user_da: bool = True):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.max_hist_num_turns = max_hist_num_turns
		# accepted for parity with the esc/cb players, which use it to drop the [act]
		# tags from their prompts. The p4g prompts always show them, so nothing reads it.
		self.infer_user_da = infer_user_da
		self.da_prompts_mapping = {
			PersuasionGame.S_Greeting:	 				"The Persuader greets the Persuadee.",
			PersuasionGame.S_CredibilityAppeal:	 		"The Persuader establishes credibility of Save the Children by citing its impact.",
			PersuasionGame.S_EmotionAppeal:	 			"The Persuader uses an emotion appeal to convince the Persuadee.",
			PersuasionGame.S_LogicalAppeal:	 			"The Persuader use of reasoning and evidence to convince the Persuadee.",
			PersuasionGame.S_TaskRelatedInquiry:	 	"The Persuader asks about the Persuadee's knowledge or opinion related to Save the Children.",
			PersuasionGame.S_PropositionOfDonation:	 	"The Persuader asks if the Persuadee would like to make a small donation.",
			PersuasionGame.S_Other:	 					"The Persuader responds to the Persuadee without using any persuaive strategy.",
		}
		self.dialog_acts = [da for da in dialog_acts if da in self.da_prompts_mapping]
		
		logger.debug(self.dialog_acts)
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise the tree never reaches the next level
			"return_full_text": False,
			**inference_args
		}
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += self._process_turns(exp) + "\n"
		return prompt_exps.strip()

	def _process_turns(self, exp:DialogSession, max_hist_num_turns: int = -1):
		prompt_exp = ""
		for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
			if role == PersuasionGame.SYS:
				prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
			else:
				prompt_exp += f"{role}: {utt}\n"
		return prompt_exp.strip()
	
	def get_utterance(self, state:DialogSession, action:int, mode='train') -> str:
		"""Realize the act the planner chose as an utterance."""
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			{da_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{self._process_turns(state, max_hist_num_turns=self.max_hist_num_turns)}
			{da_prompt}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		with role(SYSTEM_UTTERANCE):
			data = self.backbone_model.generate(prompt, **self.inference_args)
		return self.backbone_model._cleaned_resp(data, prompt)[0]

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError
	

class PersuaderChatModel(PersuaderModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {},
			infer_user_da: bool = True):
		super().__init__(
			dialog_acts=dialog_acts,
			backbone_model=backbone_model,
			max_hist_num_turns=max_hist_num_turns,
			conv_examples=conv_examples,
			inference_args=inference_args,
			infer_user_da=infer_user_da
		)
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise the tree never reaches the next level, open-loop search aside
			"return_full_text": False,
			**inference_args
		}
		self.task_prompt = """
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children.
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and another Persuadee.\nThe Persuader greets the persuadee."
		self.prompt_examples = self.process_chat_exp()
		return

	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self._process_chat_turns(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def _process_chat_turns(self, exp:DialogSession, da_prompt: str = '', max_hist_num_turns: int = -1):
		"""``exp`` as chat messages, each user turn followed by the instruction for the
		persuader turn that answers it. The final user turn is the one being answered now, so
		it takes ``da_prompt`` -- the act the planner just chose -- rather than an act read off
		the history; without that the persuader saw the same prompt for every action.

		Guarded on the raw history, not ``len(exp)``: that counts turns, so a state ending
		mid-turn read as empty and dropped the conversation from the prompt.
		"""
		if len(exp.history) == 0:
			return []
		assert(exp[0][0] == PersuasionGame.SYS)  # P4G dialogues open with the Persuader

		prompt_messages = []
		for i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
			if role == PersuasionGame.SYS:
				prompt_messages.append({"role": "assistant", "content": f"{role}: {utt}".strip()})
				continue
			is_last = i + 1 >= len(exp.history)
			instruction = da_prompt if is_last else self.da_prompts_mapping[exp[i + 1][1]]
			prompt_messages.append({
				"role": "user",
				"content": f"{role}: {utt}\n{instruction}".strip(),
			})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action:int, mode='train') -> str:
		return self.get_utterance_batched(state, action, batch=1)[0]
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3) -> List[str]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.\n{da_prompt}'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self._process_chat_turns(state, da_prompt, max_hist_num_turns=self.max_hist_num_turns)
		gen_args = {**self.inference_args, "num_return_sequences": batch}
		with role(SYSTEM_UTTERANCE):
			data = self.backbone_model.chat_generate(messages, **gen_args)
		sys_resps = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.SYS}:", user_role=f"{PersuasionGame.USR}:"
		)
		return sys_resps

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError


class PersuadeeModel(DialogModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel,
			conv_examples: List[DialogSession] = [],
			max_hist_num_turns=5,
			infer_user_da: bool = True,
			persona: str = None,
	):
		super().__init__()
		# Who this persuadee is, from the p4g pre-task survey (utils/p4g_personas.py);
		# None gives the unconditioned simulator.
		self.persona = persona or None
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		# accepted for parity with the esc/cb players, which use it to drop the [act]
		# tags from their prompts. The p4g prompts always show them, so nothing reads it.
		self.infer_user_da = infer_user_da
		self.task_prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = inference_args
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
		return prompt_exps.strip()
	
	def get_utterance(self, state:DialogSession, action=None, mode='train') -> str:
		assert(state[-1][0] == PersuasionGame.SYS)
		persona_line = persona_suffix(self.persona, prefix="\nThe Persuadee: ")
		prompt = f"""
		{self.task_prompt}{persona_line}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()
		with role(USER_SIMULATOR):
			data = self.backbone_model.generate(prompt, **self.inference_args)
		return self.backbone_model._cleaned_resp(data, prompt)[0]

	def get_utterance_w_da(self, state:DialogSession, action=None, mode='train') -> "Tuple[str, str]":
		user_resp = self.get_utterance(state, action, mode=mode)
		return split_da(user_resp, self.dialog_acts, PersuasionGame.U_Neutral)


class PersuadeeChatModel(PersuadeeModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel,
			conv_examples: List[DialogSession] = [],
			max_hist_num_turns=5,
			infer_user_da: bool = True,
			persona: str = None):
		super().__init__(
			dialog_acts=dialog_acts,
			inference_args=inference_args,
			backbone_model=backbone_model,
			conv_examples=conv_examples,
			max_hist_num_turns=max_hist_num_turns,
			infer_user_da=infer_user_da,
			persona=persona,
		)
		self.inference_args = inference_args
		self.task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and some Persuadee.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you). You may or may not want to donate to Save the Children."
		self.heuristic_args: dict = {
			"max_hist_num_turns": 2,
			"example_pred_turn": [[0, 2, 3, 4]]
		}
		self.prompt_examples = self.process_chat_exp()
		return

	def live_task_prompt(self) -> str:
		"""``new_task_prompt`` plus this persuadee's persona, if it has one.

		This is the message that hands over to the live dialogue, so the persona describes the
		persuadee being replayed. ``prompt_examples`` stays unconditioned: the demo is someone
		else.
		"""
		return self.new_task_prompt + persona_suffix(self.persona)

	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self._process_chat_turns(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def _process_chat_turns(self, exp:DialogSession, max_hist_num_turns: int = -1):
		"""``exp`` as chat messages with the persuadee -- the simulator itself -- as assistant.

		Guarded on the raw history, not ``len(exp)``: that counts turns, so a state ending
		mid-turn, which is exactly what the simulator is asked about, read as empty and
		dropped the conversation from the prompt.
		"""
		if len(exp.history) == 0:
			return []
		assert(exp[0][0] == PersuasionGame.SYS)  # P4G dialogues open with the Persuader

		prompt_messages = []
		for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
			if role == PersuasionGame.SYS:
				prompt_messages.append({"role": "user", "content": f"{role}: {utt}".strip()})
			else:
				prompt_messages.append({
					"role": "assistant",
					"content": f"{role}: [{da}] {utt}".strip(),
				})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action=None, mode='train') -> str:
		assert(state[-1][0] == PersuasionGame.SYS)  # next turn is user's turn
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.live_task_prompt()}
		]
		messages += self._process_chat_turns(state, max_hist_num_turns=self.max_hist_num_turns)
		with role(USER_SIMULATOR):
			data = self.backbone_model.chat_generate(messages, **self.inference_args)
		user_resp = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
		)[0]
		return user_resp
	
	def get_utterance_from_batched_states(self, states:List[DialogSession], action=None) -> List[str]:
		assert(all([state[-1][0] == PersuasionGame.SYS for state in states]))
		all_prompts = []
		for state in states:
			messages = [
				{'role': 'system', 'content': self.task_prompt},
				*self.prompt_examples,
				{'role': 'system', 'content': self.live_task_prompt()}
			]
			messages += self._process_chat_turns(state, max_hist_num_turns=self.max_hist_num_turns)
			all_prompts.append(messages)
		with role(USER_SIMULATOR):
			datas = self.backbone_model.chat_generate_batched(all_prompts, **self.inference_args)
		user_resps = []
		for data in datas:
			user_resp = self.backbone_model._cleaned_chat_resp(
				data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
			)
			user_resps.append(user_resp[0])
		return user_resps
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		split = [
			split_da(resp, self.dialog_acts, PersuasionGame.U_Neutral)
			for resp in self.get_utterance_from_batched_states(states, action)
		]
		das = [da for da, _ in split]
		user_resps = [utt for _, utt in split]
		return das, user_resps

	def _heuristics_qa_pair(self, dialog:DialogSession):
		"""One (dialogue, act) demonstration for predict_da: the turns as a single question,
		answered by the act the closing user turn actually carried."""
		if len(dialog) == 0:
			return []
		assert(dialog[0][0] == PersuasionGame.SYS)
		assert(dialog[-1][0] == PersuasionGame.USR)

		lines = [f"{role}: {utt}".strip() for role, _da, utt in dialog]
		lines.append(f"{dialog.USR} feeling:")
		return [
			{"role": 'user', "content": "\n".join(lines)},
			{"role": 'assistant', "content": f"{dialog[-1][1]}"},
		]

	def _heuristics_window(self, dialog:DialogSession, pred_end_idx=-1):
		"""``dialog`` cut to the last few turns, ending on the user turn to predict."""
		max_history_length = self.heuristic_args['max_hist_num_turns']
		if pred_end_idx == -1:
			pred_end_idx = len(dialog.history) - 1
		start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
		new_history = [turn for j, turn in enumerate(dialog) if start_idx <= j <= pred_end_idx]
		return DialogSession(dialog.SYS, dialog.USR).from_history(new_history)

	def process_heurstics_chat_exp(self, new_task_prompt: str):
		prompt_exps = []
		for i, exp in enumerate(self.conv_examples):
			for pred_end_turn in self.heuristic_args['example_pred_turn'][i]:
				window = self._heuristics_window(exp, pred_end_turn * 2 + 1)
				prompt_exps += self._heuristics_qa_pair(window)
				prompt_exps.append({"role": "system", "content": new_task_prompt})
		return prompt_exps[:-1]

	def predict_da(self, state:DialogSession, never_end=True) -> str:
		"""The persuadee act for the last user turn, by majority vote over 5 samples.

		``never_end`` keeps the terminal acts out of the vote so a live chat ends when the
		human says so, not when the classifier does.
		"""
		assert(state[-1][0] == PersuasionGame.USR)

		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
			{'role': 'system', 'content': self.new_task_prompt}
		]
		messages += self._heuristics_qa_pair(self._heuristics_window(state, -1))[:-1]

		inf_args = {
			"max_new_tokens": 5,
			"temperature": 0.7,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 5,
		}
		with role(USER_SIMULATOR):
			datas = self.backbone_model.chat_generate(messages, **inf_args)
		sampled_das: list = []
		for resp in datas:
			user_da = resp['generated_text'].strip()
			if user_da not in self.dialog_acts:
				sampled_das.append(PersuasionGame.U_Neutral)
			if never_end:
				if user_da == PersuasionGame.U_Donate:
					sampled_das.append(PersuasionGame.U_PositiveReaction)
				elif user_da == PersuasionGame.U_NoDonation:
					sampled_das.append(PersuasionGame.U_NegativeReaction)
				else:
					sampled_das.append(user_da)
			else:
				sampled_das.append(user_da)
		logger.info(f"sampled das: {sampled_das}")
		user_da = Counter(sampled_das).most_common(1)[0][0]
		return user_da