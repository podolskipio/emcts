import numpy as np
import logging

from collections import defaultdict as ddict
from games.game import DialogGame
from utils.gen_models import DialogModel
from utils.sessions import EmotionSupportDialogSession, DialogSession

logger = logging.getLogger(__name__)


class EmotionalSupportGame(DialogGame):
	SYS = "Therapist"
	USR = "Patient"

	S_Question = "Question"
	S_SelfDisclosure = "Self-disclosure"
	S_AffirmationAndReassurance = "Affirmation and Reassurance"
	S_ReflectionOfFeelings = "Reflection of feelings"
	S_ProvidingSuggestions = "Providing Suggestions"
	S_Information = "Information"
	S_RestatementOrParaphrasing = "Restatement or Paraphrasing"
	S_Others = "Others"

	U_FeelWorse = "Feel worse"
	U_FeelTheSame = "Feel the same"
	U_FeelBetter = "Feel better"
	U_Solved = "Solved"

	def __init__(
			self,
			system_agent:DialogModel,
			user_agent:DialogModel,
            planner,
			zero_shot,
			max_conv_turns=15,
			success_base=0.1
	):
		super().__init__('esc', EmotionalSupportGame.SYS, system_agent, EmotionalSupportGame.USR, user_agent, planner, zero_shot, success_base)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
     	# ['Affirmation and Reassurance', 'Information', 'Others', 'Providing Suggestions', 'Question', 'Reflection of feelings', 'Restatement or Paraphrasing', 'Self-disclosure']
		return {
			"system": {
				"dialog_acts": [
					EmotionalSupportGame.S_AffirmationAndReassurance, EmotionalSupportGame.S_Information, EmotionalSupportGame.S_Others,
     				EmotionalSupportGame.S_ProvidingSuggestions, EmotionalSupportGame.S_Question, EmotionalSupportGame.S_ReflectionOfFeelings,
					EmotionalSupportGame.S_RestatementOrParaphrasing, EmotionalSupportGame.S_SelfDisclosure
				],
			},
			"user": {
				"dialog_acts": [
					EmotionalSupportGame.U_FeelWorse, EmotionalSupportGame.U_FeelTheSame,
     				EmotionalSupportGame.U_FeelBetter, EmotionalSupportGame.U_Solved
				]
			}
		}

	def get_dialog_ended(self, state) -> float:
		# terminate if the patient reports the issue is solved
		# allow only max_conv_turns turns
		for (_, da, _) in state:
			if da == EmotionalSupportGame.U_Solved:
				logger.info("Dialog ended with being solved")
				return 1.0
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended with failure for reaching maximum turns")
			return -1.0
		return 0.0

	def map_user_action(self, v, sampled_das):
		if v > self.success_base:
			return EmotionalSupportGame.U_Solved
		da_dict = ddict(int)
		for sample_da in sampled_das:
			if sample_da != 'Solved':
				da_dict[sample_da] += 1
		max_freq_da = max(da_dict, key=lambda x: da_dict[x])
		return max_freq_da

	def init_dialog(self, emotion_type, problem_type) -> EmotionSupportDialogSession:
    	# [(sys_act, sys_utt, user_act, user_utt), ...]
		return EmotionSupportDialogSession(self.SYS, self.USR, emotion_type, problem_type)

	def get_next_state(self, state:DialogSession, action, agent_state: list = None, mode: str = 'train') -> "Tuple[DialogSession, list, float]":
		next_state = state.copy()
		next_agent_state = agent_state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		next_agent_state.append({'role': state.SYS, 'content': sys_utt})

		# state in user's perspective
		if not self.zero_shot:
			user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, user_da, user_resp)
			v = None
		else:
			user_resp = self.user_agent.get_utterance(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, None, user_resp)
			# v, sampled_das = 0.1, ["No, better"] * 10
			# if len(state) == 8:
			# 	v = 0.19
			# 	sampled_das[-1] = "Yes, Solved"
			v, sampled_das = self.planner.heuristic(next_state)
			user_da = self.map_user_action(v, sampled_das)
			next_state[-1][1] = user_da
		next_agent_state.append({'role': state.USR, 'content': user_resp})
		return next_state, next_agent_state, v
