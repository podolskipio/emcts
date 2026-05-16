import numpy as np
import logging

from games.game import DialogGame
from utils.gen_models import DialogModel
from utils.sessions import CBDialogSession, DialogSession

logger = logging.getLogger(__name__)


class CBGame(DialogGame):
	SYS = "Buyer"
	USR = "Seller"

	S_Affirm = "affirm"
	S_Agree = "agree"
	S_Confirm = "confirm"
	S_Counter = "counter"
	S_Counter_noprice = "counter-noprice"
	S_Deny = "deny"
	S_Disagree = "disagree"
	S_Greet = "greet"
	S_Information = "inform"
	S_Inquire = "inquire"
	S_Propose = "propose"

	U_Deal = "deal"
	U_No_deal = "no deal"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel,
              planner, zero_shot, max_conv_turns=15, success_base=0.1):
		super().__init__('cb', CBGame.SYS, system_agent, CBGame.USR, user_agent, planner, zero_shot, success_base)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
		return {
			"system": {
				"dialog_acts": [
					CBGame.S_Affirm, CBGame.S_Agree, CBGame.S_Confirm, CBGame.S_Counter,
     				CBGame.S_Counter_noprice, CBGame.S_Deny, CBGame.S_Disagree, CBGame.S_Greet,
					CBGame.S_Information, CBGame.S_Inquire, CBGame.S_Propose,
				],
			},
			"user": {
				"dialog_acts": [
					CBGame.U_Deal, CBGame.U_No_deal,
				]
			}
		}

	def map_user_action(self, v, sampled_das):
		if v > self.success_base:
			logger.info("evaluation value {} surpasses success base {}".format(v, self.success_base))
			return CBGame.U_Deal
		elif v > 0:
			logger.info("evaluation value {} fails to surpasses success base {}".format(v, self.success_base))
			return CBGame.U_No_deal
		return CBGame.U_No_deal

	def get_dialog_ended(self, state) -> float:
		# terminate if the seller agrees to a deal
		# allow only max_conv_turns turns
		for (_, da, _) in state:
			if da == CBGame.U_Deal:
				logger.info("Dialog ended with deal")
				return 1.0
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended with failure for reaching maximum turns")
			return -1.0
		return 0.0

	def init_dialog(self, item_name, buyer_item_description, buyer_price,
                 seller_item_description, seller_price) -> CBDialogSession:
		# [(sys_act, sys_utt, user_act, user_utt), ...]
		return CBDialogSession(self.SYS, self.USR, item_name, buyer_item_description, buyer_price,
                         seller_item_description, seller_price)

	def get_next_state(self, state:DialogSession, action, mode: str = 'train') -> "Tuple[DialogSession, float]":
		next_state = state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)

		# state in user's perspective
		if not self.zero_shot:
			user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, user_da, user_resp)
			v = None
		else:
			user_resp = self.user_agent.get_utterance(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, None, user_resp)
			v, sampled_das = self.planner.heuristic(next_state)
			user_da = self.map_user_action(v, sampled_das)
			next_state[-1][1] = user_da
		return next_state, v
