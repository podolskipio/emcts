import numpy as np
import logging

from collections import defaultdict as ddict
from games.game import DialogGame
from utils.gen_models import DialogModel
from utils.sessions import DialogSession

logger = logging.getLogger(__name__)


class PersuasionGame(DialogGame):
    SYS = "Persuader"
    USR = "Persuadee"

    S_PersonalStory = "personal story"
    S_CredibilityAppeal = "credibility appeal"
    S_EmotionAppeal = "emotion appeal"
    S_PropositionOfDonation = "proposition of donation"
    S_FootInTheDoor = "foot in the door"
    S_LogicalAppeal = "logical appeal"
    S_SelfModeling = "self modeling"
    S_TaskRelatedInquiry = "task related inquiry"
    S_SourceRelatedInquiry = "source related inquiry"
    S_PersonalRelatedInquiry = "personal related inquiry"
    S_NeutralToInquiry = "neutral to inquiry"
    S_Greeting = "greeting"
    S_Other = "other"

    U_NoDonation = "no donation"
    U_NegativeReaction = "negative reaction"
    U_Neutral = "neutral"
    U_PositiveReaction = "positive reaction"
    U_Donate = "donate"

    def __init__(
            self,
            system_agent: DialogModel,
            user_agent: DialogModel,
            planner,
            zero_shot,
            max_conv_turns=15,
            success_base=0.1
    ):
        super().__init__('p4g', PersuasionGame.SYS, system_agent, PersuasionGame.USR, user_agent, planner, zero_shot, success_base)
        self.max_conv_turns = max_conv_turns
        return

    @staticmethod
    def get_game_ontology() -> dict:
        return {
            "system": {
                "dialog_acts": [
                    PersuasionGame.S_PersonalStory, PersuasionGame.S_CredibilityAppeal, PersuasionGame.S_EmotionAppeal,
                    PersuasionGame.S_PropositionOfDonation, PersuasionGame.S_FootInTheDoor, PersuasionGame.S_LogicalAppeal,
                    PersuasionGame.S_SelfModeling, PersuasionGame.S_TaskRelatedInquiry, PersuasionGame.S_SourceRelatedInquiry,
                    PersuasionGame.S_PersonalRelatedInquiry, PersuasionGame.S_NeutralToInquiry, PersuasionGame.S_Greeting,
                    PersuasionGame.S_Other
                ],
            },
            "user": {
                "dialog_acts": [
                    PersuasionGame.U_NoDonation, PersuasionGame.U_NegativeReaction, PersuasionGame.U_Neutral,
                    PersuasionGame.U_PositiveReaction, PersuasionGame.U_Donate
                ]
            }
        }

    def map_user_action(self, v, sampled_das):
        if v > self.success_base:
            return PersuasionGame.U_Donate
        da_dict = ddict(int)
        for sample_da in sampled_das:
            if sample_da != PersuasionGame.U_Donate:
                da_dict[sample_da] += 1
        if len(da_dict) == 0:
            return PersuasionGame.U_Neutral
        max_freq_da = max(da_dict, key=lambda x: da_dict[x])
        return max_freq_da

    def get_dialog_ended(self, state) -> float:
        # terminate if there is a <donate> action in the persuadee's response
        # allow only max_conv_turns turns
        if len(state) >= self.max_conv_turns:
            logger.info("Dialog ended with persuasion failure")
            return -1.0
        for (_, da, _) in state:
            if da == PersuasionGame.U_Donate:
                logger.info("Dialog ended with donate")
                return 1.0
            if da == PersuasionGame.U_NoDonation:
                logger.info("Dialog ended with no-donation")
                return -1.0
        return 0.0

    def get_next_state(self, state: DialogSession, action, mode: str = 'train') -> "Tuple[DialogSession, float]":
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
