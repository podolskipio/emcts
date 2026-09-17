import logging
from typing import Tuple

from emotion_classifiers.llm_emotion import Emotions
from games.game import DialogGame
from games.p4g_success import SUCCESS_CRITERIA, donation_counts
from utils.gen_models import DialogModel
from utils.sessions import DialogSession, EmotionAwareDialogSession

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
            infer_user_da,
            max_conv_turns=15,
            success_base=0.1,
            end_on_no_donation=False,
            success_criterion="tag",
    ):
        super().__init__('p4g', PersuasionGame.SYS, system_agent, PersuasionGame.USR, user_agent,
                         planner, infer_user_da, success_base, max_conv_turns)
        # GDP-Zero ended the dialogue in failure as soon as the persuadee said [no donation].
        self.end_on_no_donation = end_on_no_donation
        # --p4g_success: which [donate]-tagged turns end the episode (and a search branch) in
        # success. DEFAULT "tag" is GDP-Zero's reading; see games/p4g_success.py.
        if success_criterion not in SUCCESS_CRITERIA:
            raise ValueError(f"success_criterion must be one of {SUCCESS_CRITERIA}, got {success_criterion!r}")
        self.success_criterion = success_criterion

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
        return self._modal_user_action(v, sampled_das,
                                       success_da=PersuasionGame.U_Donate,
                                       default_da=PersuasionGame.U_Neutral)

    def get_dialog_ended(self, state) -> float:
        last_user_utt = ""
        for _role, da, _utt in state:
            if da == PersuasionGame.U_Donate and (
                    self.success_criterion == "tag"
                    or donation_counts(self.success_criterion, _utt, last_user_utt)):
                logger.info("p4g: dialog ended with donate")
                return 1.0
            if _role == PersuasionGame.USR:
                last_user_utt = _utt
            if self.end_on_no_donation and da == PersuasionGame.U_NoDonation:
                logger.info("p4g: dialog ended with no-donation")
                return -1.0
        return self._failure_or_continue(state)

    def get_next_state(self, state: DialogSession, action, mode: str = 'train') -> DialogSession:
        next_state = state.copy()
        sys_da = self.system_agent.dialog_acts[action]
        sys_utt = self.system_agent.get_utterance(next_state, action)
        next_state.add_single(state.SYS, sys_da, sys_utt)

        # p4g is the only game that will trust the simulator's own tag: GDP-Zero's persuadee
        # emits [donate] readily, so its self-report separates the cases. esc and cb always
        # relabel through the critic. Either way the search's leaf value comes from
        # planner.predict at node expansion, never from here.
        if self.infer_user_da:
            user_resp = self.user_agent.get_utterance(next_state, None, mode)
            next_state.add_single(state.USR, None, user_resp)
            v, sampled_das = self.planner.heuristic(next_state)
            next_state[-1][1] = self.map_user_action(v, sampled_das)
        else:
            user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)
            next_state.add_single(state.USR, user_da, user_resp)
        return next_state


class EmotionAwarePersuasionGame(PersuasionGame):
    """PersuasionGame with the persuadee's emotion classified on every turn.

    States are ``EmotionAwareDialogSession``s and ``get_next_state`` returns the emotion
    alongside the state, which is what the emotion-aware MCTS backs up into its Q_emo channel.
    """

    def __init__(
        self,
        system_agent: DialogModel,
        user_agent: DialogModel,
        planner,
        infer_user_da,
        emotion_classifier,
        max_conv_turns=15,
        success_base=0.1,
        success_criterion="tag",
    ):
        super().__init__(system_agent, user_agent, planner, infer_user_da,
                         max_conv_turns=max_conv_turns, success_base=success_base,
                         success_criterion=success_criterion)
        self.emotion_classifier = emotion_classifier

    def init_dialog(self) -> EmotionAwareDialogSession:
        return EmotionAwareDialogSession(self.SYS, self.USR)

    def _classify_user(self, state, user_resp) -> dict:
        """The emotion distribution over ``user_resp``, recorded on the run's classifier.

        Called before the turn is appended, so the recorded context is the dialogue the
        persuadee was reacting to.
        """
        dist = self.emotion_classifier.predict_distribution_from_full_history(state, user_resp)
        self.emotion_classifier.records.append({
            "utterance": user_resp,
            "emotion": str(max(dist, key=dist.get)),
            "context": state.to_string_rep(),
            "distribution": dist,
        })
        return dist

    def get_next_state(self, state: EmotionAwareDialogSession, action,
                       mode: str = 'train') -> Tuple[EmotionAwareDialogSession, Emotions]:
        next_state = state.copy()
        sys_da = self.system_agent.dialog_acts[action]
        sys_utt = self.system_agent.get_utterance(next_state, action)
        # only the user's emotion is classified; the system turn carries a placeholder
        next_state.add_single(state.SYS, sys_da, "Neutral", sys_utt)

        if self.infer_user_da:
            user_da, user_resp = None, self.user_agent.get_utterance(next_state, None, mode)
        else:
            user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)

        user_dist = self._classify_user(next_state, user_resp)
        user_emotion = max(user_dist, key=user_dist.get)
        next_state.add_single(state.USR, user_da, user_emotion, user_resp, user_dist)

        if self.infer_user_da:
            v, sampled_das = self.planner.heuristic(next_state)
            next_state.history[-1].da = self.map_user_action(v, sampled_das)
        return next_state, user_emotion
