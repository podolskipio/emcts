import logging

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
            system_agent: DialogModel,
            user_agent: DialogModel,
            planner,
            infer_user_da,
            max_conv_turns=15,
            success_base=0.1,
    ):
        super().__init__('esc', EmotionalSupportGame.SYS, system_agent, EmotionalSupportGame.USR,
                         user_agent, planner, infer_user_da, success_base, max_conv_turns)

    @staticmethod
    def get_game_ontology() -> dict:
        return {
            "system": {
                "dialog_acts": [
                    EmotionalSupportGame.S_AffirmationAndReassurance, EmotionalSupportGame.S_Information,
                    EmotionalSupportGame.S_Others, EmotionalSupportGame.S_ProvidingSuggestions,
                    EmotionalSupportGame.S_Question, EmotionalSupportGame.S_ReflectionOfFeelings,
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

    def map_user_action(self, v, sampled_das):
        return self._modal_user_action(v, sampled_das,
                                       success_da=EmotionalSupportGame.U_Solved,
                                       default_da=EmotionalSupportGame.U_FeelTheSame)

    def get_dialog_ended(self, state) -> float:
        for _role, da, _utt in state:
            if da == EmotionalSupportGame.U_Solved:
                logger.info("esc: dialog ended with the issue solved")
                return 1.0
        return self._failure_or_continue(state)

    def init_dialog(self, emotion_type, problem_type) -> EmotionSupportDialogSession:
        return EmotionSupportDialogSession(self.SYS, self.USR, emotion_type, problem_type)

    def get_next_state(self, state: DialogSession, action, mode: str = 'train') -> DialogSession:
        next_state = state.copy()
        sys_da = self.system_agent.dialog_acts[action]
        sys_utt = self.system_agent.get_utterance(next_state, action)
        next_state.add_single(state.SYS, sys_da, sys_utt)

        # The critic labels the patient's turn either way -- infer_user_da only decides whether
        # the patient was asked to tag itself at all, and any tag it gives is discarded. It
        # role-plays its problem persistently and so never self-reports Solved -- the one act
        # get_dialog_ended scores as success -- which made ESC success rate 0 by construction.
        # Probed on a conversation where the patient had just said it felt much better and knew
        # what to do: the patient self-tagged Feel the same / Feel worse 12/12, while the critic
        # returned Solved 10/10 there and Feel worse 10/10 on an unresolved one. cb works the
        # same way; p4g keeps GDP-Zero's self-tagging, where the persuadee does say [donate].
        if self.infer_user_da:
            user_resp = self.user_agent.get_utterance(next_state, None, mode)
        else:
            _self_tag, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)
        next_state.add_single(state.USR, None, user_resp)
        v, sampled_das = self.planner.heuristic(next_state)
        next_state[-1][1] = self.map_user_action(v, sampled_das)
        return next_state
