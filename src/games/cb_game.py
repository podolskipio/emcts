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

    def __init__(
            self,
            system_agent: DialogModel,
            user_agent: DialogModel,
            planner,
            infer_user_da,
            max_conv_turns=15,
            success_base=0.1,
    ):
        super().__init__('cb', CBGame.SYS, system_agent, CBGame.USR, user_agent,
                         planner, infer_user_da, success_base, max_conv_turns)

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
        """Did the pair close? Decided by the critic's majority verdict, not by the value.

        The value here is the price-normalized reward, i.e. how good the deal was for the
        buyer, so a deal at the seller's asking price scores 0.0. Testing it against
        success_base recorded those as no-deal, the negotiation ran to the turn cap, and CB
        success rate was 0.
        """
        n_deal = sum(1 for da in sampled_das if da == CBGame.U_Deal)
        n_no_deal = sum(1 for da in sampled_das if da == CBGame.U_No_deal)
        if n_deal > n_no_deal:
            logger.info(f"critic majority says deal ({n_deal}/{len(sampled_das)}), value {v}")
            return CBGame.U_Deal
        return CBGame.U_No_deal

    def get_dialog_ended(self, state) -> float:
        for _role, da, _utt in state:
            if da == CBGame.U_Deal:
                logger.info("cb: dialog ended with a deal")
                return 1.0
        return self._failure_or_continue(state)

    def init_dialog(self, item_name, buyer_item_description, buyer_price,
                    seller_item_description, seller_price) -> CBDialogSession:
        return CBDialogSession(self.SYS, self.USR, item_name, buyer_item_description,
                               buyer_price, seller_item_description, seller_price)

    def get_next_state(self, state: DialogSession, action, mode: str = 'train') -> DialogSession:
        next_state = state.copy()
        sys_da = self.system_agent.dialog_acts[action]
        sys_utt = self.system_agent.get_utterance(next_state, action)
        next_state.add_single(state.SYS, sys_da, sys_utt)

        # The seller writes the utterance and the critic decides whether it closed the deal,
        # whatever infer_user_da says: both branches used to run exactly this path.
        user_resp = self.user_agent.get_utterance(next_state, None, mode)
        next_state.add_single(state.USR, None, user_resp)
        v, sampled_das = self.planner.heuristic(next_state)
        next_state[-1][1] = self.map_user_action(v, sampled_das)
        return next_state
