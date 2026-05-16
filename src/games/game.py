import numpy as np
import logging

from utils.gen_models import DialogModel
from utils.sessions import DialogSession
from abc import ABC, abstractmethod
from collections import defaultdict as ddict

logger = logging.getLogger(__name__)


class DialogGame(ABC):
    def __init__(
            self,
            dataset: str,
            system_name: str,
            system_agent: DialogModel,
            user_name: str,
            user_agent: DialogModel,
            planner,
            zero_shot: bool,
            success_base: float
    ):
        self.dataset = dataset
        self.SYS = system_name
        self.system_agent = system_agent
        self.USR = user_name
        self.user_agent = user_agent
        self.planner = planner
        self.zero_shot = zero_shot
        self.success_base = success_base
        return

    @staticmethod
    @abstractmethod
    def get_game_ontology() -> dict:
        """returns game related information such as dialog acts, slots, etc.
        """
        raise NotImplementedError

    def init_dialog(self) -> DialogSession:
        # [(sys_act, sys_utt, user_act, user_utt), ...]
        return DialogSession(self.SYS, self.USR)

    def map_user_action(self, v, sampled_das):
        raise NotImplementedError

    def get_next_state(self, state:DialogSession, action, mode: str = 'train') -> "Tuple[DialogSession, float]":
        # returns (next_state, v)
        raise NotImplementedError

    def display(self, state: DialogSession):
        string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
        print(string_rep)
        return

    @abstractmethod
    def get_dialog_ended(self, state) -> float:
        """returns 0 if not ended, then (in general) 1 if system success, -1 if failure
        """
        raise NotImplementedError