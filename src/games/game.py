import logging
from abc import ABC, abstractmethod
from collections import Counter

from utils.gen_models import DialogModel
from utils.sessions import DialogSession

logger = logging.getLogger(__name__)


class DialogGame(ABC):
    """One dialogue task: who the two speakers are, what ends the conversation, and how a
    system dialog act turns into the next state.

    A subclass supplies its two act sets (``get_game_ontology``), what counts as success
    (``get_dialog_ended``), and the wiring that decides who labels the user's turn
    (``get_next_state``).
    """

    def __init__(
            self,
            dataset: str,
            system_name: str,
            system_agent: DialogModel,
            user_name: str,
            user_agent: DialogModel,
            planner,
            infer_user_da: bool,
            success_base: float,
            max_conv_turns: int = 15,
    ):
        self.dataset = dataset
        self.SYS = system_name
        self.system_agent = system_agent
        self.USR = user_name
        self.user_agent = user_agent
        self.planner = planner
        # True when nobody tags the user's turn with a [dialog act]: the user agent writes
        # only an utterance and the planner's critic infers the act from it. False is
        # GDP-Zero's arrangement, where the user simulator declares its own act.
        self.infer_user_da = infer_user_da
        self.success_base = success_base
        self.max_conv_turns = max_conv_turns

    @staticmethod
    @abstractmethod
    def get_game_ontology() -> dict:
        """The dialog acts available to each side: ``{"system": {...}, "user": {...}}``."""
        raise NotImplementedError

    def init_dialog(self) -> DialogSession:
        return DialogSession(self.SYS, self.USR)

    def get_next_state(self, state: DialogSession, action, mode: str = 'train') -> DialogSession:
        raise NotImplementedError

    @abstractmethod
    def get_dialog_ended(self, state) -> float:
        """0.0 while the dialogue is live, 1.0 on system success, -1.0 on failure."""
        raise NotImplementedError

    @staticmethod
    def state_of(result):
        """The state out of a ``get_next_state`` return, which the emotion-aware games
        widen to ``(state, emotion)``."""
        return result[0] if isinstance(result, tuple) else result

    def map_user_action(self, v, sampled_das):
        raise NotImplementedError

    def _modal_user_action(self, v, sampled_das, success_da, default_da):
        """The critic's verdict on the user's turn.

        The success act is reachable only through the value clearing ``success_base``, so it
        is kept out of the vote below; otherwise the user's turn takes whichever act the
        critic's samples named most often. ``default_da`` covers the case where every sample
        was the success act, or none of them parsed.
        """
        if v > self.success_base:
            return success_da
        counts = Counter(da for da in sampled_das if da != success_da)
        if not counts:
            return default_da
        return max(counts, key=counts.get)

    def _failure_or_continue(self, state) -> float:
        """-1.0 once the dialogue is out of turns or looping, else 0.0.

        The shared tail of every ``get_dialog_ended``: each game checks its own success act
        first and falls through to here.
        """
        if len(state) >= self.max_conv_turns:
            logger.info(f"{self.dataset}: dialog ended in failure (turn limit)")
            return -1.0
        if self.is_stalled(state):
            logger.info(f"{self.dataset}: dialog ended in failure (verbatim repetition loop)")
            return -1.0
        return 0.0

    def is_stalled(self, state) -> bool:
        """True once both speakers have started repeating themselves word for word."""
        history = list(state)  # (role, da, utt) triples on both session types
        if len(history) < 4:
            return False
        sys_utts, usr_utts = [], []
        for role, _da, utt in history:
            (sys_utts if role == self.SYS else usr_utts).append(self._norm_utt(utt))
        if len(sys_utts) < 2 or len(usr_utts) < 2:
            return False
        return sys_utts[-1] in sys_utts[:-1] and usr_utts[-1] in usr_utts[:-1]

    @staticmethod
    def _norm_utt(utt) -> str:
        return " ".join(str(utt).split()).strip().lower()

    def display(self, state: DialogSession):
        print(state.to_string_rep(keep_sys_da=True, keep_user_da=True))
