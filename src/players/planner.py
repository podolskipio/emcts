from abc import ABC, abstractmethod


class DialogPlanner(ABC):
    @abstractmethod
    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        pass

    @abstractmethod
    def predict(self, state, policy=None, ent_bound=None):
        # returns (prob, value): a prior distribution over dialog acts and a scalar value estimate.
        # policy / ent_bound are optional and only used by chat planners that support a learned
        # policy prior; chat-formatted history is derived from `state.to_chat_messages()` on demand.
        pass
