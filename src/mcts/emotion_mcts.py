import logging
import random
from collections import defaultdict

import numpy as np
import math


from emotion_classifiers.llm_emotion import NEGATIVE_EMOTIONS, Emotions
from mcts.mcts import OpenLoopMCTS
from utils.sessions import EmotionAwareDialogSession


logger = logging.getLogger(__name__)


class EmotionAwareOpenLoopMCTS(OpenLoopMCTS):
	def __init__(self, game, player, configs, emotion_classifier) -> None:
		super().__init__(game, player, configs)
		self.emotion_classifier = emotion_classifier
		self.emotions_count = defaultdict(self.create_emotions_dict) # state -> emotion_distribution (open loop variant)


	def create_emotions_dict(self):
		return {emotion: 0 for emotion in self.emotion_classifier.emotions}

	def _to_string_rep(self, state: EmotionAwareDialogSession) -> str:
		# for tree search, key a node by its system dialog-act prefix
		das = []
		for rec in state.history:
			if rec.role == state.SYS:
				das.append(rec.da)
		return "__".join(das)

	def _init_node(self, state: EmotionAwareDialogSession):
		hashable_state: str = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		self.realizations[hashable_state] = [state.copy()]

		# keeping emotional responses across nodes
		# self.emotions_count[hashable_state] = {emotion: 0 for emotion in self.emotion_classifier.emotions}
		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v

	def _sample_realization(self, hashable_state):
		rand_i = np.random.randint(len(self.realizations[hashable_state]))
		return self.realizations[hashable_state][rand_i]

	def _add_new_realizations(self, state: EmotionAwareDialogSession):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = []
		if state in self.realizations[hashable_state]:
			return

		self.realizations[hashable_state].append(state.copy())
		if len(self.realizations[hashable_state]) > self.max_realizations:
			# should never happen
			logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
			self.realizations[hashable_state].pop(0)
		return

	def _get_next_state_emotions(self, state: EmotionAwareDialogSession, action: int) -> dict:
		# emotions_count is keyed by the DA-prefix of the *child* state (parent prefix + "__" + da),
		# so look up using the same key builder update_emotions writes with.
		next_state_hash = self._get_hash_for_next_action(self._to_string_rep(state), action)
		return self.emotions_count.get(next_state_hash, {})

	def _get_hash_for_next_action(self, hashable_state, action):
		return hashable_state + "__" + self.player.dialog_acts[action]

	def update_emotions(self, current_state: EmotionAwareDialogSession, next_action: int, emotion: Emotions) -> None:
		next_state_hash = self._get_hash_for_next_action(self._to_string_rep(current_state), next_action)
		self.emotions_count[next_state_hash][emotion] += 1

	def _get_next_state(self, state: EmotionAwareDialogSession, best_action: int):
		prefetch_state = self._get_hash_for_next_action(self._to_string_rep(state), best_action)
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			return self._sample_realization(prefetch_state)

		# otherwise, generate a new realization
		next_state, _, emotion = self.game.get_next_state(state, best_action)
		self.update_emotions(state, best_action, emotion)
		return next_state

	def _update_realizations_Vs(self, state: EmotionAwareDialogSession, v: float):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations_Vs:
			self.realizations_Vs[hashable_state] = {}
			self.realizations_Ns[hashable_state] = {}
		sys_utt = state.get_turn_utt(
			turn=-1,
			role=state.SYS,
		)
		if sys_utt not in self.realizations_Vs[hashable_state]:
			self.realizations_Vs[hashable_state][sys_utt] = 0
			self.realizations_Ns[hashable_state][sys_utt] = 0
		# update
		self.realizations_Ns[hashable_state][sys_utt] += 1
		self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / \
														 self.realizations_Ns[hashable_state][sys_utt]
		return

	def _calculate_uct(self, hashable_state: str, action: int) -> float:
		Ns = self.Ns[hashable_state]
		if Ns == 0:
			Ns = 1e-8
		# a variant of PUCT
		uct = self.Q[hashable_state][action] + self.configs.cpuct * self.P[hashable_state][action] * math.sqrt(Ns) / (
				1 + self.Nsa[hashable_state][action])
		return uct


	def search(self, state: EmotionAwareDialogSession):
		hashable_state = self._to_string_rep(state)

		# check everytime since state is stochastic, does not map to hashable_state
		terminated_v = self.game.get_dialog_ended(state)
		# check if it is terminal node
		if terminated_v == 1.0:
			logger.debug("ended")
			return terminated_v

		# otherwise, if is nontermial leaf node, we initialize and return v
		if hashable_state not in self.P:
			# selected leaf node, expand it
			# first visit V because v is only evaluated once for a hashable_state
			v = self._init_node(state)
			return v
		else:
			# add only when it is new
			self._add_new_realizations(state)

		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			uct = self._calculate_uct(hashable_state, a)
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition. For open loop, first sample from an existing realization
		state = self._sample_realization(hashable_state)
		next_state = self._get_next_state(state, best_action)
		emotion = next_state.predicted_emotion()

		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][
			best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# update v to realizations for NLG at inference
		self._update_realizations_Vs(next_state, v)
		# now we are single player, hence just v instead of -v
		return v

	def get_best_realization(self, state: EmotionAwareDialogSession, action: int):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[action]
		if prefetch_state not in self.realizations_Vs:
			raise Exception("querying a state that has no realizations sampled before")
		# get the counts for all moves
		# convert to prob
		curr_best_v = -float('inf')
		curr_best_realization = None
		for sys_utt, v in self.realizations_Vs[prefetch_state].items():
			if v > curr_best_v:
				curr_best_v = v
				curr_best_realization = sys_utt
		return curr_best_realization


class EmotionAwareDiscountQOpenLoopMCTS(EmotionAwareOpenLoopMCTS):
	def __init__(self, game, player, configs, emotion_classifier, emo_lambda: float | None = None) -> None:
		super().__init__(game, player, configs, emotion_classifier)
		self.emo_lambda = emo_lambda

	def _get_emotion_penalty(self, emotion: Emotions) -> float:
		# Penalties scaled to impact a standard [0, 1] 'v' value
		penalties = {
			Emotions.Anger: -1.0,
			Emotions.Fear: 0.2,
			Emotions.Disgust: -0.7,
			Emotions.Contempt: -0.7,
			Emotions.Sadness: 0.3,
			Emotions.Surprise: 0.1,
			Emotions.Neutral: -0.1,
			Emotions.Happiness: 0,
		}
		return penalties.get(emotion, 0.0)

	def search(self, state: EmotionAwareDialogSession):
		hashable_state = self._to_string_rep(state)

		# check everytime since state is stochastic, does not map to hashable_state
		terminated_v = self.game.get_dialog_ended(state)
		# check if it is terminal node
		if terminated_v == 1.0:
			logger.debug("ended")
			return terminated_v

		# otherwise, if is nontermial leaf node, we initialize and return v
		if hashable_state not in self.P:
			# selected leaf node, expand it
			# first visit V because v is only evaluated once for a hashable_state
			v = self._init_node(state)
			return v
		else:
			# add only when it is new
			self._add_new_realizations(state)

		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			uct = self._calculate_uct(hashable_state, a)
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition. For open loop, first sample from an existing realization
		state = self._sample_realization(hashable_state)
		next_state = self._get_next_state(state, best_action)
		emotion = next_state.predicted_emotion()

		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# update stats. Apply the emotion penalty *locally* to Q(s, a) and realizations_Vs (it
		# credits/discredits the action that just produced this emotion), but return the leaf v
		# unchanged so the penalty does not compound up the backup chain. Returning v + penalty
		# from search() would stack penalties through nested recursive calls and pull Q below the
		# [-1, +1] range that the PUCT formula assumes.

		# emotion_penalty = self._get_emotion_penalty(emotion)
		# blended_v = v + emotion_penalty

		# Convex blend of task value with emotion penalty: v~ = (1-λ)·v + λ·π(e).
		# With π(e) ∈ [-1, +1] and v ∈ [-1, +1], v~ is bounded in [-1, +1] for any λ ∈ [0, 1],
		# so Q stays in the PUCT-calibrated range and c_p stays valid. λ=0 collapses to GDPZero;
		# applied locally (we return v, not v~) so the penalty does not compound up the backup.
		emotion_penalty = self._get_emotion_penalty(emotion)
		if self.emo_lambda is not None:
			blended_v = (1.0 - self.emo_lambda) * v + self.emo_lambda * emotion_penalty
		else:
			blended_v = v + emotion_penalty

		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][
			best_action] + blended_v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# update v to realizations for NLG at inference
		self._update_realizations_Vs(next_state, blended_v)
		# now we are single player, hence just v instead of -v
		return v