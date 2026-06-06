import logging
import random
from collections import defaultdict

import numpy as np
import math


from emotion_classifiers.llm_emotion import Emotions
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

		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])

		# Hard-prune to the LLM's top-K when --llm_prior_topk is set. Matches the
		# behaviour in OpenLoopMCTS._init_node so both gdpzero and emomcts paths
		# get pruning identically.
		topk = getattr(self.player, "llm_prior_topk", None)
		if topk is not None and 0 < topk < len(self.valid_moves[hashable_state]):
			ranked = sorted(self.valid_moves[hashable_state], key=lambda a: -self.P[hashable_state][a])
			kept = np.array(sorted(ranked[:topk]), dtype=self.valid_moves[hashable_state].dtype)
			self.valid_moves[hashable_state] = kept
			self.Nsa[hashable_state] = {a: 0 for a in kept}
			self.Q[hashable_state] = {a: self.configs.Q_0 for a in kept}
			mask = np.zeros_like(self.P[hashable_state])
			mask[kept] = 1.0
			self.P[hashable_state] = self.P[hashable_state] * mask
			s = self.P[hashable_state].sum()
			if s > 0:
				self.P[hashable_state] /= s
			else:
				for a in kept:
					self.P[hashable_state][a] = 1.0 / len(kept)
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


EMOTION_VALENCE_MINED = {
    Emotions.Fear:      +1.07,   # was -0.30 — flip sign
    Emotions.Happiness: +0.59,   # was +1.00 — halve
    Emotions.Anger:     +0.41,   # was -1.00 — flip; n is small (88), keep skeptical
    Emotions.Disgust:   +0.39,   # was -0.70 — flip; small n
    Emotions.Surprise:  +0.16,   # was +0.40 — halve
    Emotions.Neutral:   -0.09,   # was  0.00 — slight tax on apathy
    Emotions.Sadness:   -0.35,   # was -0.20 — moderate, data agrees on direction
    Emotions.Contempt:  -0.60,   # HF never emits — kept as hand value for LLM-classifier fallback
}

class EmotionAwareMultiObjectiveQ(EmotionAwareOpenLoopMCTS):
	"""
	Tracks TWO independent backups per (state, action):
	  Q[s][a]      — donation-rollout reward (existing behaviour, unchanged)
	  Q_emo[s][a]  — expected emotional valence at leaf, computed from
	                 next_state.predicted_distribution() via EMOTION_VALENCE

	PUCT becomes:
	    score(a) = Q[s][a]
	             + beta_emo * Q_emo[s][a]
	             + cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])
	"""

	def __init__(self, game, player, configs, emotion_classifier,
	             beta_emo: float = 0.3) -> None:
		super().__init__(game, player, configs, emotion_classifier)
		# Weight on the emotional-valence Q channel in PUCT. Explicit constructor arg
		# wins over configs; falls back to configs.beta_emo if present.
		self.beta_emo = float(
			beta_emo if beta_emo is not None
			else getattr(configs, "beta_emo", 0.3)
		)
		# Parallel value table, same shape as self.Q. Initialised lazily in _init_node.
		self.Q_emo: dict = {}

	def _emotion_quality(self, dist) -> float:
		"""E[valence] under a predicted emotion distribution. Bounded in [-1, +1];
		returns 0.0 when no distribution is attached (e.g. SYS turns / placeholders)."""
		if not dist:
			return 0.0
		return sum(p * EMOTION_VALENCE_MINED.get(e, 0.0) for e, p in dist.items())

	def _init_node(self, state):
		# Parent does Q / Nsa / P / valid_moves / realizations / topk-prune. Wrap to
		# ALSO init Q_emo on the (possibly-pruned) valid moves so the action sets
		# stay in lock-step between the two Q channels.
		v = super()._init_node(state)
		hashable_state = self._to_string_rep(state)
		self.Q_emo[hashable_state] = {a: 0.0 for a in self.valid_moves[hashable_state]}
		return v

	def _calculate_uct(self, hashable_state: str, action: int) -> float:
		Ns = self.Ns[hashable_state] or 1e-8
		explore = math.sqrt(Ns) / (1 + self.Nsa[hashable_state][action])
		q_emo = self.Q_emo.get(hashable_state, {}).get(action, 0.0)
		return (
			self.Q[hashable_state][action]
			+ self.beta_emo * q_emo
			+ self.configs.cpuct * self.P[hashable_state][action] * explore
		)

	def search(self, state):
		"""Same selection / expansion / backup structure as the parent, with a
		PARALLEL Q_emo backup alongside the donation Q backup. Both updates use
		the SAME old Nsa[a] (before increment) so the running-mean formula is
		consistent across channels.
		"""
		hashable_state = self._to_string_rep(state)

		terminated_v = self.game.get_dialog_ended(state)
		if terminated_v == 1.0:
			logger.debug("ended")
			return terminated_v

		if hashable_state not in self.P:
			v = self._init_node(state)
			return v
		else:
			self._add_new_realizations(state)

		# PUCT selection — _calculate_uct already folds in beta_emo * Q_emo.
		best_uct, best_action = -float("inf"), -1
		for a in self.valid_moves[hashable_state]:
			uct = self._calculate_uct(hashable_state, a)
			if uct > best_uct:
				best_uct, best_action = uct, a

		state = self._sample_realization(hashable_state)
		next_state = self._get_next_state(state, best_action)

		v = self.search(next_state)

		# Donation backup (parent's formula, unchanged).
		nsa_old = self.Nsa[hashable_state][best_action]
		self.Q[hashable_state][best_action] = (
			nsa_old * self.Q[hashable_state][best_action] + v
		) / (nsa_old + 1)

		# Parallel emotion-quality backup. Distribution is already cached on
		# next_state by EmotionAwarePersuasionGame.get_next_state — no extra
		# classifier call. Backed up *locally*: emo_v is from the immediate
		# child's user reaction, NOT the leaf's. This means Q_emo[s][a] is the
		# running mean of "how the user emotionally reacted when we took a
		# from s," which is what we want for selection bias.
		emo_v = self._emotion_quality(next_state.predicted_distribution())
		self.Q_emo[hashable_state][best_action] = (
			nsa_old * self.Q_emo[hashable_state][best_action] + emo_v
		) / (nsa_old + 1)

		# Increment counters AFTER both updates so they share the same old Nsa.
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# Realization V tracker keeps the donation v for inference-time NLG choice
		# (utterance pick still optimises donation; Q_emo only shapes search).
		self._update_realizations_Vs(next_state, v)
		return v

