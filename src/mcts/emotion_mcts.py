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
		self.node_V[hashable_state] = v  # same value, keyed for the subtree log
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
		topk = self.player.llm_prior_topk
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
			self._record_cache(prefetch_state, True)
			return self._sample_realization(prefetch_state)

		# otherwise, generate a new realization
		self._record_cache(prefetch_state, False)
		next_state, emotion = self.game.get_next_state(state, best_action)
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


# ---------------------------------------------------------------------------
# Welford accumulators for the emotion channel
#
# The open-loop tree keys a node by its system-DA prefix, so a single edge (s,a) is an
# *action sequence*, not a state: across simulations it genuinely sees different user
# reactions and therefore different z. The running mean Q_emo throws that spread away.
# These three functions keep it, at a cost of one extra float per edge (M2_emo).
# ---------------------------------------------------------------------------
def welford_update(n_old: int, mean: float, m2: float, z: float) -> "tuple[float, float]":
	"""One Welford step. ``n_old`` is the visit count BEFORE this observation.

	The mean line is algebraically identical to the running mean it replaces,
	``(n_old*mean + z) / (n_old + 1)`` -- that identity is what guarantees the
	default behaviour of the planner is unchanged (see the unit tests).

	M2 needs the deviation from the OLD mean times the deviation from the NEW mean.
	Reusing the same deviation twice is the classic Welford bug and silently yields
	the wrong variance, so the two are named apart here on purpose.
	"""
	delta = z - mean                 # deviation from the pre-update mean
	mean = mean + delta / (n_old + 1)
	delta2 = z - mean                # deviation from the UPDATED mean
	m2 = m2 + delta * delta2
	return mean, m2


def welford_variance(n: int, m2: float) -> float:
	"""Sample variance (ddof=1) from the accumulator. 0.0 below two observations."""
	if n < 2:
		return 0.0
	return m2 / (n - 1)


def welford_sigma(n: int, m2: float) -> float:
	"""sigma_emo. Derived, never stored. max(0,...) guards float noise on near-zero M2."""
	return math.sqrt(max(0.0, welford_variance(n, m2)))


# --emo_signal choices. `level` is the shipped behaviour; `delta` is the Tier-C arm.
EMO_SIGNALS = ("level", "delta")

# z_delta = nu(d_s') - nu(d_parent) lives in [-2, +2] because nu lives in [-1, +1].
# Dividing by this keeps Q_emo in [-1, +1] under BOTH settings, so beta_emo means the
# same thing in the `level` and `delta` arms and the two are comparable. Reported in
# the paper alongside the delta results.
EMO_DELTA_RANGE_SCALE = 2.0


def check_emo_signal_flags(emo_signal: str, emo_n_step: int = 1, default_n_step: int = 1) -> None:
	"""Refuse to cross `--emo_signal delta` with a non-default n-step horizon.

	With z_j = nu_{j+1} - nu_j the discounted n-step return telescopes toward
	nu_{t+n} - nu_t as gamma -> 1. At gamma=0.9 it does not fully collapse, but
	n-step-over-deltas is close to terminal-minus-initial, so the delta arm and the
	n-step arm (B2) are not independent conditions. B2 runs on `level`.

	No runner exposes an n-step flag yet, so every caller passes the signal alone and this
	sees the default horizon -- the guard never fires today. It is here so it goes live the
	moment B2's flag lands, without touching backup code mid-grid.
	"""
	if emo_signal == "delta" and emo_n_step != default_n_step:
		raise ValueError(
			f"--emo_signal delta cannot be combined with a non-default n-step horizon "
			f"(n={emo_n_step}, default {default_n_step}): the discounted n-step return over "
			f"deltas telescopes toward nu_terminal - nu_initial, so the two conditions are not "
			f"independent. The n-step experiment (B2) runs on --emo_signal level."
		)


# The ARGMAX variant of the same mining, kept selectable so the two can be compared without
# editing a module constant mid-grid (which is exactly what the Week-1 schema freeze forbids).
# Same corpus (all 300 annotated p4g dialogs), same base rate 0.493, same alpha=50 shrinkage --
# the ONLY difference is Algorithm 2 line 7's assignment rule: argmax credits a whole turn to
# argmax_e Phi(e|u), while soft credits every emotion its posterior mass.
#
# Soft is the default because it matches what the planner consumes: _emotion_quality scores
# nu(d) = sum_e d(e)*w(e) over the FULL softmax, so mining by argmax fits w(e) on a different
# functional of Phi than deployment reads. The argmax table is retained as the ablation that
# shows how much that mismatch was worth -- most visibly on the thin cells, where argmax's
# w(fear)=+0.62 rests on the 70 turns where fear happened to win a close posterior.
EMOTION_VALENCE_ARGMAX = {
    Emotions.Fear:      +0.62,   # n_turns=70,  straddles base — an argmax artefact; soft puts it at +0.24
    Emotions.Happiness: +0.56,   # n_turns=1145, the only cell whose CI excludes the base rate
    Emotions.Anger:     +0.26,   # n_turns=88,  straddles base
    Emotions.Disgust:   +0.25,   # n_turns=92,  straddles base
    Emotions.Surprise:  +0.15,   # n_turns=469,  straddles base
    Emotions.Neutral:   -0.09,   # n_turns=2660, straddles base
    Emotions.Sadness:   -0.30,   # n_turns=323,  straddles base
    Emotions.Contempt:  -0.60,   # HF never emits — hand value, identical in both tables
}

# Mined by src/emotion_mining/mine_emotion_donation_p4g.py --soft over all 300 annotated p4g
# dialogs, base rate 0.493. w(e) = 10 * lift(e), where the lift is shrunk toward the base rate
# with alpha=50 pseudo-observations so under-evidenced cells cannot dominate the channel.
#
# SOFT assignment (2026-09-06). Mining now credits every emotion its posterior mass --
# T(e) += d(e), S(e) += d(e)*y -- instead of crediting only argmax_e Phi(e|u). This matches
# what the planner actually consumes below: _emotion_value() scores nu(d) = sum_e d(e)*w(e)
# over the full softmax, so mining by argmax was fitting w(e) on a different functional of
# Phi than deployment reads. Total mass is unchanged (4847 turn-equivalents either way), but
# it redistributes from the dominant cells to the thin ones: fear's effective n goes 70 -> 120,
# anger 88 -> 187, disgust 92 -> 203, while happiness 1145 -> 1064 and neutral 2660 -> 2294.
# Largest weight change: fear +0.62 -> +0.24. The old +0.62 was an argmax artifact -- those 70
# turns were the ones where fear happened to win a close posterior, and their donation rate
# (0.600) regresses to 0.527 once fear's partial mass across all turns is counted.

EMOTION_VALENCE_MINED = {
    Emotions.Happiness: +0.54,   # n_eff=1064, the only cell whose CI excludes the base rate
    Emotions.Fear:      +0.24,   # n_eff=120,  straddles base — was +0.62 under argmax
    Emotions.Disgust:   +0.13,   # n_eff=203,  straddles base
    Emotions.Anger:     +0.11,   # n_eff=187,  straddles base
    Emotions.Surprise:  +0.09,   # n_eff=589,  straddles base
    Emotions.Neutral:   -0.07,   # n_eff=2294, straddles base — slight tax on apathy
    Emotions.Sadness:   -0.14,   # n_eff=390,  straddles base
    Emotions.Contempt:  -0.60,   # HF never emits — kept as hand value for LLM-classifier fallback
}

# --emo_valence_table selects between them. "soft" is the deployed default and is the exact
# object EMOTION_VALENCE_MINED names, so the default path is bit-identical to before.
EMOTION_VALENCE_TABLES = {
    "soft": EMOTION_VALENCE_MINED,
    "argmax": EMOTION_VALENCE_ARGMAX,
}
EMO_VALENCE_TABLES = tuple(EMOTION_VALENCE_TABLES)


class EmotionAwareMultiObjectiveQ(EmotionAwareOpenLoopMCTS):
	"""
	Tracks TWO independent backups per (state, action):
	  Q[s][a]      — donation-rollout reward (existing behaviour, unchanged)
	  Q_emo[s][a]  — expected emotional valence at leaf, computed from
	                 next_state.predicted_distribution() via EMOTION_VALENCE

	PUCT becomes:
	    score(a) = Q[s][a]
	             + beta_emo * (Q_emo[s][a] - emo_risk_lambda * sigma_emo[s][a])
	             + cpuct * P[s][a] * sqrt(Ns) / (1 + Nsa[s][a])

	with ``emo_risk_lambda = 0.0`` (the default) reducing the middle term to
	``beta_emo * Q_emo[s][a]`` exactly -- ``x - 0.0*sigma`` is bit-identical to ``x``
	for every finite sigma, so no branch and no epsilon are needed to recover the
	pre-change score.

	Alongside the two means each edge carries a Welford accumulator ``M2_emo[s][a]``,
	from which sigma_emo is derived at read time. It costs two floats per edge and
	makes the within-node spread of the emotion signal -- which the running mean
	otherwise discards -- available to the variance analysis straight out of the main
	grid, and to the risk-adjusted selection rule above.
	"""

	# w(e) used by _emotion_quality. __init__ replaces it with the table --emo_valence_table
	# names; the class-level default is the deployed "soft" one.
	valence_weights: dict = EMOTION_VALENCE_MINED

	def __init__(self, game, player, configs, emotion_classifier,
	             beta_emo: float = 0.3,
	             emo_risk_lambda: float = 0.0,
	             emo_signal: str = "level",
	             emo_valence_table: str = "soft") -> None:
		super().__init__(game, player, configs, emotion_classifier)
		# The four knobs below are constructor arguments, not configs entries: every caller
		# passes them explicitly (see runners/emomcts.py and runners/rollout.py), and the
		# defaults here are the shipped pre-change behaviour.
		#
		# Weight on the emotional-valence Q channel in PUCT.
		self.beta_emo = float(beta_emo)
		# Risk aversion on the emotion channel: Q_emo - lambda*sigma_emo. DEFAULT 0.0,
		# which is exactly the pre-change rule. Instrumentation for a later Tier-C
		# experiment; not swept now.
		self.emo_risk_lambda = float(emo_risk_lambda)
		# Which quantity gets backed up into Q_emo. DEFAULT "level" = the shipped
		# absolute valence. "delta" is the Tier-C potential-function arm.
		if emo_signal not in EMO_SIGNALS:
			raise ValueError(f"emo_signal must be one of {EMO_SIGNALS}, got {emo_signal!r}")
		self.emo_signal = emo_signal
		# Which mined w(e) table the valence functional reads. DEFAULT "soft" resolves to
		# EMOTION_VALENCE_MINED itself -- the same dict object as before, so nu() is unchanged.
		if emo_valence_table not in EMOTION_VALENCE_TABLES:
			raise ValueError(f"emo_valence_table must be one of {EMO_VALENCE_TABLES}, got {emo_valence_table!r}")
		self.emo_valence_table = emo_valence_table
		self.valence_weights = EMOTION_VALENCE_TABLES[emo_valence_table]
		# Parallel value table, same shape as self.Q. Initialised lazily in _init_node.
		self.Q_emo: dict = {}
		# Welford sum-of-squared-deviations for the SAME edges as Q_emo. sigma_emo is
		# derived from this and Nsa at read time; it is never stored.
		self.M2_emo: dict = {}
		# Every individual z backed up into each edge, in visit order. Q_emo/M2_emo are the
		# O(1) summaries search actually reads; this is the raw tape the frozen subtree log
		# ships as per_step_valences, for the W3 n-step and variance analyses. Write-only --
		# nothing in selection or backup reads it.
		self.emo_valences: dict = {}

	def _emotion_quality(self, dist) -> float:
		"""E[valence] under a predicted emotion distribution. Bounded in [-1, +1];
		returns 0.0 when no distribution is attached (e.g. SYS turns / placeholders)."""
		if not dist:
			return 0.0
		return sum(p * self.valence_weights.get(e, 0.0) for e, p in dist.items())

	@staticmethod
	def _attached_distribution(state):
		"""The emotion distribution on ``state``'s last turn, or None if there is none.

		Returns None rather than raising for an empty history / a turn that was never
		classified, so the delta signal can apply the nu(empty) = 0 convention.
		"""
		if state is None or not state.history:
			return None
		return state.predicted_distribution()

	def _emotion_signal(self, parent_state, next_state) -> float:
		"""The z backed up into Q_emo for the edge that led to ``next_state``.

		``level`` (default): z = nu(d_s')                       in [-1, +1]
		``delta``:           z = (nu(d_s') - nu(d_parent)) / 2  in [-1, +1]

		The /2 is the renormalisation from (a) in the spec: nu is in [-1, +1] so the raw
		difference spans [-2, +2], and without rescaling beta_emo would mean a different
		thing in the two arms.

		Missing parent distribution -> z = 0.0, matching the existing nu(empty) = 0
		convention. It deliberately does NOT fall back to the level value: that would
		silently mix the two signals within one run.
		"""
		z_level = self._emotion_quality(self._attached_distribution(next_state))
		if self.emo_signal == "level":
			return z_level
		parent_dist = self._attached_distribution(parent_state)
		if not parent_dist:
			return 0.0
		return (z_level - self._emotion_quality(parent_dist)) / EMO_DELTA_RANGE_SCALE

	def _init_node(self, state):
		# Parent does Q / Nsa / P / valid_moves / realizations / topk-prune. Wrap to
		# ALSO init Q_emo on the (possibly-pruned) valid moves so the action sets
		# stay in lock-step between the two Q channels.
		v = super()._init_node(state)
		hashable_state = self._to_string_rep(state)
		self.Q_emo[hashable_state] = {a: 0.0 for a in self.valid_moves[hashable_state]}
		self.M2_emo[hashable_state] = {a: 0.0 for a in self.valid_moves[hashable_state]}
		self.emo_valences[hashable_state] = {a: [] for a in self.valid_moves[hashable_state]}
		return v

	def var_emo(self, hashable_state: str, action: int) -> float:
		"""Sample variance (ddof=1) of the z values backed up into Q_emo[s][a].
		0.0 below two visits -- never a division by zero."""
		m2 = self.M2_emo.get(hashable_state, {}).get(action, 0.0)
		n = self.Nsa.get(hashable_state, {}).get(action, 0)
		return welford_variance(n, m2)

	def sigma_emo(self, hashable_state: str, action: int) -> float:
		"""sqrt(var_emo). Derived at read time; nothing stores sigma."""
		return math.sqrt(max(0.0, self.var_emo(hashable_state, action)))

	def _update_emo_channel(self, hashable_state: str, action: int, z: float, n_old: int) -> None:
		"""Welford backup of the emotion channel. ``n_old`` is Nsa BEFORE the increment,
		exactly the count the pre-change running mean used, so the mean is unchanged."""
		mean, m2 = welford_update(
			n_old,
			self.Q_emo[hashable_state][action],
			self.M2_emo.setdefault(hashable_state, {}).get(action, 0.0),
			z,
		)
		self.Q_emo[hashable_state][action] = mean
		self.M2_emo[hashable_state][action] = m2
		self.emo_valences.setdefault(hashable_state, {}).setdefault(action, []).append(float(z))

	def _calculate_uct(self, hashable_state: str, action: int) -> float:
		Ns = self.Ns[hashable_state] or 1e-8
		explore = math.sqrt(Ns) / (1 + self.Nsa[hashable_state][action])
		q_emo = self.Q_emo.get(hashable_state, {}).get(action, 0.0)
		# Risk-adjusted emotion channel. At the default emo_risk_lambda = 0.0 this is
		# bit-identical to q_emo (IEEE-754: 0.0 * finite = 0.0, and x - 0.0 == x), so the
		# pre-change score is recovered without a branch or an epsilon.
		q_emo_adj = q_emo - self.emo_risk_lambda * self.sigma_emo(hashable_state, action)
		return (
			self.Q[hashable_state][action]
			+ self.beta_emo * q_emo_adj
			+ self.configs.cpuct * self.P[hashable_state][action] * explore
		)

	def search(self, state):
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
		# from s," which is what we want for selection bias. `state` here is the
		# sampled parent realization, which is what --emo_signal delta differences
		# against. The Welford step also accumulates M2_emo for sigma_emo.
		emo_v = self._emotion_signal(state, next_state)
		self._update_emo_channel(hashable_state, best_action, emo_v, nsa_old)

		# Increment counters AFTER both updates so they share the same old Nsa.
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# Realization V tracker keeps the donation v for inference-time NLG choice
		# (utterance pick still optimises donation; Q_emo only shapes search).
		self._update_realizations_Vs(next_state, v)
		return v
