import numpy as np
import logging
import math

from utils.sessions import DialogSession
from games import DialogGame
from players import DialogPlanner


logger = logging.getLogger(__name__)

# --search_horizon choices: which get_dialog_ended values stop a simulated branch.
#   legacy  -- only success (+1.0; > 0 in the closed-loop MCTS) is terminal. This is GDP-Zero's rule
#              and the default. It ignores the game's -1.0 at the turn limit and on a verbatim stall,
#              so search keeps expanding and valuing states past --max_turns that no real episode can
#              reach (analysis/phase1/SEARCH_HORIZON_BUG.md).
#   episode -- any non-zero get_dialog_ended is terminal and its value is backed up: search stops
#              exactly where rollout.py's episode loop stops, so it only visits reachable states.
SEARCH_HORIZONS = ("legacy", "episode")


class MCTS:
	# Optional bookkeeping tables, declared here so EVERY planner exposes the whole
	# subtree-log schema (runners/_common.build_subtree_records) as plain attributes.
	# Only the subclasses named below ever write to them, and they write by assigning
	# an instance attribute in __init__ -- these class-level dicts are read-only
	# placeholders that stay empty on the planners that have no such channel.
	#   realizations / realizations_Vs / realizations_Ns / cache_hits -> OpenLoopMCTS
	#   Q_emo / M2_emo / emo_valences / root_decision                 -> EmotionAwareMultiObjectiveQ
	realizations: dict = {}
	realizations_Vs: dict = {}
	realizations_Ns: dict = {}
	cache_hits: dict = {}
	Q_emo: dict = {}
	M2_emo: dict = {}
	emo_valences: dict = {}
	root_decision: dict = {}  # Constrain's root decision, filled by EmotionAwareMultiObjectiveQ.get_action_prob
	# True on the open-loop variants, where a node is a dialog-act prefix with a pool of
	# concrete realizations behind it. The subtree log records action_seq / depth only for
	# those, so it asks this instead of sniffing for a `realizations` attribute.
	is_open_loop: bool = False

	def __init__(self, game:DialogGame, player:DialogPlanner, configs) -> None:
		self.game = game
		self.player = player
		self.configs = configs
		# configs is a dotdict whose attribute access raises on a missing key; runners that
		# predate the flag build configs without it and keep the legacy rule.
		self.search_horizon = configs.get("search_horizon", "legacy") if hasattr(configs, "get") else "legacy"
		if self.search_horizon not in SEARCH_HORIZONS:
			raise ValueError(f"search_horizon must be one of {SEARCH_HORIZONS}, got {self.search_horizon!r}")
		# U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
		self.Ns: dict = {}  # saves compute, total visit count at that node.
		self.Nsa: dict = {}  # visit count for each action at that node.
		self.Q: dict = {}  # running mean return for each action.
		self.P: dict = {}  # prior policy from player.predict, masked to valid moves and renormalized.
		# utility
		self.valid_moves: dict = {}  # indices of legal dialog acts at the node.
		self.terminals: dict = {}  # cached terminal-value lookup (used by base MCTS; OpenLoopMCTS re-checks every visit because terminality depends on the concrete realization, not the DA prefix).
		# debugging / more information
		self.Vs: dict = {}  # leaf values, kept only for debugging/inspection.
		# Same leaf values as self.Vs, but keyed by the TREE key (the DA prefix for the
		# open-loop variants) instead of the full string rep. self.Vs' key space cannot be
		# joined against Nsa/Q, so the frozen subtree log reads leaf_value from here.
		# Write-only bookkeeping: nothing in selection or backup reads it.
		self.node_V: dict = {}
		return

	def _ends_search(self, terminated_v) -> bool:
		"""Open-loop terminal test on a get_dialog_ended value (see SEARCH_HORIZONS).
		Under "legacy" this is exactly the pre-flag `terminated_v == 1.0`."""
		if self.search_horizon == "episode":
			return terminated_v != 0.0
		return terminated_v == 1.0

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		return state.to_string_rep(keep_sys_da=True, keep_user_da=True, max_turn_to_display=-1)

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}

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
		return v

	def search(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		
		is_leaf_node = False
		v = 0.0
		if hashable_state not in self.terminals:
			# selected leaf node, expand
			self.terminals[hashable_state] = self.game.get_dialog_ended(state)
			v = self._init_node(state)
			is_leaf_node = True
		# if this leaf node is terminal, return the value
		# closed loop: legacy rule is "> 0"; --search_horizon episode stops on any non-zero value
		if (self.terminals[hashable_state] != 0 if self.search_horizon == "episode" else self.terminals[hashable_state] > 0):
			# terminal node
			logger.debug("ended")
			return self.terminals[hashable_state]
		# otherwise, return v
		if is_leaf_node:
			return v
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition
		next_state = self.game.get_next_state(state, best_action)

		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1
		
		# now we are single player, hence just v instead of -v
		return v

	def get_action_prob(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.Ns:
			# selected leaf node, expand
			logging.warn("querying a state that has not been visited")
			self._init_node(state)
		# get the counts for all moves
		# convert to prob
		prob = np.zeros(self.player.get_valid_moves(state).shape)
		for a in self.valid_moves[hashable_state]:
			prob[a] = self.Nsa[hashable_state][a]
		prob /= prob.sum()
		return prob


class OpenLoopMCTS(MCTS):
	is_open_loop = True

	def __init__(self, game, player, configs) -> None:
		super().__init__(game, player, configs)
		# Pool of concrete dialog trajectories that all collapse to this DA-prefix node.
		# Sampled from during selection so the tree explores the stochasticity of NLG without exploding. Capped at max_realizations.
		self.realizations: dict = {}  # state -> list of real DialogSessions

		# Running mean MCTS-backed value of each distinct last system utterance observed at this node.
		# Used at inference time to pick the best concrete utterance to actually say.
		self.realizations_Vs: dict = {}  # state -> {realization: V(realization)}

		# Visit counts that go with realizations_Vs so the means in realizations_Vs can be updated incrementally.
		self.realizations_Ns: dict = {}  # state -> {realization: N(realization)}
		self.max_realizations = configs.max_realizations

		# node -> [served_from_cache, freshly_generated] transitions INTO that node.
		# The frozen subtree log reports cache_hit = served_from_cache > 0. Write-only.
		self.cache_hits: dict = {}

		# Cache fixes (analysis/phase1/p_depth.md). Both DEFAULT off, which is the frozen planner.
		#   cache_ended_children: file a generated reply that ends the search under its node at generation
		#     time. Otherwise only a non-terminal re-entry adds to the pool (search returns before it on a
		#     terminal state), so the cache never holds, and never serves, a reply that ends the episode.
		#   cache_fresh_depth1: never serve the search root's outgoing edges from the cache. The root pool
		#     holds one state, so a cached depth-1 edge replays the same R replies on every visit; with this
		#     on each visit generates, and the depth-1 child's pool keeps every reply instead of R.
		has_get = hasattr(configs, "get")
		self.cache_ended_children = bool(configs.get("cache_ended_children", False)) if has_get else False
		self.cache_fresh_depth1 = bool(configs.get("cache_fresh_depth1", False)) if has_get else False
		# node -> generated replies that end the search there. Served from the cache alongside
		# self.realizations, never sampled as a parent: nothing is searched past them.
		self.ended_realizations: dict = {}
		# depth-1 nodes whose pool is not capped at max_realizations (cache_fresh_depth1)
		self._uncapped_pools: set = set()
		# tree key of the first search() call, i.e. the search root (a planner is built per turn)
		self._sim_root_key = None
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		das = []
		for (speaker, da, _) in state:
			if speaker == state.SYS:
				das.append(da)
		return "__".join(das)

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		self.realizations[hashable_state] = [state.copy()]

		prior, v = self.player.predict(state)
		self.Vs[st := state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.node_V[hashable_state] = v  # same value, keyed for the subtree log
		logger.debug(f"State: {st}\n\n")
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])

		# Hard-prune to the LLM's top-K when --llm_prior_topk is set. After this,
		# valid_moves / Nsa / Q only contain the K highest-prior actions; dropped
		# actions are unreachable for this node. Eliminates the PUCT round-robin
		# waste on actions the LLM said were bad. See debug.md "llm_prior_topk
		# should do hard pruning".
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
				# safety: shouldn't happen since kept has at least one >floor action,
				# but if it does, fall back to uniform over kept.
				for a in kept:
					self.P[hashable_state][a] = 1.0 / len(kept)
		return v

	def _sample_realization(self, hashable_state):
		rand_i = np.random.randint(len(self.realizations[hashable_state]))
		return self.realizations[hashable_state][rand_i]

	def _add_new_realizations(self, state):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = []
		if state in self.realizations[hashable_state]:
			return
		
		self.realizations[hashable_state].append(state.copy())
		if len(self.realizations[hashable_state]) > self.max_realizations and hashable_state not in self._uncapped_pools:
			# should never happen
			logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
			self.realizations[hashable_state].pop(0)
		return

	def _record_cache(self, node: str, hit: bool):
		counts = self.cache_hits.setdefault(node, [0, 0])
		counts[0 if hit else 1] += 1

	def _cached_children(self, prefetch_state) -> list:
		"""Every reply the cache can serve for this child node: its pool, plus the replies that ended
		the search there (always empty unless --cache_ended_children)."""
		return self.realizations.get(prefetch_state, []) + self.ended_realizations.get(prefetch_state, [])

	def _serves_from_cache(self, parent_key, prefetch_state) -> bool:
		"""The edge parent_key -> prefetch_state is served from the cache instead of generating.
		With both cache flags off this is the frozen rule: the child's pool holds max_realizations."""
		if self.cache_fresh_depth1 and parent_key == self._sim_root_key:
			return False
		return len(self._cached_children(prefetch_state)) >= self.max_realizations

	def _file_generated_child(self, parent_key, prefetch_state, next_state):
		"""Bookkeeping for a freshly generated reply, before search descends into it."""
		if self.cache_fresh_depth1 and parent_key == self._sim_root_key:
			self._uncapped_pools.add(prefetch_state)
		# Non-terminal replies reach the pool through _init_node / _add_new_realizations when search
		# enters them; a terminal one returns before that, so it is filed here or never.
		if self.cache_ended_children and self._ends_search(self.game.get_dialog_ended(next_state)):
			ended = self.ended_realizations.setdefault(prefetch_state, [])
			if next_state not in ended:
				ended.append(next_state.copy())

	def _get_next_state(self, state, best_action):
		parent_key = self._to_string_rep(state)
		prefetch_state = parent_key + "__" + self.player.dialog_acts[best_action]
		if self._serves_from_cache(parent_key, prefetch_state):
			# use a cached realization
			self._record_cache(prefetch_state, True)
			children = self._cached_children(prefetch_state)
			return children[np.random.randint(len(children))]

		# otherwise, generate a new realization
		self._record_cache(prefetch_state, False)
		next_state = self.game.get_next_state(state, best_action)
		self._file_generated_child(parent_key, prefetch_state, next_state)
		return next_state

	def _update_realizations_Vs(self, state: DialogSession, v: float):
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
		self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / self.realizations_Ns[hashable_state][sys_utt]
		return

	def search(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		if self._sim_root_key is None:
			self._sim_root_key = hashable_state

		# check everytime since state is stochastic, does not map to hashable_state
		terminated_v = self.game.get_dialog_ended(state)
		# check if it is terminal node (--search_horizon decides whether failure ends the branch)
		if self._ends_search(terminated_v):
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
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			# a variant of PUCT
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition. For open loop, first sample from an existing realization
		state = self._sample_realization(hashable_state)
		next_state = self._get_next_state(state, best_action)
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# ------- backfilling ---------

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# update v to realizations for NLG at inference
		self._update_realizations_Vs(next_state, v)
		# now we are single player, hence just v instead of -v
		return v
	
	def get_best_realization(self, state:DialogSession, action: int):
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


class OpenLoopMCTSParallel(OpenLoopMCTS):
	def __init__(self, game, player, configs) -> None:
		super().__init__(game, player, configs)

	def _populate_next_realizations(self, state, next_action, num_to_add):
		next_states = self.game.get_next_state_batched(state, next_action, batch=num_to_add)
		for next_state in next_states:
			self._add_new_realizations(next_state)
		return

	def _get_next_state(self, state, best_action):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			self._record_cache(prefetch_state, True)
			return self._sample_realization(prefetch_state)

		self._record_cache(prefetch_state, False)
		self._populate_next_realizations(state, best_action, self.max_realizations)
		return self._sample_realization(prefetch_state)
	
	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		# should have been initialized during _get_next_state, except for the root node
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = [state.copy()]

		# TODO: batch predict value function
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
		return v