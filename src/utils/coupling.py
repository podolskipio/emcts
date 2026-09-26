"""Coupled seeds (--coupled_seeds): common random numbers across arms, so two arms that make the same
choice receive the same LLM reply and the same tree draws, and their difference is method, not luck.

Why a reply store and not a server seed. SGLang honours a per-request seed only under
--enable-deterministic-inference, and on this GPU (RTX 4090, Ada) that mode does not start: its
batch-invariant matmul needs 106,496 B of shared memory against a 101,376 B limit (readiness phase3.md).
Even where it runs, a seed alone would not remove batch-dependent numerics. So coupling is done on the
client: every LLM call inside a coupled dialogue is keyed by

    (--seed, dialogue id, the exact request: messages + sampling parameters, occurrence index k)

where k counts how many times this dialogue has already sent that exact request. The first arm to make a
call stores the reply; every later call with the same key -- the same arm re-run, or another arm that
reached the same prompt -- gets that reply back. Each distinct key is still one independent draw from the
model, so each arm's own distribution is unchanged; only the draws are shared.

Tree draws (realization sampling, cache draws) come from a RandomState per (seed, dialogue, turn) instead
of the process-wide numpy stream that --num_workers dialogues interleave on.

Off by default: with no active dialogue context every function here is a pass-through.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sqlite3
import threading

import numpy as np

_CTX: contextvars.ContextVar = contextvars.ContextVar("coupling_ctx", default=None)


class ReplyStore:
	"""sqlite key -> JSON reply. Safe for the runner's worker threads and for two runs (arms) sharing one
	file concurrently: a key is written once (INSERT OR IGNORE) and every caller returns what is stored."""

	def __init__(self, path: str):
		os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
		self.path = path
		self._lock = threading.Lock()
		self._conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
		with self._lock:
			self._conn.execute("PRAGMA journal_mode=WAL")
			self._conn.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
			self._conn.commit()

	def get(self, key: str):
		with self._lock:
			row = self._conn.execute("SELECT value FROM replies WHERE key = ?", (key,)).fetchone()
		return None if row is None else json.loads(row[0])

	def put_if_absent(self, key: str, value) -> None:
		with self._lock:
			self._conn.execute("INSERT OR IGNORE INTO replies (key, value) VALUES (?, ?)", (key, json.dumps(value)))
			self._conn.commit()


class _Dialogue:
	def __init__(self, store: ReplyStore, seed: int, dlg_id: str):
		self.store, self.seed, self.dlg_id = store, int(seed), str(dlg_id)
		self.counts: dict = {}
		self.lock = threading.Lock()
		self.hits = 0
		self.misses = 0


@contextlib.contextmanager
def dialogue(store: "ReplyStore | None", seed, dlg_id):
	"""Couple every LLM call and planner RNG made inside this block (one dialogue, one thread).
	``store=None`` makes it a no-op, so callers need no branch."""
	if store is None:
		yield None
		return
	token = _CTX.set(_Dialogue(store, seed, dlg_id))
	try:
		yield _CTX.get()
	finally:
		_CTX.reset(token)


def current():
	return _CTX.get()


def active() -> bool:
	return _CTX.get() is not None


def _digest(*parts) -> str:
	return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def call(request: dict, generate):
	"""``generate()`` coupled on ``request`` (JSON-able: messages + sampling parameters). Returns what the
	store holds for this call's key, generating and storing it first on a miss. Pass-through when no
	dialogue context is active."""
	ctx = _CTX.get()
	if ctx is None:
		return generate()
	base = _digest(ctx.seed, ctx.dlg_id, request)
	with ctx.lock:
		k = ctx.counts.get(base, 0)
		ctx.counts[base] = k + 1
	key = f"{base}:{k}"
	value = ctx.store.get(key)
	if value is not None:
		ctx.hits += 1
		return value
	try:
		value = generate()
	except BaseException:
		# a failed request (the caller retries) must not use up occurrence k, or this arm's later keys
		# would shift against every other arm's
		with ctx.lock:
			ctx.counts[base] -= 1
		raise
	ctx.misses += 1
	ctx.store.put_if_absent(key, value)
	return ctx.store.get(key)  # the stored one: another arm may have written this key first


def run_in(ctx, fn, *args, **kwargs):
	"""fn(*args, **kwargs) inside ``ctx`` -- for worker threads, which do not inherit context variables."""
	if ctx is None:
		return fn(*args, **kwargs)
	token = _CTX.set(ctx)
	try:
		return fn(*args, **kwargs)
	finally:
		_CTX.reset(token)


def planner_rng(turn: int):
	"""The RandomState for the tree searched at ``turn`` of the active dialogue, or None when uncoupled."""
	ctx = _CTX.get()
	if ctx is None:
		return None
	return np.random.RandomState(int(_digest("planner_rng", ctx.seed, ctx.dlg_id, int(turn))[:8], 16))
