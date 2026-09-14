import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

POLICY_PRIOR = "policy_prior"
VALUE_ESTIMATOR = "value_estimator"
USER_SIMULATOR = "user_simulator"
SYSTEM_UTTERANCE = "system_utterance"
EMOTION_CLASSIFIER = "emotion_classifier"
UNATTRIBUTED = "unattributed"  # a call made outside any `with role(...)` block

ROLES = [POLICY_PRIOR, VALUE_ESTIMATOR, USER_SIMULATOR, SYSTEM_UTTERANCE, EMOTION_CLASSIFIER, UNATTRIBUTED]

_enabled = False
_current_role: ContextVar = ContextVar("emcts_role", default=UNATTRIBUTED)
_lock = Lock()
_stats: dict = {}
_turns = 0


def enable():
    """Start recording. Call once, before the run."""
    global _enabled
    _enabled = True
    reset()


def is_enabled():
    return _enabled


def reset():
    global _turns
    with _lock:
        _stats.clear()
        _turns = 0


@contextmanager
def role(name):
    """Bill every generation call made inside this block to ``name``."""
    if not _enabled:
        yield
        return
    token = _current_role.set(name)
    try:
        yield
    finally:
        _current_role.reset(token)


def current_role():
    """The active role. ContextVars do not cross thread boundaries, so a method that fans work
    out to a pool reads this first and re-enters it inside each worker."""
    return _current_role.get()


def mark_turn():
    """One dialogue turn finished — the denominator for calls/turn."""
    global _turns
    if not _enabled:
        return
    with _lock:
        _turns += 1


@contextmanager
def timed(name, tokens_in=0, tokens_out=0):
    """Time a block and bill it to ``name``. For roles that are not LLM calls (the HF emotion
    classifier), where nothing downstream calls record() for us."""
    if not _enabled:
        yield
        return
    t0 = time.monotonic()
    try:
        yield
    finally:
        with role(name):
            record(time.monotonic() - t0, tokens_in=tokens_in, tokens_out=tokens_out)


def record(seconds, tokens_in=0, tokens_out=0, samples=1):
    """Report one generation call. Called by the generation model, billed to the active role."""
    if not _enabled:
        return
    name = _current_role.get()
    with _lock:
        st = _stats.setdefault(name, {"calls": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0, "samples": 0})
        st["calls"] += 1
        st["seconds"] += seconds
        st["tokens_in"] += int(tokens_in or 0)
        st["tokens_out"] += int(tokens_out or 0)
        st["samples"] += int(samples or 1)


def summary():
    """{role: {calls, calls_per_turn, mean_latency_s, mean_tokens_in, mean_tokens_out, ...}}"""
    with _lock:
        raw = {k: dict(v) for k, v in _stats.items()}
        turns = _turns
    out = {}
    for name in ROLES:
        st = raw.get(name)
        if not st:
            continue
        calls = max(st["calls"], 1)
        out[name] = {
            "calls": st["calls"],
            "calls_per_turn": st["calls"] / turns if turns else None,
            "mean_latency_s": st["seconds"] / calls,
            "total_seconds": st["seconds"],
            "mean_tokens_in": st["tokens_in"] / calls,
            "mean_tokens_out": st["tokens_out"] / calls,
            "total_tokens_in": st["tokens_in"],
            "total_tokens_out": st["tokens_out"],
            "samples": st["samples"],
        }
    return {"turns": turns, "roles": out}


def report(label="run"):
    """Print the cost table and return the summary dict."""
    s = summary()
    turns = s["turns"]
    print(f"\n=== per-role cost ({label}), {turns} turns ===")
    header = f"{'role':<20}{'calls':>7}{'calls/turn':>12}{'mean lat s':>12}{'tok in':>9}{'tok out':>9}{'total s':>10}"
    print(header)
    print("-" * len(header))
    for name, v in s["roles"].items():
        cpt = f"{v['calls_per_turn']:.1f}" if v["calls_per_turn"] is not None else "-"
        print(f"{name:<20}{v['calls']:>7}{cpt:>12}{v['mean_latency_s']:>12.2f}"
              f"{v['mean_tokens_in']:>9.0f}{v['mean_tokens_out']:>9.0f}{v['total_seconds']:>10.1f}")
    total_s = sum(v["total_seconds"] for v in s["roles"].values())
    total_calls = sum(v["calls"] for v in s["roles"].values())
    print("-" * len(header))
    print(f"{'TOTAL':<20}{total_calls:>7}{'':>12}{'':>12}{'':>9}{'':>9}{total_s:>10.1f}")
    return s


def dump(output_path, label="run"):
    """Write the summary next to a run's output pickle as ``*_role_profile.json``."""
    import os
    path = os.path.splitext(output_path)[0] + "_role_profile.json"
    payload = summary()
    payload["label"] = label
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved per-role cost profile to {path}")
    return path
