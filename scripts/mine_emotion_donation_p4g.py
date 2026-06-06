from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402

EMOTION_ORDER = [
	Emotions.Happiness, Emotions.Sadness, Emotions.Fear, Emotions.Anger,
	Emotions.Surprise, Emotions.Disgust, Emotions.Contempt, Emotions.Neutral,
]
EMO_KEYS = [str(e) for e in EMOTION_ORDER]


def dialog_succeeded(dialog: dict) -> bool:
	for label in dialog["label"]:
		for tag in label.get("ee", []):
			if tag == "agree-donation":
				return True
	return False


def build_emotion_cache(dialogs: dict, cache_path: str) -> dict[str, str]:
	"""Argmax HF emotion per utterance. On-disk cache."""
	cache: dict[str, str] = {}
	if os.path.exists(cache_path):
		with open(cache_path) as f:
			cache = json.load(f)
	n_new = 0
	all_utts: list[str] = []
	for d in dialogs.values():
		for turn in d["dialog"]:
			for utt in turn.get("ee", []):
				utt = (utt or "").strip()
				if utt and utt not in cache:
					all_utts.append(utt)
	if all_utts:
		from emotion_classifiers.hf_emotion import HFEmotionClassifier
		clf = HFEmotionClassifier()
		for utt in all_utts:
			dist = clf.predict_distribution_from_utterance(utt)
			emo = max(dist, key=dist.get)
			cache[utt] = str(emo)
			n_new += 1
		Path(os.path.dirname(cache_path) or ".").mkdir(parents=True, exist_ok=True)
		with open(cache_path, "w") as f:
			json.dump(cache, f)
	print(f"emotion cache: {len(cache)} utterances ({n_new} new)")
	return cache


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
	import math
	if n == 0:
		return (0.0, 0.0)
	p = k / n
	denom = 1 + z * z / n
	center = (p + z * z / (2 * n)) / denom
	margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
	return (max(0.0, center - margin), min(1.0, center + margin))


def main():
	parser = argparse.ArgumentParser(description=__doc__,
	                                 formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--data", default="data/p4g/300_dialog_turn_based.pkl")
	parser.add_argument("--cache", default="outputs/emotion_donation_emocache.json")
	parser.add_argument("--out", default="outputs/emotion_donation_analysis.json")
	parser.add_argument("--min_obs", type=int, default=5,
	                    help="hide cells with fewer than min_obs samples from the reported table")
	args = parser.parse_args()
	os.chdir(REPO_ROOT)

	with open(args.data, "rb") as f:
		dialogs: dict = pickle.load(f)
	print(f"loaded {len(dialogs)} dialogs")

	emo_cache = build_emotion_cache(dialogs, args.cache)
	base_donation_rate = sum(dialog_succeeded(d) for d in dialogs.values()) / len(dialogs)
	print(f"base donation rate (300 dialogs): {base_donation_rate:.3f}\n")

	# --- Per-utterance: P(donation | this turn's user emotion = e) ---
	# Aggregate: count how many user turns had argmax emotion e in *successful* vs *failed* dialogs.
	utt_succ = {e: 0 for e in EMO_KEYS}
	utt_total = {e: 0 for e in EMO_KEYS}

	# --- Per-dialog: P(donation | dominant user emotion in dialog = e) ---
	# Dominant = most-frequent argmax across user turns in that dialog.
	dom_succ = {e: 0 for e in EMO_KEYS}
	dom_total = {e: 0 for e in EMO_KEYS}

	# --- Transitions: for each consecutive (prev_user_turn, next_user_turn) ---
	trans_succ: dict = defaultdict(int)   # (prev_emo, next_emo) -> count successful
	trans_total: dict = defaultdict(int)

	# --- First-vs-last user-turn emotion shift (dialog-level) ---
	shift_succ: dict = defaultdict(int)
	shift_total: dict = defaultdict(int)

	for did, d in dialogs.items():
		succ = dialog_succeeded(d)
		user_turns: list[str] = []  # argmax emotion per user turn, in order
		for turn in d["dialog"]:
			for utt in turn.get("ee", []):
				utt = (utt or "").strip()
				if not utt:
					continue
				emo = emo_cache.get(utt)
				if emo is None:
					continue
				user_turns.append(emo)
				utt_total[emo] += 1
				if succ:
					utt_succ[emo] += 1

		if not user_turns:
			continue

		# dominant emotion of this dialog
		from collections import Counter
		dom = Counter(user_turns).most_common(1)[0][0]
		dom_total[dom] += 1
		if succ:
			dom_succ[dom] += 1

		# transitions between consecutive user turns
		for a, b in zip(user_turns, user_turns[1:]):
			trans_total[(a, b)] += 1
			if succ:
				trans_succ[(a, b)] += 1

		# first -> last shift
		shift_total[(user_turns[0], user_turns[-1])] += 1
		if succ:
			shift_succ[(user_turns[0], user_turns[-1])] += 1

	print("=" * 78)
	print("Q1) Donation rate by USER emotion")
	print("=" * 78)
	print(f"\nBase rate: {base_donation_rate:.3f}\n")

	print("(a) Per-USER-TURN donation rate (turn's emotion -> outcome of its dialog):")
	print(f"  {'emotion':>10}  {'n_turns':>8}  {'P(donate)':>10}  {'lift':>7}  {'95% Wilson CI':>17}")
	per_turn_rows = []
	for e in EMO_KEYS:
		n = utt_total[e]
		if n == 0:
			continue
		p = utt_succ[e] / n
		lo, hi = wilson_ci(utt_succ[e], n)
		print(f"  {e:>10}  {n:>8}  {p:>10.3f}  {p - base_donation_rate:>+7.3f}  [{lo:.3f}, {hi:.3f}]")
		per_turn_rows.append({"emotion": e, "n_turns": n, "p_donate": p,
		                      "lift": p - base_donation_rate, "ci": [lo, hi]})

	print("\n(b) Per-DIALOG donation rate (dialog's dominant user emotion):")
	print(f"  {'dominant':>10}  {'n_dialogs':>9}  {'P(donate)':>10}  {'lift':>7}  {'95% Wilson CI':>17}")
	per_dialog_rows = []
	for e in EMO_KEYS:
		n = dom_total[e]
		if n == 0:
			continue
		p = dom_succ[e] / n
		lo, hi = wilson_ci(dom_succ[e], n)
		print(f"  {e:>10}  {n:>9}  {p:>10.3f}  {p - base_donation_rate:>+7.3f}  [{lo:.3f}, {hi:.3f}]")
		per_dialog_rows.append({"emotion": e, "n_dialogs": n, "p_donate": p,
		                        "lift": p - base_donation_rate, "ci": [lo, hi]})

	print("\n" + "=" * 78)
	print("Q2) Donation rate by USER-EMOTION TRANSITION (consecutive turns)")
	print("=" * 78)
	print(f"\n  rows = prev_emo, cols = next_emo. Cells = P(donate). "
	      f"Hidden if n < {args.min_obs}.")
	print(f"  Base rate = {base_donation_rate:.3f}.\n")

	header = "             " + " ".join(f"{e[:6]:>7}" for e in EMO_KEYS)
	print(header)
	for prev in EMO_KEYS:
		cells = []
		for nxt in EMO_KEYS:
			n = trans_total[(prev, nxt)]
			if n < args.min_obs:
				cells.append(f"{'.':>7}")
			else:
				p = trans_succ[(prev, nxt)] / n
				cells.append(f"{p:>7.2f}")
		print(f"  {prev:>10} " + " ".join(cells))

	print("\n  Counts (n):")
	print(header)
	for prev in EMO_KEYS:
		cells = []
		for nxt in EMO_KEYS:
			n = trans_total[(prev, nxt)]
			cells.append(f"{n:>7d}" if n > 0 else f"{'.':>7}")
		print(f"  {prev:>10} " + " ".join(cells))

	# Pull out transitions with biggest positive lift (n >= min_obs)
	print(f"\n  Top transitions with positive lift (n >= {args.min_obs}, ranked by lift):")
	ranked = []
	for (prev, nxt), n in trans_total.items():
		if n < args.min_obs:
			continue
		p = trans_succ[(prev, nxt)] / n
		ranked.append((p - base_donation_rate, p, n, prev, nxt))
	ranked.sort(reverse=True)
	print(f"    {'prev':>10} -> {'next':>10}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}")
	for lift, p, n, prev, nxt in ranked[:10]:
		print(f"    {prev:>10} -> {nxt:>10}  {n:>4d}  {p:>10.3f}  {lift:>+7.3f}")
	print(f"\n  Bottom transitions (negative lift, n >= {args.min_obs}):")
	print(f"    {'prev':>10} -> {'next':>10}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}")
	for lift, p, n, prev, nxt in ranked[-5:]:
		print(f"    {prev:>10} -> {nxt:>10}  {n:>4d}  {p:>10.3f}  {lift:>+7.3f}")

	print("\n" + "=" * 78)
	print("Specific test of the 'persuasion needs sympathy' hypothesis")
	print("=" * 78)

	NEG = {"sadness", "fear", "anger", "disgust", "contempt"}
	POS = {"happiness", "surprise"}

	def agg(filter_fn):
		s = sum(trans_succ[k] for k in trans_total if filter_fn(*k))
		n = sum(trans_total[k] for k in trans_total if filter_fn(*k))
		return (s, n)

	cases = [
		("happy -> negative",   lambda a, b: a == "happiness" and b in NEG),
		("happy -> happy",      lambda a, b: a == "happiness" and b == "happiness"),
		("neutral -> negative", lambda a, b: a == "neutral" and b in NEG),
		("neutral -> happy",    lambda a, b: a == "neutral" and b == "happiness"),
		("negative -> happy",   lambda a, b: a in NEG and b == "happiness"),
		("negative -> negative",lambda a, b: a in NEG and b in NEG),
	]
	print(f"\n  {'transition class':>22}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}")
	for name, fn in cases:
		s, n = agg(fn)
		if n == 0:
			print(f"  {name:>22}  {n:>4d}  {'(none)':>10}")
			continue
		p = s / n
		print(f"  {name:>22}  {n:>4d}  {p:>10.3f}  {p - base_donation_rate:>+7.3f}")

	# First-to-last shift
	print(f"\n  First-user-turn -> last-user-turn shift (n >= {args.min_obs}):")
	shift_ranked = []
	for (a, b), n in shift_total.items():
		if n < args.min_obs:
			continue
		p = shift_succ[(a, b)] / n
		shift_ranked.append((p - base_donation_rate, p, n, a, b))
	shift_ranked.sort(reverse=True)
	print(f"    {'first':>10} -> {'last':>10}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}")
	for lift, p, n, a, b in shift_ranked[:10]:
		print(f"    {a:>10} -> {b:>10}  {n:>4d}  {p:>10.3f}  {lift:>+7.3f}")

	out = {
		"base_donation_rate": base_donation_rate,
		"n_dialogs": len(dialogs),
		"per_turn_emotion": per_turn_rows,
		"per_dialog_dominant_emotion": per_dialog_rows,
		"transition_table_p_donate": {
			f"{a}->{b}": (trans_succ[(a, b)] / trans_total[(a, b)]) if trans_total[(a, b)] else None
			for a in EMO_KEYS for b in EMO_KEYS
		},
		"transition_table_n": {
			f"{a}->{b}": trans_total[(a, b)] for a in EMO_KEYS for b in EMO_KEYS
		},
		"hypothesis_tests": {
			name: {
				"n": sum(trans_total[k] for k in trans_total if fn(*k)),
				"p_donate": (
					sum(trans_succ[k] for k in trans_total if fn(*k))
					/ max(1, sum(trans_total[k] for k in trans_total if fn(*k)))
				),
			} for name, fn in cases
		},
	}
	Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(out, f, indent=2)
	print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
	main()
