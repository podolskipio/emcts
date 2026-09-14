"""Mine donation weights over the user's *cumulative* emotional state, not the current turn.

The instantaneous variant (``mine_emotion_donation_p4g.py``) scores a single emotion
distribution: ``nu(d_t) = sum_e d_t(e) w(e)``. A user who has been angry for six turns and is
momentarily neutral scores exactly like a user who has been neutral throughout. This script
scores the whole session-so-far instead.

Cumulative emotional state
--------------------------
Each user turn contributes a one-hot emotion; the state at turn t is the decay-weighted,
renormalised history

    c_t = normalize( sum_{i<=t} lam^(t-i) * onehot(e_i) ),     c_t in the 8-simplex

``--lam`` controls the memory, and it is the whole experiment:

    lam = 0    only the current turn survives -> EXACTLY the instantaneous method
    lam = 1    every turn weighted equally    -> running mean over the session
    0 < lam < 1                                 recency-biased memory

So the existing approach is the ``lam = 0`` corner of this model, and sweeping lam measures
directly what conditioning on history buys. Same EMA convention as the cum_dist vector in the
EmotionHistoryPriorMCTS design note, so the two stay comparable.

Scoring
-------
Rather than a marginal per-emotion lift, fit a ridge-regularised **logistic regression** of the
dialog's donation outcome on ``c_t``, pooled over every turn of every dialog:

    P(donate | c) = sigmoid(theta^T c + b)

Coefficients are *partial* effects, so correlated emotions stop stealing each other's credit —
the fix §1.9(a) of EmotionMCTSDoubleQ.md asks for, generalised from a one-turn histogram to a
history. Training on every prefix (not just the final state) matters because the planner has to
score partial histories mid-search.

The score is then expressed in the same units as the instantaneous ``w(e)``, so ``beta_emo``
keeps its meaning and nothing downstream needs rescaling:

    nu(c) = VALENCE_SCALE * (sigmoid(theta^T c + b) - base_rate)

Setting c = onehot(e) recovers a per-emotion table directly comparable to the old w(e).

Model selection
---------------
lam is chosen by grouped k-fold CV — folds split by *dialog*, never by turn, because all turns
of a dialog carry the same label and would otherwise leak across the fold boundary. Reported on
held-out AUC and log-loss, against the lam=0 (instantaneous) baseline.

Caveats
-------
* Correlational, exactly as in the instantaneous script: these say which emotional trajectories
  co-occur with donation, not which ones cause it.
* Emotions come from the argmax cache, so each turn is one-hot rather than a full distribution.
  Using soft distributions is a drop-in change to ``build_history_states`` when the HF model is
  available to re-label; it would mostly smooth c_t, not move the weights.
* Whether a holdout split is needed depends on the eval — see the split discussion in
  ``mine_emotion_donation_p4g.py``. Defaults match it (no holdout, correct for self-play).
* Runs on either corpus via ``--task {p4g,esc}``; ``mining_corpus.py`` owns the per-task
  outcome definitions, and the esc one is internally defined rather than annotated.

Usage
-----
    python src/emotion_mining/mine_emotion_history_donation_p4g.py
    python src/emotion_mining/mine_emotion_history_donation_p4g.py --task esc       # ESConv
    python src/emotion_mining/mine_emotion_history_donation_p4g.py --lam 1.0        # skip the sweep
    python src/emotion_mining/mine_emotion_history_donation_p4g.py --holdout_first 100

Outputs
-------
    outputs/emotion_history_donation_analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Reuse the corpus/split/cache logic so the two mining scripts cannot drift apart.
from emotion_mining.mining_corpus import EMO_KEYS, add_corpus_args, load_sequences  # noqa: E402
from emotion_mining.mine_emotion_donation_p4g import (  # noqa: E402
	VALENCE_SCALE, CONTEMPT_FALLBACK_WEIGHT)

EMO_INDEX = {e: i for i, e in enumerate(EMO_KEYS)}
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


# --------------------------------------------------------------------------------------
# Cumulative emotional state
# --------------------------------------------------------------------------------------

@dataclass
class HistoryStates:
	"""Pooled training matrix: one row per user turn, labelled by its dialog's outcome.

	``groups`` carries the dialog index per row so CV folds can split on dialogs, not turns.
	"""
	features: np.ndarray   # (n_turns, 8) cumulative emotion states, each row on the simplex
	labels: np.ndarray     # (n_turns,) 1 if the row's dialog ended in a donation
	groups: np.ndarray     # (n_turns,) dialog index


def cumulative_states(emotions: list[str], lam: float) -> np.ndarray:
	"""Decay-weighted, renormalised emotion history after each turn. Returns (n_turns, 8).

	Computed with a running accumulator: ``acc <- lam * acc + onehot(e_t)``, so the cost is
	linear in turns rather than quadratic. At lam=0 this leaves acc == onehot(e_t) exactly,
	which is the instantaneous baseline (0**0 ambiguity never arises).
	"""
	states = np.zeros((len(emotions), len(EMO_KEYS)))
	accumulator = np.zeros(len(EMO_KEYS))
	for t, emotion in enumerate(emotions):
		accumulator *= lam
		accumulator[EMO_INDEX[emotion]] += 1.0
		total = accumulator.sum()
		states[t] = accumulator / total if total > 0 else accumulator
	return states


def build_history_states(sequences, lam: float) -> HistoryStates:
	features, labels, groups = [], [], []
	for dialog_idx, seq in enumerate(sequences):
		states = cumulative_states(seq.emotions, lam)
		features.append(states)
		labels.append(np.full(len(states), float(seq.succeeded)))
		groups.append(np.full(len(states), dialog_idx))
	return HistoryStates(np.vstack(features), np.concatenate(labels), np.concatenate(groups))


# --------------------------------------------------------------------------------------
# Ridge logistic regression (IRLS) — numpy only, sklearn is not a dependency here
# --------------------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
	return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 50, tol: float = 1e-8) -> tuple[np.ndarray, float]:
	"""Newton/IRLS fit of ``sigmoid(X @ theta + b)``. Returns ``(theta, b)``.

	Ridge is required, not optional: the features sit on the simplex (they sum to 1), so they
	are exactly collinear with the intercept and the unpenalised Hessian is singular. The
	intercept itself is left unpenalised, as usual.
	"""
	n, d = X.shape
	design = np.hstack([np.ones((n, 1)), X])       # intercept first
	penalty = np.eye(d + 1) * l2
	penalty[0, 0] = 0.0                            # never shrink the intercept
	coefficients = np.zeros(d + 1)

	for _ in range(iters):
		probabilities = sigmoid(design @ coefficients)
		weights = np.clip(probabilities * (1 - probabilities), 1e-6, None)
		gradient = design.T @ (y - probabilities) - penalty @ coefficients
		hessian = (design.T * weights) @ design + penalty
		step = np.linalg.solve(hessian, gradient)
		coefficients += step
		if np.max(np.abs(step)) < tol:
			break

	return coefficients[1:], float(coefficients[0])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
	p = np.clip(p, 1e-12, 1 - 1e-12)
	return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
	"""Rank-based AUC (Mann-Whitney U), averaging ranks over ties."""
	n_pos, n_neg = int(y.sum()), int((1 - y).sum())
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	order = np.argsort(scores, kind="mergesort")
	ranks = np.empty(len(scores), dtype=float)
	ranks[order] = np.arange(1, len(scores) + 1)
	# average ranks within tied score groups
	sorted_scores = scores[order]
	start = 0
	for end in range(1, len(sorted_scores) + 1):
		if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
			ranks[order[start:end]] = ranks[order[start:end]].mean()
			start = end
	return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Grouped cross-validation
# --------------------------------------------------------------------------------------

@dataclass
class CVResult:
	lam: float
	auc: float
	logloss: float

	def __str__(self) -> str:
		return f"lam={self.lam:<5g} AUC={self.auc:.4f}  logloss={self.logloss:.4f}"


def grouped_cv(states: HistoryStates, n_folds: int, l2: float, seed: int) -> tuple[float, float]:
	"""Held-out AUC and log-loss, splitting folds by dialog so no dialog straddles a boundary."""
	dialog_ids = np.unique(states.groups)
	rng = np.random.default_rng(seed)
	shuffled = rng.permutation(dialog_ids)
	folds = np.array_split(shuffled, n_folds)

	out_of_fold = np.zeros(len(states.labels))
	for fold in folds:
		is_test = np.isin(states.groups, fold)
		theta, bias = fit_logistic(states.features[~is_test], states.labels[~is_test], l2=l2)
		out_of_fold[is_test] = sigmoid(states.features[is_test] @ theta + bias)

	return roc_auc(states.labels, out_of_fold), log_loss(states.labels, out_of_fold)


def sweep_lambda(sequences, lambdas: list[float], n_folds: int, l2: float, seed: int) -> list[CVResult]:
	results = []
	for lam in lambdas:
		states = build_history_states(sequences, lam)
		auc, loss = grouped_cv(states, n_folds, l2, seed)
		results.append(CVResult(lam, auc, loss))
		print(f"  {results[-1]}")
	return results


# --------------------------------------------------------------------------------------
# Turning the fitted model into scores in w(e) units
# --------------------------------------------------------------------------------------

def score_state(state: np.ndarray, theta: np.ndarray, bias: float, base_rate: float,
                scale: float = 1.0) -> float:
	"""nu(c) in the same units as the instantaneous w(e): 10 x (predicted lift over base)."""
	return scale * VALENCE_SCALE * float(sigmoid(state @ theta + bias) - base_rate)


def fit_output_scale(states: HistoryStates, theta: np.ndarray, bias: float,
                     base_rate: float, target: float | None) -> tuple[float, np.ndarray]:
	"""Rescale factor putting the 99th percentile of |nu(c)| at ``target``, and the raw scores.

	``EmotionAware*._emotion_quality`` documents its output as bounded in [-1, +1], and the
	instantaneous weights peak at 0.62. A logistic model is far more confident at the edges of
	the simplex than a marginal lift is, so raw nu(c) here reaches ~3.6 — pasting that in
	unscaled would multiply the emotion channel's influence without touching beta_emo. The
	rescale is a positive constant, so it preserves the ranking of states and only restores the
	magnitude contract. p99 rather than max, so a single outlier state cannot set the scale.

	``target=None`` disables rescaling (factor 1.0).
	"""
	raw = VALENCE_SCALE * (sigmoid(states.features @ theta + bias) - base_rate)
	if target is None:
		return 1.0, raw
	p99 = float(np.percentile(np.abs(raw), 99))
	return (target / p99 if p99 > 0 else 1.0), raw


def pure_emotion_weights(theta: np.ndarray, bias: float, base_rate: float,
                         scale: float) -> dict[str, float]:
	"""Score of a history consisting entirely of one emotion — directly comparable to old w(e)."""
	weights = {}
	for emotion, index in EMO_INDEX.items():
		onehot = np.zeros(len(EMO_KEYS))
		onehot[index] = 1.0
		weights[emotion] = round(score_state(onehot, theta, bias, base_rate, scale), 2)
	weights["contempt"] = CONTEMPT_FALLBACK_WEIGHT
	return weights


# Illustrative trajectories, to show that the score depends on the history and not just on
# where the user is right now. Each is a (name, {emotion: share}) mixture.
EXAMPLE_HISTORIES = [
	("all neutral",              {"neutral": 1.0}),
	("all happy",                {"happiness": 1.0}),
	("half neutral/half sad",    {"neutral": 0.5, "sadness": 0.5}),
	("half neutral/half happy",  {"neutral": 0.5, "happiness": 0.5}),
	("sustained anger",          {"anger": 0.7, "neutral": 0.3}),
	("recovered: sad -> happy",  {"sadness": 0.4, "happiness": 0.6}),
]


def mixture_state(shares: dict[str, float]) -> np.ndarray:
	state = np.zeros(len(EMO_KEYS))
	for emotion, share in shares.items():
		state[EMO_INDEX[emotion]] = share
	return state / state.sum()


# --------------------------------------------------------------------------------------

def parse_args():
	parser = argparse.ArgumentParser(description=__doc__,
	                                 formatter_class=argparse.RawDescriptionHelpFormatter)
	add_corpus_args(parser)
	parser.add_argument("--out", default=None,
	                    help="defaults to outputs/<task>_emotion_history_analysis.json")
	parser.add_argument("--holdout_first", type=int, default=0,
	                    help="[p4g] dialogs to hold out; see the split discussion in "
	                         "mine_emotion_donation_p4g.py. 0 is correct for the self-play eval.")
	parser.add_argument("--lam", type=float, default=None,
	                    help="fix the decay instead of sweeping. 0 = instantaneous baseline.")
	parser.add_argument("--l2", type=float, default=1.0,
	                    help="ridge strength; required because simplex features are collinear "
	                         "with the intercept.")
	parser.add_argument("--folds", type=int, default=5, help="grouped CV folds (split by dialog)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--rescale_p99", type=float, default=1.0,
	                    help="rescale nu(c) so the 99th percentile of |nu| over observed states "
	                         "hits this value, keeping the emotion channel inside its documented "
	                         "[-1, +1] range so beta_emo keeps its meaning.")
	parser.add_argument("--no_rescale", action="store_const", const=None, dest="rescale_p99",
	                    help="report raw nu(c) instead (will exceed the channel's stated range).")
	args = parser.parse_args()
	if args.out is None:
		args.out = ("outputs/emotion_history_donation_analysis.json" if args.task == "p4g"
		            else f"outputs/{args.task}_emotion_history_analysis.json")
	return args


def main():
	args = parse_args()
	os.chdir(REPO_ROOT)

	sequences, meta = load_sequences(args.task, args.data, args.cache,
	                                 args.holdout_first, args.esc_window, args.esc_outcome)
	base_rate = sum(seq.succeeded for seq in sequences) / len(sequences)
	n_turns = sum(len(seq.emotions) for seq in sequences)
	print(f"{len(sequences)} sessions, {n_turns} user turns, base success rate {base_rate:.3f}\n")

	# --- choose the memory length ---
	if args.lam is not None:
		lambdas, chosen = [args.lam], args.lam
		print(f"using fixed lam={chosen:g} (no sweep)")
		results = sweep_lambda(sequences, lambdas, args.folds, args.l2, args.seed)
	else:
		print(f"grouped {args.folds}-fold CV over lam (folds split by dialog):")
		results = sweep_lambda(sequences, LAMBDA_GRID, args.folds, args.l2, args.seed)
		chosen = max(results, key=lambda r: r.auc).lam

	baseline = next(r for r in results if r.lam == 0.0) if any(r.lam == 0.0 for r in results) else None
	best = next(r for r in results if r.lam == chosen)
	print(f"\nselected lam={chosen:g} (AUC {best.auc:.4f})")
	if baseline is not None and chosen != 0.0:
		print(f"  vs instantaneous lam=0: AUC {baseline.auc:.4f} -> {best.auc:.4f} "
		      f"({best.auc - baseline.auc:+.4f}), "
		      f"logloss {baseline.logloss:.4f} -> {best.logloss:.4f} "
		      f"({best.logloss - baseline.logloss:+.4f})")

	# --- refit on everything at the chosen lam ---
	states = build_history_states(sequences, chosen)
	theta, bias = fit_logistic(states.features, states.labels, l2=args.l2)

	print("\nlogistic coefficients (log-odds of donation, partial effects):")
	print(f"  {'emotion':>10}  {'theta':>8}")
	for emotion in EMO_KEYS:
		if emotion in EMO_INDEX:
			print(f"  {emotion:>10}  {theta[EMO_INDEX[emotion]]:>+8.3f}")

	scale, raw_scores = fit_output_scale(states, theta, bias, base_rate, args.rescale_p99)
	print(f"\nnu(c) over {len(raw_scores)} observed states, before rescaling:")
	percentiles = {p: float(np.percentile(raw_scores, p)) for p in (1, 25, 50, 75, 99)}
	print("  " + "  ".join(f"p{p}={v:+.2f}" for p, v in percentiles.items())
	      + f"   max|nu|={np.abs(raw_scores).max():.2f}")
	if scale != 1.0:
		print(f"  rescaled by x{scale:.3f} to put p99|nu| at {args.rescale_p99:g} "
		      f"(channel is documented as [-1, +1]; instantaneous w peaks at 0.62)")

	weights = pure_emotion_weights(theta, bias, base_rate, scale)
	print("\npure-history weights (score of an all-one-emotion session, in w(e) units):")
	for emotion in EMO_KEYS:
		if emotion in weights:
			print(f"    Emotions.{emotion.capitalize():<10} {weights[emotion]:>+6.2f},")

	print("\nmixed histories — the point of the model (same units):")
	print(f"  {'history':>26}  {'nu(c)':>7}")
	example_scores = {}
	for name, shares in EXAMPLE_HISTORIES:
		value = score_state(mixture_state(shares), theta, bias, base_rate, scale)
		example_scores[name] = round(value, 3)
		print(f"  {name:>26}  {value:>+7.2f}")

	out = {
		"corpus": meta,
		"model": {"kind": "ridge_logistic_on_cumulative_emotion", "lam": chosen,
		          "l2": args.l2, "folds": args.folds, "seed": args.seed,
		          "output_scale": scale, "rescale_p99_target": args.rescale_p99},
		"observed_score_percentiles_raw": percentiles,
		"base_donation_rate": base_rate,
		"n_dialogs": len(sequences),
		"n_turns": n_turns,
		"cv_by_lambda": [{"lam": r.lam, "auc": r.auc, "logloss": r.logloss} for r in results],
		"theta": {e: float(theta[EMO_INDEX[e]]) for e in EMO_KEYS if e in EMO_INDEX},
		"bias": bias,
		"pure_emotion_weights": weights,
		"example_histories": example_scores,
	}
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(out, f, indent=2)
	print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
	main()
