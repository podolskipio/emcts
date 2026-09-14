"""Mine the per-emotion valence weights ``w(e)`` from a dialog corpus (p4g or esc).

For each emotion ``e``, compute the empirical donation lift over the corpus base rate

    lift(e) = P(donate | user-turn emotion = e) − P(donate),

with Wilson 95% CIs, and set the valence weight ``w(e) = 10 * shrunk_lift(e)``. The result is
transcribed into ``EMOTION_VALENCE_MINED`` in ``src/mcts/emotion_mcts.py``, which
``EmotionAwareMultiObjectiveQ`` uses to score the emotion channel in PUCT.

Every number here is a *correlational* lift: "fearful users donate more often" does not
license "inducing fear causes donation". Cells whose CI contains the base rate are flagged
unstarred in the printed table and should be read as ~0 however large the point estimate.

Shrinkage
---------
Raw lifts on this corpus are unstable: the thin cells (fear n=70, anger n=88, disgust n=92)
swing wildly under resampling — mining on 200 of the 300 dialogs moves w(fear) from +1.07 to
+0.01 — while none of them has a CI excluding the base rate. Only happiness does. So ``w(e)``
is built from a lift shrunk toward the base rate by ``--alpha`` pseudo-observations
(``Tally.shrunk_lift``), which pulls under-evidenced cells toward 0 in proportion to how thin
they are and leaves the well-populated ones essentially untouched. ``--alpha 0`` recovers the
raw lifts. This is a property of the corpus, not of any train/test split.

Split convention
----------------
``--holdout_first N`` drops the first N dialogs (after removing ``P4G_BAD_DIALOGS``) from the
mining set. Whether you need it depends entirely on which evaluation the weights feed:

* **Self-play** (``runners/rollout.py``, the SR/AvgT table) — no holdout needed; default is 0.
  P4G dialogs carry ``scenario=()`` (``_common.py:171``), so ``game.init_dialog()`` starts from
  an empty session and the runner never reads ``dialog["turns"]``. The corpus contributes only
  a dialog id used as an episode label. The one piece of corpus text that does reach the
  prompt, ``EXP_DIALOG`` in ``utils/prompt_examples.py``, is hardcoded and unaffected by any
  split. Holding dialogs out therefore protects nothing and just costs a third of the data.
* **Replay** (``runners/emomcts.py`` / ``gdpzero.py``) — holdout REQUIRED, use 100. These feed
  real dialog text into the planner and score its prediction against the ground-truth next
  turn, so mining on a dialog the runner later replays is genuine leakage.

Note the filter ordering: 3 bad dialogs sit at raw indices 4, 7 and 19, so slicing the raw
pickle order would hold out 3 fewer eval dialogs than the replay runner actually visits. We
filter first, then slice — matching the runner exactly.

Structure
---------
``main`` runs three phases, in order: **split** the corpus (``select_mining_dialogs``),
**count** (the ``tally_*`` functions, which all consume one ``EmotionSequence`` pass), then
**report** (the ``report_*`` functions print; ``main`` assembles the JSON). Counting is
separated from reporting so the statistics can be checked without reading format strings.

Tasks
-----
``--task p4g`` (default) mines against the annotated ``agree-donation`` outcome. ``--task esc``
mines against an internally-defined outcome, because ESConv ships no outcome annotation at all —
see ``mining_corpus.py``, which owns every task-specific detail, for what that means and what it
costs in interpretability. The rest of this script is task-agnostic.

Usage
-----
    python src/emotion_mining/mine_emotion_donation_p4g.py                        # self-play: all 300
    python src/emotion_mining/mine_emotion_donation_p4g.py --task esc             # ESConv
    python src/emotion_mining/mine_emotion_donation_p4g.py --holdout_first 100    # replay eval: held out
    python src/emotion_mining/mine_emotion_donation_p4g.py --alpha 0              # raw, unshrunk lifts

Outputs
-------
    outputs/emotion_donation_analysis.json   lifts, weights, CIs + split metadata
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from emotion_mining.mining_corpus import EMO_KEYS, add_corpus_args, load_sequences  # noqa: E402

# Emotions the HF encoder can emit that read as negative affect. Used only by the
# transition-class hypothesis tests below.
NEGATIVE_EMOTIONS = {"sadness", "fear", "anger", "disgust", "contempt"}

# HF never emits contempt, so it gets no mined weight; this hand-set value is the fallback
# for the alternative LLM classifier, which does emit it.
CONTEMPT_FALLBACK_WEIGHT = -0.60

# w(e) = VALENCE_SCALE * lift(e). Chosen so the weights land in roughly [-1, +1], the range
# EmotionAwareMultiObjectiveQ's Q_emo channel expects.
VALENCE_SCALE = 10

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def fmt_n(n: float, width: int = 0) -> str:
	"""Format a tally count: exact integer in argmax mode, 1dp once soft mining makes it
	an effective (fractional) sample size."""
	text = f"{int(round(n))}" if float(n).is_integer() else f"{n:.1f}"
	return f"{text:>{width}}" if width else text


@dataclass
class Tally:
	"""Donation outcomes for one bucket: ``successes`` donations out of ``trials``.

	Every table in this script is the same question asked of a different bucket — a turn
	emotion, a dialog's dominant emotion, an emotion transition — so they all reduce to this.
	"""
	successes: float = 0      # int in argmax mode, float once add_soft is used
	trials: float = 0

	def add(self, donated: bool) -> None:
		self.trials += 1
		self.successes += int(donated)

	def add_soft(self, mass: float, donated: bool) -> None:
		"""Accumulate a fractional observation: ``T(e) += d(e)``, ``S(e) += d(e)*y``.

		The argmax path credits a whole turn to one emotion. This credits every emotion in
		proportion to the classifier's posterior, which is the same quantity deployment uses
		in ``nu(d) = sum_e d(e)*w(e)``. ``trials``/``successes`` become floats; every downstream
		statistic (p_donate, lift, shrunk_lift, Wilson) is already arithmetic on those two
		numbers, so it carries over unchanged — but ``trials`` is now an *effective* sample
		size, not a count of turns, and the Wilson interval should be read accordingly.
		"""
		self.trials += mass
		self.successes += mass * int(donated)

	@property
	def p_donate(self) -> float:
		return self.successes / self.trials if self.trials else 0.0

	def lift(self, base_rate: float) -> float:
		"""Raw, unsmoothed lift. Reported as the observed effect; not what w(e) is built from."""
		return self.p_donate - base_rate

	def shrunk_lift(self, base_rate: float, alpha: float) -> float:
		"""Lift after shrinking P(donate) toward the base rate with ``alpha`` pseudo-observations.

		    p_shrunk = (successes + alpha * base_rate) / (trials + alpha)

		A Beta prior centred on the base rate, so a cell with no evidence has lift exactly 0 and
		thin cells are pulled toward "indistinguishable from the corpus average" in proportion to
		how thin they are. Note this is *not* the Laplace form in mine_emotion_da_bonus_p4g.py,
		which shrinks toward 0.5 — that only coincidentally shrinks lift toward 0 here because
		this corpus's base rate happens to sit near 0.5.

		alpha=0 reproduces the raw lift.
		"""
		if self.trials == 0:
			return 0.0
		p_shrunk = (self.successes + alpha * base_rate) / (self.trials + alpha)
		return p_shrunk - base_rate

	def weight(self, base_rate: float, alpha: float) -> float:
		"""w(e), built from the shrunk lift so thin cells cannot dominate the emotion channel."""
		return VALENCE_SCALE * self.shrunk_lift(base_rate, alpha)

	def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
		"""Wilson score interval — behaves at the small n (fear: 47 turns) this corpus forces."""
		if self.trials == 0:
			return (0.0, 0.0)
		n, p = self.trials, self.p_donate
		denom = 1 + z * z / n
		center = (p + z * z / (2 * n)) / denom
		margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
		return (max(0.0, center - margin), min(1.0, center + margin))

	def is_significant(self, base_rate: float) -> bool:
		"""True when the CI excludes the base rate, i.e. the lift is distinguishable from chance."""
		lo, hi = self.wilson_ci()
		return not (lo <= base_rate <= hi)


def sum_tallies(tallies) -> Tally:
	total = Tally()
	for t in tallies:
		total.successes += t.successes
		total.trials += t.trials
	return total


# --------------------------------------------------------------------------------------
# Tallies — one function per question asked of the corpus
# --------------------------------------------------------------------------------------

def tally_per_turn(sequences: list[EmotionSequence]) -> dict[str, Tally]:
	"""P(donate | this user turn's emotion = e). The table that produces w(e)."""
	tallies = {e: Tally() for e in EMO_KEYS}
	for seq in sequences:
		for emotion in seq.emotions:
			tallies[emotion].add(seq.succeeded)
	return tallies


def tally_per_turn_soft(sequences: list[EmotionSequence]) -> dict[str, Tally]:
	"""Soft counterpart of ``tally_per_turn``: every turn contributes d(e) to every emotion.

	Algorithm 2 line 7 labels a turn by ``argmax_e Phi(e'|u)`` and credits that one emotion.
	Deployment never does this — it scores ``nu(d) = sum_e d(e)*w(e)`` over the full softmax.
	Mining soft removes that mismatch: the statistic w(e) is fitted on is now the same
	functional of Phi that the planner consumes.
	"""
	tallies = {e: Tally() for e in EMO_KEYS}
	for seq in sequences:
		for dist in seq.distributions:
			for emotion, mass in dist.items():
				if emotion in tallies and mass:
					tallies[emotion].add_soft(float(mass), seq.succeeded)
	return tallies


def tally_dominant_emotion(sequences: list[EmotionSequence]) -> dict[str, Tally]:
	"""P(donate | dialog's most frequent user emotion = e). One observation per dialog."""
	tallies = {e: Tally() for e in EMO_KEYS}
	for seq in sequences:
		dominant = Counter(seq.emotions).most_common(1)[0][0]
		tallies[dominant].add(seq.succeeded)
	return tallies


def tally_transitions(sequences: list[EmotionSequence]) -> dict[tuple[str, str], Tally]:
	"""P(donate | consecutive user turns went prev -> next)."""
	tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
	for seq in sequences:
		for prev, nxt in zip(seq.emotions, seq.emotions[1:]):
			tallies[(prev, nxt)].add(seq.succeeded)
	return tallies


def tally_first_last_shift(sequences: list[EmotionSequence]) -> dict[tuple[str, str], Tally]:
	"""P(donate | dialog opened in emotion `first` and closed in `last`)."""
	tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
	for seq in sequences:
		tallies[(seq.emotions[0], seq.emotions[-1])].add(seq.succeeded)
	return tallies


# Does persuasion need sympathy? Each case aggregates the transition tallies matching a
# predicate over (prev_emotion, next_emotion).
TRANSITION_CLASSES = [
	("happy -> negative",    lambda a, b: a == "happiness" and b in NEGATIVE_EMOTIONS),
	("happy -> happy",       lambda a, b: a == "happiness" and b == "happiness"),
	("neutral -> negative",  lambda a, b: a == "neutral" and b in NEGATIVE_EMOTIONS),
	("neutral -> happy",     lambda a, b: a == "neutral" and b == "happiness"),
	("negative -> happy",    lambda a, b: a in NEGATIVE_EMOTIONS and b == "happiness"),
	("negative -> negative", lambda a, b: a in NEGATIVE_EMOTIONS and b in NEGATIVE_EMOTIONS),
]


def tally_transition_classes(transitions: dict[tuple[str, str], Tally]) -> dict[str, Tally]:
	return {
		name: sum_tallies(t for (prev, nxt), t in transitions.items() if matches(prev, nxt))
		for name, matches in TRANSITION_CLASSES
	}


def mine_valence_weights(per_turn: dict[str, Tally], base_rate: float, alpha: float) -> dict[str, float]:
	"""w(e) for every emotion the classifier actually emitted, plus the contempt fallback."""
	weights = {e: round(t.weight(base_rate, alpha), 2)
	           for e, t in per_turn.items() if t.trials > 0}
	weights["contempt"] = CONTEMPT_FALLBACK_WEIGHT
	return weights


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def print_heading(title: str) -> None:
	print("\n" + "=" * 78)
	print(title)
	print("=" * 78)


def report_per_turn(per_turn: dict[str, Tally], base_rate: float, alpha: float) -> list[dict]:
	print("(a) Per-USER-TURN donation rate (turn's emotion -> outcome of its dialog):")
	print(f"  lift is raw; w is shrunk toward the base rate with alpha={alpha:g} pseudo-obs, "
	      f"so w != 10*lift for thin cells.")
	print(f"  {'emotion':>10}  {'n_turns':>8}  {'P(donate)':>10}  {'lift':>7}  {'w':>7}"
	      f"  {'95% Wilson CI':>17}  sig")
	rows = []
	for emotion in EMO_KEYS:
		tally = per_turn[emotion]
		if tally.trials == 0:
			continue
		lo, hi = tally.wilson_ci()
		significant = tally.is_significant(base_rate)
		print(f"  {emotion:>10}  {fmt_n(tally.trials, 8)}  {tally.p_donate:>10.3f}"
		      f"  {tally.lift(base_rate):>+7.3f}  {tally.weight(base_rate, alpha):>+7.2f}"
		      f"  [{lo:.3f}, {hi:.3f}]  {'*' if significant else ''}")
		rows.append({
			"emotion": emotion, "n_turns": tally.trials, "p_donate": tally.p_donate,
			"lift": tally.lift(base_rate),
			"lift_shrunk": tally.shrunk_lift(base_rate, alpha),
			"w": round(tally.weight(base_rate, alpha), 2),
			"ci": [lo, hi], "ci_excludes_base_rate": significant,
		})
	return rows


def report_dominant_emotion(dominant: dict[str, Tally], base_rate: float) -> list[dict]:
	print("\n(b) Per-DIALOG donation rate (dialog's dominant user emotion):")
	print(f"  {'dominant':>10}  {'n_dialogs':>9}  {'P(donate)':>10}  {'lift':>7}  {'95% Wilson CI':>17}")
	rows = []
	for emotion in EMO_KEYS:
		tally = dominant[emotion]
		if tally.trials == 0:
			continue
		lo, hi = tally.wilson_ci()
		print(f"  {emotion:>10}  {fmt_n(tally.trials, 9)}  {tally.p_donate:>10.3f}"
		      f"  {tally.lift(base_rate):>+7.3f}  [{lo:.3f}, {hi:.3f}]")
		rows.append({"emotion": emotion, "n_dialogs": tally.trials, "p_donate": tally.p_donate,
		             "lift": tally.lift(base_rate), "ci": [lo, hi]})
	return rows


def report_transition_matrix(transitions: dict[tuple[str, str], Tally],
                             base_rate: float, min_obs: int) -> None:
	print(f"\n  rows = prev_emo, cols = next_emo. Cells = P(donate). Hidden if n < {min_obs}.")
	print(f"  Base rate = {base_rate:.3f}.\n")

	header = "             " + " ".join(f"{e[:6]:>7}" for e in EMO_KEYS)
	print(header)
	for prev in EMO_KEYS:
		cells = [f"{transitions[(prev, nxt)].p_donate:>7.2f}"
		         if transitions[(prev, nxt)].trials >= min_obs else f"{'.':>7}"
		         for nxt in EMO_KEYS]
		print(f"  {prev:>10} " + " ".join(cells))

	print("\n  Counts (n):")
	print(header)
	for prev in EMO_KEYS:
		cells = [fmt_n(transitions[(prev, nxt)].trials, 7)
		         if transitions[(prev, nxt)].trials > 0 else f"{'.':>7}"
		         for nxt in EMO_KEYS]
		print(f"  {prev:>10} " + " ".join(cells))


def report_ranked_pairs(tallies: dict[tuple[str, str], Tally], base_rate: float, min_obs: int,
                        left_label: str, right_label: str, top: int, bottom: int = 0) -> None:
	"""Print the best (and optionally worst) emotion pairs by lift, skipping thin cells.

	Ties on lift break on sample size (more evidence ranks first), then alphabetically, so the
	table is reproducible run to run rather than dependent on corpus iteration order.
	"""
	ranked = sorted(
		((t.lift(base_rate), t, pair) for pair, t in tallies.items() if t.trials >= min_obs),
		key=lambda row: (-row[0], -row[1].trials, row[2]),
	)
	columns = f"    {left_label:>10} -> {right_label:>10}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}"

	def print_rows(rows):
		print(columns)
		for lift, tally, (left, right) in rows:
			print(f"    {left:>10} -> {right:>10}  {fmt_n(tally.trials, 4)}"
			      f"  {tally.p_donate:>10.3f}  {lift:>+7.3f}")

	print_rows(ranked[:top])
	if bottom:
		print(f"\n  Bottom transitions (negative lift, n >= {min_obs}):")
		print_rows(ranked[-bottom:])


def report_transition_classes(classes: dict[str, Tally], base_rate: float) -> None:
	print(f"\n  {'transition class':>22}  {'n':>4}  {'P(donate)':>10}  {'lift':>7}")
	for name, tally in classes.items():
		if tally.trials == 0:
			print(f"  {name:>22}  {fmt_n(tally.trials, 4)}  {'(none)':>10}")
			continue
		print(f"  {name:>22}  {fmt_n(tally.trials, 4)}  {tally.p_donate:>10.3f}"
		      f"  {tally.lift(base_rate):>+7.3f}")


def report_valence_weights(weights: dict[str, float]) -> None:
	print("\n  EMOTION_VALENCE_MINED (paste into src/mcts/emotion_mcts.py):")
	for emotion in EMO_KEYS:
		if emotion in weights:
			print(f"    Emotions.{emotion.capitalize():<10} {weights[emotion]:>+6.2f},")


# --------------------------------------------------------------------------------------

def build_output(args, meta, base_rate, weights, per_turn_rows, dominant_rows,
                 transitions, transition_classes) -> dict:
	return {
		"corpus": meta,
		"shrinkage_alpha": args.alpha,
		"base_donation_rate": base_rate,
		"n_dialogs": meta["n_sessions"],
		"valence_weights": weights,
		"per_turn_emotion": per_turn_rows,
		"per_dialog_dominant_emotion": dominant_rows,
		"transition_table_p_donate": {
			f"{a}->{b}": transitions[(a, b)].p_donate if transitions[(a, b)].trials else None
			for a in EMO_KEYS for b in EMO_KEYS
		},
		"transition_table_n": {
			f"{a}->{b}": transitions[(a, b)].trials for a in EMO_KEYS for b in EMO_KEYS
		},
		"hypothesis_tests": {
			name: {"n": tally.trials, "p_donate": tally.p_donate}
			for name, tally in transition_classes.items()
		},
	}


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__,
	                                 formatter_class=argparse.RawDescriptionHelpFormatter)
	add_corpus_args(parser)
	parser.add_argument("--out", default=None,
	                    help="defaults to outputs/<task>_emotion_donation_analysis.json")
	parser.add_argument("--min_obs", type=int, default=5,
	                    help="hide cells with fewer than min_obs samples from the reported table")
	parser.add_argument("--holdout_first", type=int, default=0,
	                    help="hold out the first N dialogs *after* dropping P4G_BAD_DIALOGS — "
	                         "the exact set the REPLAY runners (emomcts.py/gdpzero.py) score "
	                         "against. Set to 100 when the mined weights feed a replay eval. "
	                         "Default 0 (mine on all 300) is correct for the self-play eval, "
	                         "which replays no corpus text — see the module docstring.")
	parser.add_argument("--soft", action="store_true",
	                    help="SOFT assignment: credit every emotion its posterior mass, "
	                         "T(e) += d(e) and S(e) += d(e)*y, instead of crediting only "
	                         "argmax_e Phi(e|u). Matches what deployment consumes, "
	                         "nu(d) = sum_e d(e)*w(e); the argmax default does not.")
	parser.add_argument("--dist_cache", default=None,
	                    help="[--soft] distribution cache path; defaults to the argmax cache "
	                         "path with a _dist suffix")
	parser.add_argument("--exclude_ids", default=None,
	                    help="path to a file of dialog ids (one per line, or a JSON list) to "
	                         "drop from the mining corpus — leakage insurance when the eval "
	                         "set is defined by id rather than by position")
	parser.add_argument("--assert_no_exclusions", action="store_true",
	                    help="fail if --exclude_ids removes any dialog. Use on the production "
	                         "run, where the eval set is disjoint by construction and the id "
	                         "list is a tripwire rather than a filter")
	parser.add_argument("--alpha", type=float, default=50.0,
	                    help="pseudo-observations of shrinkage toward the base rate when "
	                         "computing w(e). 0 = raw lifts.")
	args = parser.parse_args()
	if args.out is None:
		# p4g keeps its historical filename (referenced by README and the analysis notes);
		# other tasks get a prefixed one.
		args.out = ("outputs/emotion_donation_analysis.json" if args.task == "p4g"
		            else f"outputs/{args.task}_emotion_donation_analysis.json")
	return args


def main():
	args = parse_args()
	os.chdir(REPO_ROOT)

	# --- load + split ---
	exclude_ids = None
	if args.exclude_ids:
		raw = Path(args.exclude_ids).read_text().strip()
		ids = json.loads(raw) if raw.startswith("[") else [
			ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
		exclude_ids = set(ids)
		print(f"leakage insurance: excluding {len(exclude_ids)} dialog ids "
		      f"listed in {args.exclude_ids}")
	sequences, meta = load_sequences(args.task, args.data, args.cache,
	                                 args.holdout_first, args.esc_window, args.esc_outcome,
	                                 soft=args.soft, dist_cache_path=args.dist_cache,
	                                 exclude_ids=exclude_ids,
	                                 assert_no_exclusions=args.assert_no_exclusions)

	# Base rate over exactly the sessions the tallies cover, so every lift is measured against
	# its own population.
	base_rate = sum(seq.succeeded for seq in sequences) / len(sequences)
	print(f"base success rate ({len(sequences)} {args.task} sessions): {base_rate:.3f}")

	# Algorithm 2 line 7 vs. deployment: argmax credits one emotion per turn, soft credits
	# every emotion its posterior mass. Only this table feeds w(e); the diagnostic tables
	# below stay on argmax labels so they remain comparable across runs.
	per_turn = tally_per_turn_soft(sequences) if args.soft else tally_per_turn(sequences)
	print(f"per-turn assignment: {'SOFT (T(e)+=d(e), S(e)+=d(e)*y)' if args.soft else 'argmax'}")
	dominant = tally_dominant_emotion(sequences)
	transitions = tally_transitions(sequences)
	shifts = tally_first_last_shift(sequences)
	transition_classes = tally_transition_classes(transitions)
	weights = mine_valence_weights(per_turn, base_rate, args.alpha)

	# --- report ---
	print_heading("Q1) Donation rate by USER emotion")
	print(f"\nBase rate: {base_rate:.3f}\n")
	per_turn_rows = report_per_turn(per_turn, base_rate, args.alpha)
	dominant_rows = report_dominant_emotion(dominant, base_rate)

	print_heading("Q2) Donation rate by USER-EMOTION TRANSITION (consecutive turns)")
	report_transition_matrix(transitions, base_rate, args.min_obs)
	print(f"\n  Top transitions with positive lift (n >= {args.min_obs}, ranked by lift):")
	report_ranked_pairs(transitions, base_rate, args.min_obs, "prev", "next", top=10, bottom=5)

	print_heading("Specific test of the 'persuasion needs sympathy' hypothesis")
	report_transition_classes(transition_classes, base_rate)

	print(f"\n  First-user-turn -> last-user-turn shift (n >= {args.min_obs}):")
	report_ranked_pairs(shifts, base_rate, args.min_obs, "first", "last", top=10)

	report_valence_weights(weights)

	out = build_output(args, meta, base_rate, weights,
	                   per_turn_rows, dominant_rows, transitions, transition_classes)
	Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(out, f, indent=2)
	print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
	main()
