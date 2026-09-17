"""Mine an empirical (emotion, DA) bonus matrix from the p4g training dialogs.

For each cell ``(last_user_emotion, next_persuader_DA)`` compute the lift in donation
rate vs the marginal:

    lift(e, da) = P(donate | (e, da) appears in dialog)
                 − P(donate | (e, da) does NOT appear in dialog)

Positive lift = the pair is more frequent in successful dialogs than failed ones
              → use as a persuasion booster.
Negative lift = the pair is more frequent in failed dialogs
              → use as a per-step penalty (already covered by π, but informative).

Output is a JSON file shaped exactly like the hand-seeded ``EMOTION_DA_BONUS`` in
``src/mcts/emotion_mcts.py``, drop-in via ``LearnedBonus`` (Item 5b in
``EMOMCTS_RESEARCH_DIRECTIONS.md``).

Why this exists
---------------
The hand-seeded matrix encodes communication-theory normative claims about good
persuasion. The data-mined matrix encodes empirical claims about *what historically
led to donation* — which is the actual MCTS optimisation target. The 2026-05-30
debug.md entry documents how the hand-seeded matrix's Neutral row was mis-calibrated;
mining gives a principled fix.

Usage
-----
    python src/emotion_mining/mine_emotion_da_bonus_p4g.py
    python src/emotion_mining/mine_emotion_da_bonus_p4g.py --classifier hf --scale 0.4 \
        --min_obs 5 --out outputs/learned_emotion_da_bonus.json

Outputs
-------
    outputs/learned_emotion_da_bonus.json           drop-in matrix
    outputs/learned_emotion_da_bonus_debug.json     raw cell counts + lift, for review
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict, Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402

POSITIVE_DONATION_LABELS = {
    "agree-donation",
    "provide-donation-amount",
    "confirm-donation",
}

# The persuader DA vocabulary the MCTS actually picks among (matches
# PersuasionGame.get_game_ontology()["system"]["dialog_acts"]). The p4g label set
# is broader and includes *response* DAs (`confirm donation`, `thank`, `praise user`,
# `ask donate more`, `donation information`, `you are welcome`, `closing`,
# `acknowledgement`, `ask donation amount`, `off task`) that the planner never plays
# strategically — they appear in successful dialogs because they FOLLOW the donation,
# not because they cause it. Including them in the bonus matrix would teach the planner
# to play closing-language at random times.
STRATEGIC_DAS = {
    "personal story", "credibility appeal", "emotion appeal", "proposition of donation",
    "foot in the door", "logical appeal", "self modeling", "task related inquiry",
    "source related inquiry", "personal related inquiry", "neutral to inquiry",
    "greeting", "other",
}

# p4g labels use hyphenated DA tags; map to spaced form the MCTS uses.
def _normalize_da(tag: str) -> str:
    return tag.replace("-", " ").strip().lower()


def build_classifier(kind: str):
    if kind == "hf":
        from emotion_classifiers.hf_emotion import HFEmotionClassifier
        return HFEmotionClassifier()
    from emotion_classifiers.llm_emotion import P4GLLMEmotionClassifier
    from utils.gen_models import OpenAIModel
    return P4GLLMEmotionClassifier(OpenAIModel(model_name="gpt-3.5-turbo"))


def label_emotions(classifier, dialogs: dict, cache_path: str) -> dict[str, str]:
    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    n_new = 0
    for did, d in dialogs.items():
        for turn in d["dialog"]:
            for utt in turn.get("ee", []):
                utt = (utt or "").strip()
                if utt and utt not in cache:
                    cache[utt] = str(classifier.predict_from_single_utterance(utt))
                    n_new += 1
    if n_new:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    print(f"emotion cache: {len(cache)} utterances ({n_new} new)")
    return cache


def dialog_donated(label_turns: list[dict]) -> bool:
    for turn in label_turns:
        for tag in turn.get("ee", []):
            if tag in POSITIVE_DONATION_LABELS:
                return True
    return False


def collect_cells(dialogs: dict, emo_cache: dict[str, str]) -> dict:
    """For each dialog, collect the set of (persuadee_emotion_at_t, persuader_DA_at_t+1)
    pairs that appear at least once. Then aggregate across dialogs by donation outcome.

    The "pair appears" framing (vs counting every occurrence) matches the lift formula
    and reduces sensitivity to dialog length.
    """
    # (emotion, da) -> {with_donate, with_no_donate, without_donate, without_no_donate}
    counts: dict = defaultdict(lambda: dict(with_d=0, with_nd=0, without_d=0, without_nd=0))
    n_total = 0
    n_donate = 0
    for did, d in dialogs.items():
        n_total += 1
        donate = dialog_donated(d["label"])
        n_donate += int(donate)
        # walk through pairs
        cells_in_dialog: set = set()
        last_persuadee_emotion: str | None = None
        for t, (turn, labels) in enumerate(zip(d["dialog"], d["label"])):
            # the DA being "considered/played" at turn t is the *persuader* DA at t
            sys_das = [_normalize_da(x) for x in labels.get("er", [])]
            sys_da = sys_das[0] if sys_das else None
            # condition on the last classified persuadee emotion *before* this sys turn
            if last_persuadee_emotion is not None and sys_da is not None:
                cells_in_dialog.add((last_persuadee_emotion, sys_da))
            # roll forward: classify this turn's persuadee utterance for the next iteration
            user_utts = turn.get("ee", [])
            user_utt = (user_utts[0].strip() if user_utts else "")
            if user_utt and user_utt in emo_cache:
                last_persuadee_emotion = emo_cache[user_utt]

        # increment counts based on whether the dialog donated
        for cell in cells_in_dialog:
            if donate:
                counts[cell]["with_d"] += 1
            else:
                counts[cell]["with_nd"] += 1
    # for each cell, compute "without" buckets as totals minus with-buckets
    for cell, c in counts.items():
        c["without_d"] = n_donate - c["with_d"]
        c["without_nd"] = (n_total - n_donate) - c["with_nd"]
    return counts, n_total, n_donate


def laplace_lift(c: dict, alpha: float = 1.0) -> float:
    """Smoothed lift: P(d | with) − P(d | without). Laplace alpha on both numerator and
    denominator of each conditional, so cells with tiny sample sizes shrink toward 0."""
    with_n = c["with_d"] + c["with_nd"]
    without_n = c["without_d"] + c["without_nd"]
    p_with = (c["with_d"] + alpha) / (with_n + 2 * alpha)
    p_without = (c["without_d"] + alpha) / (without_n + 2 * alpha)
    return p_with - p_without


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/p4g/300_dialog_turn_based.pkl")
    parser.add_argument("--classifier", choices=["hf", "llm"], default="hf")
    parser.add_argument("--cache", default="outputs/learned_emotion_da_bonus_emocache.json")
    parser.add_argument("--out", default="outputs/learned_emotion_da_bonus.json")
    parser.add_argument("--debug_out", default="outputs/learned_emotion_da_bonus_debug.json")
    parser.add_argument("--min_obs", type=int, default=5,
                        help="drop cells with fewer than this many dialogs in `with`")
    parser.add_argument("--scale", type=float, default=0.4,
                        help="rescale so max|bonus| == this. matches hand-seeded magnitude.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Laplace smoothing strength (higher = more shrinkage to 0)")
    parser.add_argument("--strategy_only", action="store_true", default=True,
                        help="restrict the output to strategic-action DAs only (see "
                             "STRATEGIC_DAS). Filters out response DAs whose lift is "
                             "confounded by reverse causality (they appear because the "
                             "user agreed, not because they caused agreement).")
    parser.add_argument("--no_strategy_only", action="store_false", dest="strategy_only",
                        help="disable the strategy-only filter (use all p4g DAs)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    with open(args.data, "rb") as f:
        dialogs: dict = pickle.load(f)
    print(f"loaded {len(dialogs)} dialogs")

    print(f"building classifier: {args.classifier}")
    classifier = build_classifier(args.classifier)
    emo_cache = label_emotions(classifier, dialogs, args.cache)

    counts, n_total, n_donate = collect_cells(dialogs, emo_cache)
    print(f"\ndialogs: {n_total}; donated: {n_donate} ({100*n_donate/n_total:.1f}%)")
    print(f"unique (emotion, DA) cells observed: {len(counts)}")

    # compute lifts
    lifts: dict = {}
    dropped_obs = 0
    dropped_nonstrategic = 0
    for cell, c in counts.items():
        emotion, da = cell
        if args.strategy_only and da not in STRATEGIC_DAS:
            dropped_nonstrategic += 1
            continue
        with_n = c["with_d"] + c["with_nd"]
        if with_n < args.min_obs:
            dropped_obs += 1
            continue
        lifts[cell] = laplace_lift(c, alpha=args.alpha)
    print(f"cells with ≥{args.min_obs} obs: {len(lifts)}; "
          f"dropped (low obs): {dropped_obs}; dropped (non-strategic): {dropped_nonstrategic}")

    # rescale so max|lift| == scale
    max_abs = max(abs(v) for v in lifts.values()) if lifts else 1.0
    scale_factor = args.scale / max_abs if max_abs > 0 else 1.0

    # nested dict shaped like EMOTION_DA_BONUS
    matrix: dict = defaultdict(dict)
    for (e, da), v in lifts.items():
        matrix[e][da] = round(scale_factor * v, 3)

    # output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_blob = {
        "config": vars(args),
        "n_dialogs": n_total,
        "n_donate": n_donate,
        "donate_rate": round(n_donate / n_total, 3),
        "n_cells_observed": len(counts),
        "n_cells_kept": len(lifts),
        "scale_factor_applied": round(scale_factor, 4),
        "matrix": {str(e): {da: v for da, v in das.items()} for e, das in matrix.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out_blob, f, indent=2)

    # debug dump: raw counts + raw lifts
    debug_blob = {
        "n_total": n_total, "n_donate": n_donate,
        "cells": [
            {
                "emotion": str(e), "da": da,
                "with_d": c["with_d"], "with_nd": c["with_nd"],
                "without_d": c["without_d"], "without_nd": c["without_nd"],
                "raw_lift": round(laplace_lift(c, alpha=args.alpha), 4),
                "scaled_bonus": round(scale_factor * laplace_lift(c, alpha=args.alpha), 4)
                                if c["with_d"] + c["with_nd"] >= args.min_obs else None,
            }
            for (e, da), c in sorted(counts.items())
        ],
    }
    with open(args.debug_out, "w") as f:
        json.dump(debug_blob, f, indent=2)

    print(f"\nsaved -> {args.out}")
    print(f"saved -> {args.debug_out}")

    # print summary table
    print("\nlearned bonus matrix:")
    for e in sorted(matrix.keys()):
        print(f"  [{e}]")
        for da, v in sorted(matrix[e].items(), key=lambda kv: -kv[1]):
            marker = "+" if v >= 0 else ""
            print(f"    {da:>30}  {marker}{v:.3f}")


if __name__ == "__main__":
    main()
