"""Estimate a tabular Markov dynamics model from p4g for emotion imagination rollouts.

For each (last_user_emotion, sys_DA) compute the empirical distribution over the
NEXT user emotion:

    P(emotion_{t+1} | emotion_t, sys_DA_t)

from the 250 training dialogs, with Laplace alpha smoothing so unobserved cells
get nonzero mass.

This is the "tabular fallback" mentioned in EMOMCTS_RESEARCH_DIRECTIONS.md as a
cheaper start than training the full DistilRoBERTa anticipatory model from
``scripts/train_anticipatory_emotion_p4g.py``. The two are interchangeable from
the MCTS side:
- Tabular: 8 emotions x 13 DAs x 8 emotions = 832 cells, no model dependency.
- BERT:   richer (uses dialog text as context), but ~80MB model to load.

EmotionImaginationMCTS uses the table here for its imagination side-tree.

Also computes the empirical policy P(sys_DA) marginal in *successful* dialogs
(any user turn labelled ``agree-donation``), which is used by the imagination
rollout as the rollout policy for future sys actions. Reflects "what a good
persuader plays" rather than "what an arbitrary persuader plays."

Split convention
----------------
Mirrors scripts/train_emotion_conditioned_prior_p4g.py: first 50 dialogs (in
pickle iteration order) = TEST, last 250 = TRAIN. So this transition table is
by-construction held-out wrt the existing emomcts pickles for the first 20
test dialogs.

Outputs
-------
    outputs/emotion_transition_p4g.json   the table + policy + metadata
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from emotion_classifiers.llm_emotion import Emotions  # noqa: E402

# Mirrors PersuasionGame.get_game_ontology()['system']['dialog_acts']
SYS_DAS = [
    "personal story", "credibility appeal", "emotion appeal", "proposition of donation",
    "foot in the door", "logical appeal", "self modeling", "task related inquiry",
    "source related inquiry", "personal related inquiry", "neutral to inquiry",
    "greeting", "other",
]
EMO_LIST = [str(e) for e in Emotions]

DEFAULT_LAST_EMOTION = str(Emotions.Neutral)


def _normalize_da(tag: str) -> str:
    """Map raw p4g labels (hyphenated, off-ontology) to the spaced PersuasionGame DAs."""
    spaced = (tag or "").replace("-", " ").strip().lower()
    return spaced if spaced in SYS_DAS else "other"


def _dialog_is_successful(dialog: dict) -> bool:
    for label in dialog["label"]:
        for tag in label.get("ee", []):
            if tag == "agree-donation":
                return True
    return False


def build_classifier(kind: str):
    if kind == "hf":
        from emotion_classifiers.hf_emotion import HFEmotionClassifier
        return HFEmotionClassifier()
    if kind == "llm":
        from emotion_classifiers.llm_emotion import P4GLLMEmotionClassifier
        from utils.gen_models import OpenAIModel
        return P4GLLMEmotionClassifier(OpenAIModel(model_name="gpt-3.5-turbo"))
    raise ValueError(f"unknown classifier kind: {kind!r}")


def label_emotions(classifier, dialogs: dict, cache_path: str) -> dict[str, str]:
    """utt -> emotion. Shared cache across train scripts so labelling cost is paid once."""
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


def extract_transitions(d: dict, emo_cache: dict[str, str]):
    """Yield (prev_emo, sys_da, next_emo) triples from one dialog.

    For each system turn t with a labelled persuadee response:
        prev_emo = persuadee emotion at user turn t-1 (or DEFAULT_LAST_EMOTION for t=0)
        sys_da   = normalized DA played at sys turn t (first label)
        next_emo = persuadee emotion at user turn t

    Matches the conditioning the MCTS uses at expansion time (we know the user's
    emotion from before our move, we choose a DA, we observe what emotion comes back).
    """
    prev_emo = DEFAULT_LAST_EMOTION
    for turn, labels in zip(d["dialog"], d["label"]):
        user_utts = turn.get("ee", [])
        sys_das_raw = labels.get("er", [])
        sys_da = _normalize_da(sys_das_raw[0]) if sys_das_raw else "other"
        user_utt = (user_utts[0].strip() if user_utts else "")

        if user_utt and user_utt in emo_cache:
            next_emo = emo_cache[user_utt]
            yield (prev_emo, sys_da, next_emo)
            prev_emo = next_emo


def fit_transition_table(dialogs, train_ids, emo_cache, alpha: float):
    """Returns P[prev_emo][sys_da][next_emo] with Laplace alpha smoothing."""
    counts: dict = {e: {da: Counter() for da in SYS_DAS} for e in EMO_LIST}
    total_observed = 0
    for did in train_ids:
        for prev_emo, sys_da, next_emo in extract_transitions(dialogs[did], emo_cache):
            counts[prev_emo][sys_da][next_emo] += 1
            total_observed += 1

    # Laplace smoothing + normalize: each cell counts[prev][da] -> distribution over EMO_LIST
    table: dict = {}
    for prev_emo in EMO_LIST:
        table[prev_emo] = {}
        for da in SYS_DAS:
            c = counts[prev_emo][da]
            smoothed = {e: c.get(e, 0) + alpha for e in EMO_LIST}
            total = sum(smoothed.values())
            table[prev_emo][da] = {e: v / total for e, v in smoothed.items()}
    return table, total_observed


def fit_successful_da_policy(dialogs, train_ids, alpha: float):
    """Empirical sys_DA marginal from SUCCESSFUL dialogs only (used as the
    imagination rollout's future-DA policy). Laplace-smoothed so all DAs have
    nonzero mass even if a DA never appeared in a successful train dialog."""
    counts: Counter = Counter()
    n_success = 0
    for did in train_ids:
        if not _dialog_is_successful(dialogs[did]):
            continue
        n_success += 1
        for labels in dialogs[did]["label"]:
            for raw_da in labels.get("er", []):
                counts[_normalize_da(raw_da)] += 1

    smoothed = {da: counts.get(da, 0) + alpha for da in SYS_DAS}
    total = sum(smoothed.values())
    return {da: c / total for da, c in smoothed.items()}, n_success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/p4g/300_dialog_turn_based.pkl")
    parser.add_argument("--classifier", choices=["hf", "llm"], default="hf",
                        help="emotion classifier used to label persuadee turns")
    parser.add_argument("--n_test", type=int, default=50,
                        help="first N dialogs (pickle iteration order) held out as test "
                             "(matches train_emotion_conditioned_prior_p4g.py's split).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Laplace smoothing prior on transition counts")
    parser.add_argument("--policy_alpha", type=float, default=1.0,
                        help="Laplace smoothing prior on successful-DA marginal")
    parser.add_argument("--out", default="outputs/emotion_transition_p4g.json")
    parser.add_argument("--cache", default="outputs/anticipatory_pemo_emocache.json",
                        help="shared emotion-label cache with the other train scripts")
    parser.add_argument("--print_table", action="store_true",
                        help="print the full smoothed table to stdout (~832 lines).")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    with open(args.data, "rb") as f:
        dialogs: dict = pickle.load(f)
    print(f"loaded {len(dialogs)} dialogs")

    print(f"building emotion classifier for labelling: {args.classifier}")
    classifier = build_classifier(args.classifier)
    emo_cache = label_emotions(classifier, dialogs, args.cache)

    ids = list(dialogs.keys())
    test_ids, train_ids = ids[: args.n_test], ids[args.n_test:]
    n_train_success = sum(_dialog_is_successful(dialogs[i]) for i in train_ids)
    n_test_success = sum(_dialog_is_successful(dialogs[i]) for i in test_ids)
    print(f"split: train={len(train_ids)} dialogs ({n_train_success} successful), "
          f"test={len(test_ids)} dialogs ({n_test_success} successful)")

    print(f"fitting transition table (alpha={args.alpha}) ...")
    table, total = fit_transition_table(dialogs, train_ids, emo_cache, args.alpha)
    print(f"  {total} (prev_emo, sys_da, next_emo) triples observed in train split")

    print(f"fitting successful-dialog sys_DA policy (alpha={args.policy_alpha}) ...")
    policy, n_success = fit_successful_da_policy(dialogs, train_ids, args.policy_alpha)
    print(f"  policy estimated from {n_success} successful train dialogs")

    # Per-row diagnostic: how concentrated is each transition row? Useful sanity
    # check that we're not just spitting out uniform distributions due to
    # over-smoothing on rare cells.
    print("\ntop-3 next-emotion modes for select (prev_emo, sys_DA) cells:")
    for prev_emo in EMO_LIST:
        for da in ("emotion appeal", "proposition of donation", "credibility appeal"):
            dist = table[prev_emo][da]
            top3 = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
            top_str = ", ".join(f"{e}={p:.2f}" for e, p in top3)
            print(f"  P(next | prev={prev_emo:>10}, da={da:>25}): {top_str}")

    print("\nsuccessful-dialog sys_DA policy (sorted):")
    for da, p in sorted(policy.items(), key=lambda kv: -kv[1]):
        print(f"  {da:>30}: {p:.3f}")

    if args.print_table:
        print("\nfull transition table:")
        print(json.dumps(table, indent=2))

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "n_train_dialogs": len(train_ids),
            "n_test_dialogs": len(test_ids),
            "n_train_success": n_train_success,
            "n_test_success": n_test_success,
            "n_transitions_observed": total,
            "alpha_transition": args.alpha,
            "alpha_policy": args.policy_alpha,
            "emo_vocab": EMO_LIST,
            "sys_da_vocab": SYS_DAS,
            "default_last_emotion": DEFAULT_LAST_EMOTION,
            "label_classifier": args.classifier,
            "test_ids": test_ids,
        },
        # P[prev_emo][sys_da] -> distribution over next_emo
        "transition_table": table,
        # P(sys_da) marginal from successful dialogs (rollout policy)
        "successful_da_policy": policy,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
