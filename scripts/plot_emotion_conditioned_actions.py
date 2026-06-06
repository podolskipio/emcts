import argparse, glob, os, sys, pickle, json, csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.getcwd())

NEG = {"fear", "sadness", "anger", "disgust", "contempt"}
GROUPS = ["trust-building", "proposition", "other"]
GCOL = {"trust-building": "#d73027", "proposition": "#1a9850", "other": "#9e9e9e"}
TRUST = {"emotion appeal", "credibility appeal"}
PROP = {"proposition of donation"}


def load(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    return o if isinstance(o, list) else o.get("episodes", o)


def _find(stem):
    g = [p for p in glob.glob("**/*.pkl", recursive=True) if os.path.basename(p) == stem + ".pkl"]
    if not g:
        raise FileNotFoundError(stem)
    return g[0]


def _group(da):
    if da in TRUST: return "trust-building"
    if da in PROP:  return "proposition"
    return "other"


def _emo_str(x):
    # EmoMCTS logs an Emotions enum or its str; normalise to lowercase name
    s = str(x).split(".")[-1].lower() if x is not None else ""
    return s


class _HFCache:
    """Lazy HF classifier with a JSON utterance->emotion cache."""
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.clf = None

    def label(self, utt):
        utt = (utt or "").strip()
        if not utt:
            return "neutral"
        if utt in self.cache:
            return self.cache[utt]
        if self.clf is None:
            from emotion_classifiers.hf_emotion import HFEmotionClassifier
            self.clf = HFEmotionClassifier()
        dist = self.clf.predict_distribution_from_utterance(utt)
        emo = _emo_str(max(dist, key=dist.get))
        self.cache[utt] = emo
        return emo

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        json.dump(self.cache, open(self.cache_path, "w"))


def tally(eps, has_emotion, hf=None):
    """Return counts[bucket][group] over system turns, keyed by the *preceding* user emotion."""
    counts = {"negative": defaultdict(int), "non-negative": defaultdict(int)}
    for e in eps:
        last_emo = None
        for turn in e["history"]:
            role = turn[0]
            if role == "Persuader":
                da = turn[1]
                if last_emo is not None:                 # skip the opening greeting
                    bucket = "negative" if last_emo in NEG else "non-negative"
                    counts[bucket][_group(da)] += 1
            else:  # Persuadee
                if has_emotion:
                    last_emo = _emo_str(turn[2])
                else:
                    last_emo = hf.label(turn[-1])        # classify the user utterance
    return counts


def _frac(counts_bucket):
    tot = sum(counts_bucket.values()) or 1
    return np.array([counts_bucket[g] / tot for g in GROUPS]), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gdpzero", default="rollout_p4g_gdpzero_vicuna_50d_40s")
    ap.add_argument("--emomcts", default="rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims")
    ap.add_argument("--cache", default="outputs/gdpzero_user_emocache.json")
    ap.add_argument("--out", default="outputs/emotion_conditioned_actions_40s.png")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    gd = load(args.gdpzero if os.path.exists(args.gdpzero) else _find(args.gdpzero))
    em = load(args.emomcts if os.path.exists(args.emomcts) else _find(args.emomcts))

    hf = _HFCache(args.cache)
    cg = tally(gd, has_emotion=False, hf=hf); hf.save()
    ce = tally(em, has_emotion=True)

    # 4 bars: GDPZero|neg, EmoMCTS|neg, GDPZero|non-neg, EmoMCTS|non-neg
    bars = [("GDP-Zero\n(after neg.)", cg["negative"]),
            ("EmoMCTS\n(after neg.)", ce["negative"]),
            ("GDP-Zero\n(after non-neg.)", cg["non-negative"]),
            ("EmoMCTS\n(after non-neg.)", ce["non-negative"])]
    labels = [b[0] for b in bars]
    fracs = [_frac(b[1])[0] for b in bars]
    ns = [_frac(b[1])[1] for b in bars]
    F = np.vstack(fracs)  # (4, 3)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(bars)); bottom = np.zeros(len(bars))
    for j, g in enumerate(GROUPS):
        ax.bar(x, F[:, j], bottom=bottom, color=GCOL[g], label=g, width=0.7,
               edgecolor="white", linewidth=0.4)
        bottom += F[:, j]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of next system actions"); ax.set_ylim(0, 1.08)
    ax.set_title("Action choice conditioned on the user's last emotion (40 sims)", fontsize=11, pad=14)
    for i, n in enumerate(ns):
        ax.text(i, 1.015, f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")
    ax.axvline(1.5, color="gray", lw=0.6, ls="--")
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["bar", "n"] + GROUPS)
            for lab, fr, n in zip([b[0].replace("\n", " ") for b in bars], fracs, ns):
                w.writerow([lab, n] + [f"{v:.4f}" for v in fr])
        print("saved", args.csv)

    print("\nbar".ljust(28), "n".rjust(5), "  ", "  ".join("%-14s" % g for g in GROUPS))
    for lab, fr, n in zip([b[0].replace("\n", " ") for b in bars], fracs, ns):
        print(lab.ljust(28), str(n).rjust(5), "  ", "  ".join("%-14.3f" % v for v in fr))


if __name__ == "__main__":
    main()
