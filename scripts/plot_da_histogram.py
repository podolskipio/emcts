import argparse, glob, os, sys, pickle, csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.getcwd())

ACTS = ["greeting", "task related inquiry", "logical appeal",
        "credibility appeal", "emotion appeal", "proposition of donation"]
COL = {"greeting": "#bdbdbd", "task related inquiry": "#74add1", "logical appeal": "#4575b4",
       "credibility appeal": "#fdae61", "emotion appeal": "#d73027",
       "proposition of donation": "#1a9850"}


def load(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    return o if isinstance(o, list) else o.get("episodes", o)


def _find(stem):
    g = [p for p in glob.glob("**/*.pkl", recursive=True) if os.path.basename(p) == stem + ".pkl"]
    if not g:
        raise FileNotFoundError(stem)
    return g[0]


def per_turn(eps, max_turn):
    counts = [defaultdict(int) for _ in range(max_turn)]
    for e in eps:
        st = 0
        for role, *rest in e["history"]:
            if role == "Persuader":
                if st < max_turn:
                    da = rest[0]
                    counts[st][da if da in ACTS else "other"] += 1
                st += 1
    M = np.zeros((max_turn, len(ACTS)))
    for t in range(max_turn):
        tot = sum(counts[t].values()) or 1
        for j, a in enumerate(ACTS):
            M[t, j] = counts[t][a] / tot
    return M


def _stacked(ax, M, title, max_turn):
    x = np.arange(max_turn); bottom = np.zeros(max_turn)
    for j, a in enumerate(ACTS):
        ax.bar(x, M[:, j], bottom=bottom, color=COL[a], label=a,
               width=0.8, edgecolor="white", linewidth=0.3)
        bottom += M[:, j]
    ax.set_title(title); ax.set_xlabel("system turn"); ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gdpzero", default="rollout_p4g_gdpzero_vicuna_50d_40s",
                    help="GDP-Zero pickle path or run-id stem")
    ap.add_argument("--emomcts", default="rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims",
                    help="EmoMCTS pickle path or run-id stem")
    ap.add_argument("--max_turn", type=int, default=6)
    ap.add_argument("--title", default="System dialogue-act distribution by turn",
                    help="figure suptitle (e.g. add the sim count)")
    ap.add_argument("--out", default="outputs/da_hist_gdpzero_vs_emomcts_40s.png")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    gd_path = args.gdpzero if os.path.exists(args.gdpzero) else _find(args.gdpzero)
    em_path = args.emomcts if os.path.exists(args.emomcts) else _find(args.emomcts)
    gd, em = load(gd_path), load(em_path)
    Mg, Me = per_turn(gd, args.max_turn), per_turn(em, args.max_turn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    _stacked(axes[0], Mg, "GDP-Zero", args.max_turn)
    _stacked(axes[1], Me, r"EmoMCTS ($\beta$=0.7)", args.max_turn)
    axes[0].set_ylabel("fraction of dialogues")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(args.title, y=1.02, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["planner", "turn"] + ACTS)
            for name, M in [("gdpzero", Mg), ("emomcts", Me)]:
                for t in range(args.max_turn):
                    w.writerow([name, t] + [f"{M[t, j]:.4f}" for j in range(len(ACTS))])
        print("saved", args.csv)

    for name, M in [("GDPZero", Mg), ("EmoMCTS", Me)]:
        print(f"\n=== {name} (rows=turn, cols=acts) ===")
        print("turn  " + " ".join("%-8s" % a[:8] for a in ACTS))
        for t in range(args.max_turn):
            print("%2d   " % t + " ".join("%7.2f " % M[t, j] for j in range(len(ACTS))))


if __name__ == "__main__":
    main()
