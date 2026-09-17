"""P-RELABEL (§10): does the paper's emotion-conditioned action finding survive a change of labeller?

    python analysis/phase1/scripts/p_relabel.py

Source runs (copied into analysis/phase1/relabel_src/ from an earlier session's scratchpad, md5
identical across all three copies found):
  EmoMCTS : rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims  (Ollama vicuna:13b, hf classifier, beta 0.7, 40 sims, 50 dlg)
  GDP-Zero: rollout_p4g_gdpzero_vicuna_50d_40s                 (Ollama vicuna:13b, 40 sims, 50 dlg; user turns HF-labelled post hoc)

Step 0 reproduces the published table (src/outputs/old/emotion_conditioned_actions_40s.csv) with the
ORIGINAL script's own tally() -- logged labels for EmoMCTS, the utterance cache for GDP-Zero -- and
stops if it does not match. Step 1 re-derives the same table from text with a fresh DistilRoBERTa run.
Steps 2-4 relabel with the two alternative classifiers from the C4 audit and recompute.
"""
import csv, json, os, pickle, sys
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
P1 = os.path.join(REPO, "analysis", "phase1")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import plot_emotion_conditioned_actions as orig  # the paper's pipeline, imported not re-typed

SRC = os.path.join(P1, "relabel_src")
EMO_RUN = "rollout_emo_p4g_multiobjq_beta07_50dialog_40_sims"
GDP_RUN = "rollout_p4g_gdpzero_vicuna_50d_40s"
PUBLISHED = os.path.join(REPO, "src", "outputs", "old", "emotion_conditioned_actions_40s.csv")
EMO7 = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
NEG = orig.NEG  # {fear, sadness, anger, disgust, contempt}; no classifier here emits contempt
GROUPS = orig.GROUPS
B = 1000
RNG = np.random.default_rng(20260914)

CLASSIFIERS = {
    # name: (model id, revision = refs/main snapshot in the local HF cache, head type)
    "distilroberta": ("j-hartmann/emotion-english-distilroberta-base", "0e1cd914e3d46199ed785853e12b57304e04178b", "softmax7"),
    "goemotions":    ("SamLowe/roberta-base-go_emotions", "d75048347613a25d77de8cf6412eaae9fa7b26be", "goemotions"),
    "emoberta":      ("tae898/emoberta-large", "8934b68e8b0d9fc3cd961cc7e7605533c7081e59", "softmax7"),
}
# verbatim from analysis/c4/phase2_relabel.py (google-research goemotions ekman_mapping.json + neutral)
GO2EK = {
    "anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
    "happiness": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief",
                  "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"], "neutral": ["neutral"],
}


def load_eps(run):
    with open(os.path.join(SRC, run, run + ".pkl"), "rb") as f:
        return pickle.load(f)


def classify(texts, name):
    mid, rev, head = CLASSIFIERS[name]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    clf = AutoModelForSequenceClassification.from_pretrained(mid, revision=rev).eval()
    labels = [clf.config.id2label[i] for i in range(clf.config.num_labels)]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=256)
            logits = clf(**enc).logits.float()
            if head == "goemotions":
                s = torch.sigmoid(logits)
                ix = {l: j for j, l in enumerate(labels)}
                g = torch.stack([s[:, [ix[x] for x in GO2EK[e]]].sum(1) for e in EMO7], 1)
                p = g / g.sum(1, keepdim=True)
                out += [{e: float(v) for e, v in zip(EMO7, row)} for row in p]
            else:
                p = logits.softmax(-1)
                names = ["happiness" if l == "joy" else l for l in labels]
                assert sorted(names) == sorted(EMO7), names
                ix = {l: j for j, l in enumerate(names)}
                out += [{e: float(row[ix[e]]) for e in EMO7} for row in p]
    return out


def system_turn_rows(eps, label_of):
    """One row per system turn that has a preceding user turn: (dialogue idx, bucket, group).
    Same walk as orig.tally: the opening greeting has no preceding user emotion and is skipped;
    last_emo is the label of the most recent persuadee turn."""
    rows = []
    for d, e in enumerate(eps):
        last = None
        for turn in e["history"]:
            if turn[0] == "Persuader":
                if last is not None:
                    rows.append((d, "negative" if last in NEG else "non-negative", orig._group(turn[1])))
            else:
                last = label_of(turn[-1])
    return rows


def table(rows, n_dlg, idx=None):
    """counts[bucket][group] from rows restricted to dialogue multiset idx (bootstrap)."""
    by_d = defaultdict(list)
    for d, b, g in rows:
        by_d[d].append((b, g))
    c = {"negative": Counter(), "non-negative": Counter()}
    for d in (range(n_dlg) if idx is None else idx):
        for b, g in by_d.get(d, []):
            c[b][g] += 1
    return c


def props(c):
    out = {}
    for b in ("negative", "non-negative"):
        n = sum(c[b].values())
        out[b] = {"n": n, **{g: (c[b][g] / n if n else float("nan")) for g in GROUPS}}
    return out


def contrasts(pe, pg):
    """EmoMCTS - GDP-Zero per bucket x group, plus the difference-in-differences (neg - non-neg)."""
    out = {}
    for g in GROUPS:
        dn = pe["negative"][g] - pg["negative"][g]
        dp = pe["non-negative"][g] - pg["non-negative"][g]
        out[f"{g}|negative"] = dn
        out[f"{g}|non-negative"] = dp
        out[f"{g}|DiD"] = dn - dp
    return out


def bootstrap(rows_e, rows_g, n_e, n_g):
    reps = defaultdict(list)
    empty = 0
    for _ in range(B):
        ie = RNG.integers(0, n_e, n_e)
        ig = RNG.integers(0, n_g, n_g)
        pe, pg = props(table(rows_e, n_e, ie)), props(table(rows_g, n_g, ig))
        if pe["negative"]["n"] == 0 or pg["negative"]["n"] == 0:
            empty += 1
            continue
        for k, v in contrasts(pe, pg).items():
            reps[k].append(v)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()}
    return ci, empty


def kappa(a, b, cats):
    ix = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[ix[x], ix[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(1) @ m.sum(0)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan"), m.astype(int).tolist()


def main():
    eps_e, eps_g = load_eps(EMO_RUN), load_eps(GDP_RUN)

    # ---- step 0: exact reproduction with the paper's own tally ----
    cache = orig._HFCache(os.path.join(SRC, "gdpzero_user_emocache.json"))
    ce = orig.tally(eps_e, has_emotion=True)
    cg = orig.tally(eps_g, has_emotion=False, hf=cache)
    pub = {r["bar"]: r for r in csv.DictReader(open(PUBLISHED))}
    repro = {"GDP-Zero (after neg.)": cg["negative"], "EmoMCTS (after neg.)": ce["negative"],
             "GDP-Zero (after non-neg.)": cg["non-negative"], "EmoMCTS (after non-neg.)": ce["non-negative"]}
    repro_ok = True
    repro_rows = []
    for bar, c in repro.items():
        n = sum(c.values())
        fr = [c[g] / n for g in GROUPS]
        match = n == int(pub[bar]["n"]) and all(f"{x:.4f}" == pub[bar][g] for x, g in zip(fr, GROUPS))
        repro_ok &= match
        repro_rows.append({"bar": bar, "n": n, **{g: round(x, 4) for x, g in zip(fr, GROUPS)},
                           "published_n": int(pub[bar]["n"]), "match": match})
    print("step 0 reproduction:", "MATCH" if repro_ok else "MISMATCH")
    for r in repro_rows:
        print("  ", r)
    if not repro_ok:
        json.dump({"reproduced": False, "rows": repro_rows}, open(os.path.join(P1, "p_relabel.json"), "w"), indent=1)
        sys.exit("§IV-D does not reproduce with the original pipeline -- stopping per §12.")

    # ---- steps 1-2: label every user utterance of both runs with all three classifiers ----
    utts = sorted({t[-1] for eps in (eps_e, eps_g) for e in eps for t in e["history"] if t[0] == "Persuadee"})
    nonempty = [u for u in utts if (u or "").strip()]
    dists = {name: dict(zip(nonempty, classify(nonempty, name))) for name in CLASSIFIERS}
    label = {name: {u: max(d, key=d.get) for u, d in dists[name].items()} for name in CLASSIFIERS}
    for name in CLASSIFIERS:  # orig._HFCache convention: empty utterance -> neutral
        for u in utts:
            if not (u or "").strip():
                label[name][u] = "neutral"

    # fresh DistilRoBERTa vs what the paper used (logged labels / cache)
    logged_e = [(orig._emo_str(t[2]), label["distilroberta"][t[-1]]) for e in eps_e for t in e["history"] if t[0] == "Persuadee"]
    cached_g = [(cache.cache.get((t[-1] or "").strip(), "neutral") if (t[-1] or "").strip() else "neutral",
                 label["distilroberta"][t[-1]]) for e in eps_g for t in e["history"] if t[0] == "Persuadee"]
    agree_e = float(np.mean([a == b for a, b in logged_e]))
    agree_g = float(np.mean([a == b for a, b in cached_g]))
    print(f"fresh DistilRoBERTa vs logged labels (EmoMCTS): {agree_e:.4f}; vs cache (GDP-Zero): {agree_g:.4f}")

    # ---- step 3: the conditional table under each classifier ----
    panels = {}
    for name in CLASSIFIERS:
        lab = label[name].__getitem__
        re_, rg = system_turn_rows(eps_e, lab), system_turn_rows(eps_g, lab)
        pe, pg = props(table(re_, len(eps_e))), props(table(rg, len(eps_g)))
        ci, empty = bootstrap(re_, rg, len(eps_e), len(eps_g))
        user_labels_e = [lab(t[-1]) for e in eps_e for t in e["history"] if t[0] == "Persuadee"]
        user_labels_g = [lab(t[-1]) for e in eps_g for t in e["history"] if t[0] == "Persuadee"]
        panels[name] = {
            "model": CLASSIFIERS[name][0], "revision": CLASSIFIERS[name][1],
            "EmoMCTS": pe, "GDP-Zero": pg,
            "contrasts": contrasts(pe, pg), "contrast_ci95": ci, "bootstrap_reps_dropped_empty_neg_cell": empty,
            "neg_base_rate_user_turns": {"EmoMCTS": float(np.mean([l in NEG for l in user_labels_e])),
                                         "GDP-Zero": float(np.mean([l in NEG for l in user_labels_g])),
                                         "both": float(np.mean([l in NEG for l in user_labels_e + user_labels_g]))},
            "neg_rate_conditioning_system_turns": {"EmoMCTS": pe["negative"]["n"] / (pe["negative"]["n"] + pe["non-negative"]["n"]),
                                                   "GDP-Zero": pg["negative"]["n"] / (pg["negative"]["n"] + pg["non-negative"]["n"])},
            "label_distribution_nonempty": dict(Counter(label[name][u] for eps in (eps_e, eps_g) for e in eps
                                                         for t in e["history"] if t[0] == "Persuadee" and (t[-1] or "").strip() for u in [t[-1]])),
        }

    # ---- step 4: agreement on this exact utterance set (per occurrence, non-empty) ----
    occ = [t[-1] for eps in (eps_e, eps_g) for e in eps for t in e["history"] if t[0] == "Persuadee" and (t[-1] or "").strip()]
    agreement = {}
    names = list(CLASSIFIERS)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = [label[names[i]][u] for u in occ], [label[names[j]][u] for u in occ]
            k7, m7 = kappa(a, b, EMO7)
            kn, mn = kappa(["neg" if x in NEG else "non" for x in a], ["neg" if x in NEG else "non" for x in b], ["neg", "non"])
            agreement[f"{names[i]}|{names[j]}"] = {"kappa_7class": k7, "confusion_7class_rows_first": m7, "labels": EMO7,
                                                   "top1_agreement": float(np.mean([x == y for x, y in zip(a, b)])),
                                                   "kappa_negative_binary": kn, "confusion_negative_binary": mn,
                                                   "pneg_pearson_r": float(np.corrcoef([sum(dists[names[i]][u][e] for e in NEG if e in EMO7) for u in occ],
                                                                                      [sum(dists[names[j]][u][e] for e in NEG if e in EMO7) for u in occ])[0, 1])}

    result = {"reproduced": True, "reproduction_rows": repro_rows,
              "fresh_distilroberta_agreement": {"EmoMCTS_logged": agree_e, "GDP-Zero_cache": agree_g},
              "n_user_occurrences_nonempty": len(occ), "n_unique_utterances": len(utts),
              "n_dialogues": {"EmoMCTS": len(eps_e), "GDP-Zero": len(eps_g)},
              "bootstrap": {"replicates": B, "cluster": "dialogue, resampled within each planner's run", "interval": "percentile 95%"},
              "panels": panels, "agreement": agreement}
    json.dump(result, open(os.path.join(P1, "p_relabel.json"), "w"), indent=1)
    for name, p in panels.items():
        print(f"\n== {name} ==  neg base rate (user turns, both runs) {p['neg_base_rate_user_turns']['both']:.3f}")
        for planner in ("GDP-Zero", "EmoMCTS"):
            for b in ("negative", "non-negative"):
                r = p[planner][b]
                print(f"  {planner:8s} after {b:12s} n={r['n']:3d}  " + "  ".join(f"{g}={r[g]:.3f}" for g in GROUPS))
        for k in ("trust-building|negative", "other|negative", "trust-building|DiD"):
            print(f"  diff {k:26s} {p['contrasts'][k]:+.3f}  CI {p['contrast_ci95'][k][0]:+.3f}..{p['contrast_ci95'][k][1]:+.3f}")
    for k, v in agreement.items():
        print(f"{k}: kappa7={v['kappa_7class']:.3f} kappa_neg={v['kappa_negative_binary']:.3f} top1={v['top1_agreement']:.3f} pneg_r={v['pneg_pearson_r']:.3f}")


if __name__ == "__main__":
    main()
