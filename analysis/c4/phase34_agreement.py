"""C4 Phases 3+4 - classifier agreement, and recomputation of the spike verdicts.

Phase 3: top-1 accuracy + Cohen's kappa, mean JSD, P(neg) Pearson/Spearman, confusion
         matrices, neutral fraction, MICA-protocol cosine of score-change vectors.
Phase 4: raw and length-standardised P(neg) ratios with bootstrap CIs, and the H1-H3
         directional correlations, recomputed under each classifier.

Analysis choices are held fixed to spike/phase567_analysis.py and spike/phase7b_robustness.py;
only the `emo_dist` field varies.
"""
import json, numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO  = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
NEG  = ["fear", "sadness", "anger", "disgust"]
NEGIX = [EMO.index(e) for e in NEG]
CLFS = [("distilroberta", "emo_dist_distilroberta"),
        ("alt1_goemotions", "emo_dist_alt1"),
        ("alt1b_goemotions_max", "emo_dist_alt1b_max"),
        ("alt2_emoberta", "emo_dist_alt2")]
MODELS = ["vicuna-13b-v1.5", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]
NBOOT = 2000
rng = np.random.default_rng(0)

recs = [json.loads(l) for l in open(f"{ROOT}/c4/relabelled.jsonl")]
personas = json.load(open(f"{ROOT}/spike/personas_rendered.json"))
pid_ix = {p["persona_id"]: i for i, p in enumerate(personas)}
NP_ = len(personas)

D = {name: np.array([[r[key][e] for e in EMO] for r in recs]) for name, key in CLFS}
mdl  = np.array([r["model"] for r in recs])
arm  = np.array([r["arm"] for r in recs])
tok  = np.array([r["token_count"] for r in recs], float)
pid  = np.array([pid_ix[r["persona_id"]] for r in recs])
NEGM = {n: M[:, NEGIX].sum(1) for n, M in D.items()}
TOP1 = {n: M.argmax(1) for n, M in D.items()}
out = {}

# ---------------- helpers ----------------
def H(P):
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log2(P)).sum(-1)

def jsd(A, B):
    return H((A + B) / 2) - (H(A) + H(B)) / 2

def kappa(a, b, k=7):
    n = len(a)
    C = np.zeros((k, k))
    np.add.at(C, (a, b), 1)
    po = np.trace(C) / n
    pe = (C.sum(1) @ C.sum(0)) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

PAIRS = [("distilroberta", "alt1_goemotions"), ("distilroberta", "alt2_emoberta"),
         ("alt1_goemotions", "alt2_emoberta"),
         ("distilroberta", "alt1b_goemotions_max")]

def agree_block(mask):
    b = {}
    for x, y in PAIRS:
        ax, ay = TOP1[x][mask], TOP1[y][mask]
        b[f"{x}|{y}"] = dict(
            n=int(mask.sum()),
            top1_acc=float((ax == ay).mean()),
            kappa=kappa(ax, ay),
            mean_jsd=float(jsd(D[x][mask], D[y][mask]).mean()),
            neg_pearson=float(pearsonr(NEGM[x][mask], NEGM[y][mask]).statistic),
            neg_spearman=float(spearmanr(NEGM[x][mask], NEGM[y][mask]).statistic))
    return b

# ---------------- PHASE 3 ----------------
ALL = np.ones(len(recs), bool)
out["agreement_overall"] = agree_block(ALL)
out["agreement_by_model"] = {m: agree_block(mdl == m) for m in MODELS}
out["agreement_by_arm"] = {a: agree_block(arm == a) for a in ("persona", "baseline")}
out["agreement_by_model_arm"] = {f"{m}|{a}": agree_block((mdl == m) & (arm == a))
                                 for m in MODELS for a in ("persona", "baseline")}

# confusion matrices, distilroberta (rows) vs each alternative (cols)
out["confusion"] = {}
for alt in ("alt1_goemotions", "alt1b_goemotions_max", "alt2_emoberta"):
    C = np.zeros((7, 7), int)
    np.add.at(C, (TOP1["distilroberta"], TOP1[alt]), 1)
    out["confusion"][alt] = dict(labels=EMO, matrix=C.tolist())

# neutral fraction (mean mass and top-1 rate) per classifier, per arm, per model
out["neutral"] = {}
ni = EMO.index("neutral")
for n, M in D.items():
    e = {"overall": dict(mass=float(M[:, ni].mean()), top1=float((TOP1[n] == ni).mean()))}
    for a in ("persona", "baseline"):
        k = arm == a
        e[a] = dict(mass=float(M[k, ni].mean()), top1=float((TOP1[n] == ni)[k].mean()))
    for m in MODELS:
        for a in ("persona", "baseline"):
            k = (mdl == m) & (arm == a)
            e[f"{m}|{a}"] = dict(mass=float(M[k, ni].mean()), top1=float((TOP1[n] == ni)[k].mean()))
    out["neutral"][n] = e

# MICA protocol: persona-minus-baseline change vector per model, cosine across classifiers
chg = {n: {m: D[n][(mdl == m) & (arm == "persona")].mean(0)
                - D[n][(mdl == m) & (arm == "baseline")].mean(0) for m in MODELS}
       for n in D}
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
out["mica_cosine"] = {
    "per_model": {m: {f"{x}|{y}": cos(chg[x][m], chg[y][m]) for x, y in PAIRS} for m in MODELS},
    "pooled_21d": {f"{x}|{y}": cos(np.concatenate([chg[x][m] for m in MODELS]),
                                   np.concatenate([chg[y][m] for m in MODELS])) for x, y in PAIRS},
    "change_vectors": {n: {m: dict(zip(EMO, chg[n][m].round(5).tolist())) for m in MODELS} for n in chg}}

# ---------------- PHASE 4 ----------------
def std_ratio(neg, t, a, edges=None):
    """Length-standardised P(neg) ratio: pooled token deciles, arm mean per bin,
    direct standardisation to the pooled length distribution (spike/phase7b §8b)."""
    if edges is None:
        edges = np.unique(np.quantile(t, np.linspace(0, 1, 11)))
    nb = len(edges) - 1
    idx = np.clip(np.digitize(t, edges[1:-1]), 0, nb - 1)
    w = np.array([(idx == b).sum() for b in range(nb)], float)
    P, B = [], []
    for b in range(nb):
        p, q = (idx == b) & (a == "persona"), (idx == b) & (a == "baseline")
        P.append(neg[p].mean() if p.sum() else np.nan)
        B.append(neg[q].mean() if q.sum() else np.nan)
    P, B = np.array(P), np.array(B)
    ok = ~(np.isnan(P) | np.isnan(B))
    ww = w[ok] / w[ok].sum()
    pd_, bd_ = (P[ok] * ww).sum(), (B[ok] * ww).sum()
    return pd_, bd_, pd_ / bd_, edges, int(ok.sum())

out["phase4"] = {}
for n in D:
    out["phase4"][n] = {}
    for m in MODELS:
        k = mdl == m
        neg, t, a = NEGM[n][k], tok[k], arm[k]
        rp, rb = neg[a == "persona"].mean(), neg[a == "baseline"].mean()
        pd_, bd_, sr, edges, nb = std_ratio(neg, t, a)
        # bootstrap: resample records within arm, bin edges held at the observed deciles
        ip, ib = np.flatnonzero(a == "persona"), np.flatnonzero(a == "baseline")
        bs = []
        for _ in range(NBOOT):
            s = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ib, len(ib), True)])
            bs.append(std_ratio(neg[s], t[s], a[s], edges)[2])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["phase4"][n][m] = dict(
            raw_persona=float(rp), raw_baseline=float(rb), raw_ratio=float(rp / rb),
            std_persona=float(pd_), std_baseline=float(bd_), std_ratio=float(sr),
            std_ci=[float(lo), float(hi)], boot_median=float(np.median(bs)), n_bins=nb,
            mean_tok_persona=float(t[a == "persona"].mean()),
            mean_tok_baseline=float(t[a == "baseline"].mean()))

        # directional validity H1-H3, persona arm, per-persona means (spike Phase 7)
        pa = k & (arm == "persona")
        per = np.array([D[n][pa & (pid == i)].mean(0) for i in range(NP_)])
        tr = {d: np.array([p[d] for p in personas]) for d in ("neurotic", "agreeable", "open")}
        def corr(x, y):
            rho, p2 = spearmanr(x, y)
            p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
            b = [spearmanr(x[i], y[i]).statistic
                 for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
            lo, hi = np.percentile(b, [2.5, 97.5])
            return dict(rho=float(rho), p_one_sided=float(p1), ci=[float(lo), float(hi)],
                        sig=bool(p1 < 0.05 and rho > 0))
        h = dict(H1_neurotic_neg=corr(tr["neurotic"], per[:, NEGIX].sum(1)),
                 H2_agreeable_happy=corr(tr["agreeable"], per[:, EMO.index("happiness")]),
                 H3_open_surprise=corr(tr["open"], per[:, EMO.index("surprise")]))
        h["n_sig"] = int(sum(v["sig"] for v in h.values() if isinstance(v, dict)))
        out["phase4"][n][m]["directional"] = h
        print(f"{n:16s} {m:24s} raw={rp/rb:.2f}x std={sr:.2f}x "
              f"[{lo:.2f},{hi:.2f}] nsig={h['n_sig']}", flush=True)

json.dump(out, open(f"{ROOT}/c4/agreement.json", "w"), indent=1)
print("\nwrote c4/agreement.json")
