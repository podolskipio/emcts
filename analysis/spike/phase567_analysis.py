"""Phases 5-7: noise floor + permutation test, affective range, directional validity."""
import json, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO  = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]
NEG  = ["fear","sadness","anger","disgust"]
NPERM, NBOOT = 1000, 1000
rng = np.random.default_rng(0)

recs = [json.loads(l) for l in open(f"{ROOT}/spike/generations.jsonl")]
personas = json.load(open(f"{ROOT}/spike/personas_rendered.json"))
MODELS = ["vicuna-13b-v1.5","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct"]
UTTS   = [u["utterance_id"] for u in json.load(open(f"{ROOT}/spike/system_utterances.json"))]
PIDS   = [p["persona_id"] for p in personas]
pid_ix = {p: i for i, p in enumerate(PIDS)}
utt_ix = {u: i for i, u in enumerate(UTTS)}
NP_, NU, NS = len(PIDS), len(UTTS), 3

def H(P):
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log2(P)).sum(-1)

def jsd_pairwise(M):          # M: (n,7) -> mean pairwise JS divergence (base 2)
    A, B = M[:, None, :], M[None, :, :]
    D = H((A + B) / 2) - (H(A) + H(B)) / 2
    iu = np.triu_indices(len(M), 1)
    return float(D[iu].mean())

def tensor(model, arm):
    """(n_personas, n_utts, n_seeds, 7); NaN where a cell is missing."""
    T = np.full((NP_, NU, NS, 7), np.nan)
    for r in recs:
        if r["model"] != model or r["arm"] != arm: continue
        T[pid_ix[r["persona_id"]], utt_ix[r["utterance_id"]], r["seed"]] = \
            [r["emo_dist"][e] for e in EMO]
    return T

def toks(model, arm):
    return np.array([r["token_count"] for r in recs
                     if r["model"] == model and r["arm"] == arm], float)

report = {}
for model in MODELS:
    if not any(r["model"] == model for r in recs): continue
    P, B = tensor(model, "persona"), tensor(model, "baseline")
    assert not np.isnan(P).any() and not np.isnan(B).any(), f"missing cells for {model}"

    # ---------- PHASE 5: noise floor + effect + permutation ----------
    noise = float(np.mean([H((P[p,u,a]+P[p,u,b])/2) - (H(P[p,u,a])+H(P[p,u,b]))/2
                           for p in range(NP_) for u in range(NU)
                           for a, b in ((0,1),(0,2),(1,2))]))
    Pbar = P.mean(2)                                   # (persona, utt, 7)
    effect = float(np.mean([jsd_pairwise(Pbar[:, u]) for u in range(NU)]))
    Bbar = B.mean(2)
    baseline_between = float(np.mean([jsd_pairwise(Bbar[:, u]) for u in range(NU)]))

    null = np.empty(NPERM)
    flat = P.reshape(NP_, NU, NS, 7)
    for k in range(NPERM):                             # shuffle persona labels within utterance
        vals = []
        for u in range(NU):
            X = flat[:, u].reshape(NP_ * NS, 7)
            X = X[rng.permutation(NP_ * NS)].reshape(NP_, NS, 7).mean(1)
            vals.append(jsd_pairwise(X))
        null[k] = np.mean(vals)
    p_value = float(np.mean(null >= effect))

    plt.figure(figsize=(6,3.4))
    plt.hist(null, bins=40, color="#9bb8d3", edgecolor="white")
    plt.axvline(effect, color="#c0392b", lw=2, label=f"observed = {effect:.5f}")
    plt.axvline(noise,  color="#7f8c8d", lw=1.2, ls="--", label=f"noise floor = {noise:.5f}")
    plt.title(f"{model} — permutation null (n={NPERM})", fontsize=10)
    plt.xlabel("mean pairwise JSD between persona-mean emotion distributions")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{ROOT}/spike/null_{model}.png", dpi=140); plt.close()

    m = dict(noise=noise, effect=effect, ratio=effect/noise, p_value=p_value,
             baseline_between=baseline_between,
             gate="PASS" if p_value < 0.05 else "FAIL")

    # ---------- PHASE 6: affective range ----------
    for arm, T in (("persona", P), ("baseline", B)):
        flat_all = T.reshape(-1, 7)
        neg_per_persona = T[..., [EMO.index(e) for e in NEG]].sum(-1).reshape(NP_, -1).mean(1)
        m[arm] = dict(
            neg_fraction = float(flat_all[:, [EMO.index(e) for e in NEG]].sum(-1).mean()),
            entropy      = float(H(flat_all.mean(0))),
            between_persona_var = float(neg_per_persona.var(ddof=1)),
            neutral_fraction = float(flat_all[:, EMO.index("neutral")].mean()),
            mean_tokens  = float(toks(model, arm).mean()),
            mean_dist    = {e: float(v) for e, v in zip(EMO, flat_all.mean(0))})

    # ---------- PHASE 7: directional validity ----------
    negix = [EMO.index(e) for e in NEG]
    P_neg   = Pbar[..., negix].sum(-1).mean(1)                 # per persona, over 5 utts
    P_happy = Pbar[..., EMO.index("happiness")].mean(1)
    P_surp  = Pbar[..., EMO.index("surprise")].mean(1)
    trait = {d: np.array([p[d] for p in personas]) for d in ("neurotic","agreeable","open")}

    def corr(x, y):
        rho, p2 = spearmanr(x, y)
        p1 = p2 / 2 if rho > 0 else 1 - p2 / 2          # one-sided, predicted direction >0
        bs = []
        for _ in range(NBOOT):
            i = rng.integers(0, len(x), len(x))
            if len(np.unique(x[i])) < 3: continue
            bs.append(spearmanr(x[i], y[i]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(rho=float(rho), p_one_sided=float(p1),
                    ci=[float(lo), float(hi)], sig=bool(p1 < 0.05 and rho > 0))

    m["H1_neurotic_neg"]    = corr(trait["neurotic"],  P_neg)
    m["H2_agreeable_happy"] = corr(trait["agreeable"], P_happy)
    m["H3_open_surprise"]   = corr(trait["open"],      P_surp)

    nsig = sum(m[h]["sig"] for h in ("H1_neurotic_neg","H2_agreeable_happy","H3_open_surprise"))
    m["n_sig"] = nsig
    if p_value >= 0.05:                      verdict = "FAIL"
    elif nsig >= 2:
        verdict = "STRONG PASS" if m["persona"]["neg_fraction"] > 1.5*m["baseline"]["neg_fraction"] else "PASS"
    else:                                    verdict = "WEAK"
    m["verdict"] = verdict
    report[model] = m
    print(f"{model:<24} noise={noise:.5f} effect={effect:.5f} ratio={effect/noise:5.2f} "
          f"p={p_value:.4f} nsig={nsig} -> {verdict}", flush=True)

json.dump(report, open(f"{ROOT}/spike/analysis.json","w"), indent=1)
print("\nwrote spike/analysis.json")
