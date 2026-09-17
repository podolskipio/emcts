"""Recompute the spike's full verdict rule under each classifier.

Spike rule (spike/phase567_analysis.py): FAIL if permutation p >= .05; else
STRONG PASS if n_sig >= 2 and raw P(neg) persona > 1.5x baseline; else PASS if n_sig >= 2;
else WEAK. Only the emo_dist field varies.
"""
import json, numpy as np
ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]
CLFS = ["distilroberta","alt1_goemotions","alt1b_goemotions_max","alt2_emoberta"]
KEY = {"distilroberta":"emo_dist_distilroberta","alt1_goemotions":"emo_dist_alt1",
       "alt1b_goemotions_max":"emo_dist_alt1b_max","alt2_emoberta":"emo_dist_alt2"}
MODELS = ["vicuna-13b-v1.5","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct"]
NPERM = 1000
rng = np.random.default_rng(0)

recs = [json.loads(l) for l in open(f"{ROOT}/c4/relabelled.jsonl")]
personas = json.load(open(f"{ROOT}/spike/personas_rendered.json"))
UTTS = [u["utterance_id"] for u in json.load(open(f"{ROOT}/spike/system_utterances.json"))]
pid_ix = {p["persona_id"]: i for i, p in enumerate(personas)}
utt_ix = {u: i for i, u in enumerate(UTTS)}
NP_, NU, NS = len(personas), len(UTTS), 3

def H(P):
    P = np.clip(P, 1e-12, 1.0); return -(P * np.log2(P)).sum(-1)

def jsd_pairwise(M):
    A, B = M[:, None, :], M[None, :, :]
    D = H((A + B) / 2) - (H(A) + H(B)) / 2
    return float(D[np.triu_indices(len(M), 1)].mean())

phase4 = json.load(open(f"{ROOT}/c4/agreement.json"))["phase4"]
res = {}
for clf in CLFS:
    res[clf] = {}
    for model in MODELS:
        T = np.full((NP_, NU, NS, 7), np.nan)
        for r in recs:
            if r["model"] == model and r["arm"] == "persona":
                T[pid_ix[r["persona_id"]], utt_ix[r["utterance_id"]], r["seed"]] = \
                    [r[KEY[clf]][e] for e in EMO]
        assert not np.isnan(T).any()
        effect = float(np.mean([jsd_pairwise(T.mean(2)[:, u]) for u in range(NU)]))
        null = np.empty(NPERM)
        for k in range(NPERM):
            null[k] = np.mean([jsd_pairwise(T[:, u].reshape(NP_*NS, 7)
                               [rng.permutation(NP_*NS)].reshape(NP_, NS, 7).mean(1))
                               for u in range(NU)])
        p = float(np.mean(null >= effect))
        d = phase4[clf][model]
        nsig = d["directional"]["n_sig"]
        if p >= 0.05:      v = "FAIL"
        elif nsig >= 2:    v = "STRONG PASS" if d["raw_persona"] > 1.5*d["raw_baseline"] else "PASS"
        else:              v = "WEAK"
        res[clf][model] = dict(effect=effect, p_value=p, n_sig=nsig, verdict=v,
                               raw_ratio=d["raw_ratio"], std_ratio=d["std_ratio"],
                               std_ci=d["std_ci"])
        print(f"{clf:22s} {model:24s} p={p:.4f} nsig={nsig} raw={d['raw_ratio']:.2f}x "
              f"std={d['std_ratio']:.2f}x -> {v}", flush=True)
json.dump(res, open(f"{ROOT}/c4/verdicts.json","w"), indent=1)
print("\nwrote c4/verdicts.json")
