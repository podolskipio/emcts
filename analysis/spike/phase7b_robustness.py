"""Robustness check (POST-HOC, clearly labelled): is the persona effect just response length?

Persona responses are materially longer than baseline in all three models. Length shifts the
emotion classifier independently of affect, so:
  (a) partial Spearman for H1-H3, controlling per-persona mean token count;
  (b) length-standardised P(negative) per arm: bin by pooled token deciles, compute the arm
      mean inside each bin, then average bins with equal weight so both arms are compared at
      the same length distribution.
Neither replaces the pre-registered tests; both are reported separately.
"""
import json, numpy as np
from scipy.stats import rankdata, spearmanr

ROOT="/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO=["anger","disgust","fear","happiness","neutral","sadness","surprise"]
NEG=["fear","sadness","anger","disgust"]
recs=[json.loads(l) for l in open(f"{ROOT}/spike/generations.jsonl")]
personas=json.load(open(f"{ROOT}/spike/personas_rendered.json"))
MODELS=["vicuna-13b-v1.5","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct"]
pid_ix={p["persona_id"]:i for i,p in enumerate(personas)}; NP_=len(personas)

def partial_spearman(x,y,z):
    rx,ry,rz=(rankdata(v) for v in (x,y,z))
    Z=np.c_[np.ones(len(rz)),rz]
    ex=rx-Z@np.linalg.lstsq(Z,rx,rcond=None)[0]
    ey=ry-Z@np.linalg.lstsq(Z,ry,rcond=None)[0]
    return spearmanr(ex,ey)

out={}
for m in MODELS:
    R=[r for r in recs if r["model"]==m]
    neg_p=np.zeros(NP_); hap_p=np.zeros(NP_); sur_p=np.zeros(NP_); tok_p=np.zeros(NP_); n=np.zeros(NP_)
    for r in R:
        if r["arm"]!="persona": continue
        i=pid_ix[r["persona_id"]]; e=r["emo_dist"]
        neg_p[i]+=sum(e[k] for k in NEG); hap_p[i]+=e["happiness"]; sur_p[i]+=e["surprise"]
        tok_p[i]+=r["token_count"]; n[i]+=1
    neg_p/=n; hap_p/=n; sur_p/=n; tok_p/=n
    tr={d:np.array([p[d] for p in personas]) for d in ("neurotic","agreeable","open")}
    res={}
    for name,x,y in (("H1_neurotic_neg",tr["neurotic"],neg_p),
                     ("H2_agreeable_happy",tr["agreeable"],hap_p),
                     ("H3_open_surprise",tr["open"],sur_p)):
        rho,p2=partial_spearman(x,y,tok_p)
        p1=p2/2 if rho>0 else 1-p2/2
        res[name]=dict(partial_rho=float(rho),p_one_sided=float(p1),sig=bool(p1<0.05 and rho>0))
    # length-standardised P(negative)
    toks=np.array([r["token_count"] for r in R],float)
    edges=np.unique(np.quantile(toks,np.linspace(0,1,11)))
    b_by_arm={}
    for arm in ("persona","baseline"):
        sub=[r for r in R if r["arm"]==arm]
        t=np.array([r["token_count"] for r in sub],float)
        g=np.array([sum(r["emo_dist"][k] for k in NEG) for r in sub])
        idx=np.clip(np.digitize(t,edges[1:-1]),0,len(edges)-2)
        b_by_arm[arm]=[(g[idx==b].mean() if (idx==b).sum() else np.nan) for b in range(len(edges)-1)]
    P=np.array(b_by_arm["persona"]); B=np.array(b_by_arm["baseline"])
    ok=~(np.isnan(P)|np.isnan(B))
    negs=np.array([sum(r["emo_dist"][k] for k in NEG) for r in R])
    arms=np.array([r["arm"] for r in R])
    # pooled-weighted direct standardisation (target = pooled length distribution)
    w=np.array([((np.digitize(np.array([r["token_count"] for r in R],float),edges[1:-1]).clip(0,len(edges)-2))==b).sum()
                for b in range(len(edges)-1)],float)
    ww=w[ok]/w[ok].sum()
    res["length_standardised_neg"]=dict(
        persona=float(P[ok].mean()),baseline=float(B[ok].mean()),n_bins=int(ok.sum()),
        persona_dstd=float((P[ok]*ww).sum()),baseline_dstd=float((B[ok]*ww).sum()),
        raw_ratio=float(np.mean(negs[arms=="persona"])/np.mean(negs[arms=="baseline"])),
        corr_tok_neg=float(spearmanr(toks,negs).statistic),
        corr_tok_neg_persona=float(spearmanr(toks[arms=="persona"],negs[arms=="persona"]).statistic),
        corr_tok_neg_baseline=float(spearmanr(toks[arms=="baseline"],negs[arms=="baseline"]).statistic),
        mean_tok_persona=float(toks[arms=="persona"].mean()),
        mean_tok_baseline=float(toks[arms=="baseline"].mean()))
    out[m]=res
    print(f"{m}")
    for k in ("H1_neurotic_neg","H2_agreeable_happy","H3_open_surprise"):
        c=res[k];print(f"   {k:<20} partial_rho={c['partial_rho']:+.3f} p={c['p_one_sided']:.4f} sig={c['sig']}")
    d=res["length_standardised_neg"]
    print(f"   tokens: persona={d['mean_tok_persona']:.1f} baseline={d['mean_tok_baseline']:.1f} "
          f"(+{100*(d['mean_tok_persona']/d['mean_tok_baseline']-1):.1f}%)")
    print(f"   rho(tok,neg) pooled={d['corr_tok_neg']:+.3f} | within-arm "
          f"persona={d['corr_tok_neg_persona']:+.3f} baseline={d['corr_tok_neg_baseline']:+.3f}")
    print(f"   P(neg) raw ratio={d['raw_ratio']:.2f}x  length-standardised="
          f"{d['persona_dstd']/d['baseline_dstd']:.2f}x")
json.dump(out,open(f"{ROOT}/spike/robustness.json","w"),indent=1)
