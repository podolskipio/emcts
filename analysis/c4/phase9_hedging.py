"""C4 Phase 9 - is DistilRoBERTa's `fear` label a hedging detector?

Pull every utterance DistilRoBERTa labelled fear and measure the fraction containing
hedging markers. Reported against the corpus base rate: in a donation-refusal corpus
hedging is common everywhere, so a raw fraction means nothing without the base rate and
the per-label comparison. Also checks whether fear-labelled text contains actual fear
vocabulary, which is the competing explanation.

No GPU. Reads c4/relabelled.jsonl only.
"""
import json, re, numpy as np
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]

# --- marker lists, fixed before looking at results -----------------------------------
# The four the analysis was asked for, plus their obvious morphological variants and the
# same epistemic-hedge family. Kept to deliberation/uncertainty markers; no affect words.
HEDGE = {
 "hesitant":       r"\bhesitan|\bhesitate",
 "not sure":       r"\bnot sure\b|\bunsure\b|\bnot certain\b|\bnot entirely sure\b",
 "maybe":          r"\bmaybe\b|\bperhaps\b|\bpossibly\b",
 "know more":      r"\bknow more\b|\bmore about\b|\bmore information\b|\bmore info\b|"
                   r"\blearn more\b|\bhear more\b|\bmore details\b|\btell me more\b",
 "would need to":  r"\b(?:i'?d|i would|i'?ll) (?:need|have) to\b|\bneed to know\b|"
                   r"\bwould want to know\b|\bneed more\b",
 "think about it": r"\bthink about it\b|\bthinking about it\b|\bconsider it\b|"
                   r"\bhave to think\b",
 "do research":    r"\bdo (?:my|some) research\b|\bresearch first\b|\blook into\b|"
                   r"\bdo my due diligence\b",
 "might/guess":    r"\bi might\b|\bi guess\b|\bi suppose\b|\bprobably\b|\bit depends\b",
 "a bit/a little": r"\ba bit\b|\ba little\b|\bsomewhat\b|\bkind of\b|\bsort of\b",
 "reluctant":      r"\breluctan|\bcautious\b|\bwary\b|\bskeptical\b|\bsceptical\b",
}
# Competing explanation: the text really is fearful.
FEARWORD = (r"\bafraid\b|\bscared\b|\bworried\b|\bworry\b|\banxious\b|\banxiety\b|"
            r"\bterrified\b|\bfrightened\b|\bnervous\b|\bfearful\b|\bpanic\b|\bdread\b")

recs = [json.loads(l) for l in open(f"{ROOT}/c4/relabelled.jsonl")]
txt  = [r["response_text"].lower() for r in recs]
D    = np.array([[r["emo_dist_distilroberta"][e] for e in EMO] for r in recs])
A1   = np.array([[r["emo_dist_alt1"][e] for e in EMO] for r in recs])
A2   = np.array([[r["emo_dist_alt2"][e] for e in EMO] for r in recs])
top1 = D.argmax(1)
FI   = EMO.index("fear")

hit  = {k: np.array([bool(re.search(p, t)) for t in txt]) for k, p in HEDGE.items()}
anyh = np.any(list(hit.values()), 0)
nh   = np.sum(list(hit.values()), 0)                 # number of distinct marker families
fw   = np.array([bool(re.search(FEARWORD, t)) for t in txt])

out = {}
N = len(recs)
print(f"n = {N} utterances\n")

# ---------- 1. the headline: fear-labelled vs base rate ----------
fear = top1 == FI
print("=== 1. Hedging in DistilRoBERTa `fear` utterances vs corpus base rate ===")
print(f"  fear-labelled (top-1):        n={fear.sum():5d}  hedged={anyh[fear].mean():.1%}")
print(f"  corpus base rate (all):       n={N:5d}  hedged={anyh.mean():.1%}")
print(f"  non-fear-labelled:            n={(~fear).sum():5d}  hedged={anyh[~fear].mean():.1%}")
odds, p = fisher_exact([[int(anyh[fear].sum()), int((~anyh[fear]).sum())],
                        [int(anyh[~fear].sum()), int((~anyh[~fear]).sum())]])
print(f"  enrichment vs non-fear: odds ratio={odds:.2f}  Fisher p={p:.3e}")
out["headline"] = dict(n_fear=int(fear.sum()), hedged_fear=float(anyh[fear].mean()),
                       base_rate=float(anyh.mean()), hedged_nonfear=float(anyh[~fear].mean()),
                       odds_ratio=float(odds), fisher_p=float(p))

# ---------- 2. control: hedge rate by DistilRoBERTa top-1 label ----------
print("\n=== 2. Control - hedge rate for EVERY DistilRoBERTa label ===")
print(f"  {'label':<12}{'n':>6}{'hedged':>9}{'mean #markers':>15}{'fear-word':>11}")
out["by_label"] = {}
for i, e in enumerate(EMO):
    k = top1 == i
    if not k.sum(): continue
    print(f"  {e:<12}{k.sum():6d}{anyh[k].mean():9.1%}{nh[k].mean():15.2f}{fw[k].mean():11.1%}")
    out["by_label"][e] = dict(n=int(k.sum()), hedged=float(anyh[k].mean()),
                              mean_markers=float(nh[k].mean()), fear_word=float(fw[k].mean()))

# ---------- 3. which markers drive it ----------
print("\n=== 3. Marker breakdown inside fear-labelled utterances ===")
print(f"  {'marker':<16}{'in fear':>10}{'in corpus':>11}{'lift':>8}")
out["markers"] = {}
for k in HEDGE:
    a, b = hit[k][fear].mean(), hit[k].mean()
    print(f"  {k:<16}{a:10.1%}{b:11.1%}{(a/b if b else float('nan')):8.2f}x")
    out["markers"][k] = dict(in_fear=float(a), in_corpus=float(b), lift=float(a/b) if b else None)

# ---------- 4. the competing explanation: is it actually fear? ----------
print("\n=== 4. Competing explanation - does fear-labelled text contain fear vocabulary? ===")
print(f"  fear-labelled containing a fear word: {fw[fear].mean():.1%} ({fw[fear].sum()}/{fear.sum()})")
print(f"  corpus base rate for fear words:      {fw.mean():.1%}")
print(f"  fear-labelled that are hedged but have NO fear word: "
      f"{(anyh & ~fw)[fear].mean():.1%}")
print(f"  fear-labelled with NEITHER hedge nor fear word:      {(~anyh & ~fw)[fear].mean():.1%}")
out["fear_vocab"] = dict(fear_word_in_fear=float(fw[fear].mean()),
                         fear_word_base=float(fw.mean()),
                         hedged_no_fearword=float((anyh & ~fw)[fear].mean()),
                         neither=float((~anyh & ~fw)[fear].mean()))

# ---------- 5. dose-response on the continuous quantity Q_emo consumes ----------
print("\n=== 5. Dose-response: fear MASS vs number of hedge markers (all 8,820) ===")
rho, pv = spearmanr(nh, D[:, FI])
print(f"  Spearman rho(#hedge markers, P(fear)) = {rho:+.3f}  p={pv:.2e}")
print(f"  mean P(fear): hedged={D[anyh, FI].mean():.4f}  unhedged={D[~anyh, FI].mean():.4f}  "
      f"ratio={D[anyh, FI].mean()/D[~anyh, FI].mean():.2f}x")
u, pu = mannwhitneyu(D[anyh, FI], D[~anyh, FI], alternative="greater")
print(f"  Mann-Whitney p={pu:.2e}")
print(f"  by marker count: " + "  ".join(
    f"{c}:{D[nh==c, FI].mean():.4f}(n={int((nh==c).sum())})" for c in range(0, 5)))
out["dose"] = dict(rho=float(rho), p=float(pv),
                   mean_fear_hedged=float(D[anyh, FI].mean()),
                   mean_fear_unhedged=float(D[~anyh, FI].mean()),
                   mw_p=float(pu),
                   by_count={int(c): float(D[nh==c, FI].mean()) for c in range(0,5) if (nh==c).sum()})

# ---------- 6. do the alternatives do the same thing? ----------
print("\n=== 6. Do the alternative classifiers show the same hedge->fear coupling? ===")
for nm, M in (("distilroberta", D), ("alt1_goemotions", A1), ("alt2_emoberta", A2)):
    r_f, _ = spearmanr(nh, M[:, FI])
    neg = M[:, [EMO.index(e) for e in ("fear","sadness","anger","disgust")]].sum(1)
    r_n, _ = spearmanr(nh, neg)
    print(f"  {nm:18s} rho(hedges,P(fear))={r_f:+.3f}   rho(hedges,P(neg))={r_n:+.3f}   "
          f"P(neg) hedged/unhedged={neg[anyh].mean()/neg[~anyh].mean():.2f}x")
    out.setdefault("cross_clf", {})[nm] = dict(rho_fear=float(r_f), rho_neg=float(r_n),
        neg_ratio=float(neg[anyh].mean()/neg[~anyh].mean()))

# ---------- 7. does hedging drive the persona-arm gap? ----------
print("\n=== 7. Is the persona/baseline hedging rate different? (would confound the ratio) ===")
arm = np.array([r["arm"] for r in recs]); mdl = np.array([r["model"] for r in recs])
for m in ["vicuna-13b-v1.5","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct"]:
    pm, bm = (mdl==m)&(arm=="persona"), (mdl==m)&(arm=="baseline")
    print(f"  {m:24s} hedged persona={anyh[pm].mean():.1%} baseline={anyh[bm].mean():.1%} "
          f"| markers {nh[pm].mean():.2f}/{nh[bm].mean():.2f}")
    out.setdefault("by_arm", {})[m] = dict(hedged_persona=float(anyh[pm].mean()),
        hedged_baseline=float(anyh[bm].mean()),
        markers_persona=float(nh[pm].mean()), markers_baseline=float(nh[bm].mean()))

# ---------- 8. samples ----------
out["examples"] = dict(
    fear_hedged_no_fearword=[recs[i]["response_text"] for i in np.flatnonzero(fear & anyh & ~fw)[:15]],
    fear_unhedged=[recs[i]["response_text"] for i in np.flatnonzero(fear & ~anyh)[:15]])
print("\n=== 8. fear-labelled, hedged, NO fear vocabulary (first 8) ===")
for t in out["examples"]["fear_hedged_no_fearword"][:8]: print(f"   - {t[:130]}")
print("\n=== 8b. fear-labelled, NO hedge marker (first 8) ===")
for t in out["examples"]["fear_unhedged"][:8]: print(f"   - {t[:130]}")

json.dump(out, open(f"{ROOT}/c4/hedging.json","w"), indent=1)
print("\nwrote c4/hedging.json")
