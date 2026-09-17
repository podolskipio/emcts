"""PHASE 1 gate: has the harness drifted since the spike ran?

Runs the check the task specifies -- regenerate records with the *AWQ* Llama checkpoint
through the bf16 harness and compare to spike/generations.jsonl for the same
(persona_id, utterance_id, seed, arm) -- and reports the exact-match rate.

It also runs the check that can actually discriminate drift. spike/gen_lib.one() sends only
{model, messages, temperature, top_p, max_tokens}; job["seed"] is a replicate index and never
reaches the sampler, so at temperature 0.7 with no per-request seed two runs of the same cell
are independent draws and exact match is not expected even with a byte-identical harness
(spike/phase8_report.py note 4 documents this). Distributional agreement between a fresh AWQ
replicate and the recorded AWQ run is the available drift signal.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_lib_bf16 as B
import serve_lib as S

ROOT = B.ROOT
KEY = "Llama-3.1-8B-Instruct"
AWQ = B.MODELS[KEY]["awq_path"]
N_EXACT, N_DIST = 20, 300
EMO = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
NEG = ["fear", "sadness", "anger", "disgust"]

jobs, personas, utts = B.build_jobs()
rec_by_key = {}
for line in open(f"{ROOT}/spike/generations.jsonl"):
    r = json.loads(line)
    if r["model"] == KEY:
        rec_by_key[(r["persona_id"], r["utterance_id"], r["seed"], r["arm"])] = r
print(f"recorded AWQ {KEY} records: {len(rec_by_key)}")

rng = np.random.default_rng(0)
dist_ix = sorted(rng.choice(len(jobs), N_DIST, replace=False).tolist())
need = sorted(set(range(N_EXACT)) | set(dist_ix))
print(f"regenerating {len(need)} cells with AWQ checkpoint {AWQ}")

S.preflight()
proc = S.launch(AWQ, f"{HERE}/server_verify_awq.log", dtype=None, mem_fraction=0.82)
try:
    S.wait_up(proc, KEY, f"{HERE}/server_verify_awq.log")
    S.assert_identity(AWQ)
    t0 = time.time()
    out = B.run_all([jobs[i] for i in need], AWQ, workers=10)
    print(f"  regenerated {len(out)} in {time.time()-t0:.0f}s", flush=True)
finally:
    S.teardown(proc, KEY)

fresh = {}
for i, (txt, n) in zip(need, out):
    j = jobs[i]
    fresh[(j["persona_id"], j["utterance_id"], j["seed"], j["arm"])] = {
        "response_text": txt.strip(), "token_count": n}

# ---- classify the fresh replicate with the same checkpoint the spike used ----
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
CLF = "j-hartmann/emotion-english-distilroberta-base"
tok = AutoTokenizer.from_pretrained(CLF)
clf = AutoModelForSequenceClassification.from_pretrained(CLF)
dev = "cuda" if torch.cuda.is_available() else "cpu"
clf.to(dev).eval()
RENAME = {"joy": "happiness"}
labels = [RENAME.get(clf.config.id2label[i], clf.config.id2label[i])
          for i in range(clf.config.num_labels)]
keys = list(fresh)
with torch.no_grad():
    for i in range(0, len(keys), 128):
        chunk = keys[i:i+128]
        enc = tok([fresh[k]["response_text"] for k in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(dev)
        for k, p in zip(chunk, clf(**enc).logits.softmax(-1).cpu().tolist()):
            fresh[k]["emo_dist"] = {l: float(v) for l, v in zip(labels, p)}

# ---- gate A: exact match on the first N_EXACT cells (as specified) ----
exact_rows = []
for i in range(N_EXACT):
    j = jobs[i]
    k = (j["persona_id"], j["utterance_id"], j["seed"], j["arm"])
    exact_rows.append({"key": list(k), "recorded": rec_by_key[k]["response_text"],
                       "fresh": fresh[k]["response_text"],
                       "match": rec_by_key[k]["response_text"] == fresh[k]["response_text"]})
n_exact_match = sum(r["match"] for r in exact_rows)

# ---- gate B: distributional agreement on the N_DIST cells ----
def stats(getter, keys_):
    return float(np.mean([getter(x) for x in keys_]))

rec_sub = [rec_by_key[k] for k in fresh]
fre_sub = [fresh[k] for k in fresh]
def neg(r): return sum(r["emo_dist"][e] for e in NEG)
summary = {}
for name, S_ in (("recorded_awq", rec_sub), ("fresh_awq", fre_sub)):
    summary[name] = dict(
        n=len(S_),
        mean_tokens=float(np.mean([r["token_count"] for r in S_])),
        neg_fraction=stats(neg, S_),
        neutral=float(np.mean([r["emo_dist"]["neutral"] for r in S_])),
        mean_dist={e: float(np.mean([r["emo_dist"][e] for r in S_])) for e in EMO})

from scipy.stats import mannwhitneyu, ks_2samp
tok_r = np.array([r["token_count"] for r in rec_sub], float)
tok_f = np.array([r["token_count"] for r in fre_sub], float)
neg_r = np.array([neg(r) for r in rec_sub])
neg_f = np.array([neg(r) for r in fre_sub])
tests = dict(
    tokens_mannwhitney_p=float(mannwhitneyu(tok_r, tok_f).pvalue),
    tokens_ks_p=float(ks_2samp(tok_r, tok_f).pvalue),
    neg_mannwhitney_p=float(mannwhitneyu(neg_r, neg_f).pvalue),
    neg_ks_p=float(ks_2samp(neg_r, neg_f).pvalue))

res = dict(model=KEY, awq_checkpoint=AWQ, n_exact_checked=N_EXACT,
           n_exact_match=n_exact_match, n_dist=len(fresh),
           exact_rows=exact_rows, summary=summary, tests=tests)
json.dump(res, open(f"{HERE}/phase1_verify.json", "w"), indent=1)

print(f"\n--- GATE A (exact match, as specified) ---")
print(f"  exact matches: {n_exact_match}/{N_EXACT}")
for r in exact_rows[:3]:
    print(f"   key={r['key']}\n     recorded: {r['recorded'][:90]!r}\n     fresh   : {r['fresh'][:90]!r}")
print(f"\n--- GATE B (distributional, n={len(fresh)}) ---")
for name in ("recorded_awq", "fresh_awq"):
    s = summary[name]
    print(f"  {name:<13} tokens={s['mean_tokens']:6.2f}  neg={s['neg_fraction']:.4f}  neutral={s['neutral']:.4f}")
for k, v in tests.items():
    print(f"  {k:<26} = {v:.4f}")
print(f"\nwrote bf16/phase1_verify.json")
