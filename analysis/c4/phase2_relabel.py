"""C4 Phase 2 - re-label every spike generation with two alternative emotion classifiers.

No text generation. Reads spike/generations.jsonl, writes c4/relabelled.jsonl with the
original DistilRoBERTa distribution plus alt1 (GoEmotions) and alt2 (EmoBERTa).
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
EMO  = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
DEV  = "cuda" if torch.cuda.is_available() else "cpu"

ALT1 = "SamLowe/roberta-base-go_emotions"      # multi-label, 28 GoEmotions labels
ALT2 = "tae898/emoberta-large"                 # single-label softmax, native Ekman-7

# canonical google-research/goemotions ekman_mapping.json + neutral, joy renamed happiness
GO2EK = {
    "anger":     ["anger", "annoyance", "disapproval"],
    "disgust":   ["disgust"],
    "fear":      ["fear", "nervousness"],
    "happiness": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
                  "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness":   ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise":  ["surprise", "realization", "confusion", "curiosity"],
    "neutral":   ["neutral"],
}

recs = [json.loads(l) for l in open(f"{ROOT}/spike/generations.jsonl")]
texts = [r["response_text"] for r in recs]
print(f"{len(recs)} records to re-label on {DEV}", flush=True)


def load(mid):
    tok = AutoTokenizer.from_pretrained(mid)
    clf = AutoModelForSequenceClassification.from_pretrained(mid).to(DEV).eval()
    labels = [clf.config.id2label[i] for i in range(clf.config.num_labels)]
    return tok, clf, labels


def run(mid, to_dist, B=128):
    tok, clf, labels = load(mid)
    print(f"  {mid}: {len(labels)} labels", flush=True)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), B):
            enc = tok(texts[i:i + B], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(DEV)
            logits = clf(**enc).logits.float().cpu()
            out += to_dist(logits, labels)
            if i % 2560 == 0:
                print(f"    {i}/{len(texts)}", flush=True)
    del clf
    torch.cuda.empty_cache()
    return out


def dist_goemotions(logits, labels):
    """28 sigmoid scores -> sum within Ekman group -> renormalise to a 7-dim distribution."""
    s = torch.sigmoid(logits)
    ix = {l: i for i, l in enumerate(labels)}
    grouped = torch.stack([s[:, [ix[g] for g in GO2EK[e]]].sum(1) for e in EMO], 1)
    grouped = grouped / grouped.sum(1, keepdim=True)
    return [{e: float(v) for e, v in zip(EMO, row)} for row in grouped]


def dist_softmax7(logits, labels):
    """native Ekman-7 softmax; only rename is joy -> happiness."""
    p = logits.softmax(-1)
    names = ["happiness" if l == "joy" else l for l in labels]
    assert sorted(names) == sorted(EMO), names
    ix = {l: i for i, l in enumerate(names)}
    return [{e: float(row[ix[e]]) for e in EMO} for row in p]


d1 = run(ALT1, dist_goemotions)
d2 = run(ALT2, dist_softmax7)

with open(f"{ROOT}/c4/relabelled.jsonl", "w") as f:
    for k, (r, a1, a2) in enumerate(zip(recs, d1, d2)):
        f.write(json.dumps({
            "record_id": f"{r['model']}|{r['persona_id']}|{r['utterance_id']}|{r['seed']}|{r['arm']}",
            "model": r["model"], "arm": r["arm"], "persona_id": r["persona_id"],
            "dialogue_id": r["dialogue_id"], "utterance_id": r["utterance_id"],
            "seed": r["seed"], "token_count": r["token_count"],
            "response_text": r["response_text"],
            "emo_dist_distilroberta": {e: r["emo_dist"][e] for e in EMO},
            "emo_dist_alt1": a1, "emo_dist_alt2": a2}) + "\n")
print(f"wrote c4/relabelled.jsonl ({len(recs)} records)")
