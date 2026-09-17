"""Classify every generation with the shared emotion model; store the full 7-dim softmax."""
import json, glob, os, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
MODEL = "j-hartmann/emotion-english-distilroberta-base"
RENAME = {"joy": "happiness"}          # label rename only; distribution untouched

tok = AutoTokenizer.from_pretrained(MODEL)
clf = AutoModelForSequenceClassification.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
clf.to(dev).eval()
labels = [RENAME.get(clf.config.id2label[i], clf.config.id2label[i])
          for i in range(clf.config.num_labels)]
print(f"classifier on {dev}; labels = {labels}")

recs = []
for f in sorted(glob.glob(f"{ROOT}/spike/raw_*.jsonl")):
    recs += [json.loads(l) for l in open(f)]
print(f"{len(recs)} generations from {len(glob.glob(f'{ROOT}/spike/raw_*.jsonl'))} model files")

B = 128
with torch.no_grad():
    for i in range(0, len(recs), B):
        batch = recs[i:i+B]
        enc = tok([r["response_text"] for r in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(dev)
        probs = clf(**enc).logits.softmax(-1).cpu().tolist()
        for r, p in zip(batch, probs):
            r["emo_dist"] = {l: float(v) for l, v in zip(labels, p)}
        if i % 2560 == 0: print(f"  {i}/{len(recs)}", flush=True)

with open(f"{ROOT}/spike/generations.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
print(f"wrote spike/generations.jsonl ({len(recs)} records)")
for m in sorted({r["model"] for r in recs}):
    print(f"  {m}: {sum(r['model']==m for r in recs)}")
