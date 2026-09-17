"""Store alt1's raw 28-dim GoEmotions sigmoid scores so the Ekman collapse can be varied.

The canonical google-research Ekman groups are very unequal in size (happiness 12 labels,
neutral 1), so a group-SUM collapse is size-biased. This dumps the raw scores so both
SUM and MAX collapses can be evaluated from the same forward pass.
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
MID = "SamLowe/roberta-base-go_emotions"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
recs = [json.loads(l) for l in open(f"{ROOT}/spike/generations.jsonl")]
tok = AutoTokenizer.from_pretrained(MID)
clf = AutoModelForSequenceClassification.from_pretrained(MID).to(DEV).eval()
labels = [clf.config.id2label[i] for i in range(clf.config.num_labels)]
rows = []
with torch.no_grad():
    for i in range(0, len(recs), 128):
        enc = tok([r["response_text"] for r in recs[i:i+128]], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(DEV)
        rows += torch.sigmoid(clf(**enc).logits.float()).cpu().tolist()
with open(f"{ROOT}/c4/alt1_raw_goemotions.jsonl", "w") as f:
    f.write(json.dumps({"_labels": labels}) + "\n")
    for r in rows: f.write(json.dumps([round(v, 6) for v in r]) + "\n")
print(f"wrote {len(rows)} x {len(labels)} raw sigmoid scores")
