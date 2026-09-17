# C4 Phase 1 — classifier selection, pre-committed before any re-labelling

Written before running Phase 2. Recorded so it is verifiable that the alternatives were
chosen on stated criteria, not on whether they agreed with DistilRoBERTa.

## Reference classifier (the one under test)
`j-hartmann/emotion-english-distilroberta-base` — DistilRoBERTa-base, single-label softmax,
Ekman-7 {anger, disgust, fear, joy, neutral, sadness, surprise}. Trained on a mix of 6 English
emotion datasets (Twitter, Reddit, TV dialogue, self-report).

## alt1 — `SamLowe/roberta-base-go_emotions`
- Architecture: RoBERTa-base, `problem_type = multi_label_classification` (28 sigmoid heads).
- Training data: **GoEmotions** (Reddit, 58k comments, 27 emotions + neutral) — a corpus that
  does not overlap the reference model's training mix. This is the "different training data" axis.
- Emits a full 28-dim score vector, not top-1.
- Requires a 28 -> Ekman-7 mapping (below).

## alt2 — `tae898/emoberta-large`
- Architecture: **RoBERTa-large** (355M vs 82M) — different capacity/architecture axis.
- Training data: **MELD + IEMOCAP** — conversational/dialogue emotion recognition, i.e. a
  different domain *and* different labelling protocol from the reference model.
- Native Ekman-7 label set, single-label **softmax**: directly comparable, no mapping needed.
- Chosen deliberately as the dialogue-domain check, since the spike's text is dialogue turns.

## Considered and not used
- `monologg/bert-base-cased-goemotions-ekman` (BERT, GoEmotions-Ekman). Viable on the criteria,
  but redundant with alt1 on training data, and ships a custom `BertForMultiLabelClassification`
  head requiring remote code. Not run; not swapped in for any reason related to its output.
- `michellejieli/emotion_text_classifier` — rejected: DistilRoBERTa near-clone of the reference.
- `bhadresh-savani/distilbert-base-uncased-emotion` — rejected: dair-ai/emotion 6-label set has
  no `neutral` and no `disgust`, so it neither maps onto nor subsumes Ekman-7.

**Exactly two alternatives were run. No third classifier was run and discarded.**

## GoEmotions -> Ekman-7 mapping (alt1)
Not invented here. Taken verbatim from Google Research's own published grouping,
`google-research/goemotions/data/ekman_mapping.json`, plus `neutral -> neutral`:

| Ekman-7 | GoEmotions labels folded in |
|---|---|
| anger | anger, annoyance, disapproval |
| disgust | disgust |
| fear | fear, nervousness |
| happiness (joy) | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| sadness | sadness, disappointment, embarrassment, grief, remorse |
| surprise | surprise, realization, confusion, curiosity |
| neutral | neutral |

27 emotions + neutral = 28, every label assigned exactly once, none dropped.

**Sigmoid -> distribution.** alt1 is multi-label, so its 28 sigmoid scores do not sum to 1.
Procedure: sum the sigmoid scores within each Ekman-7 group, then renormalise the 7 sums to 1.
This preserves the model's relative confidence across labels and yields the distribution
`Q_emo` would consume. The renormalisation is the only transformation applied.

## Label rename
`joy -> happiness` for all three classifiers, matching the spike convention. Rename only.
