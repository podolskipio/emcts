from emotion_classifiers.llm_emotion import Emotions
from utils.sessions import DialogSession

_HF_LABEL_MAP = {
	"anger": Emotions.Anger,
	"disgust": Emotions.Disgust,
	"fear": Emotions.Fear,
	"joy": Emotions.Happiness,
	"neutral": Emotions.Neutral,
	"sadness": Emotions.Sadness,
	"surprise": Emotions.Surprise,
}


class HFEmotionClassifier:
	"""Same public interface as ``BaseLLMEmotionClassifier``; uses an HF text-classification
	pipeline for emotion prediction. See module docstring for rationale."""

	user_role: str | None = None
	DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

	def __init__(self, _unused_backbone=None, model_name: str = DEFAULT_MODEL):
		from transformers import pipeline
		# top_k=None returns scores for every label in one call (no need to invoke twice).
		self.pipe = pipeline("text-classification", model=model_name, top_k=None)
		self.emotions = list(Emotions)
		self.records: list[dict] = []
		self._dist_cache: dict[str, dict] = {}

	def predict_distribution_from_utterance(self, utterance: str) -> dict:
		"""Return the full {Emotions: probability} dict for a single utterance."""
		if utterance in self._dist_cache:
			return self._dist_cache[utterance]
		scores = self.pipe(utterance)[0]   # list[{label, score}]
		dist = {e: 0.0 for e in self.emotions}
		for s in scores:
			mapped = _HF_LABEL_MAP.get(s["label"].lower())
			if mapped is not None:
				dist[mapped] = s["score"]
		total = sum(dist.values()) or 1.0
		dist = {e: p / total for e, p in dist.items()}
		self._dist_cache[utterance] = dist
		return dist

	def predict_distribution(self, state: DialogSession) -> dict:
		if len(state) == 0:
			return {e: 1.0 / len(self.emotions) for e in self.emotions}
		last_utt = state.history[-1][-1]
		return self.predict_distribution_from_utterance(last_utt)

	def predict(self, state: DialogSession):
		dist = self.predict_distribution(state)
		return max(dist, key=dist.get)

	def predict_from_single_utterance(self, utterance: str):
		dist = self.predict_distribution_from_utterance(utterance)
		emotion = max(dist, key=dist.get)
		self.records.append({"utterance": utterance, "emotion": str(emotion), "distribution": dist})
		return emotion

	def get_emotion(self, utterance: str):
		return self.predict_from_single_utterance(utterance)

	def predict_distribution_from_full_history(self, state: DialogSession, utterance: str) -> dict:
		# TODO for future use add full history, now single utterance
		return self.predict_distribution_from_utterance(utterance)

	def get_emotion_from_full_history(self, state: DialogSession, utterance: str):
		dist = self.predict_distribution_from_full_history(state, utterance)
		emotion = max(dist, key=dist.get)
		self.records.append({
			"utterance": utterance,
			"emotion": str(emotion),
			"context": state.to_string_rep(),
			"distribution": dist,
		})
		return emotion
