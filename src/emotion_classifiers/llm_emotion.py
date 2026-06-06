from enum import StrEnum, auto
import hashlib

import numpy as np

from utils.gen_models import GenerationModel
from utils.sessions import DialogSession, EmotionAwareDialogSession


class Emotions(StrEnum):
    Happiness = auto()
    Sadness = auto()
    Fear = auto()
    Anger = auto()
    Surprise = auto()
    Disgust = auto()
    Contempt = auto()
    Neutral = auto()


NEGATIVE_EMOTIONS = [Emotions.Sadness, Emotions.Fear, Emotions.Anger, Emotions.Disgust, Emotions.Contempt]


class BaseLLMEmotionClassifier:
    user_role: str | None = None

    # Subclasses override with task-specific few-shot examples (see P4GLLMEmotionClassifier).
    # Folded into both the single-utterance and history prompts so the model sees
    # task-appropriate exemplars before predicting.
    few_shot_examples: str = ""

    def __init__(
		self,
		generation_model: GenerationModel,
	):
        self.emotions = [
			Emotions.Happiness, Emotions.Sadness, Emotions.Fear, Emotions.Anger, Emotions.Surprise, Emotions.Disgust, Emotions.Contempt, Emotions.Neutral,
		]
        self.generation_model = generation_model
        # smoothing: prior added to each emotion bin before argmax. Was 1.0 (with 10 samples this
        # made the prior contribute ~44% of mass — way too strong, weak signals tied a lot).
        # 0.1 lets the actual sampled votes drive the argmax.
        self.smoothing = 0.1
        # every single-utterance classification is logged here as {"utterance", "emotion"};
        # the runner reads this at the end to print the distribution + dump it to JSON.
        # ``get_emotion_from_full_history`` also adds a ``context`` field to its records.
        self.records: list[dict] = []
        # cache of full distributions ({Emotions: prob}) keyed by hash of (mode, payload).
        # Both predict_*() and predict_distribution_*() read from this cache; one LLM call
        # serves every later lookup of the same input. Big win for MCTS rollouts that
        # repeatedly classify the same utterance.
        self._dist_cache: dict[str, dict] = {}
        self.task_prompt = f"""
        Your task is to predict emotions of the {self.user_role} from his message from possible emotions:
        [Happiness]: Often expressed through smiles and laughter, happiness is a positive emotional state associated with feelings of joy, satisfaction, and contentment. It is considered one of the most important emotions as it promotes well-being and social bonding.
        [Sadness]: This emotion is characterized by feelings of loss, disappointment, or helplessness. It can lead to crying and withdrawal from social interactions. While often viewed negatively, sadness is a natural response to certain life events and can foster empathy and connection with others.
        [Fear]: Fear is a response to perceived threats and is often accompanied by physiological changes such as increased heart rate. It prepares individuals to react to danger, either by fighting or fleeing. Fear can also lead to avoidance behaviors.
        [Anger]: This emotion arises in response to perceived injustice or frustration. It can manifest in various ways, from irritation to rage, and is often expressed through aggressive behaviors or confrontational attitudes. Anger can motivate individuals to address grievances.
        [Surprise]: Surprise is a brief emotional state that occurs in response to unexpected events. It can be positive (pleasant surprise) or negative (shock) and often leads to a quick reassessment of the situation.
        [Disgust]: This emotion is typically a reaction to something considered offensive or repulsive, whether it be food, behavior, or ideas. Disgust serves as a protective mechanism, helping individuals avoid harmful substances or situations.
        [Contempt]: Contempt is a complex emotion that combines feelings of disdain and superiority over others. It often manifests in dismissive behaviors and can damage relationships if expressed openly.
        [Neutral]: The user does not show any emotions in his response.
        Output only the emotion code, nothing else.
        Example Output: [Surprise]
        Example Output: [Disgust]
        Example Output: [Contempt]
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()

        # temperature lowered from 1.0 -> 0.3 (much less per-call jitter);
        # num_return_sequences dropped from 10 -> 5 (tighter prob estimates at the new temp,
        # and half the LLM cost per classification).
        self.inf_args = {
            "max_new_tokens": 8,
            "temperature": 0.3,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 5,
        }
        return

    @staticmethod
    def _match_emotion(token: str):
        # the prompt labels / model output are capitalized ("Happiness") but Emotions values are
        # lowercased (StrEnum + auto()), so match case-insensitively and return the canonical member
        token = token.strip().lower()
        for emotion in Emotions:
            if emotion.value == token:
                return emotion
        return None

    def _get_generated_emotion(self, data) -> list:
        pred_emo = []
        for resp in data:
            resp = resp['generated_text'].strip()
            start_idx = resp.find("[")
            end_idx = resp.find("]")
            if start_idx == -1 or end_idx == -1:
                continue
            found_emo = self._match_emotion(resp[start_idx + 1: end_idx])
            if found_emo is not None:
                pred_emo.append(found_emo)
        return pred_emo

    @staticmethod
    def _cache_key(*parts: str) -> str:
        return hashlib.sha1("\n###\n".join(parts).encode("utf-8")).hexdigest()

    def _classify_distribution(self, prompt: str) -> dict:
        """Run the LLM on ``prompt`` and return a normalized {Emotions: probability} dict."""
        data = self.generation_model.generate(prompt, **self.inf_args)
        sampled_emotions = self._get_generated_emotion(data)
        prob = np.zeros(len(self.emotions))
        prob += self.smoothing
        for emo in sampled_emotions:
            prob[self.emotions.index(emo)] += 1
        prob /= prob.sum()
        return dict(zip(self.emotions, prob))

    def _build_single_prompt(self, utterance: str) -> str:
        prompt = f"""
        {self.task_prompt}
        {self.few_shot_examples}
        User Utterance to predict emotions from:
        {utterance}
        Predicted Emotion code:
        """
        return prompt.replace("\t", "").strip()

    def _build_history_prompt(self, state: DialogSession) -> str:
        prompt = f"""
        {self.task_prompt}
        {self.few_shot_examples}

        Conversation so far (read it carefully — the {self.user_role}'s emotion is shaped both
        by what the other speaker just said and by what the {self.user_role} themselves said
        before):
        {state.to_string_rep()}

        Predict the emotion of the {self.user_role}'s LAST utterance in the conversation above.
        Emotion:
        """
        return prompt.replace("\t", "").strip()

    def predict_distribution_from_utterance(self, utterance: str) -> dict:
        """Return the full {Emotions: probability} dict for a single utterance (no context)."""
        key = self._cache_key("single", utterance)
        if key in self._dist_cache:
            return self._dist_cache[key]
        dist = self._classify_distribution(self._build_single_prompt(utterance))
        self._dist_cache[key] = dist
        return dist

    def predict_distribution(self, state: DialogSession) -> dict:
        """Return the full {Emotions: probability} dict given the dialog state.

        ``state``'s last record is expected to be the user utterance being classified.
        """
        if len(state) == 0:
            return {e: 1.0 / len(self.emotions) for e in self.emotions}
        key = self._cache_key("history", state.to_string_rep())
        if key in self._dist_cache:
            return self._dist_cache[key]
        dist = self._classify_distribution(self._build_history_prompt(state))
        self._dist_cache[key] = dist
        return dist

    def predict(self, state: DialogSession) -> str:
        # argmax over the cached distribution. Kept for backward compatibility with callers
        # that just want a single label; new MCTS/penalty paths read predict_distribution().
        dist = self.predict_distribution(state)
        emotion = max(dist, key=dist.get)
        print(f"Given state:\n{state.to_string_rep()}\nEmotion is {emotion}")
        return emotion

    def get_emotion(self, utterance: str):
        # classify a single utterance (e.g. the user's latest response); used by the game transition
        return self.predict_from_single_utterance(utterance)


    def predict_distribution_from_full_history(self, state: DialogSession, utterance: str) -> dict:
        tmp_state = state.copy()
        if isinstance(tmp_state, EmotionAwareDialogSession):
            tmp_state.add_single(tmp_state.USR, None, "Neutral", utterance)
        else:
            tmp_state.add_single(tmp_state.USR, None, utterance)
        return self.predict_distribution(tmp_state)

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


    def predict_from_single_utterance(self, utterance: str):
        dist = self.predict_distribution_from_utterance(utterance)
        emotion = max(dist, key=dist.get)
        self.records.append({"utterance": utterance, "emotion": str(emotion), "distribution": dist})
        return emotion



class P4GLLMEmotionClassifier(BaseLLMEmotionClassifier):
    user_role = "Persuadee"

    few_shot_examples = """
    Examples:
    [Persuadee]: "I'm not sure I have money to spare right now."
    Emotion: [Sadness]

    [Persuadee]: "How do I know this isn't a scam?"
    Emotion: [Disgust]

    [Persuadee]: "I would love to help kids in need."
    Emotion: [Happiness]

    [Persuadee]: "Hmm. Tell me how the donations get used."
    Emotion: [Neutral]

    [Persuadee]: "You're trying to guilt-trip me into this."
    Emotion: [Anger]
    """.strip()
