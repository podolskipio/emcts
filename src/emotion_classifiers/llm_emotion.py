from enum import StrEnum, auto

import numpy as np

from utils.gen_models import GenerationModel
from utils.sessions import DialogSession


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

    def __init__(
		self,
		generation_model: GenerationModel,
	):
        self.emotions = [
			Emotions.Happiness, Emotions.Sadness, Emotions.Fear, Emotions.Anger, Emotions.Surprise, Emotions.Disgust, Emotions.Contempt, Emotions.Neutral,
		]
        self.generation_model = generation_model
        self.smoothing = 1.0
        # every single-utterance classification is logged here as {"utterance", "emotion"};
        # the runner reads this at the end to print the distribution + dump it to JSON
        self.records: list[dict] = []
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

        self.inf_args = {
            "max_new_tokens": 8,
            "temperature": 1.0,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 10,
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

    def predict(self, state: DialogSession) -> str:
        print(f"Given state:\n{state.to_string_rep()}")
        # test k times and compute prob. See num_return_sequences in the API
        if len(state) == 0:
            prompt = f"""
            User Utterance to predict emotions from:
            {self.task_prompt}
            Emotion:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            Utterances history as context and use mainly the last Utterance to predict emotions from:
            {state.to_string_rep()}
            Emotion:
            """
        prompt = prompt.replace("\t", "").strip()
        data = self.generation_model.generate(prompt, **self.inf_args)
        sampled_emotions = self._get_generated_emotion(data)
        # convert to prob distribution
        prob = np.zeros(len(self.emotions))
        prob += self.smoothing
        for emo in sampled_emotions:
            prob[self.emotions.index(emo)] += 1
        prob /= prob.sum()
        return self.emotions[np.argmax(prob)]

    def get_emotion(self, utterance: str):
        # classify a single utterance (e.g. the user's latest response); used by the game transition
        return self.predict_from_single_utterance(utterance)

    def predict_from_single_utterance(self, utterance: str):
        # test k times and compute prob. See num_return_sequences in the API

        prompt = f"""
        {self.task_prompt}
        User Utterance to predict emotions from:
        {utterance}
        Predicted Emotion code:
        """
        prompt = prompt.replace("\t", "").strip()
        data = self.generation_model.generate(prompt, **self.inf_args)
        sampled_emotions = self._get_generated_emotion(data)
        # convert to prob distribution
        prob = np.zeros(len(self.emotions))
        prob += self.smoothing
        for emo in sampled_emotions:
            prob[self.emotions.index(emo)] += 1
        prob /= prob.sum()
        emotion = self.emotions[np.argmax(prob)]
        self.records.append({"utterance": utterance, "emotion": str(emotion)})
        return emotion



class P4GLLMEmotionClassifier(BaseLLMEmotionClassifier):
    user_role = "Persuadee"