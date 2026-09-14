import logging
import numpy as np
import torch

from typing import List, Tuple

from players.planner import DialogPlanner
from utils.sessions import DialogSession
from utils.gen_models import GenerationModel, DialogModel
from games import EmotionalSupportGame
from collections import Counter
from utils.rewards import reward_dict
from players.prompting import parse_das, recent_turns, split_da

def esc_patient_scenario(state) -> str:
    return (
        "You are the patient who is looking for help from the therapist, because you have "
        "the emotional issue about %s regarding %s."
        % (getattr(state, "emotion_type", "distress") or "distress",
           getattr(state, "problem_type", "an ongoing problem") or "an ongoing problem")
    )

logger = logging.getLogger(__name__)

_ESC_REWARDS = {
    EmotionalSupportGame.U_FeelWorse: 'worse',
    EmotionalSupportGame.U_FeelTheSame: 'same',
    EmotionalSupportGame.U_FeelBetter: 'better',
    EmotionalSupportGame.U_Solved: 'solved',
}


def _mean_reward(sampled_das) -> float:
    """Mean reward over the patient acts that carry one; 0.0 if none of them do."""
    scores = [reward_dict['esc'][_ESC_REWARDS[da]] for da in sampled_das if da in _ESC_REWARDS]
    return float(np.mean(scores)) if scores else 0.0


class ESCSystemPlanner(DialogPlanner):
    def __init__(
        self,
        dialog_acts,
        max_hist_num_turns,
        user_dialog_acts,
        user_max_hist_num_turns,
        generation_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
    ) -> None:
        super().__init__()
        self.dialog_acts = dialog_acts
        self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
        self.user_dialog_acts = user_dialog_acts
        self.user_max_hist_num_turns = (
            user_max_hist_num_turns  # used in heuristic function
        )
        self.conv_examples = conv_examples
        self.generation_model = generation_model
        self.smoothing = 1.0
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is a new conversation between Therapist and a Patient.
        {self.process_exp()}
        The following is a new conversation between another Therapist and Patient.
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()

        self.inf_args = {
            "max_new_tokens": 8,
            "temperature": 1.0,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 15,
        }
        return

    def process_exp(self, keep_sys_da=True, keep_user_da=False):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += (
                exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da)
                + "\n"
            )
        return prompt_exps.strip()

    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        turn = len(state)
        if turn < 1:
            return np.array(
                [1 if da == EmotionalSupportGame.S_Others else 0 for da in self.dialog_acts]
            )
        return np.array([1 for _ in self.dialog_acts])

    def get_utterance(self, state, action) -> str:
        return ""  # should not be called

    def predict(self, state: DialogSession, policy=None, ent_bound=None) -> "Tuple[np.ndarray, float]":
        if len(state) == 0:
            prompt = f"""
            {self.task_prompt}
            {EmotionalSupportGame.SYS}:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {state.to_string_rep(keep_sys_da=True)}
            {EmotionalSupportGame.SYS}:
            """
        prompt = prompt.replace("\t", "").strip()
        logger.debug(prompt)
        data = self.generation_model.generate(prompt, **self.inf_args)
        sampled_das = parse_das(data, self.dialog_acts)
        logger.debug(f"sampled das: {sampled_das}")
        v, _ = self.heuristic(state)
        return self._histogram(sampled_das), v

    def _histogram(self, sampled_das) -> "np.ndarray":
        """Sampled acts as a distribution, with `self.smoothing` added so an act that happened
        not to be drawn is not assigned probability zero."""
        prob = np.zeros(len(self.dialog_acts)) + self.smoothing
        for da in sampled_das:
            prob[self.dialog_acts.index(da)] += 1
        return prob / prob.sum()

    def heuristic(self, state: DialogSession) -> float:
        """Ask the patient simulator how it feels now; score its answer."""
        assert state[-1][0] == EmotionalSupportGame.USR
        prompt = f"""
        The following is background information about the task.
        A Therapist is having a counselling conversation with a Patient to help reduce the Patient's emotional distress.
        The Patient can choose amongst the following actions during a conversation to respond to the Therapist:
        {" ".join([f"[{da}]" for da in self.user_dialog_acts])}
        The following is a conversation between a Therapist and a Patient.
        {self.process_exp(keep_sys_da=False, keep_user_da=True)}
        The following is a new conversation between another Therapist and Patient.
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
        {EmotionalSupportGame.SYS}: How do you feel now? Has your issue been solved?
        {EmotionalSupportGame.USR}:
        """
        prompt = prompt.replace("\t", "").strip()

        inf_args = {
            "max_new_tokens": 8,
            "temperature": 1.1,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 10,
        }
        data = self.generation_model.generate(prompt, **inf_args)
        sampled_das = parse_das(data, self.user_dialog_acts)

        logger.debug(f"patient prompt: {prompt}")
        logger.debug(f"sampled das: {sampled_das}")
        # this planner keeps GDP-Zero's fixed scale rather than reward_dict['esc']
        scale = {EmotionalSupportGame.U_FeelWorse: -1.0, EmotionalSupportGame.U_FeelTheSame: -0.5,
                 EmotionalSupportGame.U_FeelBetter: 0.5, EmotionalSupportGame.U_Solved: 1.0}
        score = [scale[da] for da in sampled_das if da in scale]
        v = float(np.mean(score)) if score else 0.0
        logger.debug(f"sampled das to v: {v}")
        return v, sampled_das


class ESCChatSystemPlanner(ESCSystemPlanner):
    def __init__(
        self,
        dialog_acts,
        max_hist_num_turns,
        user_dialog_acts,
        user_max_hist_num_turns,
        generation_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        infer_user_da = True,
        use_policy_prior = True,
        action_temperature = 1.0,
        action_num_return_sequences = 15,
        eval_temperature = 1.0,
        eval_num_return_sequences = 10,
    ) -> None:
        super().__init__(
            dialog_acts,
            max_hist_num_turns,
            user_dialog_acts,
            user_max_hist_num_turns,
            generation_model,
            conv_examples,
        )
        # True makes the critic below ask its question in plain English instead of with
        # the [act] labels, and read the verdict out of the answer's wording.
        self.infer_user_da = infer_user_da
        self.use_policy_prior = use_policy_prior
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is an example conversation between a Therapist and a Patient.
        """.replace(
            "\t", ""
        ).strip()
        self.new_task_prompt = "The following is a new conversation between Therapist (you) and a Patient."
        self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

        self.inf_args = {
            "max_new_tokens": 16,
            "temperature": action_temperature,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": action_num_return_sequences,
        }
        self.eval_args = {
            "max_new_tokens": 16,
            "temperature": eval_temperature,
            "num_return_sequences": eval_num_return_sequences,
        }
        return

    def process_chat_exp(
        self,
        new_task_prompt,
        assistant_role=EmotionalSupportGame.SYS,
        keep_sys_da=True,
        keep_user_da=False,
    ):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self._process_chat_turns(
                exp, keep_sys_da, keep_user_da, assistant_role
            )
            prompt_exps.append({"role": "system", "content": new_task_prompt})
        return prompt_exps[:-1]

    def _process_chat_turns(
        self,
        exp: DialogSession,
        keep_sys_da,
        keep_user_da,
        assistant_role=EmotionalSupportGame.SYS,
        max_hist_num_turns: int = -1,
    ):
        """``exp`` as chat messages, with whichever speaker ``assistant_role`` names cast as
        the assistant. Guarded on the raw history, not ``len(exp)``: that counts turns, so a
        state ending mid-turn read as empty and dropped the conversation from the prompt."""
        if len(exp.history) == 0:
            return []
        assert exp[0][0] == EmotionalSupportGame.SYS  # dialogues open with the Therapist

        prompt_messages = []
        for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            keep_da = keep_sys_da if role == EmotionalSupportGame.SYS else keep_user_da
            content = f"{role}: [{da}] {utt}" if keep_da else f"{role}: {utt}"
            prompt_messages.append({
                "role": "assistant" if role == assistant_role else "user",
                "content": content.strip(),
            })
        return prompt_messages

    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        turn = len(state)
        if turn < 1:
            return np.array(
                [
                    1 if da == EmotionalSupportGame.S_Others else 0
                    for da in self.dialog_acts
                ]
            )
        return np.array([1 for _ in self.dialog_acts])

    def get_utterance(self, state, action) -> str:
        return ""  # should not be called

    def predict(self, state: DialogSession, policy=None, ent_bound=None) -> "Tuple[np.ndarray, float]":
        if self.use_policy_prior and policy is not None:
            logger.info('Apply policy model to calculate prior')
            with torch.no_grad():
                agent_dist, _ = policy.apply_policy(state.to_chat_messages())
            logger.info('Apply the policy network (Roberta-large) to predict prior distribution.')
            if len(agent_dist.shape) > 1:
                agent_dist = agent_dist.squeeze(dim=0)
            prob = agent_dist.detach().cpu().numpy()
        else:
            logger.info('Apply LLM to calculate prior')
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {'role': 'system', 'content': self.new_task_prompt}
            ]
            if len(state) == 0:
                messages.append(
                    {"role": "user", "content": f"{EmotionalSupportGame.USR}: Hello."}
                )
            else:
                assert state[-1][0] == EmotionalSupportGame.USR
                messages += self._process_chat_turns(
                    state, keep_sys_da=True, keep_user_da=False,
                    max_hist_num_turns=self.max_hist_num_turns,
                )
            data = self.generation_model.chat_generate(messages, **self.inf_args)
            sampled_das = parse_das(data, self.dialog_acts)
            logger.info(f"sampled das: {sampled_das}")
            prob = self._histogram(sampled_das)
        v, _ = self.heuristic(state)
        return prob, v

    def _parse_free_text_da(self, data) -> list:
        """Read the patient's verdict out of a free-text answer, for the zero-shot prompt
        that asks the question in plain English instead of with act labels."""
        pred_da = []
        for resp in data:
            resp = resp['generated_text'].lower()
            if 'no' in resp and 'worse' in resp:
                pred_da.append(EmotionalSupportGame.U_FeelWorse)
            elif 'no' in resp and 'better' in resp:
                pred_da.append(EmotionalSupportGame.U_FeelBetter)
            elif 'yes' in resp and 'solved' in resp:
                pred_da.append(EmotionalSupportGame.U_Solved)
            else:
                pred_da.append(EmotionalSupportGame.U_FeelTheSame)
        return pred_da

    def heuristic(self, state: DialogSession) -> float:
        # ask the patient simulator whether its emotional issue has been solved at this point
        assert state[-1][0] == EmotionalSupportGame.USR
        if not self.infer_user_da:
            user_task_prompt = f"""
            Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation.
            You can choose amongst the following actions during a conversation to respond to the Therapist:
            {" ".join([f"[{da}]" for da in self.user_dialog_acts])}
            The following is a example conversation between a Therapist and a Patient.
            """.replace(
                "\t", ""
            ).strip()
            user_new_task_prompt = "The following is a new conversation between a Therapist and a Patient (you)."

            messages = [
                {"role": "system", "content": user_task_prompt},
                *self.process_chat_exp(
                    new_task_prompt=user_new_task_prompt,
                    assistant_role=EmotionalSupportGame.USR,
                    keep_sys_da=False,
                    keep_user_da=True,
                ),
                {"role": "system", "content": user_new_task_prompt},
            ]
            messages += self._process_chat_turns(
                state,
                assistant_role=EmotionalSupportGame.USR,
                keep_sys_da=False,
                keep_user_da=True,
                max_hist_num_turns=self.user_max_hist_num_turns,
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{EmotionalSupportGame.SYS}: Has the your issue been solved?",
                }
            )
        else:
            # the branch that actually runs: build_agents leaves infer_user_da at its default.
            # The window matters here -- unwindowed this was the one prompt that grew with the
            # whole dialogue, past vicuna's context at 15 turns.
            conversation = self._process_chat_turns(
                state,
                assistant_role=EmotionalSupportGame.USR,
                keep_sys_da=False,
                keep_user_da=False,
                max_hist_num_turns=self.user_max_hist_num_turns,
            )
            dial = "".join("\n{}".format(turn['content']) for turn in conversation)
            messages = [
                {"role": "system", "content": "Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation."},
                {"role": "user", "content": "You can only reply with one of the following sentences: No, the Patient feels worse. No, the Patient feels the same. No, but the Patient feels better. Yes, the Patient's issue has been solved.\n\nThe following is a conversation about %s regarding %s: %s\nQuestion: Has the Patient's issue been solved? Answer: " % (state.emotion_type, state.problem_type, dial)}
            ]

        data = self.generation_model.chat_generate(messages, **self.eval_args)
        if self.infer_user_da:
            sampled_das = self._parse_free_text_da(data)
        else:
            sampled_das = parse_das(data, self.user_dialog_acts)

        logger.info(f"patient prompt: {messages}")
        logger.info(f"sampled das: {sampled_das}")
        return _mean_reward(sampled_das), sampled_das


class TherapistModel(DialogModel):
    def __init__(
        self,
        dialog_acts: List[str],
        backbone_model: GenerationModel,
        max_hist_num_turns: int = 5,
        conv_examples: List[DialogSession] = [],
        inference_args: dict = {},
        infer_user_da: bool = True,
    ):
        super().__init__()
        self.conv_examples = conv_examples
        self.backbone_model = backbone_model
        self.max_hist_num_turns = max_hist_num_turns
        # True drops the few-shot demo and the [act] tags from this agent's prompts,
        # so the patient's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        # prompts and DAs
        self.da_prompts_mapping = {
            EmotionalSupportGame.S_Question: "The Therapist asks the Patient to elaborate on the situation they just described.",
            # start of persuasion strategies
            EmotionalSupportGame.S_SelfDisclosure: "The Therapist provides a statement relating to the Patient about the situation they just described.",
            EmotionalSupportGame.S_AffirmationAndReassurance: "The Therapist provides affirmation and reassurance to the Patient on the situation they just described.",
            EmotionalSupportGame.S_ReflectionOfFeelings: "The Therapist acknowledges the Patient's feelings about the situation they described.",
            EmotionalSupportGame.S_ProvidingSuggestions: "The Therapist provides suggestions to the Patient on the situation they just described.",
            EmotionalSupportGame.S_Information: "The Therapist provides factual information to help the Patient with their situation.",
            EmotionalSupportGame.S_RestatementOrParaphrasing: "The Therapist acknowledges the Patient's feelings by paraphrasing their situation.",
            # end of persuasion strategies
            EmotionalSupportGame.S_Others: "The Therapist chats with the Patient.",
        }
        # only allow da that has the mapping
        # ['Affirmation and Reassurance', 'Information', 'Others', 'Providing Suggestions', 'Question', 'Reflection of feelings', 'Restatement or Paraphrasing', 'Self-disclosure']
        self.dialog_acts = sorted([da for da in dialog_acts if da in self.da_prompts_mapping])

        logger.debug(self.dialog_acts)
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in self.dialog_acts])}
        The following is an example conversation between a Therapist and a Patient.
        {self.process_exp()}
        The following is a new conversation between another Therapist and Patient.
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "do_sample": False,  # otherwise tree will never go to the next level
            "return_full_text": False,
            **inference_args,
        }
        return

    def process_exp(self):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += self._process_turns(exp) + "\n"
        return prompt_exps.strip()

    def _process_turns(self, exp: DialogSession, max_hist_num_turns: int = -1):
        prompt_exp = ""
        for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            if role == EmotionalSupportGame.SYS:
                prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
            else:
                prompt_exp += f"{role}: {utt}\n"
        return prompt_exp.strip()

    def get_utterance(self, state: DialogSession, action: int, mode: str = 'train') -> str:
        """Realize the act the planner chose as an utterance."""
        da = self.dialog_acts[action]
        da_prompt = self.da_prompts_mapping[da]
        if len(state) == 0:
            prompt = f"""
            {self.task_prompt}
            {da_prompt}
            {EmotionalSupportGame.SYS}:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {self._process_turns(state, max_hist_num_turns=self.max_hist_num_turns)}
            {da_prompt}
            {EmotionalSupportGame.SYS}:
            """
        prompt = prompt.replace("\t", "").strip()
        data = self.backbone_model.generate(prompt, **self.inference_args)
        return self.backbone_model._cleaned_resp(data, prompt)[0]

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class TherapistChatModel(TherapistModel):
    def __init__(
        self,
        dialog_acts: List[str],
        backbone_model: GenerationModel,
        max_hist_num_turns: int = 5,
        conv_examples: List[DialogSession] = [],
        inference_args: dict = {},
        infer_user_da = True,
    ):
        super().__init__(
            dialog_acts=dialog_acts,
            backbone_model=backbone_model,
            max_hist_num_turns=max_hist_num_turns,
            conv_examples=conv_examples,
            inference_args=inference_args,
        )
        # True drops the few-shot demo and the [act] tags from this agent's prompts,
        # so the patient's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        if self.infer_user_da:
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."
        else:
            self.task_prompt = """
            Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
            You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges.
            The following is an example conversation between a Therapist and a Patient.
            """.replace(
                "\t", ""
            ).strip()
            self.new_task_prompt = "The following is a new conversation between Therapist (you) and another Patient.\nThe Therapist greets the Patient."
            self.prompt_examples = self.process_chat_exp()
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self._process_chat_turns(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def _process_chat_turns(self, exp: DialogSession, da_prompt: str = '', max_hist_num_turns: int = -1, use_role: bool = True):
        """``exp`` as chat messages, each patient turn followed by the instruction for the
        therapist turn that answers it. The final patient turn is the one being answered now,
        so it takes ``da_prompt`` -- the act the planner just chose.

        Guarded on the raw history, not ``len(exp)``: that counts turns, so a state ending
        mid-turn read as empty and dropped the conversation from the prompt.
        """
        if len(exp.history) == 0:
            return []
        assert exp[0][0] == EmotionalSupportGame.SYS  # dialogues open with the Therapist

        prompt_messages = []
        for i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            prefix = f"{role}: " if use_role else ""
            if role == EmotionalSupportGame.SYS:
                prompt_messages.append({"role": "assistant", "content": f"{prefix}{utt}".strip()})
                continue
            is_last = i + 1 >= len(exp.history)
            instruction = da_prompt if is_last else self.da_prompts_mapping[exp[i + 1][1]]
            prompt_messages.append({
                "role": "user",
                "content": f"{prefix}{utt}\n{instruction}".strip(),
            })
        return prompt_messages

    def get_utterance(self, state: DialogSession, action: int, mode='train') -> str:
        return self.get_utterance_batched(state, action, batch=1, mode=mode)[0]

    def get_utterance_batched(
        self, state: DialogSession, action: int, batch: int = 3, mode="train",
    ) -> List[str]:
        da = self.dialog_acts[action]
        da_prompt = self.da_prompts_mapping[da]
        if self.infer_user_da:
            messages = [
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. Please reply with only one short and succinct sentence. %s" % da_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": self.new_task_prompt},
            ]
        if len(state) == 0:
            content = f"Hello.\n{da_prompt}" if self.infer_user_da else f"{EmotionalSupportGame.USR}: Hello.\n{da_prompt}"
            messages.append(
                {"role": "user", "content": content,}
            )
        else:
            assert state[-1][0] == EmotionalSupportGame.USR
            messages += self._process_chat_turns(
                state, da_prompt, max_hist_num_turns=self.max_hist_num_turns, use_role=not self.infer_user_da
            )
        gen_args = {**self.inference_args, "num_return_sequences": batch}
        if mode != 'train':
            gen_args['temperature'] = 0.0
        data = self.backbone_model.chat_generate(messages, **gen_args)
        sys_resps = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{EmotionalSupportGame.SYS}:",
            user_role=f"{EmotionalSupportGame.USR}:",
        )
        return sys_resps

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class PatientModel(DialogModel):
    def __init__(
        self,
        dialog_acts: List[str],
        inference_args: dict,
        backbone_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        max_hist_num_turns=5,
        infer_user_da=True,
    ):
        super().__init__()
        self.conv_examples = conv_examples
        self.backbone_model = backbone_model
        self.dialog_acts = dialog_acts
        self.max_hist_num_turns = max_hist_num_turns
        # prompts
        self.task_prompt = f"""
		Now enter the role-playing mode. In the following conversation, you will play as a Patient in a counselling conversation with a therapist. 
        You are the patient who is looking for the help from the therapist, because you have the emotional issue about depression regarding ongoing depression.
		The Patient (you) can choose amongst the following actions during a conversation to respond to the Therapist:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation.
		{self.process_exp()}
		The following is a new conversation between another Patient and Therapist.
		"""
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = inference_args
        # True drops the few-shot demo and the [act] tags from this agent's prompts,
        # so the patient's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        return

    def process_exp(self):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
        return prompt_exps.strip()

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == EmotionalSupportGame.SYS
        prompt = f"""
        {self.task_prompt}
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
        Patient:
        """
        prompt = prompt.replace("\t", "").strip()
        # produce a response
        data = self.backbone_model.generate(prompt, **self.inference_args)
        user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
        return user_resp

    def get_utterance_w_da(
        self, state: DialogSession, action=None, mode='train'
    ) -> "Tuple[str, str]":
        user_resp = self.get_utterance(state, action, mode=mode)
        # extract da
        start_idx = user_resp.find("[")
        end_idx = user_resp.find("]")
        if start_idx == -1 or end_idx == -1:
            da = EmotionalSupportGame.U_FeelTheSame
        else:
            da = user_resp[start_idx + 1 : end_idx]
            user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
            if da not in self.dialog_acts:
                da = EmotionalSupportGame.U_FeelTheSame
        return da, user_resp


class PatientChatModel(PatientModel):
    def __init__(
        self,
        dialog_acts: List[str],
        inference_args: dict,
        backbone_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        max_hist_num_turns=5,
        infer_user_da = True,
    ):
        super().__init__(
            dialog_acts=dialog_acts,
            inference_args=inference_args,
            backbone_model=backbone_model,
            conv_examples=conv_examples,
            max_hist_num_turns=max_hist_num_turns,
            infer_user_da=infer_user_da,
        )
        self.inference_args = inference_args
        if self.infer_user_da:
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist."
        else:
            self.task_prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Patient in a counselling conversation with a therapist. 
            You can choose amongst the following actions during a conversation to respond to the Therapist:
            {" ".join([f"[{da}]" for da in self.dialog_acts])}
            """.replace(
                "\t", ""
            ).strip()
        self.new_task_prompt = "The following is a new conversation between a Therapist and a Patient (you)."
        self.prompt_examples = self.process_chat_exp()
        
        self.heuristic_args: dict = {
            "max_hist_num_turns": 2,
            "example_pred_turn": [[0, 2, 3, 4]],
        }
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self._process_chat_turns(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def _process_chat_turns(self, exp: DialogSession, max_hist_num_turns: int = -1, use_da: bool = True, use_role: bool = True):
        """``exp`` as chat messages with the patient -- the simulator itself -- as assistant.

        Guarded on the raw history, not ``len(exp)``: that counts turns, so a state ending
        mid-turn, which is exactly what the simulator is asked about, read as empty and
        dropped the conversation from the prompt.
        """
        if len(exp.history) == 0:
            return []

        prompt_messages = []
        for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            if not use_role:
                content = f"{utt}".strip()
            elif role == EmotionalSupportGame.SYS or not use_da:
                content = f"{role}: {utt}".strip()
            else:
                content = f"{role}: [{da}] {utt}".strip()
            speaker = "user" if role == EmotionalSupportGame.SYS else "assistant"
            prompt_messages.append({"role": speaker, "content": content})
        return prompt_messages

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == EmotionalSupportGame.SYS  # next turn is user's turn
        if self.infer_user_da:
            messages = [
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": "You are the patient who is looking for the help from the therapist, because you have the emotional issue about %s regarding %s. Please reply with only one short and succinct sentence." % (state.emotion_type, state.problem_type)}
            ]
            state_ = state.copy()
            state_.history = state_.history[1:]
            messages += self._process_chat_turns(
                state_, max_hist_num_turns=self.max_hist_num_turns, use_da=False, use_role=False,
            )
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": f"{self.new_task_prompt}\n{esc_patient_scenario(state)}"},
            ]
            messages += self._process_chat_turns(
                state, max_hist_num_turns=self.max_hist_num_turns,
            )

        # copy, never mutate: build_agents shares self.inference_args across every player, so
        # assigning into it made one mode!='train' call switch the whole run to greedy decoding.
        gen_args = dict(self.inference_args)
        if mode != 'train':
            gen_args['temperature'] = 0.0
        data = self.backbone_model.chat_generate(messages, **gen_args)
        user_resp = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{EmotionalSupportGame.USR}:",
            user_role=f"{EmotionalSupportGame.SYS}:",
        )[0]
        return user_resp

    def get_utterance_from_batched_states(
        self, states: List[DialogSession], action=None
    ) -> List[str]:
        assert all([state[-1][0] == EmotionalSupportGame.SYS for state in states])
        all_prompts = []
        for state in states:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": self.new_task_prompt},
            ]
            messages += self._process_chat_turns(
                state, max_hist_num_turns=self.max_hist_num_turns
            )
            all_prompts.append(messages)
        datas = self.backbone_model.chat_generate_batched(
            all_prompts, **self.inference_args
        )
        user_resps = []
        for data in datas:
            user_resp = self.backbone_model._cleaned_chat_resp(
                data,
                assistant_role=f"{EmotionalSupportGame.USR}:",
                user_role=f"{EmotionalSupportGame.SYS}:",
            )
            user_resps.append(user_resp[0])
        return user_resps

    def get_utterance_w_da_from_batched_states(
        self, states: List[DialogSession], action=None
    ):
        split = [
            split_da(resp, self.dialog_acts, EmotionalSupportGame.U_FeelTheSame)
            for resp in self.get_utterance_from_batched_states(states, action)
        ]
        das = [da for da, _ in split]
        user_resps = [utt for _, utt in split]
        return das, user_resps

    def _heuristics_qa_pair(self, dialog: DialogSession):
        """One (dialogue, act) demonstration for predict_da: the turns as a single question,
        answered by the act the closing patient turn actually carried."""
        if len(dialog) == 0:
            return []
        assert dialog[0][0] == EmotionalSupportGame.SYS
        assert dialog[-1][0] == EmotionalSupportGame.USR

        lines = [f"{role}: {utt}".strip() for role, _da, utt in dialog]
        lines.append(f"{dialog.USR} feeling:")
        return [
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": f"{dialog[-1][1]}"},
        ]

    def _heuristics_window(self, dialog: DialogSession, pred_end_idx=-1):
        """``dialog`` cut to the last few turns, ending on the patient turn to predict."""
        max_history_length = self.heuristic_args["max_hist_num_turns"]
        if pred_end_idx == -1:
            pred_end_idx = len(dialog.history) - 1
        start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
        new_history = [turn for j, turn in enumerate(dialog) if start_idx <= j <= pred_end_idx]
        return DialogSession(dialog.SYS, dialog.USR).from_history(new_history)

    def process_heurstics_chat_exp(self, new_task_prompt: str):
        prompt_exps = []
        for i, exp in enumerate(self.conv_examples):
            for pred_end_turn in self.heuristic_args["example_pred_turn"][i]:
                window = self._heuristics_window(exp, pred_end_turn * 2 + 1)
                prompt_exps += self._heuristics_qa_pair(window)
                prompt_exps.append({"role": "system", "content": new_task_prompt})
        return prompt_exps[:-1]

    def predict_da(self, state: DialogSession, never_end=True) -> str:
        """The patient act for the last user turn, by majority vote over 5 samples.

        ``never_end`` keeps the terminal act out of the vote so a live chat ends when the
        human says so, not when the classifier does.
        """
        assert state[-1][0] == EmotionalSupportGame.USR

        messages = [
            {"role": "system", "content": self.critic_task_prompt},
            *self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
            {"role": "system", "content": self.new_task_prompt},
        ]
        messages += self._heuristics_qa_pair(self._heuristics_window(state, -1))[:-1]

        inf_args = {
            "max_new_tokens": 5,
            "temperature": 0.7,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 5,
        }
        datas = self.backbone_model.chat_generate(messages, **inf_args)
        sampled_das: list = []
        for resp in datas:
            user_da = resp["generated_text"].strip()
            if user_da not in self.dialog_acts:
                sampled_das.append(EmotionalSupportGame.U_FeelTheSame)
            if never_end:
                if user_da == EmotionalSupportGame.U_Solved:
                    sampled_das.append(EmotionalSupportGame.U_FeelBetter)
                else:
                    sampled_das.append(user_da)
            else:
                sampled_das.append(user_da)
        logger.info(f"sampled das: {sampled_das}")
        user_da = Counter(sampled_das).most_common(1)[0][0]
        return user_da