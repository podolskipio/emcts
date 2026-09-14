import logging, re
import numpy as np
import torch

from typing import List, Tuple

from players.planner import DialogPlanner
from utils.sessions import DialogSession
from utils.gen_models import GenerationModel, DialogModel
from utils.rewards import reward_dict
from games import CBGame
from players.prompting import parse_das, recent_turns


_CB_DESC_MAX_CHARS = 1200  # longest listing in cb-valid.txt is 1419 chars; keep prompts bounded


def _clip_desc(desc) -> str:
    desc = " ".join(str(desc or "").split())
    return desc if len(desc) <= _CB_DESC_MAX_CHARS else desc[:_CB_DESC_MAX_CHARS].rstrip() + " ..."


def cb_buyer_scenario(state) -> str:
    return (
        "You are the buyer who is trying to buy the %s at the lowest price you can, "
        "and you cannot pay more than %s. Product description: %s"
        % (getattr(state, "item_name", "item"), getattr(state, "buyer_price", "?"),
           _clip_desc(getattr(state, "buyer_item_description", "")))
    )


def cb_seller_scenario(state) -> str:
    return (
        "You are the seller who is trying to sell the %s at the highest price you can, "
        "and it is listed at %s. Product description: %s"
        % (getattr(state, "item_name", "item"), getattr(state, "seller_price", "?"),
           _clip_desc(getattr(state, "seller_item_description", "")))
    )


logger = logging.getLogger(__name__)


def _mean_reward(sampled_das) -> float:
    """Mean reward over the seller acts that carry one; 0.0 if none of them do."""
    scores = [reward_dict['cb'][da] for da in sampled_das if da in reward_dict['cb']]
    return float(np.mean(scores)) if scores else 0.0


class CBSystemPlanner(DialogPlanner):
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
        Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating with a seller to buy an item on an online marketplace.
        You can choose amongst the following actions during a conversation to respond to the seller:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is an example conversation between a Buyer and a Seller.
        {self.process_exp()}
        The following is a new conversation between another Buyer and Seller.
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
                [1 if da == CBGame.S_Greet else 0 for da in self.dialog_acts]
            )
        return np.array([1 for _ in self.dialog_acts])

    def get_utterance(self, state, action) -> str:
        return ""  # should not be called

    def predict(self, state: DialogSession, policy=None, ent_bound=None) -> "Tuple[np.ndarray, float]":
        if len(state) == 0:
            prompt = f"""
            {self.task_prompt}
            {CBGame.SYS}:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {state.to_string_rep(keep_sys_da=True)}
            {CBGame.SYS}:
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
        """Ask the seller simulator whether it would close the deal now; score its answer."""
        assert state[-1][0] == CBGame.USR
        prompt = f"""
        The following is background information about the task.
        A Buyer is negotiating with a Seller to buy an item on an online marketplace.
        The Seller can choose amongst the following actions during a conversation to respond to the Buyer:
        {" ".join([f"[{da}]" for da in self.user_dialog_acts])}
        The following is a conversation between a Buyer and a Seller about an item for sale.
        {self.process_exp(keep_sys_da=False, keep_user_da=True)}
        The following is a new conversation between another Buyer and Seller.
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
        Buyer: Alright, do we have a deal then?
        Seller:
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

        logger.debug(f"seller prompt: {prompt}")
        logger.debug(f"sampled das: {sampled_das}")
        return _mean_reward(sampled_das), sampled_das


class CBChatSystemPlanner(CBSystemPlanner):
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
        neg_reward = -1.0,
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
        Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating with a seller to buy an item on an online marketplace.
        You can choose amongst the following actions during a conversation to respond to the seller:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is an example conversation between a Buyer and a Seller.
        """.replace(
            "\t", ""
        ).strip()
        self.new_task_prompt = "The following is a new conversation between Buyer (you) and a Seller."
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
        self.neg_reward = neg_reward
        return

    def process_chat_exp(
        self,
        new_task_prompt,
        assistant_role=CBGame.SYS,
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
        assistant_role=CBGame.SYS,
        max_hist_num_turns: int = -1,
    ):
        """``exp`` as chat messages, with whichever speaker ``assistant_role`` names cast as
        the assistant. Guarded on the raw history, not ``len(exp)``: that counts turns, so a
        state ending mid-turn read as empty and dropped the conversation from the prompt."""
        if len(exp.history) == 0:
            return []
        assert exp[0][0] == CBGame.SYS  # the conversation always opens with the Buyer

        prompt_messages = []
        for _i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            keep_da = keep_sys_da if role == CBGame.SYS else keep_user_da
            content = f"{role}: [{da}] {utt}" if keep_da else f"{role}: {utt}"
            prompt_messages.append({
                "role": "assistant" if role == assistant_role else "user",
                "content": content.strip(),
            })
        return prompt_messages

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
                    {"role": "user", "content": f"{CBGame.USR}: Hello."}
                )
            else:
                assert state[-1][0] == CBGame.USR
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
        """Read the deal verdict out of a free-text answer, for the zero-shot prompt that
        asks the question in plain English instead of with act labels."""
        pred_da = []
        for resp in data:
            resp = resp['generated_text'].lower()
            if 'have no' in resp:
                pred_da.append(CBGame.U_No_deal)
            else:
                pred_da.append(CBGame.U_Deal)
        return pred_da

    def heuristic(self, state: DialogSession) -> float:
        """Ask whether the two sides have closed; score the deal price against the spread."""
        assert state[-1][0] == CBGame.USR
        if not self.infer_user_da:
            user_task_prompt = f"""
            Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, also extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.
            The following is an example conversation between a Buyer and a Seller.
            """.replace(
                "\t", ""
            ).strip()
            user_new_task_prompt = "The following is a new conversation between a Buyer and a Seller."

            messages = [
                {"role": "system", "content": user_task_prompt},
                *self.process_chat_exp(
                    new_task_prompt=user_new_task_prompt,
                    assistant_role=CBGame.USR,
                    keep_sys_da=False,
                    keep_user_da=False,
                ),
                {"role": "system", "content": user_new_task_prompt},
            ]
            messages += self._process_chat_turns(
                state,
                assistant_role=CBGame.USR,
                keep_sys_da=False,
                keep_user_da=False,
                max_hist_num_turns=self.user_max_hist_num_turns,
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Question: Have the Buyer and the Seller reached a deal? Answer: ",
                }
            )
        else:
            conversation = self._process_chat_turns(
                state,
                assistant_role=CBGame.USR,
                keep_sys_da=False,
                keep_user_da=False,
                max_hist_num_turns=self.user_max_hist_num_turns,
            )
            dial = "".join("\n{}".format(turn['content']) for turn in conversation)
            messages = [
                {"role": "system", "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."},
                {"role": "user", "content": "Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15.\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal.\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % dial}
            ]

        data = self.generation_model.chat_generate(messages, **self.eval_args)
        logger.info(f"deal-eval prompt: {messages}")

        # The answer is free text ("They have reached a deal at $15"), so read the verdict and
        # the price out of the wording. Any sample saying no deal vetoes the whole batch.
        deals, rewards, sampled_das = [], [], []
        for resp in data:
            text = resp['generated_text'].lower()
            if 'have not' in text:
                deals.append(-1)
                sampled_das.append('no deal')
            elif 'have reached' in text:
                deals.append(1)
                sampled_das.append('deal')

            prices = re.findall(r"[-+]?\d*\.?\d+", resp['generated_text'].replace(",", ""))
            if prices:
                deal_price = float(prices[0])
                rewards.append((deal_price - state.seller_price) / (state.buyer_price - state.seller_price))

        if -1 in deals:
            v = self.neg_reward
        elif not rewards:
            v = 0
        else:
            v = max(set(rewards), key=rewards.count)  # modal price across the samples
        logger.info(f"sampled das to v: {v}")
        return float(v), sampled_das


class BuyerModel(DialogModel):
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
        # so the seller's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        self.da_prompts_mapping = {
            CBGame.S_Greet: 'Please say hello or chat randomly.',
            CBGame.S_Inquire: 'Please ask any question about product, year, price, usage, etc.',
            CBGame.S_Information: 'Please provide information about the product, year, usage, etc.',
            CBGame.S_Propose: 'Please initiate a price or a price range for the product.',
            CBGame.S_Counter: 'Please propose a new price or a new price range.',
            CBGame.S_Counter_noprice: 'Please propose a vague price by using comparatives with existing price.',
            CBGame.S_Confirm: 'Please ask a question about the information to be confirmed.',
            CBGame.S_Affirm: 'Please give an affirmative response to a confirm.',
            CBGame.S_Deny: 'Please give a negative response to a confirm.',
            CBGame.S_Agree: 'Please agree with the proposed price.',
            CBGame.S_Disagree: 'Please disagree with the proposed price.'
        }
        self.dialog_acts = sorted([da for da in dialog_acts if da in self.da_prompts_mapping])

        logger.debug(self.dialog_acts)
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating with a seller to buy an item on an online marketplace.
        You can choose amongst the following actions during a conversation to respond to the seller:
        {" ".join([f"[{da}]" for da in self.dialog_acts])}
        The following is an example conversation between a Buyer and a Seller.
        {self.process_exp()}
        The following is a new conversation between another Buyer and Seller.
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "do_sample": False,  # otherwise the tree never reaches the next level
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
            if role == CBGame.SYS:
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
            {CBGame.SYS}:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {self._process_turns(state, max_hist_num_turns=self.max_hist_num_turns)}
            {da_prompt}
            {CBGame.SYS}:
            """
        prompt = prompt.replace("\t", "").strip()
        data = self.backbone_model.generate(prompt, **self.inference_args)
        return self.backbone_model._cleaned_resp(data, prompt)[0]

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class BuyerChatModel(BuyerModel):
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
        # so the seller's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        if self.infer_user_da:
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game."
        else:
            self.task_prompt = """
            Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating with a seller to buy an item on an online marketplace.
            You are the buyer who is trying to buy the item at the lowest price you can while still closing a deal.
            The following is an example conversation between a Buyer and a Seller.
            """.replace(
                "\t", ""
            ).strip()
            self.new_task_prompt = "The following is a new conversation between Buyer (you) and another Seller.\nThe Buyer greets the Seller."
            self.prompt_examples = self.process_chat_exp()
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self._process_chat_turns(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def _process_chat_turns(self, exp: DialogSession, da_prompt: str = '', max_hist_num_turns: int = -1, use_role: bool = True):
        """``exp`` as chat messages, each seller turn followed by the instruction for the buyer
        turn that answers it. The final seller turn is the one being answered now, so it takes
        ``da_prompt`` -- the act the planner just chose.

        Guarded on the raw history, not ``len(exp)``: that counts turns, so a state ending
        mid-turn read as empty and dropped the conversation from the prompt.
        """
        if len(exp.history) == 0:
            return []
        assert exp[0][0] == CBGame.SYS  # the conversation always opens with the Buyer

        prompt_messages = []
        for i, (role, da, utt) in recent_turns(exp, max_hist_num_turns):
            prefix = f"{role}: " if use_role else ""
            if role == CBGame.SYS:
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
                {"role": "user", "content": "You are the buyer who is trying to buy the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. %s Now start the game." % (state.item_name, state.buyer_price, state.buyer_item_description, da_prompt)}
            ]
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": f"{self.new_task_prompt}\n{cb_buyer_scenario(state)}"},
            ]
        if len(state) == 0:
            content = f"Hello.\n{da_prompt}" if self.infer_user_da else f"{CBGame.USR}: Hello.\n{da_prompt}"
            messages.append(
                {"role": "user", "content": content,}
            )
        else:
            assert state[-1][0] == CBGame.USR
            messages += self._process_chat_turns(
                state, da_prompt, max_hist_num_turns=self.max_hist_num_turns, use_role=not self.infer_user_da
            )
        gen_args = {**self.inference_args, "num_return_sequences": batch}
        if mode != 'train':
            gen_args['temperature'] = 0.0
        data = self.backbone_model.chat_generate(messages, **gen_args)
        sys_resps = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{CBGame.SYS}:",
            user_role=f"{CBGame.USR}:",
        )
        return sys_resps

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class SellerModel(DialogModel):
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
        self.task_prompt = f"""
		Now enter the role-playing mode. In the following conversation, you will play as a Seller negotiating with a Buyer who wants to buy your item on an online marketplace.
        You are the seller who wants to sell the item at the highest price you can while still closing a deal.
		The Seller (you) can choose amongst the following actions during a conversation to respond to the Buyer:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation.
		{self.process_exp()}
		The following is a new conversation between another Buyer and Seller.
		"""
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = inference_args
        # True drops the few-shot demo and the [act] tags from this agent's prompts,
        # so the seller's reply carries no act for anyone to read off it.
        self.infer_user_da = infer_user_da
        return

    def process_exp(self):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
        return prompt_exps.strip()

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == CBGame.SYS
        prompt = f"""
        {self.task_prompt}
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
        {CBGame.USR}:
        """
        prompt = prompt.replace("\t", "").strip()
        data = self.backbone_model.generate(prompt, **self.inference_args)
        return self.backbone_model._cleaned_resp(data, prompt)[0]

    def get_utterance_w_da(self, state: DialogSession, action=None, mode: str = 'train') -> Tuple[str, str]:
        raise NotImplementedError


class SellerChatModel(SellerModel):
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
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game."
        else:
            self.task_prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Seller negotiating with a Buyer who wants to buy your item on an online marketplace.
            You can choose amongst the following actions during a conversation to respond to the Buyer:
            {" ".join([f"[{da}]" for da in self.dialog_acts])}
            """.replace(
                "\t", ""
            ).strip()
        self.new_task_prompt = "The following is a new conversation between a buyer and a seller (you)."
        self.prompt_examples = self.process_chat_exp()
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self._process_chat_turns(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def _process_chat_turns(self, exp: DialogSession, max_hist_num_turns: int = -1, use_da: bool = True, use_role: bool = True):
        """``exp`` as chat messages with the seller -- the simulator itself -- as assistant.

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
            elif role == CBGame.SYS or not use_da:
                content = f"{role}: {utt}".strip()
            else:
                content = f"{role}: [{da}] {utt}".strip()
            speaker = "user" if role == CBGame.SYS else "assistant"
            prompt_messages.append({"role": speaker, "content": content})
        return prompt_messages

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == CBGame.SYS  # next turn is user's turn
        if self.infer_user_da:
            messages = [
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": "You are the seller who is trying to sell the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. Are you ready to play the game?" % (state.item_name, state.seller_price, state.seller_item_description)},
                {"role": "assistant", "content":"Yes, I'm ready to play the game!"}
            ]
            state_ = state.copy()
            messages += self._process_chat_turns(
                state_, max_hist_num_turns=self.max_hist_num_turns, use_da=False, use_role=False,
            )
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": f"{self.new_task_prompt}\n{cb_seller_scenario(state)}"},
            ]
            messages += self._process_chat_turns(
                state, max_hist_num_turns=self.max_hist_num_turns,
            )

        gen_args = dict(self.inference_args)
        if mode != 'train':
            gen_args['temperature'] = 0.0
        data = self.backbone_model.chat_generate(messages, **gen_args)
        user_resp = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{CBGame.USR}:",
            user_role=f"{CBGame.SYS}:",
        )[0]
        # The few-shot seller turns render as "Seller: [no deal] <text>", so the model emits the
        # tag too; left in, it leaks into the stored utterance and into every later prompt.
        # Stripped here, not in get_utterance_w_da, because CBGame.get_next_state takes the
        # utterance from this method and the label from the critic.
        start_idx, end_idx = user_resp.find("["), user_resp.find("]")
        if 0 <= start_idx < end_idx:
            tag = user_resp[start_idx + 1:end_idx].strip()
            if tag.lower() in {da.lower() for da in self.dialog_acts}:
                user_resp = user_resp.replace(f"[{tag}]", "", 1).strip()
        return user_resp

    def get_utterance_w_da(self, state: DialogSession, action=None, mode: str = 'train') -> Tuple[str, str]:
        """Generate the seller's reply and classify whether they just accepted the deal.
        """
        user_resp = self.get_utterance(state, action, mode=mode)
        # build the truncated dialog snippet (with the just-generated seller turn appended)
        snippet_state = state.copy()
        snippet_state.add_single(CBGame.USR, None, user_resp)
        snippet = snippet_state.to_string_rep(
            keep_user_da=False, max_turn_to_display=self.max_hist_num_turns,
        ).strip()
        messages = [
            {"role": "system",
             "content": "You decide whether a Seller has just agreed to close a deal with a Buyer in a price-negotiation dialogue."},
            {"role": "user",
             "content": (
                "You can only reply with YES or NO. "
                "YES means the Seller has accepted the Buyer's offer (or agreed on a price). "
                "NO means the Seller has not yet accepted.\n\n"
                f"Conversation:\n{snippet}\n\n"
                "Has the Seller just agreed to close the deal? Answer:"
            )},
        ]
        classify_args = {**self.inference_args, "num_return_sequences": 1, "temperature": 0.0}
        data = self.backbone_model.chat_generate(messages, **classify_args)
        answer = self.backbone_model._cleaned_chat_resp(data, assistant_role="", user_role="")[0]
        da = CBGame.U_Deal if answer.strip().lower().startswith("yes") else CBGame.U_No_deal
        return da, user_resp

    def get_utterance_from_batched_states(
        self, states: List[DialogSession], action=None
    ) -> List[str]:
        assert all([state[-1][0] == CBGame.SYS for state in states])
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
                assistant_role=f"{CBGame.USR}:",
                user_role=f"{CBGame.SYS}:",
            )
            user_resps.append(user_resp[0])
        return user_resps
