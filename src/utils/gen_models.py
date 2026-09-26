import requests
import logging
import re
import numpy as np
import torch
import openai
import os
import multiprocessing as mp
import nltk
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from typing import List, Tuple, Dict
from utils.sessions import DialogSession
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed  # for exponential backoff
from utils.utils import hashabledict
from utils import role_profiler, coupling

logger = logging.getLogger(__name__)


def _softmax(logprobs: "np.ndarray", temperature: float = 1.0) -> "np.ndarray":
    scaled = np.asarray(logprobs, dtype=float) / temperature
    shifted = np.exp(scaled - scaled.max())
    return shifted / shifted.sum()


@dataclass
class LabelScores:
    """The result of scoring a closed label set against one prompt.

    ``probs`` is the softmax over each label's full sequence logprob -- i.e. P(label | prompt,
    the answer is one of these labels), which is the quantity the sampling path was estimating
    by histogram. ``first_token_probs`` is the same normalization applied to first tokens only;
    it is carried for diagnostics, since the two agreeing is the evidence that the prompt makes
    the label set explicit enough.
    """
    labels: "List[str]"
    logprobs: "np.ndarray"
    probs: "np.ndarray"
    first_token_probs: "np.ndarray"
    temperature: float = 1.0
    #: Share of the model's own next-token probability mass (over the top-N tokens it was asked
    #: for) that falls on the label set. Near 1.0 means renormalizing over the labels throws
    #: nothing away, i.e. the prompt already makes the label set explicit. None unless
    #: ``score_labels(..., top_logprobs_num=N)`` asked for it.
    label_set_mass: "float | None" = None

    def at_temperature(self, temperature: float) -> "LabelScores":
        """The same scores read at a different sampling temperature.

        Only approximately what temperature does to a *multi-token* label -- sampling at T is
        per-token, ``prod_t softmax(logits_t / T)``, not ``softmax(sum_t logprob_t / T)``. The
        two coincide on the first token and the labels here are near-deterministic after it
        (first-token and full-sequence distributions correlate at r > 0.999 on p4g), so the
        approximation is tight where it is used.
        """
        return LabelScores(
            labels=self.labels,
            logprobs=self.logprobs,
            probs=_softmax(self.logprobs, temperature),
            first_token_probs=_softmax(np.log(np.clip(self.first_token_probs, 1e-30, None)), temperature),
            temperature=temperature,
            label_set_mass=self.label_set_mass,
        )

    def prob_of(self, label: str) -> float:
        return float(self.probs[self.labels.index(label)])

    def expectation(self, values: "Dict[str, float]") -> float:
        """Expected value of a per-label score under ``probs``. Labels absent from ``values``
        contribute nothing and their mass is dropped, matching the sampling path, which
        discards samples whose DA is not in ``reward_dict``."""
        return float(sum(p * values[l] for l, p in zip(self.labels, self.probs) if l in values))


class GenerationModel(ABC):
    # used to generate text in general. e.g. could be using API, or local model
    @abstractmethod
    def generate(self, input_text, **gen_args):
        """
        Generate text from the model.
        """
        raise NotImplementedError

    def chat_generate(self, messages, **gen_args):
        """
        Generate text from the model. Used for chatbot.
        """
        raise NotImplementedError

    def chat_generate_batched(self, messages_list, **gen_args):
        """
        Generate text from the model when you have multiple message histories
        """
        raise NotImplementedError

    def score_labels(self, messages, labels, prefill="", close="", **kwargs):
        """Score a fixed set of label strings by likelihood instead of generating them.
        """
        raise NotImplementedError

    def supports_label_scoring(self) -> bool:
        return False

    @staticmethod
    def _normalize_chat_messages(messages: "List[Dict]") -> "List[Dict]":
        """Flatten a message list so it survives single-system-slot chat templates.

        Every prompt builder in ``players/`` separates the few-shot demo from the live
        conversation with a *mid-conversation* ``{"role": "system"}`` marker
        ("The following is a new conversation between ..."). That works on the OpenAI API,
        which keeps such a message in place, but not on local servers:

        * SGLang renders vicuna through ``vicuna_v1.1``, whose ``generate_chat_conv`` does
          ``conv.system_message = message.content`` for *every* system message. The last one
          wins and is hoisted to the top, so the **leading task prompt is silently dropped**
          (role description + the list of dialog acts) and the demo runs straight into the live
          conversation with no boundary. Verified against the server: with two system messages
          the model cannot recall anything from the first one.
        * Ollama concatenates all system messages at the top, so it keeps the text but still
          loses the *position* of the separator.

        Normalizing here makes both backends see the prompt the builders intended, and makes
        the two comparable:

        1. leading system messages are merged into the one system message the template has a
           slot for;
        2. a later system message is folded into the text of the next non-system message (it
           introduces what follows), or the previous one if it is trailing;
        3. consecutive same-role messages are merged, so user/assistant strictly alternate --
           ``ADD_COLON_TWO`` picks its separator by message index, so a repeated role would
           otherwise desynchronize the ``</s>`` turn boundaries.
        """
        system_prefix: List[str] = []
        body: List[Dict] = []
        pending_system: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "system":
                if not body:
                    if content:
                        system_prefix.append(content)  # still in the leading run
                else:
                    if content:
                        pending_system.append(content)  # separator: attach to what follows
                continue
            if pending_system:
                content = "\n".join(pending_system + ([content] if content else [])).strip()
                pending_system = []
            body.append({"role": role, "content": content})
        if pending_system:
            # trailing separator with nothing after it: keep the text on the last turn
            if body:
                body[-1] = {**body[-1], "content": f"{body[-1]['content']}\n" + "\n".join(pending_system)}
            else:
                system_prefix.extend(pending_system)

        merged: List[Dict] = []
        for msg in body:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1] = {"role": msg["role"], "content": f"{merged[-1]['content']}\n{msg['content']}".strip()}
            else:
                merged.append(dict(msg))

        out: List[Dict] = []
        if system_prefix:
            out.append({"role": "system", "content": "\n".join(system_prefix)})
        out.extend(merged)
        return out

    def _cleaned_resp(self, data, prompt) -> "List[str]":
        # default helper function to clean extract the generated text from the returned json
        logger.debug("promopt:")
        logger.debug(prompt)
        cleaned_resps = []
        for gen_resp in data:
            logger.debug("raw response:")
            logger.debug(gen_resp['generated_text'])
            cleaned_resp = gen_resp['generated_text'].strip()
            if "\n" in cleaned_resp:
                cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
            logger.debug(f"cleaned response: {cleaned_resp}")
            cleaned_resps.append(cleaned_resp)
        return cleaned_resps

    def _cleaned_chat_resp(self, data, assistant_role="Persuader:", user_role="Persuadee:") -> "List[str]":
        # remove the user_role and keep the assistant_role
        # default helper function to clean extract the generated text from the returned json
        cleaned_resps = []
        for gen_resp in data:
            logger.debug("raw response:")
            logger.debug(gen_resp['generated_text'])
            cleaned_resp = gen_resp['generated_text'].strip()
            if "\n" in cleaned_resp:
                cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
            if assistant_role in cleaned_resp:
                cleaned_resp = cleaned_resp[cleaned_resp.index(assistant_role) + len(assistant_role):].strip()
            if user_role in cleaned_resp:
                cleaned_resp = cleaned_resp[:cleaned_resp.index(user_role)].strip()
            logger.debug(f"cleaned response: {cleaned_resp}")
            cleaned_resps.append(cleaned_resp)
        return cleaned_resps


class DialogModel(ABC):
    # used to play DialogGame
    def __init__(self):
        self.dialog_acts = []
        return

    @abstractmethod
    def get_utterance(self, state: DialogSession, action, mode: str = 'train') -> str:
        raise NotImplementedError

    def get_utterance_batched(self, state: DialogSession, action: int, batch: int, mode: str = 'train') -> List[str]:
        raise NotImplementedError

    def get_utterance_from_batched_states(self, states: List[DialogSession], action=None, mode: str = 'train') -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        # this is used for user agent. should not be used for system agent
        raise NotImplementedError

    def get_utterance_w_da_from_batched_states(self, states: List[DialogSession], action=None):
        # this is used for user agent. should not be used for system agent
        raise NotImplementedError

    def predict_da(self, state: DialogSession, never_end: bool = True) -> str:
        # this is used for user agent. should not be used for system agent
        raise NotImplementedError


class APIModel(GenerationModel):
    API_TOKEN = os.environ.get("HF_API_KEY")

    def __init__(self):
        # self.API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
        self.API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
        self.headers: dict[str, str] = {"Authorization": f"Bearer {APIModel.API_TOKEN}"}
        self.inference_args = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "return_full_text": False
        }
        return

    def generate(self, input_text, **_args):
        data = {
            "inputs": input_text,
            "parameters": _args or self.inference_args
        }
        response = requests.post(self.API_URL, headers=self.headers, json=data)
        return response.json()


class OpenAIModel(GenerationModel):
    API_TOKEN = os.environ.get("OPENAI_API_KEY")

    def __init__(self, model_name="text-curie-001"):
        # check if model exists
        openai.api_key = OpenAIModel.API_TOKEN
        models = openai.Engine.list()
        if model_name not in [model.id for model in models.data]:
            raise ValueError(f"model {model_name} not found")

        self.inference_args = {
            "model": model_name,
            "max_tokens": 64,
            "temperature": 0.7,
            "echo": False,
            "n": 1,
            "stop": "\n"
        }
        return

    def _update_args(self, new_args):
        args = {**self.inference_args}
        from_cache = False
        if "max_new_tokens" in new_args:
            new_args["max_tokens"] = new_args.pop("max_new_tokens")
        if "return_full_text" in new_args:
            new_args["echo"] = new_args.pop("return_full_text")
        if "do_sample" in new_args:
            from_cache = not new_args.pop("do_sample")  # rely on caching
        if "num_return_sequences" in new_args:
            new_args["n"] = new_args.pop("num_return_sequences")
        if "repetition_penalty" in new_args:
            new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
        new_args.pop("top_k", None)  # not an OpenAI API parameter
        return from_cache, {**args, **new_args}

    @lru_cache(maxsize=None)
    def _cached_generate(**parameters):
        response = openai.Completion.create(**parameters)
        return response

    # tried custom implementation of waiting before request, but I think openai is lying about how it calculates the rate limit
    # takes 3 trials to reach 2^3=8. Then 7 * 8 = 56 sec max. Just to safe we wait a bit more than 10 times
    @retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
    def generate(self, input_text, **_args):
        from_cache, parameters = self._update_args(_args)
        parameters["prompt"] = input_text
        if from_cache:
            response = OpenAIModel._cached_generate(**parameters)
        else:
            response = openai.Completion.create(**parameters)

        # format to a common format
        gen_output = []
        for resp in response.choices:
            text = resp.text
            gen_output.append({"generated_text": text})
        return gen_output


class OpenAIChatModel(OpenAIModel):
    def __init__(self, model_name="gpt-3.5-turbo", gen_sentences=-1):
        # check if model exists
        openai.api_key = self.API_TOKEN

        self.inference_args = {
            "model": model_name,
            "max_tokens": 64,
            "temperature": 0.7,
            "n": 1,
            # "stop": "\n"  # no longer need since we are using chat
            # "echo": False,
        }
        self.gen_sentences = None if gen_sentences < 0 else gen_sentences
        return

    def _update_args(self, new_args):
        if "stop" in new_args:
            new_args.pop("stop")
        if "echo" in new_args:
            new_args.pop("echo")
        if "return_full_text" in new_args:
            new_args.pop("return_full_text")
        return super()._update_args(new_args)

    def generate(self, input_text, **_args):
        logging.info("It is recommended to use chat_generate instead of generate for OpenAIChatModel")
        messages = [{
            "role": "user",
            "content": input_text
        }]
        return self.chat_generate(messages, **_args)

    @lru_cache(maxsize=None)
    def _cached_generate(**parameters):
        parameters["messages"] = list(parameters["messages"])
        response = openai.chat.completions.create(**parameters)
        return response

    @retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(3))
    def chat_generate(self, messages: List[Dict], **gen_args):
        # generate in a chat format
        from_cache, parameters = self._update_args(gen_args)
        hashable_messages = [hashabledict(m) for m in messages]
        parameters["messages"] = hashable_messages
        if from_cache:
            parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
            response = OpenAIChatModel._cached_generate(**parameters)
        else:
            response = openai.chat.completions.create(**parameters)

        # format to a common format
        gen_output = []
        for resp in response.choices:
            text = resp.message.content
            if self.gen_sentences is not None:
                sentences = nltk.sent_tokenize(text)
                if len(sentences) > self.gen_sentences:
                    text = " ".join(sentences[:self.gen_sentences])
            gen_output.append({"generated_text": text})
        return gen_output

    def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
        pool = mp.Pool(processes=len(messages_list))
        results = []
        for messages in messages_list:
            results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
        pool.close()
        pool.join()
        return [r.get() for r in results]


class AzureOpenAIModel(OpenAIModel):
    API_TOKEN = os.environ.get("MS_OPENAI_API_KEY")
    API_BASE = os.environ.get("MS_OPENAI_API_BASE")
    API_TYPE = "azure"
    API_VERSION = "2022-12-01"

    def __init__(self, model_name="chatgpt-turbo"):
        # check if model exists
        openai.api_key = AzureOpenAIModel.API_TOKEN
        openai.api_base = AzureOpenAIModel.API_BASE
        openai.api_type = AzureOpenAIModel.API_TYPE
        openai.api_version = AzureOpenAIModel.API_VERSION

        self.inference_args = {
            "engine": model_name,
            "max_tokens": 64,
            "temperature": 0.7,
            "echo": False,
            "n": 1,
            "stop": "\n"
        }
        return


class AzureOpenAIChatModel(AzureOpenAIModel):
    def __init__(self, model_name="chatgpt", gen_sentences=-1):
        # check if model exists
        openai.api_key = self.API_TOKEN
        openai.api_base = self.API_BASE
        openai.api_type = self.API_TYPE
        openai.api_version = "2023-03-15-preview"

        self.inference_args = {
            "engine": model_name,
            "max_tokens": 64,
            "temperature": 0.7,
            "n": 1,
            # "stop": "\n"  # no longer need since we are using chat
            # "echo": False,
        }
        self.gen_sentences = None if gen_sentences < 0 else gen_sentences
        return

    def _update_args(self, new_args):
        if "stop" in new_args:
            new_args.pop("stop")
        if "echo" in new_args:
            new_args.pop("echo")
        if "return_full_text" in new_args:
            new_args.pop("return_full_text")
        return super()._update_args(new_args)

    @lru_cache(maxsize=None)
    def _cached_generate(**parameters):
        parameters["messages"] = list(parameters["messages"])
        response = openai.ChatCompletion.create(**parameters)
        return response

    @retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
    def chat_generate(self, messages: List[Dict], **gen_args):
        # generate in a chat format
        from_cache, parameters = self._update_args(gen_args)
        hashable_messages = [hashabledict(m) for m in messages]
        parameters["messages"] = hashable_messages
        if from_cache:
            parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
            response = AzureOpenAIChatModel._cached_generate(**parameters)
        else:
            response = openai.ChatCompletion.create(**parameters)

        # format to a common format
        gen_output = []
        for resp in response.choices:
            text = resp['message']['content']
            if self.gen_sentences is not None:
                sentences = nltk.sent_tokenize(text)
                if len(sentences) > self.gen_sentences:
                    text = " ".join(sentences[:self.gen_sentences])
            gen_output.append({"generated_text": text})
        return gen_output

    def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
        pool = mp.Pool(processes=len(messages_list))
        results = []
        for messages in messages_list:
            results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
        pool.close()
        pool.join()
        return [r.get() for r in results]

    def generate(self, input_text, **_args):
        messages = [{
            "role": "user",
            "content": input_text
        }]
        return self.chat_generate(messages, **_args)


class OllamaModel(GenerationModel):
    """Talk to a locally running `Ollama <https://ollama.com>`_ server.

    Start the server and make the model available first, e.g.::

        ollama serve              # often already running as a service
        ollama pull llama3.1      # pull the model once
        # then in code:  OllamaModel(model_name="llama3.1")

    The HTTP endpoint defaults to ``http://localhost:11434`` and can be overridden
    with the ``base_url`` argument or the ``OLLAMA_HOST`` environment variable.

    This is the completion variant (``/api/generate``). For chat-style usage see
    :class:`OllamaChatModel`. Note that Ollama has no native ``n`` parameter, so
    multiple samples are obtained by calling the server repeatedly (and just once,
    replicated, when sampling is disabled).
    """

    def __init__(self, model_name="llama3.1", base_url=None, request_timeout=600):
        self.model_name = model_name
        base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        # sanity check: is the server reachable, and does it know about this model?
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=10).json()
            available = {m.get("name", "") for m in tags.get("models", [])}
            available |= {n.split(":", 1)[0] for n in available}  # also match without the :tag suffix
            if available and model_name not in available and model_name.split(":", 1)[0] not in available:
                logger.warning(
                    f"ollama model '{model_name}' is not present on {self.base_url} "
                    f"(available: {sorted(n for n in available if n)}). "
                    f"It will need to be pulled (`ollama pull {model_name}`) before use."
                )
        except requests.RequestException as e:
            raise ConnectionError(
                f"could not reach an Ollama server at {self.base_url}. "
                f"Start it with `ollama serve` (and `ollama pull {model_name}`), "
                f"or point OLLAMA_HOST / base_url at the right address. Original error: {e}"
            )
        # defaults; same keys as the rest of the codebase, translated in _build_options
        self.inference_args = {
            "max_new_tokens": 64,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
        }
        return

    def _build_options(self, gen_args):
        """Translate the project-wide generation kwargs into Ollama ``options``.

        Returns ``(options, n, deterministic)`` where ``n`` is how many samples the
        caller wants and ``deterministic`` is True when sampling was disabled.
        """
        args = {**self.inference_args, **gen_args}
        n = 1
        deterministic = False
        options: dict = {}
        # number of samples (Ollama has no native `n`, so we loop)
        for k in ("num_return_sequences", "n"):
            if k in args:
                n = int(args.pop(k))
        if "do_sample" in args:
            deterministic = not args.pop("do_sample")
        # token budget
        for k in ("max_new_tokens", "max_tokens", "num_predict"):
            if k in args:
                options["num_predict"] = int(args.pop(k))
        if "temperature" in args:
            options["temperature"] = args.pop("temperature")
        if "repetition_penalty" in args:
            options["repeat_penalty"] = args.pop("repetition_penalty")
        if "top_p" in args:
            options["top_p"] = args.pop("top_p")
        if "top_k" in args:
            options["top_k"] = args.pop("top_k")
        if "seed" in args:
            options["seed"] = args.pop("seed")
        if "stop" in args:
            stop = args.pop("stop")
            if stop:
                options["stop"] = [stop] if isinstance(stop, str) else list(stop)
        # arguments that don't apply to a local server / completion endpoint
        for k in ("return_full_text", "echo", "model", "engine"):
            args.pop(k, None)
        # anything left over is passed through as-is (best effort; Ollama ignores unknown options)
        options.update(args)
        if deterministic:
            options["temperature"] = 0.0
        return options, n, deterministic

    def _post(self, path, payload):
        # single HTTP chokepoint for /api/generate and /api/chat, so the per-role cost profile
        # (utils/role_profiler, paper §W5) is recorded here: one record per request, including
        # each iteration of the n-samples loop, since Ollama has no native `n`.
        t0 = time.monotonic()
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        role_profiler.record(time.monotonic() - t0,
                             tokens_in=data.get("prompt_eval_count"),
                             tokens_out=data.get("eval_count"))
        return data

    def generate(self, input_text, **gen_args):
        options, n, deterministic = self._build_options(gen_args)
        payload = {"model": self.model_name, "prompt": input_text, "stream": False, "options": options}
        if deterministic:
            text = self._post("/api/generate", payload).get("response", "")
            return [{"generated_text": text} for _ in range(n)]
        return [{"generated_text": self._post("/api/generate", payload).get("response", "")} for _ in range(n)]


class OllamaChatModel(OllamaModel):
    """Chat-style access to a local Ollama model (uses ``/api/chat``).

    Mirrors :class:`OpenAIChatModel`: ``generate`` wraps the prompt in a single
    user message, ``chat_generate`` takes a list of ``{"role", "content"}`` dicts,
    and ``gen_sentences`` (>=0) truncates each response to that many sentences.
    """

    def __init__(self, model_name="llama3.1", base_url=None, request_timeout=600, gen_sentences=-1):
        super().__init__(model_name=model_name, base_url=base_url, request_timeout=request_timeout)
        self.gen_sentences = None if gen_sentences < 0 else gen_sentences
        return

    def _truncate_sentences(self, text):
        if self.gen_sentences is None:
            return text
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > self.gen_sentences:
            text = " ".join(sentences[:self.gen_sentences])
        return text

    def generate(self, input_text, **gen_args):
        return self.chat_generate([{"role": "user", "content": input_text}], **gen_args)

    def chat_generate(self, messages: List[Dict], **gen_args):
        options, n, deterministic = self._build_options(gen_args)
        messages = self._normalize_chat_messages(messages)
        payload = {"model": self.model_name, "messages": [dict(m) for m in messages], "stream": False, "options": options}

        def _one():
            data = self._post("/api/chat", payload)
            return self._truncate_sentences(data.get("message", {}).get("content", ""))

        if deterministic:
            text = _one()
            return [{"generated_text": text} for _ in range(n)]
        return [{"generated_text": _one()} for _ in range(n)]

    def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
        # threads rather than processes: these are HTTP waits, and per-role cost records made in a
        # child *process* would be lost. The active role is captured here and re-entered inside
        # each worker, since ContextVars do not cross thread boundaries.
        if len(messages_list) == 0:
            return []
        active_role = role_profiler.current_role()

        def _one(messages):
            with role_profiler.role(active_role):
                return self.chat_generate(messages, **dict(gen_args))

        pool = ThreadPool(processes=max(1, len(messages_list)))
        results = [pool.apply_async(_one, args=(messages,)) for messages in messages_list]
        pool.close()
        pool.join()
        return [r.get() for r in results]


class LocalModel(GenerationModel):
    def __init__(self, model_name="EleutherAI/gpt-neo-2.7B", input_max_len=512, stop_symbol="\n", cuda=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        stop_token_ids = self.tokenizer.encode(stop_symbol)[0]
        set_seed(42)
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.model = self.model.cuda()
        else:
            self.cuda = False

        self.input_max_len = input_max_len
        self.inference_args = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "eos_token_id": stop_token_ids,
            "pad_token_id": self.tokenizer.eos_token_id
            # "return_full_text": False  # not available for manual generation
        }

    def generate(self, input_text: str, **gen_args):
        # override if gen_args specified
        gen_params = {**self.inference_args, **gen_args}
        inputs = self.tokenizer([input_text], return_tensors='pt', truncation=True, max_length=self.input_max_len)
        if self.cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, **gen_params)
        gen_only_outputs = outputs[:, len(inputs['input_ids'][0]):]
        gen_resps = self.tokenizer.batch_decode(gen_only_outputs, skip_special_tokens=True)

        # format output
        gen_output = []
        for resp in gen_resps:
            gen_output.append({"generated_text": resp})
        return gen_output


@lru_cache(maxsize=None)
def _sglang_cached_chat_completion(client, **parameters):
    # deterministic (do_sample=False) requests are memoized so that repeated visits to the
    # same tree node don't hit the server again. messages/extra_body arrive hashable.
    parameters["messages"] = list(parameters["messages"])
    return client.chat.completions.create(**parameters)


class SGLangChatModel(GenerationModel):
    def __init__(
            self,
            model_name="TheBloke/vicuna-13B-v1.5-AWQ",
            gen_sentences=-1,
            base_url=None,
            request_timeout=600
    ):
        base_url = base_url or os.environ.get("SGLANG_HOST", "http://127.0.0.1:30000")
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.base_url = base_url
        self.client = openai.Client(base_url=self.base_url, api_key="None", timeout=request_timeout)
        # sanity check: is the server up, and is it serving the model we are about to ask for?
        try:
            served = [m.id for m in self.client.models.list().data]
        except Exception as e:
            raise ConnectionError(
                f"could not reach an SGLang server at {self.base_url}. Start it with "
                f"`python -m sglang.launch_server --model-path {model_name} --port 30000`, "
                f"or point SGLANG_HOST / base_url at the right address. Original error: {e}"
            )
        if served and model_name not in served:
            logger.warning(
                f"SGLang at {self.base_url} serves {served}, not '{model_name}'. Using '{served[0]}' instead."
            )
            model_name = served[0]

        self.inference_args = {
            "model": model_name,
            "max_tokens": 64,
            "temperature": 0.7,
            "n": 1,
            # "stop": "\n"  # no longer need since we are using chat
            # "echo": False,
        }
        self.gen_sentences = None if gen_sentences < 0 else gen_sentences
        return

    def _update_args(self, new_args):
        new_args = dict(new_args)  # never mutate the caller's dict (shared across batched workers)
        args = {**self.inference_args}
        from_cache = False
        # completion-only / HF-only arguments that the chat endpoint has no use for
        for k in ("stop", "echo", "return_full_text"):
            new_args.pop(k, None)
        if "max_new_tokens" in new_args:
            new_args["max_tokens"] = new_args.pop("max_new_tokens")
        if "do_sample" in new_args:
            from_cache = not new_args.pop("do_sample")  # greedy + memoized
        if "num_return_sequences" in new_args:
            new_args["n"] = new_args.pop("num_return_sequences")
        # neither repetition_penalty nor top_k is an OpenAI chat parameter, so the client would
        # reject them as unexpected kwargs; SGLang accepts both as extra fields.
        extra_body = dict(new_args.pop("extra_body", {}) or {})
        if "repetition_penalty" in new_args:
            # HF semantics (multiplicative, neutral 1.0) != OpenAI frequency_penalty (additive,
            # neutral 0.0), but SGLang accepts repetition_penalty itself as an extra field.
            repetition_penalty = new_args.pop("repetition_penalty")
            if repetition_penalty != 1.0:
                extra_body["repetition_penalty"] = repetition_penalty
        if "top_k" in new_args:
            extra_body["top_k"] = new_args.pop("top_k")
        if extra_body:
            new_args["extra_body"] = hashabledict(extra_body)
        parameters = {**args, **new_args}
        if from_cache:
            parameters["temperature"] = 0.0  # do_sample=False means greedy decoding
        return from_cache, parameters

    def generate(self, input_text, **_args):
        logging.info("It is recommended to use chat_generate instead of generate for SGLangChatModel")
        messages = [{
            "role": "user",
            "content": input_text
        }]
        return self.chat_generate(messages, **_args)

    @retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(3))
    def chat_generate(self, messages: List[Dict], **gen_args):
        # generate in a chat format
        from_cache, parameters = self._update_args(gen_args)
        # vicuna_v1.1 keeps only ONE system message and hoists it to the top, so the builders'
        # mid-conversation "new conversation" separators must be flattened first -- see
        # GenerationModel._normalize_chat_messages.
        hashable_messages = [hashabledict(m) for m in self._normalize_chat_messages(messages)]
        # --coupled_seeds: the request below is shared with every arm that sends it (utils/coupling.py);
        # a pass-through outside a coupled dialogue
        request = {"messages": hashable_messages, "parameters": {k: v for k, v in parameters.items()}}
        return coupling.call(request, lambda: self._chat_request(hashable_messages, from_cache, parameters))

    def _chat_request(self, hashable_messages, from_cache, parameters):
        parameters = dict(parameters)
        parameters["messages"] = hashable_messages
        t0 = time.monotonic()
        if from_cache:
            parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
            response = _sglang_cached_chat_completion(self.client, **parameters)
        else:
            response = self.client.chat.completions.create(**parameters)
        # per-role cost profile (utils/role_profiler, paper §W5). One record per request; unlike
        # Ollama a request carries all n samples, so `samples` is where the fan-out shows up.
        usage = getattr(response, "usage", None)
        role_profiler.record(time.monotonic() - t0,
                             tokens_in=getattr(usage, "prompt_tokens", 0),
                             tokens_out=getattr(usage, "completion_tokens", 0),
                             samples=len(response.choices))

        # format to a common format
        gen_output = []
        for resp in response.choices:
            text = resp.message.content or ""
            if self.gen_sentences is not None:
                sentences = nltk.sent_tokenize(text)
                if len(sentences) > self.gen_sentences:
                    text = " ".join(sentences[:self.gen_sentences])
            gen_output.append({"generated_text": text})
        return gen_output

    def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
        # threads, not processes: the openai client holds an httpx client (locks, sockets) that
        # cannot be pickled, and these calls are pure IO that SGLang batches server-side anyway.
        if len(messages_list) == 0:
            return []
        active_role = role_profiler.current_role()  # ContextVars do not cross thread boundaries
        active_coupling = coupling.current()

        def _one(messages):
            with role_profiler.role(active_role):
                return coupling.run_in(active_coupling, self.chat_generate, messages, **dict(gen_args))

        pool = ThreadPool(processes=len(messages_list))
        results = [pool.apply_async(_one, args=(messages,)) for messages in messages_list]
        pool.close()
        pool.join()
        return [r.get() for r in results]

    # ------------------------------------------------------------------
    # Logit scoring: score a fixed label set instead of generating it
    # ------------------------------------------------------------------
    # The value estimator and the policy prior both ask the model for a *label* out of a
    # closed set -- one of 5 persuadee donation acts, one of 13 persuader dialog acts -- and
    # then throw the rest of the generation away. Sampling N completions to histogram those
    # labels is the expensive way to read a distribution the model already computes in its
    # logits. These two methods read it directly.
    #
    # Why two HTTP requests rather than one:
    #
    #   * Request 1 prefills the shared prompt (`max_new_tokens=0`) and asks for the logprobs
    #     of each label's *first* token. It costs exactly a prefill, and it puts the prompt
    #     into the radix cache.
    #   * Request 2 sends one item per label -- prompt + the label's own tokens -- and reads
    #     the input logprobs of those tokens, giving the exact sequence logprob of each label.
    #
    #
    # First-token logprobs alone are *not* a safe shortcut, which is why request 2 exists.
    # They are only a good approximation when the label set is explicit in the prompt: on a
    # prompt that listed the labels, first-token and full-sequence scores agreed to ~0.1 nats;
    # on one that did not, they disagreed on the argmax ("no donation" leads on `no` but loses
    # once `ation` is charged for). `first_token_probs` is returned so that gap stays visible.

    @property
    def _native_url(self) -> str:
        # self.base_url ends with /v1 (OpenAI-compatible); /generate lives at the root.
        return self.base_url[:-len("/v1")].rstrip("/")

    def _get_tokenizer(self):
        if getattr(self, "_tokenizer", None) is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.inference_args["model"])
        return self._tokenizer

    def render_prompt(self, messages: List[Dict]) -> str:
        """Render a chat message list to the exact raw prompt the server would build.

        Only the templates this repo actually serves are implemented, because a wrong render
        is a silently wrong score. A tokenizer that ships a `chat_template` is authoritative;
        otherwise SGLang falls back to a built-in conversation template chosen by model name,
        and for vicuna that is `vicuna_v1.1` (SeparatorStyle.ADD_COLON_TWO) -- the same
        template whose single-system-slot behaviour fix 9 is about. Anything else refuses
        rather than guessing.
        """
        messages = self._normalize_chat_messages(messages)
        tokenizer = self._get_tokenizer()
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_name = self.inference_args["model"]
        if re.search(r"vicuna|llava-v1\.5|llava-next-video-7b", model_name, re.IGNORECASE) is None:
            raise NotImplementedError(
                f"label scoring needs the raw prompt, and '{model_name}' neither ships a "
                f"chat_template nor matches SGLang's vicuna template. Add its template to "
                f"render_prompt() before using --logit_scoring with it."
            )
        # sglang/srt/parser/conversation.py :: vicuna_v1.1 + Conversation.get_prompt()
        system_message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        turns = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]  # last one wins, exactly as generate_chat_conv does
            elif msg["role"] == "user":
                turns.append(("USER", msg["content"]))
            else:
                turns.append(("ASSISTANT", msg["content"]))
        turns.append(("ASSISTANT", None))  # the blank message generation continues from
        seps = (" ", "</s>")
        rendered = system_message + seps[0]
        for i, (speaker, content) in enumerate(turns):
            if content:
                rendered += speaker + ": " + content + seps[i % 2]
            else:
                rendered += speaker + ":"
        return rendered

    def supports_label_scoring(self) -> bool:
        try:
            self.render_prompt([{"role": "user", "content": "hi"}])
            return True
        except NotImplementedError:
            return False

    def _post_generate(self, payload: dict, samples: int, tokens_in: int):
        """One /generate call. ``tokens_in`` is passed in rather than read back from the
        response because the response over-counts: every item in the scoring batch reports the
        whole shared prefix as its own prompt, so summing them bills an ~1.2k-token prompt
        seven times for a request the radix cache prefills once. The caller knows how many
        tokens are actually new, and that is what the §W5 table wants next to a generation row
        that reports its prompt once."""
        t0 = time.monotonic()
        response = requests.post(f"{self._native_url}/generate", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        results = data if isinstance(data, list) else [data]
        role_profiler.record(
            time.monotonic() - t0,
            tokens_in=tokens_in,
            tokens_out=0,  # nothing is generated: max_new_tokens=0
            samples=samples,
        )
        return results

    def score_labels(self, messages: List[Dict], labels: List[str], prefill: str = "", close: str = "",
                     top_logprobs_num: int = 0, **kwargs):
        """P(label | prompt) for each label in ``labels``, by logprob rather than by sampling.

        ``prefill`` is text the assistant is taken to have already written before the label
        (e.g. ``"Persuadee: ["``); ``close`` is what follows it (e.g. ``"]"``). Scoring the
        closing bracket matters: without it a label that is a token-prefix of the model's
        preferred continuation is over-credited.

        ``top_logprobs_num`` > 0 additionally returns how much of the model's own next-token
        mass the label set accounts for (``label_set_mass``). Renormalizing over a closed label
        set is only honest if the model was going to answer inside that set anyway; this is the
        measurement of that, and it is off by default because it is a diagnostic, not something
        the search needs on every node.
        """
        if not labels:
            raise ValueError("score_labels needs at least one label")
        tokenizer = self._get_tokenizer()
        prefix = self.render_prompt(messages)
        if prefill:
            prefix = prefix + " " + prefill  # ADD_COLON_TWO writes "ASSISTANT: <content>"
        prefix_ids = tokenizer(prefix).input_ids

        label_ids = []
        for label in labels:
            full_ids = tokenizer(prefix + label + close).input_ids
            if full_ids[: len(prefix_ids)] != prefix_ids:
                # the label merged into the last prefix token, so "prefix tokens + label
                # tokens" is not the tokenization the model would actually see.
                raise ValueError(
                    f"label {label!r} re-tokenizes the prefix boundary; end `prefill` on a "
                    f"character that does not merge (e.g. '[')."
                )
            label_ids.append(full_ids[len(prefix_ids):])

        # Request 1: prefill the shared prompt, and read each label's first-token logprob.
        first_ids = sorted({ids[0] for ids in label_ids})
        warm_payload = {
            "input_ids": prefix_ids,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": -1,
            "token_ids_logprob": first_ids,
        }
        if top_logprobs_num:
            warm_payload["top_logprobs_num"] = top_logprobs_num
        warm = self._post_generate(warm_payload, samples=1, tokens_in=len(prefix_ids))[0]
        first_logprob = {tok_id: lp for lp, tok_id, _ in warm["meta_info"]["output_token_ids_logprobs"][0]}
        label_set_mass = None
        if top_logprobs_num:
            top = warm["meta_info"]["output_top_logprobs"][0]
            in_set = {tok_id for ids in label_ids for tok_id in [ids[0]]}
            total = sum(np.exp(lp) for lp, _, _ in top)
            label_set_mass = float(sum(np.exp(lp) for lp, tok_id, _ in top if tok_id in in_set) / total) if total else None

        # Request 2: one item per label, on the now-cached prefix.
        scored = self._post_generate({
            "input_ids": [prefix_ids + ids for ids in label_ids],
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            # start one token early: the entry at logprob_start_len carries no logprob of its
            # own (nothing conditions it inside the returned window), so it is dropped below.
            "logprob_start_len": len(prefix_ids) - 1,
        }, samples=len(labels), tokens_in=sum(len(ids) for ids in label_ids))

        logprobs = np.zeros(len(labels))
        for i, (ids, result) in enumerate(zip(label_ids, scored)):
            entries = result["meta_info"]["input_token_logprobs"][1:]
            got = [tok_id for _, tok_id, _ in entries]
            if got != ids:
                raise RuntimeError(
                    f"logprob window misaligned for {labels[i]!r}: server returned tokens {got}, "
                    f"expected {ids}"
                )
            logprobs[i] = sum(lp for lp, _, _ in entries)

        return LabelScores(
            labels=list(labels),
            logprobs=logprobs,
            probs=_softmax(logprobs),
            first_token_probs=_softmax(np.array([first_logprob[ids[0]] for ids in label_ids])),
            label_set_mass=label_set_mass,
        )
