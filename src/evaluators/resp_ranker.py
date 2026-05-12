"""Pairwise response rankers used to evaluate policy-planning outputs.

Given a dialog ``context`` and two candidate system responses ``resp_a`` / ``resp_b``, a ranker
asks an LLM which one better serves the task goal, A/B-swapping to debias position and majority-
voting over several samples. ``evaluate`` returns ``(preference, debug)`` where ``preference`` is
``0`` (A is better), ``1`` (B is better) or ``2`` ("can't tell" / tie).

All three tasks share the same machinery; only the framing of the question differs, so each
concrete evaluator just implements :meth:`LLMRespRanker.build_prompt`.
"""
import logging
import random

from abc import ABC, abstractmethod
from typing import List

from utils.gen_models import GenerationModel

logger = logging.getLogger(__name__)


class RespRanker(ABC):
    @abstractmethod
    def evaluate(self, context, resp_a, resp_b):
        """Compare two responses and return ``(preference, debug)``."""
        raise NotImplementedError


class LLMRespRanker(RespRanker):
    """Shared LLM-based pairwise ranker. Subclasses set :attr:`system_role` and implement
    :meth:`build_prompt`."""

    system_role = "Speaker"  # the role label used for the A./B. options, e.g. "Persuader"

    def __init__(self, gen_model: GenerationModel, inference_args: dict = None):
        super().__init__()
        self.gen_model = gen_model
        self.inference_args = inference_args or {
            "max_tokens": 2,
            "temperature": 0.7,
            "echo": False,
            "n": 5,
            "stop": "",
        }

    @abstractmethod
    def build_prompt(self, context, resp_a, resp_b) -> str:
        """Return the ranking prompt comparing ``resp_a`` (option A) and ``resp_b`` (option B)."""
        raise NotImplementedError

    def evaluate(self, context, resp_a, resp_b):
        do_swap = random.random() < 0.5
        if do_swap:
            resp_a, resp_b = resp_b, resp_a
        prompt = self.build_prompt(context, resp_a, resp_b)
        logger.debug(f"prompt: {prompt}")
        resps = self.gen_model.generate(prompt, **self.inference_args)
        choices, rationales = self._process_resps(resps)
        preference = self._majority_vote(choices, do_swap)
        return preference, {"choices": choices, "rationales": rationales, "do_swap": do_swap}

    def _process_resps(self, resps: List[dict]):
        choices, rationales = [], []
        for resp in resps:
            gen = resp["generated_text"].strip()
            if len(gen) == 0:
                logger.warning("empty response from the ranking model")
                choice = "c"
            else:
                choice = gen[0].lower()
            if choice not in ("a", "b", "c"):
                logger.warning(f"invalid ranking choice: {choice!r}")
                choice = "c"
            choices.append(choice)
            rationales.append(gen)  # just keep the raw response
        return choices, rationales

    def _majority_vote(self, choices: List[str], do_swap=False):
        a_cnt = sum(1 for c in choices if c == "a")
        b_cnt = sum(1 for c in choices if c == "b")
        if a_cnt > b_cnt:
            return 1 if do_swap else 0
        if b_cnt > a_cnt:
            return 0 if do_swap else 1
        return 2
