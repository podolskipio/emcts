from evaluators.resp_ranker import RespRanker, LLMRespRanker
from evaluators.p4g_evaluator import P4GEvaluator
from evaluators.esc_evaluator import ESCEvaluator
from evaluators.cb_evaluator import CBEvaluator

EVALUATORS = {
	"p4g": P4GEvaluator,
	"esc": ESCEvaluator,
	"cb": CBEvaluator,
}


def get_evaluator(task, gen_model, **kwargs):
	"""Return the :class:`LLMRespRanker` for ``task`` ("p4g" | "esc" | "cb")."""
	if task not in EVALUATORS:
		raise ValueError(f"unknown evaluator task {task!r}; choose from {list(EVALUATORS)}")
	return EVALUATORS[task](gen_model, **kwargs)
