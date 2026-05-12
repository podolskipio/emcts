from evaluators.resp_ranker import RespRanker, LLMRespRanker  # noqa: F401  (RespRanker re-exported for back-compat)


class P4GEvaluator(LLMRespRanker):
    """Pairwise ranker for PersuasionForGood: which response better persuades the Persuadee
    to donate to Save the Children?"""

    system_role = "Persuader"

    def build_prompt(self, context, resp_a, resp_b) -> str:
        return f"""
		The following is background information about Save the Children.
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The following is a conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{context}
		Which of the following responses can better help the Persuader convince the Persuadee to donate to Save the Children?
		A. Persuader: {resp_a}
		B. Persuader: {resp_b}
		C. Can't tell.
		You can choose from either A, B, or C.
		Your choice:
		""".replace('\t', '').strip()
