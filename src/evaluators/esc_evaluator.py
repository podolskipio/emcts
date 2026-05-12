from evaluators.resp_ranker import LLMRespRanker


class ESCEvaluator(LLMRespRanker):
    """Pairwise ranker for ESConv: which response better supports the Patient and helps them
    work through (and ultimately resolve) their emotional problem?"""

    system_role = "Therapist"

    def build_prompt(self, context, resp_a, resp_b) -> str:
        return f"""
		The following is a conversation between a Therapist and a Patient. The Therapist is providing emotional support to help the Patient feel better and ultimately resolve their emotional problem.
		{context}
		Which of the following responses can better help the Therapist support the Patient and reduce their emotional distress?
		A. Therapist: {resp_a}
		B. Therapist: {resp_b}
		C. Can't tell.
		You can choose from either A, B, or C.
		Your choice:
		""".replace('\t', '').strip()
