from evaluators.resp_ranker import LLMRespRanker


class CBEvaluator(LLMRespRanker):
    """Pairwise ranker for CraigslistBargain: which response better helps the Buyer reach a
    favourable deal (a lower price) while still closing the sale?

    The buyer/seller item description and target prices are not part of the dialog history, so
    include them in ``context`` if you want the ranker to take them into account."""

    system_role = "Buyer"

    def build_prompt(self, context, resp_a, resp_b) -> str:
        return f"""
		The following is a conversation between a Buyer and a Seller negotiating the price of an item on an online marketplace. The Buyer is trying to reach a deal at the lowest price possible while still closing the sale.
		{context}
		Which of the following responses can better help the Buyer reach a favourable deal (a lower price) while still closing the sale?
		A. Buyer: {resp_a}
		B. Buyer: {resp_b}
		C. Can't tell.
		You can choose from either A, B, or C.
		Your choice:
		""".replace('\t', '').strip()
