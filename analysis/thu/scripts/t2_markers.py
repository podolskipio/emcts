"""TASK 2 -- the resistance/receptivity lexical scorer.

Transparent regex, no model. Every pattern is listed here verbatim and reproduced in the report
appendix. Patterns were inspected ONLY on the mine200 split (the 197 dialogues that also re-mine
w(e)); the test100 fold and the eval 100 were never looked at.

    r(utterance) = (receptivity hits - resistance hits) / (1 + tokens/10)
"""
import re

# ---------------------------------------------------------------------------------------------
# RESISTANCE
# ---------------------------------------------------------------------------------------------
RESISTANCE = {
	# counterargument connectives -- word-initial or clause-initial only, so "but" inside
	# "nothing but good" does not fire
	"counterargument": [
		r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b", r"\bnevertheless\b",
		r"\bon the other hand\b", r"\bthat said\b", r"\beven so\b",
	],
	# deflection / postponement
	"deflection": [
		r"\bmaybe later\b", r"\bi'?ll think about it\b", r"\bnot (?:right )?now\b",
		r"\bsome other time\b", r"\banother time\b", r"\bi'?ll (?:have to )?(?:consider|look into)\b",
		r"\bnot (?:really )?(?:interested|sure i)\b", r"\bi (?:can'?t|cannot) (?:afford|right now)\b",
		r"\bi'?m (?:a bit )?(?:tight|broke)\b", r"\bdon'?t have (?:the )?(?:money|funds|cash)\b",
		r"\bi'?ll pass\b", r"\bno thank(?:s| you)\b",
	],
	# source doubt / scepticism about the charity or the persuader
	"source_doubt": [
		r"\bhow do i know\b", r"\bare you sure\b", r"\bscam\b", r"\bfraud\b", r"\blegit(?:imate)?\b",
		r"\btrust(?:worthy)?\b", r"\bskeptic(?:al)?\b", r"\bsceptic(?:al)?\b",
		r"\bhow much (?:of it |actually )?goes\b", r"\bwhere does the money go\b",
		r"\boverhead\b", r"\badministrat(?:ive|ion) costs?\b", r"\bproof\b", r"\bverify\b",
	],
	# hedges
	"hedge": [
		r"\bi guess\b", r"\bsort of\b", r"\bkind of\b", r"\bkinda\b", r"\bi suppose\b",
		r"\bprobably\b", r"\bperhaps\b", r"\bmight\b", r"\bnot sure\b", r"\bi dunno\b",
		r"\bi don'?t know\b",
	],
}

# ---------------------------------------------------------------------------------------------
# RECEPTIVITY
# ---------------------------------------------------------------------------------------------
RECEPTIVITY = {
	# questions directed at the cause (a "?" alone is too coarse -- it also catches deflecting
	# challenges, which resistance already scores)
	"cause_question": [
		r"\b(?:how|where|what|who|which)\b[^?]{0,80}\b(?:charity|children|donate|donation|fund|"
		r"money|organi[sz]ation|save the children|help|support|cause)\b[^?]{0,40}\?",
		r"\bhow (?:can|do) i (?:help|donate|give|contribute)\b",
		r"\btell me more\b", r"\bi'?d like to (?:know|hear) more\b",
	],
	# self-disclosure
	"self_disclosure": [
		r"\bi once\b", r"\bmy (?:family|mother|father|mom|dad|son|daughter|kids?|children|wife|"
		r"husband|brother|sister|friend)\b", r"\bi (?:have|had) (?:a |an )?(?:child|kids?|son|daughter)\b",
		r"\bi work (?:with|for|at)\b", r"\bi volunteer\b", r"\bwhen i was\b", r"\bin my (?:own )?experience\b",
		r"\bi (?:also )?(?:donate|give|support)(?:d|s)? (?:to|regularly|monthly|every)\b",
		r"\bgrew up\b", r"\bi'?ve been\b",
	],
	# commitment language
	"commitment": [
		r"\bi will\b", r"\bi'?ll donate\b", r"\bi'?d like to\b", r"\bsign me up\b", r"\bcount me in\b",
		r"\bi'?m in\b", r"\bi want to (?:help|donate|give|contribute)\b",
		r"\bi can (?:donate|give|do)\b", r"\bput me down\b", r"\bhappy to (?:help|donate|give)\b",
		r"\bi'?ll give\b", r"\bsounds good\b", r"\bdefinitely\b", r"\babsolutely\b",
		r"\ball of it\b", r"\bthe (?:full|whole) (?:amount|two dollars|\$2)\b",
	],
}

_RES = {k: [re.compile(p, re.I) for p in v] for k, v in RESISTANCE.items()}
_REC = {k: [re.compile(p, re.I) for p in v] for k, v in RECEPTIVITY.items()}


def hits(text):
	"""Per-family hit counts (number of matches, not number of patterns that matched)."""
	out = {}
	for fam, pats in _RES.items():
		out[f"res_{fam}"] = sum(len(p.findall(text)) for p in pats)
	for fam, pats in _REC.items():
		out[f"rec_{fam}"] = sum(len(p.findall(text)) for p in pats)
	out["res_total"] = sum(v for k, v in out.items() if k.startswith("res_"))
	out["rec_total"] = sum(v for k, v in out.items() if k.startswith("rec_"))
	out["tokens"] = len(text.split())
	return out


def r_score(text):
	h = hits(text)
	return (h["rec_total"] - h["res_total"]) / (1.0 + h["tokens"] / 10.0)


def marker_lists():
	"""The lists, verbatim, for the appendix."""
	return {"resistance": RESISTANCE, "receptivity": RECEPTIVITY}
