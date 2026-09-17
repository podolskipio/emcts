"""When does a persuadee turn tagged [donate] count as a donation?

The simulator tags its own turn; GDP-Zero ends the episode on the tag alone. Measured on the episode
pilots (analysis/thu/success_criteria_pilots.tsv), the tag fires on deferrals ("I will consider donating
and will be in touch") and names amounts the P4G task could never pay. `--p4g_success` picks a stricter
reading. Shared by the environment (games.p4g_game.PersuasionGame.get_dialog_ended, which also ends
search) and the offline scorer (analysis/thu/scripts/t10_success_criteria.py), so the two cannot drift.

  tag        the [donate] tag alone                                   (DEFAULT, GDP-Zero)
  committed  tag AND no deferral/hedge language in the tagged turn
  amount     committed AND an explicit amount in the tagged turn or the persuadee turn before it
"""
import re

SUCCESS_CRITERIA = ("tag", "committed", "amount")

# v2 (2026-09-17): the hedge must GOVERN the donation, not merely co-occur with it. v1 was a bare
# substring match and flagged "After considering the information ..., I have decided to donate $5"; on
# the 134 logged successes it had 2 false positives and 1 miss (analysis/thu/success_criterion_fix.md).
_DONATION = r"(?:donat\w*|giv(?:e|ing)|contribut\w*)"
# a hedge word with a donation verb at most 3 words after it: "consider making a donation", "possibly donate"
HEDGE_GOVERNS = re.compile(r"\b(consider(?:ing)?|think(?:ing)? about|maybe|perhaps|possibly|might)\b"
						   r"(?:\s+[\w']+){0,3}?\s+" + _DONATION, re.I)
# deferral with no donation verb needed: "look into it and see how I can", "check my finances"
DEFERRAL = re.compile(r"\b(look into it|see how I can|check my finances|see what I can afford|be in touch|"
					  r"get back to you|think about it|another time|not (?:right )?now)\b", re.I)
# a firm first-person commitment. Only these words may sit between the modal and the verb, so
# "I will consider donating" is NOT firm while "I will definitely donate" is.
FIRM = re.compile(r"\bI(?:'ll| will| would like to|'d like to| want to| have decided to|'ve decided to| am going to|"
				  r"'m going to| am ready to|'m ready to| am also ready to)\s+"
				  r"(?:(?:definitely|now|also|happily|gladly|go ahead and|really)\s+)*"
				  r"(?:" + _DONATION + r"|make a (?:small |one-time )?donation)", re.I)
HEDGE = HEDGE_GOVERNS  # kept for callers that report the matched hedge phrase


def hedge_match(text: str):
	"""The phrase that makes ``text`` a hedged non-commitment, or None.

	Hedged = (a hedge word governing a donation verb, OR a deferral phrase) AND no firm first-person
	commitment anywhere in the turn.
	"""
	text = text or ""
	if FIRM.search(text):
		return None
	m = HEDGE_GOVERNS.search(text) or DEFERRAL.search(text)
	return m.group(0) if m else None


_NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "twenty": 20, "fifty": 50,
			"hundred": 100, "a": 1, "half a": 0.5}
AMOUNT = re.compile(r"\$\s?(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(dollars?|bucks|cents?)\b|"
					r"\b(one|two|three|four|five|ten|twenty|fifty|hundred|a|half a)\s+(dollars?|bucks|cents?)\b", re.I)


def amounts(text: str) -> list:
	"""Every money amount named in ``text``, in dollars."""
	out = []
	for m in AMOUNT.finditer(text or ""):
		if m.group(1):
			out.append(float(m.group(1)))
		elif m.group(2):
			v = float(m.group(2))
			out.append(v / 100 if m.group(3).lower().startswith("cent") else v)
		else:
			v = _NUMWORD[m.group(4).lower()]
			out.append(v / 100 if m.group(5).lower().startswith("cent") else v)
	return out


def donation_counts(criterion: str, donate_turn: str, lead_in: str = "") -> bool:
	"""Whether a [donate]-tagged turn counts as success under ``criterion``."""
	if criterion == "tag":
		return True
	if hedge_match(donate_turn):
		return False
	if criterion == "committed":
		return True
	if criterion == "amount":
		return bool(amounts(donate_turn) or amounts(lead_in))
	raise ValueError(f"success criterion must be one of {SUCCESS_CRITERIA}, got {criterion!r}")
