"""Real persuadee profiles from the Persuasion for Good release.

GDP-Zero's ``data/p4g/300_dialog_turn_based.pkl`` keeps only ``dialog`` and ``label`` per
dialogue, so the p4g user simulator runs with no per-dialogue grounding at all (``read_p4g``
returns an empty ``scenario`` tuple, unlike esc/cb which ground on the case). The upstream
dataset does ship the missing half: every participant filled in a pre-task survey, giving 23
psychological attributes plus demographics.

    Wang et al., ACL 2019 -- https://aclanthology.org/P19-1566/
    https://gitlab.com/ucdavisnlp/persuasionforgood  (data/FullData/full_info.csv)

``data/p4g_personas/full_info.csv`` has one row per participant, two rows per dialogue:

    B2   dialogue id      -- same string the pickle keys on, e.g. 20180904-045349_715_live
    B3   user id
    B4   role             -- "0" persuader, "1" persuadee
    B6   actual donation
    B7   number of turns
    *.x  survey responses (below)

All 300 dialogues in the pickle have a persuadee row, so the join is total.

Unlike a hand-authored persona list this is an *empirical* distribution: replaying dialogue
``did`` with the profile of the person who actually took part conditions the simulator on the
real user rather than on an invented archetype.

Reading a profile does not change anything on its own: the rendered text reaches a prompt
only by being passed to ``build_agents(..., persona=...)``, which the runners do per dialogue
under ``--p4g_persona``.
"""
import bisect
import csv
import logging
import os

logger = logging.getLogger(__name__)

# repo root is two levels up from this file (src/utils/p4g_personas.py)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_INFO_CSV = os.path.join(REPO_ROOT, "data", "p4g_personas", "full_info.csv")

PERSUADER, PERSUADEE = "0", "1"

# --- survey scales -------------------------------------------------------
# Big Five (Goldberg 1992), 1-5 Likert
BIG_FIVE = ["extrovert", "agreeable", "conscientious", "neurotic", "open"]
# Moral Foundations (Graham et al. 2011), 1-6
MORAL_FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity"]
# Schwartz Portrait Values (Cieciuch & Davidov 2012), 1-6
SCHWARTZ_VALUES = [
	"freedom", "conform", "tradition", "benevolence", "universalism", "self_direction",
	"stimulation", "hedonism", "achievement", "power", "security",
]
# Decision-making style (Hamilton et al. 2016), 1-5
DECISION_STYLE = ["rational", "intuitive"]

PSYCH_TRAITS = BIG_FIVE + MORAL_FOUNDATIONS + SCHWARTZ_VALUES + DECISION_STYLE  # the paper's 23
DEMOGRAPHICS = ["age", "sex", "race", "edu", "marital", "employment", "income", "religion", "ideology"]

_SCALE_MAX = {}
_SCALE_MAX.update({t: 5.0 for t in BIG_FIVE})
_SCALE_MAX.update({t: 6.0 for t in MORAL_FOUNDATIONS})
_SCALE_MAX.update({t: 6.0 for t in SCHWARTZ_VALUES})
_SCALE_MAX.update({t: 5.0 for t in DECISION_STYLE})

# how each Big Five trait reads at the low / high end, in second person
_BIG_FIVE_PHRASING = {
	"extrovert":     ("reserved and quiet with strangers", "outgoing and talkative"),
	"agreeable":     ("blunt, and slow to trust people who want something from you",
	                  "warm, trusting and eager to cooperate"),
	"conscientious": ("casual about plans and commitments", "organised and follow through on what you commit to"),
	"neurotic":      ("emotionally steady and hard to rattle", "anxious and easily worried"),
	"open":          ("practical and prefer the familiar", "curious and open to new ideas"),
}
# the survey's religion buckets, phrased for second person. "Other religion" is the largest
# bucket and says only "religious, not one of the three named" -- too vague to role-play, so
# it renders as nothing rather than as the literal label.
_RELIGION_PHRASING = {
	"Protestant": "a protestant",
	"Catholic": "a catholic",
	"Atheist": "an atheist",
}
_MORAL_PHRASING = {
	"care":     "the suffering of vulnerable people",
	"fairness": "whether people are treated justly",
	"loyalty":  "loyalty to your own group and community",
	"authority": "respect for authority and tradition",
	"purity":   "decency and moral purity",
}
# the Schwartz values, phrased as what the person actually weighs rather than as the scale's
# own label -- "benevolence" and "self-direction" are psychology jargon a 13B simulator will
# not act on, the paraphrase is something it can role-play.
_SCHWARTZ_PHRASING = {
	"freedom":        "your own freedom and independence",
	"conform":        "behaving properly and not upsetting people",
	"tradition":      "tradition and the customs you were raised with",
	"benevolence":    "looking after the people close to you",
	"universalism":   "the wellbeing of people everywhere and of the planet",
	"self_direction": "thinking for yourself and making your own choices",
	"stimulation":    "excitement, novelty and taking risks",
	"hedonism":       "enjoying yourself and having a good time",
	"achievement":    "being successful and having other people recognise it",
	"power":          "status, influence and being in control",
	"security":       "safety, stability and avoiding risk",
}
# 11 values is enough to bury the Big Five clauses if every non-mid one is emitted, so only
# the most distinctive few are kept per direction.
_MAX_VALUE_CLAUSES = 3


def _to_float(raw):
	"""Survey cells are blank for the handful of participants who skipped the survey."""
	if raw is None:
		return None
	raw = raw.strip()
	if raw in ("", "NA", "NaN", "nan", "None"):
		return None
	try:
		return float(raw)
	except ValueError:
		return None


def _clean_str(raw):
	if raw is None:
		return None
	raw = raw.strip()
	return raw or None


def load_profiles(path: str = None, role: str = PERSUADEE) -> dict:
	"""``{dialogue_id: profile}`` for one role (persuadee by default).

	A profile is a plain dict: the traits in ``PSYCH_TRAITS`` as floats (``None`` when the
	participant skipped the survey), the fields in ``DEMOGRAPHICS`` as floats/strings, plus
	``dialogue_id``, ``user_id``, ``role``, ``donation`` and ``num_turns``.
	"""
	path = path or DEFAULT_INFO_CSV
	if not os.path.exists(path):
		raise FileNotFoundError(
			f"p4g participant survey not found at {path}. Download it with:\n"
			f"  mkdir -p data/p4g_personas && curl -fsSL -o data/p4g_personas/full_info.csv \\\n"
			f"    https://gitlab.com/ucdavisnlp/persuasionforgood/-/raw/master/data/FullData/full_info.csv"
		)
	profiles = {}
	with open(path, "r", encoding="utf-8") as f:
		for row in csv.DictReader(f):
			if row.get("B4") != role:
				continue
			profile = {
				"dialogue_id": row["B2"],
				"user_id": _clean_str(row.get("B3")),
				"role": "persuadee" if role == PERSUADEE else "persuader",
				"donation": _to_float(row.get("B6")),
				"num_turns": _to_float(row.get("B7")),
			}
			for trait in PSYCH_TRAITS:
				profile[trait] = _to_float(row.get(f"{trait}.x"))
			for field in DEMOGRAPHICS:
				raw = row.get(f"{field}.x")
				profile[field] = _to_float(raw) if field in ("age", "income") else _clean_str(raw)
			profiles[profile["dialogue_id"]] = profile
	role_name = "persuadee" if role == PERSUADEE else "persuader"
	logger.info("loaded %d %s profiles from %s", len(profiles), role_name, path)
	return profiles


_CORPUS_CACHE = {}


def corpus_bands(path: str = None, role: str = PERSUADEE) -> dict:
	"""``{trait: sorted responses}`` over every participant -- the reference distribution the
	tertile split is taken against.

	Splitting the *scale* into thirds instead mislabels almost everyone, because Likert
	responses to these inventories are heavily skewed: on the 1017 persuadees, scale-thirds
	calls 830 of them "high" on ``rational`` and 754 "high" on ``fairness``. A clause four
	persuadees in five also get is not a distinctive trait, it is a constant, and it costs
	prompt tokens to say. Against the corpus, "high" means high *for a p4g participant*.
	"""
	path = path or DEFAULT_INFO_CSV
	key = (os.path.abspath(path), role)
	if key not in _CORPUS_CACHE:
		values = {trait: [] for trait in PSYCH_TRAITS}
		for profile in load_profiles(path, role=role).values():
			for trait in PSYCH_TRAITS:
				if profile.get(trait) is not None:
					values[trait].append(profile[trait])
		_CORPUS_CACHE[key] = {trait: sorted(v) for trait, v in values.items() if v}
	return _CORPUS_CACHE[key]


def _band(value, trait, corpus: dict = None) -> str:
	"""Split a Likert response into low / mid / high thirds.

	Against ``corpus`` when one is given (see ``corpus_bands``), else against the trait's own
	scale -- the fallback for a profile that did not come from the full survey.
	"""
	if value is None:
		return None
	if corpus and trait in corpus:
		responses = corpus[trait]
		low, high = responses[len(responses) // 3], responses[2 * len(responses) // 3]
		if low >= high:  # a trait the whole corpus answered alike separates nobody
			return "mid"
	else:
		top = _SCALE_MAX[trait]
		low, high = 1.0 + (top - 1.0) / 3.0, 1.0 + 2.0 * (top - 1.0) / 3.0
	if value <= low:
		return "low"
	if value >= high:
		return "high"
	return "mid"


def _distinctiveness(value, trait, corpus: dict = None) -> float:
	"""How far this response sits from the middle -- 0 at the median, 0.5 at either extreme.

	Only used to rank the Schwartz values against each other when there are more non-mid ones
	than ``_MAX_VALUE_CLAUSES`` has room for.
	"""
	if corpus and trait in corpus:
		responses = corpus[trait]
		return abs(bisect.bisect_left(responses, value) / len(responses) - 0.5)
	span = _SCALE_MAX[trait] - 1.0
	return abs(value - (1.0 + _SCALE_MAX[trait]) / 2.0) / span


def _and_list(items) -> str:
	"""['a', 'b', 'c'] -> 'a, b and c'.

	Most of the phrasings above are themselves multi-word and contain their own commas and
	"and"s ("safety, stability and avoiding risk"), and comma-joining those produces a run-on
	the simulator has to disentangle before it can role-play any of it. Whenever an item is
	already compound the list separates on semicolons instead, which keeps the boundaries
	visible.
	"""
	if len(items) == 1:
		return items[0]
	compound = any("," in item or " and " in item for item in items)
	sep, final = ("; ", "; and ") if compound else (", ", " and ")
	return sep.join(items[:-1]) + final + items[-1]


def describe_profile(profile: dict, include_demographics: bool = True,
					 include_moral: bool = True, include_values: bool = True,
					 corpus="auto") -> str:
	"""Render a profile as a second-person persona fragment for the user simulator.

	Only the traits that sit at an end of the *corpus* distribution are mentioned -- a mid-band
	response says little, and listing all 23 attributes buries the informative ones. The
	numbers themselves never reach the prompt: a 13B simulator does not act on "agreeable:
	4.4", it acts on "you are warm, trusting and eager to cooperate".

	``corpus`` is the reference distribution the low/mid/high split is taken against:
	``"auto"`` loads the full survey (the default, and what you want), a dict is one you
	already built with ``corpus_bands``, and ``None`` falls back to splitting each trait's own
	scale into thirds. Returns "" when the participant skipped the survey and there is nothing
	to condition on.
	"""
	if corpus == "auto":
		corpus = corpus_bands()

	def band(trait):
		return _band(profile.get(trait), trait, corpus)

	clauses = []

	if include_demographics:
		bits = []
		age = profile.get("age")
		if age:
			bits.append(f"{int(age)} years old")
		edu = profile.get("edu")
		if edu:
			bits.append({
				"Less than four-year college": "did not finish a four-year degree",
				"Four-year college": "have a four-year degree",
				"Postgraduate": "have a postgraduate degree",
			}.get(edu, edu.lower()))
		employment = profile.get("employment")
		if employment and employment != "Other":  # "Other" carries nothing to role-play
			bits.append(f"currently {employment.lower()}")
		if bits:
			clauses.append(f"You are {_and_list(bits)}.")
		ideology = profile.get("ideology")
		religion = _RELIGION_PHRASING.get(profile.get("religion"))
		if ideology and religion:
			clauses.append(f"Politically you are {ideology.lower()} and you are {religion}.")
		elif ideology:
			clauses.append(f"Politically you are {ideology.lower()}.")
		elif religion:
			clauses.append(f"You are {religion}.")

	traits = []
	for trait in BIG_FIVE:
		if band(trait) == "low":
			traits.append(_BIG_FIVE_PHRASING[trait][0])
		elif band(trait) == "high":
			traits.append(_BIG_FIVE_PHRASING[trait][1])
	if traits:
		clauses.append(f"You are {_and_list(traits)}.")

	if include_moral:
		cares = [_MORAL_PHRASING[t] for t in MORAL_FOUNDATIONS if band(t) == "high"]
		if cares:
			clauses.append(f"You care a great deal about {_and_list(cares)}.")

	if include_values:
		# 11 of the 23 attributes live here, so a participant can land outside the mid band on
		# most of them at once. Keep the ones furthest from the corpus median and drop the
		# rest, rather than spending a dozen clauses on one inventory.
		ranked = {"high": [], "low": []}
		for trait in SCHWARTZ_VALUES:
			side = band(trait)
			if side in ranked:
				ranked[side].append(trait)
		for side, lead in (("high", "You place a lot of weight on"),
						   ("low", "You put little weight on")):
			traits = sorted(ranked[side], key=lambda t: (
				-_distinctiveness(profile[t], t, corpus), SCHWARTZ_VALUES.index(t)))
			picked = [_SCHWARTZ_PHRASING[t] for t in traits[:_MAX_VALUE_CLAUSES]]
			if picked:
				clauses.append(f"{lead} {_and_list(picked)}.")

	rational, intuitive = band("rational"), band("intuitive")
	if rational == "high" and intuitive != "high":
		clauses.append("You make decisions by reasoning them through rather than going with your gut.")
	elif intuitive == "high" and rational != "high":
		clauses.append("You go with your gut rather than reasoning decisions through.")

	return " ".join(clauses)


def load_persona_texts(path: str = None, **describe_kwargs) -> dict:
	"""``{dialogue_id: persona text}`` -- ``load_profiles`` then ``describe_profile``.

	Dialogues whose participant skipped the survey render as "" and are dropped, so a caller
	can treat a missing key as "no persona for this dialogue".
	"""
	describe_kwargs.setdefault("corpus", corpus_bands(path))
	texts = {}
	for did, profile in load_profiles(path).items():
		text = describe_profile(profile, **describe_kwargs)
		if text:
			texts[did] = text
	return texts


def persona_suffix(persona: str, prefix: str = " ") -> str:
	"""``prefix + persona``, or "" when there is none -- so a prompt can append it
	unconditionally and stay byte-identical for an unconditioned simulator."""
	return f"{prefix}{persona}" if persona else ""


if __name__ == "__main__":  # quick look: python src/utils/p4g_personas.py
	import sys
	sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
	profiles = load_profiles()
	print(f"{len(profiles)} persuadee profiles from {DEFAULT_INFO_CSV}\n")
	for did in list(profiles)[:5]:
		print(f"--- {did} (donated ${profiles[did]['donation']}) ---")
		print(describe_profile(profiles[did]))
		print()
