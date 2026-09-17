"""A deterministic, offline stand-in for the chat backbone, for the pre/post-freeze
end-to-end regression.

Why not a real LLM. The acceptance test in Part 3.2 asks whether the CODE changed the
run. A sampled LLM answers a different question: two runs of the *unmodified* code
already differ, so "identical SR / AvgT / action sequences" is not even checkable
against it -- any difference would be indistinguishable from sampling noise, and any
match would be luck. This model returns the same text for the same prompt, forever,
so a difference in the output can only come from the code.

It is a real ``GenerationModel``: the games, players, planner and emotion classifier
call it through their normal interfaces and parse its output with their normal
parsers. Only the token source is fake.

Responses are drawn from ONE process-global RNG seeded at construction, not from a
per-prompt hash. That choice matters:

  * a prompt-hashed stub returns the same utterance for the same prompt forever, so every
    realization of a node collapses to one string, ``max_realizations`` stops meaning
    anything, and -- because p4g self-play starts every dialogue from the same empty
    scenario -- all ten dialogues become byte-identical. SR pins to a single value and the
    regression can no longer discriminate;
  * a seeded global stream reproduces exactly under the same seed (the call sequence is
    deterministic), while still giving each node genuinely different realizations and each
    dialogue a different trajectory -- which is what the emotion channel needs in order to
    have any within-node variance at all.

The property the regression relies on is therefore: same seed + same call sequence =>
identical output. If a code change alters how many draws are taken, the streams
desynchronise and the runs diverge -- which is precisely the failure the test is for.
"""
import random

from games import PersuasionGame
from utils.gen_models import OpenAIChatModel


SYS_ACTS = [
	PersuasionGame.S_PropositionOfDonation, PersuasionGame.S_CredibilityAppeal,
	PersuasionGame.S_EmotionAppeal, PersuasionGame.S_LogicalAppeal,
	PersuasionGame.S_TaskRelatedInquiry, PersuasionGame.S_PersonalRelatedInquiry,
	PersuasionGame.S_Greeting, PersuasionGame.S_Other,
]
USR_ACTS = [
	PersuasionGame.U_NoDonation, PersuasionGame.U_NegativeReaction,
	PersuasionGame.U_Neutral, PersuasionGame.U_PositiveReaction,
	PersuasionGame.U_Donate,
]
# "donate" ends the episode, so a uniform draw would make every rollout succeed on turn
# one or two and collapse SR to 1.0 -- a metric that cannot discriminate anything. These
# weights keep agreement rare enough that SR and AvgT actually vary across dialogues.
USR_ACT_WEIGHTS = [3, 3, 6, 3, 1]
EMOTIONS = ["Happiness", "Sadness", "Fear", "Anger", "Surprise", "Disgust", "Contempt", "Neutral"]

SYS_LINES = [
	"Save the Children works in the hardest places to reach.",
	"Even a small amount goes a long way for a child in crisis.",
	"They have been doing this since 1919 and publish their accounts.",
	"Can I tell you a little about what the money actually buys?",
	"What matters most to you when you pick a charity?",
	"No pressure at all, I just wanted to share it with you.",
]
USR_LINES = [
	"That does sound like a good cause.",
	"I am not sure I can afford anything right now.",
	"Tell me more about where the money goes.",
	"I have heard of them before, I think.",
	"Honestly I get asked for donations a lot.",
	"Alright, I could put a couple of dollars in.",
]


class DeterministicChatModel(OpenAIChatModel):
	"""Same interface as ``OpenAIChatModel``; never touches the network."""

	def __init__(self, gen_sentences=-1, seed=0):
		# mirror OpenAIChatModel.__init__ without the API-key / client setup
		self.inference_args = {"model": "stub-deterministic", "max_tokens": 64,
							   "temperature": 0.7, "n": 1}
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		self.call_count = 0
		self.rng = random.Random(seed)

	@staticmethod
	def _last_speaker_line(messages) -> str:
		for msg in reversed(messages):
			content = (msg.get("content") or "").strip()
			if content.startswith(f"{PersuasionGame.SYS}:") or content.startswith(f"{PersuasionGame.USR}:"):
				return content
		return ""

	def _kind(self, messages) -> str:
		"""Which of the four callers is asking, read off the prompt they built.

		persuadee-side prompts end on a ``Persuader:`` line (user simulator, and the
		value estimator, which is the same role-play closed with the donation ask);
		persuader-side prompts end on a ``Persuadee:`` line. Within the persuader side,
		the utterance model appends the chosen dialog act's instruction on its own line
		while the policy prior does not -- the prior is the one that wants a ``[act]``
		label back.
		"""
		line = self._last_speaker_line(messages)
		if line.startswith(f"{PersuasionGame.SYS}:"):
			return "usr_turn"          # expects "Persuadee: [act] text"
		if "\n" in line:
			return "sys_utterance"     # expects plain persuader text
		return "sys_prior"             # expects "Persuader: [act] text"

	def _one(self, kind: str, rng: random.Random) -> str:
		if kind == "usr_turn":
			da = rng.choices(USR_ACTS, weights=USR_ACT_WEIGHTS)[0]
			return f"{PersuasionGame.USR}: [{da}] {rng.choice(USR_LINES)}"
		if kind == "sys_utterance":
			return f"{PersuasionGame.SYS}: {rng.choice(SYS_LINES)}"
		return f"{PersuasionGame.SYS}: [{rng.choice(SYS_ACTS)}] {rng.choice(SYS_LINES)}"

	# -- GenerationModel interface ----------------------------------------
	def chat_generate(self, messages, **gen_args):
		self.call_count += 1
		n = int(gen_args.get("num_return_sequences", gen_args.get("n", 1)) or 1)
		kind = self._kind(messages)
		return [{"generated_text": self._one(kind, self.rng)} for _ in range(n)]

	def chat_generate_batched(self, messages_list, **gen_args):
		return [self.chat_generate(m, **gen_args) for m in messages_list]

	def generate(self, input_text, **gen_args):
		"""The emotion classifier's entry point: a plain prompt, ``[Emotion]`` back."""
		self.call_count += 1
		n = int(gen_args.get("num_return_sequences", gen_args.get("n", 1)) or 1)
		rng = self.rng
		# a skewed draw, so the resulting softmax is neither one-hot nor uniform and the
		# valence channel sees genuinely different z across an edge's simulations
		weights = [rng.random() ** 2 for _ in EMOTIONS]
		return [{"generated_text": f"[{rng.choices(EMOTIONS, weights=weights)[0]}]"} for _ in range(n)]

	def supports_label_scoring(self) -> bool:
		return False


def install(runner_module, gen_sentences=-1, seed=0):
	"""Point ``runner_module.make_backbone_model`` at the stub. Returns the model so the
	caller can read ``call_count``."""
	model = DeterministicChatModel(gen_sentences=gen_sentences, seed=seed)
	runner_module.make_backbone_model = lambda **kwargs: (model, "chat")
	return model
