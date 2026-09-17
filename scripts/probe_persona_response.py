"""Does the user simulator actually *use* a persona? A cheap, model-agnostic probe.

Before spending a full planner run on ``--p4g_persona``, check the premise: given the same
Persuader utterance, does the persuadee simulator answer differently when it is told who it
is? A model that ignores the persona makes the whole persona experiment a no-op, and that
failure is silent -- the run completes, the numbers just do not move.

Design (one condition = one pass over the grid):

  20 profiles   -- a balanced 2x2 over the *extremes* of agreeableness x neuroticism, five
                   profiles per cell, taken from the real p4g pre-task survey.
  5 utterances  -- one fixed Persuader turn per act type (greeting, credibility appeal,
                   logical appeal, emotion appeal, proposition of donation). Fixed, so the
                   only thing that varies across profiles is the persona.
  n samples     -- repeats per cell, to separate persona effects from sampling noise.

  => 20 x 5 x n generations per condition, run with and without the persona.

Three measurements, in increasing order of how much they demand of the model:

  1. cross-persona divergence -- for one fixed utterance, how far apart are the 20 answers?
     The no-persona condition is the control: there all 20 prompts are byte-identical, so its
     divergence is pure sampling noise. persona >> no-persona means the text is being read.
  2. emotion entropy + negative mass -- does the affective range actually widen, or does the
     model stay in its polite-neutral attractor whatever it is told?
  3. directional sanity -- do high-neuroticism profiles yield more fear/sadness, and
     low-agreeableness ones more anger/disgust? Divergence only says the persona changed the
     output; this says it changed it in the right direction.

Generation goes through the real ``PersuadeeChatModel``, so the probe tests the production
prompt path rather than a bespoke one.

    # against whatever SGLang is serving
    python scripts/probe_persona_response.py --model_label vicuna-13b

    # a different model: restart SGLang on that model first, then
    python scripts/probe_persona_response.py --model_label llama-3.1-8b

Writes ``<out_dir>/<model_label>.json`` (every generation + all metrics).
``scripts/report_persona_probe.py`` turns one or more of those into the analysis document.
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from games import PersuasionGame
from players.p4g_players import PersuadeeChatModel
from utils.gen_models import SGLangChatModel
from utils.p4g_personas import load_profiles, describe_profile
from utils.prompt_examples import EXP_DIALOG
from utils.sessions import DialogSession

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# One fixed Persuader turn per act. Held constant across profiles and conditions so the only
# thing that varies is the persona; worded in the style of the p4g corpus.
FIXED_UTTERANCES = {
	"greeting": "Hello! How are you doing today?",
	"credibility appeal": (
		"Save the Children is one of the largest children's charities in the world, and "
		"they publish audited reports on exactly how donations are spent."
	),
	"logical appeal": (
		"Even a dollar or two goes a long way here -- that amount can cover a day of "
		"clean water and food for a child in a war zone."
	),
	"emotion appeal": (
		"Right now there are children going to sleep hungry and frightened, with no one "
		"coming to help them. It is hard to think about."
	),
	"proposition of donation": (
		"Would you be willing to donate a small part of your task payment to Save the "
		"Children today?"
	),
}

# Emotions enum values are lowercase. `contempt` is in the enum but HFEmotionClassifier has no
# mapping for it (its model emits 7 labels), so it is always exactly 0.0 -- carrying it would
# deflate every entropy by a constant. Metrics run over the 7 reachable labels; the raw
# distributions in the JSON keep the full 8-way axis.
UNREACHABLE = ["contempt"]
NEGATIVE = ["anger", "disgust", "fear", "sadness"]
ANXIOUS = ["fear", "sadness"]        # what high neuroticism should raise
HOSTILE = ["anger", "disgust"]       # what low agreeableness should raise


# ---------------------------------------------------------------------------
# profile selection: a balanced 2x2 over the extremes of agreeableness x neuroticism
# ---------------------------------------------------------------------------
def select_extreme_profiles(n_per_cell=5, restrict_to_annotated=True):
	"""Five profiles per corner of the (agreeableness, neuroticism) square.

	A strict decile *intersection* is far too unbalanced on this data (2 / 10 / 8 / 4 of the
	annotated 300), so instead each profile is placed in percentile space and the ones nearest
	each corner are taken. Balanced by construction, and still genuinely extreme -- the report
	prints the per-cell means so the reader can judge how extreme.
	"""
	profiles = load_profiles()
	if restrict_to_annotated:
		with open(os.path.join(REPO_ROOT, "data", "p4g", "300_dialog_turn_based.pkl"), "rb") as f:
			keep = set(pickle.load(f))
		profiles = {d: p for d, p in profiles.items() if d in keep}

	rows = [
		(d, p) for d, p in profiles.items()
		if p["agreeable"] is not None and p["neurotic"] is not None and describe_profile(p)
	]
	agree = np.array([p["agreeable"] for _, p in rows])
	neuro = np.array([p["neurotic"] for _, p in rows])

	def pct_rank(v):  # 0..1, ties averaged
		order = v.argsort()
		ranks = np.empty(len(v), float)
		ranks[order] = np.arange(len(v))
		for x in np.unique(v):
			m = v == x
			ranks[m] = ranks[m].mean()
		return ranks / (len(v) - 1)

	pa, pn = pct_rank(agree), pct_rank(neuro)

	selected, taken = [], set()
	for cell, (ta, tn) in {
		"lowA_lowN":   (0.0, 0.0),
		"lowA_highN":  (0.0, 1.0),
		"highA_lowN":  (1.0, 0.0),
		"highA_highN": (1.0, 1.0),
	}.items():
		dist = np.hypot(pa - ta, pn - tn)
		for i in dist.argsort():
			if len(([s for s in selected if s["cell"] == cell])) >= n_per_cell:
				break
			if i in taken:
				continue
			taken.add(i)
			did, prof = rows[i]
			selected.append({
				"dialogue_id": did,
				"cell": cell,
				"agreeable": prof["agreeable"],
				"neurotic": prof["neurotic"],
				"donation": prof["donation"],
				"persona": describe_profile(prof),
			})
	return selected


# ---------------------------------------------------------------------------
# embeddings, for measurement 1
# ---------------------------------------------------------------------------
class Embedder:
	"""Mean-pooled MiniLM if it can be loaded, else a pure-numpy character-ngram TF-IDF.

	sentence-transformers is not installed here, so the transformer path is hand-rolled on
	AutoModel. The fallback is worse at paraphrase but still separates "different content"
	from "same content", which is all measurement 1 needs.
	"""

	def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
		self.kind = "tfidf-char"
		self.model = None
		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
			self.torch = torch
			self.tok = AutoTokenizer.from_pretrained(model_name)
			self.model = AutoModel.from_pretrained(model_name).eval()
			self.kind = f"minilm:{model_name}"
		except Exception as e:  # offline, or no weights cached
			print(f"  [embedder] {model_name} unavailable ({type(e).__name__}); "
				  f"falling back to char-ngram TF-IDF")

	def __call__(self, texts):
		if self.model is not None:
			return self._minilm(texts)
		return self._tfidf(texts)

	def _minilm(self, texts):
		out = []
		with self.torch.no_grad():
			for i in range(0, len(texts), 32):
				batch = self.tok(texts[i:i + 32], padding=True, truncation=True,
								 max_length=256, return_tensors="pt")
				hidden = self.model(**batch).last_hidden_state
				mask = batch["attention_mask"].unsqueeze(-1).float()
				pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
				out.append(pooled.cpu().numpy())
		return np.vstack(out)

	def _tfidf(self, texts, n=4):
		grams = [{t[i:i + n] for i in range(max(1, len(t) - n + 1))} for t in map(str.lower, texts)]
		vocab = sorted(set().union(*grams)) if grams else []
		index = {g: i for i, g in enumerate(vocab)}
		X = np.zeros((len(texts), len(vocab)))
		for r, gs in enumerate(grams):
			for g in gs:
				X[r, index[g]] = 1.0
		df = X.sum(0)
		idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
		return X * idf


def mean_pairwise_cosine_distance(vectors):
	"""Average 1 - cos(u, v) over all distinct pairs. 0 = identical, 1 = orthogonal."""
	if len(vectors) < 2:
		return float("nan")
	V = np.asarray(vectors, dtype=float)
	V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
	sim = V @ V.T
	# off-diagonal mean
	iu = np.triu_indices(len(V), k=1)
	return float(1.0 - sim[iu].mean())


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def make_state(sys_utterance):
	"""A one-turn session ending on the Persuader, i.e. the persuadee's turn to speak."""
	state = DialogSession(PersuasionGame.SYS, PersuasionGame.USR)
	state.add_single(PersuasionGame.SYS, "greeting", sys_utterance)
	return state


def generate_condition(backbone, profiles, acts, n_samples, use_persona, user_dialog_acts,
					   example, inference_args):
	"""One pass over the 20 x 5 x n grid. Returns a flat list of records."""
	records = []
	for pi, prof in enumerate(profiles):
		# a fresh simulator per profile -- persona is a constructor argument
		user = PersuadeeChatModel(
			user_dialog_acts, inference_args, backbone_model=backbone,
			conv_examples=[example],
			persona=prof["persona"] if use_persona else None,
		)
		# batch this profile's whole row (acts x samples) into one server call
		states, meta = [], []
		for act in acts:
			for s in range(n_samples):
				states.append(make_state(FIXED_UTTERANCES[act]))
				meta.append((act, s))
		das, responses = user.get_utterance_w_da_from_batched_states(states)
		for (act, s), da, resp in zip(meta, das, responses):
			records.append({
				"dialogue_id": prof["dialogue_id"], "cell": prof["cell"],
				"agreeable": prof["agreeable"], "neurotic": prof["neurotic"],
				"act": act, "sample": s, "persona_used": bool(use_persona),
				"da": da, "response": resp,
			})
		print(f"    profile {pi + 1}/{len(profiles)} ({prof['cell']}) "
			  f"{'persona' if use_persona else 'no-persona'}: {len(responses)} generations")
	return records


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def entropy(dist, base=2.0):
	p = np.asarray([max(float(v), 0.0) for v in dist], dtype=float)
	p = p / (p.sum() or 1.0)
	nz = p[p > 0]
	return float(-(nz * np.log(nz)).sum() / math.log(base))


def bootstrap_diff(a, b, n=10000, seed=0):
	"""95% CI for mean(a) - mean(b) by resampling each group. numpy only (no scipy here)."""
	rng = np.random.default_rng(seed)
	a, b = np.asarray(a, float), np.asarray(b, float)
	diffs = np.array([
		rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
		for _ in range(n)
	])
	return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def measure(records, emotions_order, embedder):
	"""All three measurements, computed per condition."""
	out = {}
	for cond in (True, False):
		rs = [r for r in records if r["persona_used"] == cond]
		key = "persona" if cond else "no_persona"
		if not rs:
			continue

		# --- 1. cross-persona divergence, per act, per sample index -------------
		divs = []
		for act in sorted({r["act"] for r in rs}):
			for s in sorted({r["sample"] for r in rs}):
				texts = [r["response"] for r in rs if r["act"] == act and r["sample"] == s]
				texts = [t for t in texts if t and t.strip()]
				if len(texts) >= 2:
					divs.append(mean_pairwise_cosine_distance(embedder(texts)))
		# how often the model returned the exact same string for different personas
		dup = []
		for act in sorted({r["act"] for r in rs}):
			texts = [r["response"] for r in rs if r["act"] == act]
			dup.append(1.0 - len(set(texts)) / max(1, len(texts)))

		# --- 2. emotion entropy + negative mass ---------------------------------
		dists = np.array([[r["emotion_dist"][e] for e in emotions_order] for r in rs])
		aggregate = dists.mean(0)
		neg_idx = [emotions_order.index(e) for e in NEGATIVE]
		argmax_labels = [emotions_order[i] for i in dists.argmax(1)]

		out[key] = {
			"n": len(rs),
			"divergence_mean": float(np.mean(divs)) if divs else None,
			"divergence_std": float(np.std(divs)) if divs else None,
			"duplicate_response_rate": float(np.mean(dup)),
			"aggregate_emotion": {e: float(v) for e, v in zip(emotions_order, aggregate)},
			"aggregate_entropy": entropy(aggregate),
			"mean_per_response_entropy": float(np.mean([entropy(d) for d in dists])),
			"negative_mass": float(dists[:, neg_idx].sum(1).mean()),
			"negative_argmax_frac": float(np.mean([l in NEGATIVE for l in argmax_labels])),
			"distinct_argmax_emotions": len(set(argmax_labels)),
			"da_distribution": {
				da: sum(1 for r in rs if r["da"] == da) / len(rs)
				for da in sorted({r["da"] for r in rs})
			},
		}

	# --- 3. directional sanity, within each condition -------------------------
	directional = {}
	for cond in (True, False):
		rs = [r for r in records if r["persona_used"] == cond]
		if not rs:
			continue
		key = "persona" if cond else "no_persona"

		def mass(subset, labels):
			idx = [emotions_order.index(e) for e in labels]
			return [sum(r["emotion_dist"][emotions_order[i]] for i in idx) for r in subset]

		hi_n = [r for r in rs if r["cell"].endswith("highN")]
		lo_n = [r for r in rs if r["cell"].endswith("lowN")]
		lo_a = [r for r in rs if r["cell"].startswith("lowA")]
		hi_a = [r for r in rs if r["cell"].startswith("highA")]

		d1, l1, u1 = bootstrap_diff(mass(hi_n, ANXIOUS), mass(lo_n, ANXIOUS))
		d2, l2, u2 = bootstrap_diff(mass(lo_a, HOSTILE), mass(hi_a, HOSTILE))
		directional[key] = {
			"fear_sadness_highN_minus_lowN": {"diff": d1, "ci95": [l1, u1],
											  "highN_mean": float(np.mean(mass(hi_n, ANXIOUS))),
											  "lowN_mean": float(np.mean(mass(lo_n, ANXIOUS)))},
			"anger_disgust_lowA_minus_highA": {"diff": d2, "ci95": [l2, u2],
											   "lowA_mean": float(np.mean(mass(lo_a, HOSTILE))),
											   "highA_mean": float(np.mean(mass(hi_a, HOSTILE)))},
		}
	out["directional"] = directional
	return out


def main(args):
	backbone = SGLangChatModel(args.sglang_model, base_url=args.sglang_host)
	served = backbone.inference_args["model"]
	print(f"model under test: {served}  (label: {args.model_label})")

	profiles = select_extreme_profiles(n_per_cell=args.n_per_cell)
	print(f"selected {len(profiles)} extreme profiles "
		  f"({args.n_per_cell} per cell of agreeableness x neuroticism)")
	for cell in ("lowA_lowN", "lowA_highN", "highA_lowN", "highA_highN"):
		c = [p for p in profiles if p["cell"] == cell]
		print(f"  {cell:12} agreeable={np.mean([p['agreeable'] for p in c]):.2f} "
			  f"neurotic={np.mean([p['neurotic'] for p in c]):.2f}")

	acts = list(FIXED_UTTERANCES)
	example = DialogSession(PersuasionGame.SYS, PersuasionGame.USR).from_history(EXP_DIALOG)
	user_das = PersuasionGame.get_game_ontology()["user"]["dialog_acts"]
	# matches build_agents' usr_inference_args, so the probe measures the production sampler
	inference_args = {
		"max_new_tokens": 128, "temperature": 1.1, "repetition_penalty": 1.0,
		"do_sample": True, "return_full_text": False, "top_p": 0.9, "top_k": 40,
	}

	records = []
	for use_persona in (True, False):
		print(f"\n=== condition: {'persona' if use_persona else 'no persona'} ===")
		records += generate_condition(backbone, profiles, acts, args.n_samples, use_persona,
									  user_das, example, inference_args)

	print(f"\nclassifying emotions for {len(records)} generations")
	from emotion_classifiers.hf_emotion import HFEmotionClassifier
	clf = HFEmotionClassifier()
	emotions_order = [str(e) for e in clf.emotions]
	for r in records:
		dist = clf.predict_distribution_from_utterance(r["response"] or "")
		r["emotion_dist"] = {str(k): float(v) for k, v in dist.items()}
		r["emotion"] = max(r["emotion_dist"], key=r["emotion_dist"].get)

	print("computing measurements")
	embedder = Embedder()
	metric_emotions = [e for e in emotions_order if e not in UNREACHABLE]
	metrics = measure(records, metric_emotions, embedder)

	os.makedirs(args.out_dir, exist_ok=True)
	path = os.path.join(args.out_dir, f"{args.model_label}.json")
	with open(path, "w", encoding="utf-8") as f:
		json.dump({
			"model_label": args.model_label,
			"served_model": served,
			"embedder": embedder.kind,
			"emotions_order": emotions_order,
			"metric_emotions": metric_emotions,
			"n_profiles": len(profiles),
			"n_acts": len(acts),
			"n_samples": args.n_samples,
			"fixed_utterances": FIXED_UTTERANCES,
			"profiles": profiles,
			"inference_args": inference_args,
			"metrics": metrics,
			"records": records,
		}, f, indent=2)
	print(f"\nwrote {path}")

	p, n = metrics.get("persona", {}), metrics.get("no_persona", {})
	print(f"\n  divergence   persona={p.get('divergence_mean'):.4f}  "
		  f"no-persona={n.get('divergence_mean'):.4f}")
	print(f"  entropy      persona={p.get('aggregate_entropy'):.3f}  "
		  f"no-persona={n.get('aggregate_entropy'):.3f}")
	print(f"  negative     persona={p.get('negative_mass'):.3f}  "
		  f"no-persona={n.get('negative_mass'):.3f}")
	d = metrics["directional"]["persona"]["fear_sadness_highN_minus_lowN"]
	print(f"  highN-lowN fear+sadness = {d['diff']:+.4f}  CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--sglang_model", default="TheBloke/vicuna-13B-v1.5-AWQ",
					help="model name; ignored with a warning if the server serves another one")
	ap.add_argument("--sglang_host", default=None, help="default $SGLANG_HOST or http://127.0.0.1:30000")
	ap.add_argument("--model_label", required=True, help="short name for the output file, e.g. vicuna-13b")
	ap.add_argument("--n_per_cell", type=int, default=5, help="profiles per 2x2 cell (20 total at 5)")
	ap.add_argument("--n_samples", type=int, default=2, help="repeats per (profile, act) cell")
	ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "outputs", "persona_probe"))
	main(ap.parse_args())
