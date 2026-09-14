"""CLI: pairwise LLM-judge comparison of two policy-planner outputs (GDP-Zero style).

Input is one or two pickle files produced by the offline runners
(``runners/raw_prompting.py`` / ``runners/gdpzero*.py``). Each pickle is a list of per-turn
records written by those runners' replay loop:

    {"did", "context", "ori_da", "ori_resp", "new_da", "new_resp", "debug"}

Modes:
  * default (vs. human):   A = record["ori_resp"]  (human ground-truth)
                           B = record["new_resp"]  (the ``-f`` model)
  * head-to-head:          A = h2h_record["new_resp"]   (the ``--h2h`` model)
                           B = record["new_resp"]       (the ``-f`` model)

The judge LLM is asked which response better serves the task goal (see
``evaluators/{p4g,esc,cb}_evaluator.py`` for the task-specific prompts). A/B order is
swapped at random and a majority vote over N samples decides the winner, so a "win" always
means *B beat A*, i.e. the model passed via ``-f`` won.

Reported stats:
  * win   : count where the ``-f`` model's response was preferred
  * lose  : count where the reference (human / ``--h2h``) was preferred
  * draw  : count where the judge could not tell
  * win-rate = win / (win + draw + lose)

Always also writes ``<output_file>_metadata.json`` containing the same headline stats
*plus* a DA-distribution breakdown:
  * per-side DA histogram (A vs B)
  * DA-level agreement count + rate
  * top disagreement transitions (A's pick -> B's pick) with counts
  * **per-(da_a, da_b) win-rate stratification** — which DA swaps cost wins
  * marginal win rate of B per DA it played, and per DA A played

The per-transition stratification is the key debugging signal: e.g., "every time B
played task-inquiry instead of A's proposition-of-donation, B lost 4 out of 6" tells
you the DA mix is the bottleneck, while "B wins 80% on task-inquiry vs A's other but
loses 80% on inquiry vs A's proposition" tells you the bonus matrix needs cell-level
re-tuning. Without this you only see the top-line win rate.

    cd src
    python evaluators/run_judge.py --task p4g -f outputs/gdpzero_p4g.pkl --output outputs/eval_p4g.pkl
    python evaluators/run_judge.py --task esc -f outputs/gdpzero_esc.pkl --h2h outputs/raw_esc.pkl --output outputs/h2h_esc.pkl
    python evaluators/run_judge.py --task cb  -f outputs/gdpzero_cb.pkl  --judge ollama --ollama_model llama3.1
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import json
import pickle
import logging
import argparse
from collections import Counter, defaultdict

from tqdm.auto import tqdm

from evaluators import get_evaluator
from utils.gen_models import OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel


logger = logging.getLogger(__name__)


def _load_records(path):
	with open(path, "rb") as f:
		return pickle.load(f)


def _build_da_metadata(results: list, ref_label: str) -> dict:
	"""Compute DA-distribution + stratified-win-rate metadata from the per-record results.

	Each record in ``results`` is expected to have ``da_a``, ``da_b``, ``winner`` set
	(see the main loop). The summary is the kind of comparison the user would otherwise
	have to recompute by hand against the input pickles — DA histograms, agreement rate,
	top disagreements, and per-(da_a, da_b) win-rate stratification. This is what tells
	you *which DA swaps cost wins*, not just whether the run won overall.
	"""
	# top-line counts per side
	hist_a = Counter(r["da_a"] for r in results if r.get("da_a") is not None)
	hist_b = Counter(r["da_b"] for r in results if r.get("da_b") is not None)

	# DA-level agreement
	paired = [r for r in results if r.get("da_a") is not None and r.get("da_b") is not None]
	agree = sum(1 for r in paired if r["da_a"] == r["da_b"])
	agreement_rate = (agree / len(paired)) if paired else 0.0

	# disagreement transitions (A's DA -> B's DA), sorted by count desc
	transitions = Counter(
		(r["da_a"], r["da_b"]) for r in paired if r["da_a"] != r["da_b"]
	)

	# stratified win rates: for each (da_a, da_b), how often did B win?
	# `winner` convention from the main loop: 1=B wins, 0=A wins, -1/None=draw.
	# Win rate is computed against the (win+lose) denominator so it's directly comparable
	# to the top-line "win rate" — draws are reported separately.
	def _stratify(records):
		w = sum(1 for r in records if r["winner"] == 1)
		l = sum(1 for r in records if r["winner"] == 0)
		d = sum(1 for r in records if r["winner"] not in (0, 1))
		total = w + l + d
		win_rate = w / total if total else 0.0
		return {"win": w, "lose": l, "draw": d, "n": total, "win_rate": round(win_rate, 4)}

	# per (da_a, da_b) — the headline diagnostic table
	per_transition: dict = {}
	for (a_da, b_da), n in transitions.most_common():
		bucket = [r for r in paired if r["da_a"] == a_da and r["da_b"] == b_da]
		per_transition[f"A={a_da} | B={b_da}"] = {"n": n, **_stratify(bucket)}

	# marginal: when B chose DA X, what was its win rate?
	per_da_b: dict = {}
	for da, n in hist_b.most_common():
		bucket = [r for r in paired if r["da_b"] == da]
		per_da_b[da] = _stratify(bucket)

	# marginal on the reference side too — useful when reference is h2h (not human),
	# to spot which DAs the *opponent* is winning with.
	per_da_a: dict = {}
	for da, n in hist_a.most_common():
		bucket = [r for r in paired if r["da_a"] == da]
		# from B's perspective: win rate is the fraction where B beat this A DA choice
		per_da_a[da] = _stratify(bucket)

	# top-K disagreements for the human-readable summary printed to stdout
	top_disagreements = [
		{"A": a_da, "B": b_da, "n": n}
		for (a_da, b_da), n in transitions.most_common(10)
	]

	# Tier-1 ablation attribution: if the input pickle carried counterfactual_da_b
	# (written by runners/emomcts.py), surface "when the bonus changed the chosen DA,
	# what was the win rate?" — a retrospective ablation that needs no extra runs.
	ablation_blocks: dict = {}
	cf_results = [r for r in results if r.get("counterfactual_da_b")]
	if cf_results:
		def _ablation(strategy_key: str) -> dict:
			"""For records where counterfactual[strategy_key] differs from the actually
			played DA, report B's win/lose/draw count + win rate. The interpretation:
			"these are the turns where this shaping term changed the action; did the
			change pay off?"""
			diverged = [
				r for r in cf_results
				if r["counterfactual_da_b"].get(strategy_key) is not None
				and r["counterfactual_da_b"].get(strategy_key) != r["da_b"]
			]
			return {
				"n_total_with_counterfactual": len(cf_results),
				"n_diverged": len(diverged),
				**_stratify(diverged),
			}
		ablation_blocks = {
			# rows where the bonus changed which DA got played
			"bonus_changed_action":   _ablation("puct_no_bonus"),
			# rows where the value-only argmax disagrees with the visit-based pick;
			# Q-vs-exploration trade-off attribution
			"q_vs_visits_disagree":   _ablation("argmax_q"),
		}

	# also stratify by last_user_emotion when available — per-cell calibration signal
	per_emotion_winrate: dict = {}
	emo_paired = [r for r in results if r.get("last_user_emotion") is not None]
	if emo_paired:
		emo_groups: dict = defaultdict(list)
		for r in emo_paired:
			emo_groups[r["last_user_emotion"]].append(r)
		for emo, bucket in sorted(emo_groups.items()):
			per_emotion_winrate[emo] = {"n_records": len(bucket), **_stratify(bucket)}

	meta = {
		"reference_label": ref_label,                  # "human" or "h2h:<path>"
		"da_histogram_A": dict(hist_a.most_common()),
		"da_histogram_B": dict(hist_b.most_common()),
		"da_agreement_count": agree,
		"da_agreement_rate": round(agreement_rate, 4),
		"n_paired": len(paired),
		"top_disagreements": top_disagreements,
		"per_transition_winrate": per_transition,      # the headline debugging table
		"marginal_winrate_when_B_played": per_da_b,
		"marginal_winrate_when_A_played": per_da_a,
	}
	if ablation_blocks:
		meta["shaping_ablation"] = ablation_blocks
	if per_emotion_winrate:
		meta["winrate_by_last_user_emotion"] = per_emotion_winrate
	return meta


def _print_da_summary(meta: dict, label_a: str, label_b: str) -> None:
	"""Compact stdout view of the per-DA breakdown — same info as the JSON, glanceable."""
	print()
	print(f"--- DA distribution ({label_a} vs {label_b}) ---")
	all_das = sorted(set(meta["da_histogram_A"]) | set(meta["da_histogram_B"]))
	for da in all_das:
		a = meta["da_histogram_A"].get(da, 0)
		b = meta["da_histogram_B"].get(da, 0)
		delta = b - a
		marker = f"  ({delta:+d})" if delta else ""
		print(f"  {da:>30}: A={a:>3}  B={b:>3}{marker}")
	print(f"  DA agreement: {meta['da_agreement_count']}/{meta['n_paired']} = "
	      f"{meta['da_agreement_rate']*100:.1f}%")

	print()
	print(f"--- Top DA disagreements (A's pick -> B's pick) ---")
	for row in meta["top_disagreements"][:10]:
		print(f"  A={row['A']:>30}  ->  B={row['B']:<30}  ({row['n']:>3})")

	print()
	print(f"--- B's win rate stratified by (A_da, B_da) — top 10 by count ---")
	for key, stats in list(meta["per_transition_winrate"].items())[:10]:
		print(f"  {key:<70}  win-rate={stats['win_rate']*100:5.1f}%  "
		      f"(w={stats['win']}, l={stats['lose']}, d={stats['draw']})")

	print()
	print(f"--- Marginal: when B played DA X, B's win rate ---")
	for da, stats in meta["marginal_winrate_when_B_played"].items():
		print(f"  {da:>30}: win-rate={stats['win_rate']*100:5.1f}%  "
		      f"(w={stats['win']}, l={stats['lose']}, d={stats['draw']})")

	if "shaping_ablation" in meta:
		print()
		print(f"--- Shaping ablation (retrospective; from Tier-1 counterfactual_da logging) ---")
		for name, stats in meta["shaping_ablation"].items():
			frac = (stats['n_diverged'] / stats['n_total_with_counterfactual']
			        if stats['n_total_with_counterfactual'] else 0)
			print(f"  {name:<30}: diverged on {stats['n_diverged']}/"
			      f"{stats['n_total_with_counterfactual']} ({frac*100:.1f}%)  "
			      f"win-rate on those={stats['win_rate']*100:5.1f}%  "
			      f"(w={stats['win']}, l={stats['lose']}, d={stats['draw']})")

	if "winrate_by_last_user_emotion" in meta:
		print()
		print(f"--- B's win rate by last user emotion ---")
		for emo, stats in meta["winrate_by_last_user_emotion"].items():
			print(f"  {emo:>12}: win-rate={stats['win_rate']*100:5.1f}%  "
			      f"(n={stats['n']}, w={stats['win']}, l={stats['lose']}, d={stats['draw']})")


# Direct OpenAI chat model identifiers that we know to be valid as judges. New OpenAI
# models can be added here without further code changes — they pass through to
# OpenAIChatModel(judge), which validates against the API's live model list. Listed
# explicitly so --judge --help shows them and argparse rejects typos.
_OPENAI_DIRECT_JUDGES = {
	"gpt-3.5-turbo",
	"gpt-4o-mini",
	"gpt-4o",
	"gpt-4-turbo",
	"gpt-4",
}


def _build_judge(judge, ollama_model=None, ollama_host=None):
	"""Construct the LLM backend used as the judge.

	Supported values for ``judge``:
	  * direct OpenAI chat model id (gpt-3.5-turbo, gpt-4o-mini, gpt-4o, gpt-4-turbo,
	    gpt-4) — uses OpenAIChatModel with the model name verbatim. Cost rises with
	    model size; gpt-4o is the recommended upgrade from gpt-3.5-turbo for stronger
	    persuasion-vs-prose discrimination (see debug.md "judge ablation").
	  * "chatgpt" — Azure OpenAI deployment alias (deployment-name configured elsewhere).
	  * "ollama" — locally-hosted model via OllamaChatModel; specify which model with
	    --ollama_model.
	"""
	if judge in _OPENAI_DIRECT_JUDGES:
		return OpenAIChatModel(judge)
	if judge == "chatgpt":
		return AzureOpenAIChatModel(judge)
	if judge == "ollama":
		return OllamaChatModel(ollama_model or "llama3.1", base_url=ollama_host)
	raise ValueError(f"unknown --judge {judge!r}")


def main():
	parser = argparse.ArgumentParser(description="pairwise LLM-judge comparison of two planner outputs")
	parser.add_argument("--task", required=True, choices=["p4g", "esc", "cb"], help="which task evaluator to use")
	parser.add_argument("-f", required=True,
						help="path to the comparison pickle from an offline runner — B = its 'new_resp'")
	parser.add_argument("--h2h", default="",
						help="optional second pickle: A = its 'new_resp' (otherwise A = human 'ori_resp' from -f)")
	parser.add_argument("--judge", default="gpt-3.5-turbo",
						choices=sorted(_OPENAI_DIRECT_JUDGES) + ["chatgpt", "ollama"],
						help="which LLM to use as the judge. Direct OpenAI: gpt-3.5-turbo "
							 "(default, fastest), gpt-4o-mini (cheap upgrade), gpt-4o "
							 "(recommended — stronger persuasion-vs-prose discrimination), "
							 "gpt-4-turbo, gpt-4. 'chatgpt' uses the Azure deployment. "
							 "'ollama' uses a local model (set --ollama_model).")
	parser.add_argument("--ollama_model", default="llama3.1", help="[--judge ollama] model name served by Ollama")
	parser.add_argument("--ollama_host", default=None, help="[--judge ollama] server URL")
	parser.add_argument("--output", default="", help="output pickle path (default: <dir of -f>/evaluation/<name>_evaluated.pkl)")
	parser.add_argument("--out_json", default="", help="also write the summary {win, draw, lose, win_rate, n} to this JSON file")
	parser.add_argument("--limit", type=int, default=-1, help="evaluate at most this many records (-1 = all)")
	parser.add_argument("--debug", action="store_true", help="verbose logging from the ranker")
	args = parser.parse_args()

	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
		logger.setLevel(logging.DEBUG)

	judge_model = _build_judge(args.judge, args.ollama_model, args.ollama_host)
	evaluator = get_evaluator(args.task, judge_model)

	data = _load_records(args.f)
	h2h_data = []
	if args.h2h:
		h2h_data = _load_records(args.h2h)
		if len(data) != len(h2h_data):
			raise ValueError(
				f"--h2h size mismatch: {len(data)} records in {args.f} vs {len(h2h_data)} in {args.h2h}; "
				"both files must come from the same dataset/runner config."
			)
		if not args.output:
			raise ValueError("--output is required when using --h2h (default path collides with vs-human runs)")

	if args.limit > 0:
		data = data[: args.limit]
		if h2h_data:
			h2h_data = h2h_data[: args.limit]

	stats = {"win": 0, "draw": 0, "lose": 0}
	results = []
	for i, d in tqdm(enumerate(data), total=len(data), desc=f"judge:{args.task}"):
		context = d["context"]
		resp_b = d["new_resp"]                                  # the -f model's response
		resp_a = h2h_data[i]["new_resp"] if h2h_data else d["ori_resp"]  # reference (h2h model or human)
		# DA tags: A side is either the h2h model's planned DA or the human's logged DA.
		da_b = d.get("new_da")
		da_a = h2h_data[i].get("new_da") if h2h_data else d.get("ori_da")

		try:
			preference, info = evaluator.evaluate(context, resp_a, resp_b)
		except Exception as e:
			logger.exception(f"judge failed on record {i} (did={d.get('did')}): {e}")
			continue

		if preference == 1:
			stats["win"] += 1
		elif preference == 0:
			stats["lose"] += 1
		else:
			stats["draw"] += 1

		info["winner"] = preference
		info["did"] = d.get("did")
		info["context"] = context
		info["resp_a"] = resp_a
		info["resp_b"] = resp_b
		info["da_a"] = da_a
		info["da_b"] = da_b
		# Tier-1 attribution fields written by runners/emomcts.py — if present, they
		# let _build_da_metadata produce a per-ablation win/lose table without re-running.
		info["counterfactual_da_b"] = d.get("counterfactual_da")
		info["last_user_emotion"] = d.get("last_user_emotion")
		info["turn_index"] = d.get("turn_index")
		results.append(info)

	# resolve output path
	if args.output:
		output_file = args.output
	else:
		out_dir = os.path.join(os.path.dirname(args.f) or ".", "evaluation")
		os.makedirs(out_dir, exist_ok=True)
		output_file = os.path.join(out_dir, os.path.basename(args.f).replace(".pkl", "_evaluated.pkl"))

	out_parent = os.path.dirname(output_file)
	if out_parent:
		os.makedirs(out_parent, exist_ok=True)
	with open(output_file, "wb") as f:
		pickle.dump(results, f)

	total = sum(stats.values())
	win_rate = (stats["win"] / total) if total else 0.0

	# DA-distribution metadata: histograms per side, agreement, top disagreements, and
	# (most useful) per-(da_a, da_b) win-rate stratification. Tells us *which* DA swaps
	# cost wins, not just whether the run won. Always computed; always written to a
	# sibling _metadata.json next to the output pickle so downstream diffing scripts
	# don't have to re-derive it from the pickles.
	ref_label = f"h2h:{args.h2h}" if h2h_data else "human"
	da_meta = _build_da_metadata(results, ref_label=ref_label)

	summary = {
		**stats,
		"n": total,
		"win_rate": win_rate,
		"task": args.task,
		"judge": args.judge,
		"reference": ref_label,
		"f_input": args.f,
		"h2h_input": args.h2h or None,
		"output_file": output_file,
		"da_metadata": da_meta,
	}

	print(f"task={args.task}  judge={args.judge}  vs={'h2h' if h2h_data else 'human'}")
	print(f"win rate: {win_rate * 100.0:.2f}%")
	print(f"stats: {stats}")
	print(f"saved per-record decisions to {output_file}")

	label_a = "h2h" if h2h_data else "human"
	label_b = "f"
	_print_da_summary(da_meta, label_a=label_a, label_b=label_b)

	# default metadata sidecar: <output_file>_metadata.json (replaces .pkl extension).
	metadata_file = (
		args.out_json
		or output_file.replace(".pkl", "_metadata.json")
	)
	# if --out_json was given, also write the metadata file alongside the pickle so the
	# DA breakdown is always discoverable next to the per-record decisions.
	if args.out_json and args.out_json != output_file.replace(".pkl", "_metadata.json"):
		alt_metadata_file = output_file.replace(".pkl", "_metadata.json")
		with open(alt_metadata_file, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2)
		print(f"saved DA metadata to {alt_metadata_file}")
	with open(metadata_file, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)
	print(f"saved summary + DA metadata to {metadata_file}")


if __name__ == "__main__":
	main()
