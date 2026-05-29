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

from tqdm.auto import tqdm

from evaluators import get_evaluator
from utils.gen_models import OpenAIChatModel, AzureOpenAIChatModel, OllamaChatModel


logger = logging.getLogger(__name__)


def _load_records(path):
	with open(path, "rb") as f:
		return pickle.load(f)


def _build_judge(judge, ollama_model=None, ollama_host=None):
	"""Construct the LLM backend used as the judge."""
	if judge == "gpt-3.5-turbo":
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
						choices=["gpt-3.5-turbo", "chatgpt", "ollama"], help="which LLM to use as the judge")
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
	summary = {**stats, "n": total, "win_rate": win_rate}
	print(f"task={args.task}  judge={args.judge}  vs={'h2h' if h2h_data else 'human'}")
	print(f"win rate: {win_rate * 100.0:.2f}%")
	print(f"stats: {stats}")
	print(f"saved per-record decisions to {output_file}")

	if args.out_json:
		with open(args.out_json, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2)


if __name__ == "__main__":
	main()
