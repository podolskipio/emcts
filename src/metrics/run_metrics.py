"""CLI: compute SR / AT (and SL for CraigslistBargain) for a file of episode records.

The episode records are what ``runners/rollout.py`` writes (a pickle, or a .json / .jsonl
list); see ``metrics/dialog_metrics.py`` for the expected schema.

    cd src
    python metrics/run_metrics.py --episodes outputs/rollout_cb.pkl --task cb --max_turns 8
    python metrics/run_metrics.py --episodes outputs/rollout_p4g.pkl --max_turns 10 --out outputs/metrics_p4g.json
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import json
import pickle
import argparse

from metrics.dialog_metrics import compute_metrics, format_metrics


def load_episodes(path):
	if path.endswith(".jsonl"):
		with open(path, "r", encoding="utf-8") as f:
			return [json.loads(line) for line in f if line.strip()]
	if path.endswith(".json"):
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		return obj if isinstance(obj, list) else obj.get("episodes", [obj])
	with open(path, "rb") as f:
		obj = pickle.load(f)
	return obj if isinstance(obj, list) else obj.get("episodes", obj)


def main():
	parser = argparse.ArgumentParser(description="compute episode-level metrics (SR / AT / SL)")
	parser.add_argument("--episodes", required=True, help="path to the episode records (pickle / json / jsonl)")
	parser.add_argument("--task", choices=["p4g", "esc", "cb"], default=None,
						help="task (only affects whether SL is reported; auto-detected otherwise)")
	parser.add_argument("--max_turns", type=int, default=None, help="turn limit for SR / AT (default: none)")
	parser.add_argument("--at_successes_only", action="store_true",
						help="average AT over successful episodes only (default: failures count as --max_turns)")
	parser.add_argument("--no_sl_clip", action="store_true", help="report the raw (unclipped) SL ratio")
	parser.add_argument("--out", default=None, help="also write the metrics dict to this json file")
	args = parser.parse_args()

	episodes = load_episodes(args.episodes)
	metrics = compute_metrics(
		episodes,
		task=args.task,
		max_turns=args.max_turns,
		failures_as_max=not args.at_successes_only,
		sl_clip=None if args.no_sl_clip else (0.0, 1.0),
	)
	print(format_metrics(metrics))
	print(json.dumps(metrics, indent=2))
	if args.out:
		out_dir = os.path.dirname(args.out)
		if out_dir:
			os.makedirs(out_dir, exist_ok=True)
		with open(args.out, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)


if __name__ == "__main__":
	main()
