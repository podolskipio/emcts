"""Raw-prompting baseline: greedy one-step policy planning (no tree search).

Replays a held-out dataset and, at every turn, asks the chat planner for its prior over
dialog acts, picks the argmax, and realizes that act with `game.get_next_state`.

Works for all three tasks::

    cd src
    python runners/raw_prompting.py --game p4g --data data/p4g/300_dialog_turn_based.pkl
    python runners/raw_prompting.py --game esc --data data/esc/esc-valid.json
    python runners/raw_prompting.py --game cb  --data data/cb/cb-valid.txt --llm ollama --ollama_model llama3.1
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import argparse
import numpy as np

from runners._common import run_eval, add_common_args, finalize_args


def plan_turn(game, system, planner, backbone_model, state, cmd_args):
	prior, v = planner.predict(state)
	best = int(np.argmax(prior))
	da = system.dialog_acts[best]
	next_state, _agent_state, _ = game.get_next_state(state, best, agent_state=[])
	utt = next_state.history[-2][2]  # [sys_turn, simulated_user_turn] -> the system utterance
	return da, utt, {"prior": prior, "da": da, "v": v}


def main():
	parser = argparse.ArgumentParser(description="raw-prompting (greedy one-step) policy planning")
	add_common_args(parser, default_output="outputs/raw_prompt.pkl")
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)
	# raw prompting uses the model's built-in inference defaults for the system agent
	run_eval(cmd_args.game, cmd_args, plan_turn, sys_inference_args={})


if __name__ == "__main__":
	main()
