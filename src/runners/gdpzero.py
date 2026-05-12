"""GDP-Zero: open-loop MCTS policy planning with realization sampling.

Replays a held-out dataset and, at every turn, runs `OpenLoopMCTS` for `--num_mcts_sims`
simulations, takes the argmax of the resulting visit-count policy as the next dialog act,
and uses `get_best_realization` to pick the concrete utterance.

    cd src
    python runners/gdpzero.py --game p4g --data data/p4g/300_dialog_turn_based.pkl --num_mcts_sims 20
    python runners/gdpzero.py --game esc --data data/esc/esc-valid.json
    python runners/gdpzero.py --game cb  --data data/cb/cb-valid.txt

NOTE: the MCTS classes in `mcts/mcts.py` currently call `game.get_next_state(state, action)`
with two positional args and `game.get_next_state_batched(...)`, neither of which the game
classes support yet -- so the MCTS-based runners need that reconciliation before they run
end-to-end. `runners/raw_prompting.py` works today.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import argparse
import numpy as np
from tqdm.auto import tqdm

from utils.utils import dotdict
from mcts.mcts import OpenLoopMCTS
from runners._common import run_eval, add_common_args, finalize_args


def _mcts_configs(cmd_args):
	return dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})


def plan_turn(game, system, planner, backbone_model, state, cmd_args):
	configs = _mcts_configs(cmd_args)
	dp = OpenLoopMCTS(game, planner, configs)
	for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="mcts"):
		dp.search(state)
	policy = dp.get_action_prob(state)
	best = int(np.argmax(policy))
	da = system.dialog_acts[best]
	utt = dp.get_best_realization(state, best)
	debug = {
		"probs": policy, "da": da,
		"search_tree": {
			"Ns": dp.Ns, "Nsa": dp.Nsa, "Q": dp.Q, "P": dp.P, "Vs": dp.Vs,
			"realizations": dp.realizations,
			"realizations_Vs": dp.realizations_Vs,
			"realizations_Ns": dp.realizations_Ns,
		},
	}
	return da, utt, debug


def main():
	parser = argparse.ArgumentParser(description="GDP-Zero (open-loop MCTS w/ realization sampling)")
	add_common_args(parser, default_output="outputs/gdpzero.pkl")
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="number of MCTS simulations per turn")
	parser.add_argument("--max_realizations", type=int, default=3, help="number of realizations per MCTS state")
	parser.add_argument("--Q_0", type=float, default=0.0, help="initial Q value for uninitialized states")
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)
	run_eval(cmd_args.game, cmd_args, plan_turn)


if __name__ == "__main__":
	main()
