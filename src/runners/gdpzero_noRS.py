"""GDP-Zero ablation: open-loop MCTS, but the final utterance is freshly generated.

Builds the open-loop MCTS tree exactly like `runners/gdpzero.py`, but instead of returning
the best-scoring realization from the tree it re-generates the chosen dialog act with
`game.get_next_state` (no realization *selection*).

    cd src
    python runners/gdpzero_noRS.py --game p4g --data data/p4g/300_dialog_turn_based.pkl

NOTE: see the caveat in `runners/gdpzero.py` about the MCTS <-> game `get_next_state` API
mismatch -- the MCTS runners need that reconciliation before they run end-to-end.
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


def plan_turn(game, system, planner, backbone_model, state, cmd_args):
	configs = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})
	dp = OpenLoopMCTS(game, planner, configs)
	for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="mcts"):
		dp.search(state)
	policy = dp.get_action_prob(state)
	best = int(np.argmax(policy))
	da = system.dialog_acts[best]
	next_state, _ = game.get_next_state(state, best)
	utt = next_state.history[-2][2]
	debug = {
		"probs": policy, "da": da,
		"search_tree": {
			"Ns": dp.Ns, "Nsa": dp.Nsa, "Q": dp.Q, "P": dp.P, "Vs": dp.Vs,
			"realizations": dp.realizations,
		},
	}
	return da, utt, debug


def main():
	parser = argparse.ArgumentParser(description="GDP-Zero ablation: open-loop MCTS, no realization selection")
	add_common_args(parser, default_output="outputs/gdpzero_noRS.pkl")
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="number of MCTS simulations per turn")
	parser.add_argument("--max_realizations", type=int, default=3, help="number of realizations per MCTS state")
	parser.add_argument("--Q_0", type=float, default=0.0, help="initial Q value for uninitialized states")
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)
	run_eval(cmd_args.game, cmd_args, plan_turn)


if __name__ == "__main__":
	main()
