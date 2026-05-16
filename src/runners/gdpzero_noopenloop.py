"""GDP-Zero ablation: closed-loop MCTS (no open-loop realization pool).

Same as `runners/gdpzero.py` but uses the plain `MCTS` class (one concrete trajectory per
DA-prefix node) and realizes the chosen dialog act once via `game.get_next_state`. The user
simulator is run deterministically (`do_sample=False`) since there is no open-loop sampling.

    cd src
    python runners/gdpzero_noopenloop.py --game p4g --data data/p4g/300_dialog_turn_based.pkl

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
from mcts.mcts import MCTS
from runners._common import run_eval, add_common_args, finalize_args


def plan_turn(game, system, planner, backbone_model, state, cmd_args):
	configs = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": 1,  # unused by MCTS, kept for config compatibility
	})
	dp = MCTS(game, planner, configs)
	for _ in tqdm(range(configs.num_MCTS_sims), leave=False, desc="mcts"):
		dp.search(state)
	policy = dp.get_action_prob(state)
	best = int(np.argmax(policy))
	da = system.dialog_acts[best]
	next_state, _ = game.get_next_state(state, best)
	utt = next_state.history[-2][2]
	debug = {
		"probs": policy, "da": da,
		"search_tree": {"Ns": dp.Ns, "Nsa": dp.Nsa, "Q": dp.Q, "P": dp.P, "Vs": dp.Vs},
	}
	return da, utt, debug


def main():
	parser = argparse.ArgumentParser(description="GDP-Zero ablation: closed-loop MCTS")
	add_common_args(parser, default_output="outputs/gdpzero_noopenloop.pkl")
	parser.add_argument("--num_mcts_sims", type=int, default=20, help="number of MCTS simulations per turn")
	parser.add_argument("--Q_0", type=float, default=0.0, help="initial Q value for uninitialized states")
	cmd_args = finalize_args(parser.parse_args())
	print("saving to", cmd_args.output)
	# closed-loop: deterministic user simulator; system uses model defaults
	run_eval(
		cmd_args.game, cmd_args, plan_turn,
		sys_inference_args={},
		usr_inference_args={
			"max_new_tokens": 128, "temperature": 1.0, "repetition_penalty": 1.0,
			"do_sample": False, "return_full_text": False,
		},
	)


if __name__ == "__main__":
	main()
