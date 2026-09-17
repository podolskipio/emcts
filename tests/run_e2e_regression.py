"""Driver for the Part 3.2 end-to-end regression: run ``runners/rollout.py`` against a
given source tree with the deterministic stub backbone and a fixed seed.

    python tests/run_e2e_regression.py --src src --out outputs/post.pkl \
           --dialogs 10 --sims 50 --seed 0 [--beta_emo 0.3]

The point is to run the SAME driver against the pre-change and the post-change tree and
diff the results. The seed is applied here, in this process, rather than through a
``--seed`` flag, because the pre-change tree does not have that flag -- seeding outside
the runner keeps the two invocations byte-identical in every other respect.

``runpy`` is used instead of importing ``main`` so the runner's own argparse block runs:
every default the runner declares is exercised, not a hand-copied subset that could drift
between the two trees.
"""
import argparse
import os
import random
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--src", required=True, help="source tree to run (pre- or post-change)")
	ap.add_argument("--out", required=True, help="output pickle path")
	ap.add_argument("--dialogs", type=int, default=10)
	ap.add_argument("--sims", type=int, default=50)
	ap.add_argument("--max_turns", type=int, default=10)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--beta_emo", type=float, default=0.0)
	ap.add_argument("--runner", default="rollout", choices=["rollout", "emomcts"],
					help="rollout = self-play (SR/AvgT); emomcts = the replay runner")
	ap.add_argument("--game", default="emo_p4g")
	ap.add_argument("--algo", default="emomcts")
	ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
					help="extra flags passed straight to the runner (post-change tree only)")
	args = ap.parse_args()

	src = os.path.abspath(args.src)
	sys.path.insert(0, HERE)
	sys.path.insert(0, src)

	import runners._common as common
	import fake_backbone

	model = fake_backbone.DeterministicChatModel(seed=args.seed)
	common.make_backbone_model = lambda **kwargs: (model, "chat")

	random.seed(args.seed)
	np.random.seed(args.seed)

	runner_path = os.path.join(src, "runners", f"{args.runner}.py")
	sys.argv = [runner_path, "--game", args.game]
	if args.runner == "rollout":
		sys.argv += ["--algo", args.algo,
					 "--max_conv", str(args.dialogs),
					 "--max_turns", str(args.max_turns)]
	else:
		sys.argv += ["--num_dialogs", str(args.dialogs)]
	sys.argv += [
		"--num_mcts_sims", str(args.sims),
		"--num_workers", "1",
		"--beta_emo", str(args.beta_emo),
		"--output", os.path.abspath(args.out),
	] + [a for a in args.extra if a != "--"]

	runpy.run_path(runner_path, run_name="__main__")
	print(f"[regression] stub backbone calls: {model.call_count}")


if __name__ == "__main__":
	main()
