#!/usr/bin/env python3
"""How reproducible is a logit-scored value, exactly? (paper §W5)

Replacing sampling with scoring removes the sampler's variance, but it does not make the pipeline
bit-exact: the logprobs come off GPU kernels whose reduction order depends on how the request was
batched and how much of the prefix was already cached. This script measures that residual, so the
reproducibility claim in W5_COST_TABLE.md is a number rather than an assertion.

Three conditions, same states, same prompt each time:

  * **warm**        -- repeated back to back, prefix resident in the radix cache
  * **flushed**     -- `/flush_cache` before each repeat, so every call re-prefills from scratch
  * **concurrent**  -- all states in flight at once, so requests batch with each other

Reported per condition: the spread of the raw label logprobs, and the spread of `v` -- the number
that actually reaches MCTS. Compare against the sampled estimator's own spread, which the same
script measures by running the generation path repeatedly.

    cd src && python ../scripts/logit_stability.py --n_states 30 --repeats 5
"""
import argparse
import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from runners._common import TASKS, build_agents, make_backbone_model  # noqa: E402
from utils.rewards import reward_dict                                # noqa: E402

from validate_logit_scoring import collect_states, run_pass          # noqa: E402


def spreads(values):
    """values: (repeats, states). Per-state peak-to-peak, summarized across states."""
    ptp = values.max(0) - values.min(0)
    return {"mean_spread": float(ptp.mean()), "max_spread": float(ptp.max()),
            "exactly_identical_states": int((ptp == 0).sum()), "n_states": int(ptp.shape[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_states", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sglang_model", type=str, default="TheBloke/vicuna-13B-v1.5-AWQ")
    ap.add_argument("--host", type=str, default="http://127.0.0.1:30000")
    ap.add_argument("--explicit_value_labels", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/logit_stability.json")
    args = ap.parse_args()

    backbone, family = make_backbone_model("sglang", sglang_model=args.sglang_model)
    _, system, _, gen_planner = build_agents("p4g", backbone, family, logit_scoring=False,
                                             explicit_value_labels=args.explicit_value_labels)
    _, _, _, logit_planner = build_agents("p4g", backbone, family, logit_scoring="both",
                                          explicit_value_labels=args.explicit_value_labels)
    states = collect_states(args.n_states, args.seed, TASKS["p4g"].default_data, system.dialog_acts)
    print(f"{len(states)} states x {args.repeats} repeats\n")

    def score_all(workers, flush):
        if flush:
            requests.post(f"{args.host}/flush_cache", timeout=60)
        out, _ = run_pass("score", lambda s: logit_planner.score_value_labels(s), states, workers)
        return (np.array([o.logprobs for o in out]),
                np.array([o.expectation(reward_dict["p4g"]) for o in out]))

    report = {"n_states": len(states), "repeats": args.repeats,
              "explicit_value_labels": args.explicit_value_labels, "conditions": {}}

    for name, workers, flush in (("warm", 1, False), ("flushed", 1, True), ("concurrent", 10, False)):
        L, V = [], []
        for _ in range(args.repeats):
            lp, v = score_all(workers, flush)
            L.append(lp); V.append(v)
        L = np.array(L)                       # (repeats, states, labels)
        report["conditions"][name] = {
            "logprob": spreads(L.reshape(args.repeats, -1)),
            "v": spreads(np.array(V)),
        }
        c = report["conditions"][name]
        print(f"{name:11s} logprob spread mean {c['logprob']['mean_spread']:.2e} "
              f"max {c['logprob']['max_spread']:.2e}   |   "
              f"v spread mean {c['v']['mean_spread']:.2e} max {c['v']['max_spread']:.2e}   |   "
              f"identical {c['v']['exactly_identical_states']}/{c['v']['n_states']} states")

    # The thing it replaces, measured the same way.
    Vg = []
    for i in range(args.repeats):
        out, _ = run_pass(f"gen {i+1}", lambda s: gen_planner.heuristic(s), states, 10)
        Vg.append([v for v, _ in out])
    report["sampled_baseline_v"] = spreads(np.array(Vg))
    b = report["sampled_baseline_v"]
    print(f"\n{'sampled':11s} v spread mean {b['mean_spread']:.2e} max {b['max_spread']:.2e}   |   "
          f"identical {b['exactly_identical_states']}/{b['n_states']} states")

    worst = max(report["conditions"][c]["v"]["max_spread"] for c in report["conditions"])
    print(f"\nworst-case v spread, scored: {worst:.2e};  sampled: {b['max_spread']:.2e}"
          f"  ({b['max_spread'] / worst:.0f}x wider)" if worst else "")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
