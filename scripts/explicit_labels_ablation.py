#!/usr/bin/env python3
"""Does making the label set explicit at the answer position raise value agreement? (§W5)

The logit-scoring validation lands at r ~= 0.94 against a single 10-sample generation run, below
the 0.95 acceptance bar, and the prescribed remedy is "the prompt needs the label set made
explicit". This script tests that remedy instead of arguing about it.

In the value prompt as built, the five donation labels appear exactly once -- in the leading
system message, ~1000 tokens before the position where the answer is read -- and the few-shot demo
exhibits only four of them (`no donation` never occurs in it). Each variant below restates the set
closer to the answer, and is applied to **both** arms, so the sampling baseline and the scored
value always see the identical prompt and the comparison stays honest.

Reported per variant: r against a single generation run (the acceptance criterion as written), the
baseline's own test-retest reliability (the ceiling that criterion is measured against), and the
share of the model's next-token mass already sitting on the label set.

    cd src && python ../scripts/explicit_labels_ablation.py --n_states 200 --num_workers 10
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from games.p4g_game import PersuasionGame                    # noqa: E402
from players.p4g_players import P4GChatSystemPlanner         # noqa: E402
from runners._common import TASKS, build_agents, make_backbone_model  # noqa: E402
from utils.rewards import reward_dict                        # noqa: E402

from validate_logit_scoring import collect_states, pearson, run_pass, spearman  # noqa: E402


def variants(labels):
    """(name, text appended to the final user message). '' is the prompt as it ships."""
    listing = " ".join(f"[{da}]" for da in labels)
    return [
        ("as-is", ""),
        # The set, restated where the answer is given rather than 1000 tokens earlier.
        ("labels restated",
         f"\n(Reply as the Persuadee. Begin your reply with exactly one of these labels: {listing})"),
        # As above, plus naming what the label is *for* -- the question being answered is
        # "would you donate", and the labels are the permitted answers to it.
        ("labels + answer framing",
         f"\nAnswer with your reaction to that question, beginning with exactly one of these "
         f"labels: {listing}. Use [donate] only if you are agreeing to donate, and "
         f"[no donation] only if you are refusing."),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_states", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--gen_replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sglang_model", type=str, default="TheBloke/vicuna-13B-v1.5-AWQ")
    ap.add_argument("--out", type=str, default="outputs/explicit_labels_ablation.json")
    args = ap.parse_args()

    backbone, family = make_backbone_model("sglang", sglang_model=args.sglang_model)
    _, system, _, gen_planner = build_agents("p4g", backbone, family, logit_scoring=False)
    _, _, _, logit_planner = build_agents("p4g", backbone, family, logit_scoring=True)
    states = collect_states(args.n_states, args.seed, TASKS["p4g"].default_data, system.dialog_acts)
    print(f"{len(states)} states, {args.gen_replicates} generation replicates per variant\n")

    original = P4GChatSystemPlanner._build_value_messages
    report = {"n_states": len(states), "gen_replicates": args.gen_replicates, "variants": {}}

    for name, suffix in variants(logit_planner.user_dialog_acts):
        # Patch the shared builder, so the sampling arm and the scored arm get the same prompt.
        def patched(self, state, _suffix=suffix):
            messages = original(self, state)
            if not _suffix:
                return messages
            messages = list(messages)
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + _suffix}
            return messages
        P4GChatSystemPlanner._build_value_messages = patched
        try:
            print(f"== {name} ==")
            V = []
            for i in range(args.gen_replicates):
                out, _ = run_pass(f"{name} gen {i+1}", lambda s: gen_planner.heuristic(s),
                                  states, args.num_workers)
                V.append([v for v, _ in out])
            V = np.array(V)
            scored, _ = run_pass(f"{name} logit",
                                 lambda s: logit_planner.score_value_labels(s, top_logprobs_num=20),
                                 states, args.num_workers)
            v_logit = np.array([s.expectation(reward_dict["p4g"]) for s in scored])
            mass = [s.label_set_mass for s in scored if s.label_set_mass is not None]
            kept = np.array([len(das) for _, das in out])

            pairs = [pearson(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V))]
            ceiling = float(np.mean(pairs))
            r1 = pearson(V[0], v_logit)
            entry = {
                "r_vs_single_gen_run": r1,
                "spearman_vs_single_gen_run": spearman(V[0], v_logit),
                "r_vs_mean_of_all_gen_runs": pearson(V.mean(0), v_logit),
                "gen_test_retest_reliability": ceiling,
                "disattenuated": r1 / np.sqrt(ceiling) if ceiling > 0 else float("nan"),
                "label_set_mass_mean": float(np.mean(mass)) if mass else None,
                "label_set_mass_min": float(np.min(mass)) if mass else None,
                "mean_usable_gen_samples": float(kept.mean()),
                "p_donate_mean": float(np.mean([s.prob_of(PersuasionGame.U_Donate) for s in scored])),
            }
            report["variants"][name] = entry
            print(f"  r vs single gen run        {entry['r_vs_single_gen_run']:.4f}")
            print(f"  gen test-retest ceiling    {entry['gen_test_retest_reliability']:.4f}")
            print(f"  r vs mean of {len(V)} gen runs   {entry['r_vs_mean_of_all_gen_runs']:.4f}")
            print(f"  disattenuated              {entry['disattenuated']:.4f}")
            print(f"  label-set mass             {entry['label_set_mass_mean']:.4f} "
                  f"(min {entry['label_set_mass_min']:.4f})")
            print(f"  usable gen samples         {entry['mean_usable_gen_samples']:.2f}/10\n")
        finally:
            P4GChatSystemPlanner._build_value_messages = original

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== summary (acceptance criterion: r vs a single generation run >= 0.95) ===")
    print(f"{'variant':26s} {'r':>8s} {'ceiling':>9s} {'vs mean':>9s} {'disatt':>8s} {'mass':>7s}")
    for name, e in report["variants"].items():
        print(f"{name:26s} {e['r_vs_single_gen_run']:8.4f} {e['gen_test_retest_reliability']:9.4f} "
              f"{e['r_vs_mean_of_all_gen_runs']:9.4f} {e['disattenuated']:8.4f} "
              f"{e['label_set_mass_mean']:7.3f}")
    best = max(report["variants"], key=lambda n: report["variants"][n]["r_vs_single_gen_run"])
    print(f"\nbest by the criterion as written: {best!r} at "
          f"{report['variants'][best]['r_vs_single_gen_run']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
