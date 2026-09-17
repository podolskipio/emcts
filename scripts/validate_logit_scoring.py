#!/usr/bin/env python3
"""Validate logit-scored value / prior against the sampling paths they replace (paper §W5).

The two roles being replaced are Monte-Carlo estimators of quantities the model's logits give
exactly:

  * **value estimator** -- 10 persuadee completions at temperature 1.1, DA read out of each,
    ``reward_dict['p4g']`` averaged over them. The logit path scores the 5 donation labels
    against the same prompt and takes the exact expectation.
  * **policy prior** -- 15 persuader completions, DA histogrammed, ``self.smoothing`` added.
    The logit path scores the persuader's act set against the same prompt.

Same prompt in both arms: the sampling and logit paths share ``_build_value_messages`` /
``_build_prior_messages``, so any difference is estimator, not prompt.

The headline is the value correlation. Read it against the **noise ceiling** this script also
measures: two *independent* 10-sample runs of the generation path against each other. A
10-sample mean of a 5-valued variable is a noisy thing to correlate with, and no estimator --
including a perfect one -- can correlate with it above that ceiling. Reporting r without the
ceiling makes an estimator look worse the noisier the baseline is.

Usage (server must be up -- scripts/serve_sglang.sh):

    cd src && python ../scripts/validate_logit_scoring.py --n_states 200 --num_workers 10
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from games.p4g_game import PersuasionGame           # noqa: E402
from runners._common import TASKS, build_agents, make_backbone_model, resolve_data_path  # noqa: E402
from utils.rewards import reward_dict                # noqa: E402
from utils.sessions import DialogSession            # noqa: E402


def collect_states(n_states, seed, data_path, system_dialog_acts):
    """``n_states`` dialogue prefixes that end on a persuadee turn -- the states MCTS asks the
    value estimator about. Sampled one prefix per dialog, cycling over dialogs so the sample
    spreads across conversations and depths rather than piling into a few long ones."""
    dialogs = TASKS["p4g"].read_dialogs(resolve_data_path(data_path), set(system_dialog_acts))
    rng = random.Random(seed)
    candidates = []
    for dialog in dialogs:
        turns = dialog["turns"]
        for depth in range(1, len(turns) + 1):
            candidates.append((dialog["id"], turns[:depth]))
    rng.shuffle(candidates)

    seen_per_dialog = {}
    picked = []
    # round 1: at most one prefix per dialog, then relax
    for did, prefix in candidates:
        if seen_per_dialog.get(did):
            continue
        seen_per_dialog[did] = True
        picked.append((did, prefix))
        if len(picked) == n_states:
            break
    for did, prefix in candidates:
        if len(picked) == n_states:
            break
        picked.append((did, prefix))

    states = []
    for did, prefix in picked[:n_states]:
        state = DialogSession(PersuasionGame.SYS, PersuasionGame.USR)
        for turn in prefix:
            state.add_single(PersuasionGame.SYS, turn["sys_da"], turn["sys_utt"])
            state.add_single(PersuasionGame.USR, turn["usr_da"], turn["usr_utt"])
        states.append((did, len(prefix), state))
    return states


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def ranks(x):
        order = np.argsort(np.asarray(x, float), kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties, otherwise ties are ranked by input order and the coefficient is wrong
        x = np.asarray(x, float)
        for value in np.unique(x):
            mask = x == value
            r[mask] = r[mask].mean()
        return r
    return pearson(ranks(a), ranks(b))


def run_pass(label, fn, states, num_workers):
    """Apply ``fn`` to every state, ``num_workers`` at a time, and time the whole pass."""
    results = [None] * len(states)
    done = [0]
    lock = Lock()

    def one(i):
        out = fn(states[i][2])
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(states):
                print(f"  [{label}] {done[0]}/{len(states)}", flush=True)
        return i, out

    t0 = time.monotonic()
    pool = ThreadPool(processes=max(1, num_workers))
    for i, out in pool.imap_unordered(one, range(len(states))):
        results[i] = out
    pool.close()
    pool.join()
    return results, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_states", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sglang_model", type=str, default="TheBloke/vicuna-13B-v1.5-AWQ")
    ap.add_argument("--data", type=str, default=TASKS["p4g"].default_data)
    ap.add_argument("--gen_replicates", type=int, default=5,
                    help="independent 10-sample runs of the generation value. One is the "
                         "baseline as the pipeline actually uses it; the extra ones measure how "
                         "much of the disagreement is that baseline's own sampling noise.")
    ap.add_argument("--skip_prior", action="store_true", help="value estimator only")
    ap.add_argument("--prior_topk", type=int, default=5,
                    help="also score the --llm_prior_topk ranking call against the same "
                         "histogram baseline. That call is what the deployed configuration "
                         "actually runs, so this says whether the logit prior is closer or "
                         "further from the prior both of them approximate. 0 skips it.")
    ap.add_argument("--out", type=str, default="outputs/logit_scoring_validation.json")
    args = ap.parse_args()

    backbone, family = make_backbone_model("sglang", sglang_model=args.sglang_model)
    # Two planners over one backbone: the flag is read per call, and the passes below run
    # threaded, so flipping it on a shared planner would race.
    _, system, _, gen_planner = build_agents("p4g", backbone, family, logit_scoring=False)
    _, _, _, logit_planner = build_agents("p4g", backbone, family, logit_scoring=True)
    assert logit_planner.logit_scoring, "backbone cannot score labels -- is --llm sglang serving?"

    states = collect_states(args.n_states, args.seed, args.data, system.dialog_acts)
    print(f"{len(states)} states, depths {min(d for _, d, _ in states)}-{max(d for _, d, _ in states)}, "
          f"{len({i for i, _, _ in states})} distinct dialogs")
    print(f"acts scored: {len(logit_planner.dialog_acts)} persuader "
          f"({', '.join(logit_planner.dialog_acts)})")
    print(f"labels scored: {len(logit_planner.user_dialog_acts)} persuadee "
          f"({', '.join(logit_planner.user_dialog_acts)})\n")

    report = {"n_states": len(states), "model": args.sglang_model, "seed": args.seed}

    # ---- value estimator -------------------------------------------------
    print("== value estimator ==")
    replicates, gen_seconds = [], []
    for r_i in range(args.gen_replicates):
        out, secs = run_pass(f"gen {r_i + 1}/{args.gen_replicates}",
                             lambda s: gen_planner.heuristic(s), states, args.num_workers)
        replicates.append(out)
        gen_seconds.append(secs)
    scored, t_logit = run_pass("logit", lambda s: logit_planner.score_value_labels(s, top_logprobs_num=20),
                               states, args.num_workers)

    V = np.array([[v for v, _ in rep] for rep in replicates])   # (replicates, states)
    v_gen_a = V[0]
    v_logit = np.array([s.expectation(reward_dict["p4g"]) for s in scored])
    t_gen_a = gen_seconds[0]

    # How much of the disagreement is the baseline arguing with itself. `r_ceiling` is the mean
    # correlation between two independent 10-sample runs -- the reliability of the number the
    # pipeline actually uses -- so it bounds what *any* estimator, including a noiseless one,
    # can score against a single run. Averaging m runs lifts that bound, and the logit value's
    # correlation against the m-run mean rising with m is the evidence that what remains is the
    # baseline's noise rather than a disagreement about the state.
    pairwise = [pearson(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V))]
    r_ceiling = float(np.mean(pairwise)) if pairwise else float("nan")
    vs_mean_of_m = {}
    for m in range(1, len(V) + 1):
        vs_mean_of_m[m] = pearson(V[:m].mean(0), v_logit)
    r_logit = pearson(v_gen_a, v_logit)
    # Classic attenuation correction: r_true = r_observed / sqrt(reliability of the baseline).
    disattenuated = r_logit / np.sqrt(r_ceiling) if r_ceiling == r_ceiling and r_ceiling > 0 else float("nan")

    # P(donate) each way: the logit path reads it off directly, the sampling path can only
    # estimate it as a fraction of 10 draws.
    donate_logit = [s.prob_of(PersuasionGame.U_Donate) for s in scored]
    donate_gen = [np.mean([d == PersuasionGame.U_Donate for d in das]) if das else 0.0 for _, das in replicates[0]]
    first_vs_full = [pearson(s.probs, s.first_token_probs) for s in scored]
    # Is the label set explicit enough for renormalizing over it to be honest? This is the share
    # of the model's own next-token mass (top 20 tokens) that already sits on one of the five
    # labels. Near 1.0 and the closed-set assumption costs nothing; well below and the prompt --
    # not the estimator -- is what needs fixing.
    label_mass = [s.label_set_mass for s in scored if s.label_set_mass is not None]
    # Samples the generation path had to throw away: `_get_user_generated_da` keeps only
    # completions whose bracketed DA is one of the five labels. Those are states where the two
    # paths are answering slightly different questions -- the logit path conditions on the label
    # set by construction and cannot discard anything.
    kept = np.array([len(das) for _, das in replicates[0]])
    report["value"] = {
        "gen_replicates": args.gen_replicates,
        "pearson_logit_vs_gen": r_logit,
        "spearman_logit_vs_gen": spearman(v_gen_a, v_logit),
        "pearson_gen_vs_gen_noise_ceiling": r_ceiling,
        "pearson_logit_vs_mean_of_m_gen_runs": {str(m): v for m, v in vs_mean_of_m.items()},
        "pearson_logit_vs_gen_disattenuated": float(disattenuated),
        "pearson_donate_prob": pearson(donate_gen, donate_logit),
        "distinct_values_gen": len(set(np.round(v_gen_a, 6))),
        "distinct_values_logit": len(set(np.round(v_logit, 6))),
        "mean_usable_samples_per_call": float(kept.mean()),
        "states_with_zero_usable_samples": int((kept == 0).sum()),
        "mean_label_set_mass": float(np.mean(label_mass)) if label_mass else None,
        "min_label_set_mass": float(np.min(label_mass)) if label_mass else None,
        "mean_first_token_vs_full_sequence_pearson": float(np.mean(first_vs_full)),
        "min_first_token_vs_full_sequence_pearson": float(np.min(first_vs_full)),
        "seconds": {"gen_per_replicate": gen_seconds, "logit": t_logit},
        "speedup_wall": t_gen_a / t_logit if t_logit else None,
    }
    print(f"\n  logit vs one generation run   pearson {r_logit:.4f}   spearman {report['value']['spearman_logit_vs_gen']:.4f}")
    print(f"  noise ceiling (gen vs gen, {len(pairwise)} pairs)  pearson {r_ceiling:.4f}")
    print("  logit vs the mean of m generation runs:")
    for m, v in vs_mean_of_m.items():
        print(f"      m={m} ({10 * m:3d} samples)  {v:.4f}")
    print(f"  disattenuated (r / sqrt(reliability))  {disattenuated:.4f}")

    # The one substantive difference between the two estimators, as opposed to noise: the
    # sampling path draws at temperature 1.1, so the label distribution it is estimating is the
    # logits flattened by 1.1, not the raw softmax. Reading the same logprobs back at a matching
    # temperature costs nothing (no extra call) and is the honest like-for-like comparison.
    v_ref = V.mean(0)
    temp_sweep = {}
    for T in (0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0):
        v_T = np.array([sc.at_temperature(T).expectation(reward_dict["p4g"]) for sc in scored])
        temp_sweep[T] = {"vs_single_run": pearson(v_gen_a, v_T),
                         "vs_mean_of_all": pearson(v_ref, v_T)}
    best_T = max(temp_sweep, key=lambda T: temp_sweep[T]["vs_mean_of_all"])
    report["value"]["temperature_sweep"] = {str(T): d for T, d in temp_sweep.items()}
    report["value"]["best_temperature"] = best_T
    print(f"  read-back temperature (the sampling path drew at 1.1):")
    for T, d in temp_sweep.items():
        mark = "  <-- best" if T == best_T else ""
        print(f"      T={T:<4} vs one run {d['vs_single_run']:.4f}   "
              f"vs {len(V)}-run mean {d['vs_mean_of_all']:.4f}{mark}")
    print(f"  P(donate) correlation  {report['value']['pearson_donate_prob']:.4f}")
    print(f"  distinct v: {report['value']['distinct_values_gen']} sampled "
          f"vs {report['value']['distinct_values_logit']} scored (out of {len(states)})")
    print(f"  usable samples per generation call: {kept.mean():.2f}/10 "
          f"({report['value']['states_with_zero_usable_samples']} states got none)")
    if label_mass:
        print(f"  next-token mass already on the label set: mean {np.mean(label_mass):.4f}, "
              f"min {np.min(label_mass):.4f}")
    print(f"  first-token vs full-sequence probs: mean r {report['value']['mean_first_token_vs_full_sequence_pearson']:.4f}, "
          f"min {report['value']['min_first_token_vs_full_sequence_pearson']:.4f}")
    print(f"  wall {t_gen_a:.1f}s sampled -> {t_logit:.1f}s scored "
          f"({report['value']['speedup_wall']:.2f}x)\n")

    # ---- policy prior ----------------------------------------------------
    if not args.skip_prior:
        print("== policy prior ==")
        prior_gen, t_pgen = run_pass("gen", lambda s: gen_planner.sample_prior_probs(s), states, args.num_workers)
        prior_logit, t_plogit = run_pass("logit", lambda s: logit_planner.score_prior_labels(s), states, args.num_workers)
        P_gen = np.array(prior_gen)
        P_logit = np.array([s.probs for s in prior_logit])
        acts = logit_planner.dialog_acts
        top1 = float(np.mean(P_gen.argmax(1) == P_logit.argmax(1)))
        k = min(5, len(acts))
        topk_overlap = float(np.mean([
            len(set(np.argsort(-g)[:k]) & set(np.argsort(-l)[:k])) / k for g, l in zip(P_gen, P_logit)
        ]))
        report["prior"] = {
            "n_acts": len(acts),
            "pearson_flat": pearson(P_gen.ravel(), P_logit.ravel()),
            "spearman_flat": spearman(P_gen.ravel(), P_logit.ravel()),
            "mean_per_state_pearson": float(np.mean([pearson(g, l) for g, l in zip(P_gen, P_logit)])),
            "top1_agreement": top1,
            f"top{k}_overlap": topk_overlap,
            "mean_total_variation": float(np.mean(0.5 * np.abs(P_gen - P_logit).sum(1))),
            "seconds": {"gen": t_pgen, "logit": t_plogit},
            "speedup_wall": t_pgen / t_plogit if t_plogit else None,
            "mean_prob_gen": dict(zip(acts, np.round(P_gen.mean(0), 4).tolist())),
            "mean_prob_logit": dict(zip(acts, np.round(P_logit.mean(0), 4).tolist())),
        }
        print(f"\n  pearson over all state x act cells  {report['prior']['pearson_flat']:.4f}")
        print(f"  mean per-state pearson              {report['prior']['mean_per_state_pearson']:.4f}")
        print(f"  top-1 agreement {top1:.3f}   top-{k} overlap {topk_overlap:.3f}   "
              f"mean TV {report['prior']['mean_total_variation']:.3f}")
        print(f"  wall {t_pgen:.1f}s sampled -> {t_plogit:.1f}s scored "
              f"({report['prior']['speedup_wall']:.2f}x)")

        if args.prior_topk:
            # The deployed configuration does not run the histogram prior -- --llm_prior_topk
            # replaced it with one ranking call. Measured against the same histogram baseline,
            # this says whether the logit prior is a better approximation of the prior both are
            # standing in for, not just a cheaper one.
            prior_topk, t_ptopk = run_pass(f"topk k={args.prior_topk}",
                                           lambda s: gen_planner.topk_prior_probs(s, args.prior_topk),
                                           states, args.num_workers)
            P_topk = np.array(prior_topk)
            report["prior"]["topk_baseline"] = {
                "k": args.prior_topk,
                "pearson_flat_vs_histogram": pearson(P_gen.ravel(), P_topk.ravel()),
                "top1_agreement_vs_histogram": float(np.mean(P_gen.argmax(1) == P_topk.argmax(1))),
                "mean_total_variation_vs_histogram": float(np.mean(0.5 * np.abs(P_gen - P_topk).sum(1))),
                "seconds": t_ptopk,
            }
            tb = report["prior"]["topk_baseline"]
            print(f"\n  against the same histogram baseline, the deployed --llm_prior_topk "
                  f"{args.prior_topk} ranking call:")
            print(f"      pearson {tb['pearson_flat_vs_histogram']:.4f} "
                  f"(logit {report['prior']['pearson_flat']:.4f})")
            print(f"      top-1   {tb['top1_agreement_vs_histogram']:.3f} "
                  f"(logit {top1:.3f})")
            print(f"      mean TV {tb['mean_total_variation_vs_histogram']:.3f} "
                  f"(logit {report['prior']['mean_total_variation']:.3f})")
            print(f"      wall    {t_ptopk:.1f}s (logit {t_plogit:.1f}s)")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")

    target = 0.95
    best_m = max(vs_mean_of_m, key=lambda m: vs_mean_of_m[m])
    print(f"\n=== verdict (target r >= {target}) ===")
    print(f"  against a single 10-sample generation run   {r_logit:.4f}")
    print(f"  against the mean of {best_m} runs ({10 * best_m} samples)          {vs_mean_of_m[best_m]:.4f}")
    print(f"  disattenuated for baseline noise            {disattenuated:.4f}")
    print(f"  the baseline's own reliability              {r_ceiling:.4f}")
    if r_logit >= target:
        print("  PASS outright.")
    elif vs_mean_of_m[best_m] >= target and disattenuated >= target:
        print(f"  PASS once the baseline's sampling noise is accounted for. The single-run number "
              f"falls short because the baseline is itself only reliable to {r_ceiling:.4f}: a "
              f"noiseless estimator could not score better than about {np.sqrt(r_ceiling):.3f} "
              f"against one run.")
        if label_mass and np.mean(label_mass) > 0.95:
            print(f"  The prompt is not the problem: {np.mean(label_mass):.1%} of the model's own "
                  f"next-token mass already sits on the label set, so the closed-set assumption "
                  f"throws away almost nothing.")
    else:
        print("  BELOW TARGET even after correcting for baseline noise -- make the label set "
              "explicit in the prompt and re-run."
              + (f" (label-set mass is only {np.mean(label_mass):.1%}.)" if label_mass else ""))


if __name__ == "__main__":
    main()
