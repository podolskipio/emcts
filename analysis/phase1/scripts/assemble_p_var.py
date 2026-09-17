"""Assemble analysis/phase1/p_var.json (the §11 schema) from the per-run p_var_<RUN>.json files.

    python analysis/phase1/scripts/assemble_p_var.py [--runs D1 D2]

Gate components are placed beside the proposed thresholds with a met/not-met flag. The gates are
NOT applied (no overall verdict is computed) -- the human decides. D1 and D2 are reported side by
side under "by_run" and never averaged; the top-level §11 fields are filled from D1 (primary).
"""
import argparse, json, os

P1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(tag):
    p = os.path.join(P1, tag, f"p_var_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def meta(tag):
    run = os.path.join(P1, "runs", tag)
    out = {}
    for name in ("wall_clock.txt", "started_at.txt"):
        f = os.path.join(run, name)
        if os.path.exists(f):
            out[name.split(".")[0]] = open(f).read().strip()
    return out


def excl0(ci):
    return ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)


def gates(r, split="median", pop="all"):
    nk = r["nodekey"][split][pop]
    wp, om = nk["within_prefix"], nk["pooled_omega2"]
    conf = r["nodekey"][split]["confound_depth"]["r_bucket_depth"]
    bnd = r["nodekey"][split]["boundary"]["fraction_within_0.1"]
    ap = r["affpool"][split]["plain"]
    node = {
        "discordance_rate_depth_ge_2": {"value": wp["discordance_rate"], "ci": wp["discordance_rate_ci"], "threshold": ">= 0.30",
                                        "met": wp["discordance_rate"] >= 0.30},
        "within_prefix_mean_delta": {"value": wp["delta_mean"], "ci": wp["delta_mean_ci"], "std_effect": wp["std_effect"],
                                     "threshold": "CI excludes 0 and |mean|/sd(z) >= 0.2",
                                     "met": excl0(wp["delta_mean_ci"]) and abs(wp["std_effect"]) >= 0.2},
        "pooled_omega2": {"value": om["visit_weighted_omega2"], "ci": om["visit_weighted_omega2_ci"],
                          "clipped": om["visit_weighted_omega2_clipped"], "threshold": ">= 0.05 and CI lower > 0",
                          "met": om["visit_weighted_omega2"] >= 0.05 and om["visit_weighted_omega2_ci"][0] > 0},
        "boundary_fraction_0.1": {"value": bnd, "threshold": "<= 0.35", "met": bnd <= 0.35},
        "bucket_depth_r": {"value": conf["value"], "ci": conf["ci"], "threshold": "|r| <= 0.3", "met": abs(conf["value"]) <= 0.3},
        "n_discordant_edges": {"value": wp["n_discordant_edges"], "stop_if_below": 200},
    }
    aff = {
        "n_prefixes_median": {"value": ap["n_prefixes_median"]["value"], "ci": ap["n_prefixes_median"]["ci"], "threshold": ">= 4",
                              "met": ap["n_prefixes_median"]["value"] >= 4},
        "evidence_multiplier_median": {"value": ap["evidence_multiplier_median"]["value"], "ci": ap["evidence_multiplier_median"]["ci"],
                                       "threshold": ">= 3", "met": ap["evidence_multiplier_median"]["value"] >= 3},
        "top_prefix_share_median": {"value": ap["top_prefix_share_median"]["value"], "ci": ap["top_prefix_share_median"]["ci"],
                                    "threshold": "<= 0.6", "met": ap["top_prefix_share_median"]["value"] <= 0.6},
        "between_within_ratio": {"value": ap["between_within_ratio_wmedian"]["value"], "ci": ap["between_within_ratio_wmedian"]["ci"],
                                 "noise_corrected": ap["between_within_ratio_noise_corrected_wmedian"]["value"],
                                 "threshold": "<= 1.0",
                                 "met": (ap["between_within_ratio_wmedian"]["value"] is not None and ap["between_within_ratio_wmedian"]["value"] <= 1.0)},
    }
    return node, aff


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", default=["D1", "D2"]); a = ap.parse_args()
    runs = {t: load(t) for t in a.runs}
    runs = {t: r for t, r in runs.items() if r}
    assert "D1" in runs, "D1 results missing"
    ids = [l.split("\t")[1].strip() for l in open(os.path.join(P1, "runs", "dialogue_ids.txt")) if not l.startswith("#")]
    gpu = json.load(open(os.path.join(P1, "runs", "gpu_hours.json"))) if os.path.exists(os.path.join(P1, "runs", "gpu_hours.json")) else {}
    by_run = {}
    for t, r in runs.items():
        node, aff = gates(r)
        node_f, _ = gates(r, pop="fresh")
        node_l, aff_l = gates(r, split="label")
        by_run[t] = {"gate_components": {"nodekey": node, "affpool": aff},
                     "gate_components_fresh_steps_only": {"nodekey": node_f},
                     "gate_components_label_split": {"nodekey": node_l, "affpool": aff_l},
                     "meta": meta(t)}
    d1 = runs["D1"]
    nk = d1["nodekey"]["median"]["all"]
    strat = nk["within_depth_median_split"]
    out = {
        "data_source": {"runs": list(runs), "n_dialogues": d1["dialogues"], "dialogue_ids": ids,
                        "dialogue_positions": "101-130 of the non-annotated p4g pool (eval set = 1-100)",
                        "beta_emo": [0.7, 0.0][:len(runs)], "gpu_hours": gpu.get("total_gpu_hours", None), "gpu_hours_detail": gpu,
                        "from_eval_set": False, "classifier": "hf",
                        "config": "vicuna-13B-v1.5-AWQ via SGLang (all roles one server); logit_scoring off (sampled value + top-K prior); "
                                  "R=max_realizations=4; n_sims 50; K 5; Tmax=max_turns=10; persona; cpuct 1.0; Q_0 0.0; "
                                  "emo_signal level; soft table; 10 workers; seed 0"},
        "depth_profile": {"visits_by_depth": {k: v["visits"] for k, v in d1["depth_profile"].items()},
                          "edges_by_depth": {k: v["edges"] for k, v in d1["depth_profile"].items()},
                          "edges_ge_depth2": nk["within_prefix"]["edges_depth_ge_2"]},
        "tau_med": d1["tau_med"],
        "nodekey": {"restricted_to_depth_ge_2": True,
                    "discordance_rate": nk["within_prefix"]["discordance_rate"],
                    "within_prefix_delta": {"mean": nk["within_prefix"]["delta_mean"], "ci": nk["within_prefix"]["delta_mean_ci"],
                                            "std_effect": nk["within_prefix"]["std_effect"]},
                    "pooled_omega2": {"value": nk["pooled_omega2"]["visit_weighted_omega2"], "ci": nk["pooled_omega2"]["visit_weighted_omega2_ci"]},
                    "boundary_fraction_0.1": d1["nodekey"]["median"]["boundary"]["fraction_within_0.1"],
                    "bucket_depth_r": d1["nodekey"]["median"]["confound_depth"]["r_bucket_depth"],
                    "within_depth_stratified": {"delta_mean": strat["within_prefix"]["delta_mean"],
                                                "omega2": strat["pooled_omega2"]["visit_weighted_omega2"],
                                                "definition": "bucket = parent_nu < median(parent_nu | depth)"},
                    "pooled_omega2_edge_demeaned": {"value": nk["pooled_omega2_edge_demeaned"]["visit_weighted_omega2"],
                                                    "ci": nk["pooled_omega2_edge_demeaned"]["visit_weighted_omega2_ci"],
                                                    "note": "z minus its own edge mean: the part of pooled omega^2 not attributable to prefix identity"},
                    "post_split_visits_depth_ge_2": nk["post_split_visits"],
                    "fresh_steps_only": {"discordance_rate": d1["nodekey"]["median"]["fresh"]["within_prefix"]["discordance_rate"],
                                         "delta_mean": d1["nodekey"]["median"]["fresh"]["within_prefix"]["delta_mean"],
                                         "delta_ci": d1["nodekey"]["median"]["fresh"]["within_prefix"]["delta_mean_ci"],
                                         "omega2": d1["nodekey"]["median"]["fresh"]["pooled_omega2"]["visit_weighted_omega2"]}},
        "affpool": {"sharing_measured": "within_tree",
                    "n_prefixes_median": d1["affpool"]["median"]["plain"]["n_prefixes_median"]["value"],
                    "evidence_multiplier_median": d1["affpool"]["median"]["plain"]["evidence_multiplier_median"]["value"],
                    "top_prefix_share_median": d1["affpool"]["median"]["plain"]["top_prefix_share_median"]["value"],
                    "between_within_ratio": d1["affpool"]["median"]["plain"]["between_within_ratio_wmedian"]["value"],
                    "depth_keyed": {"evidence_multiplier_median": d1["affpool"]["median"]["depth_keyed"]["evidence_multiplier_median"]["value"],
                                    "between_within_ratio": d1["affpool"]["median"]["depth_keyed"]["between_within_ratio_wmedian"]["value"]}},
        "splits": {"median": {"occupancy": d1["splits"]["median"]["occupancy"]["overall"]},
                   "label": {"occupancy": d1["splits"]["label"]["occupancy"]["overall"]},
                   "kappa_between_splits": d1["splits"]["agreement"]["kappa"]},
        "channel_dominance": {"argmax_flip_fraction": d1["channel_dominance"]["argmax_flip_fraction_full_puct"],
                              "argmax_flip_fraction_exploit_expanded": d1["channel_dominance"]["argmax_flip_fraction_exploit_expanded"],
                              "median_ratio": d1["channel_dominance"]["median_ratio"],
                              "flip_kinds_share_of_flips": d1["channel_dominance"]["flip_kinds_share_of_flips"]},
        "sigma": {"ratio_median": d1["sigma"]["ratio_median"], "ratio_p90": d1["sigma"]["ratio_p90"], "median_N": d1["sigma"]["median_N"],
                  "note": "supersedes the 1.43 / 9.10 stub-classifier figures, which must not be cited"},
        "gate_components": by_run["D1"]["gate_components"],
        "by_run": by_run,
        "gates_applied": False,
    }
    # search-horizon sensitivity (SEARCH_HORIZON_BUG.md §4): same components with steps past Tmax removed
    hs = {}
    for t in runs:
        f = os.path.join(P1, "horizon_sensitivity", t, f"p_var_{t}.json")
        if os.path.exists(f):
            node_h, aff_h = gates(json.load(open(f)))
            hs[t] = {"gate_components": {"nodekey": node_h, "affpool": aff_h}}
    out["search_horizon_bug"] = {
        "doc": "analysis/phase1/SEARCH_HORIZON_BUG.md",
        "share_steps_child_past_Tmax": {"D1": 0.197, "D2": 0.127},
        "share_trees_with_impossible_state": {"D1": 132 / 175, "D2": 89 / 177},
        "note": "all gate components above were computed under the legacy search rule; the block below drops step rows whose "
                "child is past Tmax (a lower bound on the bug's effect). Pilot under --search_horizon episode: Thursday 2026-09-17.",
        "sensitivity_child_len_le_Tmax": hs,
    }
    json.dump(out, open(os.path.join(P1, "p_var.json"), "w"), indent=1)
    print("wrote", os.path.join(P1, "p_var.json"))
    for t, b in by_run.items():
        print(f"== {t}")
        for g, comps in b["gate_components"].items():
            for k, v in comps.items():
                print(f"  {g:8s} {k:32s} {v.get('value')!s:>24}  ci={v.get('ci')}  thr={v.get('threshold', '')}  met={v.get('met', '')}")


if __name__ == "__main__":
    main()
