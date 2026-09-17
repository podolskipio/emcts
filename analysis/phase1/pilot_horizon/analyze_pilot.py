"""Pilot P1 analysis: `--search_horizon episode` on 10 dialogues vs the same 10 dialogues of D1 (legacy).

    python analysis/phase1/pilot_horizon/analyze_pilot.py \
        --pilot analysis/phase1/runs/P1/P1 --ref analysis/phase1/runs/D1/D1 --T 10 \
        --out analysis/phase1/pilot_horizon [--B 1000]

Section A is a correctness gate on the fix itself and exits non-zero if it fails. Sections B-E are
descriptive: n = 10 paired dialogues cannot establish an SR difference, so no test is run on
outcomes. The P-VAR components are recomputed with the same scripts on both sides, with
dialogue-clustered CIs.
"""
import argparse, glob, gzip, json, os, pickle, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCRIPTS = os.path.join(REPO, "analysis", "phase1", "scripts")
sys.path.insert(0, os.path.join(REPO, "src"))


def load_steps(run):
    return [json.loads(l) for p in sorted(glob.glob(f"{run}/simlog/*.ndjson.gz")) for l in gzip.open(p, "rt")
            if '"record_type": "step"' in l]


def episodes(run):
    pk = glob.glob(f"{run}/*.pkl")
    return {e["did"]: e for e in pickle.load(open(pk[0], "rb"))}


def acts(ep):
    return [h[1] for h in ep["history"] if h[0] == "Persuader"]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--T", type=int, default=10); ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--beta", type=float, default=0.7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}

    # ---- A. correctness of the fix ----
    meta = json.load(open(f"{a.pilot}/metadata.json"))
    steps = load_steps(a.pilot)
    child_len = np.array([s["turn_index"] + s["depth"] for s in steps])
    A = {"metadata_search_horizon": meta["mcts_args"].get("search_horizon"),
         "steps": int(len(steps)), "max_child_len": int(child_len.max()),
         "steps_child_len_gt_T": int((child_len > a.T).sum()),
         "steps_parent_len_ge_T": int((child_len - 1 >= a.T).sum())}
    align = subprocess.run([sys.executable, f"{SCRIPTS}/check_instrumentation.py", "-", a.pilot], capture_output=True, text=True)
    A["alignment_check"] = align.stdout.strip().splitlines()[-1]
    A["pass"] = (A["metadata_search_horizon"] == "episode" and A["steps_child_len_gt_T"] == 0
                 and A["steps_parent_len_ge_T"] == 0 and A["alignment_check"].endswith("PASS"))
    res["A_correctness"] = A
    print(json.dumps(A, indent=1))

    # ---- B. paired outcomes (descriptive) ----
    pe, re_ = episodes(a.pilot), episodes(a.ref)
    ids = sorted(pe)
    missing = [d for d in ids if d not in re_]
    rows = []
    for d in ids:
        if d in missing:
            continue
        s1, s0 = acts(pe[d]), acts(re_[d])
        div = next((i for i, (x, y) in enumerate(zip(s1, s0)) if x != y), None if len(s1) == len(s0) else min(len(s1), len(s0)))
        rows.append({"dlg_id": d, "ref_success": bool(re_[d]["success"]), "pilot_success": bool(pe[d]["success"]),
                     "ref_turns": re_[d]["num_turns"], "pilot_turns": pe[d]["num_turns"], "first_divergent_system_turn": div,
                     "ref_acts": s0, "pilot_acts": s1})
    B = pd.DataFrame(rows)
    res["B_outcomes"] = {"n_paired": int(len(B)), "missing_in_ref": missing,
                         "SR_ref": float(B.ref_success.mean()), "SR_pilot": float(B.pilot_success.mean()),
                         "mean_turns_ref": float(B.ref_turns.mean()), "mean_turns_pilot": float(B.pilot_turns.mean()),
                         "discordant_pairs_ref_only_success": int((B.ref_success & ~B.pilot_success).sum()),
                         "discordant_pairs_pilot_only_success": int((~B.ref_success & B.pilot_success).sum()),
                         "per_dialogue": rows}

    # ---- C/D/E. search structure, cost, P-VAR components: same scripts on both sides ----
    pdir, rdir = os.path.join(a.out, "pilot_steps"), os.path.join(a.out, "ref_steps")
    sh([sys.executable, f"{SCRIPTS}/build_steps.py", a.pilot, pdir])
    sh([sys.executable, f"{SCRIPTS}/build_steps.py", a.ref, rdir])
    # restrict the reference to the pilot's dialogues, then recompute tau_med on that subset
    r = pd.read_parquet(f"{rdir}/steps.parquet")
    r = r[r.dlg_id.isin(ids)].reset_index(drop=True)
    tau = float(np.median(r.parent_nu))
    r["tau_med"] = tau; r["bucket_med"] = (r.parent_nu < tau).astype(int)
    r.to_parquet(f"{rdir}/steps.parquet", index=False)
    sh([sys.executable, f"{SCRIPTS}/p_var.py", "P1", pdir, "--beta", str(a.beta), "--B", str(a.B)])
    sh([sys.executable, f"{SCRIPTS}/p_var.py", "REF", rdir, "--beta", str(a.beta), "--B", str(a.B)])
    P, R = json.load(open(f"{pdir}/p_var_P1.json")), json.load(open(f"{rdir}/p_var_REF.json"))

    p = pd.read_parquet(f"{pdir}/steps.parquet")
    def structure(df, run):
        cl = df.turn_index + df.depth
        wc = os.path.join(os.path.dirname(os.path.abspath(run.rstrip("/"))), "wall_clock.txt")  # run_diag.sh writes runs/<TAG>/wall_clock.txt
        wall = open(wc).read().strip() if os.path.exists(wc) else None
        sims = df.groupby(["tree_key", "simulation_index"])
        return {"steps": int(len(df)), "trees": int(df.tree_key.nunique()), "steps_per_tree": float(len(df) / df.tree_key.nunique()),
                "share_steps_child_len_gt_T": float((cl > a.T).mean()), "share_steps_child_len_eq_T": float((cl == a.T).mean()),
                "share_simulations_ending_at_horizon_failure": float(sims.apply(lambda g: ((g.turn_index + g.depth == a.T) & (g.v == -1.0)).any()).mean()),
                "max_depth": int(df.depth.max()), "share_visits_depth1": float((df.depth == 1).mean()),
                "median_visits_parent_node_depth2": float(df[df.depth == 2].groupby(["tree_key", "prefix_key"]).size().median()) if (df.depth == 2).any() else None,
                "fresh_share": float((~df.child_from_cache).mean()), "run_wall_clock": wall}
    res["C_structure"] = {"pilot": structure(p, a.pilot), "ref_same_dialogues": structure(r, a.ref)}

    def comp(x):
        nk, ap_ = x["nodekey"]["median"]["all"], x["affpool"]["median"]["plain"]
        cd = x["channel_dominance"]
        return {"tau_med": x["tau_med"],
                "discordance": [nk["within_prefix"]["discordance_rate"], nk["within_prefix"]["discordance_rate_ci"]],
                "delta_mean": [nk["within_prefix"]["delta_mean"], nk["within_prefix"]["delta_mean_ci"]],
                "std_effect": nk["within_prefix"]["std_effect"],
                "omega2": [nk["pooled_omega2"]["visit_weighted_omega2"], nk["pooled_omega2"]["visit_weighted_omega2_ci"]],
                "omega2_edge_demeaned": nk["pooled_omega2_edge_demeaned"]["visit_weighted_omega2"],
                "boundary_0.1": x["nodekey"]["median"]["boundary"]["fraction_within_0.1"],
                "bucket_depth_r": [x["nodekey"]["median"]["confound_depth"]["r_bucket_depth"]["value"], x["nodekey"]["median"]["confound_depth"]["r_bucket_depth"]["ci"]],
                "affpool_n_prefixes": ap_["n_prefixes_median"]["value"], "affpool_multiplier": ap_["evidence_multiplier_median"]["value"],
                "affpool_top_share": ap_["top_prefix_share_median"]["value"], "affpool_ratio": ap_["between_within_ratio_wmedian"]["value"],
                "flip_full_puct": cd["argmax_flip_fraction_full_puct"], "flip_exploit_expanded": cd["argmax_flip_fraction_exploit_expanded"],
                "flip_visited_over_unvisited": cd["flip_kinds_share_of_flips"].get("affect_picks_visited_over_unvisited"),
                "sigma_ratio_median": x["sigma"]["ratio_median"]}
    res["D_components"] = {"pilot": comp(P), "ref_same_dialogues": comp(R)}

    json.dump(res, open(os.path.join(a.out, "pilot_results.json"), "w"), indent=1, default=str)
    f = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    fc = lambda v: f"{f(v[0])} [{f(v[1][0])}, {f(v[1][1])}]" if isinstance(v, list) else f(v)
    with open(os.path.join(a.out, "pilot_tables.md"), "w") as out:
        out.write(f"## A. Correctness: {'PASS' if A['pass'] else 'FAIL'}\n\n")
        out.write("\n".join(f"- {k}: {v}" for k, v in A.items()) + "\n\n## B. Paired outcomes (descriptive, n = %d)\n\n" % len(B))
        out.write("| | D1 legacy, same dialogues | P1 episode |\n|---|---|---|\n")
        bo = res["B_outcomes"]
        out.write(f"| SR | {f(bo['SR_ref'])} | {f(bo['SR_pilot'])} |\n| mean turns | {f(bo['mean_turns_ref'])} | {f(bo['mean_turns_pilot'])} |\n")
        out.write(f"| discordant pairs (success only in legacy / only in episode) | {bo['discordant_pairs_ref_only_success']} | {bo['discordant_pairs_pilot_only_success']} |\n\n")
        out.write("| dialogue | legacy success / turns | episode success / turns | first divergent system turn |\n|---|---|---|---|\n")
        for rw in rows:
            out.write(f"| {rw['dlg_id']} | {rw['ref_success']} / {rw['ref_turns']} | {rw['pilot_success']} / {rw['pilot_turns']} | {rw['first_divergent_system_turn']} |\n")
        out.write("\n## C. Search structure\n\n| | D1 legacy, same dialogues | P1 episode |\n|---|---|---|\n")
        for k in res["C_structure"]["pilot"]:
            out.write(f"| {k} | {f(res['C_structure']['ref_same_dialogues'][k])} | {f(res['C_structure']['pilot'][k])} |\n")
        out.write("\n## D. P-VAR components (median split; dialogue-clustered 95 % CI)\n\n| | D1 legacy, same dialogues | P1 episode |\n|---|---|---|\n")
        for k in res["D_components"]["pilot"]:
            out.write(f"| {k} | {fc(res['D_components']['ref_same_dialogues'][k])} | {fc(res['D_components']['pilot'][k])} |\n")
    print(open(os.path.join(a.out, "pilot_tables.md")).read())
    return 0 if A["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
