"""Flatten a run's simlog into steps.parquet (§7) + turns.parquet.

    python analysis/phase1/scripts/build_steps.py <run_dir> <out_dir>
    e.g. python analysis/phase1/scripts/build_steps.py analysis/phase1/runs/D1/D1 analysis/phase1/D1

One row per simulation step. Adds:
  edge_key, tree_key, prefix_key       string keys (prefix within the tree)
  tau_med, bucket_med                  parent_nu < median(parent_nu over all rows of THIS run)
  parent_label, bucket_lab             argmax of parent_emotion_dist in {fear,sadness,anger,disgust}
  generating_parent_nu                 for every step, the parent_nu of the fresh step that generated
                                       this child realization (== parent_nu on fresh steps)
  root_nu                              the tree's observed root nu (from the turn record)
Siblings stay as a JSON string column (siblings_json); p_var.py parses them for §7.6.
"""
import glob, gzip, json, os, sys
import numpy as np
import pandas as pd

NEG_LABELS = {"fear", "sadness", "anger", "disgust"}


def load_simlog(run_dir):
    steps, turns = [], []
    files = sorted(glob.glob(os.path.join(run_dir, "simlog", "*.ndjson.gz")))
    if not files:
        sys.exit(f"no simlog files under {run_dir}/simlog")
    for p in files:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                (steps if r["record_type"] == "step" else turns).append(r)
    return steps, turns


def argmax_label(d):
    return max(d, key=d.get).lower() if d else ""


def main(run_dir, out_dir, max_child_len=None):
    os.makedirs(out_dir, exist_ok=True)
    steps, turns = load_simlog(run_dir)
    t = pd.DataFrame(turns)
    df = pd.DataFrame(steps)
    n_raw = len(df)
    if max_child_len is not None:
        # sensitivity for SEARCH_HORIZON_BUG.md: drop steps whose child state is longer than the episode
        # horizon (turn_index + depth > Tmax), i.e. states no real episode reaches. Applied BEFORE
        # tau_med, so the median is that of the retained rows. This removes the impossible rows; it
        # cannot remove their influence on the Q values and selections of the rows that remain.
        df = df[(df["turn_index"] + df["depth"]) <= max_child_len].reset_index(drop=True)
    df["tree_key"] = df["dlg_id"] + "|" + df["turn_index"].astype(str)
    df["prefix_key"] = df["action_prefix"].map(lambda p: "__".join(p))
    df["edge_key"] = df["tree_key"] + "|" + df["prefix_key"] + "|" + df["action"]
    df["parent_label"] = df["parent_emotion_dist"].map(argmax_label)
    df["child_label"] = df["child_emotion_dist"].map(argmax_label)
    tau = float(np.median(df["parent_nu"]))
    df["tau_med"] = tau
    df["bucket_med"] = (df["parent_nu"] < tau).astype(int)
    df["bucket_lab"] = df["parent_label"].isin(NEG_LABELS).astype(int)
    # generating parent: join cached children to the fresh step (same tree + edge) that created them
    fresh = df[~df["child_from_cache"]]
    gen = fresh.drop_duplicates(["edge_key", "child_realization_id"]).set_index(["edge_key", "child_realization_id"])["parent_nu"]
    df["generating_parent_nu"] = [gen.get((e, c), np.nan) for e, c in zip(df["edge_key"], df["child_realization_id"])]
    df["generating_bucket_med"] = np.where(df["generating_parent_nu"].isna(), np.nan, (df["generating_parent_nu"] < tau).astype(float))
    root = t[t["planned"]].set_index(["dlg_id", "turn_index"])["root_nu"]
    df["root_nu"] = [root.get((d, i), np.nan) for d, i in zip(df["dlg_id"], df["turn_index"])]
    df["siblings_json"] = df["siblings"].map(json.dumps)
    df["parent_emotion_dist_json"] = df["parent_emotion_dist"].map(json.dumps)
    df["child_emotion_dist_json"] = df["child_emotion_dist"].map(json.dumps)
    df["action_prefix_json"] = df["action_prefix"].map(json.dumps)
    df = df.drop(columns=["siblings", "parent_emotion_dist", "child_emotion_dist", "action_prefix", "record_type"])
    df.to_parquet(os.path.join(out_dir, "steps.parquet"), index=False)
    t["root_emotion_dist_json"] = t["root_emotion_dist"].map(json.dumps)
    t["user_emotion_dist_json"] = t["user_emotion_dist"].map(json.dumps)
    t["root_visits_json"] = t["root_visits"].map(json.dumps)
    t.drop(columns=["root_emotion_dist", "user_emotion_dist", "root_visits", "record_type"]).to_parquet(
        os.path.join(out_dir, "turns.parquet"), index=False)
    summary = {
        "run_dir": run_dir, "rows": len(df), "rows_before_filter": n_raw, "max_child_len": max_child_len, "dialogues": int(df["dlg_id"].nunique()),
        "trees": int(df["tree_key"].nunique()), "prefixes": int((df["tree_key"] + "|" + df["prefix_key"]).nunique()),
        "edges": int(df["edge_key"].nunique()), "tau_med": tau,
        "fresh_rows": int((~df["child_from_cache"]).sum()),
        "unresolved_generating_parent": int(df["generating_parent_nu"].isna().sum()),
        "turn_records": len(t),
    }
    json.dump(summary, open(os.path.join(out_dir, "steps_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out_dir")
    ap.add_argument("--max_child_len", type=int, default=None,
                    help="drop steps with turn_index + depth > this (the episode horizon, e.g. 10)")
    a = ap.parse_args()
    main(a.run_dir, a.out_dir, a.max_child_len)
