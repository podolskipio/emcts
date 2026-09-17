"""P-VAR (§7-§9): NodeKey and AffPool gate components from one run's steps.parquet.

    python analysis/phase1/scripts/p_var.py <run_tag> <steps_dir> [--beta 0.7] [--figdir DIR] [--B 1000]

Rules (§6), applied everywhere: every CI resamples dialogue_id with replacement (1000 replicates,
percentile 95%); omega^2, never eta^2. Output: <steps_dir>/p_var_<tag>.json.

Every NodeKey statistic is computed on two step populations (see definitions.md):
  all   -- every simulation step (what Q_emo averages; what a NodeKey split would see under the
           current prefix-keyed realization cache)
  fresh -- child_from_cache == False (the child was generated from THIS parent realization;
           the causal parent->child premise)
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import pandas as pd

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
DEPTH_BINS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 99, "6+")]


def depth_bin(d):
    for lo, hi, name in DEPTH_BINS:
        if lo <= d <= hi:
            return name


def pct(a, q):
    a = np.asarray([x for x in a if np.isfinite(x)])
    return float(np.percentile(a, q)) if len(a) else float("nan")


def ci(reps):
    return [pct(reps, 2.5), pct(reps, 97.5)]


class Boot:
    """Cluster bootstrap over dialogues. stat(ids) gets a list of dialogue ids WITH multiplicity."""

    def __init__(self, dialogues, B, seed=20260914):
        self.d = list(dialogues)
        self.B = B
        self.rng = np.random.default_rng(seed)
        self.draws = [self.rng.integers(0, len(self.d), len(self.d)) for _ in range(B)]

    def run(self, stat):
        return [stat([self.d[i] for i in idx]) for idx in self.draws]


# ----------------------------------------------------------------------------------------------
# §7.1 visit profile
# ----------------------------------------------------------------------------------------------
def visit_profile(df):
    g = df.groupby("depth")
    edges = df.groupby("depth")["edge_key"].nunique()
    out = {}
    for d, sub in g:
        per_edge = sub.groupby("edge_key").size()
        out[int(d)] = {"visits": int(len(sub)), "edges": int(edges[d]), "mean_visits_per_edge": float(per_edge.mean()),
                       "median_visits_per_edge": float(per_edge.median()),
                       "fresh_visits": int((~sub["child_from_cache"]).sum())}
    # per-node (parent) visits: visits into a node at depth d = steps whose parent sits at depth d-1
    node_visits = df.groupby(["depth", "tree_key", "prefix_key"]).size().groupby("depth").median()
    for d in out:
        out[d]["median_visits_per_parent_node"] = float(node_visits.get(d, np.nan))
    return out


# ----------------------------------------------------------------------------------------------
# §7.2 within-prefix bucket contrast
# ----------------------------------------------------------------------------------------------
def edge_contrasts(df, bucket):
    """Per edge at depth>=2: n per bucket, mean z per bucket, discordant flag, delta = mean(b=1) - mean(b=0)."""
    sub = df[df["depth"] >= 2]
    g = sub.groupby(["dlg_id", "depth", "edge_key", bucket])["z"].agg(["size", "mean"]).unstack(bucket)
    n0 = g[("size", 0)].fillna(0) if ("size", 0) in g else 0
    n1 = g[("size", 1)].fillna(0) if ("size", 1) in g else 0
    e = pd.DataFrame({"n0": n0, "n1": n1}).reset_index()
    e["m0"] = g[("mean", 0)].values if ("mean", 0) in g else np.nan
    e["m1"] = g[("mean", 1)].values if ("mean", 1) in g else np.nan
    e["N"] = e["n0"] + e["n1"]
    e["discordant"] = (e["n0"] > 0) & (e["n1"] > 0)
    e["delta"] = np.where(e["discordant"], e["m1"] - e["m0"], np.nan)
    return e


def nodekey_contrast(df, bucket, boot):
    e = edge_contrasts(df, bucket)
    sub = df[df["depth"] >= 2]
    zstats = sub.groupby("dlg_id")["z"].agg(n="size", s="sum", s2=lambda x: float((x ** 2).sum()))
    by_d = {d: g for d, g in e.groupby("dlg_id")}
    empty = e.iloc[0:0]

    def sd_z(ids):
        z = zstats.reindex(ids).fillna(0)
        n, s, s2 = z["n"].sum(), z["s"].sum(), z["s2"].sum()
        return float(np.sqrt((s2 - s * s / n) / (n - 1))) if n > 1 else np.nan

    def stats(ids):
        ee = pd.concat([by_d.get(d, empty) for d in ids])
        dd = ee["delta"].dropna()
        m = float(dd.mean()) if len(dd) else np.nan
        return m, (m / sd_z(ids) if len(dd) else np.nan), float(ee["discordant"].mean()) if len(ee) else np.nan

    reps = boot.run(stats)
    disc = e["discordant"]
    delta = e["delta"].dropna()
    sdz = float(sub["z"].std(ddof=1))
    by_depth = {}
    for d, g in e.groupby("depth"):
        dd = g["delta"].dropna()
        by_depth[int(d)] = {"edges": int(len(g)), "edges_N_ge_2": int((g["N"] >= 2).sum()),
                            "discordance_rate": float(g["discordant"].mean()),
                            "discordance_rate_N_ge_2": float(g.loc[g["N"] >= 2, "discordant"].mean()) if (g["N"] >= 2).any() else np.nan,
                            "n_discordant": int(g["discordant"].sum()),
                            "delta_mean": float(dd.mean()) if len(dd) else np.nan}
    return {
        "edges_depth_ge_2": int(len(e)), "edges_N_ge_2": int((e["N"] >= 2).sum()),
        "n_discordant_edges": int(disc.sum()),
        "discordance_rate": float(disc.mean()), "discordance_rate_ci": ci([r[2] for r in reps]),
        "discordance_rate_among_N_ge_2": float(e.loc[e["N"] >= 2, "discordant"].mean()) if (e["N"] >= 2).any() else np.nan,
        "delta_mean": float(delta.mean()) if len(delta) else np.nan,
        "delta_median": float(delta.median()) if len(delta) else np.nan,
        "delta_mean_ci": ci([r[0] for r in reps]),
        "sd_z_depth_ge_2": sdz,
        "std_effect": float(delta.mean() / sdz) if len(delta) else np.nan,
        "std_effect_ci": ci([r[1] for r in reps]),
        "by_depth": by_depth,
    }


# ----------------------------------------------------------------------------------------------
# §7.3 pooled decomposition, omega^2 from sufficient statistics
# ----------------------------------------------------------------------------------------------
def omega2_from_suff(n, s, s2):
    """n,s,s2: arrays over the k bucket levels of one group. Returns omega^2 (unclipped) or nan."""
    keep = n > 0
    n, s, s2 = n[keep], s[keep], s2[keep]
    k = len(n)
    N = n.sum()
    if k < 2 or N - k < 1:
        return np.nan, int(N)
    grand = s.sum() / N
    sst = s2.sum() - N * grand ** 2
    ssb = float(((s / n) ** 2 * n).sum() - N * grand ** 2)
    ssw = sst - ssb
    msw = ssw / (N - k)
    denom = sst + msw
    if denom <= 0:
        return np.nan, int(N)
    return float((ssb - (k - 1) * msw) / denom), int(N)


def pooled_omega2(df, bucket, keys, boot, min_depth=2):
    sub = df[df["depth"] >= min_depth]
    suff = sub.assign(z2=sub["z"] ** 2).groupby(["dlg_id", *keys, bucket]).agg(n=("z", "size"), s=("z", "sum"), s2=("z2", "sum"))
    by_d = {d: g.droplevel(0) for d, g in suff.groupby(level=0)}

    def compute(ids):
        agg = pd.concat([by_d[d] for d in ids if d in by_d]).groupby(level=list(range(len(keys) + 1))).sum()
        vals, ws, rows = [], [], []
        for key, g in agg.groupby(level=list(range(len(keys)))):
            w, N = omega2_from_suff(g["n"].values.astype(float), g["s"].values, g["s2"].values)
            if np.isfinite(w):
                vals.append(w); ws.append(N); rows.append((key, w, N))
        if not vals:
            return np.nan, np.nan, rows
        vals, ws = np.array(vals), np.array(ws, float)
        return float((vals * ws).sum() / ws.sum()), float((np.clip(vals, 0, None) * ws).sum() / ws.sum()), rows

    w_un, w_cl, rows = compute(list(by_d))
    reps = boot.run(lambda ids: compute(ids)[:2])
    per_group = [{"group": list(k) if isinstance(k, tuple) else [k], "omega2": w, "n": N} for k, w, N in rows]
    arr = np.array([r["omega2"] for r in per_group]) if per_group else np.array([np.nan])
    depth_idx = keys.index("depth") if "depth" in keys else None
    by_depth = {}
    if depth_idx is not None:
        tmp = defaultdict(list)
        for r in per_group:
            tmp[depth_bin(int(r["group"][depth_idx]))].append(r)
        for b, rs in tmp.items():
            w = np.array([r["omega2"] for r in rs]); n = np.array([r["n"] for r in rs], float)
            by_depth[b] = {"groups": len(rs), "visit_weighted_omega2": float((w * n).sum() / n.sum()),
                           "visit_weighted_omega2_clipped": float((np.clip(w, 0, None) * n).sum() / n.sum())}
    return {
        "keys": keys, "n_groups": len(per_group),
        "visit_weighted_omega2": w_un, "visit_weighted_omega2_ci": ci([r[0] for r in reps]),
        "visit_weighted_omega2_clipped": w_cl, "visit_weighted_omega2_clipped_ci": ci([r[1] for r in reps]),
        "group_distribution": {"p10": pct(arr, 10), "p25": pct(arr, 25), "median": pct(arr, 50), "p75": pct(arr, 75), "p90": pct(arr, 90),
                               "share_positive": float(np.mean(arr > 0)) if len(per_group) else np.nan},
        "by_depth_bin": by_depth,
    }


# ----------------------------------------------------------------------------------------------
# §7.5 confound: bucket vs depth / turn
# ----------------------------------------------------------------------------------------------
def point_biserial(b, x):
    b, x = np.asarray(b, float), np.asarray(x, float)
    if b.std() == 0 or x.std() == 0:
        return np.nan
    return float(np.corrcoef(b, x)[0, 1])


def confound(df, bucket, boot):
    out = {"parent_nu_by_depth": {}}
    for d, g in df.groupby("depth"):
        q = g["parent_nu"].quantile([0.25, 0.5, 0.75])
        out["parent_nu_by_depth"][int(d)] = {"n": int(len(g)), "mean": float(g["parent_nu"].mean()), "q25": float(q[0.25]),
                                             "median": float(q[0.5]), "q75": float(q[0.75]),
                                             "bucket1_occupancy": float(g[bucket].mean())}
    by_d = {d: g for d, g in df.groupby("dlg_id")}
    for name, col in (("depth", "depth"), ("turn_index", "turn_index"), ("absolute_position", "abs_pos")):
        r = point_biserial(df[bucket], df[col])
        reps = boot.run(lambda ids: point_biserial(pd.concat([by_d[i][bucket] for i in ids]), pd.concat([by_d[i][col] for i in ids])))
        out[f"r_bucket_{name}"] = {"value": r, "ci": ci(reps)}
        sub = df[df["depth"] >= 2]
        out[f"r_bucket_{name}_depth_ge_2"] = point_biserial(sub[bucket], sub[col])
    return out


# ----------------------------------------------------------------------------------------------
# §7.6 channel dominance
# ----------------------------------------------------------------------------------------------
def first_argmax(vals):
    best, bi = -np.inf, -1
    for i, v in enumerate(vals):
        if v > best:
            best, bi = v, i
    return bi


def channel_dominance(df, beta_run, beta_cf, boot):
    """beta_run: the beta the run used (its logged uct contains beta_run*Q_emo).
    beta_cf: the beta whose influence is measured (== beta_run for D1; 0.7 counterfactually for D2)."""
    rows = []
    for dlg, depth, sj in zip(df["dlg_id"], df["depth"], df["siblings_json"]):
        sib = json.loads(sj)
        uct = np.array([s["uct"] for s in sib])
        qemo = np.array([s["Q_emo"] for s in sib])
        base = uct - beta_run * qemo                  # Q + cpuct*P*sqrt(Ns)/(1+N): the score without the affect term
        full = base + beta_cf * qemo                  # the score with beta_cf
        flip_full = first_argmax(full) != first_argmax(base)
        exp = [i for i, s in enumerate(sib) if s["N"] > 0]
        rec = {"dlg_id": dlg, "depth": int(depth), "flip_full_puct": bool(flip_full), "n_expanded": len(exp)}
        if flip_full:
            ca, cb = sib[first_argmax(full)], sib[first_argmax(base)]
            rec["flip_kind"] = ("affect_picks_visited_over_unvisited" if ca["N"] > 0 and cb["N"] == 0 else
                                "affect_picks_unvisited_over_visited" if ca["N"] == 0 and cb["N"] > 0 else
                                "both_visited" if ca["N"] > 0 else "both_unvisited")
        if len(exp) >= 2:
            order = sorted(exp, key=lambda i: -full[i])[:2]
            a, b = order
            dq = abs(sib[a]["Q"] - sib[b]["Q"])
            dqe = abs(beta_cf * (sib[a]["Q_emo"] - sib[b]["Q_emo"]))
            q = [sib[i]["Q"] for i in exp]
            qe = [sib[i]["Q"] + beta_cf * sib[i]["Q_emo"] for i in exp]
            rec.update({"abs_dQ": dq, "abs_beta_dQemo": dqe, "ratio": (dqe / dq) if dq > 0 else np.inf,
                        "flip_exploit_expanded": first_argmax(qe) != first_argmax(q)})
        rows.append(rec)
    r = pd.DataFrame(rows)
    two = r[r["n_expanded"] >= 2]
    by_d = {d: g for d, g in r.groupby("dlg_id")}

    def stats(ids):
        g = pd.concat([by_d[i] for i in ids])
        t = g[g["n_expanded"] >= 2]
        return float(g["flip_full_puct"].mean()), float(t["flip_exploit_expanded"].mean()), float(np.median(t["ratio"]))

    reps = boot.run(stats)
    by_depth = {}
    for d, g in r.groupby("depth"):
        t = g[g["n_expanded"] >= 2]
        by_depth[int(d)] = {"n": int(len(g)), "n_ge2_expanded": int(len(t)), "flip_full_puct": float(g["flip_full_puct"].mean()),
                            "flip_exploit_expanded": float(t["flip_exploit_expanded"].mean()) if len(t) else np.nan,
                            "median_abs_dQ": float(t["abs_dQ"].median()) if len(t) else np.nan,
                            "median_abs_beta_dQemo": float(t["abs_beta_dQemo"].median()) if len(t) else np.nan,
                            "median_ratio": float(t["ratio"].median()) if len(t) else np.nan}
    fin = two["ratio"].replace(np.inf, np.nan)
    return {
        "beta_run": beta_run, "beta_measured": beta_cf, "counterfactual": beta_run != beta_cf,
        "selection_points": int(len(r)), "points_with_ge2_expanded": int(len(two)),
        "argmax_flip_fraction_full_puct": float(r["flip_full_puct"].mean()), "argmax_flip_fraction_full_puct_ci": ci([x[0] for x in reps]),
        "argmax_flip_fraction_exploit_expanded": float(two["flip_exploit_expanded"].mean()), "argmax_flip_fraction_exploit_expanded_ci": ci([x[1] for x in reps]),
        "abs_dQ": {"median": float(two["abs_dQ"].median()), "p25": pct(two["abs_dQ"], 25), "p75": pct(two["abs_dQ"], 75), "p90": pct(two["abs_dQ"], 90)},
        "abs_beta_dQemo": {"median": float(two["abs_beta_dQemo"].median()), "p25": pct(two["abs_beta_dQemo"], 25), "p75": pct(two["abs_beta_dQemo"], 75), "p90": pct(two["abs_beta_dQemo"], 90)},
        "median_ratio": float(two["ratio"].median()), "median_ratio_ci": ci([x[2] for x in reps]),
        "share_ratio_gt_1": float((two["ratio"] > 1).mean()), "share_dQ_zero": float((two["abs_dQ"] == 0).mean()),
        "flip_kinds_share_of_flips": r.loc[r["flip_full_puct"], "flip_kind"].value_counts(normalize=True).to_dict() if r["flip_full_puct"].any() else {},
        "mean_Q_emo_of_visited_siblings": float(np.mean([s_["Q_emo"] for sj in df["siblings_json"] for s_ in json.loads(sj) if s_["N"] > 0])),
        "ratio_finite_p90": pct(fin, 90),
        "by_depth": by_depth,
    }, two


# ----------------------------------------------------------------------------------------------
# §8 AffPool (per tree)
# ----------------------------------------------------------------------------------------------
def affpool_cells(df, bucket, depth_keyed):
    edge_N = df.groupby("edge_key").size()
    d = df.assign(edge_N=df["edge_key"].map(edge_N))
    if depth_keyed:
        d = d.assign(dbin=d["depth"].map(depth_bin))
        cell_cols = ["dlg_id", "tree_key", bucket, "action", "dbin"]
    else:
        cell_cols = ["dlg_id", "tree_key", bucket, "action"]
    cells = []
    for key, g in d.groupby(cell_cols):
        per_prefix = g.groupby("prefix_key")
        sizes = per_prefix.size()
        n_pref = len(sizes)
        N_pool = int(len(g))
        med_node = float(g.drop_duplicates("edge_key")["edge_N"].median())
        med_incell = float(g.groupby("edge_key").size().median())
        shares = sizes / N_pool
        rec = {"dlg_id": key[0], "tree_key": key[1], "bucket": int(key[2]), "action": key[3],
               "dbin": key[4] if depth_keyed else None,
               "n_prefixes": n_pref, "N_pool": N_pool, "median_N_node": med_node,
               "evidence_multiplier": N_pool / med_node, "evidence_multiplier_incell": N_pool / med_incell,
               "top_prefix_share": float(shares.max()),
               "herfindahl": float((shares ** 2).sum()), "min_depth": int(g["depth"].min())}
        if n_pref >= 3:
            means = per_prefix["z"].mean()
            vars_ = per_prefix["z"].var(ddof=1)
            ns = sizes
            dfw = (ns - 1).clip(lower=0)
            within = float((vars_.fillna(0) * dfw).sum() / dfw.sum()) if dfw.sum() > 0 else np.nan
            between = float(means.var(ddof=1))
            rec["between_var"] = between
            rec["within_var"] = within
            rec["ratio"] = between / within if within and within > 0 else (np.inf if between > 0 else np.nan)
            # noise-corrected variance component: var(means) includes within/n_i sampling noise
            rec["ratio_noise_corrected"] = ((between - float((within / ns).mean())) / within) if within and within > 0 else np.nan
        cells.append(rec)
    return pd.DataFrame(cells)


def wmedian(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x)
    x, w = x[m], w[m]
    if not len(x):
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def affpool(df, bucket, boot, depth_keyed):
    cells = affpool_cells(df, bucket, depth_keyed)
    by_d = {d: g for d, g in cells.groupby("dlg_id")}

    def summ(c):
        h = c[c["n_prefixes"] >= 3]
        return {"n_prefixes_median": float(c["n_prefixes"].median()), "evidence_multiplier_median": float(c["evidence_multiplier"].median()),
                "top_prefix_share_median": float(c["top_prefix_share"].median()), "herfindahl_median": float(c["herfindahl"].median()),
                "between_within_ratio_wmedian": wmedian(h["ratio"], h["N_pool"]) if len(h) else np.nan,
                "between_within_ratio_noise_corrected_wmedian": wmedian(h["ratio_noise_corrected"], h["N_pool"]) if len(h) else np.nan}

    reps = boot.run(lambda ids: summ(pd.concat([by_d[i] for i in ids if i in by_d])))
    out = summ(cells)
    out = {k: {"value": v, "ci": ci([r[k] for r in reps])} for k, v in out.items()}
    dist = lambda s: {"p10": pct(s, 10), "p25": pct(s, 25), "median": pct(s, 50), "p75": pct(s, 75), "p90": pct(s, 90)}
    out["n_cells"] = int(len(cells))
    out["n_trees"] = int(cells["tree_key"].nunique())
    out["cells_with_ge3_prefixes"] = int((cells["n_prefixes"] >= 3).sum())
    out["share_cells_single_prefix"] = float((cells["n_prefixes"] == 1).mean())
    out["distributions"] = {c: dist(cells[c]) for c in ("n_prefixes", "N_pool", "evidence_multiplier", "evidence_multiplier_incell", "top_prefix_share", "herfindahl")}
    out["cells_excluding_depth1_only"] = {"n_cells": int((cells["min_depth"] >= 2).sum()),
                                          "n_prefixes_median": float(cells.loc[cells["min_depth"] >= 2, "n_prefixes"].median()),
                                          "evidence_multiplier_median": float(cells.loc[cells["min_depth"] >= 2, "evidence_multiplier"].median())}
    # within-tree distribution: per tree medians, then their spread across trees
    per_tree = cells.groupby("tree_key").agg(n_prefixes=("n_prefixes", "median"), evidence_multiplier=("evidence_multiplier", "median"))
    out["per_tree_median_distribution"] = {c: dist(per_tree[c]) for c in per_tree.columns}
    out["by_bucket"] = {int(b): {"n_cells": int(len(g)), "n_prefixes_median": float(g["n_prefixes"].median()),
                                 "evidence_multiplier_median": float(g["evidence_multiplier"].median())} for b, g in cells.groupby("bucket")}
    if depth_keyed:
        out["by_depth_bin"] = {}
        for b, g in cells.groupby("dbin"):
            h = g[g["n_prefixes"] >= 3]
            out["by_depth_bin"][b] = {"n_cells": int(len(g)), "n_prefixes_median": float(g["n_prefixes"].median()),
                                      "evidence_multiplier_median": float(g["evidence_multiplier"].median()),
                                      "top_prefix_share_median": float(g["top_prefix_share"].median()),
                                      "cells_ge3": int(len(h)),
                                      "between_within_ratio_wmedian": wmedian(h["ratio"], h["N_pool"]) if len(h) else np.nan}
    return out, cells


# ----------------------------------------------------------------------------------------------
# §9
# ----------------------------------------------------------------------------------------------
def split_agreement(df):
    a, b = df["bucket_med"].values, df["bucket_lab"].values
    m = np.array([[np.sum((a == i) & (b == j)) for j in (0, 1)] for i in (0, 1)])
    n = m.sum(); po = np.trace(m) / n; pe = (m.sum(1) @ m.sum(0)) / n ** 2
    return {"kappa": float((po - pe) / (1 - pe)) if pe < 1 else np.nan,
            "table_rows_med_cols_lab": m.tolist(), "row_col_labels": ["0 (not neg)", "1 (neg)"]}


def occupancy(df, bucket):
    return {"overall": [float(1 - df[bucket].mean()), float(df[bucket].mean())],
            "depth_ge_2": [float(1 - df.loc[df.depth >= 2, bucket].mean()), float(df.loc[df.depth >= 2, bucket].mean())],
            "by_depth": {int(d): float(g[bucket].mean()) for d, g in df.groupby("depth")}}


def sigma_emo(df):
    """Per edge with N >= 2 at the end of its tree, from the z tape (identical to Welford M2/(N-1))."""
    g = df.groupby("edge_key")["z"].agg(N="size", q="mean", var=lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan)
    g = g[g["N"] >= 2].copy()
    g["sigma"] = np.sqrt(g["var"].clip(lower=0))
    g["absq"] = g["q"].abs()
    tiny = g["absq"] < 1e-9
    g["ratio"] = np.where(tiny, np.nan, g["sigma"] / g["absq"])
    r = g["ratio"].dropna()
    byN = {}
    for lo, hi, name in ((2, 2, "N=2"), (3, 3, "N=3"), (4, 5, "N=4-5"), (6, 10, "N=6-10"), (11, 10 ** 9, "N>10")):
        s = g[(g["N"] >= lo) & (g["N"] <= hi)]
        byN[name] = {"edges": int(len(s)), "sigma_median": float(s["sigma"].median()) if len(s) else np.nan,
                     "ratio_median": float(s["ratio"].median()) if len(s) else np.nan}
    return {"edges_N_ge_2": int(len(g)), "excluded_abs_Qemo_zero": int(tiny.sum()),
            "ratio_median": float(r.median()), "ratio_p25": pct(r, 25), "ratio_p75": pct(r, 75), "ratio_p90": pct(r, 90),
            "sigma_median": float(g["sigma"].median()), "abs_Qemo_median": float(g["absq"].median()),
            "median_N": float(g["N"].median()), "share_N_eq_2": float((g["N"] == 2).mean()), "share_N_le_3": float((g["N"] <= 3).mean()),
            "N_distribution": {"p25": pct(g["N"], 25), "median": pct(g["N"], 50), "p75": pct(g["N"], 75), "p90": pct(g["N"], 90)},
            "by_N": byN}


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(tag, df, dom_two, cells, figdir, beta_cf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    # depths past 15 are a handful of pathological chains (definitions.md: the depth cap does not bound search);
    # they are pooled into one "16+" bin so the axis stays readable
    df = df.assign(depth_plot=df["depth"].clip(upper=16))
    depths = sorted(df["depth_plot"].unique())
    dlabel = lambda d: "16+" if d == 16 else str(d)
    # nu by depth
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [df.loc[df.depth_plot == d, "parent_nu"].values for d in depths]
    bp = ax.boxplot(data, positions=range(len(depths)), widths=0.55, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE, lw=1.2)
    for k in ("whiskers", "caps"):
        for x in bp[k]:
            x.set(color=INK2, lw=1)
    for m in bp["medians"]:
        m.set(color=INK, lw=2)
    tau = df["tau_med"].iloc[0]
    ax.axhline(tau, color=ORANGE, lw=2, ls="--")
    ax.text(-0.45, tau, f"τ_med = {tau:+.3f} ", color=INK2, va="bottom", ha="left", fontsize=8)
    ax.set_xticks(range(len(depths)), [dlabel(d) for d in depths], fontsize=7)
    top = ax.get_ylim()[1]
    for i, x in enumerate(data):  # row counts above the plot, rotated so many depths do not collide
        ax.text(i, top, f"n={len(x)}", rotation=90, fontsize=6, color=INK2, ha="center", va="bottom")
    ax.set_xlabel("edge depth (1 = root's outgoing edges; parent is the observed root)", color=INK2, fontsize=8)
    ax.set_ylabel("parent ν", color=INK2, fontsize=9)
    ax.set_title(f"{tag}: parent-realization valence by depth", color=INK, fontsize=10, loc="left", pad=28)
    style(ax); fig.tight_layout(); fig.savefig(f"{figdir}/nu_by_depth_{tag}.png", dpi=150); plt.close(fig)
    # occupancy by depth, both splits
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, col, name in zip(axes, ("bucket_med", "bucket_lab"), ("median split (ν < τ_med)", "label split (neg argmax)")):
        occ = [df.loc[df.depth_plot == d, col].mean() for d in depths]
        ax.bar(range(len(depths)), occ, color=ORANGE, width=0.7, label="bucket 1 (negative side)")
        ax.bar(range(len(depths)), [1 - o for o in occ], bottom=occ, color=BLUE, width=0.7, label="bucket 0",
               edgecolor="white", linewidth=2)
        ax.set_xticks(range(len(depths)), [dlabel(d) for d in depths], fontsize=7)
        ax.set_title(name, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("edge depth", color=INK2, fontsize=8)
        style(ax)
    axes[0].set_ylabel("share of step rows", color=INK2, fontsize=9)
    fig.suptitle(f"{tag}: bucket occupancy by depth", color=INK, fontsize=10, x=0.02, ha="left")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{figdir}/bucket_occupancy_by_depth_{tag}.png", dpi=150); plt.close(fig)
    # dQ vs beta dQemo
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    x, y = dom_two["abs_dQ"].values, dom_two["abs_beta_dQemo"].values
    lim = max(np.nanmax(x) if len(x) else 1, np.nanmax(y) if len(y) else 1) * 1.05
    hb = ax.hexbin(x, y, gridsize=45, extent=(0, lim, 0, lim), bins="log", mincnt=1, cmap="Blues", linewidths=0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("selection points (log)", color=INK2, fontsize=8); cb.ax.tick_params(labelsize=7, colors=INK2)
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    ax.text(lim * 0.97, lim * 0.97, "equal", color=INK2, fontsize=8, ha="right", va="top")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("|ΔQ| between top-2 siblings", color=INK2, fontsize=8)
    ax.set_ylabel(f"|β·ΔQ_emo|  (β = {beta_cf})", color=INK2, fontsize=8)
    ax.set_title(f"{tag}: task vs affect term at selection", color=INK, fontsize=10, loc="left")
    style(ax); ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(f"{figdir}/delta_q_vs_beta_delta_qemo_{tag}.png", dpi=150); plt.close(fig)
    # prefix sharing
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    vc = cells["n_prefixes"].value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color=BLUE, width=0.7)
    axes[0].set_xlabel("distinct prefixes per (bucket, act) cell, within tree", color=INK2, fontsize=8)
    axes[0].set_ylabel("cells", color=INK2, fontsize=9)
    em = np.clip(cells["evidence_multiplier"].values, 0, 20)
    axes[1].hist(em, bins=40, color=BLUE)
    axes[1].axvline(3, color=ORANGE, lw=2, ls="--")
    axes[1].text(3, axes[1].get_ylim()[1] * 0.95, " proposed ≥3", color=INK2, fontsize=8, va="top")
    axes[1].set_xlabel("evidence multiplier N_pool / median N_node (clipped at 20)", color=INK2, fontsize=8)
    for ax in axes:
        style(ax)
    fig.suptitle(f"{tag}: AffPool prefix sharing (median split)", color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"{figdir}/prefix_sharing_{tag}.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def nodekey_block(df, bucket, boot):
    out = {}
    for pop, d in (("all", df), ("fresh", df[~df["child_from_cache"]])):
        blk = {"rows": int(len(d)), "within_prefix": nodekey_contrast(d, bucket, boot),
               "pooled_omega2": pooled_omega2(d, bucket, ["depth", "action"], boot)}
        # proxy check: remove each edge's own mean z, so only within-edge variation is left to attribute to the bucket.
        # A pooled omega^2 that collapses here was prefix identity, not within-prefix affect.
        de = d.assign(z=d["z"] - d.groupby("edge_key")["z"].transform("mean"))
        blk["pooled_omega2_edge_demeaned"] = pooled_omega2(de, bucket, ["depth", "action"], boot)
        # what a NodeKey split would leave per (edge, bucket) child at depth >= 2, against the planned m=3 guard
        sv = d[d["depth"] >= 2].groupby(["depth", "edge_key", bucket]).size()
        blk["post_split_visits"] = {"median": float(sv.median()), "share_ge_3": float((sv >= 3).mean()),
                                    "by_depth": {int(k): {"median": float(g.median()), "share_ge_3": float((g >= 3).mean()), "cells": int(len(g))}
                                                 for k, g in sv.groupby(level=0) if k <= 6}}
        # §7.5.4 stratified re-runs
        #  (a) groups additionally keyed on absolute dialogue position (turn + depth), global tau
        blk["pooled_omega2_abs_position_strata"] = pooled_omega2(d, bucket, ["abs_pos", "depth", "action"], boot)
        #  (b) depth-specific median split: the bucket cannot encode depth at all (diagnostic, not a gate tau)
        dd = d.copy()
        dd["bucket_depth_med"] = (dd["parent_nu"] < dd.groupby("depth")["parent_nu"].transform("median")).astype(int)
        if bucket == "bucket_med":
            blk["within_depth_median_split"] = {"within_prefix": nodekey_contrast(dd, "bucket_depth_med", boot),
                                                "pooled_omega2": pooled_omega2(dd, "bucket_depth_med", ["depth", "action"], boot)}
        out[pop] = blk
    out["boundary"] = {
        "fraction_within_0.1": float((abs(df["parent_nu"] - df["tau_med"]) < 0.1).mean()),
        "fraction_within_0.05": float((abs(df["parent_nu"] - df["tau_med"]) < 0.05).mean()),
        "fraction_within_0.1_depth_ge_2": float((abs(df.loc[df.depth >= 2, "parent_nu"] - df["tau_med"].iloc[0]) < 0.1).mean()),
        "fraction_within_0.05_depth_ge_2": float((abs(df.loc[df.depth >= 2, "parent_nu"] - df["tau_med"].iloc[0]) < 0.05).mean()),
        "parent_nu_range_observed": [float(df["parent_nu"].min()), float(df["parent_nu"].max())],
        "parent_nu_iqr": float(df["parent_nu"].quantile(0.75) - df["parent_nu"].quantile(0.25)),
    }
    out["confound_depth"] = confound(df, bucket, boot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("steps_dir")
    ap.add_argument("--beta", type=float, required=True, help="beta_emo the run used")
    ap.add_argument("--beta_measure", type=float, default=0.7, help="beta whose affect term §7.6 measures")
    ap.add_argument("--figdir", default=None); ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.steps_dir, "steps.parquet"))
    df["abs_pos"] = df["turn_index"] + df["depth"]
    boot = Boot(sorted(df["dlg_id"].unique()), a.B)
    res = {"tag": a.tag, "rows": int(len(df)), "dialogues": int(df["dlg_id"].nunique()), "trees": int(df["tree_key"].nunique()),
           "prefixes": int((df["tree_key"] + "|" + df["prefix_key"]).nunique()), "edges": int(df["edge_key"].nunique()),
           "tau_med": float(df["tau_med"].iloc[0]), "fresh_rows": int((~df["child_from_cache"]).sum()),
           "bootstrap": {"B": a.B, "cluster": "dlg_id", "interval": "percentile 95%"}}
    res["depth_profile"] = visit_profile(df)
    res["splits"] = {"median": {"occupancy": occupancy(df, "bucket_med")}, "label": {"occupancy": occupancy(df, "bucket_lab")},
                     "agreement": split_agreement(df), "agreement_depth_ge_2": split_agreement(df[df.depth >= 2])}
    res["nodekey"] = {}
    res["affpool"] = {}
    cells_fig = None
    for bname, col in (("median", "bucket_med"), ("label", "bucket_lab")):
        print(f"[{a.tag}] nodekey {bname}", flush=True)
        res["nodekey"][bname] = nodekey_block(df, col, boot)
        print(f"[{a.tag}] affpool {bname}", flush=True)
        ap_plain, cells = affpool(df, col, boot, depth_keyed=False)
        ap_depth, _ = affpool(df, col, boot, depth_keyed=True)
        res["affpool"][bname] = {"sharing_measured": "within_tree", "plain": ap_plain, "depth_keyed": ap_depth}
        if bname == "median":
            cells_fig = cells
    print(f"[{a.tag}] channel dominance", flush=True)
    res["channel_dominance"], two = channel_dominance(df, a.beta, a.beta_measure, boot)
    res["sigma"] = sigma_emo(df)
    with open(os.path.join(a.steps_dir, f"p_var_{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))
    if a.figdir:
        figures(a.tag, df, two, cells_fig, a.figdir, a.beta_measure)
    print(f"wrote {a.steps_dir}/p_var_{a.tag}.json")


if __name__ == "__main__":
    main()
