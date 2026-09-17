#!/usr/bin/env python3
"""Build ``gridA/RESULTS.md`` from the Grid A run directories.

Reads every ``gridA/runs/<stage>__<model>__<persona>persona__<method>__<sims>s__seed<n>/``
written by ``scripts/run_sweep_experiments.sh`` and reports:

  * completeness / stop conditions first (a missing cell destroys the interaction, so it is
    reported at the top, not buried under the numbers)
  * the factorial table: SR and AvgT per method per model x persona cell, plus
    ``delta = SR(EmoMCTS+topK) - SR(GDP-Zero+topK)`` -- the headline is the model x persona
    interaction in delta, not absolute SR
  * mean +/- std across seeds wherever a cell has more than one
  * paired McNemar (exact binomial on the discordant pairs) on per-dialogue outcomes, valid
    because every method in a cell sees the same scenarios in the same order with the same
    persona assignment
  * Wilson CIs on every SR, normal CIs on every AvgT, and a paired CI on every delta
  * the no-MCTS baseline against DialogXpert's published figures
  * the cost table: wall-clock and LLM calls per role, per model, per persona

No scipy: the Wilson interval and the exact two-sided binomial are computed here.

    python3 scripts/gridA_report.py --grid-dir gridA
"""
import argparse
import json
import math
import os
import pickle
import re
import statistics
import sys
from collections import defaultdict

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))   # metrics.dialog_metrics
sys.path.insert(0, SCRIPTS_DIR)                      # gridA_cache_hit

Z = 1.959963984540054  # 95%

METHODS = ["gdpzero", "gdpzero_topk", "emomcts_topk"]
METHOD_LABEL = {
    "gdpzero": "GDP-Zero",
    "gdpzero_topk": "GDP-Zero + top-K",
    "emomcts_topk": "EmoMCTS + top-K",
    "llm_raw": "llm_raw (no search)",
}
MODEL_LABEL = {
    "vicuna": "Vicuna-13B",
    "llama31": "Llama-3.1-8B-Instruct",
    "qwen25": "Qwen2.5-7B-Instruct",
}
MODEL_ORDER = ["vicuna", "llama31", "qwen25"]
PERSONA_ORDER = ["no", "yes"]

# DialogXpert's published PersuasionForGood figures, quoted for the anchor cell only.
DIALOGXPERT_PUBLISHED = {"SR": 0.8132, "AvgT": 5.07}

RUN_RE = re.compile(
    r"^(?P<stage>a\d)__(?P<model>[^_]+)__(?P<persona>no|yes)persona__"
    r"(?P<method>.+?)__(?P<sims>\d+)s__seed(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# statistics (pure python -- scipy is not a dependency of this repo)
# ---------------------------------------------------------------------------
def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion. Better than Wald at n=100/SR near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values, z=Z):
    """Normal CI on a mean (used for AvgT, where n=100 per run makes the CLT fine)."""
    n = len(values)
    if n == 0:
        return (float("nan"), (float("nan"), float("nan")))
    m = statistics.fmean(values)
    if n < 2:
        return (m, (float("nan"), float("nan")))
    se = statistics.stdev(values) / math.sqrt(n)
    return (m, (m - z * se, m + z * se))


def binom_two_sided(b, c):
    """Exact two-sided binomial p on the discordant pairs -- McNemar's exact test.

    ``b`` = pairs the first method won, ``c`` = pairs the second won. Exact rather than the
    chi-square approximation because ``b + c`` is routinely under 25 here, which is where the
    approximation misbehaves.
    """
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def paired_diff_ci(b, c, n, z=Z):
    """CI for the paired difference of proportions (Wald form for matched pairs).

    ``b``/``c`` are the discordant counts, ``n`` the number of pairs. The variance term uses
    the discordant counts only, which is what makes the paired test buy power over two
    independent proportions on the same data.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    d = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return (d - half, d + half)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, run_dir, name, meta):
        self.dir = run_dir
        self.name = name
        for k, v in meta.items():
            setattr(self, k, v)
        self.sims = int(self.sims)
        self.seed = int(self.seed)
        self.episodes = []
        self.metadata = {}
        self.profile = {}
        self.cache = (0, 0)

    # per-dialogue outcome map, the unit the paired test works on
    @property
    def outcomes(self):
        return {ep["did"]: bool(ep.get("success")) for ep in self.episodes}

    @property
    def n(self):
        return len(self.episodes)

    @property
    def cache_rate(self):
        hits, edges = self.cache
        return hits / edges if edges else None


def load_runs(runs_dir, expected_dialogs):
    from metrics.dialog_metrics import compute_metrics
    from gridA_cache_hit import cache_hit_rate

    runs = []
    for name in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        run_dir = os.path.join(runs_dir, name)
        m = RUN_RE.match(name)
        if not os.path.isdir(run_dir) or not m:
            continue
        run = Run(run_dir, name, m.groupdict())
        pkl = os.path.join(run_dir, f"{name}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                run.episodes = pickle.load(f)
        meta_path = os.path.join(run_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                run.metadata = json.load(f)
        prof_path = os.path.join(run_dir, f"{name}_role_profile.json")
        if os.path.exists(prof_path):
            with open(prof_path, encoding="utf-8") as f:
                run.profile = json.load(f)
        run.cache = cache_hit_rate(run_dir)
        max_turns = (run.metadata.get("args") or {}).get("max_turns")
        run.metrics = compute_metrics(run.episodes, task=(run.metadata.get("args") or {}).get("game"),
                                      max_turns=max_turns) if run.episodes else {}
        run.complete = run.n >= expected_dialogs
        runs.append(run)
    return runs


def load_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    if not rows:
        return []
    head, body = rows[0], rows[1:]
    return [dict(zip(head, r)) for r in body]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def fmt_sr(run):
    k = sum(1 for v in run.outcomes.values() if v)
    lo, hi = wilson(k, run.n)
    return f"{run.metrics['SR']:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_at(run):
    max_turns = (run.metadata.get("args") or {}).get("max_turns")
    per_dialog = _turns_vector(run.episodes, max_turns)
    m, (lo, hi) = mean_ci(per_dialog)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def _turns_vector(episodes, max_turns, failures_as_max=True):
    """Per-dialogue turn counts under the same convention as metrics.average_turn.

    Duplicated here (rather than calling ``average_turn``) because the CI needs the vector,
    not the mean; the two must agree, so any change to the convention has to be made in both.
    """
    out = []
    use_cap = failures_as_max and max_turns is not None
    for ep in episodes:
        n = ep.get("num_turns", 0)
        reached = ep.get("success") and (max_turns is None or n <= max_turns)
        if reached:
            out.append(min(n, max_turns) if max_turns is not None else n)
        elif use_cap:
            out.append(max_turns)
    return out


def pooled_sr(runs):
    """SR over every (seed, dialogue) in a cell, with a Wilson CI."""
    k = sum(1 for r in runs for v in r.outcomes.values() if v)
    n = sum(r.n for r in runs)
    if n == 0:
        return "--"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def pooled_at(runs):
    """AvgT over every (seed, dialogue) in a cell, with a normal CI."""
    vec = []
    for r in runs:
        vec += _turns_vector(r.episodes, (r.metadata.get("args") or {}).get("max_turns"))
    m, (lo, hi) = mean_ci(vec)
    return "--" if not vec else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def mean_std(values, digits=3):
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.fmean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------
def section_completeness(out, runs, args, skipped, timing):
    out.append("## 0. Completeness and stop conditions\n")
    by_cell = defaultdict(list)
    for r in runs:
        by_cell[(r.stage, r.model, r.persona, r.method, r.sims, r.seed)].append(r)

    expected = []
    for seed in args.anchor_seeds:
        for meth in METHODS:
            expected.append(("a1", "vicuna", "no", meth, args.sims, seed))
    for model in MODEL_ORDER:
        for seed in args.a2_seeds:
            for meth in METHODS:
                expected.append(("a2", model, "no", meth, args.a2_sims, seed))
    for model in MODEL_ORDER:
        for persona in PERSONA_ORDER:
            if model == "vicuna" and persona == "no":
                continue
            for meth in METHODS:
                expected.append(("a4", model, persona, meth, args.sims, args.seed))

    missing = [k for k in expected if not by_cell.get(k)]
    incomplete = [r for r in runs if not r.complete]
    if not missing and not incomplete and not skipped:
        out.append("All expected runs are present and complete.\n")
    if missing:
        out.append(f"**{len(missing)} expected run(s) MISSING.** A missing cell destroys the "
                   "model x persona interaction; no substitute configuration has been used in "
                   "its place.\n")
        out.append("| stage | model | persona | method | sims | seed |")
        out.append("|---|---|---|---|---|---|")
        for stage, model, persona, meth, sims, seed in missing:
            out.append(f"| {stage} | {model} | {persona} | {meth} | {sims} | {seed} |")
        out.append("")
    if incomplete:
        out.append(f"**{len(incomplete)} run(s) INCOMPLETE** (fewer than {args.dialogs} dialogues):\n")
        for r in incomplete:
            out.append(f"- `{r.name}` — {r.n}/{args.dialogs} episodes")
        out.append("")
    if skipped:
        out.append("**Cells skipped because the SGLang server held a different backbone.** "
                   "They were not substituted; re-run them with that model served.\n")
        out.append("```")
        out += skipped
        out.append("```\n")

    out.append("### Realization-cache hit rate (stop condition: below 0.50)\n")
    out.append("Fraction of logged tree edges served from the MCTS realization cache. A collapse "
               "here means prompt serialization changed, which would split the grid into two "
               "incomparable batches.\n")
    out.append("| run | cache hit rate | edges |")
    out.append("|---|---|---|")
    for r in runs:
        rate = r.cache_rate
        cell = "n/a (no tree)" if rate is None else f"{rate:.3f}" + ("  **BELOW 0.50**" if rate < 0.50 else "")
        out.append(f"| `{r.name}` | {cell} | {r.cache[1]} |")
    out.append("")
    failures = [row for row in timing if row.get("status") not in ("0", "", None)]
    if failures:
        out.append("### Runner exits with non-zero status\n")
        for row in failures:
            out.append(f"- {row['stage']}/{row['model']}/{row['persona']}/{row['method']} "
                       f"seed {row['seed']}: exit {row['status']}")
        out.append("")


def cell_runs(runs, model, persona, sims, stages=("a1", "a4")):
    """Every run in one model x persona cell at one budget, grouped by method."""
    by_method = defaultdict(list)
    for r in runs:
        if r.model == model and r.persona == persona and r.sims == sims and r.stage in stages:
            by_method[r.method].append(r)
    for v in by_method.values():
        v.sort(key=lambda r: r.seed)
    return by_method


def section_factorial(out, runs, args):
    out.append("## 1. The factorial grid — SR, AvgT, and Δ per cell\n")
    out.append(f"n_sims={args.sims}, {args.dialogs} dialogues per run, `--emo_signal level`, "
               "`--emo_risk_lambda 0.0`, `w(e) = soft` (post-freeze).\n")
    out.append("> **Comparability rule.** Numbers are comparable *within* a model × persona cell "
               "and never across cells: the backbone changes the user simulator, the value "
               "estimator and the utterance model at once, so cross-cell differences in absolute "
               "SR are not attributable to the planner. The headline result is the model × "
               "persona interaction in Δ, not absolute SR. Only the Vicuna / no-persona cell is "
               "comparable to DialogXpert's published figures (§3).\n")
    out.append("Δ = SR(EmoMCTS + top-K) − SR(GDP-Zero + top-K), paired on dialogue id. "
               "Brackets are 95% CIs (Wilson for SR, normal for AvgT, matched-pairs for Δ). "
               "`±` is the std across seeds where a cell has more than one.\n")

    for model in MODEL_ORDER:
        for persona in PERSONA_ORDER:
            by_method = cell_runs(runs, model, persona, args.sims)
            if not any(by_method.values()):
                continue
            anchor = (model == "vicuna" and persona == "no")
            title = f"### Cell: {MODEL_LABEL[model]} / persona={persona}"
            out.append(title + ("  — **ANCHOR CELL**" if anchor else "") + "\n")
            out.append("| method | SR (mean ± std) | SR pooled [95% CI] | AvgT (mean ± std) "
                       "| AvgT pooled [95% CI] | seeds |")
            out.append("|---|---|---|---|---|---|")
            for meth in METHODS:
                rs = [r for r in by_method.get(meth, []) if r.complete]
                if not rs:
                    out.append(f"| {METHOD_LABEL[meth]} | *missing* | *missing* | *missing* "
                               "| *missing* | 0 |")
                    continue
                out.append(f"| {METHOD_LABEL[meth]} | {mean_std([r.metrics['SR'] for r in rs])} "
                           f"| {pooled_sr(rs)} | {mean_std([r.metrics['AT'] for r in rs], 2)} "
                           f"| {pooled_at(rs)} | {len(rs)} |")
            out.append("")
            if max((len(v) for v in by_method.values()), default=0) > 1:
                out.append("The pooled CI treats every (seed, dialogue) as one independent "
                           "observation. Seeds replay the same scenarios, so it is mildly "
                           "optimistic; the `± std` across seeds is the honest run-to-run "
                           "spread.\n")

            emo = [r for r in by_method.get("emomcts_topk", []) if r.complete]
            base = [r for r in by_method.get("gdpzero_topk", []) if r.complete]
            if emo and base:
                out.append(_delta_block(emo, base))
            else:
                out.append("Δ: *not computable — one of the two arms is missing.*\n")


def _pair(run_a, run_b):
    """Discordant counts over the dialogues both runs actually played."""
    oa, ob = run_a.outcomes, run_b.outcomes
    shared = sorted(set(oa) & set(ob))
    b = sum(1 for d in shared if oa[d] and not ob[d])   # A wins
    c = sum(1 for d in shared if ob[d] and not oa[d])   # B wins
    return b, c, len(shared)


def _delta_block(emo_runs, base_runs):
    lines = []
    by_seed_emo = {r.seed: r for r in emo_runs}
    by_seed_base = {r.seed: r for r in base_runs}
    seeds = sorted(set(by_seed_emo) & set(by_seed_base))
    if not seeds:
        return "Δ: *no seed is present for both arms.*\n"
    deltas = []
    lines.append("| seed | Δ SR | pairs | EmoMCTS wins (b) | GDP-Zero+topK wins (c) | "
                 "McNemar exact p | 95% CI on Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    tot_b = tot_c = tot_n = 0
    for s in seeds:
        a, bb = by_seed_emo[s], by_seed_base[s]
        b, c, n = _pair(a, bb)
        tot_b, tot_c, tot_n = tot_b + b, tot_c + c, tot_n + n
        d = (b - c) / n if n else float("nan")
        deltas.append(d)
        lo, hi = paired_diff_ci(b, c, n)
        lines.append(f"| {s} | {d:+.3f} | {n} | {b} | {c} | {binom_two_sided(b, c):.4f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] |")
    if len(seeds) > 1:
        lo, hi = paired_diff_ci(tot_b, tot_c, tot_n)
        d = (tot_b - tot_c) / tot_n
        lines.append(f"| **pooled** | **{d:+.3f}** | {tot_n} | {tot_b} | {tot_c} | "
                     f"{binom_two_sided(tot_b, tot_c):.4f} | [{lo:+.3f}, {hi:+.3f}] |")
        lines.append(f"| mean ± std | {mean_std(deltas)} | | | | | |")
    lines.append("")
    lines.append("Pairs are dialogues both arms played; the two arms share scenario order and "
                 "persona assignment, which is what makes the pairing valid. Pooling across "
                 "seeds treats (seed, dialogue) as the pair and so counts each scenario once per "
                 "seed.\n")
    return "\n".join(lines)


def section_a2(out, runs, args):
    a2 = [r for r in runs if r.stage == "a2" and r.complete]
    if not a2:
        return
    out.append(f"## 2. A2 — replicate grid at n_sims={args.a2_sims}\n")
    out.append("Three seeds per method per model, so every number carries an error bar. This is "
               "the answer to \"all three methods read exactly 0.58\": with one seed that "
               "coincidence is unfalsifiable, with three it is a std.\n")
    out.append("| model | method | SR (mean ± std) | AvgT (mean ± std) | seeds |")
    out.append("|---|---|---|---|---|")
    for model in MODEL_ORDER:
        for meth in METHODS:
            rs = sorted([r for r in a2 if r.model == model and r.method == meth],
                        key=lambda r: r.seed)
            if not rs:
                continue
            srs = [r.metrics["SR"] for r in rs]
            ats = [r.metrics["AT"] for r in rs]
            out.append(f"| {MODEL_LABEL[model]} | {METHOD_LABEL[meth]} | {mean_std(srs)} "
                       f"| {mean_std(ats, 2)} | {len(rs)} |")
    out.append("")
    for model in MODEL_ORDER:
        emo = [r for r in a2 if r.model == model and r.method == "emomcts_topk"]
        base = [r for r in a2 if r.model == model and r.method == "gdpzero_topk"]
        if emo and base:
            out.append(f"**Δ — {MODEL_LABEL[model]}, n_sims={args.a2_sims}**\n")
            out.append(_delta_block(emo, base))


def section_dialogxpert(out, runs, args):
    out.append("## 3. No-MCTS baseline vs DialogXpert (published)\n")
    out.append("> **DEVIATION.** DialogXpert is not implemented in this repository — there is no "
               "trained dialogue policy here, only `llm_raw` / `gdpzero` / `emomcts`. Stage a3 "
               "runs the nearest available no-search baseline (`--algo llm_raw`) under the anchor "
               "cell's conditions. It is **not** a DialogXpert reproduction and is not labelled "
               "as one; the published figures below are quoted, not reproduced.\n")
    out.append("| system | SR | AvgT | source |")
    out.append("|---|---|---|---|")
    out.append(f"| DialogXpert | {DIALOGXPERT_PUBLISHED['SR']:.4f} | "
               f"{DIALOGXPERT_PUBLISHED['AvgT']:.2f} | published |")
    a3 = [r for r in runs if r.stage == "a3" and r.complete]
    if a3:
        for r in a3:
            out.append(f"| {METHOD_LABEL.get(r.method, r.method)} on {MODEL_LABEL[r.model]} | "
                       f"{fmt_sr(r)} | {fmt_at(r)} | this work (a3) |")
    else:
        out.append("| *(a3 not run)* | -- | -- | -- |")
    anchor = cell_runs(runs, "vicuna", "no", args.sims)
    for meth in METHODS:
        rs = [r for r in anchor.get(meth, []) if r.complete]
        if rs:
            srs = [r.metrics["SR"] for r in rs]
            ats = [r.metrics["AT"] for r in rs]
            out.append(f"| {METHOD_LABEL[meth]} on {MODEL_LABEL['vicuna']} (anchor) | "
                       f"{mean_std(srs)} | {mean_std(ats, 2)} | this work (a1) |")
    out.append("")
    out.append("Only this cell — Vicuna, no persona — is comparable to the published figures; "
               "every other cell in §1 uses a different backbone or conditions the persuadee on "
               "a real participant profile, and neither matches DialogXpert's setup.\n")


def section_cost(out, runs, timing, args):
    out.append("## 4. Cost — wall-clock and LLM calls per role\n")
    out.append("Per-dialogue figures are run totals divided by the number of dialogues. "
               "`--profile_roles` was on for every run. Note that with `--num_workers > 1` "
               "dialogues overlap, so *mean latency* per call is inflated by contention; "
               "**call and token counts are unaffected** and are the figures the paper's cost "
               "claim rests on.\n")
    workers = sorted({(r.metadata.get("args") or {}).get("num_workers")
                      for r in runs if r.metadata} - {None})
    if workers and workers != [1]:
        out.append(f"> **`--num_workers` = {workers} in this grid.** Wall-clock per run is real "
                   "elapsed time with that many dialogues in flight, so it reflects throughput, "
                   "not per-dialogue serial latency. For a serial-latency figure, re-run one cell "
                   "per backbone with `NUM_WORKERS=1`.\n")
    wall = {}
    for row in timing:
        key = (row["stage"], row["model"], row["persona"], row["method"], row["sims"], row["seed"])
        wall[key] = float(row["wall_s"])

    out.append("| model | persona | method | sims | wall-clock (s) | s/dialogue | "
               "policy_prior | value_est | user_sim | sys_utt | emo_clf | total calls | "
               "calls/dialogue |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    roles = ["policy_prior", "value_estimator", "user_simulator",
             "system_utterance", "emotion_classifier"]
    for model in MODEL_ORDER:
        for persona in PERSONA_ORDER:
            budgets = sorted({r.sims for r in runs if r.model == model
                              and r.persona == persona and r.complete})
            for sims in budgets:
              for meth in METHODS + ["llm_raw"]:
                # keyed by budget as well: a2 (n_sims=10) and a4 (n_sims=50) are different costs
                # and averaging them together would understate the n_sims=50 rows.
                rs = [r for r in runs if r.model == model and r.persona == persona
                      and r.method == meth and r.sims == sims and r.complete]
                if not rs:
                    continue
                # average over seeds; a cell with one seed reports that run
                per_role = {role: [] for role in roles}
                totals, walls, secs_per = [], [], []
                for r in rs:
                    rp = (r.profile or {}).get("roles", {})
                    for role in roles:
                        per_role[role].append(rp.get(role, {}).get("calls", 0) / r.n)
                    totals.append(sum(v.get("calls", 0) for v in rp.values()) / r.n)
                    key = (r.stage, r.model, r.persona, r.method, str(r.sims), str(r.seed))
                    if key in wall:
                        walls.append(wall[key])
                        secs_per.append(wall[key] / r.n)
                w = f"{statistics.fmean(walls):.0f}" if walls else "--"
                sp = f"{statistics.fmean(secs_per):.1f}" if secs_per else "--"
                cells = " | ".join(f"{statistics.fmean(per_role[role]):.1f}" for role in roles)
                tot = statistics.fmean(totals)
                out.append(f"| {MODEL_LABEL[model]} | {persona} | {METHOD_LABEL[meth]} | "
                           f"{rs[0].sims} | {w} | {sp} | {cells} | "
                           f"{tot * rs[0].n:.0f} | {tot:.1f} |")
    out.append("")
    out.append("Columns `policy_prior` … `emo_clf` are **calls per dialogue** for that role.\n")


def section_deviations(out, grid_dir, runs, args):
    out.append("## 5. Deviations from the spec\n")
    out.append("1. **DialogXpert (A3) is not reproduced.** The repository contains no trained "
               "dialogue policy. Stage a3 runs `--algo llm_raw` as the no-search baseline and the "
               "published 0.8132 / 5.07 are quoted as published. See §3.\n")
    out.append("2. **Cache hit rate is a proxy.** The frozen NDJSON subtree schema stores a "
               "per-node `cache_hit` boolean, not raw hit/miss counts, and the schema may not be "
               "extended after the Week-1 freeze. §0 therefore reports the fraction of logged "
               "edges whose node was served from the realization cache at least once. It moves "
               "with the true hit rate and is measured identically for every run, so the 0.50 "
               "stop condition is applied consistently.\n")
    out.append("3. **Per-dialogue cost is derived, not measured per dialogue.** "
               "`utils/role_profiler` accumulates per-role totals for a run; §4 divides by the "
               "dialogue count. Per-run wall-clock is measured directly (`gridA/TIMING.tsv`).\n")
    prov = os.path.join(grid_dir, "PROVENANCE.tsv")
    if os.path.exists(prov):
        with open(prov, encoding="utf-8") as f:
            entries = [l.split("\t") for l in f.read().splitlines() if l.strip()]
        heads = {e[1] for e in entries if len(e) > 1}
        if len(heads) > 1:
            out.append(f"4. **The code revision changed between passes** ({len(heads)} distinct "
                       "commits in `gridA/PROVENANCE.tsv`). Grid parameters were held fixed by the "
                       "`BATCH.json` guard, but check the intervening diff before treating every "
                       "cell as one batch.\n")
        out.append("\n<details><summary>Pass provenance</summary>\n\n```")
        out += ["\t".join(e) for e in entries]
        out.append("```\n</details>\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", default=os.path.join(REPO_ROOT, "gridA"))
    ap.add_argument("--out", default=None, help="default: <grid-dir>/RESULTS.md")
    ap.add_argument("--dialogs", type=int, default=100)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--a2-sims", dest="a2_sims", type=int, default=10)
    ap.add_argument("--anchor-seeds", dest="anchor_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--a2-seeds", dest="a2_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0, help="the single seed used outside the anchor cell")
    args = ap.parse_args()

    grid_dir = os.path.abspath(args.grid_dir)
    runs_dir = os.path.join(grid_dir, "runs")
    out_path = args.out or os.path.join(grid_dir, "RESULTS.md")

    runs = load_runs(runs_dir, args.dialogs)
    timing = load_tsv(os.path.join(grid_dir, "TIMING.tsv"))
    skipped_path = os.path.join(grid_dir, "SKIPPED.log")
    skipped = open(skipped_path, encoding="utf-8").read().splitlines() if os.path.exists(skipped_path) else []

    batch = {}
    batch_path = os.path.join(grid_dir, "BATCH.json")
    if os.path.exists(batch_path):
        with open(batch_path, encoding="utf-8") as f:
            batch = json.load(f)

    out = ["# Grid A — results\n"]
    shown = os.path.relpath(runs_dir, REPO_ROOT)
    if shown.startswith(".."):
        shown = runs_dir
    out.append(f"{len(runs)} run directories under `{shown}`.\n")
    out.append("> The pre-freeze **0.70 SR / 6.72 AvgT** numbers are discarded: they were produced "
               "with the argmax-mined w(·) table. Every run here pins `--emo_valence_table soft` "
               "and the anchor cell was re-run from scratch.\n")
    if batch:
        out.append("<details><summary>Frozen batch parameters</summary>\n\n```json")
        out.append(json.dumps(batch, indent=2))
        out.append("```\n</details>\n")

    section_completeness(out, runs, args, skipped, timing)
    section_factorial(out, runs, args)
    section_a2(out, runs, args)
    section_dialogxpert(out, runs, args)
    section_cost(out, runs, timing, args)
    section_deviations(out, grid_dir, runs, args)

    os.makedirs(grid_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {out_path}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
