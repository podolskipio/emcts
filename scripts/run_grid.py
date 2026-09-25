"""Run the §4c grid (analysis/thu/plan_4c_run_table.md, Thu Sep 17 revision) in the plan's order.

    python3 scripts/run_grid.py --dry_run          # print every run + check it against its frozen config
    python3 scripts/run_grid.py                    # start from the first run
    python3 scripts/run_grid.py --continue         # after Ctrl+C / crash: skip finished runs, redo the cut one
    python3 scripts/run_grid.py --steps 1 2        # only the plan's Order steps 1 and 2

Each run gets its own folder analysis/grid/runs/<tag>/ with
    run_record.json   what the run was (plan reference, one-sentence description, arguments),
                      its status, and its results (SR + 95% CI, AvgT, wall-clock)
    runner.log        everything the runner printed
    <tag>/<tag>.pkl   the episodes (the runner nests its output one folder deeper)

A run counts as finished only when the runner exited cleanly AND wrote all 100 episodes -- the
runner skips a dialogue that raises and still exits 0. --continue re-runs every unfinished run from
scratch; the partial folder is moved to analysis/grid/runs/_interrupted/, never deleted.
"""
import os
import sys
import json
import math
import time
import shutil
import pickle
import signal
import argparse
import subprocess
import urllib.request
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
RUNS_DIR = os.path.join(REPO, "analysis", "grid", "runs")
FROZEN_DIR = os.path.join(REPO, "analysis", "grid", "configs", "frozen")
PLAN_FILE = "analysis/thu/plan_4c_run_table.md"

VICUNA = "TheBloke/vicuna-13B-v1.5-AWQ"
QWEN = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SGLANG_HOST = os.environ.get("SGLANG_HOST", "http://127.0.0.1:30000")
N_DIALOGUES = 100


# ---------------------------------------------------------------------------------------------
# The plan. Everything below this line up to "Running" is a transcription of plan §4c.
# ---------------------------------------------------------------------------------------------

# "Shared arguments -- every grid run, explicit, no inheritance". Runner defaults differ from
# these in several places (max_conv 20, R 3, llm classifier, ...), so every one is written out.
SHARED_ARGS = [
    "--game", "emo_p4g",
    "--algo", "emomcts",
    "--llm", "sglang",
    "--data", os.path.join(REPO, "data", "p4g", "rollout_evalset_nonannotated.jsonl"),
    "--max_conv", "100",               # DEFAULT IS 20 -- omitting it evaluates a fifth of the set
    "--search_horizon", "episode",
    "--max_turns", "10",
    "--max_realizations", "4",         # NOT the runner default of 3
    "--llm_prior_topk", "5",           # K; 0 only for B1
    "--emotion_classifier", "hf",      # NOT the runner default of llm
    "--emo_valence_table", "generic",
    "--cpuct", "1.0",
    "--Q_0", "0.0",
    "--emo_risk_lambda", "0.0",
    "--logit_scoring", "off",
    "--p4g_persona",
    "--p4g_success", "tag",
    "--num_workers", "10",
]

# "Pool bias is 0.25, derived not chosen": beta = 0.5 at depth 2 (n ~ 2, n_pool ~ 5) gives b ~ 0.27.
# One constant on purpose -- it must be identical for AffPool and ActPool, or the grid's central
# contrast is confounded.
POOL_BIAS = "0.25"

# Block A: arm-specific arguments -- "nothing else differs between arms".
# DOSE BUG, measured 2026-09-21, NOT fixed here. The three selection arms were matched at a
# 15.2 % argmax flip rate on the P1 pilot trees. Replayed off the grid's own simlogs, the
# achieved live rates are:
#
#       arm             beta    s20      s50
#       Bias            0.70    19.0 %   27.2 %
#       CenteredBias    1.03    13.1 %   24.2 %
#       Momentum        1.11    13.3 %   20.2 %
#
# The doses do not transfer across budget, and Bias intervenes 45 % more than CenteredBias at
# s20 -- the budget where Bias appears to beat it (+0.09). Bias vs CenteredBias is therefore
# dose-confounded and uninterpretable at any n. The betas are deliberately left alone: changing
# them here would silently break comparability with the twelve seed-1 runs already in the record
# and with their frozen configs. The fix is to re-derive the doses per budget off the grid's
# NoEmo trees (the replay is offline and costs no GPU), then regenerate the affected frozen
# configs under a new tag -- not to edit these constants in place.
ARM_ARGS = {
    "NoEmo":        ["--beta_emo", "0.0"],
    "Bias":         ["--beta_emo", "0.7", "--emo_signal", "level"],
    "Momentum":     ["--beta_emo", "1.11", "--emo_signal", "delta"],
    "CenteredBias": ["--beta_emo", "1.03", "--emo_signal", "level", "--emo_centre"],
    "AffPool":      ["--beta_emo", "0.0", "--aff_pool", "--aff_pool_bias", POOL_BIAS,
                     "--aff_pool_tau", "0.263", "--aff_pool_key", "affect"],
    "ActPool":      ["--beta_emo", "0.0", "--aff_pool", "--aff_pool_bias", POOL_BIAS,
                     "--aff_pool_key", "act"],     # no --aff_pool_tau: the act key has no bucket
}

ARM_DESCRIPTION = {
    "NoEmo":        "GDP-Zero search + top-K with no affect signal -- the control every affect arm is read against.",
    "Bias":         "Adds beta*Q_emo (generic valence, level signal) to selection -- affect plus the flat visited-edge bonus.",
    "Momentum":     "Selection bonus from the mean-zero local affective change (delta signal) -- direction instead of level, bonus-free.",
    "CenteredBias": "Bias with the flat visited-edge component removed (--emo_centre) -- isolates the differential affect signal.",
    "AffPool":      "RAVE-style pooling of task returns keyed on (affect bucket tau=0.263, act) -- the affective pooling arm.",
    "ActPool":      "The same pooling keyed on dialogue act alone -- the placebo partition AffPool is read against.",
}

# Block A cost table, hours per run (the table's per-arm total divided by its seeds).
EXPECTED_HOURS = {
    "NoEmo":        {20: 2.6, 50: 6.4},
    "Bias":         {20: 3.3, 50: 8.2},
    "Momentum":     {20: 2.7, 50: 6.7},
    "CenteredBias": {20: 3.1, 50: 7.8},
    "AffPool":      {20: 3.1, 50: 7.8},
    "ActPool":      {20: 3.1, 50: 7.8},
}

ORDER_STEPS = {
    0: "post-freeze priority: NoEmo's own seed spread, then ActPool vs NoEmo at s20",
    1: "n_sims=20, seed 1, all six arms",
    2: "n_sims=50, seed 1, all six arms",
    3: "B1, B2 -- baselines",
    4: "seed 2, both budgets",
    5: "B3-B5 -- Qwen, the second backbone",
    6: "seed 3 on NoEmo and Bias, both budgets",
    7: "Block C in priority order (only C3 and C4 have configs)",
}


def set_arg(args, flag, value):
    """Return a copy of args with flag's value replaced (the flag must already be there)."""
    args = list(args)
    args[args.index(flag) + 1] = value
    return args


def make_run(tag, step, block, arm, n_sims, seed, description, plan_row,
             expected_hours, backbone=VICUNA, shared=SHARED_ARGS):
    argv = (shared
            + ["--sglang_model", backbone,
               "--num_mcts_sims", str(n_sims),
               "--seed", str(seed)]
            + ARM_ARGS[arm]
            + ["--output", os.path.join(RUNS_DIR, tag, tag + ".pkl"),
               "--frozen_config", os.path.join(FROZEN_DIR, tag + ".json")])
    return {
        "tag": tag,
        "description": description,
        "plan_ref": f"Plan §4c ({PLAN_FILE}) > Order step {step} ({ORDER_STEPS[step]}) > {block} > {plan_row}",
        "order_step": step,
        "block": block,
        "arm": arm,
        "backbone": backbone,
        "num_mcts_sims": n_sims,
        "seed": seed,
        "expected_hours": expected_hours,
        "argv": argv,
    }


def block_a(arm, n_sims, seed, step):
    return make_run(
        tag=f"A_{arm}_s{n_sims}_seed{seed}", step=step,
        block="Block A -- the full budget factorial (primary)", arm=arm, n_sims=n_sims, seed=seed,
        description=ARM_DESCRIPTION[arm] + f" n_sims={n_sims}, seed {seed}.",
        plan_row=f"row {arm}, column n_sims={n_sims}, seed {seed}",
        expected_hours=EXPECTED_HOURS[arm][n_sims])


def qwen(arm):
    return make_run(
        tag=f"B_Qwen_{arm}", step=5,
        block="Block B -- baselines and replication", arm=arm, n_sims=50, seed=1, backbone=QWEN,
        description=f"{arm} on Qwen2.5-7B -- does the arm ordering hold on a backbone without Vicuna's happiness drift?",
        plan_row=f"B3-B5 (second backbone, Qwen2.5-7B-Instruct), arm {arm}",
        expected_hours=4.0)


def third_seed(arm, step=7):
    return make_run(
        tag=f"C3_{arm}_s20_seed3", step=step,
        block="Block C -- slack", arm=arm, n_sims=20, seed=3,
        description=ARM_DESCRIPTION[arm] + " Third seed at n_sims=20, for power on the arms carrying the claim.",
        plan_row=f"C3 (third seeds), arm {arm}",
        expected_hours=3.0)


# B1 runs GDP-Zero's own planner on the plain game with top-K off -- not emomcts at beta 0.
B1 = make_run(
    tag="B1_GDPZero_plain", step=3,
    block="Block B -- baselines and replication", arm="NoEmo", n_sims=50, seed=1,
    shared=set_arg(set_arg(set_arg(SHARED_ARGS, "--game", "p4g"), "--algo", "gdpzero"), "--llm_prior_topk", "0"),
    description="Plain GDP-Zero planner, top-K off -- the published baseline; B1 -> NoEmo isolates the top-K contribution.",
    plan_row="B1 (GDP-Zero, plain)",
    expected_hours=8.0)

# B2 is the only run not on the episode horizon. Paired with A_NoEmo_s50_seed1.
B2 = make_run(
    tag="B2_legacy_horizon", step=3,
    block="Block B -- baselines and replication", arm="NoEmo", n_sims=50, seed=1,
    shared=set_arg(SHARED_ARGS, "--search_horizon", "legacy"),
    description="NoEmo on the legacy search horizon -- re-measures the P1 horizon contrast at n=100, paired with A_NoEmo_s50_seed1.",
    plan_row="B2 (legacy horizon ablation)",
    expected_hours=11.0)

# C4: Bias under the predecision table. NOTE the plan asks for tau and dose re-derived for this
# table; the frozen config keeps beta 0.7 (7.5 % flips under predecision, not 15.2 %).
C4 = make_run(
    tag="C4_Bias_predecision", step=7,
    block="Block C -- slack", arm="Bias", n_sims=50, seed=1,
    shared=set_arg(SHARED_ARGS, "--emo_valence_table", "predecision"),
    description="Bias under the predecision valence table -- turns the leakage finding into an experiment.",
    plan_row="C4 (predecision valence ablation)",
    expected_hours=8.0)

SIX_ARMS = ["NoEmo", "Bias", "Momentum", "CenteredBias", "AffPool", "ActPool"]

# ---------------------------------------------------------------------------------------------
# 2026-09-21 amendment to the plan's order. Nothing about any run changed -- every tag below
# still validates against the same frozen config it always did. Only WHICH runs the default
# queue offers, and in what order, changed.
#
# The seed-1 diagnostics found that per-dialogue success carries no signal. Cohen's kappa over
# the 15 arm pairs is +0.087 (s20) / +0.023 (s50), and at s50 the number of arms succeeding per
# dialogue fits Binomial(6, 0.735) with chi2 = 4.3 on ~5 df: knowing which dialogue it is tells
# you nothing about whether an arm succeeds. Pairing therefore buys no power -- McNemar's
# advantage comes from positive correlation between the paired measurements -- which is why all
# nine paired tests came back p > 0.08. At independent Bernoulli(0.73) the 80 %-power detectable
# difference is 0.175 at n=100 and 0.101 at n=300 (three seeds).
#
#   * Step 0 (new) runs first. Its first two runs answer the question that decides the rest:
#     can ONE unchanged arm span the spread the grid is reading as an effect? Its last two
#     complete ActPool vs NoEmo at s20 -- the only contrast that is both unconfounded and large
#     enough (+0.11, McNemar p=0.080 at seed 1) that n=300 could resolve it.
#   * Step 4 (seed 2, both budgets, all six arms, ~63 h) is DEFERRED out of the default queue.
#     Its two headline contrasts are confounded: Bias and CenteredBias are not dose-matched in
#     practice (see ARM_ARGS above), and AffPool/ActPool hold the pool bias b fixed rather than
#     the RAVE weight beta, which is the quantity selection actually uses. More seeds would
#     measure two biased estimands more precisely. Reachable with `--steps 4`.
# ---------------------------------------------------------------------------------------------

# The order their results land in: NoEmo's seed spread first (~6 h), then the ActPool contrast.
PRIORITY = [
    block_a("NoEmo", 20, 2, step=0),
    block_a("NoEmo", 20, 3, step=0),
    block_a("ActPool", 20, 2, step=0),
    third_seed("ActPool", step=0),
]
PROMOTED = {run["tag"] for run in PRIORITY}

# Seed 2 at both budgets, minus the two runs step 0 promoted. Not in the default queue.
DEFERRED = [run for run in (block_a(arm, n_sims, 2, step=4)
                            for n_sims in (20, 50) for arm in SIX_ARMS)
            if run["tag"] not in PROMOTED]

# "Order -- any prefix is submittable". C1, C2, C4b and C5 need code or a pilot first and have
# no frozen config, so they are not here.
RUN_ORDER = (
    [block_a(arm, 20, 1, step=1) for arm in SIX_ARMS]
    + [block_a(arm, 50, 1, step=2) for arm in SIX_ARMS]
    + PRIORITY
    + [B1, B2]
    + [qwen("NoEmo"), qwen("Bias"), qwen("ActPool")]
    + [run for run in (block_a(arm, n_sims, 3, step=6)
                       for n_sims in (20, 50) for arm in ("NoEmo", "Bias"))
       if run["tag"] not in PROMOTED]
    + [run for run in [third_seed("Momentum"), third_seed("ActPool"), third_seed("AffPool")]
       if run["tag"] not in PROMOTED]
    + [C4]
)


# ---------------------------------------------------------------------------------------------
# Checks before any GPU time is spent
# ---------------------------------------------------------------------------------------------

def args_as_dict(argv):
    """['--seed', '1', '--p4g_persona', ...] -> {'seed': '1', 'p4g_persona': True, ...}"""
    out, i = {}, 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[key] = argv[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def same_value(ours, frozen):
    if isinstance(frozen, bool):
        return ours is frozen
    if isinstance(frozen, (int, float)):
        return float(ours) == float(frozen)
    return str(ours) == str(frozen)


def frozen_config_problems(run):
    """Differences between the plan's arguments for this run and its frozen config."""
    path = os.path.join(FROZEN_DIR, run["tag"] + ".json")
    if not os.path.exists(path):
        return [f"no frozen config at {path}"]
    frozen = json.load(open(path))["frozen"]
    ours = args_as_dict(run["argv"])
    problems = []
    for key, want in sorted(frozen.items()):
        have = ours.get(key, False if isinstance(want, bool) else None)
        if have is None:
            continue    # a runner default the frozen file pins; the runner checks it at start
        if not same_value(have, want):
            problems.append(f"--{key}: plan says {have!r}, frozen config says {want!r}")
    return problems


def served_model():
    """The model the SGLang server is serving, or None if it cannot be reached."""
    try:
        with urllib.request.urlopen(SGLANG_HOST.rstrip("/") + "/v1/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------------------------

def record_path(tag):
    return os.path.join(RUNS_DIR, tag, "run_record.json")


def load_record(tag):
    try:
        with open(record_path(tag)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_record(record):
    os.makedirs(os.path.dirname(record_path(record["tag"])), exist_ok=True)
    tmp = record_path(record["tag"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, record_path(record["tag"]))     # never leave a half-written record behind


def is_done(tag):
    record = load_record(tag)
    return record is not None and record["status"] == "done"


def archive_partial_run(tag):
    """Move an unfinished run's folder aside so the re-run starts from an empty folder."""
    src = os.path.join(RUNS_DIR, tag)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(RUNS_DIR, "_interrupted", f"{tag}_{stamp}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"  moved the unfinished run's files to {os.path.relpath(dst, REPO)}")


def git_state():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "src_has_uncommitted_changes": bool(dirty)}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def compute_results(tag, wall_seconds):
    sys.path.insert(0, SRC)
    from metrics.dialog_metrics import compute_metrics

    # the runner turns --output runs/<tag>/<tag>.pkl into runs/<tag>/<tag>/<tag>.pkl
    pkl = os.path.join(RUNS_DIR, tag, tag, tag + ".pkl")
    if not os.path.exists(pkl):
        return {"n": 0, "episodes_file": None}
    with open(pkl, "rb") as f:
        episodes = pickle.load(f)
    m = compute_metrics(episodes, task="p4g", max_turns=10)
    n = m["n"]
    n_success = round(m["SR"] * n)
    return {
        "n": n,
        "SR": round(m["SR"], 4),
        "SR_95CI_wilson": wilson_ci(n_success, n),
        "AvgT": round(m["AT"], 3),
        "wall_hours": round(wall_seconds / 3600, 2),
        "seconds_per_dialogue": round(wall_seconds / n, 1) if n else None,
        "episodes_file": os.path.relpath(pkl, REPO),
    }


def print_results(record):
    r = record["results"]
    print()
    print("=" * 100)
    print(f"  {record['tag']}   [{record['status'].upper()}]")
    print(f"  {record['description']}")
    print(f"  {record['plan_ref']}")
    if r.get("n"):
        lo, hi = r["SR_95CI_wilson"]
        ratio = r["wall_hours"] / record["expected_hours"]
        print(f"  SR {r['SR']:.2f} [{lo:.2f}, {hi:.2f}]   AvgT {r['AvgT']:.2f}   n={r['n']}   "
              f"wall {r['wall_hours']:.2f} h ({r['seconds_per_dialogue']:.0f} s/dialogue, "
              f"{ratio:.2f}x the plan's {record['expected_hours']} h)")
    print("=" * 100)


def print_summary_table():
    print()
    print(f"{'run':<26} {'step':>4} {'status':<12} {'n':>4} {'SR':>6} {'95% CI':>13} {'AvgT':>6} {'hours':>6}")
    for run in RUN_ORDER:
        record = load_record(run["tag"])
        if record is None:
            continue
        r = record.get("results") or {}
        if r.get("n"):
            ci = f"[{r['SR_95CI_wilson'][0]:.2f}, {r['SR_95CI_wilson'][1]:.2f}]"
            print(f"{run['tag']:<26} {run['order_step']:>4} {record['status']:<12} {r['n']:>4} "
                  f"{r['SR']:>6.2f} {ci:>13} {r['AvgT']:>6.2f} {r['wall_hours']:>6.2f}")
        else:
            print(f"{run['tag']:<26} {run['order_step']:>4} {record['status']:<12}")


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------

def run_one(run):
    """Run the runner for one grid cell, tee its output to runner.log, and return its exit code.

    A KeyboardInterrupt (Ctrl+C, or the SIGTERM handler below) propagates after the runner has
    been stopped, so the caller can mark the run as interrupted.
    """
    log_path = os.path.join(RUNS_DIR, run["tag"], "runner.log")
    command = [sys.executable, "runners/rollout.py"] + run["argv"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(command, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = os.read(proc.stdout.fileno(), 4096)   # chunks, not lines: keeps tqdm's bar live
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                log.write(chunk)
            return proc.wait()
        except KeyboardInterrupt:
            stop_runner(proc)
            raise


def stop_runner(proc):
    """Ctrl+C already reached the runner (same process group); give it time, then force it."""
    for stop in (lambda: proc.send_signal(signal.SIGINT), proc.terminate, proc.kill):
        stop()
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            pass


def sigterm_as_ctrl_c(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="skip runs already finished and re-run the one that was cut off")
    parser.add_argument("--steps", type=int, nargs="+", choices=sorted(ORDER_STEPS),
                        help="only these Order steps of the plan (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the runs and check them against the frozen configs; run nothing")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, sigterm_as_ctrl_c)

    # DEFERRED is out of the default queue but still selectable by its step number, so
    # `--steps 4` runs seed 2 exactly as the plan wrote it.
    if args.steps is None:
        runs = RUN_ORDER
    else:
        runs = [r for r in RUN_ORDER + DEFERRED if r["order_step"] in args.steps]

    # 1. Every run must match its frozen config -- found now, not twelve GPU-hours in.
    mismatches = {r["tag"]: frozen_config_problems(r) for r in runs}
    mismatches = {tag: p for tag, p in mismatches.items() if p}
    if mismatches:
        print("The plan and the frozen configs disagree -- nothing was started:\n")
        for tag, problems in mismatches.items():
            print(f"  {tag}")
            for p in problems:
                print(f"      {p}")
        print("\nFix the plan in this script or regenerate the configs (analysis/thu/scripts/gen_grid_configs.py).")
        sys.exit(1)

    if args.dry_run:
        for r in runs:
            state = load_record(r["tag"])["status"] if load_record(r["tag"]) else "not started"
            print(f"step {r['order_step']}  {r['tag']:<26} ~{r['expected_hours']:>4} h  [{state}]")
            print(f"        {r['description']}")
            print(f"        python3 runners/rollout.py {' '.join(r['argv'])}\n")
        print(f"{len(runs)} runs, ~{sum(r['expected_hours'] for r in runs):.0f} h; all match their frozen configs.")
        return

    # 2. A fresh start must not overwrite runs that already exist.
    existing = [r["tag"] for r in runs if load_record(r["tag"])]
    if existing and not args.resume:
        print(f"{len(existing)} run(s) already have results in {os.path.relpath(RUNS_DIR, REPO)} "
              f"(first: {existing[0]}). Use --continue to pick up where the grid stopped.")
        sys.exit(1)

    todo = [r for r in runs if not is_done(r["tag"])]
    print(f"{len(runs) - len(todo)} of {len(runs)} runs already finished; {len(todo)} to go "
          f"(~{sum(r['expected_hours'] for r in todo):.0f} h).")

    for i, run in enumerate(todo, 1):
        tag = run["tag"]
        print(f"\n##### [{i}/{len(todo)}] {tag} -- {run['description']}")

        # 3. The runner silently falls back to whatever model the server has, so check it here.
        model = served_model()
        if model != run["backbone"]:
            print(f"The SGLang server at {SGLANG_HOST} serves {model!r}; {tag} needs {run['backbone']!r}.\n"
                  f"Switch the server and carry on with:\n"
                  f"    MODEL={run['backbone']} scripts/run_grid.sh --continue")
            sys.exit(1)

        if load_record(tag):                        # cut off last time: start it again from scratch
            print(f"  {tag} did not finish last time ({load_record(tag)['status']}); running it again")
            archive_partial_run(tag)

        record = dict(run, status="running", started_at=now(), finished_at=None,
                      exit_code=None, git=git_state(), results={},
                      command="cd src && python3 runners/rollout.py " + " ".join(run["argv"]),
                      arguments=args_as_dict(run["argv"]))
        save_record(record)

        start = time.time()
        try:
            exit_code = run_one(run)
        except KeyboardInterrupt:
            record.update(status="interrupted", finished_at=now(),
                          results=compute_results(tag, time.time() - start))
            save_record(record)
            print(f"\nStopped during {tag}. Run again with --continue to redo it and carry on.")
            sys.exit(130)

        results = compute_results(tag, time.time() - start)
        finished = exit_code == 0 and results["n"] == N_DIALOGUES
        record.update(status="done" if finished else "failed", exit_code=exit_code,
                      finished_at=now(), results=results)
        save_record(record)
        print_results(record)
        print_summary_table()

        if not finished:
            print(f"\n{tag} did not finish: exit code {exit_code}, {results['n']}/{N_DIALOGUES} dialogues. "
                  f"See {os.path.relpath(os.path.join(RUNS_DIR, tag, 'runner.log'), REPO)}.\n"
                  f"Stopping here; --continue will run {tag} again.")
            sys.exit(1)

    print("\nAll selected runs are finished.")


if __name__ == "__main__":
    main()
