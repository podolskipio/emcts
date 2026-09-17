#!/usr/bin/env python3
"""Phase-1 calibration driver: run one cell, record what it cost.

One invocation = one row in ``calib/throughput.json``. Everything except the three axes the
plan varies (model, persona, workers) is pinned to the grid values below, so a row prices a
grid cell directly.

    python3 calib/run_calib.py --run 1 --arch 8b --workers 10 --tag run1_w10
    python3 calib/run_calib.py --run 2 --arch 8b --workers 24 --persona --tag run2
    python3 calib/run_calib.py --run 3 --arch 13b --workers 24 --persona --tag run3

Assumes the matching server is already up (``calib/serve.sh 8b``); it refuses to run against
a server that is serving a different checkpoint, because that is a silently wrong row.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import requests

REPO = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
CALIB = os.path.join(REPO, "calib")
THROUGHPUT = os.path.join(CALIB, "throughput.json")

MODELS = {
    "8b": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "13b": "TheBloke/vicuna-13B-v1.5-AWQ",
}

# --- the frozen grid configuration -------------------------------------------------------
# Anything here is IDENTICAL in every calibration run and in every grid cell. The depth cap
# is `max_conv_turns`, which build_agents wires to --max_turns (W5 fix 5), so Tmax and the
# search depth cap are the same number and there is no second knob to set.
# ERRATUM 2026-09-14: that is only true under --search_horizon episode. Under the legacy default,
# search ignores the -1.0 at the turn limit and expands past Tmax (analysis/phase1/SEARCH_HORIZON_BUG.md).
GRID = dict(
    game="emo_p4g",
    algo="emomcts",
    num_mcts_sims=50,      # n_sims
    llm_prior_topk=5,      # K
    max_turns=10,          # Tmax == depth cap (max_conv_turns)
    max_realizations=4,    # R
    beta_emo=0.7,          # beta
    Q_0=0.0,
    cpuct=1.0,
    emo_signal="level",
    emo_risk_lambda=0.0,
    emo_valence_table="soft",
    emotion_classifier="hf",
    logit_scoring="both",
    seed=0,
)

METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+|NaN)$")


def scrape(base):
    """{metric name: summed value} from the SGLang prometheus endpoint."""
    out = {}
    try:
        text = requests.get(f"{base}/metrics", timeout=10).text
    except Exception as e:
        print(f"  ! /metrics unreachable: {e}")
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        try:
            v = float(m.group("value"))
        except ValueError:
            continue
        out[m.group("name")] = out.get(m.group("name"), 0.0) + v
    return out


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return max(int(x.strip()) for x in out.strip().splitlines() if x.strip())
    except Exception:
        return 0


class Sampler(threading.Thread):
    """Poll nvidia-smi and the SGLang gauges while the run is in flight.

    Peak VRAM alone cannot answer 'are we KV-cache bound', because --mem-fraction-static
    preallocates the pool: nvidia-smi reads the same number at 10 and at 24 workers. The
    gauge that moves is token_usage (fraction of the KV pool in use), so both are sampled.
    """

    def __init__(self, base, period=2.0):
        super().__init__(daemon=True)
        self.base, self.period, self.stop_flag = base, period, threading.Event()
        self.peak_vram_mib = 0
        self.samples = []   # (t, token_usage, num_running_reqs, num_queue_reqs, gen_throughput, cache_hit_rate)

    def run(self):
        t0 = time.time()
        while not self.stop_flag.is_set():
            self.peak_vram_mib = max(self.peak_vram_mib, gpu_used_mib())
            g = scrape(self.base)
            if g:
                self.samples.append((
                    round(time.time() - t0, 1),
                    g.get("sglang:token_usage", 0.0),
                    g.get("sglang:num_running_reqs", 0.0),
                    g.get("sglang:num_queue_reqs", 0.0),
                    g.get("sglang:gen_throughput", 0.0),
                    g.get("sglang:cache_hit_rate", 0.0),
                ))
            self.stop_flag.wait(self.period)

    def summary(self):
        if not self.samples:
            return {}
        busy = [s for s in self.samples if s[2] > 0]           # only while requests are running
        col = lambda i, rows: [r[i] for r in rows] or [0.0]
        return {
            "peak_vram_gb": round(self.peak_vram_mib / 1024.0, 2),
            "peak_kv_token_usage": round(max(col(1, self.samples)), 3),
            "mean_kv_token_usage_busy": round(sum(col(1, busy)) / max(len(busy), 1), 3),
            "peak_running_reqs": int(max(col(2, self.samples))),
            "mean_running_reqs_busy": round(sum(col(2, busy)) / max(len(busy), 1), 1),
            "peak_queue_reqs": int(max(col(3, self.samples))),
            "mean_gen_throughput_busy": round(sum(col(4, busy)) / max(len(busy), 1), 1),
            "mean_gauge_cache_hit_rate_busy": round(sum(col(5, busy)) / max(len(busy), 1), 4),
            "n_samples": len(self.samples),
        }


def build_cmd(args, out_pkl):
    cmd = [
        sys.executable, os.path.join(REPO, "src", "runners", "rollout.py"),
        "--game", GRID["game"], "--algo", GRID["algo"],
        "--llm", "sglang", "--sglang_model", MODELS[args.arch],
        "--emotion_classifier", GRID["emotion_classifier"],
        "--num_mcts_sims", str(GRID["num_mcts_sims"]),
        "--llm_prior_topk", str(GRID["llm_prior_topk"]),
        "--max_realizations", str(GRID["max_realizations"]),
        "--beta_emo", str(GRID["beta_emo"]),
        "--Q_0", str(GRID["Q_0"]), "--cpuct", str(GRID["cpuct"]),
        "--emo_signal", GRID["emo_signal"],
        "--emo_risk_lambda", str(GRID["emo_risk_lambda"]),
        "--emo_valence_table", GRID["emo_valence_table"],
        "--logit_scoring", GRID["logit_scoring"],
        "--max_turns", str(GRID["max_turns"]),
        "--max_conv", str(args.dialogues),
        "--num_workers", str(args.workers),
        "--seed", str(GRID["seed"]),
        "--profile_roles",
        "--output", out_pkl,
    ]
    if args.persona:
        cmd.append("--p4g_persona")
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, required=True, help="calibration run number (1/2/3)")
    p.add_argument("--arch", choices=list(MODELS), required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--persona", action="store_true")
    p.add_argument("--dialogues", type=int, default=10)
    p.add_argument("--tag", required=True, help="run directory name under calib/runs/")
    p.add_argument("--host", default="http://127.0.0.1:30000")
    p.add_argument("--note", default="")
    args = p.parse_args()

    # refuse to price the wrong checkpoint
    info = requests.get(f"{args.host}/get_model_info", timeout=10).json()
    served = info.get("model_path", "")
    if os.path.basename(served.rstrip("/")) != os.path.basename(MODELS[args.arch]):
        sys.exit(f"server is serving {served!r}, not {MODELS[args.arch]!r}")

    out_pkl = os.path.join(CALIB, "runs", args.tag + ".pkl")
    log_path = os.path.join(CALIB, "logs", args.tag + ".log")
    cmd = build_cmd(args, out_pkl)
    print(" ".join(cmd))

    before = scrape(args.host)
    sampler = Sampler(args.host)
    sampler.start()
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=os.path.join(REPO, "src"), stdout=log,
                             stderr=subprocess.STDOUT)
    wall = time.time() - t0
    sampler.stop_flag.set()
    sampler.join(timeout=10)
    after = scrape(args.host)

    d = lambda k: after.get(k, 0.0) - before.get(k, 0.0)
    prompt_tokens = d("sglang:prompt_tokens_total")
    cached_tokens = d("sglang:cached_tokens_total")
    gen_tokens = d("sglang:generation_tokens_total")
    hit = cached_tokens / prompt_tokens if prompt_tokens else 0.0

    # role profile: calls/turn per role, and what the run actually generated
    # setup_output_dir re-points --output into runs/<tag>/<tag>.pkl
    run_dir = os.path.splitext(out_pkl)[0]
    real_pkl = os.path.join(run_dir, args.tag + ".pkl")
    prof_path = os.path.join(run_dir, args.tag + "_role_profile.json")
    prof = {}
    if os.path.exists(prof_path):
        with open(prof_path) as f:
            prof = json.load(f)
    roles = prof.get("roles", {})
    cpt = lambda r: round(roles.get(r, {}).get("calls_per_turn") or 0.0, 2)
    llm_roles = ["policy_prior", "value_estimator", "user_simulator", "system_utterance"]
    total_samples = sum(roles.get(r, {}).get("samples", 0) for r in llm_roles)
    total_out = sum(roles.get(r, {}).get("total_tokens_out", 0) for r in llm_roles)
    total_in = sum(roles.get(r, {}).get("total_tokens_in", 0) for r in llm_roles)
    total_calls = sum(roles.get(r, {}).get("calls", 0) for r in llm_roles)

    # episode-level outcome, so a row that got cheap by ending dialogues early is visible
    sr = at = turns = None
    try:
        import pickle
        sys.path.insert(0, os.path.join(REPO, "src"))   # the pickle references src/ modules
        with open(real_pkl, "rb") as f:
            eps = pickle.load(f)
        turns = sum(e["num_turns"] for e in eps)
        sr = round(sum(bool(e["success"]) for e in eps) / max(len(eps), 1), 3)
        at = round(sum(e["num_turns"] if e["success"] else GRID["max_turns"] for e in eps)
                   / max(len(eps), 1), 2)
    except Exception as e:
        print(f"  ! could not read episodes: {e}")

    rec = {
        "run": args.run,
        "tag": args.tag,
        "model": MODELS[args.arch],
        "arch": args.arch,
        "persona": bool(args.persona),
        "workers": args.workers,
        "wall_clock_s": round(wall, 1),
        "dialogues": args.dialogues,
        "generations_per_s": round(total_samples / wall, 3) if wall else 0.0,
        "cache_hit_rate": round(hit, 4),
        "peak_vram_gb": sampler.summary().get("peak_vram_gb", 0.0),
        "llm_calls_per_turn": {
            "value": cpt("value_estimator"),
            "prior": cpt("policy_prior"),
            "user_sim": cpt("user_simulator"),
            "system": cpt("system_utterance"),
        },
        "mean_tokens_generated": round(total_out / total_samples, 2) if total_samples else 0,
        # --- everything below is extra context, not part of the required schema ---
        "exit_code": rc,
        "realized_turns": prof.get("turns"),
        "episode_turns": turns,
        "SR": sr,
        "AT": at,
        "server_prompt_tokens": int(prompt_tokens),
        "server_cached_tokens": int(cached_tokens),
        "server_generation_tokens": int(gen_tokens),
        "client_tokens_in": total_in,
        "client_tokens_out": total_out,
        "client_llm_calls": total_calls,
        "client_llm_samples": total_samples,
        "output_tokens_per_s": round(total_out / wall, 1) if wall else 0.0,
        "gpu": sampler.summary(),
        "grid": GRID,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    }

    rows = []
    if os.path.exists(THROUGHPUT):
        with open(THROUGHPUT) as f:
            rows = json.load(f)
    rows = [r for r in rows if r.get("tag") != args.tag] + [rec]
    with open(THROUGHPUT, "w") as f:
        json.dump(rows, f, indent=1)

    print(json.dumps({k: rec[k] for k in (
        "run", "tag", "workers", "persona", "wall_clock_s", "generations_per_s",
        "cache_hit_rate", "peak_vram_gb", "llm_calls_per_turn", "mean_tokens_generated",
        "SR", "AT", "realized_turns", "exit_code")}, indent=1))
    with open(os.path.join(CALIB, "logs", args.tag + "_gauges.json"), "w") as f:
        json.dump({"summary": sampler.summary(), "samples": sampler.samples}, f)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
