"""Run the readiness program's real-tree runs, one after another, with the grid runner's own machinery.

    python analysis/readiness/scripts/run_readiness.py --phase 2 [--dry_run] [--continue]

Everything but the destination is scripts/run_grid.py: the same shared arguments (the frozen grid's), the
same arm arguments, the same run record (git commit + dirty flag, argv, status, SR/AvgT), the same
runner.log. Runs go to analysis/readiness/runs/<tag>/ -- never analysis/grid/. No --frozen_config: these
runs differ from the frozen cells on purpose (the cache flags, and Phase 2's dialogue count), and each
record carries its full argv instead.

A run refuses to start if src/ has uncommitted changes: the record's commit must be the code that ran.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import run_grid as rg  # noqa: E402

rg.RUNS_DIR = os.path.join(REPO, "analysis", "readiness", "runs")

FIXES_ALL = ["--cache_ended_children", "--cache_fresh_depth1"]
DRAW = ["--cache_draw", "bucket_kernel", "--cache_bucket_tau", "0.263", "--cache_kernel_h", "0.2"]


def make(tag, phase, arm, extra, n_dialogues, n_sims=20, seed=1, description="", hours=1.0):
	shared = rg.set_arg(rg.SHARED_ARGS, "--max_conv", str(n_dialogues))
	argv = (shared + ["--sglang_model", rg.VICUNA, "--num_mcts_sims", str(n_sims), "--seed", str(seed)]
			+ rg.ARM_ARGS[arm] + extra
			+ ["--output", os.path.join(rg.RUNS_DIR, tag, tag + ".pkl")])
	return {"tag": tag, "description": description, "plan_ref": f"readiness brief v2, Phase {phase}",
			"phase": phase, "arm": arm, "backbone": rg.VICUNA, "num_mcts_sims": n_sims, "seed": seed,
			"n_dialogues": n_dialogues, "expected_hours": hours, "argv": argv}


COUPLING_DIR = os.path.join(REPO, "analysis", "readiness", "coupling")


def coupled(store):
	return ["--coupled_seeds", "--coupling_store", os.path.join(COUPLING_DIR, store + ".sqlite")]


# Arm fixes from Phase 3 on: NoEmo stays affect-free (#6, #5); the pooling arms add the #1+#2 draw, which
# reads the parent's nu (brief v2 Phase 5: "AffPool/ActPool ± all fixes").
NOEMO_FIXES = FIXES_ALL
POOL_FIXES = FIXES_ALL + DRAW

PHASES = {
	2: [
		make("P2_NoEmo_fixes_n25", 2, "NoEmo", FIXES_ALL, 25, hours=1.0,
			 description="Phase 2 mechanical verification: NoEmo with #6 retention and #5 fresh depth-1. "
						 "Decision-level only, no SR."),
		make("P2_AffPool_fixes_n25", 2, "AffPool", FIXES_ALL + DRAW, 25, hours=1.2,
			 description="Phase 2 mechanical verification: AffPool with #6, #5 and the #1+#2 bucket_kernel "
						 "draw (tau 0.263, h 0.2). Decision-level only, no SR."),
	],
	# 3A acceptance (PREREG Entry 10): same arm + seed twice -> identical dialogues; NoEmo vs ActPool on the
	# same store -> identical up to the first differing act. Run in this order.
	3: [
		make("P3A_NoEmo_a", 3, "NoEmo", NOEMO_FIXES + coupled("p3a_seed1"), 5, hours=0.3,
			 description="3A acceptance: NoEmo, coupled, first run (records the store)."),
		make("P3A_NoEmo_b", 3, "NoEmo", NOEMO_FIXES + coupled("p3a_seed1"), 5, hours=0.3,
			 description="3A acceptance: NoEmo, coupled, identical re-run on the same store and seed."),
		make("P3A_ActPool", 3, "ActPool", POOL_FIXES + coupled("p3a_seed1"), 5, hours=0.3,
			 description="3A acceptance: ActPool on the same store and seed as P3A_NoEmo_a."),
	],
}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
	ap.add_argument("--dry_run", action="store_true")
	ap.add_argument("--continue", dest="resume", action="store_true")
	a = ap.parse_args()
	runs = PHASES[a.phase]
	if a.dry_run:
		for r in runs:
			print(f"{r['tag']}  ~{r['expected_hours']} h\n    cd src && python3 runners/rollout.py {' '.join(r['argv'])}\n")
		return
	git = rg.git_state()
	if git["src_has_uncommitted_changes"]:
		sys.exit("src/ has uncommitted changes; commit before any real-tree run (readiness rule 3).")
	for r in runs:
		tag = r["tag"]
		if rg.is_done(tag):
			print(f"{tag} already done")
			continue
		if rg.load_record(tag):
			if not a.resume:
				sys.exit(f"{tag} has a record already; use --continue")
			rg.archive_partial_run(tag)
		if rg.served_model() != r["backbone"]:
			sys.exit(f"SGLang is not serving {r['backbone']}")
		record = dict(r, status="running", started_at=rg.now(), finished_at=None, exit_code=None, git=rg.git_state(),
					  results={}, command="cd src && python3 runners/rollout.py " + " ".join(r["argv"]),
					  arguments=rg.args_as_dict(r["argv"]))
		rg.save_record(record)
		start = time.time()
		code = rg.run_one(r)
		results = rg.compute_results(tag, time.time() - start)
		ok = code == 0 and results.get("n") == r["n_dialogues"]
		record.update(status="done" if ok else "failed", exit_code=code, finished_at=rg.now(), results=results)
		rg.save_record(record)
		print(f"{tag}: {record['status']}  n={results.get('n')}  wall {results.get('wall_hours')} h")
		if not ok:
			sys.exit(1)


if __name__ == "__main__":
	main()
