"""Generate every grid config from ONE frozen template (plan §4c, Thu Sep 17 revision). Nothing hand-edited.

    python analysis/thu/scripts/gen_grid_configs.py --p4g_success tag --out analysis/grid/DRAFT

For each run: the full argv (every shared value written out, no runner default relied on) and a
`--frozen_config` JSON holding every value that must hold -- shared block plus the arm's own settings.
Each config is then VALIDATED without running anything: parsed by runners/rollout.py's real argparse
block, passed through finalize_args (inert-arm-flag check + frozen-config check). A config that fails
is not written.

`--p4g_success` is required so the criterion is always an explicit choice (PREREG Entry 6: tag).
Primary budget n_sims 50 (PREREG Entry 6) sets B1, B2, B3-B5 and C4; C3 is n_sims 20 by the plan.
"""
import argparse
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

VICUNA = "TheBloke/vicuna-13B-v1.5-AWQ"
QWEN = "Qwen/Qwen2.5-7B-Instruct-AWQ"
EVAL = os.path.join(REPO, "data", "p4g", "rollout_evalset_nonannotated.jsonl")

# ---- the frozen template: shared block (PREREG Entry 5, FREEZE_NOTES §10.1) ------------------------
SHARED = {
	"game": "emo_p4g", "algo": "emomcts", "llm": "sglang", "sglang_model": VICUNA,
	"data": EVAL, "max_conv": 100,                      # runner default 20
	"search_horizon": "episode", "max_turns": 10,
	"max_realizations": 4,                              # runner default 3
	"llm_prior_topk": 5,                                # runner default None
	"emotion_classifier": "hf",                         # runner default llm
	"emo_valence_table": "generic",
	"cpuct": 1.0, "Q_0": 0.0, "emo_risk_lambda": 0.0,
	"logit_scoring": "off", "explicit_value_labels": False,
	"p4g_persona": True, "num_workers": 10,
}
FLAG = {"Q_0": "--Q_0", "max_conv": "--max_conv"}  # dest -> spelling where it is not --<dest>

# ---- arms: ONLY what differs between arms (plan §4c Block A) ---------------------------------------
ARMS = {
	"NoEmo": {"beta_emo": 0.0},
	"Bias": {"beta_emo": 0.70, "emo_signal": "level"},
	"Momentum": {"beta_emo": 1.11, "emo_signal": "delta"},
	"CenteredBias": {"beta_emo": 1.03, "emo_signal": "level", "emo_centre": True},
	"AffPool": {"beta_emo": 0.0, "aff_pool": True, "aff_pool_bias": 0.25, "aff_pool_tau": 0.263, "aff_pool_key": "affect"},
	"ActPool": {"beta_emo": 0.0, "aff_pool": True, "aff_pool_bias": 0.25, "aff_pool_key": "act"},
}
# resolved values the frozen check asserts for arms that do NOT pass a flag (so a stray flag is caught)
ARM_OFF = {"emo_signal": "level", "emo_centre": False, "aff_pool": False, "aff_pool_key": "affect"}


def runs():
	out = []
	A_SEEDS = {"NoEmo": 3, "Bias": 3, "Momentum": 2, "CenteredBias": 2, "AffPool": 2, "ActPool": 2}
	for sims in (20, 50):
		for arm, n in A_SEEDS.items():
			for seed in range(1, n + 1):
				out.append((f"A_{arm}_s{sims}_seed{seed}", arm, {"num_mcts_sims": sims, "seed": seed}, {}))
	primary = 50  # PREREG Entry 6: n_sims 50 primary, 20 the contrast
	# B1 is GDP-Zero's own OpenLoopMCTS on the plain p4g game: with top-K off, emomcts beta 0 breaks
	# exact PUCT ties differently at 0.27 % of selections (analysis/thu/b1_diagnosis.md)
	out.append(("B1_GDPZero_plain", "NoEmo", {"num_mcts_sims": primary, "seed": 1},
				{"llm_prior_topk": 0, "algo": "gdpzero", "game": "p4g"}))
	out.append(("B2_legacy_horizon", "NoEmo", {"num_mcts_sims": primary, "seed": 1}, {"search_horizon": "legacy"}))
	for arm in ("NoEmo", "Bias", "ActPool"):
		out.append((f"B_Qwen_{arm}", arm, {"num_mcts_sims": primary, "seed": 1}, {"sglang_model": QWEN}))
	for arm in ("Momentum", "ActPool", "AffPool"):
		out.append((f"C3_{arm}_s20_seed3", arm, {"num_mcts_sims": 20, "seed": 3}, {}))
	out.append(("C4_Bias_predecision", "Bias", {"num_mcts_sims": primary, "seed": 1}, {"emo_valence_table": "predecision"}))
	return out


def argv_for(values):
	argv = []
	for dest, v in values.items():
		flag = FLAG.get(dest, f"--{dest}")
		if isinstance(v, bool):
			if v:
				argv.append(flag)
		else:
			argv += [flag, str(v)]
	return argv


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--p4g_success", required=True, choices=["tag", "committed", "amount"])
	ap.add_argument("--out", required=True)
	a = ap.parse_args()
	os.makedirs(os.path.join(a.out, "frozen"), exist_ok=True)

	import runners.rollout as mod
	src = open(mod.__file__).read()
	block = src[src.index('if __name__ == "__main__":'):]
	block = block.replace("cmd_args = finalize_args(parser.parse_args())", "ARGS = parser").split("ARGS = parser")[0] + "ARGS = parser"
	ns = dict(vars(mod))
	ns["__name__"] = "__main__"
	exec(compile(block, mod.__file__, "exec"), ns)
	parser = ns["ARGS"]

	cmds, manifest = [], []
	for tag, arm, run_vals, overrides in runs():
		values = {**SHARED, **run_vals, **ARMS[arm], **overrides, "p4g_success": a.p4g_success}
		frozen = {**ARM_OFF, **values}
		out_pkl = os.path.join(REPO, "analysis", "grid", "runs", tag, f"{tag}.pkl")
		fpath = os.path.abspath(os.path.join(a.out, "frozen", f"{tag}.json"))
		json.dump({"frozen": frozen, "tag": tag, "arm": arm}, open(fpath, "w"), indent=1, sort_keys=True)
		argv = argv_for(values) + ["--output", out_pkl, "--frozen_config", fpath]
		# validate exactly as the runner would, without running
		check = list(argv)
		check[check.index("--output") + 1] = os.path.join(a.out, "_validate", tag, "o.pkl")  # finalize_args makes the dir
		cmd_args = parser.parse_args(check)
		mod.finalize_args(cmd_args, argv=check)
		cmds.append(f"# {tag}\n( cd {shlex.quote(os.path.join(REPO, 'src'))} && python3 runners/rollout.py "
					f"{' '.join(shlex.quote(x) for x in argv)} )")
		manifest.append({"tag": tag, "arm": arm, "argv": argv, "frozen_config": fpath,
						 "backbone": values["sglang_model"], "needs_server": values["sglang_model"]})
	import shutil
	shutil.rmtree(os.path.join(a.out, "_validate"), ignore_errors=True)
	with open(os.path.join(a.out, "commands.sh"), "w") as f:
		f.write("#!/usr/bin/env bash\n# GENERATED by analysis/thu/scripts/gen_grid_configs.py -- do not edit.\n"
				f"# p4g_success={a.p4g_success}. Qwen runs (B_Qwen_*) need the server restarted on {QWEN}.\nset -euo pipefail\n\n")
		f.write("\n".join(cmds) + "\n")
	json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
	print(f"{len(manifest)} configs generated and validated -> {a.out}")


if __name__ == "__main__":
	main()
