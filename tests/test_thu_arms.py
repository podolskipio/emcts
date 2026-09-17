"""Acceptance tests for the Thursday arms. Same contract as tests/test_wed_arms.py: every new behaviour
sits behind a flag whose default is the shipped planner, and the first job is BIT-IDENTITY.

ActPool (--aff_pool_key act): AffPool with the affective bucket removed from the key.

  * --aff_pool off         -> NoEmo GOLDEN (the pre-change tree's fingerprints in test_wed_arms.py)
  * --aff_pool_key affect  -> AffPool as it was before --aff_pool_key existed. PRECHANGE_AFFPOOL was
                              produced from the working tree immediately before the flag landed
                              (analysis/thu/prechange_affpool_fingerprints.json).
  * --aff_pool_key act     -> one cell per act, pooled over every node and both buckets; RAVE blend,
                              scope, bias and diagnostics unchanged.

    python -m pytest tests/test_thu_arms.py -q
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_wed_arms as W  # noqa: E402  (also puts src/ on sys.path)

em = W.em
SYS_ACTS = W.SYS_ACTS

PRECHANGE_AFFPOOL = {
	(0.0, 0.05): "58f1af151f182ae9498441c677aeab7711124af707a0febd310ae5910e199dc6",
	(0.0, 0.1): "672f450435c30a7ae7e17c62a5f890c5b68c241ee0e270ad7c8b649e6e20dbdb",
	(0.0, 0.25): "6c236c9c2c7a14c56dc55e9f483525dd4e6f8e96c3aaaf65e300422bc79402f6",
	(0.7, 0.05): "5ddab4e17b715471ba54611043034b2639740ae590ae2e3610e9614ab770d55e",
	(0.7, 0.1): "c6f184f450656085dcd83914550e39e56d4fb6173ddcf95828c96760e6282a1e",
	(0.7, 0.25): "e0e976db2fc973c88b126d0b8746c0cb84b11d9fbc70a45684b36e5cd4d97554",
}


def full_fingerprint(p) -> str:
	"""W.fingerprint plus the pool tables and EVERY sim_steps field (the pool diagnostics too)."""
	def norm(x):
		if isinstance(x, float):
			return x.hex()
		if isinstance(x, dict):
			return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
		if isinstance(x, (list, tuple)):
			return [norm(v) for v in x]
		if hasattr(x, "item"):
			return norm(x.item())
		return x
	blob = {"base": W.fingerprint(p), "Q_pool": norm(p.Q_pool), "N_pool": norm(p.N_pool),
			"steps": [norm(s) for s in p.sim_steps]}
	return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# bit-identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_pool_off_is_noemo_golden(beta):
	p, _ = W.run_search(beta_emo=beta, aff_pool=False, aff_pool_key="act")  # key is inert when off
	assert W.fingerprint(p) == W.GOLDEN[f"beta{beta}"]
	assert p.Q_pool == {} and p.N_pool == {}


@pytest.mark.parametrize("beta,bias", sorted(PRECHANGE_AFFPOOL))
def test_affect_key_is_prechange_affpool(beta, bias):
	implicit, _ = W.run_search(beta_emo=beta, aff_pool=True, aff_pool_bias=bias)
	explicit, _ = W.run_search(beta_emo=beta, aff_pool=True, aff_pool_bias=bias, aff_pool_key="affect")
	assert W.fingerprint(implicit) == PRECHANGE_AFFPOOL[(beta, bias)]
	assert full_fingerprint(explicit) == full_fingerprint(implicit)


def test_constructor_default_key_is_affect():
	p, _ = W.run_search(sims=1)
	assert p.aff_pool_key == "affect"


def test_unknown_key_is_rejected():
	with pytest.raises(ValueError):
		W.run_search(sims=1, aff_pool=True, aff_pool_key="bucket")


# ---------------------------------------------------------------------------
# ActPool behaviour
# ---------------------------------------------------------------------------
def test_act_key_pools_every_step_under_one_cell_per_act():
	p, _ = W.run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1, aff_pool_key="act")
	cells = {}
	for step in p.sim_steps:
		assert step["aff_bucket"] == 0
		cells.setdefault(step["action"], []).append(step["backup_value"])
	assert {b for b, _ in p.N_pool} == {0}
	assert {SYS_ACTS[a] for _, a in p.N_pool} == set(cells)
	for act, vs in cells.items():
		key = (0, SYS_ACTS.index(act))
		assert p.N_pool[key] == len(vs)
		assert p.Q_pool[key] == pytest.approx(sum(vs) / len(vs), abs=1e-12)


def test_act_pool_is_the_union_of_the_affect_buckets_on_the_same_steps():
	"""Given identical steps, the act cell is exactly the merge of the two affect cells. Checked by
	re-keying one run's own log, so it holds regardless of how the two searches diverge."""
	p, _ = W.run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1)  # affect
	merged = {}
	for step in p.sim_steps:
		merged.setdefault(step["action"], []).append(step["backup_value"])
	for act, vs in merged.items():
		a = SYS_ACTS.index(act)
		n = sum(p.N_pool.get((b, a), 0) for b in (0, 1))
		q = sum(p.N_pool.get((b, a), 0) * p.Q_pool.get((b, a), 0.0) for b in (0, 1)) / n
		assert n == len(vs)
		assert q == pytest.approx(sum(vs) / len(vs), abs=1e-12)


def test_act_key_changes_the_search_and_keeps_affpool_diagnostics():
	aff, _ = W.run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1)
	act, _ = W.run_search(beta_emo=0.0, aff_pool=True, aff_pool_bias=0.1, aff_pool_key="act")
	assert full_fingerprint(act) != full_fingerprint(aff)
	for step in act.sim_steps:
		assert {"aff_bucket", "pool_beta_selected", "pool_flip"} <= set(step)
		assert all({"Q_pool", "N_pool", "pool_beta"} <= set(sib) for sib in step["siblings"])
	# the act cell sees at least as much evidence as either affect cell -- the point of the control
	for (b, a), n in aff.N_pool.items():
		assert act.N_pool.get((0, a), 0) > 0
	assert sum(act.N_pool.values()) == len(act.sim_steps)


def test_act_key_uses_same_blend_and_bias():
	for seed in range(50):
		p, s, acts = W.random_node_planner(seed)
		p.aff_pool_key = "act"
		assert p._pool_bucket(-0.9) == 0 and p._pool_bucket(0.9) == 0
		for a in acts:
			beta = em.rave_beta(p.Nsa[s][a], p.N_pool.get((0, a), 0), p.aff_pool_bias)
			q_eff = p.Q[s][a] if beta == 0 else (1 - beta) * p.Q[s][a] + beta * p.Q_pool[(0, a)]
			assert p._calculate_uct(s, a, bucket=0) == pytest.approx(
				W.prechange_uct(p, s, a) - p.Q[s][a] + q_eff, abs=1e-12)


# ---------------------------------------------------------------------------
# runner flags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("runner", ["rollout", "emomcts"])
def test_runner_aff_pool_key_flag(runner):
	import runners.rollout as rollout_runner
	import runners.emomcts as emomcts_runner
	mod = {"rollout": rollout_runner, "emomcts": emomcts_runner}[runner]
	src = open(mod.__file__).read()
	block = src[src.index('if __name__ == "__main__":'):]
	block = block.replace("cmd_args = finalize_args(parser.parse_args())", "ARGS = parser")
	block = block.split("ARGS = parser")[0] + "ARGS = parser"
	ns = dict(vars(mod))
	ns["__name__"] = "__main__"
	exec(compile(block, mod.__file__, "exec"), ns)
	parser = ns["ARGS"]
	assert parser.parse_args([]).aff_pool_key == "affect"
	assert parser.parse_args(["--aff_pool_key", "act"]).aff_pool_key == "act"
	assert parser.parse_args(["--aff-pool-key", "act"]).aff_pool_key == "act"
	with pytest.raises(SystemExit):
		parser.parse_args(["--aff_pool_key", "bucket"])
	# the runner must hand the key to the planner, not just parse it
	assert "aff_pool_key=" in src and '"aff_pool_key": cmd_args.aff_pool_key' in src


# ---------------------------------------------------------------------------
# --frozen_config: a run whose resolved config differs from the frozen values must not start
# ---------------------------------------------------------------------------
def _runner_parser(runner):
	import importlib
	mod = importlib.import_module(f"runners.{runner}")
	src = open(mod.__file__).read()
	block = src[src.index('if __name__ == "__main__":'):]
	block = block.replace("cmd_args = finalize_args(parser.parse_args())", "ARGS = parser")
	block = block.split("ARGS = parser")[0] + "ARGS = parser"
	ns = dict(vars(mod))
	ns["__name__"] = "__main__"
	exec(compile(block, mod.__file__, "exec"), ns)
	return ns["ARGS"], mod


GRID = ["--llm", "sglang", "--num_mcts_sims", "50", "--max_realizations", "4", "--llm_prior_topk", "5",
		"--emotion_classifier", "hf", "--seed", "0", "--search_horizon", "legacy"]
FROZEN = {"llm": "sglang", "num_mcts_sims": 50, "max_realizations": 4, "llm_prior_topk": 5,
		  "emotion_classifier": "hf", "seed": 0, "search_horizon": "legacy"}


def _frozen(tmp_path, values):
	p = tmp_path / "frozen.json"
	p.write_text(json.dumps({"frozen": values}))
	return str(p)


@pytest.mark.parametrize("runner", ["rollout", "emomcts"])
def test_frozen_config_is_off_by_default(runner, tmp_path):
	parser, mod = _runner_parser(runner)
	a = parser.parse_args(["--output", str(tmp_path / "o.pkl")])
	assert a.frozen_config is None
	out = mod.finalize_args(a)
	assert not hasattr(out, "frozen_config_sha256")


@pytest.mark.parametrize("runner", ["rollout", "emomcts"])
def test_frozen_config_passes_on_an_exact_match(runner, tmp_path):
	parser, mod = _runner_parser(runner)
	f = _frozen(tmp_path, FROZEN)
	a = mod.finalize_args(parser.parse_args(GRID + ["--output", str(tmp_path / "o.pkl"), "--frozen_config", f]))
	assert len(a.frozen_config_sha256) == 64


@pytest.mark.parametrize("runner", ["rollout", "emomcts"])
@pytest.mark.parametrize("drop,why", [
	("--emotion_classifier", "inherited llm classifier"),
	("--max_realizations", "inherited R=3"),
	("--num_mcts_sims", "inherited n_sims 20"),
	("--llm_prior_topk", "inherited no top-K prior"),
	("--seed", "inherited unseeded"),
])
def test_frozen_config_rejects_an_inherited_default(runner, drop, why, tmp_path):
	parser, mod = _runner_parser(runner)
	i = GRID.index(drop)
	argv = GRID[:i] + GRID[i + 2:]
	with pytest.raises(SystemExit, match=drop.lstrip("-")):
		mod.finalize_args(parser.parse_args(argv + ["--output", str(tmp_path / "o.pkl"),
													"--frozen_config", _frozen(tmp_path, FROZEN)]))


def test_frozen_config_rejects_unknown_keys_and_bool_int_confusion(tmp_path):
	parser, mod = _runner_parser("rollout")
	base = GRID + ["--output", str(tmp_path / "o.pkl")]
	with pytest.raises(SystemExit, match="not an argument"):
		mod.finalize_args(parser.parse_args(base + ["--frozen_config", _frozen(tmp_path, {**FROZEN, "n_sims": 50})]))
	with pytest.raises(SystemExit, match="aff_pool"):
		mod.finalize_args(parser.parse_args(base + ["--frozen_config", _frozen(tmp_path, {**FROZEN, "aff_pool": 0})]))
	with pytest.raises(SystemExit, match="search_horizon"):
		mod.finalize_args(parser.parse_args(base + ["--frozen_config",
													_frozen(tmp_path, {**FROZEN, "search_horizon": "episode"})]))


def test_every_runner_goes_through_the_check():
	import glob
	for path in glob.glob(os.path.join(os.path.dirname(__file__), "..", "src", "runners", "*.py")):
		src = open(path).read()
		if "add_common_args(parser" in src and not path.endswith("_common.py"):
			assert "cmd_args = finalize_args(parser.parse_args())" in src, path


# ---------------------------------------------------------------------------
# --emo_valence_table predecision: w(e) re-mined on pre-decision turns
# ---------------------------------------------------------------------------
def test_predecision_table_is_the_mined_output():
	path = os.path.join(os.path.dirname(__file__), "..", "analysis", "thu", "remine",
						"predecision_soft_all300_unit.json")
	mined = json.load(open(path))["valence_weights"]
	table = {str(getattr(k, "value", k)).lower(): v for k, v in em.EMOTION_VALENCE_TABLES["predecision"].items()}
	assert table == mined


def test_default_table_unchanged_by_predecision():
	assert em.EMOTION_VALENCE_TABLES["soft"] is em.EMOTION_VALENCE_MINED
	p, _ = W.run_search(beta_emo=0.7)
	assert W.fingerprint(p) == W.GOLDEN["beta0.7"]


def test_predecision_table_changes_the_search_and_parses():
	base, _ = W.run_search(beta_emo=0.7)
	pre, _ = W.run_search(beta_emo=0.7, emo_valence_table="predecision")
	assert pre.valence_weights is em.EMOTION_VALENCE_PREDECISION
	assert W.fingerprint(pre) != W.fingerprint(base)
	for runner in ("rollout", "emomcts"):
		parser, _ = _runner_parser(runner)
		assert parser.parse_args(["--emo_valence_table", "predecision"]).emo_valence_table == "predecision"


# ---------------------------------------------------------------------------
# --p4g_success: stricter success detection, in the environment (ends episodes AND search branches)
# ---------------------------------------------------------------------------
from games import PersuasionGame as _PG  # noqa: E402
from utils.sessions import DialogSession as _DS  # noqa: E402


def _session(turns):
	s = _DS(_PG.SYS, _PG.USR)
	for i, (da, utt) in enumerate(turns):
		s.add_single(_PG.SYS if i % 2 == 0 else _PG.USR, da, utt)
	return s


def _game(criterion=None):
	kw = {} if criterion is None else {"success_criterion": criterion}
	return _PG(None, None, None, False, max_conv_turns=10, **kw)


HEDGED = [("greeting", "hi"), ("neutral", "hello"), ("proposition of donation", "would you give?"),
		  ("donate", "I will consider donating and will be in touch with you soon.")]
COMMITTED_NO_AMOUNT = HEDGED[:3] + [("donate", "Yes, I will donate to Save the Children.")]
WITH_AMOUNT = HEDGED[:3] + [("donate", "Alright, I'll donate $2.")]
AMOUNT_IN_LEAD_IN = [("greeting", "hi"), ("positive reaction", "I could give 50 cents."),
					 ("proposition of donation", "great?"), ("donate", "Yes, let's do it.")]


def test_default_success_is_the_tag_alone():
	g = _game()
	assert g.success_criterion == "tag"
	for turns in (HEDGED, COMMITTED_NO_AMOUNT, WITH_AMOUNT, AMOUNT_IN_LEAD_IN):
		assert g.get_dialog_ended(_session(turns)) == 1.0


@pytest.mark.parametrize("criterion,expected", [
	("committed", {"hedged": 0.0, "committed": 1.0, "amount": 1.0, "lead_in": 1.0}),
	("amount", {"hedged": 0.0, "committed": 0.0, "amount": 1.0, "lead_in": 1.0}),
])
def test_strict_success_criteria(criterion, expected):
	g = _game(criterion)
	got = {"hedged": g.get_dialog_ended(_session(HEDGED)),
		   "committed": g.get_dialog_ended(_session(COMMITTED_NO_AMOUNT)),
		   "amount": g.get_dialog_ended(_session(WITH_AMOUNT)),
		   "lead_in": g.get_dialog_ended(_session(AMOUNT_IN_LEAD_IN))}
	assert got == expected


def test_a_rejected_donate_does_not_block_a_later_one_and_limit_still_fails():
	g = _game("committed")
	later = HEDGED + [("emotion appeal", "it matters"), ("donate", "OK, I will donate $1 now.")]
	assert g.get_dialog_ended(_session(later)) == 1.0
	long = HEDGED + [("other", "x"), ("neutral", "y")] * 8
	assert g.get_dialog_ended(_session(long)) == -1.0


def test_unknown_success_criterion_rejected():
	with pytest.raises(ValueError):
		_game("strict")


def test_environment_agrees_with_offline_scorer_on_logged_pilots():
	import glob as _glob, pickle as _pickle
	root = os.path.join(os.path.dirname(__file__), "..", "analysis", "thu")
	rows = {}
	for line in open(os.path.join(root, "success_criteria_pilots.tsv")).read().splitlines()[1:]:
		run, did, lenient, committed, amount = line.split("\t")[:5]
		rows[(run, did)] = (lenient == "True", committed == "True", amount == "True")
	tags = {"legacy": "P1_legacy", "noemo": "P1_episode", "bias": "E_bias_b0.7", "momentum": "E_momentum_b1.2",
			"centre": "E_centre_b1.1", "actpool": "E_actpool", "affpool": "E_affpool"}
	checked = 0
	for tag, d in tags.items():
		pk = _glob.glob(os.path.join(root, "runs", d, d, "*.pkl"))
		if not pk:
			pytest.skip("pilot runs not present")
		for ep in _pickle.load(open(pk[0], "rb")):
			s = _DS(_PG.SYS, _PG.USR, [[h[0], h[1], h[-1]] for h in ep["history"]])
			lenient, committed, amount = rows[(tag, ep["did"])]
			if not lenient:
				continue  # a failed episode has no [donate]; nothing to disagree on
			assert (_game("tag").get_dialog_ended(s) == 1.0) == lenient
			assert (_game("committed").get_dialog_ended(s) == 1.0) == committed, (tag, ep["did"])
			assert (_game("amount").get_dialog_ended(s) == 1.0) == amount, (tag, ep["did"])
			checked += 1
	assert checked >= 50


def test_rollout_success_flag_and_build_agents_guard():
	parser, mod = _runner_parser("rollout")
	assert parser.parse_args([]).p4g_success == "tag"
	assert parser.parse_args(["--p4g_success", "committed"]).p4g_success == "committed"
	src = open(mod.__file__).read()
	assert "success_criterion=cmd_args.p4g_success" in src
	import runners._common as common
	with pytest.raises(ValueError, match="defined for p4g"):
		common.build_agents("esc", None, "chat", success_criterion="committed")


def test_inert_arm_flags_are_refused():
	parser, mod = _runner_parser("rollout")
	ok = ["--aff_pool", "--aff_pool_bias", "0.25", "--aff_pool_tau", "0.263", "--aff_pool_key", "affect"]
	mod.finalize_args(parser.parse_args(ok + ["--output", "/tmp/x/o.pkl"]), argv=ok)
	actpool_tau = ["--aff_pool", "--aff_pool_bias", "0.25", "--aff_pool_key", "act", "--aff_pool_tau", "0.263"]
	with pytest.raises(SystemExit, match="aff_pool_tau"):
		mod.finalize_args(parser.parse_args(actpool_tau), argv=actpool_tau)
	no_pool = ["--aff_pool_bias", "0.25"]
	with pytest.raises(SystemExit, match="need --aff_pool"):
		mod.finalize_args(parser.parse_args(no_pool), argv=no_pool)
	eq_form = ["--aff_pool", "--aff_pool_key=act", "--aff-pool-tau=0.3"]
	with pytest.raises(SystemExit, match="aff_pool_tau"):
		mod.finalize_args(parser.parse_args(eq_form), argv=eq_form)
	noemo = ["--beta_emo", "0", "--emo_signal", "level"]  # the pilot harness passes this; must stay legal
	mod.finalize_args(parser.parse_args(noemo + ["--output", "/tmp/x/o.pkl"]), argv=noemo)
