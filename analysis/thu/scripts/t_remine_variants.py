"""w(e) under turn-selection variants: does ANY weighting reveal a pre-decision emotion signal?

    python analysis/thu/scripts/t_remine_variants.py

Same mining code (mining_corpus.load_sequences + mine_emotion_donation_p4g.tally_per_turn_soft, soft, alpha 50,
unit-weighted base rate), annotated 300 dialogs. Variants:

  shipped         all turns, uniform            (reference: the deployed table's recipe, unit base)
  pre_uniform     pre-decision turns, uniform   (= --emo_valence_table predecision)
  pre_last        pre-decision, ONLY the last pre-decision turn
  pre_recency     pre-decision, gamma**k, gamma 0.7 (0.5 and 0.85 as sensitivity)

Per emotion: w, raw lift, weighted trials (the miner's n), effective number of DIALOGS
n_eff_dialogs = (sum_d T_d)^2 / sum_d T_d^2 over per-dialog weighted mass T_d, and intervals:
  wilson_units   the miner's Wilson interval, n = soft mass. It ignores that utterances in a dialog share
                 its outcome; in practice it comes out WIDER than the cluster bootstrap for most cells,
                 because soft mass is well below the utterance count
  cluster_boot   95 % percentile interval of the LIFT (p_e - base, base recomputed per resample) from 2000
                 bootstrap resamples of DIALOGS -- the primary interval
  cluster_bonf   the same at 1 - 0.05/7 (Bonferroni over the 7 emotions tested within a variant)
(A Kish n over soft units is not used: with masses < 1 it exceeds the weighted trials and is meaningless.)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
os.chdir(REPO)
import lib  # noqa: E402
from emotion_mining.mining_corpus import load_sequences  # noqa: E402
from emotion_mining.mine_emotion_donation_p4g import Tally, sum_tallies, tally_per_turn_soft  # noqa: E402

EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
ALPHA = 50.0
B = 2000
SEED = 20260917

VARIANTS = {
	"shipped_all_turns": dict(pre_decision_only=False, turn_weighting="uniform"),
	"pre_uniform": dict(pre_decision_only=True, turn_weighting="uniform"),
	"pre_last": dict(pre_decision_only=True, turn_weighting="last"),
	"pre_recency_0.7": dict(pre_decision_only=True, turn_weighting="recency", recency_gamma=0.7),
	"pre_recency_0.5": dict(pre_decision_only=True, turn_weighting="recency", recency_gamma=0.5),
	"pre_recency_0.85": dict(pre_decision_only=True, turn_weighting="recency", recency_gamma=0.85),
}


def wilson(p, n, z=1.96):
	if n <= 0:
		return [np.nan, np.nan]
	den = 1 + z * z / n
	c = (p + z * z / (2 * n)) / den
	h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
	return [float(c - h), float(c + h)]


def run(name, kw):
	seqs, meta = load_sequences("p4g", None, None, soft=True, **kw)
	tallies = tally_per_turn_soft(seqs)
	base = sum_tallies(tallies.values()).p_donate
	# per-dialog, per-emotion weighted mass (for Kish n and the cluster bootstrap)
	D = len(seqs)
	S = np.zeros((D, 7))
	T = np.zeros((D, 7))
	for d, seq in enumerate(seqs):
		for i, dist in enumerate(seq.distributions):
			w = seq.weights[i] if seq.weights else 1.0
			for j, e in enumerate(EMO7):
				m = float(dist.get(e, 0.0)) * w
				T[d, j] += m
				S[d, j] += m * seq.succeeded
	rng = np.random.default_rng(SEED)
	lifts = []
	for _ in range(B):
		idx = rng.integers(0, D, D)
		s, t = S[idx].sum(0), T[idx].sum(0)
		lifts.append(s / t - s.sum() / t.sum())
	lifts = np.array(lifts)
	rows = {}
	for j, e in enumerate(EMO7):
		tl = tallies[e]
		neff_d = float(T[:, j].sum() ** 2 / (T[:, j] ** 2).sum()) if T[:, j].any() else 0.0
		lo, hi = np.percentile(lifts[:, j], [2.5, 97.5])
		blo, bhi = np.percentile(lifts[:, j], [100 * 0.025 / 7, 100 - 100 * 0.025 / 7])
		wu = tl.wilson_ci()
		rows[e] = {"w": round(tl.weight(base, ALPHA), 2), "raw_lift": tl.p_donate - base, "p_donate": tl.p_donate,
				   "weighted_trials": tl.trials, "n_eff_dialogs": neff_d,
				   "wilson_units": list(wu), "wilson_units_excludes_base": not (wu[0] <= base <= wu[1]),
				   "cluster_boot_lift": [float(lo), float(hi)], "cluster_boot_excludes_0": bool(lo > 0 or hi < 0),
				   "cluster_bonf_lift": [float(blo), float(bhi)], "cluster_bonf_excludes_0": bool(blo > 0 or bhi < 0)}
	return {"base_rate_unit": base, "n_dialogs": D, "variant": kw, "emotions": rows}


def main():
	out = {}
	for name, kw in VARIANTS.items():
		out[name] = run(name, kw)
		r = out[name]
		print(f"\n== {name}  dialogs {r['n_dialogs']}  unit base {r['base_rate_unit']:.3f}")
		print(f"  {'emotion':10s} {'w':>6s} {'lift':>7s} {'trials':>7s} {'nEffDlg':>7s}  "
			  f"{'Wilson(units)':>16s} ex  {'cluster lift 95%':>18s} ex  {'cluster Bonf/7':>18s} ex")
		for e, x in r["emotions"].items():
			print(f"  {e:10s} {x['w']:+6.2f} {x['raw_lift']:+7.3f} {x['weighted_trials']:7.1f} {x['n_eff_dialogs']:7.1f}  "
				  f"[{x['wilson_units'][0]:.3f},{x['wilson_units'][1]:.3f}] {'Y' if x['wilson_units_excludes_base'] else '.'}   "
				  f"[{x['cluster_boot_lift'][0]:+.3f},{x['cluster_boot_lift'][1]:+.3f}] {'Y' if x['cluster_boot_excludes_0'] else '.'}   "
				  f"[{x['cluster_bonf_lift'][0]:+.3f},{x['cluster_bonf_lift'][1]:+.3f}] {'Y' if x['cluster_bonf_excludes_0'] else '.'}")
	lib.jdump(out, os.path.join(REPO, "analysis", "thu", "remine", "variants.json"))


if __name__ == "__main__":
	main()
