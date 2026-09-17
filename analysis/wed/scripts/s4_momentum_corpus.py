"""§4 -- the momentum hypothesis on human data (Table C only).

    python analysis/wed/scripts/s4_momentum_corpus.py

H: the direction of the user's affective change (delta_nu) modulates WHICH act works -- an
act x delta_nu interaction on donation, not a main effect of delta_nu.

Samples (every one reported; none averaged):
  primary    test100, w(e) re-mined on the other 197 (not fitted on these outcomes)
  pre        primary, turns up to the first 'agree-donation' (post-agreement turns trivially "succeed")
  single     primary, turns carrying exactly one planner act
  generic297 all 297 dialogues, generic hand-signed valence (not fitted on outcomes -> no circularity)
  soft297    all 297, deployed all-300 table (CIRCULAR on outcomes; reported for completeness only)
  last       the last system turn per dialogue (one outcome per row), test100/w200 and generic297

Decision rule, fixed before looking (calibration in lib.fit_models): H is supported in a sample only if
  (i) the F-corrected clustered Wald test of act:delta_nu has p < 0.05 WITH turn_index controlled, and
  (ii) the focused difference-in-differences DiD = [rate(propose, rising) - rate(propose, falling)]
       - [rate(emotion appeal, rising) - rate(emotion appeal, falling)] has a cluster-bootstrap CI
       above 0, and
  (iii) the level-nu version of (i) does not show the same interaction (else it is level, not direction).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import WED, Boot, ci, fit_models, update_wednesday, wilson, jdump  # noqa: E402

PROPOSE, EMO, CRED = "proposition of donation", "emotion appeal", "credibility appeal"
BUCKETS = ["falling", "flat", "rising"]
MIN_N = 30


def prep(C, table):
	d = C.dropna(subset=[f"delta_nu_{table}"]).copy()
	d["dnu"] = d[f"delta_nu_{table}"]
	d["nu"] = d[f"user_nu_{table}"]
	t = d.drop_duplicates(["dialogue_id", "turn_index"])
	cuts = [float(t["dnu"].quantile(1 / 3)), float(t["dnu"].quantile(2 / 3))]
	d["bucket"] = np.where(d["dnu"] < cuts[0], "falling", np.where(d["dnu"] <= cuts[1], "flat", "rising"))
	for c in ("dnu", "nu"):
		d[f"{c}_z"] = (d[c] - t[c].mean()) / t[c].std()
	d["turn_z"] = (d["turn_index"] - t["turn_index"].mean()) / t["turn_index"].std()
	d["act"] = d["system_act"]
	return d, cuts


def occupancy(d, cuts):
	t = d.drop_duplicates(["dialogue_id", "turn_index"])
	return {"cuts": cuts, "turns": len(t), "rows": len(d), "dialogues": int(d["dialogue_id"].nunique()),
			"bucket_share": t["bucket"].value_counts(normalize=True).reindex(BUCKETS).to_dict(),
			"share_dnu_negative": float((t["dnu"] < 0).mean()), "share_dnu_below_-0.1": float((t["dnu"] < -0.1).mean()),
			"share_dnu_exact_zero": float((t["dnu"] == 0).mean()),
			"donation_rate_by_bucket": t.groupby("bucket")["donated"].mean().reindex(BUCKETS).to_dict(),
			"turn_index_mean_by_bucket": t.groupby("bucket")["turn_index"].mean().reindex(BUCKETS).to_dict()}


def table(d):
	rows = []
	for (act, b), g in d.groupby(["act", "bucket"]):
		k, n = int(g["donated"].sum()), len(g)
		rows.append({"act": act, "bucket": b, "n": n, "rate": k / n, "wilson": wilson(k, n),
					 "n_dialogues": int(g["dialogue_id"].nunique()), "interpretable": n >= MIN_N})
	return rows


def did(d, a1=PROPOSE, a2=EMO):
	r = d.groupby(["act", "bucket"])["donated"].mean()
	try:
		return float((r[(a1, "rising")] - r[(a1, "falling")]) - (r[(a2, "rising")] - r[(a2, "falling")]))
	except KeyError:
		return np.nan


def act_slopes(res, acts, x):
	co = res["M2_coefficients"]
	base = co[x]["b"]
	return {a: base + (co[f"act[{a}]:{x}"]["b"] if a != acts[0] else 0.0) for a in acts}


def analyse(d, boot, label):
	out = {"occupancy": occupancy(d, d.attrs["cuts"]), "table": table(d)}
	# models use only acts with >= MIN_N rows and >= 10 in every tercile: a 1-row act (greeting, n = 1 in
	# test100) gets a separated coefficient with a spuriously tiny cluster SE and alone drove Wald p to 1e-14
	ct = pd.crosstab(d["act"], d["bucket"]).reindex(columns=BUCKETS, fill_value=0)
	keep = [a for a in d["act"].value_counts().index if ct.loc[a].sum() >= MIN_N and ct.loc[a].min() >= 10]
	out["model_acts"] = keep
	out["excluded_from_models"] = {a: int(ct.loc[a].sum()) for a in ct.index if a not in keep}
	d = d[d["act"].isin(keep)].copy()
	d.attrs["cuts"] = out["occupancy"]["cuts"]
	acts = keep
	models = {}
	for x in ("dnu_z", "nu_z"):
		for ctrl in ((), ("turn_z",)):
			key = f"{x}|{'turn' if ctrl else 'none'}"
			models[key] = fit_models(d, "donated", x, controls=ctrl, family="logit", acts=acts)
			models[key]["act_slopes"] = act_slopes(models[key], acts, x)
	# cluster bootstrap: increments, slopes, DiD
	reps = {k: {"inc01": [], "inc12": [], "slopes": []} for k in models}
	did_reps, did_rep_cred = [], []
	for bdf in boot.frame(d):
		did_reps.append(did(bdf))
		did_rep_cred.append(did(bdf, PROPOSE, CRED))
		for k in models:
			x, ctrl = k.split("|")
			try:
				r = fit_models(bdf, "donated", x, controls=(("turn_z",) if ctrl == "turn" else ()), family="logit", acts=acts)
			except Exception:
				continue
			reps[k]["inc01"].append(r["inc_M0_M1"])
			reps[k]["inc12"].append(r["inc_M1_M2"])
			reps[k]["slopes"].append(act_slopes(r, acts, x))
	for k, m in models.items():
		m["inc_M0_M1_ci"] = ci(reps[k]["inc01"])
		m["inc_M1_M2_ci"] = ci(reps[k]["inc12"])
		m["act_slope_ci"] = {a: ci([s[a] for s in reps[k]["slopes"]]) for a in acts}
		m["slope_contrast_propose_minus_emotion"] = {
			"value": m["act_slopes"].get(PROPOSE, np.nan) - m["act_slopes"].get(EMO, np.nan),
			"ci": ci([s.get(PROPOSE, np.nan) - s.get(EMO, np.nan) for s in reps[k]["slopes"]])}
		m.pop("M2_coefficients", None)
	out["models"] = models
	out["did_propose_minus_emotion"] = {"value": did(d), "ci": ci(did_reps)}
	out["did_propose_minus_credibility"] = {"value": did(d, PROPOSE, CRED), "ci": ci(did_rep_cred)}
	m = models["dnu_z|turn"]
	lvl = models["nu_z|turn"]
	out["verdict"] = {
		"i_interaction_wald_p_turn_controlled": m["interaction_wald_cluster"]["p"],
		"ii_did_ci_above_zero": bool(out["did_propose_minus_emotion"]["ci"][0] > 0),
		"iii_level_interaction_p": lvl["interaction_wald_cluster"]["p"],
		"i_lr_naive_p_turn_controlled": m["lr_M1_M2"]["p_naive"],
		"supported": bool(m["interaction_wald_cluster"]["p"] < 0.05 and out["did_propose_minus_emotion"]["ci"][0] > 0
						  and not lvl["interaction_wald_cluster"]["p"] < 0.05),
	}
	print(label, out["verdict"], "| inc12", round(m["inc_M1_M2"], 4), m["inc_M1_M2_ci"],
		  "| DiD", round(out["did_propose_minus_emotion"]["value"], 3), out["did_propose_minus_emotion"]["ci"])
	return out


def main():
	C = pd.read_parquet(os.path.join(WED, "corpus_turns.parquet"))
	samples = {}
	test = C[C["split"] == "test100"]
	d, cuts = prep(test, "w200")
	d.attrs["cuts"] = cuts
	samples["primary"] = d
	for name, sub in (("pre", d[d["pre_outcome"]]), ("single", d[d["n_acts_in_turn"] <= 1])):
		sub = sub.copy()
		sub.attrs["cuts"] = cuts
		samples[name] = sub
	for name, table_name in (("generic297", "generic"), ("soft297", "soft")):
		dd, cc = prep(C, table_name)
		dd.attrs["cuts"] = cc
		samples[name] = dd
	res = {}
	for name, d in samples.items():
		boot = Boot(d["dialogue_id"].unique())
		res[name] = analyse(d, boot, name)
	# last system act before the outcome: one row per dialogue (its acts exploded)
	for name, base in (("last_primary", samples["primary"]), ("last_generic297", samples["generic297"])):
		L = base[base["is_last_system_act"]].copy()
		L.attrs["cuts"] = base.attrs["cuts"]
		res[name] = {"occupancy": occupancy(L, L.attrs["cuts"]), "table": table(L),
					 "did_propose_minus_emotion": did(L), "note": "one outcome per dialogue; descriptive only"}
	res["definitions"] = __doc__
	jdump(res, os.path.join(WED, "s4_momentum_corpus.json"))
	update_wednesday("s4_momentum_corpus", res)


if __name__ == "__main__":
	main()
