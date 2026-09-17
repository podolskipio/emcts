"""TASK 2 -- construct test: does a lexical RESISTANCE signal predict donation better than EMOTION?

    python analysis/thu/scripts/t2_build_turns.py     # once: user turns + text + verified nu
    python analysis/thu/scripts/t2_construct_test.py

Design (every choice here is fixed before test100 is read):

  corpus    the 297 annotated P4G dialogues, analysis/thu/user_turns.parquet
  tune      mine200 (197). Marker lists were inspected here and ONLY here. w(e) -- and therefore
            nu -- was also mined on mine200, so on test100 both constructs are out-of-sample.
  test      test100 (100). The same held-out fold as the momentum test (build_corpus_turns.py).

  LEAKAGE CUT  features use only user turns STRICTLY BEFORE the dialogue's first explicit decision
            turn (agree-donation / disagree-donation / disagree-donation-more /
            provide-donation-amount / confirm-donation). Symmetric: donors lose "yes I'll give",
            refusers lose "no thanks". The annotated act labels locate the cut; they are never a
            predictor. The uncut ("full") version is reported beside it so the inflation shows.

  two r's   r_prior  the brief's a-priori lists, signs as theorised, zero tuning
            r_tuned  each family re-signed by a fixed rule on mine200 (below). Kept separate so a
                     win cannot be attributed to theory when it came from tuning.

  models    logit(donated) on test100
            M0  n_turns
            M1  M0 + nu_final + nu_mean          emotion
            M2  M0 + r_final  + r_mean           resistance   (same df as M1)
            M3  M0 + all four
  metrics   McFadden pseudo-R2 (full test100 fit); out-of-fold Brier and AUC (stratified 5-fold,
            repeated 20x); 95% CIs by bootstrap over dialogues (1000), PAIRED for M2 - M1.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402
import t2_markers as MK  # noqa: E402

THU = os.path.join(REPO, "analysis", "thu")
DECISION_ACTS = {"agree-donation", "disagree-donation", "disagree-donation-more",
				 "provide-donation-amount", "confirm-donation"}
FAMILIES = [f"res_{k}" for k in MK.RESISTANCE] + [f"rec_{k}" for k in MK.RECEPTIVITY]
TUNE_MIN_ABS_DIFF = 0.005  # a family whose donated-vs-not gap on mine200 is below this is dropped
B = 1000
CV_FOLDS, CV_REPEATS = 5, 20
RNG_SEED = 20260917


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def decision_turns():
	"""{dialogue_id: first user turn index carrying an explicit decision act} (absent = never)."""
	dialogs = pickle.load(open(os.path.join(REPO, "data", "p4g", "300_dialog_turn_based.pkl"), "rb"))
	out = {}
	for did, d in dialogs.items():
		for i, lab in enumerate(d["label"]):
			if DECISION_ACTS & set(lab.get("ee", []) if isinstance(lab, dict) else []):
				out[did] = i
				break
	return out


def tune_weights(u):
	"""Fixed rule on mine200 ONLY: weight = sign(mean hits/turn donated - not donated), 0 if |gap| is
	below TUNE_MIN_ABS_DIFF. Computed on pre-decision turns, the same rows the test uses."""
	m = u[(u["split"] == "mine200") & u["pre_decision"]]
	dl = m.groupby("dialogue_id").agg(**{f: (f, "mean") for f in FAMILIES}, donated=("donated", "first"))
	w, gaps = {}, {}
	for f in FAMILIES:
		gap = float(dl[dl.donated == 1][f].mean() - dl[dl.donated == 0][f].mean())
		gaps[f] = gap
		w[f] = 0.0 if abs(gap) < TUNE_MIN_ABS_DIFF else float(np.sign(gap))
	return w, gaps


def prior_weights():
	"""The brief's theory: receptivity +1, resistance -1."""
	return {f: (1.0 if f.startswith("rec_") else -1.0) for f in FAMILIES}


def score(u, w):
	num = sum(w[f] * u[f] for f in FAMILIES)
	return num / (1.0 + u["tokens"] / 10.0)


def dialogue_table(u, rows_mask):
	"""One row per dialogue from the selected user turns."""
	d = u[rows_mask].sort_values(["dialogue_id", "user_turn"])
	g = d.groupby("dialogue_id")
	t = pd.DataFrame({
		"donated": g["donated"].first().astype(float),
		"split": g["split"].first(),
		"n_turns": g.size().astype(float),
		"nu_final": g["nu"].last(), "nu_mean": g["nu"].mean(),
		"rp_final": g["r_prior"].last(), "rp_mean": g["r_prior"].mean(),
		"rt_final": g["r_tuned"].last(), "rt_mean": g["r_tuned"].mean(),
	})
	return t.reset_index()


MODELS = {
	"M0": ["n_turns"],
	"M1_emotion": ["n_turns", "nu_final", "nu_mean"],
	"M2_resistance_prior": ["n_turns", "rp_final", "rp_mean"],
	"M2_resistance_tuned": ["n_turns", "rt_final", "rt_mean"],
	"M3_both_prior": ["n_turns", "nu_final", "nu_mean", "rp_final", "rp_mean"],
	"M3_both_tuned": ["n_turns", "nu_final", "nu_mean", "rt_final", "rt_mean"],
}


# ---------------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------------
def _design(df, cols, mu=None, sd=None):
	X = df[cols].to_numpy(float)
	if mu is None:
		mu, sd = X.mean(0), X.std(0)
		sd = np.where(sd > 0, sd, 1.0)
	return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


def mcfadden(df, cols):
	y = df["donated"].to_numpy(float)
	X, _, _ = _design(df, cols)
	_, ll, _, _ = lib._logit_fit(X, y)
	p0 = y.mean()
	ll0 = float(len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
	return 1.0 - ll / ll0


def auc(y, p):
	y, p = np.asarray(y), np.asarray(p)
	pos, neg = p[y == 1], p[y == 0]
	if not len(pos) or not len(neg):
		return np.nan
	ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
	return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def oof_predictions(df, cols, rng):
	"""Stratified 5-fold, repeated; returns the mean out-of-fold probability per dialogue."""
	y = df["donated"].to_numpy(float)
	acc = np.zeros(len(df))
	for _ in range(CV_REPEATS):
		fold = np.empty(len(df), int)
		for cls in (0, 1):
			idx = np.flatnonzero(y == cls)
			rng.shuffle(idx)
			fold[idx] = np.arange(len(idx)) % CV_FOLDS
		for k in range(CV_FOLDS):
			tr, te = fold != k, fold == k
			Xtr, mu, sd = _design(df[tr], cols)
			beta, _, _, _ = lib._logit_fit(Xtr, y[tr])
			Xte, _, _ = _design(df[te], cols, mu, sd)
			acc[te] += 1 / (1 + np.exp(-np.clip(Xte @ beta, -30, 30)))
	return acc / CV_REPEATS


def evaluate(df, label):
	"""All models on one dialogue table: point estimates + paired bootstrap over dialogues."""
	rng = np.random.default_rng(RNG_SEED)
	y = df["donated"].to_numpy(float)
	oof = {m: oof_predictions(df, c, rng) for m, c in MODELS.items()}
	point = {}
	for m, c in MODELS.items():
		point[m] = {"k": len(c) + 1, "mcfadden": mcfadden(df, c),
					"brier_oof": float(np.mean((oof[m] - y) ** 2)), "auc_oof": auc(y, oof[m])}

	# paired bootstrap: same dialogue resample for every model
	brng = np.random.default_rng(RNG_SEED + 1)
	reps = {m: {"mcfadden": [], "brier_oof": [], "auc_oof": []} for m in MODELS}
	n = len(df)
	for _ in range(B):
		idx = brng.integers(0, n, n)
		yb = y[idx]
		if yb.min() == yb.max():
			continue
		sub = df.iloc[idx].reset_index(drop=True)
		for m, c in MODELS.items():
			reps[m]["mcfadden"].append(mcfadden(sub, c))
			# OOF predictions are held fixed per dialogue; resampling them gives the CI on the metric
			reps[m]["brier_oof"].append(float(np.mean((oof[m][idx] - yb) ** 2)))
			reps[m]["auc_oof"].append(auc(yb, oof[m][idx]))
	for m in MODELS:
		for s in ("mcfadden", "brier_oof", "auc_oof"):
			point[m][f"{s}_ci"] = lib.ci(reps[m][s])

	# the comparison the brief asks for: M2 vs M1 at equal df, paired
	contrasts = {}
	for m2 in ("M2_resistance_prior", "M2_resistance_tuned"):
		c = {}
		for s in ("mcfadden", "brier_oof", "auc_oof"):
			diff = np.array(reps[m2][s]) - np.array(reps["M1_emotion"][s])
			c[s] = {"diff": point[m2][s] - point["M1_emotion"][s], "ci": lib.ci(diff),
					# share of resamples where resistance is better (Brier: lower is better)
					"p_resistance_better": float(np.mean(diff < 0 if s == "brier_oof" else diff > 0))}
		contrasts[f"{m2}_minus_M1"] = c
	return {"label": label, "n_dialogues": int(n), "donation_rate": float(y.mean()),
			"models": point, "contrasts": contrasts}


# ---------------------------------------------------------------------------------------------
def main():
	u = pd.read_parquet(os.path.join(THU, "user_turns.parquet"))
	H = pd.DataFrame([MK.hits(t) for t in u["text"]])
	u = pd.concat([u.reset_index(drop=True), H], axis=1)
	dec = decision_turns()
	u["decision_turn"] = u["dialogue_id"].map(dec)
	u["pre_decision"] = u["decision_turn"].isna() | (u["user_turn"] < u["decision_turn"])

	w_tuned, gaps = tune_weights(u)
	w_prior = prior_weights()
	u["r_prior"] = score(u, w_prior)
	u["r_tuned"] = score(u, w_tuned)

	out = {"design": {"tune_split": "mine200", "test_split": "test100",
					  "decision_acts": sorted(DECISION_ACTS), "tune_min_abs_diff": TUNE_MIN_ABS_DIFF,
					  "cv": f"stratified {CV_FOLDS}-fold x {CV_REPEATS}", "bootstrap": B, "seed": RNG_SEED},
		   "weights": {"prior": w_prior, "tuned": w_tuned, "tuning_gaps_mine200": gaps},
		   "marker_lists": MK.marker_lists()}

	# how much the cut removes
	n_dec = u.groupby("dialogue_id")["decision_turn"].first()
	out["leakage_cut"] = {
		"dialogues_with_decision_turn": int(n_dec.notna().sum()),
		"dialogues_total": int(len(n_dec)),
		"user_turns_kept": int(u["pre_decision"].sum()),
		"user_turns_total": int(len(u)),
	}

	results = {}
	for cut, mask in (("pre_decision", u["pre_decision"]), ("full_dialogue_LEAKY", pd.Series(True, index=u.index))):
		tbl = dialogue_table(u, mask)
		test = tbl[tbl["split"] == "test100"].reset_index(drop=True)
		dropped = 100 - len(test)
		results[cut] = evaluate(test, f"test100 / {cut}")
		results[cut]["dialogues_without_pre_decision_turns"] = int(dropped)
		print(f"--- {cut}: n={len(test)} (dropped {dropped}) ---")
		for m, v in results[cut]["models"].items():
			print(f"  {m:22s} R2 {v['mcfadden']:.3f} {np.round(v['mcfadden_ci'], 3)}  "
				  f"Brier {v['brier_oof']:.3f}  AUC {v['auc_oof']:.3f} {np.round(v['auc_oof_ci'], 3)}")
		for k, c in results[cut]["contrasts"].items():
			print(f"  {k}: dR2 {c['mcfadden']['diff']:+.3f} {np.round(c['mcfadden']['ci'], 3)}  "
				  f"dAUC {c['auc_oof']['diff']:+.3f} {np.round(c['auc_oof']['ci'], 3)}  "
				  f"dBrier {c['brier_oof']['diff']:+.4f} {np.round(c['brier_oof']['ci'], 4)}")
	out["results"] = results

	# utterance-level: are r and nu the same construct measured twice?
	corr = {}
	for split in ("test100", "mine200"):
		s = u[(u["split"] == split) & u["pre_decision"]]
		corr[split] = {
			"n_utterances": int(len(s)),
			"pearson_nu_rprior": float(s["nu"].corr(s["r_prior"])),
			"spearman_nu_rprior": float(s["nu"].rank().corr(s["r_prior"].rank())),
			"pearson_nu_rtuned": float(s["nu"].corr(s["r_tuned"])),
			"spearman_nu_rtuned": float(s["nu"].rank().corr(s["r_tuned"].rank())),
			"share_r_prior_nonzero": float((s["r_prior"] != 0).mean()),
			"share_r_tuned_nonzero": float((s["r_tuned"] != 0).mean()),
		}
	out["utterance_correlation"] = corr
	print("utterance corr:", json.dumps(corr, indent=1))

	lib.jdump(out, os.path.join(THU, "construct_test.json"))
	with open(os.path.join(THU, "marker_lists.json"), "w") as f:
		json.dump({"lists": MK.marker_lists(), "weights_prior": w_prior, "weights_tuned": w_tuned,
				   "tuning_gaps_mine200": gaps}, f, indent=1)


if __name__ == "__main__":
	main()
