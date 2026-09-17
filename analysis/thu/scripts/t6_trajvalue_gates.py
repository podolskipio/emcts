"""TASK 6 -- TrajValue Gate A and Gate B (spec as given by the human, 2026-09-17).

    python analysis/thu/scripts/t6_trajvalue_gates.py

Corpus: the 297 annotated P4G dialogues (analysis/thu/user_turns.parquet: whole persuadee turns, the unit the
planner classifies), PRE-DECISION CUT (turns strictly before the first decision act, as Task 2 and the
re-mined table). Dialogues with no pre-decision turn are dropped.

nu is CROSS-FITTED: in every fold, w(e) is re-mined on the training dialogues only with the `predecision`
recipe (soft, alpha 50, unit base rate), and nu for the held-out dialogues uses that fold's w. So neither the
features nor g ever see the outcome of the dialogue they score.

Trajectory features per dialogue (over its pre-decision turns, in order):
  nu_final     nu of the last turn          n_turns      number of turns
  nu_mean      mean nu                      n_negative   turns with nu < 0
  neg_run_max  longest run of nu < 0        nu_slope     OLS slope of nu on turn index (0 if one turn)

Gate A  logit(donated) ~ nu_final + n_turns  vs  + nu_mean + n_negative + neg_run_max + nu_slope.
        Incremental McFadden pseudo-R2, 1000-replicate bootstrap over dialogues.
        PASS: CI excludes 0 and point >= 0.02.
        Also reported: per-feature drop-one increments and coefficients, the cross-validated increment
        (out-of-fold log-loss / Brier / AUC; an in-sample nested increment cannot be negative, so its CI
        excluding 0 is nearly automatic), and the sustained-negative vs recovering contrast.

Gate B  5-fold cross-fit. Signature = (tercile(nu_final) with edges from the training folds,
        bucket(neg_run_max) in {0, 1-2, 3+}) = 9 cells. g(cell) = training donation rate in the cell minus
        the training base rate, centred so sum_cells g = 0, clipped to +/-0.25; empty cells g = 0.
        Held-out prediction p = clip(base_train + g, 0, 1) against p = base_train.
        PASS: out-of-fold Brier improves with a bootstrap CI excluding 0, and no cell's g flips sign in more
        than one fold (sign relative to the majority sign across folds; g == 0 folds are not counted).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "analysis", "wed", "scripts"))
import lib  # noqa: E402
import t2_construct_test as CT  # noqa: E402

THU = os.path.join(REPO, "analysis", "thu")
EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
ALPHA = 50.0
K = 5
B = 1000
SEED = 20260917
G_CLIP = 0.25
BASE = ["nu_final", "n_turns"]
TRAJ = ["nu_mean", "n_negative", "neg_run_max", "nu_slope"]


# ------------------------------------------------------------------ corpus
def load_turns():
	u = pd.read_parquet(os.path.join(THU, "user_turns.parquet"))
	dec = CT.decision_turns()
	d = u["dialogue_id"].map(dec)
	u = u[d.isna() | (u["user_turn"] < d)].sort_values(["dialogue_id", "user_turn"]).reset_index(drop=True)
	return u


def mine_w(turns):
	"""predecision recipe at the turn unit: soft mass, alpha 50, unit-weighted base rate."""
	y = turns["donated"].to_numpy(float)
	P = turns[[f"p_{e}" for e in EMO7]].to_numpy()
	n = P.sum(0)
	s = (P * y[:, None]).sum(0)
	base = s.sum() / n.sum()
	p_shrunk = (s + ALPHA * base) / (n + ALPHA)
	return dict(zip(EMO7, np.round(10 * (p_shrunk - base), 2)))


def features(turns, w):
	P = turns[[f"p_{e}" for e in EMO7]].to_numpy()
	nu = P @ np.array([w[e] for e in EMO7])
	t = turns.assign(nu=nu)
	rows = []
	for did, g in t.groupby("dialogue_id", sort=False):
		v = g["nu"].to_numpy()
		neg = v < 0
		run = best = 0
		for x in neg:
			run = run + 1 if x else 0
			best = max(best, run)
		slope = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
		rows.append({"dialogue_id": did, "donated": float(g["donated"].iloc[0]), "n_turns": float(len(v)),
					 "nu_final": float(v[-1]), "nu_mean": float(v.mean()), "n_negative": float(neg.sum()),
					 "neg_run_max": float(best), "nu_slope": slope, "final_negative": bool(v[-1] < 0)})
	return pd.DataFrame(rows)


def folds(dids, y, rng):
	f = np.empty(len(dids), int)
	for cls in (0, 1):
		idx = np.flatnonzero(y == cls)
		rng.shuffle(idx)
		f[idx] = np.arange(len(idx)) % K
	return dict(zip(dids, f))


def crossfit_features(turns, fold_of):
	parts, ws = [], {}
	for k in range(K):
		tr = turns[turns["dialogue_id"].map(fold_of) != k]
		te = turns[turns["dialogue_id"].map(fold_of) == k]
		w = mine_w(tr)
		ws[k] = w
		f = features(te, w).assign(fold=k)
		parts.append(f)
	return pd.concat(parts, ignore_index=True), ws


# ------------------------------------------------------------------ Gate A
def design(df, cols, mu=None, sd=None):
	X = df[cols].to_numpy(float)
	if mu is None:
		mu, sd = X.mean(0), X.std(0)
		sd = np.where(sd > 0, sd, 1.0)
	return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


def mcfadden(df, cols):
	y = df["donated"].to_numpy(float)
	X, _, _ = design(df, cols)
	beta, ll, _, _ = lib._logit_fit(X, y)
	p0 = y.mean()
	return 1 - ll / (len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))), beta


def oof_probs(df, cols, rng, repeats=20):
	y = df["donated"].to_numpy(float)
	acc = np.zeros(len(df))
	for _ in range(repeats):
		f = np.empty(len(df), int)
		for cls in (0, 1):
			idx = np.flatnonzero(y == cls)
			rng.shuffle(idx)
			f[idx] = np.arange(len(idx)) % K
		for k in range(K):
			tr, te = f != k, f == k
			X, mu, sd = design(df[tr], cols)
			b, _, _, _ = lib._logit_fit(X, y[tr])
			Xt, _, _ = design(df[te], cols, mu, sd)
			acc[te] += 1 / (1 + np.exp(-np.clip(Xt @ b, -30, 30)))
	return np.clip(acc / repeats, 1e-6, 1 - 1e-6)


def gate_a(df):
	full = BASE + TRAJ
	r2_base, _ = mcfadden(df, BASE)
	r2_full, beta = mcfadden(df, full)
	inc = r2_full - r2_base
	brng = np.random.default_rng(SEED)
	n = len(df)
	reps, drop_reps = [], {c: [] for c in TRAJ}
	for _ in range(B):
		sub = df.iloc[brng.integers(0, n, n)].reset_index(drop=True)
		if sub["donated"].nunique() < 2:
			continue
		rf = mcfadden(sub, full)[0]
		reps.append(rf - mcfadden(sub, BASE)[0])
		for c in TRAJ:
			drop_reps[c].append(rf - mcfadden(sub, [x for x in full if x != c])[0])
	drop = {c: {"increment_when_added_last": r2_full - mcfadden(df, [x for x in full if x != c])[0],
				"ci": lib.ci(drop_reps[c]),
				"coef_std": float(beta[1 + full.index(c)])} for c in TRAJ}
	# cross-validated increment
	rng = np.random.default_rng(SEED + 7)
	y = df["donated"].to_numpy(float)
	pb, pf = oof_probs(df, BASE, rng), oof_probs(df, full, rng)
	ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
	d_ll, d_br = ll(pb) - ll(pf), (pb - y) ** 2 - (pf - y) ** 2  # positive = trajectory helps
	cv_reps = {"logloss": [], "brier": [], "auc": []}
	for _ in range(B):
		i = brng.integers(0, n, n)
		cv_reps["logloss"].append(float(d_ll[i].mean()))
		cv_reps["brier"].append(float(d_br[i].mean()))
		if len(set(y[i])) == 2:
			cv_reps["auc"].append(CT.auc(y[i], pf[i]) - CT.auc(y[i], pb[i]))
	a = {"n_dialogues": int(n), "donation_rate": float(y.mean()),
		 "mcfadden_base": r2_base, "mcfadden_full": r2_full,
		 "increment": lib.with_ci(inc, reps),
		 "PASS": bool(lib.ci(reps)[0] > 0 and inc >= 0.02),
		 "per_feature": drop,
		 "cross_validated": {"logloss_gain": lib.with_ci(float(d_ll.mean()), cv_reps["logloss"]),
							 "brier_gain": lib.with_ci(float(d_br.mean()), cv_reps["brier"]),
							 "auc_base": CT.auc(y, pb), "auc_full": CT.auc(y, pf),
							 "auc_gain": lib.with_ci(CT.auc(y, pf) - CT.auc(y, pb), cv_reps["auc"])}}
	# sustained-negative vs recovering
	sust = df[(df["neg_run_max"] >= 2) & df["final_negative"]]
	reco = df[(df["neg_run_max"] >= 2) & ~df["final_negative"]]
	never = df[df["neg_run_max"] == 0]
	a["sustained_vs_recovering"] = {
		name: {"n": int(len(g)), "donation_rate": float(g["donated"].mean()) if len(g) else None,
			   "wilson95": lib.wilson(int(g["donated"].sum()), len(g)) if len(g) else None}
		for name, g in (("sustained_negative (run>=2, final<0)", sust),
						("recovering (run>=2, final>=0)", reco), ("never_negative", never))}
	return a


# ------------------------------------------------------------------ Gate B
def run_bucket(x):
	return 0 if x == 0 else (1 if x <= 2 else 2)


def mine_g(train):
	edges = np.quantile(train["nu_final"], [1 / 3, 2 / 3])
	cell = np.digitize(train["nu_final"], edges) * 3 + train["neg_run_max"].map(run_bucket).to_numpy()
	base = float(train["donated"].mean())
	g = np.zeros(9)
	counts = np.zeros(9, int)
	for c in range(9):
		m = cell == c
		counts[c] = int(m.sum())
		if m.any():
			g[c] = train["donated"][m].mean() - base
	g = g - g.mean()
	g = np.clip(g, -G_CLIP, G_CLIP)
	return {"edges": edges.tolist(), "g": g, "counts": counts, "base": base}


def cell_of(df, edges):
	return np.digitize(df["nu_final"], edges) * 3 + df["neg_run_max"].map(run_bucket).to_numpy()


def gate_b(feat):
	rows, per_fold = [], {}
	for k in range(K):
		tr, te = feat[feat["fold"] != k], feat[feat["fold"] == k]
		m = mine_g(tr)
		per_fold[k] = m
		c = cell_of(te, m["edges"])
		p0 = np.full(len(te), m["base"])
		p1 = np.clip(m["base"] + m["g"][c], 0, 1)
		y = te["donated"].to_numpy()
		rows.append(pd.DataFrame({"dialogue_id": te["dialogue_id"].to_numpy(), "y": y, "p0": p0, "p1": p1, "cell": c}))
	r = pd.concat(rows, ignore_index=True)
	gain = (r["p0"] - r["y"]) ** 2 - (r["p1"] - r["y"]) ** 2  # positive = g helps
	brng = np.random.default_rng(SEED + 3)
	n = len(r)
	reps = [float(gain.to_numpy()[brng.integers(0, n, n)].mean()) for _ in range(B)]
	G = np.array([per_fold[k]["g"] for k in range(K)])
	flips = {}
	for c in range(9):
		s = np.sign(G[:, c])
		nz = s[s != 0]
		if not len(nz):
			flips[c] = 0
			continue
		maj = 1.0 if (nz > 0).sum() >= (nz < 0).sum() else -1.0
		flips[c] = int((nz != maj).sum())
	ci = lib.ci(reps)
	occupancy = r["cell"].value_counts(normalize=True).reindex(range(9), fill_value=0.0)
	return {"n_dialogues": int(n),
			"brier_base": float(((r["p0"] - r["y"]) ** 2).mean()), "brier_g": float(((r["p1"] - r["y"]) ** 2).mean()),
			"brier_gain": {"value": float(gain.mean()), "ci": ci},
			"sign_flips_per_cell": {cell_name(c): flips[c] for c in range(9)},
			"PASS_brier": bool(ci[0] > 0), "PASS_signs": bool(max(flips.values()) <= 1),
			"PASS": bool(ci[0] > 0 and max(flips.values()) <= 1),
			"g_by_fold": {cell_name(c): [round(float(G[k, c]), 3) for k in range(K)] for c in range(9)},
			"train_counts_by_fold": {cell_name(c): [int(per_fold[k]["counts"][c]) for k in range(K)] for c in range(9)},
			"oof_cell_occupancy": {cell_name(c): float(occupancy[c]) for c in range(9)},
			"tercile_edges_by_fold": [per_fold[k]["edges"] for k in range(K)]}


def cell_name(c):
	return f"nu_final_T{c // 3 + 1}|negrun_{['0', '1-2', '3+'][c % 3]}"


def main():
	turns = load_turns()
	per_d = turns.groupby("dialogue_id")["donated"].first()
	rng = np.random.default_rng(SEED)
	fold_of = folds(per_d.index.to_numpy(), per_d.to_numpy(), rng)
	feat, ws = crossfit_features(turns, fold_of)
	out = {"design": {"corpus": "annotated P4G, pre-decision cut, whole persuadee turns", "folds": K, "bootstrap": B,
					  "seed": SEED, "nu": "cross-fitted: predecision recipe mined on training folds (turn unit)",
					  "g_clip": G_CLIP},
		   "w_by_fold": ws,
		   "gate_A": gate_a(feat), "gate_B": gate_b(feat)}
	# sensitivity: nu under the frozen `predecision` table (mined on all 300 -> not cross-fitted, leaks)
	sys.path.insert(0, os.path.join(REPO, "src"))
	import mcts.emotion_mcts as em
	w_frozen = {str(getattr(k, "value", k)).lower(): v for k, v in em.EMOTION_VALENCE_TABLES["predecision"].items()}
	ff = features(turns, w_frozen).merge(feat[["dialogue_id", "fold"]], on="dialogue_id")
	out["sensitivity_frozen_table_not_crossfit"] = {"gate_A": gate_a(ff), "gate_B": gate_b(ff)}
	lib.jdump(out, os.path.join(THU, "trajvalue_gates.json"))
	for name, blk in (("cross-fit", out), ("frozen table (leaky)", out["sensitivity_frozen_table_not_crossfit"])):
		A, Bb = blk["gate_A"], blk["gate_B"]
		print(f"== {name}")
		print(f"  Gate A: n {A['n_dialogues']} R2 base {A['mcfadden_base']:.3f} full {A['mcfadden_full']:.3f} "
			  f"inc {A['increment']['value']:.4f} {np.round(A['increment']['ci'], 4)} PASS={A['PASS']}")
		print("   per feature:", {c: (round(v['increment_when_added_last'], 4), round(v['coef_std'], 3)) for c, v in A["per_feature"].items()})
		cv = A["cross_validated"]
		print(f"   CV: logloss gain {cv['logloss_gain']['value']:+.4f} {np.round(cv['logloss_gain']['ci'], 4)} "
			  f"brier gain {cv['brier_gain']['value']:+.4f} {np.round(cv['brier_gain']['ci'], 4)} AUC {cv['auc_base']:.3f}->{cv['auc_full']:.3f}")
		print("   sustained/recovering:", json.dumps(A["sustained_vs_recovering"]))
		print(f"  Gate B: brier {Bb['brier_base']:.4f} -> {Bb['brier_g']:.4f} gain {Bb['brier_gain']['value']:+.4f} "
			  f"{np.round(Bb['brier_gain']['ci'], 4)} flips {Bb['sign_flips_per_cell']} PASS={Bb['PASS']}")
		print("   g by fold:", Bb["g_by_fold"])
		print("   counts:", Bb["train_counts_by_fold"])


if __name__ == "__main__":
	main()
