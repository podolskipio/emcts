"""TASK 2 follow-up -- outcome leakage: how much of the apparent affective (and lexical) prediction of
donation is the outcome itself, and is w(e) partly fitted to outcome language?

    python analysis/thu/scripts/t2_leakage.py      (after t2_build_turns.py)

Three feature windows, applied IDENTICALLY to emotion and resistance features and to the mine200
tuning of r and w(e):

  full         every user turn (leaky; what w(e) was mined on)
  commit_cut   turns strictly before the first commitment-language turn (a hit on the `commitment`
               marker family); if the dialogue has none, turns strictly before the final user turn
               (human-specified, 2026-09-16)
  decision_cut turns strictly before the first annotated decision act (agree-donation,
               disagree-donation, disagree-donation-more, provide-donation-amount, confirm-donation);
               if none, every turn (construct_test.md's cut)

Leakage audit per window and outcome group: how much annotated decision language survives the cut.

w(e) re-mine: the mining formula (mine_emotion_donation_p4g.py: w = 10 * n/(n+alpha) * lift, soft
per-turn emotion mass, dialogue-level outcome, alpha 50) re-implemented, checked against the shipped
w_mined_200.json on full turns, then re-run on each window. Tests M1 with the window's own w(e).
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
import t2_construct_test as CT  # noqa: E402

THU = os.path.join(REPO, "analysis", "thu")
EMO7 = ["happiness", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
ALPHA = 50.0


def turn_acts():
	dialogs = pickle.load(open(os.path.join(REPO, "data", "p4g", "300_dialog_turn_based.pkl"), "rb"))
	return {(did, i): set(lab.get("ee", [])) for did, d in dialogs.items() for i, lab in enumerate(d["label"])
			if isinstance(lab, dict)}


def windows(u):
	w = {"full": pd.Series(True, index=u.index)}
	commit = u[u["rec_commitment"] > 0].groupby("dialogue_id")["user_turn"].min()
	last = u.groupby("dialogue_id")["user_turn"].max()
	cut = u["dialogue_id"].map(commit).fillna(u["dialogue_id"].map(last))
	w["commit_cut"] = u["user_turn"] < cut
	u["_commit_turn_found"] = u["dialogue_id"].map(commit).notna()
	dec = CT.decision_turns()
	d = u["dialogue_id"].map(dec)
	w["decision_cut"] = d.isna() | (u["user_turn"] < d)
	return w


def audit(u, mask):
	k = u[mask]
	out = {}
	for don, g in k.groupby("donated"):
		dl = g.groupby("dialogue_id")
		all_d = u[u["donated"] == don]["dialogue_id"].nunique()
		out["donors" if don == 1 else "non_donors"] = {
			"dialogues_with_any_kept_turn": int(dl.ngroups), "dialogues_total": int(all_d),
			"kept_turns_per_dialogue": float(dl.size().mean()),
			"share_dialogues_keeping_agree_or_amount": float(dl["_agree"].max().mean()),
			"share_dialogues_keeping_disagree": float(dl["_disagree"].max().mean()),
			"share_kept_turns_with_any_decision_act": float(g["_decision"].mean()),
			"share_kept_turns_with_commitment_hit": float((g["rec_commitment"] > 0).mean()),
		}
	return out


def mine_w(u, mask):
	"""w(e) = 10 * n_e/(n_e+alpha) * (p(donate | e) - base), soft per-turn emotion mass."""
	m = u[mask & (u["split"] == "mine200")]
	y = m["donated"].to_numpy(float)
	base = float(m.groupby("dialogue_id")["donated"].first().mean())
	base_turn = float(y.mean())
	w, rows = {}, {}
	for e in EMO7:
		p = m[f"p_{e}"].to_numpy()
		n = p.sum()
		pd_e = float((p * y).sum() / n)
		lift = pd_e - base
		w[e] = round(10 * n / (n + ALPHA) * lift, 2)
		rows[e] = {"n_soft": float(n), "p_donate": pd_e, "lift": lift, "w": w[e]}
	return w, {"base_dialogue": base, "base_turn": base_turn, "per_emotion": rows, "turns": int(len(m))}


def nu_from(u, w):
	return u[[f"p_{e}" for e in EMO7]].to_numpy() @ np.array([w.get(e, 0.0) for e in EMO7])


def tune_on(u, mask):
	m = u[(u["split"] == "mine200") & mask]
	dl = m.groupby("dialogue_id").agg(**{f: (f, "mean") for f in CT.FAMILIES}, donated=("donated", "first"))
	w, gaps = {}, {}
	for f in CT.FAMILIES:
		gap = float(dl[dl.donated == 1][f].mean() - dl[dl.donated == 0][f].mean())
		gaps[f] = gap
		w[f] = 0.0 if abs(gap) < CT.TUNE_MIN_ABS_DIFF else float(np.sign(gap))
	return w, gaps


def main():
	u = pd.read_parquet(os.path.join(THU, "user_turns.parquet"))
	u = pd.concat([u.reset_index(drop=True), pd.DataFrame([MK.hits(t) for t in u["text"]])], axis=1)
	acts = turn_acts()
	a = [acts.get((d, t), set()) for d, t in zip(u["dialogue_id"], u["user_turn"])]
	u["_agree"] = [bool(x & {"agree-donation", "provide-donation-amount", "confirm-donation"}) for x in a]
	u["_disagree"] = [bool(x & {"disagree-donation", "disagree-donation-more"}) for x in a]
	u["_decision"] = u["_agree"] | u["_disagree"]
	W = windows(u)
	shipped = json.load(open(os.path.join(REPO, "analysis", "wed", "corpus", "w_mined_200.json")))["valence_weights"]

	out = {"windows": {}, "shipped_w200": shipped,
		   "commit_cut_dialogues_without_commitment_turn": int((~u.groupby("dialogue_id")["_commit_turn_found"].first()).sum())}
	u["r_prior"] = CT.score(u, CT.prior_weights())
	for name, mask in W.items():
		wt, gaps = tune_on(u, mask)
		u["r_tuned"] = CT.score(u, wt)
		w_re, mine_info = mine_w(u, mask)
		res = {"audit": audit(u, mask), "kept_turns": int(mask.sum()), "tuned_r_weights": wt, "tuning_gaps": gaps,
			   "w_remined": w_re, "w_remined_info": mine_info}
		# (a) nu with the SHIPPED w200 -- as deployed
		u["nu"] = nu_from(u, shipped)
		tbl = CT.dialogue_table(u, mask)
		test = tbl[tbl["split"] == "test100"].reset_index(drop=True)
		res["n_test_dialogues"] = int(len(test))
		res["shipped_w"] = CT.evaluate(test, f"{name}/shipped w200")
		# (b) nu with w(e) re-mined on this window's mine200 turns
		u["nu"] = nu_from(u, w_re)
		tbl = CT.dialogue_table(u, mask)
		test = tbl[tbl["split"] == "test100"].reset_index(drop=True)
		res["remined_w"] = CT.evaluate(test, f"{name}/re-mined w")
		out["windows"][name] = res
		m, c = res["shipped_w"]["models"], res["shipped_w"]["contrasts"]
		print(f"== {name}: kept {mask.sum()} turns, n_test {len(test)}")
		print("   w re-mined:", w_re)
		for k in ("M0", "M1_emotion", "M2_resistance_prior", "M2_resistance_tuned", "M3_both_tuned"):
			print(f"   {k:22s} R2 {m[k]['mcfadden']:.3f}  AUC {m[k]['auc_oof']:.3f} {np.round(m[k]['auc_oof_ci'], 3)}  Brier {m[k]['brier_oof']:.3f}")
		print(f"   M1 with re-mined w: R2 {res['remined_w']['models']['M1_emotion']['mcfadden']:.3f} AUC {res['remined_w']['models']['M1_emotion']['auc_oof']:.3f}")
		for k, v in c.items():
			print(f"   {k}: dAUC {v['auc_oof']['diff']:+.3f} {np.round(v['auc_oof']['ci'], 3)} dR2 {v['mcfadden']['diff']:+.3f}")
		print("   audit:", json.dumps(res["audit"]))
	lib.jdump(out, os.path.join(THU, "leakage.json"))


if __name__ == "__main__":
	main()
