"""Turn one or more ``probe_persona_response.py`` JSONs into ``models_personality_p4g.md``.

    python scripts/report_persona_probe.py                       # every JSON in outputs/persona_probe
    python scripts/report_persona_probe.py --runs a.json b.json  # or name them explicitly

Reads only the JSONs, so re-running it after a new model lands rebuilds the whole comparison
without re-generating anything. Models are reported in the order given (default: filename).
"""
import argparse
import glob
import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEGATIVE = ["anger", "disgust", "fear", "sadness"]


def fmt(x, nd=3, signed=False):
	if x is None or (isinstance(x, float) and np.isnan(x)):
		return "n/a"
	return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def verdict(run):
	"""One-line call on whether this model uses the persona at all."""
	m = run["metrics"]
	p, n = m["persona"], m["no_persona"]
	d = m["directional"]["persona"]["fear_sadness_highN_minus_lowN"]
	lifts = []
	if p["divergence_mean"] and n["divergence_mean"]:
		lifts.append(p["divergence_mean"] / n["divergence_mean"])
	div_lift = lifts[0] if lifts else float("nan")
	directional_ok = d["ci95"][0] > 0
	if div_lift > 1.15 and directional_ok:
		return "**uses it, and in the right direction**", div_lift, directional_ok
	if div_lift > 1.15:
		return "**reads it; direction unconfirmed**", div_lift, directional_ok
	if div_lift < 0.90:
		# below the noise floor is still an effect -- the persona changed the output, but it
		# made the 20 profiles answer *more* alike rather than less. Usually a register shift
		# (longer, more uniformly polite) rather than genuine per-person differentiation.
		tail = " and the direction holds" if directional_ok else ", direction unconfirmed"
		return f"**narrows output rather than differentiating**{tail}", div_lift, directional_ok
	if directional_ok:
		return "**weak divergence, but the direction holds**", div_lift, directional_ok
	return "**no detectable persona effect**", div_lift, directional_ok


def section_overview(runs):
	lines = ["| Model | Verdict | Divergence lift | high-N → fear+sadness |",
			 "| --- | --- | --- | --- |"]
	for r in runs:
		v, lift, ok = verdict(r)
		d = r["metrics"]["directional"]["persona"]["fear_sadness_highN_minus_lowN"]
		ci = f"{fmt(d['diff'], 3, True)} [{fmt(d['ci95'][0], 3, True)}, {fmt(d['ci95'][1], 3, True)}]"
		lines.append(f"| `{r['model_label']}` | {v} | {fmt(lift, 2)}× | {ci} |")
	return "\n".join(lines)


def _len_stats(run, cond):
	rs = [r for r in run["records"] if r["persona_used"] == cond]
	return float(np.mean([len(r["response"].split()) for r in rs]))


def _unique_rate(run, cond):
	rs = [r for r in run["records"] if r["persona_used"] == cond]
	return len({r["response"].strip() for r in rs}) / max(1, len(rs))


def section_metric_table(runs):
	rows = [
		("Cross-persona divergence — persona", lambda m: m["persona"]["divergence_mean"], 4),
		("Cross-persona divergence — no persona (noise floor)", lambda m: m["no_persona"]["divergence_mean"], 4),
		("Identical-response rate — persona", lambda m: m["persona"]["duplicate_response_rate"], 3),
		("Identical-response rate — no persona", lambda m: m["no_persona"]["duplicate_response_rate"], 3),
		("Emotion entropy (bits, max 2.807) — persona", lambda m: m["persona"]["aggregate_entropy"], 3),
		("Emotion entropy — no persona", lambda m: m["no_persona"]["aggregate_entropy"], 3),
		("Negative emotion mass — persona", lambda m: m["persona"]["negative_mass"], 3),
		("Negative emotion mass — no persona", lambda m: m["no_persona"]["negative_mass"], 3),
		("Negative by argmax — persona", lambda m: m["persona"]["negative_argmax_frac"], 3),
		("Negative by argmax — no persona", lambda m: m["no_persona"]["negative_argmax_frac"], 3),
		("Distinct argmax emotions — persona", lambda m: m["persona"]["distinct_argmax_emotions"], 0),
		("Distinct argmax emotions — no persona", lambda m: m["no_persona"]["distinct_argmax_emotions"], 0),
	]
	extra = [
		("Mean response length, words — persona", lambda r: _len_stats(r, True), 1),
		("Mean response length, words — no persona", lambda r: _len_stats(r, False), 1),
		("Unique responses — persona", lambda r: _unique_rate(r, True), 3),
		("Unique responses — no persona", lambda r: _unique_rate(r, False), 3),
	]
	header = "| Measurement | " + " | ".join(f"`{r['model_label']}`" for r in runs) + " |"
	sep = "| --- |" + " --- |" * len(runs)
	out = [header, sep]
	for label, fn, nd in rows:
		cells = []
		for r in runs:
			try:
				cells.append(fmt(fn(r["metrics"]), nd))
			except Exception:
				cells.append("n/a")
		out.append(f"| {label} | " + " | ".join(cells) + " |")
	for label, fn, nd in extra:
		out.append(f"| {label} | " + " | ".join(fmt(fn(r), nd) for r in runs) + " |")
	return "\n".join(out)


def section_emotion_profile(run):
	m = run["metrics"]
	emos = [e for e in run.get("metric_emotions", run["emotions_order"])]
	lines = ["| Emotion | no persona | persona | Δ |", "| --- | --- | --- | --- |"]
	for e in emos:
		a = m["no_persona"]["aggregate_emotion"].get(e, 0.0)
		b = m["persona"]["aggregate_emotion"].get(e, 0.0)
		mark = " **←**" if abs(b - a) >= 0.02 else ""
		lines.append(f"| {e}{' *(neg)*' if e in NEGATIVE else ''} | {fmt(a)} | {fmt(b)} | {fmt(b - a, 3, True)}{mark} |")
	return "\n".join(lines)


def section_by_cell(run):
	"""Per-cell emotion means in the persona condition — the directional evidence, unpooled."""
	recs = [r for r in run["records"] if r["persona_used"]]
	emos = run.get("metric_emotions", run["emotions_order"])
	lines = ["| Cell | agreeable | neurotic | fear+sadness | anger+disgust | negative total |",
			 "| --- | --- | --- | --- | --- | --- |"]
	for cell in ("lowA_lowN", "lowA_highN", "highA_lowN", "highA_highN"):
		rs = [r for r in recs if r["cell"] == cell]
		if not rs:
			continue
		prof = [p for p in run["profiles"] if p["cell"] == cell]
		fs = np.mean([r["emotion_dist"]["fear"] + r["emotion_dist"]["sadness"] for r in rs])
		ad = np.mean([r["emotion_dist"]["anger"] + r["emotion_dist"]["disgust"] for r in rs])
		neg = np.mean([sum(r["emotion_dist"][e] for e in NEGATIVE) for r in rs])
		lines.append(
			f"| `{cell}` | {np.mean([p['agreeable'] for p in prof]):.2f} | "
			f"{np.mean([p['neurotic'] for p in prof]):.2f} | {fs:.3f} | {ad:.3f} | {neg:.3f} |"
		)
	return "\n".join(lines)


def section_interaction(run):
	"""The neuroticism effect split by agreeableness.

	Pooling high-N against low-N averages over both agreeableness halves, which hides the case
	where a model only differentiates within one of them -- an interaction that reads as "no
	effect" in the headline test. Bootstrap CIs, same procedure as the pooled test.
	"""
	import importlib.util
	spec = importlib.util.spec_from_file_location(
		"probe", os.path.join(REPO_ROOT, "scripts", "probe_persona_response.py"))
	probe = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(probe)

	recs = [r for r in run["records"] if r["persona_used"]]
	lines = ["| Agreeableness half | high-N − low-N, fear+sadness | 95% CI | clears 0 |",
			 "| --- | --- | --- | --- |"]
	for half, name in (("lowA", "low agreeableness"), ("highA", "high agreeableness")):
		hi = [r for r in recs if r["cell"] == f"{half}_highN"]
		lo = [r for r in recs if r["cell"] == f"{half}_lowN"]
		if not hi or not lo:
			continue
		mass = lambda rs: [r["emotion_dist"]["fear"] + r["emotion_dist"]["sadness"] for r in rs]
		diff, l, u = probe.bootstrap_diff(mass(hi), mass(lo))
		lines.append(f"| {name} | {fmt(diff, 3, True)} | [{fmt(l, 3, True)}, {fmt(u, 3, True)}] | "
					 f"{'yes' if l > 0 else 'no'} |")
	return "\n".join(lines)


def section_da(run):
	"""Which persuadee acts the simulator emits. Emotion is what the classifier reads off the
	text; the DA is what the *search* actually consumes, so a collapse here matters more."""
	m = run["metrics"]
	das = sorted(set(m["persona"]["da_distribution"]) | set(m["no_persona"]["da_distribution"]))
	lines = ["| Persuadee act | no persona | persona | Δ |", "| --- | --- | --- | --- |"]
	for da in das:
		a = m["no_persona"]["da_distribution"].get(da, 0.0)
		b = m["persona"]["da_distribution"].get(da, 0.0)
		lines.append(f"| {da} | {fmt(a)} | {fmt(b)} | {fmt(b - a, 3, True)} |")
	return "\n".join(lines)


def section_by_act(run):
	recs = run["records"]
	acts = list(run["fixed_utterances"])
	lines = ["| Persuader act | negative mass (no persona) | negative mass (persona) | Δ |",
			 "| --- | --- | --- | --- |"]
	for act in acts:
		a = [sum(r["emotion_dist"][e] for e in NEGATIVE) for r in recs if r["act"] == act and not r["persona_used"]]
		b = [sum(r["emotion_dist"][e] for e in NEGATIVE) for r in recs if r["act"] == act and r["persona_used"]]
		if a and b:
			lines.append(f"| {act} | {np.mean(a):.3f} | {np.mean(b):.3f} | {np.mean(b) - np.mean(a):+.3f} |")
	return "\n".join(lines)


def section_examples(run, act="proposition of donation", n=3):
	"""Same utterance, opposite personas — the 'just read 20' check."""
	out = []
	for cell in ("lowA_highN", "highA_lowN"):
		rs = [r for r in run["records"]
			  if r["persona_used"] and r["act"] == act and r["cell"] == cell][:n]
		out.append(f"\n**`{cell}`** (agreeableness/neuroticism extreme)\n")
		for r in rs:
			out.append(f"> [{r['da']}] {r['response'].strip()}  \n> `→ {r['emotion']}`\n")
	rs = [r for r in run["records"] if not r["persona_used"] and r["act"] == act][:n]
	out.append("\n**No persona** (same prompt every time)\n")
	for r in rs:
		out.append(f"> [{r['da']}] {r['response'].strip()}  \n> `→ {r['emotion']}`\n")
	return "\n".join(out)


def main(args):
	paths = args.runs or sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
	if not paths:
		raise SystemExit(f"no probe JSONs found in {args.in_dir}")
	runs = []
	for p in paths:
		with open(p, encoding="utf-8") as f:
			runs.append(json.load(f))
	print(f"reporting on {len(runs)} run(s): {', '.join(r['model_label'] for r in runs)}")

	r0 = runs[0]
	doc = [
		"# Do the models actually use a persona? (p4g persuadee simulator)",
		"",
		"Probe of whether an LLM user simulator *responds* to persona conditioning at all, before",
		"spending a planner run on it. A model that silently ignores the persona turns the whole",
		"`--p4g_persona` experiment into a no-op that still completes and still reports numbers.",
		"",
		"Generated by `scripts/report_persona_probe.py` from `scripts/probe_persona_response.py`",
		"output. Every number below comes from those JSONs; nothing is hand-entered.",
		"",
		"## Design",
		"",
		f"- **{r0['n_profiles']} profiles**, a balanced 2×2 over the extremes of agreeableness ×",
		"  neuroticism, taken from the real Persuasion for Good pre-task survey",
		"  (`data/p4g_personas/full_info.csv`), restricted to the annotated 300.",
		f"- **{r0['n_acts']} fixed Persuader utterances**, one per act type. Identical across every",
		"  profile and condition, so the persona is the only thing that varies.",
		f"- **{r0['n_samples']} samples** per (profile, act) cell →",
		f"  **{r0['n_profiles'] * r0['n_acts'] * r0['n_samples']} generations per condition**,",
		"  run with and without the persona.",
		"- Generation goes through the production `PersuadeeChatModel` at the same sampling settings",
		f"  `build_agents` uses (`temperature={r0['inference_args']['temperature']}`,",
		f"  `top_p={r0['inference_args']['top_p']}`, `top_k={r0['inference_args']['top_k']}`).",
		f"- Emotions from `j-hartmann/emotion-english-distilroberta-base`; embeddings from",
		f"  `{r0['embedder']}`.",
		"",
		"**The no-persona condition is the control.** There all profiles get a byte-identical",
		"prompt, so its cross-profile divergence is pure sampling noise. A persona is being read",
		"only if persona divergence sits meaningfully above that floor.",
		"",
		"## Verdict",
		"",
		section_overview(runs),
		"",
		"*Divergence lift = persona divergence ÷ no-persona divergence. 1.0× means the persona",
		"changed nothing beyond sampling noise. The directional column is the high-neuroticism",
		"minus low-neuroticism difference in fear+sadness mass, with a bootstrap 95% CI; the",
		"interval must clear 0 for the model to be using the persona *correctly*, not just",
		"differently.*",
		"",
		"## All measurements",
		"",
		section_metric_table(runs),
		"",
	]

	for r in runs:
		v, lift, ok = verdict(r)
		doc += [
			f"## `{r['model_label']}`",
			"",
			f"Served as `{r['served_model']}`. Verdict: {v} ({fmt(lift, 2)}× divergence lift).",
			"",
			"### Emotion distribution shift",
			"",
			section_emotion_profile(r),
			"",
			"### Directional evidence, by profile cell (persona condition)",
			"",
			section_by_cell(r),
			"",
			"### Interaction: is the neuroticism effect uniform?",
			"",
			section_interaction(r),
			"",
			"### Dialog acts emitted (what the search consumes)",
			"",
			section_da(r),
			"",
			"### Where the persona bites, by Persuader act",
			"",
			section_by_act(r),
			"",
			"### Sample generations — same utterance, opposite personas",
			"",
			f"Persuader: *\"{r['fixed_utterances']['proposition of donation']}\"*",
			section_examples(r),
			"",
		]

	if args.pending:
		doc += [
			"## Not yet probed",
			"",
			"These were requested but have not been run, so no numbers for them appear above:",
			"",
		]
		for label in args.pending:
			doc += [f"- `{label}`"]
		doc += [
			"",
			"Each needs the GPU to itself — this box is a single 24 GB RTX 4090 and the SGLang",
			"server for the model above already holds ~22 GB. To add one:",
			"",
			"```bash",
			"# stop the running SGLang server first, then",
			"python -m sglang.launch_server --model-path <repo/model> --port 30000",
			"python scripts/probe_persona_response.py --model_label <label>",
			"python scripts/report_persona_probe.py     # rebuilds this document with all runs",
			"```",
			"",
		]

	out = os.path.join(REPO_ROOT, args.out)
	with open(out, "w", encoding="utf-8") as f:
		f.write("\n".join(doc))
	print(f"wrote {out}")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--in_dir", default=os.path.join(REPO_ROOT, "outputs", "persona_probe"))
	ap.add_argument("--runs", nargs="*", help="explicit JSON paths, in report order")
	ap.add_argument("--out", default="models_personality_p4g.md")
	ap.add_argument("--pending", nargs="*", default=[],
					help="model labels that were requested but not yet run; listed as outstanding")
	main(ap.parse_args())
