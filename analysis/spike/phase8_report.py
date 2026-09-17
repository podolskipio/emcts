"""Assemble spike/RESULTS.md from analysis.json + generations.jsonl."""
import json, os, subprocess, numpy as np
ROOT="/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
A=json.load(open(f"{ROOT}/spike/analysis.json"))
recs=[json.loads(l) for l in open(f"{ROOT}/spike/generations.jsonl")]
personas=json.load(open(f"{ROOT}/spike/personas_rendered.json"))
D=json.load(open(f"{ROOT}/spike/personas.json"))
MODELS=[m for m in ["vicuna-13b-v1.5","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct"] if m in A]
CKPT={"vicuna-13b-v1.5":"TheBloke/vicuna-13B-v1.5-AWQ",
      "Llama-3.1-8B-Instruct":"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
      "Qwen2.5-7B-Instruct":"Qwen/Qwen2.5-7B-Instruct-AWQ"}
prereg_time=subprocess.run(["stat","-c","%y",f"{ROOT}/spike/preregistration.md"],
                           capture_output=True,text=True).stdout.strip()[:19]

def pct(x): return f"{100*x:.2f}%"
L=[]; W=L.append
W("# Persona Spike — Does the P4G user simulator respond to psychological profiles?\n")
W(f"Pre-registration written **{prereg_time}**, before any generation or classification "
  f"(`spike/preregistration.md`).\n")

# 1 verdicts
W("## 1. Verdict\n")
W("| Model | permutation p | pre-registered ρ significant | Verdict |")
W("|---|---|---|---|")
for m in MODELS:
    a=A[m]; W(f"| `{m}` | {a['p_value']:.4f} | {a['n_sig']}/3 | **{a['verdict']}** |")
W("")
verdicts={m:A[m]["verdict"] for m in MODELS}
passing=[m for m in MODELS if verdicts[m] in ("PASS","STRONG PASS")]
W("### Combined action\n")
W("**The near-miss the design was built to detect actually occurred, in two of three "
  "models.** Vicuna and Llama both pass the manipulation check decisively (p < 0.001) and "
  "both show a strong agreeableness→happiness correlation (ρ = +0.57 and +0.50), but their "
  "neuroticism→negative-affect correlation is indistinguishable from zero (ρ = +0.006 and "
  "−0.022). That is persona *style* adoption without persona *emotional dynamics* — exactly "
  "the pattern that passes question 1 and fails question 3. Had the spike stopped at the "
  "manipulation check, all three models would have looked like passes.\n")
if not passing:
    W("**All three models FAIL or WEAK.** See the negative-result statement in §9.\n")
else:
    best=max(passing,key=lambda m:A[m]["effect"]/A[m]["baseline_between"])
    r=A[best]["effect"]/A[best]["baseline_between"]
    W(f"**{len(passing)} of 3 models pass.** **`{best}`** is the only model to satisfy the "
      f"directional test ({A[best]['n_sig']}/3 pre-registered hypotheses, all with the "
      "predicted sign), and it also shows the widest separation from its like-for-like floor "
      f"(effect {A[best]['effect']:.5f} vs baseline-arm between-slot "
      f"{A[best]['baseline_between']:.5f}, {r:.2f}×; the other models reach "
      f"{min(A[m]['effect']/A[m]['baseline_between'] for m in MODELS if m!=best):.2f}–"
      f"{max(A[m]['effect']/A[m]['baseline_between'] for m in MODELS if m!=best):.2f}×). "
      "It is also the only model whose persona arm *raises* negative affect over baseline "
      "and it shows the largest between-persona variance in P(negative) — the quantity "
      "`Q_emo` needs in order to have anything to work with.\n")
    if best=="vicuna-13b-v1.5":
        W("Vicuna passes, so Block B keeps planner and simulator matched and DialogXpert "
          "comparability is preserved.\n")
    else:
        W(f"Vicuna's verdict is **{verdicts['vicuna-13b-v1.5']}**, so Block B should run "
          f"`{best}` as the **user simulator** with Vicuna as the planner. That is a "
          "cross-model configuration, which satisfies the B3 cross-model robustness "
          "condition in the same runs. The simulator was chosen on measured "
          "persona-responsiveness, not convenience.\n")
    W(f"The main codebase should wire `{{persona}}` into the user-simulator prompt at the "
      "Phase-2 placement (after the static preamble and in-context example, immediately "
      "before the dialogue history). **Throughput must be re-measured afterwards**: persona "
      "text is unique per dialogue and cannot be shared in the radix tree.\n")

# 2 config
W("## 2. Configuration\n")
W("| Model | Checkpoint | Quantization |")
W("|---|---|---|")
for m in MODELS: W(f"| `{m}` | `{CKPT[m]}` | AWQ, 4-bit, group size 128 |")
W("")
W("Quantization verified from each checkpoint's `quantization_config` "
  "(`quant_method=awq`, `bits=4`, `group_size=128`), not assumed from the repo name.\n")
W("Sampling parameters **identical across all three models**: `temperature=0.7`, "
  "`top_p=0.9`, `max_tokens=64`, seeds `{0,1,2}`, 10 concurrent workers under SGLang 0.5.9, "
  "`--context-length 2048`, `--mem-fraction-static 0.82`, `--random-seed 0`.\n")
W("Qwen variant used: **standard `Qwen2.5-7B-Instruct`** (AWQ), not the `-1M` long-context "
  "variant used by RLFF-ESC; prompts here are short (<400 tokens). Qwen2.5 has no reasoning "
  "mode, so no `<think>` handling was needed.\n")
W("Emotion classifier: `j-hartmann/emotion-english-distilroberta-base` for all models, "
  "full 7-dim softmax stored, argmax never used. Its native label `joy` is reported as "
  "`happiness` (rename only).\n")

# 3 personas
W("## 3. Personas\n")
W(f"- Corpus profiles with all 23 dimensions complete: **{D['n_corpus_profiles']} / 1017** "
  "persuadee dialogues.\n"
  f"- Tertile boundaries computed across all **{D['n_corpus_profiles']}** corpus profiles, "
  "not across the 100 evaluation dialogues.\n"
  f"- Evaluation dialogues: 100 (pickle order, minus the 3 content-filtered dialogues the "
  "runner already excludes).\n"
  f"- **{len(personas)} of 100 had a complete profile; {len(D['excluded_dialogues'])} excluded**: "
  f"{', '.join('`'+d+'`' for d in D['excluded_dialogues'])} (no matching persuadee survey row).\n"
  f"- Gate (≥80 of 100) **passed**. All 23 dimensions verified within 1–6.\n")
W(f"Generation counts therefore use {len(personas)} personas: "
  f"{len(personas)} × 5 utterances × 3 seeds × 2 arms = **{len(personas)*30:,} per model**, "
  f"**{len(personas)*30*len(MODELS):,} total** (vs. the 3,000/9,000 planned for a full 100).\n")

# 4 rendered personas
W("## 4. Five rendered personas\n")
for p in personas[:5]:
    W(f"**#{p['persona_id']}** ({p['n_tokens']} tok) — neurotic={p['neurotic']:.1f}, "
      f"agreeable={p['agreeable']:.1f}, open={p['open']:.1f}\n")
    W(f"> {p['text']}\n")

# 5 phase 5
W("## 5. Manipulation check with noise floor (Phase 5)\n")
W("`noise` = mean JSD between seeds *within* the same persona and utterance (pure sampling "
  "stochasticity). `effect` = mean JSD between *different* personas' seed-averaged "
  "distributions, same utterance. Permutation test shuffles persona labels within each "
  "utterance, 1,000 iterations.\n")
W("| Model | noise (seed) | effect | ratio | baseline-arm between-slot | p_value | gate |")
W("|---|---|---|---|---|---|---|")
for m in MODELS:
    a=A[m]; W(f"| `{m}` | {a['noise']:.5f} | {a['effect']:.5f} | {a['ratio']:.2f} | "
              f"{a['baseline_between']:.5f} | {a['p_value']:.4f} | {a['gate']} |")
W("")
for m in MODELS: W(f"![null {m}](null_{m}.png)\n")
W("**Reading the ratio.** `effect` compares means of 3 samples while `noise` compares single "
  "samples, so the two are not on the same scale and the ratio runs below 1 even when the "
  "permutation test is decisive. The like-for-like floor is the **baseline-arm between-slot** "
  "column: identical aggregation (3 seeds averaged, 98 slots, same utterance) but with an "
  "empty persona, so it isolates what between-group JSD looks like when there is nothing to "
  "condition on. `effect` exceeding it is the interpretable manipulation signal, and the "
  "permutation test is the formal version of the same comparison.\n")

# 6 phase 6
W("## 6. Affective range (Phase 6)\n")
W("| Model | Arm | P(negative) | neutral | entropy | between-persona var of P(neg) | mean tokens |")
W("|---|---|---|---|---|---|---|")
for m in MODELS:
    for arm in ("persona","baseline"):
        d=A[m][arm]
        W(f"| `{m}` | {arm} | {pct(d['neg_fraction'])} | {pct(d['neutral_fraction'])} | "
          f"{d['entropy']:.4f} | {d['between_persona_var']:.3e} | {d['mean_tokens']:.1f} |")
W("")
W("**Token-length confound.** Persona responses are longer than baseline in every model "
  "(see the table). Longer text shifts classifier output independently of affect, so any "
  "arm difference in P(negative) is potentially confounded by length. Flagged here and "
  "quantified in §8b rather than silently corrected.\n")
W("**Neutral fraction is reported per arm** so that classifier insensitivity stays "
  "distinguishable from a genuine null.\n")
W("### 7. Baseline affective flatness across models\n")
W("Unconditioned simulator only — useful independently of the persona question.\n")
W("| Model | baseline P(negative) | baseline neutral | baseline entropy |")
W("|---|---|---|---|")
for m in MODELS:
    d=A[m]["baseline"]
    W(f"| `{m}` | {pct(d['neg_fraction'])} | {pct(d['neutral_fraction'])} | {d['entropy']:.4f} |")
W("")

# 8 correlations
W("## 8. Directional validity (Phase 7)\n")
W("Pre-registered hypotheses only. Spearman ρ, one-sided in the predicted direction, "
  "bootstrap 95% CI over 1,000 resamples of personas.\n")
W("| Model | H1 neurotic→P(fear+sad) | H2 agreeable→P(happy) | H3 open→P(surprise) |")
W("|---|---|---|---|")
for m in MODELS:
    row=[f"| `{m}` "]
    for h in ("H1_neurotic_neg","H2_agreeable_happy","H3_open_surprise"):
        c=A[m][h]
        row.append(f"| ρ={c['rho']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
                   f"p={c['p_one_sided']:.3f}{' **✓**' if c['sig'] else ''} ")
    W("".join(row)+"|")
W("")
W("H1 uses P(fear+sadness+anger+disgust) as pre-registered for the negative composite. "
  "No exploratory correlations over other dimensions were computed; the hypothesis set was "
  "frozen at three in Phase 0 and not extended.\n")
q=A.get("Qwen2.5-7B-Instruct")
if q and q["H1_neurotic_neg"]["ci"][0] < 0 < q["H1_neurotic_neg"]["ci"][1]:
    W("**Caveat on Qwen H1.** Its one-sided p is significant (p="
      f"{q['H1_neurotic_neg']['p_one_sided']:.3f}) but the bootstrap 95% CI "
      f"[{q['H1_neurotic_neg']['ci'][0]:+.3f}, {q['H1_neurotic_neg']['ci'][1]:+.3f}] crosses "
      "zero, so H1 is the weakest of the three and should be treated as suggestive rather "
      "than established. H2 is the only hypothesis whose CI excludes zero in every model "
      "that supports it.\n")

# ---- robustness ----
try: RB=json.load(open(f"{ROOT}/spike/robustness.json"))
except Exception: RB=None
if RB:
    W("### 8b. Robustness to response length (post-hoc, not pre-registered)\n")
    W("Persona responses are longer than baseline in every model, and longer text can shift "
      "the classifier independently of affect. Two post-hoc checks, reported separately from "
      "the pre-registered tests:\n")
    W("| Model | mean tokens persona / baseline | ρ(tok,neg) pooled | ρ(tok,neg) within persona / baseline | H1 partial ρ | H2 partial ρ | H3 partial ρ |")
    W("|---|---|---|---|---|---|---|")
    for m in MODELS:
        r=RB[m]; d=r["length_standardised_neg"]
        W(f"| `{m}` | {d['mean_tok_persona']:.1f} / {d['mean_tok_baseline']:.1f} "
          f"(+{100*(d['mean_tok_persona']/d['mean_tok_baseline']-1):.0f}%) "
          f"| {d['corr_tok_neg']:+.3f} "
          f"| {d['corr_tok_neg_persona']:+.3f} / {d['corr_tok_neg_baseline']:+.3f} "
          f"| {r['H1_neurotic_neg']['partial_rho']:+.3f} "
          f"| {r['H2_agreeable_happy']['partial_rho']:+.3f} "
          f"| {r['H3_open_surprise']['partial_rho']:+.3f} |")
    W("")
    W("| Model | P(neg) raw persona / baseline | raw ratio | length-standardised persona / baseline | standardised ratio |")
    W("|---|---|---|---|---|")
    for m in MODELS:
        d=RB[m]["length_standardised_neg"]
        W(f"| `{m}` | {pct(A[m]['persona']['neg_fraction'])} / {pct(A[m]['baseline']['neg_fraction'])} "
          f"| {d['raw_ratio']:.2f}× | {pct(d['persona_dstd'])} / {pct(d['baseline_dstd'])} "
          f"| {d['persona_dstd']/d['baseline_dstd']:.2f}× |")
    W("")
    qr=RB.get("Qwen2.5-7B-Instruct")
    if qr:
        d=qr["length_standardised_neg"]
        W("**How to read this for Qwen.** Persona responses are materially longer — "
          f"{d['mean_tok_persona']:.1f} vs {d['mean_tok_baseline']:.1f} tokens, "
          f"+{100*(d['mean_tok_persona']/d['mean_tok_baseline']-1):.1f}%, Mann-Whitney "
          "p≈5e-89 — so length has to be dealt with, not waved off. Nothing is truncated: "
          "0% of responses in either arm hit the 64-token cap.\n")
        W("**The direction of the length effect is the opposite of the usual confound "
          "story, and correcting for it makes the persona effect larger.** *Within* each arm, "
          "longer Qwen responses are slightly **less** negative "
          f"(ρ = {d['corr_tok_neg_persona']:+.3f} persona, "
          f"{d['corr_tok_neg_baseline']:+.3f} baseline). The near-zero pooled correlation "
          f"(ρ = {d['corr_tok_neg']:+.3f}) is Simpson's paradox: the persona arm is both "
          "longer and more negative, so mixing the arms cancels the within-arm slope. Since "
          "persona responses are longer, and length is mildly *anti*-correlated with "
          "negativity, the raw arm gap is **suppressed** by the length difference rather "
          "than inflated by it.\n")
        W(f"Standardising both arms to a common length distribution therefore raises the "
          f"gap: **{d['persona_dstd']/d['baseline_dstd']:.2f}×** "
          f"({pct(d['persona_dstd'])} vs {pct(d['baseline_dstd'])}) weighting bins by the "
          f"pooled length distribution, {d['persona']/d['baseline']:.2f}× weighting bins "
          "equally — the two targets agree, so the choice is not doing the work. It is "
          "consistent across the whole range (**persona exceeds baseline in 10 of 10 token "
          "bins**), and dropping the sparsest bins moves it *up* (1.72× at min cell n≥20, "
          "1.76× at n≥50), so it is not an artifact of thin cells. Bootstrap 95% CI "
          "**[1.49, 1.86]**, median 1.66.\n")
        W("**Which number to quote.** The length-adjusted **1.65×** is the better estimate of "
          "the persona effect on negative affect; the raw 1.31× understates it because the "
          "persona arm's extra verbosity pulls mildly against negativity. Both are reported "
          "above so the adjustment is visible.\n")
        W("**This does not silently upgrade the verdict.** The STRONG PASS criterion was "
          "operationalised as raw P(negative) ≥ 1.5× baseline; on the raw metric Qwen is "
          "1.31×, so the pre-registered verdict remains **PASS**. The length-adjusted ratio "
          "clears 1.5×, but switching metrics after seeing the data is exactly the move the "
          "pre-registration exists to prevent, and the bootstrap lower bound (1.49) sits "
          "right on the threshold. Reported as: PASS, with a length-adjusted effect that "
          "would meet the STRONG PASS bar.\n")
        W("**Correction.** An earlier draft of this section argued length could be dismissed "
          "because the pooled ρ(tokens, P(neg)) was ≈ 0. That pooled figure masked the "
          "within-arm relationship described above and should not have been used to dismiss "
          "the confound. The per-arm analysis replaces it. The conclusion strengthens rather "
          "than weakens, but the earlier reasoning was wrong.\n")
        W("The partial correlations remain the conservative reading: partialling out "
          f"per-persona mean token count attenuates H1 to p={qr['H1_neurotic_neg']['p_one_sided']:.4f} "
          f"and H3 to p={qr['H3_open_surprise']['p_one_sided']:.4f}, both just above 0.05, "
          "leaving only H2 significant. Because verbosity is partly *caused* by persona "
          "conditioning, partialling it out removes real persona signal too, so these are a "
          "lower bound rather than a corrected estimate. H1 and H3 remain the fragile "
          "components of the PASS.\n")

# 9 examples
W("## 9. Example response pairs\n")
W("Same utterance, contrasting personas (most vs. least neurotic), persona arm, seed 0.\n")
neu=sorted(personas,key=lambda p:p["neurotic"])
lo_p,hi_p=neu[0]["persona_id"],neu[-1]["persona_id"]
by={(r["model"],r["persona_id"],r["utterance_id"],r["arm"],r["seed"]):r for r in recs}
for m in MODELS:
    W(f"**`{m}`**\n")
    for u in ("emotion_appeal","proposition"):
        for tag,pid in (("low-neuroticism",lo_p),("high-neuroticism",hi_p)):
            r=by.get((m,pid,u,"persona",0))
            if r: W(f"- *{u}* / {tag} (#{pid}): “{r['response_text']}”")
        W("")
W("")

# 10 throughput
W("## 10. Throughput\n")
W("| Model | generations | seconds | gen/s | cache hit rate |")
W("|---|---|---|---|---|")
for m in MODELS:
    t=json.load(open(f"{ROOT}/spike/throughput_{m}.json"))
    si=t.get("server_info") or {}
    ch=si.get("cache_hit_rate") or (si.get("internal_states") or [{}])[0].get("cache_hit_rate")
    W(f"| `{m}` | {t['n']} | {t['seconds']:.0f} | {t['gen_per_sec']:.1f} | "
      f"{ch if ch is not None else 'n/a'} |")
W("")
W("**Cache hit rate was not captured.** `/get_server_info` in SGLang 0.5.9 returns the "
  "launch configuration, not runtime counters, and the servers ran at `--log-level warning`, "
  "which suppresses the per-batch `#cached-token` lines. Recovering it needs a re-run at "
  "`--log-level info` (or `--enable-metrics` and a scrape of `/metrics`); it is reported as "
  "missing rather than estimated. Structurally, both arms share the static preamble + "
  "in-context example prefix by construction, and the persona arm then diverges with ~75 "
  "unique tokens per persona — so the persona arm's prefix reuse is necessarily lower, which "
  "is the same effect that will reduce throughput once `{persona}` is wired into the main "
  "codebase.\n")
W("## 11. Deviations, incidents, and integrity notes\n")
W("Recorded so the result can be judged on how it was actually produced.\n")
W("1. **98 personas, not 100.** Two of the 100 evaluation dialogues have no matching "
  "persuadee survey row, so per-model counts are 2,940 rather than 3,000 and the total is "
  "8,820 rather than 9,000. The pre-registered gate (≥80/100) passed comfortably.\n")
W("2. **Prompt structure was corrected after the Phase-4 spot check, before any analysis.** "
  "The first Vicuna spot check returned *\"I'm just a computer program, so I don't have "
  "feelings or emotions\"* in both arms. Cause: SGLang's conversation builder "
  "(`conversation.py:612`) overwrites `conv.system_message` on every system message and "
  "renders it at the top of the prompt, so my second system message was destroying the "
  "persuadee-role preamble *and* relocating the persona to position zero. Fixed by keeping "
  "exactly one system message and carrying the persona at the head of the final user turn — "
  "same specified placement, no template dependence — plus an explicit instruction not to "
  "claim to be an AI. All reported data was generated after this fix; the affected "
  "generations were discarded. This is a prompt-plumbing fix, not a tuning step: "
  "`FACET_TEXT` was **not** revised at any point, before or after seeing results.\n")
W("3. **A wrong-model incident was detected and the affected data destroyed.** The first "
  "Llama and Qwen runs were silently served by a stale Vicuna server: SGLang's launcher "
  "survived `Popen.terminate()` and kept holding port 31411, the new servers failed to bind "
  "with `address already in use`, and my health check polled that port and got the old "
  "server's `200 OK` (\"server up in 1s\"). Those 5,880 generations were **deleted, not "
  "analysed**, and both models were regenerated after adding (a) a pre-flight check that the "
  "port is free, (b) a `/get_model_info` assertion that the responding server reports the "
  "expected checkpoint, and (c) process-group `SIGKILL` teardown that waits for the port to "
  "release. The re-runs logged `identity OK` against the correct checkpoints. Vicuna's data "
  "was unaffected — the server that answered it was Vicuna under identical launch flags — "
  "and was kept.\n")
W("4. **`seed` is a replicate index, not a per-request RNG seed.** SGLang's continuous "
  "batching makes per-request determinism depend on batch composition, so the three seeds "
  "give three independent draws per cell rather than reproducible individual samples. The "
  "server ran with `--random-seed 0`. Both arms receive the same seed structure, which is "
  "what the paired design requires.\n")
W("5. **Post-hoc analysis is labelled.** §8b (length control) was added after seeing the "
  "pre-registered results, because persona responses turned out materially longer. It is "
  "reported separately and does not restate the verdict.\n")
W("6. **Relevant existing code.** The main codebase already contains "
  "`src/utils/p4g_personas.py` (profile loading, `describe_profile`, `persona_suffix`) and "
  "calls `persona_suffix` in `src/players/p4g_players.py` at exactly the placement this "
  "spike validated. Nothing in `src/` was modified by this spike; all code and data live "
  "under `spike/`. Wiring Block B should reuse that hook rather than re-implement it.\n")
open(f"{ROOT}/spike/RESULTS.md","w").write("\n".join(L)+"\n")
print("wrote spike/RESULTS.md")
