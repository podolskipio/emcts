"""Phase 2 - verbalize 23-dim profiles as natural language, capped at 80 tokens."""
import json
from transformers import AutoTokenizer

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
MAX_FACETS, MAX_TOKENS = 6, 80

FACET_TEXT = {
 ("extrovert","high"):"You are outgoing and enjoy talking with new people.",
 ("extrovert","low"):"You are reserved and keep to yourself.",
 ("agreeable","high"):"You are trusting and eager to help others.",
 ("agreeable","low"):"You are skeptical of other people's motives.",
 ("conscientious","high"):"You are organized and always follow through on what you promise.",
 ("conscientious","low"):"You are casual about plans and often leave things unfinished.",
 ("neurotic","high"):"You worry easily and are often anxious.",
 ("neurotic","low"):"You are emotionally steady and rarely rattled.",
 ("open","high"):"You are curious and drawn to new ideas.",
 ("open","low"):"You prefer the familiar and are wary of new ideas.",
 ("care","high"):"You are deeply moved by the suffering of others.",
 ("care","low"):"You are not easily swayed by appeals to suffering.",
 ("fairness","high"):"You care strongly that people are treated justly.",
 ("fairness","low"):"You think fairness matters less than other concerns.",
 ("loyalty","high"):"You are fiercely loyal to your family and country.",
 ("loyalty","low"):"You put little weight on group loyalty.",
 ("authority","high"):"You respect authority and established order.",
 ("authority","low"):"You are skeptical of authority and hierarchy.",
 ("purity","high"):"You value decency and are disgusted by degrading behaviour.",
 ("purity","low"):"You are unbothered by things others find improper.",
 ("freedom","high"):"You value personal liberty and resist being told what to do.",
 ("freedom","low"):"You accept limits on personal freedom for the common good.",
 ("conform","high"):"You follow the rules and avoid drawing attention to yourself.",
 ("conform","low"):"You do things your own way regardless of what is expected.",
 ("tradition","high"):"You hold to tradition and religious custom.",
 ("tradition","low"):"You place little value on tradition.",
 ("benevolence","high"):"You go out of your way to care for the people around you.",
 ("benevolence","low"):"You look after your own concerns before other people's.",
 ("universalism","high"):"You care about all people and the environment, not just your own circle.",
 ("universalism","low"):"You focus on your own circle rather than distant causes.",
 ("self_direction","high"):"You think for yourself and value your independence.",
 ("self_direction","low"):"You would rather be guided than decide everything alone.",
 ("stimulation","high"):"You crave excitement and are willing to take risks.",
 ("stimulation","low"):"You avoid risk and prefer a quiet, predictable life.",
 ("hedonism","high"):"You seek out pleasure and enjoy indulging yourself.",
 ("hedonism","low"):"You are restrained and rarely indulge yourself.",
 ("achievement","high"):"You are ambitious and want to be seen as successful.",
 ("achievement","low"):"You are not driven by ambition or status.",
 ("power","high"):"You want influence and control over resources.",
 ("power","low"):"You have little interest in wealth or power.",
 ("security","high"):"You need to feel safe and value stability above all.",
 ("security","low"):"You are comfortable living with uncertainty.",
 ("rational","high"):"You think decisions through carefully before acting.",
 ("rational","low"):"You do not overthink your decisions.",
 ("intuitive","high"):"You go with your gut feeling.",
 ("intuitive","low"):"You distrust gut feelings and want to see evidence.",
}

D = json.load(open(f"{ROOT}/spike/personas.json"))
DIMS, bounds = D["dims"], D["bounds"]
assert all((d,l) in FACET_TEXT for d in DIMS for l in ("high","low")), "missing facet text"
print(f"FACET_TEXT complete: {len(DIMS)} dims x 2 levels = {len(FACET_TEXT)} lines")

tok = AutoTokenizer.from_pretrained("TheBloke/vicuna-13B-v1.5-AWQ")

def render(p):
    scored = []
    for d in DIMS:
        lo, hi = bounds[d]
        if   p[d] >= hi: scored.append((p[d]-hi, FACET_TEXT[(d,"high")]))
        elif p[d] <= lo: scored.append((lo-p[d], FACET_TEXT[(d,"low")]))
    scored.sort(key=lambda x: -x[0])
    facets = [t for _, t in scored[:MAX_FACETS]]
    demo = f"You are {p['age']}, {p['employment'].lower()}, {p['edu'].lower()}."
    while facets and len(tok.encode(" ".join(facets+[demo]), add_special_tokens=False)) > MAX_TOKENS:
        facets.pop()
    return " ".join(facets + [demo])

out, lens = [], []
for p in D["personas"]:
    txt = render(p)
    n = len(tok.encode(txt, add_special_tokens=False)); lens.append(n)
    out.append({"persona_id": p["persona_id"], "dialogue_id": p["dialogue_id"],
                "text": txt, "n_tokens": n,
                **{d: p[d] for d in DIMS}})
json.dump(out, open(f"{ROOT}/spike/personas_rendered.json","w"), indent=1)

print(f"tokens: min={min(lens)} mean={sum(lens)/len(lens):.1f} max={max(lens)}  (cap {MAX_TOKENS})")
print(f"over cap: {sum(l>MAX_TOKENS for l in lens)}\n")
print("=== 5 RENDERED PERSONAS ===")
for r in out[:5]:
    print(f"\n[{r['persona_id']}] ({r['n_tokens']} tok) neurotic={r['neurotic']:.1f} "
          f"agreeable={r['agreeable']:.1f} open={r['open']:.1f}\n  {r['text']}")
