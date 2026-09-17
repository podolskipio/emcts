"""Shared prompt construction + SGLang client for the persona spike."""
import json, os, requests
from concurrent.futures import ThreadPoolExecutor

ROOT = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS"
PORT = 31411
BASE = f"http://127.0.0.1:{PORT}"

MODELS = {
 "vicuna-13b-v1.5":        dict(path="TheBloke/vicuna-13B-v1.5-AWQ",
                                chat_template="vicuna_v1.1"),
 "Llama-3.1-8B-Instruct":  dict(path="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                                chat_template=None),
 "Qwen2.5-7B-Instruct":    dict(path="Qwen/Qwen2.5-7B-Instruct-AWQ",
                                chat_template=None),
}
# identical for every model / arm / persona / seed
SAMPLING = dict(temperature=0.7, top_p=0.9, max_tokens=64)

PREAMBLE = ("You are the Persuadee: a human taking part in a conversation. A Persuader is "
            "trying to persuade you to donate to a charity called Save the Children. "
            "Reply naturally as yourself, in one or two sentences. You are a person with "
            "your own feelings and opinions -- never say that you are an AI, a language "
            "model, or a computer program. Do not write any speaker name or label.")
EX_USER = "Persuader: Have you heard of Save the Children before?"
EX_ASST = "I think I've seen the name somewhere, but I don't really know what they do."
NEW_CONV = "The following is a new conversation between a Persuader and a Persuadee (you)."

def build_messages(persona_text: str, sys_utt: str):
    """Persona goes AFTER the static preamble and AFTER the in-context example, immediately
    before the dialogue history -- so the shared prefix (preamble + example + NEW_CONV) is
    byte-identical across every request and RadixAttention can share it.

    It is carried at the head of the final *user* turn rather than as a second system
    message: SGLang's conversation builder (conversation.py:612) overwrites
    `conv.system_message` on each system message and renders it at the top of the prompt,
    which would both destroy the preamble and move the persona to position zero. Keeping a
    single system message makes placement identical and template-independent for all three
    models."""
    marker = NEW_CONV + (f" The Persuadee: {persona_text}" if persona_text else "")
    return [
        {"role": "system",    "content": PREAMBLE},
        {"role": "user",      "content": EX_USER},
        {"role": "assistant", "content": EX_ASST},
        {"role": "user",      "content": f"{marker}\n\nPersuader: {sys_utt}"},
    ]

def build_jobs():
    personas = json.load(open(f"{ROOT}/spike/personas_rendered.json"))
    utts     = json.load(open(f"{ROOT}/spike/system_utterances.json"))
    jobs = []
    for p in personas:
        for u in utts:
            for seed in (0, 1, 2):
                for arm in ("persona", "baseline"):
                    jobs.append(dict(
                        persona_id=p["persona_id"], dialogue_id=p["dialogue_id"],
                        utterance_id=u["utterance_id"], seed=seed, arm=arm,
                        messages=build_messages(p["text"] if arm == "persona" else "",
                                                u["text"])))
    return jobs, personas, utts

_sess = requests.Session()
def one(job, model_path):
    body = dict(model=model_path, messages=job["messages"], **SAMPLING)
    r = _sess.post(f"{BASE}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]

def run_all(jobs, model_path, workers=10):
    out = [None] * len(jobs)
    def work(i):
        txt, n = one(jobs[i], model_path)
        out[i] = (txt, n)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(len(jobs))))
    return out
