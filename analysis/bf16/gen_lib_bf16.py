"""bf16 control: the spike harness with ONLY the checkpoint + dtype changed.

`spike/gen_lib.py` is imported rather than copied, so PREAMBLE, EX_USER, EX_ASST, NEW_CONV,
build_messages(), build_jobs() and SAMPLING are the *same objects* the spike ran with -- there
is no copy that can silently drift. This file adds nothing but a model table pointing at the
unquantized checkpoints.
"""
import os, sys

SPIKE = "/mnt/c/Users/Piotr/PycharmProjects/EMCTS/spike"
sys.path.insert(0, SPIKE)
import gen_lib as G  # noqa: E402  -- the spike's own harness, unmodified

ROOT, PORT, BASE = G.ROOT, G.PORT, G.BASE
SAMPLING = G.SAMPLING                      # temperature=0.7, top_p=0.9, max_tokens=64
build_messages, build_jobs, run_all = G.build_messages, G.build_jobs, G.run_all

# bf16 arm. chat_template=None everywhere: both checkpoints ship a chat template in their
# tokenizer_config, and the AWQ runs for these two models used the template the same way
# (only vicuna needed an explicit --chat-template, and vicuna is out of scope at bf16).
MODELS = {
    "Llama-3.1-8B-Instruct": dict(path="meta-llama/Llama-3.1-8B-Instruct",
                                  chat_template=None, awq_path=G.MODELS["Llama-3.1-8B-Instruct"]["path"]),
    "Qwen2.5-7B-Instruct":   dict(path="Qwen/Qwen2.5-7B-Instruct",
                                  chat_template=None, awq_path=G.MODELS["Qwen2.5-7B-Instruct"]["path"]),
}
