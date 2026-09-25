"""Shared setup for the readiness Phase 1 scripts: the frozen grid's agents, built through the runners'
own build_agents so every prompt is the one the planner uses. Only the simulator temperature can be
changed, and only where a script names it (Phase 1D).

The frozen environment (analysis/grid/runs/*/run_record.json): Vicuna-13B-AWQ on SGLang, top_p 0.9 /
top_k 40, --p4g_persona ON, --logit_scoring off, --emotion_classifier hf, --emo_valence_table generic,
simulator T 1.1.
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "analysis", "thu", "scripts"))

from games import PersuasionGame  # noqa: E402
from utils.sessions import DialogSession  # noqa: E402
from runners._common import (  # noqa: E402
	build_agents, make_backbone_model, P4G_BAD_DIALOGS, _read_p4g_pickle,
)
from mcts.emotion_mcts import EMOTION_VALENCE_GENERIC  # noqa: E402

READINESS = os.path.join(REPO, "analysis", "readiness")
SGLANG_MODEL = "TheBloke/vicuna-13B-v1.5-AWQ"
FROZEN_USER_T = 1.1
SYS_ACTS = PersuasionGame.get_game_ontology()["system"]["dialog_acts"]
USER_ACTS = PersuasionGame.get_game_ontology()["user"]["dialog_acts"]

_backbone = None


def backbone():
	global _backbone
	if _backbone is None:
		_backbone, family = make_backbone_model("sglang", sglang_model=SGLANG_MODEL)
		assert family == "chat"
	return _backbone


def user_inference_args(temperature=FROZEN_USER_T):
	"""build_agents' default simulator args with only the temperature replaced."""
	return {"max_new_tokens": 128, "temperature": float(temperature), "repetition_penalty": 1.0,
			"do_sample": True, "return_full_text": False, "top_p": 0.9, "top_k": 40}


def agents(persona, user_temperature=FROZEN_USER_T):
	"""(game, system, user, planner) exactly as rollout.py builds them for the frozen grid."""
	kw = {}
	if user_temperature != FROZEN_USER_T:
		kw["usr_inference_args"] = user_inference_args(user_temperature)
	return build_agents("p4g", backbone(), "chat", persona=persona, llm_prior_topk=5, logit_scoring="off", **kw)


def personas():
	from utils.p4g_personas import load_persona_texts
	return load_persona_texts()


def annotated_dialogs():
	"""The annotated P4G dialogues as {id: [turn dicts]} (runners' reader; drops P4G_BAD_DIALOGS)."""
	path = os.path.join(REPO, "data", "p4g", "300_dialog_turn_based.pkl")
	return {d["id"]: d["turns"] for d in _read_p4g_pickle(path, set(SYS_ACTS))}


def session(turns, end_on="usr"):
	"""A DialogSession of ``turns``; end_on="sys" drops the last persuadee reply."""
	s = DialogSession(PersuasionGame.SYS, PersuasionGame.USR)
	for i, t in enumerate(turns):
		s.add_single(PersuasionGame.SYS, t["sys_da"], t["sys_utt"])
		if end_on == "sys" and i == len(turns) - 1:
			break
		s.add_single(PersuasionGame.USR, t["usr_da"], t["usr_utt"])
	return s


def nu(dist):
	"""nu under the frozen generic valence table, from an {Emotions: p} distribution."""
	return float(sum(p * EMOTION_VALENCE_GENERIC.get(e, 0.0) for e, p in dist.items()))


def hf_classifier():
	from emotion_classifiers.hf_emotion import HFEmotionClassifier
	return HFEmotionClassifier()


def git_head():
	import subprocess
	return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
