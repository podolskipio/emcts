"""One-off: convert GDP-Zero's P4G pickle to a JSON-lines .txt matching ESC / CB.

The input ``300_dialog_turn_based.pkl`` is a ``{did: {"dialog": [{er, ee}], "label": [{er, ee}]}}``
dict where ``er`` / ``ee`` are *lists* of sentences and ``label.er`` / ``label.ee`` are the
matching lists of per-sentence dialog acts.

Output is one JSON object per dialog (JSON-lines), shaped like DPDP's ``esc-valid.txt`` /
``cb-valid.txt``: each system turn (joined ``er`` sentences) and each user turn (joined ``ee``
sentences) becomes one ``{speaker, text, strategy}`` entry. The "strategy" is the *last* label
on that side of the turn — the same rule ``runners/_common.read_p4g`` already uses for the user
side; both sides keep the raw hyphenated label from the dataset (e.g. ``credibility-appeal``).

    cd src
    python runners/convert_p4g_to_jsonl.py \
        --input data/p4g/300_dialog_turn_based.pkl \
        --output data/p4g/p4g-valid.txt
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # put src/ on the path

import json
import pickle
import argparse


# Mirror of `PersuasionGame`'s system dialog acts (we don't import the game class here so the
# converter has no torch/transformers dependency). Used to pick a system-side ``strategy`` that
# resolves to a known DA, matching ``runners/_common.read_p4g``'s set-intersection logic.
P4G_SYSTEM_DAS = {
	"personal story", "credibility appeal", "emotion appeal", "proposition of donation",
	"foot in the door", "logical appeal", "self modeling", "task related inquiry",
	"source related inquiry", "personal related inquiry", "neutral to inquiry",
	"greeting", "other",
}


def _last(labels):
	"""Return the last non-empty label in ``labels`` or ``None`` if there are none."""
	for lab in reversed(labels or []):
		if lab:
			return lab
	return None


def _pick_sys_strategy(labels):
	"""Match ``read_p4g``'s pickle path: any label that is itself a known game DA, else ``"other"``.

	The raw P4G labels are hyphenated (e.g. ``credibility-appeal``) while the game ontology uses
	spaces (e.g. ``credibility appeal``), so in practice only ``greeting`` / ``other`` ever match.
	Keeping this exact behaviour means the JSON-lines file is a drop-in for the original pickle.
	"""
	for lab in (labels or []):
		if lab and lab in P4G_SYSTEM_DAS:
			return lab
	return "other"


def convert(pickle_path, output_path):
	with open(pickle_path, "rb") as f:
		all_dialogs = pickle.load(f)

	out_dir = os.path.dirname(output_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	n_dialogs = 0
	n_turns = 0
	with open(output_path, "w", encoding="utf-8") as f:
		for did, dialog in all_dialogs.items():
			turns = dialog.get("dialog", [])
			labels = dialog.get("label", [])
			out_turns = []
			for t, turn in enumerate(turns):
				# GDP-Zero convention: a turn with no user side means the dialog has ended -- stop here
				if len(turn.get("ee", [])) == 0:
					break
				er = [s.strip() for s in turn.get("er", []) if s and s.strip()]
				ee = [s.strip() for s in turn.get("ee", []) if s and s.strip()]
				er_labs = labels[t].get("er", []) if t < len(labels) else []
				ee_labs = labels[t].get("ee", []) if t < len(labels) else []
				if er:
					out_turns.append({"speaker": "sys", "text": " ".join(er), "strategy": _pick_sys_strategy(er_labs)})
				if ee:
					out_turns.append({"speaker": "usr", "text": " ".join(ee), "strategy": _last(ee_labs)})
			if not out_turns:
				continue
			f.write(json.dumps({"id": did, "dialog": out_turns}, ensure_ascii=False) + "\n")
			n_dialogs += 1
			n_turns += len(out_turns)
	print(f"wrote {n_dialogs} dialogs ({n_turns} utterances) -> {output_path}")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve(path):
	"""Treat relative paths as relative to the repo root, so the script works from any CWD."""
	return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--input", default="data/p4g/300_dialog_turn_based.pkl",
						help="path to the GDP-Zero P4G pickle (relative paths are resolved from the repo root)")
	parser.add_argument("--output", default="data/p4g/p4g-valid.txt",
						help="output JSON-lines file (relative paths are resolved from the repo root)")
	args = parser.parse_args()
	convert(_resolve(args.input), _resolve(args.output))


if __name__ == "__main__":
	main()
