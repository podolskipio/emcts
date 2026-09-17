"""Part 3.2 acceptance check: diff a pre-change and a post-change regression run.

    python tests/compare_regression.py <dir with {arm}_pre.pkl / {arm}_post.pkl> [arm ...]

Asserts, for each arm, that SR and AvgT are identical and that the selected action
sequences match. Also diffs the complete episode records, which is strictly stronger:
if any of them differ the change is not a strict generalization and the grid must not
be started. Exit code is 0 only when everything matches.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from metrics.dialog_metrics import compute_metrics  # noqa: E402

MAX_TURNS = 10


def load(run_dir, stem):
	"""setup_output_dir re-points --output into <dir>/<stem>/<stem>.pkl."""
	with open(os.path.join(run_dir, stem, stem + ".pkl"), "rb") as f:
		return pickle.load(f)


def action_seqs(episodes):
	return [(e["did"], tuple(rec[1] for rec in e["history"] if rec[0] == "Persuader"))
			for e in episodes]


def main():
	run_dir = sys.argv[1]
	arms = sys.argv[2:] or ["beta0", "beta03"]
	ok = True
	for arm in arms:
		pre, post = load(run_dir, f"{arm}_pre"), load(run_dir, f"{arm}_post")
		m_pre = compute_metrics(pre, task="emo_p4g", max_turns=MAX_TURNS)
		m_post = compute_metrics(post, task="emo_p4g", max_turns=MAX_TURNS)
		seq_pre, seq_post = action_seqs(pre), action_seqs(post)
		# subtree_log is a path into the run directory, which differs by construction
		strip = lambda eps: [{k: v for k, v in e.items() if k != "subtree_log"} for e in eps]
		same_metrics, same_seq = m_pre == m_post, seq_pre == seq_post
		same_all = strip(pre) == strip(post)
		ok &= same_metrics and same_seq and same_all
		print(f"[{arm}]  episodes pre={len(pre)} post={len(post)}")
		print(f"  SR    pre={m_pre['SR']:.4f}  post={m_post['SR']:.4f}   identical={m_pre['SR'] == m_post['SR']}")
		print(f"  AvgT  pre={m_pre['AT']:.4f}  post={m_post['AT']:.4f}   identical={m_pre['AT'] == m_post['AT']}")
		print(f"  action sequences identical: {same_seq} "
			  f"({sum(len(s) for _, s in seq_pre)} actions over {len(seq_pre)} dialogues)")
		print(f"  full episode records identical: {same_all}")
		for (d, s1), (_, s2) in zip(seq_pre, seq_post):
			if s1 != s2:
				print(f"   DIFF {d}: {s1} != {s2}")
	print("\nRESULT:", "IDENTICAL — strict generalization confirmed" if ok
		  else "DIFFERENT — STOP, do not proceed to the grid")
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
