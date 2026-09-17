"""§4 acceptance: (1) pre/post subtree logs + emotion records identical, (2) simlog internal alignment.

    python analysis/phase1/scripts/check_instrumentation.py <pre_run_dir> <post_run_dir>

Alignment checks need no ordering assumption:
  a. per edge (turn, prefix, action), the step records' z in simulation order == subtree per_step_valences
  b. every step's parent realization id is traceable: depth 1 -> the turn's observed root
     (parent_nu == turn.root_nu); depth >= 2 -> some step in the same tree at depth-1 whose
     child_realization_id equals it, with child_nu == parent_nu and identical distributions.
  c. z == child_nu under --emo_signal level.
"""
import glob, gzip, json, os, sys
from collections import defaultdict

def load(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]

def main(pre, post):
    """pre == "-" skips the pre/post comparison and runs only the alignment checks on post (real runs)."""
    ok = True
    for sub in (("subtree",) if pre != "-" else ()):
        a = sorted(glob.glob(f"{pre}/{sub}/*.ndjson.gz")); b = sorted(glob.glob(f"{post}/{sub}/*.ndjson.gz"))
        same = [os.path.basename(x) for x in a] == [os.path.basename(x) for x in b] and all(load(x) == load(y) for x, y in zip(a, b))
        print(f"{sub} logs identical pre/post: {same} ({len(a)} files)"); ok &= same
    if pre != "-":
        ea = json.load(open(glob.glob(f"{pre}/*_emotions.json")[0])); eb = json.load(open(glob.glob(f"{post}/*_emotions.json")[0]))
        print(f"emotion classifier records identical: {ea == eb} ({len(ea)})"); ok &= ea == eb

    n_steps = n_edges = 0
    for path in sorted(glob.glob(f"{post}/simlog/*.ndjson.gz")):
        recs = load(path)
        steps = [r for r in recs if r["record_type"] == "step"]
        turns = {r["turn_index"]: r for r in recs if r["record_type"] == "turn"}
        sub_path = os.path.join(post, "subtree", os.path.basename(path))
        if not steps:  # dialogue ended on the greeting: no planned turn, no tree, no subtree file
            ok &= not os.path.exists(sub_path) and len(turns) == 1
            print(f"{os.path.basename(path)}: no planned turn (turn records={len(turns)}), no subtree file: {not os.path.exists(sub_path)}")
            continue
        subtree = load(sub_path)
        n_steps += len(steps)
        # a
        tape = defaultdict(list)
        for s in sorted(steps, key=lambda s: (s["turn_index"], s["simulation_index"], s["depth"])):
            tape[(s["turn_index"], tuple(s["action_prefix"]), s["action"])].append(s["z"])
        root_len = {}
        for r in subtree:
            if r["parent_id"] is None:
                root_len[r["turn"]] = len(r["action_seq"])
        sub_tape = {}
        for r in subtree:
            if r["parent_id"] is None or r["N"] == 0:
                continue
            k = (r["turn"], tuple(r["action_seq"][root_len[r["turn"]]:-1]), r["action_seq"][-1])
            sub_tape[k] = r["per_step_valences"]
        a_ok = dict(tape) == sub_tape
        n_edges += len(sub_tape)
        # b, c
        b_ok = c_ok = True
        by_tree_child = defaultdict(dict)
        for s in steps:
            by_tree_child[(s["turn_index"], s["depth"])][s["child_realization_id"]] = s
        for s in steps:
            c_ok &= s["z"] == s["child_nu"]
            if s["depth"] == 1:
                b_ok &= abs(s["parent_nu"] - turns[s["turn_index"]]["root_nu"]) < 1e-12 and \
                        s["parent_emotion_dist"] == turns[s["turn_index"]]["root_emotion_dist"]
            else:
                src = by_tree_child[(s["turn_index"], s["depth"] - 1)].get(s["parent_realization_id"])
                b_ok &= src is not None and src["child_nu"] == s["parent_nu"] and src["child_emotion_dist"] == s["parent_emotion_dist"]
        print(f"{os.path.basename(path)}: steps={len(steps)} edges={len(sub_tape)}  z-tape==per_step_valences:{a_ok}  parent traceable:{b_ok}  z==child_nu:{c_ok}")
        ok &= a_ok and b_ok and c_ok
    print(f"total steps={n_steps} edges={n_edges}\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
