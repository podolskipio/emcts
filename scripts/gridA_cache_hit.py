#!/usr/bin/env python3
"""Realization-cache hit rate of one Grid A run, from its frozen subtree logs.

Grid A stop condition: "cache hit rate drops below ~50% mid-grid -> stop and diagnose;
something changed in prompt serialization." The MCTS realization cache
(``OpenLoopMCTS._get_next_state``) keys on the serialized state string, so if prompt
serialization changes, previously-shared nodes stop colliding and the hit rate falls off
a cliff. That is the cheapest available canary for it.

What is measured: the fraction of logged tree EDGES whose node was served from the
realization cache at least once (``cache_hit`` in the frozen NDJSON schema -- a bool per
node, which is what the schema freeze gives us; the raw hit/miss counts are not in the
log and the schema may not be extended). It is therefore a proxy for the true
hits/(hits+misses) ratio, but a monotone one: it moves with the same underlying quantity
and is compared only against runs measured the same way.

Exit codes: 0 = at or above the threshold (or nothing to measure, e.g. --algo llm_raw,
which builds no tree), 2 = below.

    python3 scripts/gridA_cache_hit.py gridA/runs/a1__vicuna__nopersona__gdpzero__50s__seed0 --min 0.5
"""
import argparse
import glob
import gzip
import json
import os
import sys


def cache_hit_rate(run_dir):
    """``(hits, edges)`` over every subtree record in ``run_dir/subtree/*.ndjson.gz``."""
    hits = edges = 0
    for path in sorted(glob.glob(os.path.join(run_dir, "subtree", "*.ndjson.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                edges += 1
                hits += bool(rec.get("cache_hit"))
    return hits, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", help="run directory (the one holding subtree/)")
    ap.add_argument("--min", type=float, default=0.50,
                    help="fail (exit 2) if the rate is below this. Default 0.50, the Grid A "
                         "stop condition.")
    args = ap.parse_args()

    failed = False
    for run_dir in args.run_dir:
        hits, edges = cache_hit_rate(run_dir)
        name = os.path.basename(os.path.normpath(run_dir))
        if edges == 0:
            # llm_raw builds no tree; a search run with no edges is a different problem and
            # the sweep's episode-count check catches it.
            print(f"[cache] {name}: no subtree edges logged -- nothing to check")
            continue
        rate = hits / edges
        status = "ok" if rate >= args.min else "BELOW THRESHOLD"
        print(f"[cache] {name}: {rate:.3f} ({hits}/{edges} edges) min={args.min:.2f} {status}")
        if rate < args.min:
            failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
