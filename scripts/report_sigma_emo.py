"""Read a run's frozen subtree NDJSON logs and report the sigma_emo distribution.

    python scripts/report_sigma_emo.py outputs/<run>/subtree

This is the Part-3.3 smoke test on the new instrumentation, and a preview of the
variance analysis the ICDM reviewer's objection asks for: if sigma_emo is uniformly
zero on edges with N >= 2 the Welford update is wired wrong, and if it is uniformly
tiny the risk-adjusted rule (--emo_risk_lambda) has no headroom to work with.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys


def load(subtree_dir):
	files = sorted(glob.glob(os.path.join(subtree_dir, "*.ndjson.gz")))
	if not files:
		sys.exit(f"no *.ndjson.gz under {subtree_dir}")
	records = []
	for path in files:
		with gzip.open(path, "rt", encoding="utf-8") as f:
			records += [json.loads(line) for line in f if line.strip()]
	return files, records


def histogram(values, bins=10, width=44):
	if not values:
		return
	lo, hi = min(values), max(values)
	if hi <= lo:
		print(f"    all values == {lo:.6f}")
		return
	step = (hi - lo) / bins
	counts = [0] * bins
	for v in values:
		counts[min(bins - 1, int((v - lo) / step))] += 1
	peak = max(counts) or 1
	for i, c in enumerate(counts):
		left, right = lo + i * step, lo + (i + 1) * step
		bar = "#" * int(width * c / peak)
		print(f"    [{left:6.3f}, {right:6.3f})  {c:6d}  {bar}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("subtree_dir")
	ap.add_argument("--min_n", type=int, default=2,
					help="sigma_emo is defined as 0 below this visit count (ddof=1)")
	args = ap.parse_args()

	files, records = load(args.subtree_dir)
	edges = [r for r in records if r["parent_id"] is not None]
	visited = [r for r in edges if r["N"] >= args.min_n]
	sigmas = [r["sigma_emo"] for r in visited]
	nonzero = [s for s in sigmas if s > 0.0]

	print(f"{len(files)} dialogue log(s), {len(records)} records, {len(edges)} edge records")
	print(f"edges with N >= {args.min_n}: {len(visited)}")
	print(f"  sigma_emo > 0:  {len(nonzero)}  ({100.0 * len(nonzero) / max(1, len(visited)):.1f}%)")

	# --- the smoke test itself --------------------------------------------
	if not visited:
		sys.exit("FAIL: no edge reached N >= 2 — nothing to check")
	if not nonzero:
		sys.exit("FAIL: sigma_emo is uniformly zero on every N >= 2 edge — the Welford "
				 "update is not wired to the backup")
	print("  PASS: sigma_emo is non-zero on edges with N >= 2")

	q = statistics.quantiles(sigmas, n=100) if len(sigmas) > 1 else [sigmas[0]] * 99
	print("\nsigma_emo across edges with N >= %d:" % args.min_n)
	print(f"  mean   {statistics.fmean(sigmas):.4f}")
	print(f"  median {q[49]:.4f}")
	print(f"  p10 {q[9]:.4f}   p25 {q[24]:.4f}   p75 {q[74]:.4f}   p90 {q[89]:.4f}   p99 {q[98]:.4f}")
	print(f"  min {min(sigmas):.4f}   max {max(sigmas):.4f}")
	histogram(sigmas)

	# how big is the spread relative to the mean the channel actually uses?
	q_emo = [abs(r["Q_emo"]) for r in visited]
	print(f"\n|Q_emo| on the same edges: mean {statistics.fmean(q_emo):.4f}  max {max(q_emo):.4f}")
	ratios = [r["sigma_emo"] / abs(r["Q_emo"]) for r in visited if abs(r["Q_emo"]) > 1e-9]
	if ratios:
		rq = statistics.quantiles(ratios, n=100) if len(ratios) > 1 else [ratios[0]] * 99
		print(f"sigma_emo / |Q_emo|:      median {rq[49]:.2f}   p90 {rq[89]:.2f}")

	# per_step_valences is the raw tape; cross-check a few edges against it
	checked = mismatch = 0
	for r in visited[:2000]:
		zs = r["per_step_valences"]
		if len(zs) < 2:
			continue
		mean = sum(zs) / len(zs)
		var = sum((z - mean) ** 2 for z in zs) / (len(zs) - 1)
		checked += 1
		if abs(var ** 0.5 - r["sigma_emo"]) > 1e-9:
			mismatch += 1
	print(f"\ncross-check vs per_step_valences: {checked} edges, {mismatch} mismatches")
	if mismatch:
		sys.exit("FAIL: logged sigma_emo disagrees with the raw valence tape")


if __name__ == "__main__":
	main()
