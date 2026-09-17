#!/usr/bin/env python3
"""Phase 2: recompute the grid budget from the measured Phase-1 rows.

Deliberately written out longhand -- one named variable per cell, the arithmetic spelled out
rather than folded into a loop -- because this is the number the whole grid is bought with and
it has to be checkable by eye against calib/throughput.json.
"""
import json, os

CALIB = os.path.dirname(os.path.abspath(__file__))
rows = {r["tag"]: r for r in json.load(open(os.path.join(CALIB, "throughput.json")))}

N_CELL = 100          # dialogues per grid cell (README results table)
H = 3600.0

# --- the four factorial cell costs -------------------------------------------------------
# Three are scaled from 24-dialogue calibration runs; the fourth was measured at full size.
r8n, r8p, r13p = rows["run1_d24_w10"], rows["run2_d24_w10"], rows["run3_d24_w10"]
r13n = rows["confirm_13b_nopersona_d100"]

cell_8b_nopersona  = r8n["wall_clock_s"]  * (N_CELL / r8n["dialogues"])  / H
cell_8b_persona    = r8p["wall_clock_s"]  * (N_CELL / r8p["dialogues"])  / H
cell_13b_persona   = r13p["wall_clock_s"] * (N_CELL / r13p["dialogues"]) / H
cell_13b_nopersona_measured = r13n["wall_clock_s"] / H       # already 100 dialogues
cell_13b_nopersona_plan     = 10.0                            # the plan's stated figure

# --- factorial, both ways ----------------------------------------------------------------
factorial_measured = (6 * cell_8b_nopersona + 6 * cell_8b_persona
                      + 3 * cell_13b_nopersona_measured + 3 * cell_13b_persona)
factorial_plan     = (6 * cell_8b_nopersona + 6 * cell_8b_persona
                      + 3 * cell_13b_nopersona_plan + 3 * cell_13b_persona)

BLOCKS = {"A1_extra_seeds": 20, "A2": 18, "A3": 1, "B1": 28, "B2": 30}
blocks = sum(BLOCKS.values())

total_measured = factorial_measured + blocks
total_plan     = factorial_plan + blocks

def band(t):
    if t < 230:  return "< 230 h", "Proceed. Spend slack on a 2nd seed for B2 first."
    if t <= 280: return "230-280 h", "Proceed. Cut B1 to 3 points in one cell; hold C-tier at zero."
    return "> 280 h", "Cut: persona cells -> 1 seed, B1 -> one cell, Llama cells drop A2, 8B -> 1 seed."

out = {
    "n_cell": N_CELL, "adopted_workers": 10,
    "cells_h": {
        "8b_nopersona": round(cell_8b_nopersona, 3),
        "8b_persona": round(cell_8b_persona, 3),
        "13b_nopersona_measured": round(cell_13b_nopersona_measured, 3),
        "13b_nopersona_plan": cell_13b_nopersona_plan,
        "13b_persona": round(cell_13b_persona, 3),
    },
    "factorial_measured_h": round(factorial_measured, 1),
    "factorial_plan_anchor_h": round(factorial_plan, 1),
    "blocks_h": blocks,
    "total_measured_h": round(total_measured, 1),
    "total_plan_anchor_h": round(total_plan, 1),
    "band_measured": band(total_measured),
    "band_plan_anchor": band(total_plan),
    "available_h": 280, "plan_estimate_h": 227,
    "slack_measured_h": round(280 - total_measured, 1),
}
json.dump(out, open(os.path.join(CALIB, "budget_computed.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
