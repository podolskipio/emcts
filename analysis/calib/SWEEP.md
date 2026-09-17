# 1.2 Worker-count sweep — result

Run 1 config (8B / no persona), 24 dialogues per arm, everything else at the frozen grid
values in `run_calib.py:GRID`.

**Deviation from the plan: 24 dialogues per arm, not 10.** `rollout.py:334` clamps
`workers = max(1, min(--num_workers, n_dialogues))`, so a 10-dialogue run silently executes
10 workers whatever `--num_workers` says. At 10 dialogues the 16- and 24-worker arms would
have been byte-for-byte reruns of the 10-worker arm, and the "plateau" would have been an
artefact of the clamp rather than a measurement.

| workers | wall s | turns | s/turn | gen/s | gen tok/s | prompt tok/s | cache hit | peak VRAM | peak KV pool | peak queue | mean running reqs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **10** | **1341** | 105 | **12.78** | **16.48** | **670.5** | 20,116 | 0.925 | 23.24 GB | 0.095 | 0 | 7.5 |
| 16 | 1657 | 117 | 14.16 | 15.16 | 643.6 | 19,036 | 0.920 | 23.33 GB | 0.144 | 0 | 9.3 |
| 24 | 1805 | 118 | 15.29 | 13.18 | 556.3 | 15,452 | 0.921 | 23.38 GB | 0.167 | 0 | 14.9 |

## Adopted: 10 workers

Throughput does not plateau, it **degrades monotonically** — on every measure, including the
two that are invariant to how much work an arm happened to realize (s/turn and generated
tokens/s). 10 is the best of the three and is adopted for the grid and for runs 2 and 3.

## The ceiling is not KV cache, so the plan's two remedies do not apply

The plan says "if throughput plateaus, you are KV-cache bound. Try: fp8 KV cache
quantization, and shortening the in-context example." Neither would help here:

* **KV pool peaks at 16.7%** of `max_total_num_tokens = 105,843`. There is ~6x headroom.
  fp8 KV quantization buys capacity that is not the binding constraint.
* **`peak_queue_reqs = 0`** on all three arms: the server never had a backlog. Requests are
  admitted the moment they arrive.
* **The client is idle**: 6-19% of one core against 32 cores.
* **`sglang::scheduler` sits at 100% of a single core, already at 10 workers**, while GPU
  utilization reads 79-92%.

Caveat on that last row: in SGLang the scheduler process also drives the model forward loop,
so 100% CPU there is not on its own proof of a *scheduling* bottleneck. What the four rows
together do establish is that adding client concurrency cannot help, and that KV capacity is
not what is stopping it.

**Where the headroom actually is: request count, not request size.** The scheduler's cost is
per request, and this configuration issues ~135 LLM calls per dialogue turn (policy prior
16.8, value estimator 16.8, user simulator 50.7, system utterance 50.7 — measured, run 1).
The two planner roles are already 2 HTTP requests per scoring operation by design. The
untaken lever is collapsing the R realization samples of a shared prompt into one `n=R`
request instead of R separate ones (`gen_models.chat_generate_batched` currently fans out to
individual requests). Not attempted here — it is a code change, not a calibration knob — but
it is the direction that would buy grid hours, and it is worth pricing before committing the
full grid.

## Side observation, relevant to grid design

The three arms realize different amounts of work (105 / 117 / 118 turns) and different SR
(1.000 / 1.000 / 0.958) at identical settings. In p4g self-play `scenario = ()` — the
dialogue id is used only to label the episode and to look up a persona — so **without
`--p4g_persona` every dialogue in a cell is the same starting condition** and differs only by
sampling. The no-persona cells therefore carry no scenario variance at all; their error bars
are Bernoulli sampling noise over one scenario. The persona cells are the only ones where 100
dialogues are 100 different situations.
