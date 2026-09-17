"""PHASE 2 - launch one bf16 model, spot-check, generate 2940, record cost telemetry.

Structure copied from spike/phase4_generate.py. Changed: the checkpoint (unquantized), an
explicit --dtype bfloat16 and no quantization flag, a configurable worker count, and peak-VRAM
sampling. Prompt construction and sampling come from spike/gen_lib via gen_lib_bf16, unchanged.

  usage: phase2_generate.py <model_key> [workers]
"""
import json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_lib_bf16 as B
import serve_lib as S

ROOT = B.ROOT
key = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = B.MODELS[key]
log_path = f"{HERE}/server_{key}_bf16.log"

print(f"\n===== {key}  bf16  ({cfg['path']})  workers={workers} =====", flush=True)
S.preflight()
proc = S.launch(cfg["path"], log_path, chat_template=cfg["chat_template"],
                dtype="bfloat16", mem_fraction=0.82)
vram = S.VRAMSampler(); vram.start()
try:
    load_s = S.wait_up(proc, key, log_path)
    info = S.assert_identity(cfg["path"])
    srv = S.server_info()
    kv = srv.get("max_total_num_tokens") or srv.get("max_total_tokens")
    print(f"  KV cache capacity: {kv} tokens | max_running_requests={srv.get('max_running_requests')}",
          flush=True)

    jobs, personas, utts = B.build_jobs()

    # ---- spot-check gate: first 10 generations, inspected before the full run ----
    print("  --- SPOT CHECK (first 10) ---", flush=True)
    sc = B.run_all(jobs[:10], cfg["path"], workers=workers)
    for j, (txt, n) in zip(jobs[:10], sc):
        print(f"   [{j['arm']:<8} {j['utterance_id']:<18} s{j['seed']}] ({n:>2}t) "
              f"{txt.strip()[:110]!r}", flush=True)
    json.dump([{"job": {k: v for k, v in j.items() if k != "messages"}, "text": t, "tok": n}
               for j, (t, n) in zip(jobs[:10], sc)],
              open(f"{HERE}/spotcheck_{key}_bf16.json", "w"), indent=1)

    # ---- full run ----
    t0 = time.time()
    res = B.run_all(jobs, cfg["path"], workers=workers)
    dt = time.time() - t0
    ntok = sum(n for _, n in res)
    print(f"  generated {len(res)} in {dt/60:.1f} min = {len(res)/dt:.2f} gen/s, "
          f"{ntok/dt:.1f} out-tok/s", flush=True)

    srv_after = S.server_info()
    peak = vram.stop()
    with open(f"{HERE}/raw_{key}_bf16.jsonl", "w") as f:
        for j, (txt, n) in zip(jobs, res):
            f.write(json.dumps({"model": key, "precision": "bf16",
                "persona_id": j["persona_id"], "dialogue_id": j["dialogue_id"],
                "utterance_id": j["utterance_id"], "seed": j["seed"], "arm": j["arm"],
                "response_text": txt.strip(), "token_count": n}) + "\n")
    json.dump({"model": key, "precision": "bf16", "checkpoint": cfg["path"],
               "workers": workers, "n": len(res), "seconds": dt,
               "gen_per_sec": len(res)/dt, "out_tok_per_sec": ntok/dt,
               "total_out_tokens": ntok, "load_seconds": load_s,
               "peak_vram_mib": peak, "kv_cache_tokens": kv,
               "max_running_requests": srv.get("max_running_requests"),
               "mem_fraction_static": 0.82, "context_length": 2048,
               "sampling": B.SAMPLING, "server_info": srv_after},
              open(f"{HERE}/throughput_{key}_bf16.json", "w"), indent=1)
    print(f"  peak VRAM {peak} MiB | wrote bf16/raw_{key}_bf16.jsonl", flush=True)
finally:
    vram.stop()
    S.teardown(proc, key)
