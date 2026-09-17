"""Phase 4 - launch SGLang per model, spot-check, generate 2940, save raw."""
import json, os, signal, socket, subprocess, sys, time, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_lib as G

ROOT, SC = G.ROOT, os.environ["SC"]
key = sys.argv[1]
cfg = G.MODELS[key]

def port_free():
    with socket.socket() as sk:
        return sk.connect_ex(("127.0.0.1", G.PORT)) != 0

def wait_port_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_free(): return True
        time.sleep(2)
    return False

def launch():
    cmd = ["/home/piotr/virtual_envs/SGLEnv/bin/python", "-m", "sglang.launch_server",
           "--model-path", cfg["path"], "--host", "127.0.0.1", "--port", str(G.PORT),
           "--context-length", "2048", "--mem-fraction-static", "0.82",
           "--random-seed", "0", "--log-level", "warning"]
    if cfg["chat_template"]: cmd += ["--chat-template", cfg["chat_template"]]
    log = open(f"{SC}/server_{key}.log", "w")
    # own process group so teardown can kill the whole SGLang tree, not just the launcher
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)

def wait(proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"FATAL: server for {key} died; see {SC}/server_{key}.log")
        try:
            if requests.get(f"{G.BASE}/health_generate", timeout=5).status_code == 200:
                print(f"  server up in {time.time()-t0:.0f}s", flush=True); return
        except Exception: pass
        time.sleep(5)
    sys.exit(f"FATAL: server for {key} did not come up in {timeout}s")

print(f"\n===== {key}  ({cfg['path']}) =====", flush=True)
# PRE-FLIGHT: a stale server still holding the port would answer our health check and
# silently serve the WRONG model, so refuse to start until the port is genuinely free.
if not port_free():
    sys.exit(f"FATAL: port {G.PORT} already in use before launching {key}; "
             "kill the stale SGLang server first.")
proc = launch()
try:
    wait(proc)
    # IDENTITY ASSERTION: confirm the server answering us is actually this model.
    info = requests.get(f"{G.BASE}/get_model_info", timeout=30).json()
    served = info.get("model_path", "")
    if served != cfg["path"]:
        sys.exit(f"FATAL: port {G.PORT} is serving {served!r}, expected {cfg['path']!r}. "
                 "Refusing to generate against the wrong model.")
    print(f"  identity OK: serving {served}", flush=True)
    jobs, personas, utts = G.build_jobs()

    # ---- spot-check gate: first 10 generations, inspected before the full run ----
    print("  --- SPOT CHECK (first 10) ---", flush=True)
    sc_out = G.run_all(jobs[:10], cfg["path"], workers=10)
    for j, (txt, n) in zip(jobs[:10], sc_out):
        print(f"   [{j['arm']:<8} {j['utterance_id']:<18} s{j['seed']}] ({n:>2}t) "
              f"{txt.strip()[:110]!r}", flush=True)
    json.dump([{"job": {k: v for k, v in j.items() if k != "messages"},
                "text": t, "tok": n} for j, (t, n) in zip(jobs[:10], sc_out)],
              open(f"{ROOT}/spike/spotcheck_{key}.json", "w"), indent=1)

    # ---- full run ----
    t0 = time.time()
    res = G.run_all(jobs, cfg["path"], workers=10)
    dt = time.time() - t0
    print(f"  generated {len(res)} in {dt/60:.1f} min = {len(res)/dt:.1f} gen/s", flush=True)

    # cache-hit rate straight from the server
    try: cache = requests.get(f"{G.BASE}/get_server_info", timeout=10).json()
    except Exception: cache = {}

    with open(f"{ROOT}/spike/raw_{key}.jsonl", "w") as f:
        for j, (txt, n) in zip(jobs, res):
            f.write(json.dumps({"model": key, "persona_id": j["persona_id"],
                "dialogue_id": j["dialogue_id"], "utterance_id": j["utterance_id"],
                "seed": j["seed"], "arm": j["arm"],
                "response_text": txt.strip(), "token_count": n}) + "\n")
    json.dump({"model": key, "checkpoint": cfg["path"], "n": len(res),
               "seconds": dt, "gen_per_sec": len(res)/dt,
               "sampling": G.SAMPLING, "server_info": cache},
              open(f"{ROOT}/spike/throughput_{key}.json", "w"), indent=1)
    print(f"  wrote spike/raw_{key}.jsonl", flush=True)
finally:
    # kill the whole process group; SGLang's scheduler/detokenizer children survive a
    # plain terminate() on the launcher and keep holding the port.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try: os.killpg(os.getpgid(proc.pid), sig)
        except Exception: pass
        try:
            proc.wait(timeout=45); break
        except Exception: pass
    ok = wait_port_free()
    print(f"  server for {key} stopped; port free={ok}", flush=True)
    if not ok: print(f"  WARNING: port {G.PORT} still held after teardown", flush=True)
