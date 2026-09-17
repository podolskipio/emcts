"""Launch / health-check / tear down one SGLang server, with VRAM sampling.

Lifted from spike/phase4_generate.py (launch, wait, identity assertion, process-group
teardown) so the serving path matches the spike. Additions: a dtype/quantization switch, a
configurable worker count, and a background nvidia-smi sampler for peak VRAM.
"""
import json, os, signal, socket, subprocess, sys, threading, time
import requests

SGL_PY = "/home/piotr/virtual_envs/SGLEnv/bin/python"
PORT = 31411
BASE = f"http://127.0.0.1:{PORT}"


def port_free():
    with socket.socket() as sk:
        return sk.connect_ex(("127.0.0.1", PORT)) != 0


def wait_port_free(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_free():
            return True
        time.sleep(2)
    return False


def launch(model_path, log_path, chat_template=None, dtype="bfloat16",
           mem_fraction=0.82, context_length=2048, extra=None):
    cmd = [SGL_PY, "-m", "sglang.launch_server",
           "--model-path", model_path, "--host", "127.0.0.1", "--port", str(PORT),
           "--context-length", str(context_length),
           "--mem-fraction-static", str(mem_fraction),
           "--random-seed", "0", "--log-level", "info"]
    if dtype:
        cmd += ["--dtype", dtype]
    if chat_template:
        cmd += ["--chat-template", chat_template]
    if extra:
        cmd += list(extra)
    log = open(log_path, "w")
    print("  launch:", " ".join(cmd[2:]), flush=True)
    # own process group so teardown kills the whole SGLang tree, not just the launcher
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_up(proc, key, log_path, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"FATAL: server for {key} died; see {log_path}")
        try:
            if requests.get(f"{BASE}/health_generate", timeout=5).status_code == 200:
                print(f"  server up in {time.time()-t0:.0f}s", flush=True)
                return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    sys.exit(f"FATAL: server for {key} did not come up in {timeout}s")


def assert_identity(expected):
    """A stale server on the port would answer health checks and serve the WRONG model."""
    info = requests.get(f"{BASE}/get_model_info", timeout=30).json()
    served = info.get("model_path", "")
    if served != expected:
        sys.exit(f"FATAL: port {PORT} is serving {served!r}, expected {expected!r}.")
    print(f"  identity OK: serving {served}", flush=True)
    return info


class VRAMSampler(threading.Thread):
    """Poll nvidia-smi for peak used MiB while a run is in flight."""
    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                self.peak = max(self.peak, int(out.stdout.strip().split("\n")[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        return self.peak


def server_info():
    try:
        return requests.get(f"{BASE}/get_server_info", timeout=10).json()
    except Exception:
        return {}


def teardown(proc, key):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        try:
            proc.wait(timeout=45)
            break
        except Exception:
            pass
    ok = wait_port_free()
    print(f"  server for {key} stopped; port free={ok}", flush=True)
    if not ok:
        print(f"  WARNING: port {PORT} still held after teardown", flush=True)
    return ok


def preflight():
    if not port_free():
        sys.exit(f"FATAL: port {PORT} already in use; kill the stale SGLang server first.")
