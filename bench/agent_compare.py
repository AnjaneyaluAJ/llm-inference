"""Step 4: agent-style multi-turn traffic, prefix caching and KV offload. Plan: docs/step4_agent_kv_plan.md.

Four tp2 servers per (engine, configuration), one per GPU pair, on ports 8001-8004. bench/agentload.py
routes across them (sticky or least-loaded).

  --check   smoke test: start every (engine, configuration), send one turn, then for the cpu configuration
            run the offload correctness check and 100 GSM8K questions
  default   the sweep: per repeat, (engine, configuration) order shuffled, four loads shuffled

Engine options per configuration: configs/step4_engine_args.json.

Usage: python bench/agent_compare.py [--check] [--reps 2] [--out /data/results/step4] [--only vllm:cpu-sticky,...]
       [--loads 8,32,64,128] [--duration 600 --warmup 180]
"""
import argparse, json, random, time
from pathlib import Path

import sweep
from tool_compare import ENV

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-32B"
ARGS = json.loads((HERE.parent / "configs" / "step4_engine_args.json").read_text())
CONFIGS = {  # name: (engine argument set, routing)
    "nocache": ("nocache", "ll"), "gpu-ll": ("gpu", "ll"), "gpu-sticky": ("gpu", "sticky"),
    "cpu-ll": ("cpu", "ll"), "cpu-sticky": ("cpu", "sticky"),
}
ENGINES = ["vllm", "sglang"]
LOADS = [8, 32, 64, 128]
PORTS = [8001, 8002, 8003, 8004]
REPLICAS = ",".join(f"http://localhost:{p}" for p in PORTS)


def cmd(engine, argset, port):
    a = ARGS[engine]
    if engine == "vllm":
        return ["--model", MODEL, "--port", str(port), *a["base"], *a[argset]]
    return ["python3", "-m", "sglang.launch_server", "--model-path", MODEL, "--host", "0.0.0.0", "--port", str(port),
            *a["base"], *a[argset]]


def start(engine, argset, logdir, timeout):
    sweep.stop_all()
    t0 = time.time()
    names = []
    for i, p in enumerate(PORTS):
        sweep._run(f"eng{i}", engine, cmd(engine, argset, p), f"{2 * i},{2 * i + 1}", p)
        names.append(f"eng{i}")
    if not sweep._wait(PORTS, names, timeout):
        (logdir / "server.log").write_text(sweep.server_logs())
        sweep.stop_all()
        raise RuntimeError(f"{engine}/{argset} failed to start, see {logdir}/server.log")
    return time.time() - t0


def agentload(sessions, routing, seed, out, duration, warmup, extra=()):
    r = sweep.sh([sweep.PY, str(HERE / "agentload.py"), "--replicas", REPLICAS, "--model", MODEL, "--tokenizer", MODEL,
                  "--sessions", str(sessions), "--routing", routing, "--seed", str(seed), "--out", str(out),
                  "--duration", str(duration), "--warmup", str(warmup), *extra], env=ENV)
    if r.returncode:
        Path(f"{out}.err").write_text(r.stderr[-5000:])
    return (r.stdout.strip().splitlines() or [""])[-1][:220]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--out", default="/data/results/step4")
    ap.add_argument("--only", default="", help="comma list of engine:config")
    ap.add_argument("--loads", default=",".join(map(str, LOADS)))
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=180)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    loads = [int(x) for x in a.loads.split(",")]
    root = Path(a.out) / ("check" if a.check else "")
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.txt").write_text(sweep.sh("nvidia-smi; docker images --digests; uname -a; free -g; ls /data/hf/hub").stdout)
    gpufile = open(root / "gpu.csv", "a")
    gpulog = sweep.subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "1"], stdout=gpufile, stderr=sweep.subprocess.DEVNULL)
    reps = [0] if a.check else range(a.first_rep, a.first_rep + a.reps)
    try:
        for rep in reps:
            rng = random.Random(9000 + rep)
            if a.check:  # one server set per (engine, argument set); routing does not change the server
                order = [(e, c) for e in ENGINES for c in ("nocache", "gpu-sticky", "cpu-sticky")]
            else:
                order = [(e, c) for e in ENGINES for c in CONFIGS]
                rng.shuffle(order)
            for engine, conf in order:
                tag = f"{engine}:{conf}"
                if only and tag not in only:
                    continue
                argset, routing = CONFIGS[conf]
                d = root / (f"{engine}/{argset}" if a.check else f"rep{rep}/{engine}/{conf}")
                d.mkdir(parents=True, exist_ok=True)
                try:
                    up = start(engine, argset, d, a.startup_timeout)
                except RuntimeError as e:
                    print(f"[rep{rep}] {tag} START FAILED: {str(e)[:200]}", flush=True)
                    continue
                meta = {"engine": engine, "config": conf, "argset": argset, "routing": routing, "model": MODEL,
                        "image": sweep.IMAGES[engine], "cmd": cmd(engine, argset, PORTS[0]), "startup_s": up,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"[rep{rep}] {tag} up in {up:.0f}s", flush=True)
                if a.check:
                    msg = agentload(8, "sticky", 1, d / "smoke", 90, 30)
                    print(f"[check] {tag} smoke: {msg}", flush=True)
                    if argset == "cpu":
                        r = sweep.sh([sweep.PY, str(HERE / "offload_check.py"), "--base-url", "http://localhost:8001",
                                      "--model", MODEL, "--out-file", str(d / "offload_check.json")], env=ENV)
                        print(f"[check] {tag} offload: {(r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1][:200]}", flush=True)
                        r = sweep.sh([sweep.PY, str(HERE / "quality.py"), "--base-url", "http://localhost:8001", "--model", MODEL,
                                      "--n", "100", "--concurrency", "32", "--out", str(d / "quality.jsonl")], env=ENV)
                        print(f"[check] {tag} quality: {(r.stdout.strip().splitlines() or [''])[-1]}", flush=True)
                else:
                    lo = loads[:]; rng.shuffle(lo)
                    for s in lo:
                        out = d / f"s{s}"
                        if out.with_suffix(".json").exists():
                            continue
                        seed = 400000 + rep * 10000 + s  # same sessions for every engine and configuration within a repeat
                        t0 = time.strftime("%H:%M:%S", time.gmtime())
                        msg = agentload(s, routing, seed, out, a.duration, a.warmup)
                        print(f"[rep{rep}] {tag} s={s} ({t0}): {msg}", flush=True)
                (d / "server.log").write_text(sweep.server_logs())
                sweep.stop_all()
    finally:
        gpulog.terminate(); gpulog.wait(); gpufile.close()
        sweep.stop_all()
    print("STEP4_DONE" if not a.check else "CHECK_DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


if __name__ == "__main__":
    main()
