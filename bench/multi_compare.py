"""Step 2: engines x tensor/data-parallel splits on 8 GPUs, measured with both loadgen and AIPerf.

Qwen3-32B (BF16, 64 GB) on 8x L40S (g6e.48xlarge, PCIe, no NVLink). Splits that use all 8 GPUs:
  tp8     one server, tensor parallel over 8 GPUs
  tp4dp2  two tensor-parallel-4 replicas
  tp2dp4  four tensor-parallel-2 replicas
(TP1 does not fit: 64 GB of weights > 48 GB per GPU.)

For each repeat, split and engine (both shuffled per repeat), every workload x concurrency point is
measured by loadgen and AIPerf back to back on the same running server, the tool going first
alternating, as in bench/tool_compare.py. loadgen always saves raw streams of text-less responses.
In repeat 1 each server also answers 100 GSM8K questions (correctness check across splits).

Usage: python bench/multi_compare.py [--reps 2] [--out /data/results/step2] [--only tp8:vllm] [--points ...]
"""
import argparse, json, random, time
from pathlib import Path

import sweep
from tool_compare import AIPERF, ENV, run_aiperf, run_loadgen

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-32B"
SPLITS = {"tp8": (8, 1), "tp4dp2": (4, 2), "tp2dp4": (2, 4)}
ENGINES = ["vllm", "sglang", "trtllm"]
DEFAULT_POINTS = "chat:1,16,64,256;rag:1,16,64,256;generate:1,16,64,256"


def parse_points(s):
    return {w: [int(c) for c in cs.split(",")] for w, cs in (x.split(":") for x in s.split(";"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--out", default="/data/results/step2")
    ap.add_argument("--points", default=DEFAULT_POINTS)
    ap.add_argument("--only", default="", help="comma list of split:engine to run, e.g. tp8:vllm,tp2dp4:trtllm")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--startup-timeout", type=int, default=2400)
    a = ap.parse_args()
    points = parse_points(a.points)
    only = set(filter(None, a.only.split(",")))
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.txt").write_text(sweep.sh(f"nvidia-smi; nvidia-smi topo -m; docker images --digests; uname -a; "
                                           f"lscpu | head -30; {AIPERF} --version").stdout)
    gpufile = open(root / "gpu.csv", "a")
    gpulog = sweep.subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "2"], stdout=gpufile, stderr=sweep.subprocess.DEVNULL)
    try:
        k = 0
        for rep in range(a.first_rep, a.first_rep + a.reps):
            rng = random.Random(3000 + rep)
            splits = list(SPLITS); rng.shuffle(splits)
            for split in splits:
                engines = ENGINES[:]; rng.shuffle(engines)
                for engine in engines:
                    if only and f"{split}:{engine}" not in only:
                        continue
                    tp, dp = SPLITS[split]
                    d = root / f"rep{rep}" / split / engine
                    d.mkdir(parents=True, exist_ok=True)
                    try:
                        up = sweep.start(engine, "defaults", d, MODEL, tp, dp, a.startup_timeout)
                    except RuntimeError as e:
                        print(f"[rep{rep}] {split} {engine} START FAILED: {e}", flush=True)
                        continue
                    (d / "meta.json").write_text(json.dumps({
                        "engine": engine, "split": split, "tp": tp, "dp": dp, "model": MODEL, "image": sweep.IMAGES[engine],
                        "cmd": sweep.engine_cmd(engine, "defaults", MODEL, tp, dp), "startup_s": up,
                        "dp_method": "nginx least_conn over separate servers" if engine == "trtllm" and dp > 1 else "engine built-in",
                        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
                    print(f"[rep{rep}] {split} {engine} up in {up:.0f}s", flush=True)
                    for wi, w in enumerate(points):
                        for c in points[w]:
                            n, warm = sweep.num_requests(c), min(c, 8)
                            seed = 700000 + rep * 100000 + wi * 1000 + c
                            tools = ["loadgen", "aiperf"] if k % 2 == 0 else ["aiperf", "loadgen"]
                            k += 1
                            for t in tools:
                                pd = d / f"{w}_c{c}"
                                pd.mkdir(exist_ok=True)
                                if t == "loadgen":
                                    if (pd / "loadgen.json").exists():
                                        continue
                                    msg = run_loadgen(w, c, n, warm, seed, pd / "loadgen", dump=True, model=MODEL)
                                else:
                                    if (pd / "aiperf" / "profile_export_aiperf.json").exists():
                                        continue
                                    (pd / "aiperf").mkdir(exist_ok=True)
                                    msg = run_aiperf(w, c, n, warm, seed, pd / "aiperf", model=MODEL)
                                print(f"[rep{rep}] {split} {engine} {w} c={c} {t} (first={tools[0]}): {msg[:150]}", flush=True)
                    if rep == a.first_rep and not a.no_quality and not (d / "quality.summary.txt").exists():
                        r = sweep.sh([sweep.PY, str(HERE / "quality.py"), "--model", MODEL, "--n", "100",
                                      "--concurrency", "32", "--out", str(d / "quality.jsonl")], env=ENV)
                        (d / "quality.summary.txt").write_text(r.stdout[-3000:] + r.stderr[-2000:])
                        print(f"[rep{rep}] {split} {engine} quality: {(r.stdout.strip().splitlines() or [''])[-1][:150]}", flush=True)
                    (d / "server.log").write_text(sweep.server_logs())
                    sweep.stop_all()
    finally:
        gpulog.terminate()
        gpulog.wait()
        gpufile.close()
        sweep.stop_all()
    print("STEP2_DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


if __name__ == "__main__":
    main()
