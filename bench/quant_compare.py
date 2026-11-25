"""Step 3: BF16 vs FP8 vs INT4 Qwen3-32B on 8x L40S, across splits and engines, measured with loadgen.

Plan and predictions: docs/step3_quant_plan.md.

  --check   start every combination, answer 20 GSM8K questions, stop (which engine loads which format)
  default   the sweep: for each repeat, combinations and engines shuffled; every workload x users point;
            in repeat 1 also the full GSM8K test set (1,319 questions)

Prompt seeds are the same formula as Step 2 (bench/multi_compare.py), so BF16 tp2dp4 here sees the same
prompts as Step 2's tp2dp4.

Usage: python bench/quant_compare.py [--check] [--reps 2] [--out /data/results/step3]
       [--only fp8/tp1dp8:vllm,...] [--fallback fp8:trtllm,...]
"""
import argparse, json, random, time
from pathlib import Path

import sweep
from tool_compare import ENV, run_loadgen

HERE = Path(__file__).resolve().parent
MODELS = {"bf16": "Qwen/Qwen3-32B", "fp8": "Qwen/Qwen3-32B-FP8", "int4": "Qwen/Qwen3-32B-AWQ"}
FALLBACK = {"fp8": "RedHatAI/Qwen3-32B-FP8-dynamic", "int4": "RedHatAI/Qwen3-32B-quantized.w4a16"}
SPLITS = {"tp1dp8": (1, 8), "tp2dp4": (2, 4)}
COMBOS = [("bf16", "tp2dp4"), ("fp8", "tp1dp8"), ("fp8", "tp2dp4"), ("int4", "tp1dp8"), ("int4", "tp2dp4")]
ENGINES = ["vllm", "sglang", "trtllm"]
POINTS = {"chat": [1, 16, 64, 256], "rag": [1, 16, 64, 256], "generate": [1, 16, 64, 256]}


def quality(model, n, out):
    r = sweep.sh([sweep.PY, str(HERE / "quality.py"), "--model", model, "--n", str(n), "--concurrency", "64",
                  "--out", str(out)], env=ENV)
    out.with_suffix(".summary.txt").write_text(r.stdout[-3000:] + r.stderr[-2000:])
    return (r.stdout.strip().splitlines() or [""])[-1][:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--out", default="/data/results/step3")
    ap.add_argument("--only", default="", help="comma list of fmt/split:engine, e.g. fp8/tp1dp8:vllm")
    ap.add_argument("--fallback", default="", help="comma list of fmt:engine or fmt/split:engine that use the fallback checkpoint")
    ap.add_argument("--startup-timeout", type=int, default=2400)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    fallback = set(filter(None, a.fallback.split(",")))
    root = Path(a.out) / ("check" if a.check else "")
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.txt").write_text(sweep.sh(f"nvidia-smi; docker images --digests; uname -a; "
                                           f"ls /data/hf/hub").stdout)
    gpufile = open(root / "gpu.csv", "a")
    gpulog = sweep.subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "2"], stdout=gpufile, stderr=sweep.subprocess.DEVNULL)
    reps = [0] if a.check else range(a.first_rep, a.first_rep + a.reps)
    try:
        for rep in reps:
            rng = random.Random(5000 + rep)
            combos = COMBOS[:]; rng.shuffle(combos)
            for fmt, split in combos:
                engines = ENGINES[:]; rng.shuffle(engines)
                for engine in engines:
                    tag = f"{fmt}/{split}:{engine}"
                    if only and tag not in only:
                        continue
                    model = FALLBACK[fmt] if (f"{fmt}:{engine}" in fallback or tag in fallback) else MODELS[fmt]
                    tp, dp = SPLITS[split]
                    d = root / (f"{fmt}_{split}/{engine}" if a.check else f"rep{rep}/{fmt}_{split}/{engine}")
                    if a.check and (d / "check.json").exists():
                        continue
                    d.mkdir(parents=True, exist_ok=True)
                    try:
                        up = sweep.start(engine, "defaults", d, model, tp, dp, a.startup_timeout)
                    except RuntimeError as e:
                        print(f"[rep{rep}] {tag} START FAILED: {str(e)[:200]}", flush=True)
                        if a.check:
                            (d / "check.json").write_text(json.dumps({"model": model, "started": False}))
                        continue
                    meta = {"format": fmt, "engine": engine, "split": split, "tp": tp, "dp": dp, "model": model,
                            "fallback": model != MODELS[fmt], "image": sweep.IMAGES[engine],
                            "cmd": sweep.engine_cmd(engine, "defaults", model, tp, dp), "startup_s": up,
                            "dp_method": "nginx least_conn over separate servers" if engine == "trtllm" and dp > 1 else "engine built-in",
                            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    (d / "meta.json").write_text(json.dumps(meta, indent=1))
                    print(f"[rep{rep}] {tag} ({model}) up in {up:.0f}s", flush=True)
                    if a.check:
                        q = quality(model, 20, d / "quality.jsonl")
                        (d / "check.json").write_text(json.dumps({"model": model, "started": True, "startup_s": up, "quality": q}))
                        print(f"[check] {tag} quality: {q}", flush=True)
                    else:
                        for wi, w in enumerate(POINTS):
                            for c in POINTS[w]:
                                pd = d / f"{w}_c{c}"
                                pd.mkdir(exist_ok=True)
                                if (pd / "loadgen.json").exists():
                                    continue
                                seed = 700000 + rep * 100000 + wi * 1000 + c
                                msg = run_loadgen(w, c, sweep.num_requests(c), min(c, 8), seed, pd / "loadgen", dump=True, model=model)
                                print(f"[rep{rep}] {tag} {w} c={c}: {msg[:150]}", flush=True)
                        if rep == a.first_rep and not (d / "quality.jsonl").exists():
                            print(f"[rep{rep}] {tag} quality: {quality(model, 1319, d / 'quality.jsonl')}", flush=True)
                    (d / "server.log").write_text(sweep.server_logs())
                    sweep.stop_all()
    finally:
        gpulog.terminate()
        gpulog.wait()
        gpufile.close()
        sweep.stop_all()
    print("STEP3_DONE" if not a.check else "CHECK_DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


if __name__ == "__main__":
    main()
