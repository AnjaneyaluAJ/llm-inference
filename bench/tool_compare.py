"""Cross-check: our load generator (bench/loadgen.py) vs NVIDIA AIPerf, on the same servers.

For every repeat, engine (shuffled order) and point, both tools run back to back against the same
running server, with the same request count, warm-up count, prompt length and output length.
The tool that goes first alternates from point to point so slow drift does not favour either.

Each tool is used the way it ships, apart from matching the workload: AIPerf uses its default
synthetic prompts (Shakespeare text) and counts output tokens by re-tokenizing the returned text;
loadgen uses random-token prompts with unique prefixes and the server's usage count. Those
differences are part of what is being compared.

Then a stress point per engine (generate, 256 users, 1024 requests) with raw-stream capture on both
tools, to see whether text-less streams show up in AIPerf too.

Usage: python bench/tool_compare.py [--reps 3] [--out /data/results/toolcmp]
"""
import argparse, json, os, random, time
from pathlib import Path

import sweep

HERE = Path(__file__).resolve().parent
AIPERF = "/data/aiperf-venv/bin/aiperf"
POINTS = {"chat": [1, 16, 64], "rag": [1, 16, 64], "generate": [1, 16, 64]}
ENV = {"HF_HOME": sweep.HF, "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"}


def run_loadgen(w, c, n, warm, seed, out, dump=False, model=sweep.MODEL):
    pl, ol = sweep.WORKLOADS[w]
    cmd = [sweep.PY, str(HERE / "loadgen.py"), "--model", model, "--tokenizer", model, "--prompt-len", str(pl), "--output-len", str(ol), "--concurrency", str(c),
           "--num-requests", str(n), "--warmup", str(warm), "--seed", str(seed), "--out", str(out)]
    r = sweep.sh(cmd + (["--dump-bad"] if dump else []), env=ENV)
    if r.returncode:
        out.with_suffix(".err").write_text(r.stderr[-5000:])
    return (r.stdout.strip().splitlines() or [""])[-1][:160]


def run_aiperf(w, c, n, warm, seed, outdir, raw=False, model=sweep.MODEL):
    pl, ol = sweep.WORKLOADS[w]
    cmd = [AIPERF, "profile", "--model", model, "--url", "http://localhost:8000",
           "--endpoint-type", "completions", "--streaming", "--use-legacy-max-tokens",
           "--isl", str(pl), "--isl-stddev", "0", "--osl", str(ol), "--osl-stddev", "0",
           "--extra-inputs", "ignore_eos:true", "--concurrency", str(c), "--request-count", str(n),
           "--warmup-request-count", str(warm), "--random-seed", str(seed), "--ui", "none",
           "--artifact-dir", str(outdir), "--export-level", "raw" if raw else "records"]
    r = sweep.sh(cmd, env=ENV | {"HOME": os.environ.get("HOME", "/home/ubuntu")})
    (outdir / "aiperf_stdout.txt").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-5000:])
    try:
        d = json.loads((outdir / "profile_export_aiperf.json").read_text())
        return f"out tok/s {d['output_token_throughput']['avg']:.1f}, ttft p50 {d['time_to_first_token']['p50']:.0f} ms"
    except Exception as e:  # noqa: BLE001
        return f"no summary ({e}); rc={r.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="/data/results/toolcmp")
    ap.add_argument("--skip-stress", action="store_true")
    a = ap.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.txt").write_text(sweep.sh(f"nvidia-smi; docker images --digests; uname -a; {AIPERF} --version").stdout)
    gpufile = open(root / "gpu.csv", "a")
    gpulog = sweep.subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "1"], stdout=gpufile, stderr=sweep.subprocess.DEVNULL)
    try:
        k = 0
        for rep in range(1, a.reps + 1):
            order = ["vllm", "sglang", "trtllm"]
            random.Random(2000 + rep).shuffle(order)
            for engine in order:
                d = root / f"rep{rep}" / engine
                d.mkdir(parents=True, exist_ok=True)
                up = sweep.start(engine, "defaults", d)
                print(f"[rep{rep}] {engine} up in {up:.0f}s", flush=True)
                for wi, w in enumerate(POINTS):
                    for c in POINTS[w]:
                        n, warm, seed = sweep.num_requests(c), min(c, 8), 500000 + rep * 100000 + wi * 1000 + c
                        tools = ["loadgen", "aiperf"] if k % 2 == 0 else ["aiperf", "loadgen"]
                        k += 1
                        for t in tools:
                            if t == "loadgen":
                                out = d / f"{w}_c{c}" / "loadgen"
                                if out.with_suffix(".json").exists():
                                    continue
                                msg = run_loadgen(w, c, n, warm, seed, out)
                            else:
                                outdir = d / f"{w}_c{c}" / "aiperf"
                                if (outdir / "profile_export_aiperf.json").exists():
                                    continue
                                outdir.mkdir(parents=True, exist_ok=True)
                                msg = run_aiperf(w, c, n, warm, seed, outdir)
                            print(f"[rep{rep}] {engine} {w} c={c} {t} (first={tools[0]}): {msg}", flush=True)
                (d / "server.log").write_text(sweep.sh("docker logs eng 2>&1").stdout[-2_000_000:])
                sweep.sh("docker rm -f eng")
        if not a.skip_stress:
            for engine in ("vllm", "sglang", "trtllm"):
                d = root / "stress" / engine
                d.mkdir(parents=True, exist_ok=True)
                sweep.start(engine, "defaults", d)
                for t in ("loadgen", "aiperf"):
                    if t == "loadgen":
                        msg = run_loadgen("generate", 256, 1024, 8, 990256, d / "loadgen", dump=True)
                    else:
                        (d / "aiperf").mkdir(exist_ok=True)
                        msg = run_aiperf("generate", 256, 1024, 8, 990256, d / "aiperf", raw=True)
                    print(f"[stress] {engine} generate c=256 {t}: {msg}", flush=True)
                (d / "server.log").write_text(sweep.sh("docker logs eng 2>&1").stdout[-2_000_000:])
                sweep.sh("docker rm -f eng")
    finally:
        gpulog.terminate()
        gpulog.wait()
        gpufile.close()
        sweep.sh("docker rm -f eng")
    print("TOOLCMP_DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


if __name__ == "__main__":
    main()
