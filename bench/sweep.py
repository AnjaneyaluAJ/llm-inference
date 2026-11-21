"""Concurrency sweep across engines, workloads and repeats.

For each repeat, engines run in a shuffled order (so drift over time does not favour one engine).
Each engine is started fresh in its container, health-checked, warmed up, then every workload x
concurrency point is measured with bench/loadgen.py. GPU clocks, power and temperature are logged
every second for the whole run. The same seeds are used for every engine within a repeat, so all
engines see identical prompts; seeds differ between repeats.

Usage: python bench/sweep.py --config defaults --reps 3 --out /data/results/step1
"""
import argparse, json, random, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/data/venv/bin/python"
HF = "/data/hf"
KV_TOKENS = 147456  # matched KV-cache capacity for the "matched" config: 2304 x 64 (vLLM) = 4608 x 32 (TRT-LLM) tokens

IMAGES = {
    "vllm": "vllm/vllm-openai:v0.34.0",
    "sglang": "lmsysorg/sglang:v0.5.23",
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0",
}


def engine_cmd(engine, config):
    """Server command line. 'defaults' = as shipped, only the max context length set."""
    if engine == "vllm":
        cmd = ["--model", "Qwen/Qwen3-8B", "--max-model-len", "8192"]
        if config == "matched":
            # vLLM v0.34 picks the block size itself (64 tokens for this model); set it explicitly so
            # the block count maps to exactly KV_TOKENS tokens. 9216 blocks at the auto size asked for 81 GiB.
            cmd += ["--block-size", "64", "--num-gpu-blocks-override", str(KV_TOKENS // 64)]
        return cmd
    if engine == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-8B", "--host", "0.0.0.0",
               "--port", "8000", "--context-length", "8192"]
        if config == "matched":
            cmd += ["--max-total-tokens", str(KV_TOKENS)]
        return cmd
    if engine == "trtllm":
        cmd = ["trtllm-serve", "Qwen/Qwen3-8B", "--host", "0.0.0.0", "--port", "8000", "--max_seq_len", "8192"]
        if config == "matched":
            cmd += ["--extra_llm_api_options", "/cfg/trtllm_matched.yaml"]
        return cmd
    raise ValueError(engine)


WORKLOADS = {  # name: (prompt tokens, output tokens)
    "chat": (1024, 256),
    "rag": (4096, 128),
    "generate": (128, 1024),
}
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def num_requests(c):
    return max(4 * c, 10)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def start(engine, config, logdir):
    sh("docker rm -f eng")
    cfgdir = HERE.parent / "configs"
    args = ["docker", "run", "-d", "--name", "eng", "--gpus", "all", "--ipc=host", "--shm-size", "16g",
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "-p", "8000:8000",
            "-v", f"{HF}:/root/.cache/huggingface", "-v", f"{cfgdir}:/cfg:ro", "-e", "HF_HUB_OFFLINE=1",
            IMAGES[engine], *engine_cmd(engine, config)]
    r = sh(args)
    if r.returncode:
        raise RuntimeError(r.stderr)
    t0 = time.time()
    while time.time() - t0 < 900:
        if sh("curl -sf localhost:8000/health").returncode == 0:
            break
        if "running" not in sh("docker inspect -f '{{.State.Status}}' eng").stdout:
            break
        time.sleep(5)
    else:
        raise RuntimeError("server did not become healthy")
    if sh("curl -sf localhost:8000/health").returncode:
        (logdir / "server.log").write_text(sh("docker logs eng 2>&1").stdout)
        raise RuntimeError(f"{engine} failed to start, see {logdir}/server.log")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["defaults", "matched"], required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--engines", default="vllm,sglang,trtllm")
    ap.add_argument("--workloads", default="chat,rag,generate")
    ap.add_argument("--concurrency", default=",".join(map(str, CONCURRENCY)))
    ap.add_argument("--out", default="/data/results/step1")
    a = ap.parse_args()
    engines = a.engines.split(",")
    conc = [int(c) for c in a.concurrency.split(",")]
    root = Path(a.out) / a.config
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.txt").write_text(sh("nvidia-smi; docker images --digests; uname -a").stdout)
    gpufile = open(root / "gpu.csv", "a")
    gpulog = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used",
         "--format=csv", "-l", "1"], stdout=gpufile, stderr=subprocess.DEVNULL)
    try:
        for rep in range(a.first_rep, a.first_rep + a.reps):
            order = engines[:]
            random.Random(1000 + rep).shuffle(order)
            for engine in order:
                d = root / f"rep{rep}" / engine
                d.mkdir(parents=True, exist_ok=True)
                startup = start(engine, a.config, d)
                meta = {"engine": engine, "image": IMAGES[engine], "cmd": engine_cmd(engine, a.config),
                        "startup_s": startup, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "image_digest": sh(f"docker inspect -f '{{{{.Image}}}}' eng").stdout.strip()}
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"[rep{rep}] {engine} up in {startup:.0f}s", flush=True)
                for wi, w in enumerate(a.workloads.split(",")):
                    pl, ol = WORKLOADS[w]
                    for c in conc:
                        out = d / f"{w}_c{c}"
                        if out.with_suffix(".json").exists():
                            continue
                        seed = rep * 100000 + wi * 1000 + c
                        r = sh([PY, str(HERE / "loadgen.py"), "--prompt-len", str(pl), "--output-len", str(ol),
                                "--concurrency", str(c), "--num-requests", str(num_requests(c)),
                                "--warmup", str(min(c, 8)), "--seed", str(seed), "--out", str(out)],
                               env={"HF_HOME": HF, "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
                        last = (r.stdout.strip().splitlines() or [""])[-1]
                        print(f"[rep{rep}] {engine} {w} c={c}: {last[:200]}", flush=True)
                        if r.returncode:
                            (d / f"{w}_c{c}.err").write_text(r.stderr[-5000:])
                (d / "server.log").write_text(sh("docker logs eng 2>&1").stdout[-2_000_000:])
                sh("docker rm -f eng")
    finally:
        gpulog.terminate()
        gpulog.wait()
        gpufile.close()
        sh("docker rm -f eng")


if __name__ == "__main__":
    main()
