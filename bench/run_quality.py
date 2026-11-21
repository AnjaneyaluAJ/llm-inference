"""Run the GSM8K quality check on every engine and config, using the same server commands as the sweep.

Also re-measures any sweep point that left a .err file (with the server for that engine/config up),
writing it as <point>_rerun.json so the original failure stays on record.

Usage: python bench/run_quality.py [--n 250] [--out /data/results/step1]
"""
import argparse
from pathlib import Path

import sweep

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default="/data/results/step1")
    a = ap.parse_args()
    root = Path(a.out)
    env = {"HF_HOME": sweep.HF, "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"}
    try:
        for config in ("defaults", "matched"):
            for engine in ("vllm", "sglang", "trtllm"):
                d = root / "quality" / config
                d.mkdir(parents=True, exist_ok=True)
                sweep.start(engine, config, d)
                r = sweep.sh([sweep.PY, str(HERE / "quality.py"), "--n", str(a.n), "--out", str(d / f"{engine}.jsonl")], env=env)
                print(f"{config} {engine} quality: {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:] if r.returncode else ''}", flush=True)
                (d / f"{engine}.summary.txt").write_text(r.stdout + r.stderr[-3000:])
                for err in sorted(root.glob(f"{config}/rep*/{engine}/*.err")):
                    rerun = err.with_name(err.stem + "_rerun")
                    if rerun.with_suffix(".json").exists():
                        continue
                    w, c = err.stem.rsplit("_c", 1)
                    pl, ol = sweep.WORKLOADS[w]
                    c = int(c)
                    r = sweep.sh([sweep.PY, str(HERE / "loadgen.py"), "--prompt-len", str(pl), "--output-len", str(ol),
                                  "--concurrency", str(c), "--num-requests", str(sweep.num_requests(c)),
                                  "--warmup", str(min(c, 8)), "--seed", str(int(err.parts[-3][3:]) * 100000 +
                                  list(sweep.WORKLOADS).index(w) * 1000 + c), "--out", str(rerun)], env=env)
                    print(f"rerun {err}: {(r.stdout.strip().splitlines() or [''])[-1][:200]}", flush=True)
                (d / f"{engine}.server.log").write_text(sweep.sh("docker logs eng 2>&1").stdout[-1_000_000:])
                sweep.sh("docker rm -f eng")
    finally:
        sweep.sh("docker rm -f eng")


if __name__ == "__main__":
    main()
