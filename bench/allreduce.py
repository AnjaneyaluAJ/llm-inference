"""All-reduce cost between GPUs: what tensor parallelism pays on every layer.

Qwen3-32B with tensor parallelism does 2 all-reduces per layer x 64 layers = 128 per generated token.
At one user each is small (hidden size 5120 x 2 bytes = 10 KB), so latency matters, not bandwidth.
Measures NCCL all-reduce time for group sizes 2, 4, 8 (first N GPUs) at decode-sized and large messages.

Run inside an engine container (has torch + NCCL):
  docker run --rm --gpus all --ipc=host -v $PWD:/b --entrypoint python3 vllm/vllm-openai:v0.34.0 /b/allreduce.py > allreduce.json
"""
import json, os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [10_240, 163_840, 1_310_720, 16 * 1024**2, 256 * 1024**2]  # bytes: batch 1, 16, 128 decode; large


def worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    out = []
    for nbytes in SIZES:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        iters = 200 if nbytes < 2**22 else 20
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        t = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        busbw = 2 * (world - 1) / world * nbytes / dt / 1e9
        out.append({"bytes": nbytes, "us": dt * 1e6, "bus_GBps": busbw})
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


if __name__ == "__main__":
    res = {"gpus": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "nccl": ".".join(map(str, torch.cuda.nccl.version())), "groups": {}}
    ctx = mp.get_context("spawn")
    for world in (2, 4, 8):
        if world > res["gpus"]:
            continue
        q = ctx.Queue()
        mp.start_processes(worker, args=(world, 29500 + world, q), nprocs=world, start_method="spawn", join=True)
        res["groups"][world] = q.get()
        print(world, [(r["bytes"], round(r["us"], 1), round(r["bus_GBps"], 1)) for r in res["groups"][world]], file=sys.stderr)
    print(json.dumps(res, indent=1))
