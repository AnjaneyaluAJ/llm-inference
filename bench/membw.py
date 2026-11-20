"""Measured GPU memory bandwidth, the ceiling for single-user decode speed.

Two estimates on a 4 GiB buffer: a device-to-device copy (counts read + write bytes) and a
read-only reduction (sum). Decode mostly reads weights, so the read-only number is the relevant one.
"""
import json, sys
import torch

n = 4 * 1024**3 // 2
x = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
y = torch.empty_like(x)
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "bytes": x.numel() * 2}


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000 / iters


t = timeit(lambda: y.copy_(x))
res["copy_GBps"] = 2 * res["bytes"] / t / 1e9
t = timeit(lambda: x.sum(dtype=torch.float32))
res["read_GBps"] = res["bytes"] / t / 1e9
print(json.dumps(res))
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(res, indent=1))
