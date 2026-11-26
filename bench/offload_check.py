"""Step 4 smoke check: does a prompt that comes back from the CPU KV tier give the same output?

For each of --n prompts of --len tokens: send it cold; push it out of GPU memory with --fill other prompts
of the same length; send it again (served from the CPU tier if offload works); send it once more (GPU hit).
Records time to first token and the first --out greedy tokens of each, and compares them with the cold run.
Exact equality is not guaranteed (batch shapes change the arithmetic slightly), so this is an observation.
"""
import argparse, asyncio, json, random, time

import aiohttp
from transformers import AutoTokenizer


async def ask(s, url, model, prompt, n):
    body = {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0.0, "ignore_eos": True, "stream": True}
    t0 = time.perf_counter(); ttft = None; text = []
    async with s.post(url + "/v1/completions", json=body) as r:
        async for raw in r.content:
            line = raw.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                c = json.loads(line[5:])["choices"]
                if c and c[0].get("text"):
                    ttft = ttft or time.perf_counter() - t0
                    text.append(c[0]["text"])
    return ttft, "".join(text)


async def main(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    special = set(tok.all_special_ids)
    vocab = [i for i in range(min(tok.vocab_size, 150000)) if i not in special]
    rng = random.Random(77)
    mk = lambda k: f"[{k}] " + tok.decode([rng.choice(vocab) for _ in range(a.len)])  # noqa: E731
    res = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        for i in range(a.n):
            p = mk(f"p{i}")
            t_cold, cold = await ask(s, a.base_url, a.model, p, a.out)
            for j in range(a.fill):
                await ask(s, a.base_url, a.model, mk(f"f{i}.{j}"), 1)
            t_cpu, cpu = await ask(s, a.base_url, a.model, p, a.out)
            t_gpu, gpu = await ask(s, a.base_url, a.model, p, a.out)
            res.append({"i": i, "ttft_cold": t_cold, "ttft_after_evict": t_cpu, "ttft_again": t_gpu,
                        "same_after_evict": cpu == cold, "same_again": gpu == cold,
                        "cold": cold, "after_evict": cpu, "again": gpu})
    out = {"n": a.n, "len": a.len, "fill": a.fill,
           "same_after_evict": sum(r["same_after_evict"] for r in res), "same_again": sum(r["same_again"] for r in res),
           "median_ttft": {k: sorted(r[k] for r in res)[len(res) // 2] for k in ("ttft_cold", "ttft_after_evict", "ttft_again")},
           "rows": res}
    with open(a.out_file, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("n", "same_after_evict", "same_again", "median_ttft")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--len", type=int, default=16000)
    ap.add_argument("--fill", type=int, default=6)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--out-file", "--out_file", dest="out_file", required=True)
    asyncio.run(main(ap.parse_args()))
