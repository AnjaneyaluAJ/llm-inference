"""One load generator for every engine, over the OpenAI-compatible /v1/completions API.

Each request has a prompt of exactly --prompt-len tokens (random vocabulary tokens with a unique
prefix, so no engine gets prefix-cache hits from another request) and asks for exactly
--output-len tokens with greedy decoding and EOS ignored. Every response is checked: requests that
come back with the wrong number of output tokens are counted as errors, not silently averaged in.

Modes:
  --concurrency N : closed loop, N requests in flight at all times, --num-requests total
  --rate R        : open loop, Poisson arrivals at R requests/s for --num-requests requests

Writes one JSON line per request to <out>.requests.jsonl and a summary to <out>.json.
"""
import argparse, asyncio, json, random, statistics, time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def make_prompts(tok, n, prompt_len, seed):
    """n prompts that each re-tokenize to exactly prompt_len tokens."""
    rng = random.Random(seed)
    special = set(tok.all_special_ids)
    vocab = [i for i in range(min(tok.vocab_size, 150000)) if i not in special]
    prompts = []
    while len(prompts) < n:
        ids = [rng.choice(vocab) for _ in range(prompt_len + 8)]
        text = f"[{len(prompts)}:{seed}] " + tok.decode(ids)
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) < prompt_len:
            continue
        text = tok.decode(enc[:prompt_len])
        if len(tok.encode(text, add_special_tokens=False)) == prompt_len:
            prompts.append(text)
    return prompts


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def one_request(session, url, model, prompt, output_len, extra):
    body = {
        "model": model, "prompt": prompt, "max_tokens": output_len, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True, **extra,
    }
    rec = {"start": time.perf_counter(), "ttft": None, "chunk_times": [], "output_tokens": None,
           "prompt_tokens": None, "error": None}
    try:
        async with session.post(url, json=body) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:200]}"
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                now = time.perf_counter()
                if msg.get("choices") and msg["choices"][0].get("text"):
                    if rec["ttft"] is None:
                        rec["ttft"] = now - rec["start"]
                    rec["chunk_times"].append(now - rec["start"])
                if msg.get("usage"):
                    rec["output_tokens"] = msg["usage"].get("completion_tokens")
                    rec["prompt_tokens"] = msg["usage"].get("prompt_tokens")
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["e2e"] = time.perf_counter() - rec["start"]
    if rec["error"] is None and rec["output_tokens"] != output_len:
        rec["error"] = f"output_tokens={rec['output_tokens']} != {output_len}"
    return rec


async def run(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = make_prompts(tok, args.num_requests + args.warmup, args.prompt_len, args.seed)
    url = args.base_url.rstrip("/") + "/v1/completions"
    extra = json.loads(args.extra_body) if args.extra_body else {}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        # warm-up, not recorded
        await asyncio.gather(*[one_request(s, url, args.model, p, args.output_len, extra)
                               for p in prompts[:args.warmup]])
        todo = prompts[args.warmup:]
        recs = []
        t0 = time.perf_counter()
        if args.rate:
            rng = random.Random(args.seed + 1)
            tasks = []
            for p in todo:
                tasks.append(asyncio.create_task(one_request(s, url, args.model, p, args.output_len, extra)))
                await asyncio.sleep(rng.expovariate(args.rate))
            recs = await asyncio.gather(*tasks)
        else:
            queue = list(todo)

            async def worker():
                while queue:
                    p = queue.pop()
                    recs.append(await one_request(s, url, args.model, p, args.output_len, extra))
            await asyncio.gather(*[worker() for _ in range(args.concurrency)])
        wall = time.perf_counter() - t0

    ok = [r for r in recs if r["error"] is None]
    tpot = [(r["e2e"] - r["ttft"]) / (r["output_tokens"] - 1) for r in ok if r["output_tokens"] > 1]
    itl = [b - a for r in ok for a, b in zip(r["chunk_times"], r["chunk_times"][1:])]
    ms = lambda xs, p: None if not xs else round(1000 * pct(xs, p), 2)  # noqa: E731
    summary = {
        "config": vars(args), "wall_s": wall, "requests_ok": len(ok), "requests_err": len(recs) - len(ok),
        "errors_sample": [r["error"] for r in recs if r["error"]][:5],
        "req_per_s": len(ok) / wall, "output_tok_per_s": sum(r["output_tokens"] for r in ok) / wall,
        "ttft_ms": {p: ms([r["ttft"] for r in ok], p) for p in (50, 90, 99)},
        "tpot_ms": {p: ms(tpot, p) for p in (50, 90, 99)},
        "itl_ms": {p: ms(itl, p) for p in (50, 90, 99)},
        "e2e_ms": {p: ms([r["e2e"] for r in ok], p) for p in (50, 90, 99)},
        "mean_chunks_per_request": statistics.mean(len(r["chunk_times"]) for r in ok) if ok else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    with open(out.with_suffix(".requests.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "chunk_times"} | {"n_chunks": len(r["chunk_times"])}) + "\n")
    print(json.dumps({k: summary[k] for k in ("requests_ok", "requests_err", "req_per_s", "output_tok_per_s", "ttft_ms", "tpot_ms")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=4)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--concurrency", type=int)
    g.add_argument("--rate", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-body", default="", help="JSON merged into each request body")
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))
