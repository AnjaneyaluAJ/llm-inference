"""Agent-style multi-turn load generator over /v1/completions (Step 4, docs/step4_agent_kv_plan.md).

S session slots run in a closed loop. A session has --turns turns:
  turn 1: shared prefix (identical for every session in the run) + task
  turn t: previous prompt + the model's previous output + a tool result
Each turn asks for exactly --out-len tokens (greedy, EOS ignored), then the session pauses for an
exponential "tool" time (mean --think s). A finished session is replaced by a new one in its slot.
Each slot's first session starts at a random turn with a synthetic history, so the mix of context
lengths is near steady state from the start.

Routing is done here, across --replicas:
  sticky : a slot (and every session in it) always uses replica slot % R
  ll     : each turn goes to the replica with the fewest requests in flight (ties broken at random)

Only turns that start in [--warmup, --duration) are measured. After --duration no new turns start;
in-flight turns get --grace seconds, then are cancelled and counted as censored.

Writes <out>.turns.jsonl (one line per turn), <out>.json (summary) and the replicas' /metrics text at
the start and end of the measured window (<out>.metrics_{start,end}.txt).
"""
import argparse, asyncio, json, random, time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class Text:
    """Random-vocabulary text of about n tokens, deterministic per (seed, key)."""
    def __init__(self, tok, seed):
        self.tok, self.seed = tok, seed
        special = set(tok.all_special_ids)
        self.vocab = [i for i in range(min(tok.vocab_size, 150000)) if i not in special]

    def make(self, key, n):
        rng = random.Random(f"{self.seed}:{key}")
        return f"[{key}] " + self.tok.decode([rng.choice(self.vocab) for _ in range(n)])


async def scrape(s, replicas, path):
    out = []
    for u in replicas:
        try:
            async with s.get(u + "/metrics") as r:
                out.append(f"===== {u} =====\n" + await r.text())
        except Exception as e:  # noqa: BLE001
            out.append(f"===== {u} =====\nERROR {e}")
    Path(path).write_text("\n".join(out))


async def one_turn(s, url, model, prompt, out_len):
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    rec = {"ttft": None, "output_tokens": None, "prompt_tokens": None, "cached_tokens": None, "error": None}
    text = []
    t0 = time.perf_counter()
    try:
        async with s.post(url + "/v1/completions", json=body) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:200]}"
                return rec, ""
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("choices") and msg["choices"][0].get("text"):
                    if rec["ttft"] is None:
                        rec["ttft"] = time.perf_counter() - t0
                    text.append(msg["choices"][0]["text"])
                u = msg.get("usage")
                if u:
                    rec["output_tokens"] = u.get("completion_tokens")
                    rec["prompt_tokens"] = u.get("prompt_tokens")
                    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["e2e"] = time.perf_counter() - t0
    if rec["error"] is None and rec["output_tokens"] != out_len:
        rec["error"] = f"output_tokens={rec['output_tokens']} != {out_len}"
    if rec["error"] is None and rec["ttft"] is None:
        rec["error"] = f"no text chunks received (usage said {rec['output_tokens']} tokens)"
    return rec, "".join(text)


async def run(a):
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    T = Text(tok, a.seed)
    replicas = [u.rstrip("/") for u in a.replicas.split(",")]
    R = len(replicas)
    prefix = T.make("prefix", a.prefix)
    inflight = [0] * R
    route_rng = random.Random(a.seed + 7)
    recs, censored = [], [0]
    t_start = time.perf_counter()
    now = lambda: time.perf_counter() - t_start  # noqa: E731

    def pick(slot):
        if a.routing == "sticky":
            return slot % R
        m = min(inflight)
        return route_rng.choice([i for i, v in enumerate(inflight) if v == m])

    async def slot_loop(s, slot):
        rng = random.Random(a.seed * 1000 + slot)
        n = 0
        start_turn = rng.randint(1, a.turns)  # first session of the slot starts mid-way, with a synthetic history
        while True:
            sid = f"s{slot}.{n}"
            prompt = prefix + T.make(f"{sid}:task", a.task)
            for t in range(2, start_turn + 1):
                prompt += T.make(f"{sid}:out{t - 1}", a.out_len) + T.make(f"{sid}:tool{t}", a.tool)
            for t in range(start_turn, a.turns + 1):
                if now() >= a.duration:
                    return
                i = pick(slot)
                inflight[i] += 1
                st = now()
                try:
                    rec, text = await one_turn(s, replicas[i], a.model, prompt, a.out_len)
                finally:
                    inflight[i] -= 1
                rec.update({"session": sid, "turn": t, "synthetic_history": n == 0 and t == start_turn and t > 1,
                            "replica": i, "start": st, "end": now()})
                recs.append(rec)
                if rec["error"]:
                    break  # a failed turn ends the session
                prompt += text + (T.make(f"{sid}:tool{t + 1}", a.tool) if t < a.turns else "")
                await asyncio.sleep(rng.expovariate(1 / a.think))
            n += 1
            start_turn = 1

    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        tasks = [asyncio.create_task(slot_loop(s, k)) for k in range(a.sessions)]
        await asyncio.sleep(a.warmup)
        await scrape(s, replicas, f"{a.out}.metrics_start.txt")
        await asyncio.sleep(max(0, a.duration - now()))
        await scrape(s, replicas, f"{a.out}.metrics_end.txt")
        done, pending = await asyncio.wait(tasks, timeout=a.grace)
        for p in pending:
            p.cancel()
        censored[0] = len(pending)
        await asyncio.gather(*pending, return_exceptions=True)

    W, D = a.warmup, a.duration
    meas = [r for r in recs if W <= r["start"] < D]
    ok = [r for r in meas if r["error"] is None]
    done_in = [r for r in recs if r["error"] is None and W <= r["end"] < D]
    mins = (D - W) / 60
    ms = lambda xs, p: None if not xs else round(1000 * pct(xs, p), 1)  # noqa: E731
    ttft = [r["ttft"] for r in ok]
    pt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    ct = [(r["cached_tokens"] or 0, r["prompt_tokens"]) for r in ok if r["prompt_tokens"] and r["cached_tokens"] is not None]
    summary = {
        "config": vars(a),
        "turns_started_in_window": len(meas), "turns_ok": len(ok), "turns_err": len(meas) - len(ok),
        "censored_at_end": censored[0],
        "errors_sample": [r["error"] for r in meas if r["error"]][:5],
        "turns_per_min": len(done_in) / mins,
        "output_tok_per_s": sum(r["output_tokens"] for r in done_in) / (D - W),
        "goodput_turns_per_min_ttft_2s": sum(1 for r in done_in if r["ttft"] <= 2.0) / mins,
        "ttft_ms": {p: ms(ttft, p) for p in (50, 90, 99)},
        "turn_ms": {p: ms([r["e2e"] for r in ok], p) for p in (50, 90, 99)},
        "mean_prompt_tokens": sum(pt) / len(pt) if pt else None,
        "client_cached_fraction": (sum(c for c, _ in ct) / sum(p for _, p in ct)) if ct else None,
        "turns_per_replica": [sum(1 for r in ok if r["replica"] == i) for i in range(R)],
    }
    with open(f"{a.out}.turns.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    Path(f"{a.out}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("turns_per_min", "goodput_turns_per_min_ttft_2s", "ttft_ms",
                                              "mean_prompt_tokens", "client_cached_fraction", "turns_err", "censored_at_end")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", required=True, help="comma list of base URLs")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--routing", choices=["sticky", "ll"], required=True)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--prefix", type=int, default=3000)
    ap.add_argument("--task", type=int, default=500)
    ap.add_argument("--tool", type=int, default=1500)
    ap.add_argument("--out-len", type=int, default=150)
    ap.add_argument("--think", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=180)
    ap.add_argument("--grace", type=float, default=60)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))
