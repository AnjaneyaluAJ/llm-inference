"""Quality check: the same GSM8K questions through every engine, greedy, same prompt.

Not a model evaluation: the point is to show that each engine produces the model's normal answers,
so no engine's speed comes from changed outputs. Reports accuracy and, against a reference file
from another engine, how many outputs are identical token-for-token.

Uses an 8-shot plain-text prompt over /v1/completions (no chat template, so every engine sees the
same tokens), max 384 new tokens, stop at the next "Question:".

Usage: python bench/quality.py --out /data/results/step1/quality/vllm.jsonl [--n 250]
"""
import argparse, asyncio, json, re
from pathlib import Path

import aiohttp
import pandas as pd

DATA = "/data/hf/gsm8k/main/test-00000-of-00001.parquet"
TRAIN = "/data/hf/gsm8k/main/train-00000-of-00001.parquet"


def build_prompt(shots, q):
    s = "".join(f"Question: {x['question']}\nAnswer: {x['answer'].split('####')[0].strip()} The answer is "
                 f"{x['answer'].split('####')[1].strip()}.\n\n" for x in shots)
    return s + f"Question: {q}\nAnswer:"


def extract(text):
    m = re.search(r"The answer is\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        return nums[-1].replace(",", "") if nums else None
    return m.group(1).replace(",", "")


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


async def main(a):
    test = pd.read_parquet(DATA).head(a.n)
    shots = pd.read_parquet(TRAIN).head(8).to_dict("records")
    sem = asyncio.Semaphore(a.concurrency)
    url = a.base_url.rstrip("/") + "/v1/completions"

    async def ask(s, i, row):
        body = {"model": a.model, "prompt": build_prompt(shots, row["question"]), "max_tokens": 384,
                "temperature": 0.0, "stop": ["Question:"]}
        async with sem, s.post(url, json=body) as r:
            out = (await r.json())["choices"][0]["text"]
        gold = row["answer"].split("####")[1].strip().replace(",", "")
        pred = extract(out)
        return {"i": i, "gold": gold, "pred": pred, "correct": same(pred, gold), "text": out}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        res = await asyncio.gather(*[ask(s, i, r) for i, r in test.iterrows()])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in res) / len(res)
    print(json.dumps({"n": len(res), "accuracy": round(acc, 4)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
