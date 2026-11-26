"""Analyse Step 3 (bench/quant_compare.py): formats x splits x engines x workloads x users, loadgen, two repeats.

For every point: median over repeats of output tokens/s, p50/p99 TTFT and TPOT. Quality on the full GSM8K
(repeat 1): accuracy with a 95% interval, and paired agreement of final answers with the same engine's BF16
run. Then the checks against the predictions in docs/step3_quant_plan.md.

Usage: python bench/step3_analyze.py results/step3   -> writes summary.json and summary.md there
"""
import json, math, re, statistics, sys
from collections import defaultdict
from pathlib import Path

from tool_analyze import loadgen_point

COMBOS = ["bf16_tp2dp4", "fp8_tp1dp8", "fp8_tp2dp4", "int4_tp1dp8", "int4_tp2dp4"]
ENGINES = ["vllm", "sglang", "trtllm"]
WORKLOADS = ["chat", "rag", "generate"]
USERS = [1, 16, 64, 256]
# One-user ceilings and predicted ranges from the plan (746.7 GB/s per GPU / weights on disk)
CEIL = {"bf16_tp2dp4": 22.8, "fp8_tp1dp8": 21.8, "fp8_tp2dp4": 43.5, "int4_tp1dp8": 38.7, "int4_tp2dp4": 77.4}
PRED = {"bf16_tp2dp4": (20, 22), "fp8_tp1dp8": (15, 20), "fp8_tp2dp4": (30, 38), "int4_tp1dp8": (28, 36),
        "int4_tp2dp4": (45, 60)}
PRICE_H = 30.13118  # g6e.48xlarge on-demand, us-west-2
METRICS = ("output_tok_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms")


def med(runs, k):
    xs = [r[k] for r in runs if r.get(k) is not None]
    return statistics.median(xs) if xs else None


def spread(runs):
    tp = [r["output_tok_per_s"] for r in runs]
    return 100 * (max(tp) - min(tp)) / statistics.median(tp) if len(tp) > 1 else None


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs (fixed b, broke c)."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def kv_tokens(log):
    t = log.read_text(errors="replace")
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
    if m:
        return int(m[0].replace(",", "")), "vLLM log"
    m = re.findall(r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+)", t)
    if m:
        return int(m[0]), "SGLang log"
    m = re.findall(r"Allocated [\d.]+ GiB for max tokens in paged KV cache \((\d+)\)", t)
    if m:
        return max(int(x) for x in m), "TRT-LLM log"
    # SGLang with --dp-size: the start-up line can be missing from the captured log; estimate from #token / usage
    est = [int(a) / float(b) for a, b in re.findall(r"#token: (\d+), token usage: ([\d.]+)", t) if float(b) >= 0.5]
    if est:
        return round(statistics.median(est), -2), "estimated from SGLang #token / token usage"
    return None, None


def main(root):
    root = Path(root)
    pts = defaultdict(list); errs = defaultdict(lambda: defaultdict(int)); nreq = defaultdict(int)
    for p in sorted(root.glob("rep*/*/*/*_c*")):
        combo, engine = p.parent.parent.name, p.parent.name
        w, c = p.name.rsplit("_c", 1)
        key = (combo, engine, w, int(c))
        if not (p / "loadgen.json").exists():
            continue
        lp, raw = loadgen_point(p / "loadgen.json")
        pts[key].append(lp)
        nreq[key] += raw["requests_ok"] + raw["requests_err"]
        for x in open(p / "loadgen.requests.jsonl"):
            err = json.loads(x).get("error")
            if err:
                errs[key]["timeout" if "Timeout" in err else "no_text" if "no text" in err else "other"] += 1
        bad = p / "loadgen.bad_streams.jsonl"
        if bad.exists():
            errs[key]["bad_streams_dumped"] += sum(1 for _ in open(bad))

    rows = {k: {**{m: med(r, m) for m in METRICS}, "reps": len(r), "rep_spread_pct": spread(r),
                "runs_tok_per_s": [x["output_tok_per_s"] for x in r], "requests": nreq[k],
                "errors": dict(errs[k])} for k, r in pts.items()}
    g = lambda cb, e, w, c, k="output_tok_per_s": (rows.get((cb, e, w, c)) or {}).get(k)  # noqa: E731
    have = lambda cb, e: g(cb, e, "chat", 1) is not None  # noqa: E731
    peak = lambda cb, e, w: max(g(cb, e, w, c) or 0 for c in USERS)  # noqa: E731
    peak_at = lambda cb, e, w: max(USERS, key=lambda c: g(cb, e, w, c) or 0)  # noqa: E731

    # Quality: accuracy, 95% interval, paired agreement with the same engine's BF16 answers
    qa = {}
    for q in sorted(root.glob("rep1/*/*/quality.jsonl")):
        qa[(q.parent.parent.name, q.parent.name)] = {json.loads(x)["i"]: json.loads(x) for x in open(q)}
    quality = {}
    for (cb, e), ans in qa.items():
        n = len(ans); k = sum(a["correct"] for a in ans.values())
        lo, hi = wilson(k, n)
        d = {"n": n, "accuracy": 100 * k / n, "ci95": [lo, hi]}
        base = qa.get(("bf16_tp2dp4", e))
        if base and cb != "bf16_tp2dp4":
            same = sum(ans[i]["pred"] == base[i]["pred"] for i in ans)
            d["agree_with_bf16_pct"] = 100 * same / n
            d["fixed_vs_bf16"] = sum(ans[i]["correct"] and not base[i]["correct"] for i in ans)
            d["broke_vs_bf16"] = sum(base[i]["correct"] and not ans[i]["correct"] for i in ans)
            d["mcnemar_p"] = mcnemar_p(d["fixed_vs_bf16"], d["broke_vs_bf16"])
            d["accuracy_diff_points"] = d["accuracy"] - 100 * sum(a["correct"] for a in base.values()) / len(base)
        quality[f"{cb}/{e}"] = d
    # Noise floor: BF16 engines against each other
    bf = {e: qa[("bf16_tp2dp4", e)] for e in ENGINES if ("bf16_tp2dp4", e) in qa}
    bf_pairs = {f"{a}-{b}": 100 * sum(bf[a][i]["pred"] == bf[b][i]["pred"] for i in bf[a]) / len(bf[a])
                for x, a in enumerate(bf) for b in list(bf)[x + 1:]}

    kv, meta = {}, {}
    for m in sorted(root.glob("rep1/*/*/meta.json")):
        d = json.loads(m.read_text()); key = f"{m.parent.parent.name}/{m.parent.name}"
        meta[key] = {"model": d["model"], "fallback": d.get("fallback"), "image": d.get("image"),
                     "startup_s": d.get("startup_s"), "dp_method": d.get("dp_method")}
        kv[key] = dict(zip(("tokens_per_replica", "source"), kv_tokens(m.parent / "server.log")))

    # Step 2 control: same split, same engines, earlier session
    s2 = {}
    s2f = root.parent / "step2" / "summary.json"
    if s2f.exists():
        for r in json.loads(s2f.read_text())["rows"]:
            if r["split"] == "tp2dp4":
                s2[(r["engine"], r["workload"], r["users"])] = r["loadgen"]["output_tok_per_s"]

    md = ["# Step 3 results: Qwen3-32B on 8x L40S, BF16 vs FP8 vs INT4", "",
          "Median of 2 repeats, loadgen. Output tokens/s unless stated. Default engine settings. "
          "INT4 is the Red Hat GPTQ checkpoint (`RedHatAI/Qwen3-32B-quantized.w4a16`); see the README for why.", ""]
    for w in WORKLOADS:
        md += [f"## {w}: output tokens/s", "",
               "| format/split | engine | " + " | ".join(f"{c} users" for c in USERS) + " |", "|---|---|" + "---|" * len(USERS)]
        for cb in COMBOS:
            for e in ENGINES:
                if have(cb, e):
                    md.append(f"| {cb} | {e} | " + " | ".join(f"{g(cb, e, w, c):.0f}" for c in USERS) + " |")
        md.append("")

    md += ["## Predictions", "", "### 1. One user (generate)", "",
           "| format/split | engine | tok/s | ceiling | % of ceiling | predicted | inside? |", "|---|---|---|---|---|---|---|"]
    p1 = []
    for cb in COMBOS:
        for e in ENGINES:
            if not have(cb, e):
                continue
            v = g(cb, e, "generate", 1); lo, hi = PRED[cb]; ok = lo <= v <= hi; p1.append(ok)
            md.append(f"| {cb} | {e} | {v:.1f} | {CEIL[cb]} | {100 * v / CEIL[cb]:.0f}% | {lo}-{hi} | {'yes' if ok else 'no'} |")
    fastest = max(((cb, e) for cb in COMBOS for e in ENGINES if have(cb, e)), key=lambda x: g(*x, "generate", 1))
    md.append(f"\nInside the predicted range: {sum(p1)} of {len(p1)}. Fastest: {'/'.join(fastest)}.\n")

    md += ["### 2-4. Peak throughput (best of 1/16/64/256 users) and users at the peak", "",
           "| format/split | engine | chat | rag | generate |", "|---|---|---|---|---|"]
    for cb in COMBOS:
        for e in ENGINES:
            if have(cb, e):
                md.append(f"| {cb} | {e} | " + " | ".join(f"{peak(cb, e, w):.0f} ({peak_at(cb, e, w)})" for w in WORKLOADS) + " |")
    md += ["", "Best quantized combination per engine against BF16 tp2dp4, same engine:", "",
           "| engine | workload | BF16 tp2dp4 | best FP8 | x | best INT4 | x |", "|---|---|---|---|---|---|---|"]
    for e in ENGINES:
        for w in WORKLOADS:
            b = peak("bf16_tp2dp4", e, w)
            f = max((peak(cb, e, w), cb) for cb in ("fp8_tp1dp8", "fp8_tp2dp4") if have(cb, e))
            i = max(((peak(cb, e, w), cb) for cb in ("int4_tp1dp8", "int4_tp2dp4") if have(cb, e)), default=None)
            md.append(f"| {e} | {w} | {b:.0f} | {f[0]:.0f} {f[1]} | {f[0] / b:.2f} | "
                      + (f"{i[0]:.0f} {i[1]} | {i[0] / b:.2f} |" if i else "- | - |"))
    md += ["", "At 256 users, best FP8 vs best INT4 (same engine):", "",
           "| engine | workload | FP8 | INT4 | FP8 / INT4 |", "|---|---|---|---|---|"]
    for e in ("vllm", "sglang"):
        for w in WORKLOADS:
            f = max(g(cb, e, w, 256) for cb in ("fp8_tp1dp8", "fp8_tp2dp4"))
            i = max(g(cb, e, w, 256) for cb in ("int4_tp1dp8", "int4_tp2dp4"))
            md.append(f"| {e} | {w} | {f:.0f} | {i:.0f} | {f / i:.2f} |")
    md += ["", "tp1dp8 / tp2dp4 at 256 users:", "", "| format | engine | chat | rag | generate |", "|---|---|---|---|---|"]
    for fm in ("fp8", "int4"):
        for e in ("vllm", "sglang"):
            md.append(f"| {fm} | {e} | " + " | ".join(
                f"{g(f'{fm}_tp1dp8', e, w, 256) / g(f'{fm}_tp2dp4', e, w, 256):.2f}" for w in WORKLOADS) + " |")

    md += ["", "### 5. Quality: full GSM8K (1,319 questions), repeat 1", "",
           "| format/split | engine | accuracy | 95% interval | vs BF16 (points) | same answer as BF16 | fixed | broke | McNemar p |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cb in COMBOS:
        for e in ENGINES:
            q = quality.get(f"{cb}/{e}")
            if q:
                ag = f"{q['agree_with_bf16_pct']:.1f}%" if "agree_with_bf16_pct" in q else "-"
                dd = f"{q['accuracy_diff_points']:+.1f}" if "accuracy_diff_points" in q else "-"
                pv = f"{q['mcnemar_p']:.3f}" if "mcnemar_p" in q else "-"
                md.append(f"| {cb} | {e} | {q['accuracy']:.1f}% | {q['ci95'][0]:.1f}-{q['ci95'][1]:.1f} | {dd} | {ag} | "
                          f"{q.get('fixed_vs_bf16', '-')} | {q.get('broke_vs_bf16', '-')} | {pv} |")
    md += ["", "BF16 engines against each other (the noise floor for agreement): "
           + ", ".join(f"{k} {v:.1f}%" for k, v in bf_pairs.items()) + "."]

    md += ["", "## KV-cache room per replica (tokens, repeat 1 start-up)", "", "| format/split | engine | tokens | source |", "|---|---|---|---|"]
    for k, v in kv.items():
        if v["tokens_per_replica"]:
            md.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | {v['tokens_per_replica']:,.0f} | {v['source']} |")

    md += ["", "## Cost per million output tokens at peak ($30.13/h for the instance)", "",
           "| format/split | engine | chat | rag | generate |", "|---|---|---|---|---|"]
    for cb in COMBOS:
        for e in ENGINES:
            if have(cb, e):
                md.append(f"| {cb} | {e} | " + " | ".join(f"${PRICE_H / (peak(cb, e, w) * 3600) * 1e6:.2f}" for w in WORKLOADS) + " |")

    if s2:
        md += ["", "## BF16 control: this run vs Step 2 (tp2dp4, loadgen)", ""]
        diffs = [(100 * (g("bf16_tp2dp4", e, w, c) / s2[(e, w, c)] - 1), (e, w, c))
                 for e in ENGINES for w in WORKLOADS for c in USERS if (e, w, c) in s2 and g("bf16_tp2dp4", e, w, c)]
        md.append(f"Median difference {statistics.median(abs(x) for x, _ in diffs):.1f}%; {sum(abs(x) <= 5 for x, _ in diffs)} "
                  f"of {len(diffs)} points within 5%. Largest: "
                  + ", ".join(f"{'/'.join(map(str, k))} {x:+.0f}%" for x, k in sorted(diffs, key=lambda d: -abs(d[0]))[:4]) + ".")

    md += ["", "## Time to first token, p50 (s) at 256 users", "", "| format/split | engine | chat | rag | generate |", "|---|---|---|---|---|"]
    for cb in COMBOS:
        for e in ENGINES:
            if have(cb, e):
                md.append(f"| {cb} | {e} | " + " | ".join(f"{g(cb, e, w, 256, 'ttft_p50_ms') / 1000:.1f}" for w in WORKLOADS) + " |")
    sp = [(r["rep_spread_pct"], k) for k, r in rows.items() if r["rep_spread_pct"] is not None]
    md += ["", "## Repeatability (rep1 vs rep2 throughput)", "",
           f"Median spread {statistics.median(x for x, _ in sp):.1f}%; {sum(x <= 5 for x, _ in sp)} of {len(sp)} points within 5%. "
           "Largest: " + ", ".join(f"{'/'.join(map(str, k))} {x:.0f}%" for x, k in sorted(sp, reverse=True)[:5]) + "."]
    kinds = sorted({k for v in errs.values() for k in v})
    md += ["", "## Requests and errors (both repeats)", "", f"Requests: {sum(nreq.values())}.", ""]
    if kinds:
        md += ["| point | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        md += [f"| {'/'.join(map(str, k))} | " + " | ".join(str(errs[k].get(x, 0)) for x in kinds) + " |"
               for k in sorted(errs) if any(errs[k].values())]

    out = {"rows": [{"combo": k[0], "engine": k[1], "workload": k[2], "users": k[3], **v} for k, v in sorted(rows.items())],
           "quality": quality, "bf16_engine_agreement": bf_pairs, "kv": kv, "meta": meta}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
