"""Analyse Step 2 (bench/multi_compare.py): splits x engines x workloads x users, two tools, two repeats.

For every split/engine/workload/users point: median over repeats of output tokens/s, p50/p99 TTFT and
p50/p99 TPOT, from loadgen and from AIPerf (as shipped and adjusted, see bench/tool_analyze.py).
Then the checks against the predictions in docs/step2_parallel_plan.md.

Usage: python bench/step2_analyze.py results/step2   -> writes summary.json and summary.md there
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

from tool_analyze import aiperf_point, loadgen_point

SPLITS = ["tp8", "tp4dp2", "tp2dp4"]
ENGINES = ["vllm", "sglang", "trtllm"]
WORKLOADS = ["chat", "rag", "generate"]
USERS = [1, 16, 64, 256]
# Ceilings and predicted ranges at one user, from the plan (746.7 GB/s per GPU, 65.5 GB of weights)
CEIL = {"tp2dp4": 22.8, "tp4dp2": 45.6, "tp8": 91.2}
PRED = {"tp2dp4": (19, 22), "tp4dp2": (33, 42), "tp8": (45, 70)}
METRICS = ("output_tok_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms")


def med(runs, k):
    xs = [r[k] for r in runs if r.get(k) is not None]
    return statistics.median(xs) if xs else None


def spread(runs):
    tp = [r["output_tok_per_s"] for r in runs]
    return 100 * (max(tp) - min(tp)) / statistics.median(tp) if len(tp) > 1 else None


def main(root):
    root = Path(root)
    pts = defaultdict(lambda: defaultdict(list))
    errs = defaultdict(lambda: defaultdict(int)); zero = defaultdict(int); nreq = defaultdict(lambda: defaultdict(int))
    for p in sorted(root.glob("rep*/*/*/*_c*")):
        split, engine = p.parent.parent.name, p.parent.name
        w, c = p.name.rsplit("_c", 1)
        key = (split, engine, w, int(c))
        if (p / "loadgen.json").exists():
            lp, raw = loadgen_point(p / "loadgen.json")
            pts[key]["loadgen"].append(lp)
            nreq[key]["loadgen"] += raw["requests_ok"] + raw["requests_err"]
            for x in open(p / "loadgen.requests.jsonl"):
                err = json.loads(x).get("error")
                if err:
                    errs[key]["loadgen_" + ("timeout" if "Timeout" in err else "no_text" if "no text" in err else "other")] += 1
        if (p / "aiperf" / "profile_export_aiperf.json").exists():
            asis, adj, extra = aiperf_point(p / "aiperf")
            pts[key]["aiperf"].append(asis); pts[key]["aiperf_adjusted"].append(adj)
            zero[key] += extra["zero_text"]; nreq[key]["aiperf"] += extra["requests"]
            errs[key]["aiperf_failed"] += extra["requests"] - extra["ok"]

    rows = {}
    for key, tools in pts.items():
        row = {t: {**{k: med(runs, k) for k in METRICS}, "reps": len(runs), "rep_spread_pct": spread(runs),
                   "runs_tok_per_s": [r["output_tok_per_s"] for r in runs]} for t, runs in tools.items()}
        row["errors"] = dict(errs[key]); row["aiperf_zero_text"] = zero[key]; row["requests"] = dict(nreq[key])
        rows[key] = row

    quality = {}
    for q in sorted(root.glob("rep*/*/*/quality.summary.txt")):
        d = json.loads(q.read_text())
        quality[f"{q.parent.parent.name}/{q.parent.name}"] = d
    meta = {}
    for m in sorted(root.glob("rep*/*/*/meta.json")):
        d = json.loads(m.read_text())
        meta[f"{m.parent.parent.parent.name}/{m.parent.parent.name}/{m.parent.name}"] = {
            "startup_s": d.get("startup_s"), "dp_method": d.get("dp_method"), "image": d.get("image")}

    g = lambda s, e, w, c, t="loadgen", k="output_tok_per_s": (rows.get((s, e, w, c), {}).get(t) or {}).get(k)  # noqa: E731
    md = ["# Step 2 results: Qwen3-32B on 8x L40S, tensor vs data parallel", "",
          "Median of 2 repeats. Output tokens/s unless stated. Default engine settings.", ""]

    # 1. Throughput tables per workload, loadgen
    for w in WORKLOADS:
        md += [f"## {w}: output tokens/s (loadgen / AIPerf)", "",
               "| split | engine | " + " | ".join(f"{c} users" for c in USERS) + " |", "|---|---|" + "---|" * len(USERS)]
        for s in SPLITS:
            for e in ENGINES:
                cells = []
                for c in USERS:
                    a, b = g(s, e, w, c), g(s, e, w, c, "aiperf")
                    cells.append(f"{a:.0f} / {b:.0f}" if a is not None and b is not None else "-")
                md.append(f"| {s} | {e} | " + " | ".join(cells) + " |")
        md.append("")

    # 2. Prediction checks
    md += ["## Predictions", "", "### 1. One user (generate, loadgen)", "",
           "| split | engine | tok/s | ceiling | % of ceiling | predicted | inside? |", "|---|---|---|---|---|---|---|"]
    p1 = []
    for s in SPLITS:
        for e in ENGINES:
            v = g(s, e, "generate", 1)
            lo, hi = PRED[s]
            ok = lo <= v <= hi
            p1.append(ok)
            md.append(f"| {s} | {e} | {v:.1f} | {CEIL[s]} | {100 * v / CEIL[s]:.0f}% | {lo}-{hi} | {'yes' if ok else 'no'} |")
    md.append(f"\nInside the predicted range: {sum(p1)} of {len(p1)}.\n")

    peak = lambda s, e, w, t="loadgen": max(g(s, e, w, c, t) or 0 for c in USERS)  # noqa: E731
    md += ["### 2. Peak throughput, tp2dp4 vs tp8 (loadgen, best of 1/16/64/256 users)", "",
           "| engine | workload | tp8 | tp4dp2 | tp2dp4 | tp2dp4 / tp8 |", "|---|---|---|---|---|---|"]
    for w in ["chat", "generate"]:
        for e in ENGINES:
            a, m, b = peak("tp8", e, w), peak("tp4dp2", e, w), peak("tp2dp4", e, w)
            md.append(f"| {e} | {w} | {a:.0f} | {m:.0f} | {b:.0f} | {b / a:.2f}x |")
    md += ["", "### 3. RAG at 256 users (AIPerf; loadgen timed out on part of the tp8 requests)", "",
           "| engine | tp8 | tp4dp2 | tp2dp4 | best |", "|---|---|---|---|---|"]
    for e in ENGINES:
        v = {s: g(s, e, "rag", 256, "aiperf") for s in SPLITS}
        md.append(f"| {e} | {v['tp8']:.0f} | {v['tp4dp2']:.0f} | {v['tp2dp4']:.0f} | {max(v, key=v.get)} |")
    md += ["", "### 4. Engines at the same split: spread of peak throughput (max/min - 1)", "",
           "| split | chat | rag | generate |", "|---|---|---|---|"]
    for s in SPLITS:
        cells = []
        for w in WORKLOADS:
            v = [peak(s, e, w) for e in ENGINES]
            cells.append(f"{100 * (max(v) / min(v) - 1):.0f}%")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "### 5. Tools: AIPerf / loadgen throughput", ""]
    ratios = []
    for key, r in sorted(rows.items()):
        if "loadgen" in r and "aiperf" in r and r["loadgen"]["output_tok_per_s"]:
            ratios.append((r["aiperf"]["output_tok_per_s"] / r["loadgen"]["output_tok_per_s"], key))
    within = sum(1 for x, _ in ratios if abs(x - 1) <= 0.03)
    md.append(f"Within 3%: {within} of {len(ratios)} points. Range {min(ratios)[0]:.3f} ({'/'.join(map(str, min(ratios)[1]))}) "
              f"to {max(ratios)[0]:.3f} ({'/'.join(map(str, max(ratios)[1]))}). Median {statistics.median(x for x, _ in ratios):.3f}.")
    outside = sorted((x, k) for x, k in ratios if abs(x - 1) > 0.03)
    if outside:
        md += ["", "Points outside 3%:", ""] + [f"- {'/'.join(map(str, k))}: {x:.3f}" for x, k in outside]

    # 3. Latency at load, repeatability, errors, quality
    md += ["", "## Time to first token, p50 (s, loadgen) at 256 users", "", "| split | engine | chat | rag | generate |", "|---|---|---|---|---|"]
    for s in SPLITS:
        for e in ENGINES:
            md.append(f"| {s} | {e} | " + " | ".join(f"{g(s, e, w, 256, k='ttft_p50_ms') / 1000:.1f}" for w in WORKLOADS) + " |")
    sp = [(r["loadgen"]["rep_spread_pct"], k) for k, r in rows.items() if r.get("loadgen", {}).get("rep_spread_pct") is not None]
    md += ["", "## Repeatability (loadgen throughput, rep1 vs rep2)", "",
           f"Median spread {statistics.median(x for x, _ in sp):.1f}%; {sum(1 for x, _ in sp if x <= 5)} of {len(sp)} points within 5%. "
           f"Largest: " + ", ".join(f"{'/'.join(map(str, k))} {x:.0f}%" for x, k in sorted(sp, reverse=True)[:5]) + "."]
    tot = lambda t: sum(v[t] for v in nreq.values())  # noqa: E731
    kinds = sorted({k for v in errs.values() for k in v})
    md += ["", "## Requests and errors (both repeats)", "",
           f"Requests: loadgen {tot('loadgen')}, AIPerf {tot('aiperf')}. AIPerf text-less responses: {sum(zero.values())}.", "",
           "| point | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    md += [f"| {'/'.join(map(str, k))} | " + " | ".join(str(errs[k].get(x, 0)) for x in kinds) + " |"
           for k in sorted(errs) if any(errs[k].values())]
    md.append("| total | " + " | ".join(str(sum(errs[k].get(x, 0) for k in errs)) for x in kinds) + " |")
    md += ["", "## GSM8K (100 questions, repeat 1)", "", "| split | " + " | ".join(ENGINES) + " |", "|---|---|---|---|"]
    for s in SPLITS:
        md.append(f"| {s} | " + " | ".join(f"{100 * quality[f'{s}/{e}']['accuracy']:.0f}%" if f"{s}/{e}" in quality else "-" for e in ENGINES) + " |")

    out = {"rows": [{"split": k[0], "engine": k[1], "workload": k[2], "users": k[3], **v} for k, v in sorted(rows.items())],
           "quality": quality, "meta": meta}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
