"""Compare loadgen and AIPerf results from bench/tool_compare.py.

For each engine/workload/concurrency: median over repeats of output tokens/s, p50/p99 time to first
token and p50/p99 time per output token, for
  loadgen          - our client
  aiperf           - AIPerf's own summary, as shipped
  aiperf_adjusted  - recomputed from AIPerf's per-request records: requests that re-used a warm-up
                     prompt (prefix-cache hits) dropped from latency stats, and throughput counted
                     with the server-reported output tokens instead of client re-tokenization
Plus AIPerf's output-token undercount and the stress-test error counts.

Usage: python bench/tool_analyze.py results/toolcmp   -> writes summary.json and summary.md there
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def aiperf_point(d):
    s = json.loads((d / "profile_export_aiperf.json").read_text())
    recs = [json.loads(l) for l in open(d / "profile_export.jsonl")]
    warm = {r["metadata"]["conversation_id"] for r in recs if r["metadata"]["benchmark_phase"] == "warmup"}
    prof = [r for r in recs if r["metadata"]["benchmark_phase"] != "warmup"]
    ok = [r for r in prof if "time_to_first_token" in r["metrics"]]
    fresh = [r for r in ok if r["metadata"]["conversation_id"] not in warm]
    m = lambda r, k: r["metrics"][k]["value"] if k in r["metrics"] else None  # noqa: E731
    dur = s["benchmark_duration"]["avg"]  # seconds
    server_tokens = sum(m(r, "usage_completion_tokens") or 0 for r in ok)
    client_tokens = sum(m(r, "output_sequence_length") or 0 for r in ok)
    asis = {"output_tok_per_s": s["output_token_throughput"]["avg"],
            "ttft_p50_ms": s["time_to_first_token"]["p50"], "ttft_p99_ms": s["time_to_first_token"]["p99"],
            "tpot_p50_ms": s["inter_token_latency"]["p50"], "tpot_p99_ms": s["inter_token_latency"]["p99"]}
    ttft = [m(r, "time_to_first_token") for r in fresh]
    tpot = [(m(r, "request_latency") - m(r, "time_to_first_token")) / (m(r, "usage_completion_tokens") - 1)
            for r in ok if (m(r, "usage_completion_tokens") or 0) > 1]
    adj = {"output_tok_per_s": server_tokens / dur, "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
           "tpot_p50_ms": pct(tpot, 50), "tpot_p99_ms": pct(tpot, 99)}
    starts = sorted(r["metadata"]["request_start_ns"] for r in prof)
    ends = sorted(r["metadata"]["request_end_ns"] for r in prof)
    c = int(d.parent.name.rsplit("_c", 1)[1]) if "_c" in d.parent.name else 256
    turn = [(starts[c + i] - ends[i]) / 1e6 for i in range(len(starts) - c)]
    diffs = [(m(r, "output_sequence_length") or 0) - (m(r, "usage_completion_tokens") or 0) for r in ok]
    extra = {"turnaround_ms_p50": pct(turn, 50), "requests_token_count_off": sum(1 for x in diffs if x),
             "token_diff_min": min(diffs, default=0), "token_diff_max": max(diffs, default=0),
             "zero_text": sum(1 for r in prof if (m(r, "output_sequence_length") or 0) == 0),
             "requests": len(prof), "ok": len(ok), "reused_warmup_prompts": len(ok) - len(fresh),
             "client_vs_server_tokens_pct": 100 * (client_tokens - server_tokens) / server_tokens if server_tokens else None,
             "errors": (s.get("error_summary") or [])}
    return asis, adj, extra


def loadgen_turnaround(f, c):
    recs = [json.loads(l) for l in open(f)]
    starts = sorted(r["start"] for r in recs); ends = sorted(r["start"] + r["e2e"] for r in recs if r.get("e2e"))
    return pct([(starts[c + i] - ends[i]) * 1000 for i in range(min(len(starts), len(ends)) - c)], 50)


def loadgen_point(f):
    d = json.loads(f.read_text())
    return {"output_tok_per_s": d["output_tok_per_s"], "ttft_p50_ms": d["ttft_ms"]["50"], "ttft_p99_ms": d["ttft_ms"]["99"],
            "tpot_p50_ms": d["tpot_ms"]["50"], "tpot_p99_ms": d["tpot_ms"]["99"]}, d


def main(root):
    root = Path(root)
    pts = defaultdict(lambda: defaultdict(list))
    extras = defaultdict(list)
    extras_lg = defaultdict(list)
    for p in sorted(root.glob("rep*/*/*_c*")):
        engine, (w, c) = p.parent.name, p.name.rsplit("_c", 1)
        key = (engine, w, int(c))
        if (p / "loadgen.json").exists():
            lp, raw = loadgen_point(p / "loadgen.json")
            pts[key]["loadgen"].append(lp)
            extras_lg[key].append({"errors": raw["requests_err"],
                                   "turnaround_ms_p50": loadgen_turnaround(p / "loadgen.requests.jsonl", int(c))})
        if (p / "aiperf" / "profile_export_aiperf.json").exists():
            asis, adj, extra = aiperf_point(p / "aiperf")
            pts[key]["aiperf"].append(asis); pts[key]["aiperf_adjusted"].append(adj); extras[key].append(extra)
    metrics = ("output_tok_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms")
    rows = []
    for key, tools in sorted(pts.items()):
        row = {"engine": key[0], "workload": key[1], "concurrency": key[2]}
        for t, runs in tools.items():
            row[t] = {k: statistics.median(r[k] for r in runs if r[k] is not None) for k in metrics}
            row[t]["reps"] = len(runs)
            tp = [r["output_tok_per_s"] for r in runs]
            row[t]["rep_spread_pct"] = 100 * (max(tp) - min(tp)) / statistics.median(tp)
        row["aiperf_turnaround_ms_p50"] = [e["turnaround_ms_p50"] for e in extras[key]]
        row["loadgen_turnaround_ms_p50"] = [e["turnaround_ms_p50"] for e in extras_lg[key]]
        row["loadgen_errors"] = sum(e["errors"] for e in extras_lg[key])
        row["aiperf_zero_text"] = sum(e["zero_text"] for e in extras[key])
        row["aiperf_requests_token_count_off"] = sum(e["requests_token_count_off"] for e in extras[key])
        row["aiperf_token_diff_range"] = [min(e["token_diff_min"] for e in extras[key]), max(e["token_diff_max"] for e in extras[key])]
        row["aiperf_reused_warmup_prompts"] = [e["reused_warmup_prompts"] for e in extras[key]]
        row["aiperf_client_vs_server_tokens_pct"] = [round(e["client_vs_server_tokens_pct"], 3) for e in extras[key]
                                                    if e["client_vs_server_tokens_pct"] is not None]
        rows.append(row)

    stress = {}
    for d in sorted(x for x in root.glob("stress/*") if x.is_dir()):
        e = {}
        if (d / "loadgen.json").exists():
            lg = json.loads((d / "loadgen.json").read_text())
            e["loadgen"] = {"requests": lg["requests_ok"] + lg["requests_err"], "errors": lg["requests_err"],
                            "error_sample": lg["errors_sample"][:3], "bad_streams_saved": (d / "loadgen.bad_streams.jsonl").exists()}
        if (d / "aiperf" / "profile_export_aiperf.json").exists():
            _, _, ex = aiperf_point(d / "aiperf")
            recs = [json.loads(l) for l in open(d / "aiperf" / "profile_export.jsonl")]
            prof = [r for r in recs if r["metadata"]["benchmark_phase"] != "warmup"]
            zero_text = [r for r in prof if (r["metrics"].get("output_sequence_length", {}).get("value") or 0) == 0]
            e["aiperf"] = {"requests": ex["requests"], "ok": ex["ok"], "errors": ex["errors"], "zero_text_records": len(zero_text)}
        stress[d.name] = e

    (root / "summary.json").write_text(json.dumps({"rows": rows, "stress": stress}, indent=1))
    md = ["# loadgen vs AIPerf (median of repeats, default engine settings)", "",
          "Ratio = AIPerf / loadgen. 'adj' = AIPerf recomputed without re-used warm-up prompts and with server token counts.", "",
          "| engine | workload | users | tok/s loadgen | tok/s AIPerf | ratio | ratio adj | TTFT p50 lg / AIPerf / adj (ms) | TPOT p50 lg / AIPerf / adj (ms) | rep spread lg / AIPerf (%) | warm-up prompts re-used |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "loadgen" not in r or "aiperf" not in r:
            continue
        lg, ap, aj = r["loadgen"], r["aiperf"], r["aiperf_adjusted"]
        md.append(f"| {r['engine']} | {r['workload']} | {r['concurrency']} | {lg['output_tok_per_s']:.1f} | {ap['output_tok_per_s']:.1f} | "
                  f"{ap['output_tok_per_s'] / lg['output_tok_per_s']:.3f} | {aj['output_tok_per_s'] / lg['output_tok_per_s']:.3f} | "
                  f"{lg['ttft_p50_ms']:.0f} / {ap['ttft_p50_ms']:.0f} / {aj['ttft_p50_ms']:.0f} | "
                  f"{lg['tpot_p50_ms']:.2f} / {ap['tpot_p50_ms']:.2f} / {aj['tpot_p50_ms']:.2f} | "
                  f"{lg['rep_spread_pct']:.1f} / {ap['rep_spread_pct']:.1f} | {r['aiperf_reused_warmup_prompts']} |")
    toks = [x for r in rows for x in r["aiperf_client_vs_server_tokens_pct"]]
    if toks:
        md += ["", f"AIPerf client token count vs server count: median {statistics.median(toks):+.3f}%, range {min(toks):+.3f}% to {max(toks):+.3f}%"]
    md += [f"Requests where AIPerf's count differs from the server's: {sum(r['aiperf_requests_token_count_off'] for r in rows)} of "
           f"{sum(x['ok'] for k in extras for x in extras[k])}",
           f"Client turnaround (next request start - previous end), median of points: loadgen "
           f"{statistics.median(x for r in rows for x in r['loadgen_turnaround_ms_p50']):.2f} ms, AIPerf "
           f"{statistics.median(x for r in rows for x in r['aiperf_turnaround_ms_p50'] if x is not None):.2f} ms",
           f"Errors: loadgen {sum(r['loadgen_errors'] for r in rows)}, AIPerf zero-text {sum(r['aiperf_zero_text'] for r in rows)}"]
    md += ["", "## Stress point: generate, 256 users, 1024 requests", "", "```", json.dumps(stress, indent=1), "```"]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
