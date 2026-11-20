"""Aggregate sweep results: median over repeats, with spread, per config/engine/workload/concurrency.

Also computes, per engine and workload, the highest throughput reached while meeting the
pre-registered SLO (p99 TTFT and p99 TPOT both within target), using the concurrency sweep.

Usage: python bench/analyze.py /data/results/step1  -> writes summary.json and summary.md there
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

SLO = {"chat": (2000, 50), "rag": (3000, 50), "generate": (1000, 50)}  # p99 TTFT ms, p99 TPOT ms
MEASURED_CEILING = 746.7e9 / 15.137e9  # tokens/s at concurrency 1, from docs/step1_engines_plan.md


def main(root):
    root = Path(root)
    pts = defaultdict(list)
    errors = []
    for f in root.glob("*/rep*/*/*_c*.json"):
        config, rep, engine = f.parts[-4], f.parts[-3], f.parts[-2]
        w, c = f.stem.rsplit("_c", 1)
        d = json.loads(f.read_text())
        if d["requests_err"]:
            errors.append((str(f), d["requests_err"], d["errors_sample"][:2]))
        pts[(config, engine, w, int(c))].append(d)
    rows = []
    for (config, engine, w, c), ds in sorted(pts.items()):
        med = lambda xs: statistics.median(xs)  # noqa: E731
        tput = [d["output_tok_per_s"] for d in ds]
        row = {
            "config": config, "engine": engine, "workload": w, "concurrency": c, "reps": len(ds),
            "output_tok_per_s": med(tput),
            "spread_pct": 100 * (max(tput) - min(tput)) / med(tput) if len(ds) > 1 else None,
            "req_per_s": med([d["req_per_s"] for d in ds]),
            "ttft_p50_ms": med([d["ttft_ms"]["50"] for d in ds]), "ttft_p99_ms": med([d["ttft_ms"]["99"] for d in ds]),
            "tpot_p50_ms": med([d["tpot_ms"]["50"] for d in ds]), "tpot_p99_ms": med([d["tpot_ms"]["99"] for d in ds]),
            "errors": sum(d["requests_err"] for d in ds),
        }
        ttft_t, tpot_t = SLO[w]
        row["meets_slo"] = row["ttft_p99_ms"] <= ttft_t and row["tpot_p99_ms"] <= tpot_t
        rows.append(row)

    best = {}
    for r in rows:
        k = (r["config"], r["engine"], r["workload"])
        b = best.setdefault(k, {"peak_tok_per_s": 0, "slo_tok_per_s": 0, "slo_concurrency": None, "slo_req_per_s": 0})
        b["peak_tok_per_s"] = max(b["peak_tok_per_s"], r["output_tok_per_s"])
        if r["meets_slo"] and r["output_tok_per_s"] > b["slo_tok_per_s"]:
            b.update(slo_tok_per_s=r["output_tok_per_s"], slo_concurrency=r["concurrency"], slo_req_per_s=r["req_per_s"])

    single = {f"{r['config']}/{r['engine']}": {"tok_per_s": round(1000 / r["tpot_p50_ms"], 2),
                                              "pct_of_measured_ceiling": round(100 * 1000 / r["tpot_p50_ms"] / MEASURED_CEILING, 1)}
              for r in rows if r["workload"] == "generate" and r["concurrency"] == 1}
    out = {"rows": rows, "best": [dict(zip(("config", "engine", "workload"), k)) | v for k, v in sorted(best.items())],
           "single_user_generate": single, "measured_ceiling_tok_per_s": MEASURED_CEILING, "errors": errors}
    (root / "summary.json").write_text(json.dumps(out, indent=1))

    md = ["# Step 1 summary (median of repeats)", "",
          f"Measured-bandwidth ceiling at concurrency 1: {MEASURED_CEILING:.1f} tokens/s", "",
          "## Single user, generate workload (decode speed = 1000 / median TPOT)", "",
          "| config/engine | tokens/s | % of measured ceiling |", "|---|---|---|"]
    md += [f"| {k} | {v['tok_per_s']} | {v['pct_of_measured_ceiling']} |" for k, v in single.items()]
    md += ["", "## Peak and SLO-constrained throughput (output tokens/s)", "",
           "| config | engine | workload | peak | within SLO | at concurrency |", "|---|---|---|---|---|---|"]
    md += [f"| {b['config']} | {b['engine']} | {b['workload']} | {b['peak_tok_per_s']:.0f} | {b['slo_tok_per_s']:.0f} | {b['slo_concurrency']} |"
           for b in out["best"]]
    spreads = [r["spread_pct"] for r in rows if r["spread_pct"] is not None]
    if spreads:
        md += ["", f"Repeat spread (max-min)/median of throughput: median {statistics.median(spreads):.1f}%, max {max(spreads):.1f}%"]
    md += ["", f"Points with request errors: {len(errors)}"]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
