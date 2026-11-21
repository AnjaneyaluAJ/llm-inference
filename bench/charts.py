"""Step 1 charts from results/step1/summary.json.

Usage: python bench/charts.py results/step1 results/step1/charts
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((D / "summary.json").read_text())
ROWS = S["rows"]
CEIL = S["measured_ceiling_tok_per_s"]
SPEC_CEIL = 864 / 15.137

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
ENGINES = ["vllm", "sglang", "trtllm"]
NAME = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}
COL = {"vllm": "#2a78d6", "sglang": "#eb6834", "trtllm": "#1baf7a"}
MARK = {"vllm": "o", "sglang": "s", "trtllm": "D"}
SLO = {"chat": (2000, 50), "rag": (3000, 50), "generate": (1000, 50)}
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def figure(kicker, title, sub):
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=SURFACE)
    fig.text(0.07, 0.92, kicker.upper(), fontsize=15, color=COL["vllm"], weight="bold")
    fig.text(0.07, 0.865, title, fontsize=34, weight="bold", va="center")
    fig.text(0.07, 0.81, sub, fontsize=17, color=INK2, va="center")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.08, 0.08], color="#dcdbd5", lw=1.5))
    fig.text(0.07, 0.045, "github.com/AnjaneyaluAJ/llm-inference", fontsize=16, color=INK2, va="center")
    fig.text(0.93, 0.045, "Anjaneyalu Thippaiah", fontsize=16, color=INK2, va="center", ha="right")
    return fig


def style(ax, fs=15):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs, length=0, pad=8)


def rows(config, engine, w):
    return sorted((r for r in ROWS if r["config"] == config and r["engine"] == engine and r["workload"] == w),
                  key=lambda r: r["concurrency"])


# 1. Single-user decode vs the memory-bandwidth ceiling ------------------------------
fig = figure("Inference, Part 1: serving engines", "One user: all three hit the memory wall",
             "Qwen3-8B on one L40S, generating 1,024 tokens, one request at a time")
ax = fig.add_axes((0.20, 0.22, 0.72, 0.50)); style(ax)
vals = {e: 1000 / rows("defaults", e, "generate")[0]["tpot_p50_ms"] for e in ENGINES}
ys = list(range(len(ENGINES)))[::-1]
ax.axvspan(39, 47, color="#cde2fb", alpha=0.6, lw=0, zorder=0)
ax.text(43, 2.62, "predicted\n39-47", ha="center", va="bottom", fontsize=14, color=INK2)
for y, e in zip(ys, ENGINES):
    ax.barh(y, vals[e], height=0.5, color=COL[e], zorder=3)
    ax.text(vals[e] - 0.8, y, f"{vals[e]:.1f}", ha="right", va="center", fontsize=17, color="white", weight="bold", zorder=4)
for x, lab in ((CEIL, f"measured bandwidth limit {CEIL:.1f}"), (SPEC_CEIL, f"spec-sheet limit {SPEC_CEIL:.1f}")):
    ax.axvline(x, color=INK, lw=2, ls=(0, (4, 3)), zorder=5)
    ax.text(x + 0.4, -0.55, lab, rotation=90, ha="left", va="bottom", fontsize=13, color=INK2)
ax.set_yticks(ys, [NAME[e] for e in ENGINES], fontsize=17)
ax.set_xlim(0, 60); ax.set_ylim(-0.6, 3.1)
ax.set_xlabel("output tokens per second", fontsize=15, labelpad=10)
ax.grid(axis="y", visible=False)
fig.text(0.07, 0.115, "Each token needs all 15.1 GB of weights read from GPU memory. 746.7 GB/s / 15.1 GB = 49.3 tokens/s.\n"
         "Median of 3 runs. Engines are within 1.5% of each other, at 92-94% of the measured limit.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "01_single_user.png", facecolor=SURFACE); plt.close(fig)

# 2. Throughput vs p99 time to first token, per workload -------------------------------
fig = figure("Inference, Part 1: serving engines", "Under load, the engines track each other",
             "Throughput vs worst-case wait for the first token, 1 to 256 concurrent users")
titles = {"chat": "Chat: 1k in, 256 out", "rag": "RAG: 4k in, 128 out", "generate": "Generate: 128 in, 1k out"}
for i, w in enumerate(("chat", "rag", "generate")):
    ax = fig.add_axes((0.09 + i * 0.295, 0.25, 0.25, 0.46)); style(ax, 13)
    for e in ENGINES:
        rs = rows("defaults", e, w)
        ax.plot([r["output_tok_per_s"] for r in rs], [r["ttft_p99_ms"] / 1000 for r in rs], color=COL[e], lw=2,
                marker=MARK[e], ms=8, mec=SURFACE, mew=2, zorder=3)
    ax.set_yscale("log")
    ax.axhline(SLO[w][0] / 1000, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.text(ax.get_xlim()[0] if False else 0.02, SLO[w][0] / 1000 * 1.15, f"target {SLO[w][0] / 1000:.0f} s",
            transform=ax.get_yaxis_transform(), fontsize=12, color=INK2)
    ax.set_title(titles[w], fontsize=15, loc="left", pad=10, color=INK)
    ax.set_xlim(left=0)
    ax.set_xlabel("output tokens/s", fontsize=13)
    if i == 0:
        ax.set_ylabel("p99 time to first token (s, log)", fontsize=13)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
# legend, one row, with markers (identity is never colour alone)
handles = [plt.Line2D([], [], color=COL[e], marker=MARK[e], lw=2, ms=9, mec=SURFACE, mew=2, label=NAME[e]) for e in ENGINES]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.07, 0.765), ncol=3, frameon=False, fontsize=15)
fig.text(0.07, 0.135, "Each point is one concurrency level (1, 2, 4 ... 256), median of 3 runs, default settings.\n"
         "Down and to the right is better. Points above the dashed line miss the latency target.\n"
         "Every engine flattens out: past that point more users only add waiting time.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "02_load_curves.png", facecolor=SURFACE); plt.close(fig)

# 3. Throughput within the latency targets ----------------------------------------------
fig = figure("Inference, Part 1: serving engines", "Within latency targets, vLLM leads by 2-7%",
             "Highest throughput where p99 first-token wait and p99 per-token time both meet target")
best = {(b["config"], b["engine"], b["workload"]): b for b in S["best"]}
rowmap = {(r["config"], r["engine"], r["workload"], r["concurrency"]): r for r in ROWS}
for i, w in enumerate(("chat", "rag", "generate")):
    ax = fig.add_axes((0.09 + i * 0.295, 0.27, 0.25, 0.44)); style(ax, 13)
    for j, e in enumerate(ENGINES):
        b = best[("defaults", e, w)]
        ax.bar(j, b["slo_tok_per_s"], width=0.62, color=COL[e], zorder=3)
        ax.text(j, b["slo_tok_per_s"], f"{b['slo_tok_per_s']:.0f}", ha="center", va="bottom", fontsize=14, color=INK)
        ax.text(j, 0, f"{b['slo_concurrency']} users", ha="center", va="bottom", fontsize=11, color="white", zorder=4)
    ax.set_xticks(range(3), [NAME[e].replace("TensorRT-LLM", "TRT-LLM") for e in ENGINES], fontsize=13)
    ax.grid(axis="x", visible=False)
    ax.set_title(titles[w], fontsize=15, loc="left", pad=10, color=INK)
    ax.margins(y=0.12)
    if i == 0:
        ax.set_ylabel("output tokens/s within target", fontsize=13)
    if w == "rag":
        r8 = rowmap[("defaults", "sglang", "rag", 8)]
        ax.annotate(f"missed at 8 users:\np99 {r8['tpot_p99_ms']:.2f} ms\nvs 50 ms target",
                    xy=(1, best[("defaults", "sglang", "rag")]["slo_tok_per_s"] + 12), xytext=(1, 128),
                    ha="center", fontsize=11, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
fig.text(0.07, 0.15, "Targets (set before measuring): p99 per-token time 50 ms; p99 first token 2 s chat, 3 s RAG, 1 s generate.\n"
         "Value = best of the tested levels (1, 2, 4 ... 256 users) that met target, so it moves in steps: SGLang's\n"
         "RAG bar is a cutoff effect, not a real gap. Default settings, median of 3 runs. Matching KV-cache memory\n"
         "changed every bar by about 1% or less, except that SGLang RAG then passed at 8 users (155).",
         fontsize=13.5, color=INK2, va="center")
fig.savefig(OUT / "03_within_slo.png", facecolor=SURFACE); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
