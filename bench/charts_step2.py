"""Step 2 charts from results/step2/summary.json (same style as bench/charts.py).

Usage: python bench/charts_step2.py results/step2 results/step2/charts
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((D / "summary.json").read_text())
PT = {(r["split"], r["engine"], r["workload"], r["users"]): r for r in S["rows"]}

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
ENGINES = ["vllm", "sglang", "trtllm"]
NAME = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TRT-LLM"}
COL = {"vllm": "#2a78d6", "sglang": "#eb6834", "trtllm": "#1baf7a"}
SPLITS = ["tp8", "tp4dp2", "tp2dp4"]
SNAME = {"tp8": "tp8\n1 copy x 8 GPUs", "tp4dp2": "tp4dp2\n2 copies x 4", "tp2dp4": "tp2dp4\n4 copies x 2"}
SCOL = {"tp8": "#7b5cd6", "tp4dp2": "#c98a1b", "tp2dp4": "#0f766e"}
SMARK = {"tp8": "o", "tp4dp2": "s", "tp2dp4": "D"}
USERS = [1, 16, 64, 256]
CEIL = {"tp8": 91.2, "tp4dp2": 45.6, "tp2dp4": 22.8}
TITLES = {"chat": "Chat: 1k in, 256 out", "rag": "RAG: 4k in, 128 out", "generate": "Generate: 128 in, 1k out"}
KICK = "Inference, Part 2: splitting 8 GPUs"
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def tok(s, e, w, c, tool="loadgen"):
    return PT[(s, e, w, c)][tool]["output_tok_per_s"]


def peak(s, e, w):
    # tp8 RAG at 256 users: loadgen's read timeout dropped requests, so that point is AIPerf (see README)
    vals = [tok(s, e, w, c, "aiperf" if (s, w, c) == ("tp8", "rag", 256) else "loadgen") for c in USERS]
    return max(vals)


def figure(title, sub):
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=SURFACE)
    fig.text(0.07, 0.92, KICK.upper(), fontsize=15, color=COL["vllm"], weight="bold")
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


def engine_legend(fig, y):
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[e], label=NAME[e].replace("TRT-LLM", "TensorRT-LLM")) for e in ENGINES]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.07, y), ncol=3, frameon=False, fontsize=15)


# 1. One user: more GPUs per copy is faster, but further below its ceiling --------------------
fig = figure("One user: more GPUs per copy is faster", "Qwen3-32B, generating 1,024 tokens, one request at a time")
ax = fig.add_axes((0.24, 0.22, 0.68, 0.48)); style(ax)
bh = 0.24
for i, s in enumerate(SPLITS):
    y0 = 2 - i
    for j, e in enumerate(ENGINES):
        y = y0 + (1 - j) * bh
        v = tok(s, e, "generate", 1)
        ax.barh(y, v, height=bh - 0.03, color=COL[e], zorder=3)
        ax.text(v - 0.8, y, f"{v:.1f}", ha="right", va="center", fontsize=13, color="white", weight="bold", zorder=4)
    ax.plot([CEIL[s]] * 2, [y0 - 1.6 * bh, y0 + 1.6 * bh], color=INK, lw=2, ls=(0, (4, 3)), zorder=5)
    lo, hi = min(tok(s, e, "generate", 1) for e in ENGINES), max(tok(s, e, "generate", 1) for e in ENGINES)
    ax.text(CEIL[s] + 0.8, y0 + 1.75 * bh, f"limit {CEIL[s]}: reached {100 * lo / CEIL[s]:.0f}-{100 * hi / CEIL[s]:.0f}%",
            va="center", fontsize=12.5, color=INK2)
ax.set_yticks([2, 1, 0], [SNAME[s] for s in SPLITS], fontsize=15)
ax.set_xlim(0, 118); ax.set_ylim(-0.55, 2.62)
ax.set_xlabel("output tokens per second", fontsize=15, labelpad=10)
ax.grid(axis="y", visible=False)
engine_legend(fig, 0.755)
fig.text(0.07, 0.125, "Dashed line: memory-bandwidth limit, 746.7 GB/s per GPU x GPUs per copy / 65.5 GB of weights.\n"
         "tp8 is about 2.5x faster than tp2, but every token also waits for 128 all-reduces between GPUs,\n"
         "and on this instance the GPUs have no direct link. Median of 2 runs.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "01_single_user.png", facecolor=SURFACE); plt.close(fig)

# 2. Peak throughput per split -----------------------------------------------------------------
fig = figure("Many users: four small copies win", "Peak output tokens/s, best of 1, 16, 64 and 256 users")
for i, w in enumerate(("chat", "rag", "generate")):
    ax = fig.add_axes((0.09 + i * 0.295, 0.25, 0.25, 0.46)); style(ax, 13)
    for k, s in enumerate(SPLITS):
        for j, e in enumerate(ENGINES):
            x = k + (j - 1) * 0.27
            v = peak(s, e, w)
            ax.bar(x, v, width=0.25, color=COL[e], zorder=3)
            ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(3), SPLITS, fontsize=13)
    ax.grid(axis="x", visible=False)
    ax.set_title(TITLES[w], fontsize=15, loc="left", pad=10, color=INK)
    ax.margins(y=0.14)
    if i == 0:
        ax.set_ylabel("peak output tokens/s", fontsize=13)
engine_legend(fig, 0.765)
fig.text(0.07, 0.145, "tp2dp4 over tp8: 2.2-7.6x, depending on engine and workload. Default settings, median of 2 runs.\n"
         "These peaks come with long waits: at 256 users the median first token took seconds to minutes.\n"
         "RAG on tp8 at 256 users is measured with AIPerf (loadgen timed out on part of those requests).",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "02_peak.png", facecolor=SURFACE); plt.close(fig)

# 3. The best split depends on the load (vLLM) ---------------------------------------------------
fig = figure("The best split depends on the load", "vLLM, output tokens/s at 1, 16, 64 and 256 concurrent users")
for i, w in enumerate(("chat", "rag", "generate")):
    ax = fig.add_axes((0.09 + i * 0.295, 0.25, 0.25, 0.46)); style(ax, 13)
    for s in SPLITS:
        ys = [tok(s, "vllm", w, c, "aiperf" if (s, w, c) == ("tp8", "rag", 256) else "loadgen") for c in USERS]
        ax.plot(range(4), ys, color=SCOL[s], lw=2, marker=SMARK[s], ms=8, mec=SURFACE, mew=2, zorder=3)
    ax.set_xticks(range(4), [str(c) for c in USERS], fontsize=13)
    ax.set_xlabel("concurrent users", fontsize=13)
    ax.set_title(TITLES[w], fontsize=15, loc="left", pad=10, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if i == 0:
        ax.set_ylabel("output tokens/s (log scale)", fontsize=13)
h = [plt.Line2D([], [], color=SCOL[s], marker=SMARK[s], lw=2, ms=9, mec=SURFACE, mew=2,
                label=SNAME[s].replace("\n", ": ")) for s in SPLITS]
fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.07, 0.765), ncol=3, frameon=False, fontsize=14)
fig.text(0.07, 0.14, "Chat and generate: tp8 wins at 1 user, tp4dp2 at 16, tp2dp4 at 256. RAG: tp2dp4 from 16 users.\n"
         "SGLang and TensorRT-LLM rank the splits the same way at every point, except SGLang\n"
         "generate at 256 users, where tp4dp2 stays ahead. Median of 2 runs.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "03_by_load.png", facecolor=SURFACE); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
