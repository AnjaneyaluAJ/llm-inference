"""Step 3 charts from results/step3/summary.json (same style as bench/charts_step2.py).

Usage: python bench/charts_step3.py results/step3 results/step3/charts
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((D / "summary.json").read_text())
PT = {(r["combo"], r["engine"], r["workload"], r["users"]): r for r in S["rows"]}
Q = S["quality"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
ENGINES = ["vllm", "sglang", "trtllm"]
NAME = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TRT-LLM"}
COL = {"vllm": "#2a78d6", "sglang": "#eb6834", "trtllm": "#1baf7a"}
COMBOS = ["bf16_tp2dp4", "fp8_tp2dp4", "fp8_tp1dp8", "int4_tp2dp4", "int4_tp1dp8"]
CNAME = {"bf16_tp2dp4": "BF16\n2 GPUs/copy", "fp8_tp2dp4": "FP8\n2 GPUs/copy", "fp8_tp1dp8": "FP8\n1 GPU/copy",
         "int4_tp2dp4": "INT4\n2 GPUs/copy", "int4_tp1dp8": "INT4\n1 GPU/copy"}
CEIL = {"bf16_tp2dp4": 22.8, "fp8_tp1dp8": 21.8, "fp8_tp2dp4": 43.5, "int4_tp1dp8": 38.7, "int4_tp2dp4": 77.4}
USERS = [1, 16, 64, 256]
TITLES = {"chat": "Chat: 1k in, 256 out", "rag": "RAG: 4k in, 128 out", "generate": "Generate: 128 in, 1k out"}
KICK = "Inference, Part 3: quantization"
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def tok(cb, e, w, c):
    r = PT.get((cb, e, w, c))
    return r["output_tok_per_s"] if r else None


def peak(cb, e, w):
    v = [tok(cb, e, w, c) for c in USERS]
    return max(v) if v[0] is not None else None


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


# 1. Peak throughput per format/split ---------------------------------------------------------
fig = figure("Many users: 4-bit on one GPU each wins", "Peak output tokens/s, best of 1, 16, 64 and 256 users, 8x L40S")
for i, w in enumerate(("chat", "rag", "generate")):
    ax = fig.add_axes((0.09 + i * 0.295, 0.27, 0.25, 0.44)); style(ax, 11)
    for k, cb in enumerate(COMBOS):
        for j, e in enumerate(ENGINES):
            v = peak(cb, e, w)
            if v is None:
                continue
            x = k + (j - 1) * 0.27
            ax.bar(x, v, width=0.25, color=COL[e], zorder=3)
    ax.set_xticks(range(len(COMBOS)), [CNAME[c].replace(" GPUs/copy", "").replace(" GPU/copy", "").replace("\n", "\ntp") for c in COMBOS],
                  fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_title(TITLES[w], fontsize=15, loc="left", pad=10, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if i == 0:
        ax.set_ylabel("peak output tokens/s", fontsize=13)
engine_legend(fig, 0.765)
fig.text(0.07, 0.15, "tp2 = 4 copies on 2 GPUs each; tp1 = 8 copies on 1 GPU each. TensorRT-LLM started only BF16 and FP8 on tp2.\n"
         "INT4 on one GPU per copy: 2.3-3.1x BF16 on the same engine. Best FP8: 1.1-1.7x.\n"
         "Default settings, median of 2 runs. INT4 = RedHatAI w4a16 (GPTQ) checkpoint.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "01_peak.png", facecolor=SURFACE); plt.close(fig)

# 2. One user ----------------------------------------------------------------------------------
fig = figure("One user: fewer bytes per GPU is faster", "Qwen3-32B, generating 1,024 tokens, one request at a time")
ax = fig.add_axes((0.24, 0.22, 0.68, 0.5)); style(ax)
bh = 0.24
for i, cb in enumerate(COMBOS):
    y0 = len(COMBOS) - 1 - i
    for j, e in enumerate(ENGINES):
        v = tok(cb, e, "generate", 1)
        if v is None:
            continue
        y = y0 + (1 - j) * bh
        ax.barh(y, v, height=bh - 0.03, color=COL[e], zorder=3)
        ax.text(v - 0.8, y, f"{v:.1f}", ha="right", va="center", fontsize=11.5, color="white", weight="bold", zorder=4)
    ax.plot([CEIL[cb]] * 2, [y0 - 1.6 * bh, y0 + 1.6 * bh], color=INK, lw=2, ls=(0, (4, 3)), zorder=5)
    ax.text(CEIL[cb] + 0.8, y0 + 1.2 * bh, f"limit {CEIL[cb]}", va="center", fontsize=12, color=INK2)
ax.set_yticks(range(len(COMBOS) - 1, -1, -1), [CNAME[c] for c in COMBOS], fontsize=14)
ax.set_xlim(0, 90); ax.set_ylim(-0.55, len(COMBOS) - 0.35)
ax.set_xlabel("output tokens per second", fontsize=15, labelpad=10)
ax.grid(axis="y", visible=False)
engine_legend(fig, 0.765)
fig.text(0.07, 0.125, "Dashed line: memory-bandwidth limit, 746.7 GB/s per GPU x GPUs per copy / weights on disk.\n"
         "On one GPU, vLLM FP8 and INT4 reach 96-98% of it; with 2 GPUs, the link between them costs a share.\n"
         "FP8 on one GPU is no faster than BF16 on two. Median of 2 runs.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "02_one_user.png", facecolor=SURFACE); plt.close(fig)

# 3. Quality --------------------------------------------------------------------------------------
fig = figure("Quality: FP8 holds, INT4 drops 1-2 points", "GSM8K, all 1,319 questions, accuracy with 95% interval")
ax = fig.add_axes((0.24, 0.22, 0.68, 0.5)); style(ax)
bf = [Q[f"bf16_tp2dp4/{e}"]["accuracy"] for e in ENGINES]
ax.axvspan(min(bf), max(bf), color=GRID, alpha=0.7, zorder=1)
ax.text(max(bf) + 0.08, len(COMBOS) - 0.45, "BF16 range", fontsize=12, color=INK2, va="center")
for i, cb in enumerate(COMBOS):
    y0 = len(COMBOS) - 1 - i
    for j, e in enumerate(ENGINES):
        q = Q.get(f"{cb}/{e}")
        if not q:
            continue
        y = y0 + (1 - j) * 0.22
        ax.plot(q["ci95"], [y, y], color=COL[e], lw=2, zorder=3)
        ax.plot(q["accuracy"], y, "o", color=COL[e], ms=9, mec=SURFACE, mew=2, zorder=4)
        lab = f"{q['accuracy']:.1f}%"
        if "agree_with_bf16_pct" in q:
            lab += f"   same answer as BF16: {q['agree_with_bf16_pct']:.1f}%"
        ax.text(95.4, y, lab, va="center", fontsize=11, color=INK2)
ax.set_yticks(range(len(COMBOS) - 1, -1, -1), [CNAME[c] for c in COMBOS], fontsize=14)
ax.set_xlim(88.5, 100); ax.set_ylim(-0.55, len(COMBOS) - 0.3)
ax.set_xticks([89, 90, 91, 92, 93, 94, 95])
ax.set_xlabel("GSM8K accuracy, %", fontsize=15, labelpad=10)
ax.grid(axis="y", visible=False)
engine_legend(fig, 0.765)
fig.text(0.07, 0.125, "FP8: within 1.1 points of BF16 on the same engine, none significant. INT4: 1.1-2.2 points lower,\n"
         "significant (paired McNemar p < 0.05) in 3 of 4 runs. The BF16 engines give the same final answer as\n"
         "each other on 96-97% of questions, so 100% agreement is not expected. Greedy decoding, 1 run each.",
         fontsize=14, color=INK2, va="center")
fig.savefig(OUT / "03_quality.png", facecolor=SURFACE); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
