# Step 3 results: Qwen3-32B on 8x L40S, BF16 vs FP8 vs INT4

Plan and predictions, written before the run: [docs/step3_quant_plan.md](../../docs/step3_quant_plan.md)
(committed 2026-11-25 06:36 UTC; the start-up check began at 06:37).
Every number below comes from [summary.md](summary.md) and [summary.json](summary.json), produced by
`python bench/step3_analyze.py results/step3`.

**Setup:**
- Qwen3-32B on the same AWS g6e.48xlarge as [Step 2](../step2/README.md) (8x NVIDIA L40S 48 GB, PCIe, no
  peer-to-peer), 2026-11-25 06:37 to 2026-11-26 02:18 UTC.
- Three formats:
  - BF16: `Qwen/Qwen3-32B`, the Step 2 baseline;
  - FP8: `Qwen/Qwen3-32B-FP8`, block-scaled FP8;
  - INT4: `RedHatAI/Qwen3-32B-quantized.w4a16`, 4-bit GPTQ weights with 16-bit activations. This is the
    plan's fallback; the official `Qwen/Qwen3-32B-AWQ` produced garbage (see [Support](#support)).
- Two splits of the 8 GPUs: tp2dp4 (four copies, 2 GPUs each, the best Step 2 split) and, where the model
  fits on one GPU, tp1dp8 (eight copies).
- Three engines: vLLM v0.34.0, SGLang v0.5.23 and TensorRT-LLM 1.3.0, at default settings with only the
  context length (8192) and the parallelism set. The KV cache is BF16 everywhere.
- 12 of the planned 15 format/split/engine combinations ran. TensorRT-LLM started only BF16 tp2dp4 and
  FP8 tp2dp4.
- Workloads as in Step 2: chat (1k in, 256 out), RAG (4k in, 128 out), generate (128 in, 1k out), at 1, 16,
  64 and 256 concurrent users, 2 repeats in shuffled order, measured with this repo's `loadgen`
  (97,488 requests). Numbers are medians of the 2 repeats.
- Quality: the full GSM8K test set (1,319 questions) on every combination, in repeat 1.

## Results

Charts: [peak throughput](charts/01_peak.png), [one user](charts/02_one_user.png),
[quality](charts/03_quality.png) (`python bench/charts_step3.py results/step3 results/step3/charts`).

### Peak output tokens/s (best of 1/16/64/256 users)

| Format | Split | Engine | Chat | RAG | Generate |
|---|---|---|---|---|---|
| BF16 | tp2dp4 | vLLM | 1,128 | 214 | 2,678 |
| | | SGLang | 968 | 185 | 2,098 |
| | | TRT-LLM | 1,099 | 209 | 2,664 |
| FP8 | tp1dp8 | vLLM | 1,591 | 317 | 3,610 |
| | | SGLang | 1,432 | 314 | 2,042 |
| FP8 | tp2dp4 | vLLM | 1,246 | 211 | 3,244 |
| | | SGLang | 1,239 | 227 | 2,804 |
| | | TRT-LLM | 1,268 | 233 | 3,549 |
| INT4 | tp1dp8 | vLLM | **2,794** | **501** | **6,568** |
| | | SGLang | 2,699 | 474 | 6,530 |
| INT4 | tp2dp4 | vLLM | 1,403 | 229 | 4,139 |
| | | SGLang | 1,324 | 227 | 3,701 |

Against BF16 tp2dp4 on the same engine, the best FP8 combination is 1.12-1.70x and the best INT4 one is
2.34-3.11x. INT4 on eight single-GPU copies is the highest on every workload for both engines that ran it.

### Cost per million output tokens at peak

$30.13 per hour for the instance (on-demand, us-west-2), divided by peak throughput:

| | Chat | RAG | Generate |
|---|---|---|---|
| BF16 tp2dp4, vLLM | $7.42 | $39.14 | $3.13 |
| FP8 tp1dp8, vLLM | $5.26 | $26.38 | $2.32 |
| INT4 tp1dp8, vLLM | $3.00 | $16.72 | $1.27 |

These peaks are at 256 users with long waits: median time to first token at 256 users is 0.8-3.2 s on
chat and generate for INT4 tp1dp8 on vLLM, but 26 s on RAG (all combinations are in
[summary.md](summary.md#time-to-first-token-p50-s-at-256-users)). Peak throughput is not a latency target.

### One user (generate, tokens/s)

| Format | Split | Ceiling | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|---|
| BF16 | tp2 | 22.8 | 21.0 | 21.3 | 21.2 |
| FP8 | tp1 | 21.8 | 21.0 | 15.7 | - |
| FP8 | tp2 | 43.5 | 36.0 | 26.4 | 34.3 |
| INT4 | tp1 | 38.7 | 37.2 | 38.0 | - |
| INT4 | tp2 | 77.4 | 57.2 | **60.1** | - |

Ceiling = 746.7 GB/s per GPU x GPUs per copy / weights on disk (the plan's figures; the INT4 ceiling uses
the AWQ size of 19.3 GB, and the Red Hat checkpoint vLLM actually loaded takes 18.12 GiB = 19.5 GB per GPU,
a ceiling of 38.4 tok/s). On one GPU, vLLM FP8 and both INT4 engines reach 96-98% of the ceiling. With
two GPUs per copy, every format loses a share to communication: BF16 reaches 92-93%, INT4 74-78%, FP8
79-83% on vLLM and TRT-LLM. SGLang FP8 is lower on both splits (61-72%).

### Quality: full GSM8K (1,319 questions)

| Format | Split | Engine | Accuracy | vs BF16, same engine | Same final answer as BF16 | Fixed / broke | McNemar p |
|---|---|---|---|---|---|---|---|
| BF16 | tp2dp4 | vLLM | 93.0% | | | | |
| | | SGLang | 93.3% | | | | |
| | | TRT-LLM | 93.1% | | | | |
| FP8 | tp1dp8 | vLLM | 93.5% | +0.5 | 95.0% | 22 / 16 | 0.42 |
| | | SGLang | 92.2% | -1.1 | 93.4% | 22 / 36 | 0.09 |
| FP8 | tp2dp4 | vLLM | 93.5% | +0.5 | 95.1% | 22 / 16 | 0.42 |
| | | SGLang | 93.8% | +0.5 | 94.7% | 26 / 19 | 0.37 |
| | | TRT-LLM | 93.0% | -0.1 | 94.0% | 25 / 26 | 1.00 |
| INT4 | tp1dp8 | vLLM | 91.4% | -1.7 | 91.3% | 30 / 52 | 0.02 |
| | | SGLang | 92.2% | -1.1 | 91.4% | 32 / 46 | 0.14 |
| INT4 | tp2dp4 | vLLM | 90.8% | -2.2 | 90.4% | 30 / 59 | 0.003 |
| | | SGLang | 91.3% | -2.0 | 90.7% | 31 / 57 | 0.007 |

"Fixed / broke": questions the quantized model got right and BF16 got wrong, and the reverse. The McNemar
test uses only those, so it is paired question by question. Each accuracy's 95% interval is about ±1.4
points.

- **FP8:** no measurable loss. Every difference from BF16 is within 1.1 points and none is significant
  (p > 0.08).
- **INT4:** a small, real loss of 1.1-2.2 points. Three of the four INT4 runs are significant at p < 0.05.
- **Agreement:** the three BF16 engines give the same final answer as each other on only 96.4-97.0% of
  questions. Numerical differences between engines alone change 3-4% of answers, so agreement with BF16
  cannot be expected to reach 100%. FP8 is at 93.4-95.1%, INT4 at 90.4-91.4%.

### KV-cache room per copy (tokens)

| Format | Split | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|
| BF16 | tp2 | 78,848 | 50,739 | 80,800 |
| FP8 | tp1 | 30,208 | ~20,400* | - |
| FP8 | tp2 | 193,536 | 168,467 | 186,304 |
| INT4 | tp1 | 89,600 | ~76,900* | - |
| INT4 | tp2 | 252,928 | 225,411 | - |

\*SGLang with `--dp-size` did not print its start-up KV line in the captured logs; these are estimated
from its running `#token` / `token usage` lines.

### Repeatability and the BF16 control

- **Repeats:** median difference between the two repeats 0.3%; 143 of 144 points within 5%. The one
  exception is INT4 tp1dp8/vLLM/RAG at 16 users, 13% apart.
- **BF16 control:** BF16 tp2dp4 was re-measured in this run, on the same instance. Against Step 2, the
  median difference is 0.2% and all 36 points are within 5% (the largest is 3%). The Step 2 and Step 3
  numbers are directly comparable.

## Predictions vs results

| # | Prediction | Result | |
|---|---|---|---|
| 1 | One user: INT4 tp2 fastest of all; FP8 tp1 no faster than BF16 tp2 | INT4 tp2 fastest (SGLang 60.1, vLLM 57.2). FP8 tp1 21.0 (vLLM) and 15.7 (SGLang) vs BF16 tp2 21.0-21.3. 7 of 12 inside the predicted ranges: one-GPU copies came closer to their ceilings than predicted, SGLang FP8 fell short | **held** (the ranges were off for 5 of 12) |
| 2 | Best FP8 has at least 1.4x BF16 tp2dp4's peak on chat and generate | Chat 1.41x (1,591 vs 1,128). Generate 1.35x (3,610 vs 2,678). Same engine: chat 1.15-1.48x, generate 1.33-1.35x | **wrong on generate**, barely held on chat |
| 3 | At 256 users the best FP8 combination beats the best INT4 one on chat and generate | INT4 is 1.8-2.3x FP8 on chat and generate, and 1.5-1.6x on RAG, on both engines | **wrong** |
| 4 | FP8 tp1dp8 has the least KV room (20-30k tokens per copy), so tp2dp4 beats it on RAG and chat at 256 users; INT4 tp1dp8 gives INT4's highest generate peak | KV room 30k (vLLM), ~20k (SGLang): right. But tp1dp8 beats tp2dp4 on chat (1.16-1.28x) and RAG (1.38-1.50x). INT4 tp1dp8 is INT4's best on generate (1.59-1.76x tp2dp4): right | **wrong** on the main claim |
| 5 | FP8 within 1 point of BF16, INT4 within 3; same final answer on ≥95% (FP8) and ≥90% (INT4) of questions | Accuracy: INT4 within 2.2 points, FP8 within 1 point except SGLang tp1 at -1.1 (not significant). Agreement: INT4 90.4-91.4%, FP8 93.4-95.1% (2 of 5 reach 95%) | **partly wrong**: the FP8 agreement bar was set above the 96-97% at which BF16 engines agree with each other |
| 6 | At least one engine cannot serve at least one official checkpoint without the fallback; TRT-LLM with AWQ most likely | No engine served the official AWQ checkpoint correctly. TRT-LLM could not start INT4 at all, or FP8 on one GPU | **held**, more broadly than predicted |

### Why predictions 2 and 3 were wrong: FP8 did not run as FP8 everywhere

The plan assumed FP8 would use the L40S's FP8 tensor cores. The server logs show three different paths:

- **vLLM:** "Your GPU does not have native support for FP8 computation but FP8 quantization is being
  used. Weight-only FP8 compression will be used leveraging the Marlin kernel."
  ([server.log](rep1/fp8_tp1dp8/vllm/server.log)). With this block-scaled checkpoint on the L40S,
  vLLM v0.34.0 stores the weights in FP8 and computes in 16-bit, exactly as it does with INT4 (Marlin
  kernels for both).
- **SGLang:** runs FP8 x FP8, but with "Using default W8A8 Block FP8 kernel config. Performance might be
  sub-optimal!": it has no tuned kernel configuration for the L40S.
- **TensorRT-LLM:** runs its FP8 block-scaling GEMM; its autotuner logs "using the fallback tactic" for
  some shapes.

So on vLLM, FP8 and INT4 did the same thing, reading 8-bit or 4-bit weights and multiplying in 16-bit,
and INT4 simply read half the bytes. The engines that did use FP8 arithmetic were not faster than
vLLM's weight-only path: at one user SGLang FP8 reached 61-72% of the ceiling vs vLLM's 83-96%.

Prediction 3 also assumed that unpacking 4-bit weights would cost more as the batch grew. That needs a
batch large enough for the arithmetic, not the weight reads, to set the speed. At 256 users on tp1dp8,
each copy has at most 32 requests, and INT4 also has about 3x FP8's KV room per GPU (89.6k vs 30.2k
tokens on vLLM). I did not measure the kernels separately, so the split between these two effects is not
known.

### Why prediction 4 was wrong: one GPU per copy has no communication at all

FP8 tp1dp8 does have the least KV room: 8 copies x 30k = 242k tokens in total, against 4 x 194k = 774k for
FP8 tp2dp4 on vLLM. Yet it serves 1.5x more on RAG at 256 users. Step 2 showed that on this instance
communication between GPUs is expensive, and most of all during prefill: about 32 ms per all-reduce for
a 4k-token prompt at tp8. A tp1 copy sends nothing between GPUs. That is my explanation, consistent with
Step 2, not a separate measurement. Where the KV squeeze does bite is SGLang FP8 generate at 256 users:
with ~20k tokens per copy and up to 32 requests of 1,152 tokens each, its median wait for a first token
was 63 s, and tp2dp4 beat tp1dp8 there (2,804 vs 2,042 tok/s), the one exception to the pattern.

## Support

The start-up check ran 20 GSM8K questions on every combination before the sweep
([check](check), [fallback/check](fallback/check), logs [check3.log](check3.log) and
[check3fb.log](check3fb.log)):

| Checkpoint | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| `Qwen/Qwen3-32B` (BF16), tp2 | ok | ok | ok |
| `Qwen/Qwen3-32B-FP8`, tp2 | ok | ok | ok |
| `Qwen/Qwen3-32B-FP8`, tp1 | ok | ok | out of GPU memory at start-up |
| `RedHatAI/Qwen3-32B-FP8-dynamic`, tp1 (fallback, TRT-LLM only) | | | out of GPU memory at start-up |
| `Qwen/Qwen3-32B-AWQ`, tp1 and tp2 | starts, **0 of 20 correct** | starts, **0 of 20 correct** | tp1: out of memory; tp2: `AssertionError` while loading |
| `RedHatAI/Qwen3-32B-quantized.w4a16` (fallback), tp1 and tp2 | 17-18 of 20 | 17-18 of 20 | `TypeError` in its compressed-tensors config parser |

- **The official AWQ checkpoint** loaded on vLLM and SGLang without an error and then generated
  repeated tokens ("locklocklock...", [quality.jsonl](check/int4_tp1dp8/vllm/quality.jsonl)). A server
  that starts cleanly is not proof that it works; only the quality check caught this. I did not find
  the cause. All INT4 results here use the Red Hat GPTQ checkpoint.
- **TensorRT-LLM on one GPU:** the FP8 weights take 32.5 GiB per GPU (as vLLM loaded
  them) of the L40S's 44.4 GiB, and at default
  settings TensorRT-LLM ran out of memory during start-up. Lowering its memory settings might have made
  it fit; that is a tuning step outside this plan.

## Errors

230 requests, all on generate (1,024 output tokens) and all on vLLM or TensorRT-LLM, returned 1,024
tokens with no text: the same text-less streams traced to the engines' output in
[Step 2](../step2/README.md#errors). They are counted as errors and excluded from throughput, at most
31 of 2,048 requests (1.5%) at a point. INT4 had 6, SGLang none. There were no timeouts and no other
errors.

## Deviations from the plan

- **INT4 checkpoint:** the Red Hat GPTQ fallback, as the plan allowed, because the official AWQ one
  produced garbage on vLLM and SGLang and did not start on TensorRT-LLM.
- **TensorRT-LLM:** 2 of its 5 combinations ran; 12 combinations in all instead of 15.
- **Duration:** about 20 h end to end, including the start-up checks (06:37 to 02:18 UTC), about
  $590 at $30.13/h; the plan estimated 21.5-29.5 h.
- **Plan wording:** the plan describes the FP8 checkpoint as quantizing "weights and activations".
  That is what the checkpoint allows; vLLM chose weight-only on this GPU (see above).

## Limits

- One model, one instance type, default engine settings. FP8 on H100-class GPUs, where vLLM uses FP8
  arithmetic, could rank differently.
- One INT4 checkpoint (GPTQ via Red Hat). Other 4-bit methods may differ in speed and quality.
- GSM8K is one quality measure, on short arithmetic reasoning. It does not cover long contexts, code
  or other languages.
- Two repeats; quality from one run per combination.
- The cost figures and the explanations for predictions 2-4 were worked out after the run; they were
  not predictions.
