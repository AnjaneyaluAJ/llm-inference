# Step 3: quantization of Qwen3-32B on 8x L40S (BF16 vs FP8 vs INT4)

**Question:** how much speed, memory and money does quantizing a 32B model buy on L40S GPUs, and what
does it cost in answer quality? Quantization also makes the model fit on one GPU, so: once it fits,
is one copy per GPU (tp1dp8) better than the best Step 2 split (tp2dp4)?

## Fixed setup

Same as [Step 2](step2_parallel_plan.md) unless stated:
- **Hardware:** AWS g6e.48xlarge, 8x L40S 48 GB, PCIe with no peer-to-peer. The same instance as
  Step 2.
- **Engines:** the same images (vLLM v0.34.0, SGLang v0.5.23, TensorRT-LLM 1.3.0), at default
  settings, with only the context length (8192) and the parallelism set. Each engine reads the
  quantization format from the checkpoint. The KV cache stays in BF16, the engines' default.
- **Data parallelism:** handled as in Step 2.

## Formats

| Format | Checkpoint | Weights on disk | What is quantized |
|---|---|---|---|
| BF16 | `Qwen/Qwen3-32B` | 65.5 GB | nothing (the Step 2 baseline) |
| FP8 | `Qwen/Qwen3-32B-FP8` | 34.3 GB | weights and activations, FP8 with block scales |
| INT4 | `Qwen/Qwen3-32B-AWQ` | 19.3 GB | weights only, 4-bit AWQ; activations stay 16-bit |

**Checkpoints:**
- The official Qwen checkpoints come first. If an engine cannot load one, it gets one retry with the
  Red Hat checkpoint of the same format: `RedHatAI/Qwen3-32B-FP8-dynamic` (per-channel FP8) or
  `RedHatAI/Qwen3-32B-quantized.w4a16` (GPTQ INT4). Any substitution is reported.
- **INT8 is dropped.** No INT8 checkpoint of Qwen3-32B is published by Qwen, Red Hat or NVIDIA, only
  small community uploads. The L40S has FP8 tensor cores, so FP8 is the natural 8-bit choice on it.
- **No quantization is done here.** That keeps the results about serving, not about a home-made
  checkpoint.

## Splits (all use 8 GPUs)

| Format | Splits | Why |
|---|---|---|
| BF16 | tp2dp4 | best Step 2 split; re-measured in this run as the same-session control |
| FP8 | tp1dp8, tp2dp4 | fits on one GPU (34.3 of 46 GB), with little room left for KV cache |
| INT4 | tp1dp8, tp2dp4 | fits on one GPU with about 20 GB left for KV cache |

That is 5 format/split pairs x 3 engines = 15 combinations.

## Workloads, points and tools

- **Workloads and users:** chat, RAG and generate, at 1, 16, 64 and 256 users, as in Step 2.
- **Repeats:** 2, with the order shuffled each repeat.
- **Load generator:** `bench/loadgen.py` only. Steps 1 and 2 already cross-checked it against AIPerf
  twice, and the known differences are documented. Dropping AIPerf halves the time per point.
- **Quality:** the full GSM8K test set (1,319 questions) on every combination in repeat 1. Accuracy,
  plus the share of questions whose final answer matches the BF16 run, paired question by question.
  At around 94% accuracy, the 95% interval on 1,319 questions is about ±1.3 points, so the paired
  agreement is the more sensitive measure.
- **Before the sweep:** start-up check of all 15 combinations with 20 GSM8K questions each, to find
  which engines load which formats. Then the timing of one full combination, to confirm or trim the
  plan.

## Predictions (written before any of this runs)

Ceilings at one user = memory bandwidth of the replica's GPUs (746.7 GB/s each) / weights on disk.
For tp2, the predictions include Step 2's measured communication cost, about 7 ms per token.

| Format | Split | Ceiling (tok/s) | Prediction (tok/s) |
|---|---|---|---|
| BF16 | tp2 | 22.8 | 20-22 (Step 2 measured 21) |
| FP8 | tp1 | 21.8 | 15-20 |
| FP8 | tp2 | 43.5 | 30-38 |
| INT4 | tp1 | 38.7 | 28-36 |
| INT4 | tp2 | 77.4 | 45-60 |

1. **One user:** INT4 tp2 is the fastest of all 15 combinations, and FP8 tp1 is no faster than BF16
   tp2. Reading half the bytes on half the GPUs leaves the per-token read time unchanged.
2. **Many users:** the best FP8 combination has at least 1.4x the peak throughput of BF16 tp2dp4 on
   chat and generate, because it reads half the weight bytes and the L40S has twice the FP8 tensor
   throughput.
3. **INT4 vs FP8 under load:** INT4 is faster at 1 user, but at 256 users the best FP8 combination
   beats the best INT4 one on chat and generate. 4-bit weights must be unpacked to 16-bit before the
   multiply, which costs more as the batch grows.
4. **tp1dp8 vs tp2dp4:** FP8 tp1dp8 has the least KV-cache room, about 20-30k tokens per replica,
   so tp2dp4 beats it on RAG and chat at 256 users. INT4 has more room, and INT4 tp1dp8 gives INT4's
   highest peak on generate.
5. **Quality:** on the full GSM8K, FP8 is within 1 point of BF16 and INT4 within 3 points. Final
   answers agree with BF16 on at least 95% of questions for FP8 and at least 90% for INT4.
6. **Support:** at least one engine cannot serve at least one of the official checkpoints without
   the fallback. TensorRT-LLM with AWQ is the most likely.

## Budget and how it runs

- **Duration:** the start-up check plus one timed combination takes about 1.5 h. The full sweep is
  estimated at 20-28 h at $30.13/h, about $600-850, confirmed or trimmed after the timing.
- **Data:** results are pulled to the laptop as one archive, with the checksum compared against the
  instance's copy.
- **Command:** `python bench/quant_compare.py --reps 2 --out /data/results/step3`.
