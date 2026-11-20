# Step 1: vLLM vs SGLang vs TensorRT-LLM

**Question:** for the same model on the same GPU, which engine serves fastest, and does the answer
change with the type of traffic?

## Fixed setup (identical for every engine)

- Model: Qwen3-8B, BF16, same weights for all engines.
- GPU: 1x NVIDIA L40S 48 GB on AWS g6e.4xlarge (16 vCPU, 128 GiB RAM, $3.00/h on-demand, us-east-1).
  Optional rerun of the tuned configs on 1x H100 if p5 capacity is available.
- Software: each engine's official Docker image at a pinned version; one host, one driver.
  Versions, image digests, driver and GPU clocks are recorded with every run.
- Client: one load generator for all engines, over the same OpenAI-compatible API, with a fixed
  number of output tokens per request (ignore EOS) and greedy decoding.

## Workloads

| Name | Prompt tokens | Output tokens | Stresses |
|---|---|---|---|
| chat | ~1k | ~256 | balanced |
| rag | ~4k | ~128 | prefill (reading the prompt) |
| generate | ~128 | ~1k | decode (generating tokens) |

Each workload is run (a) at fixed concurrency 1, 2, 4, ... 256, and (b) at a fixed request rate
(Poisson arrivals), so that latency under load can be read off directly.

## Metrics

- Time to first token (TTFT), p50 and p99
- Time per output token (TPOT) and inter-token latency, p50 and p99
- Output tokens/s and requests/s
- Goodput: highest request rate at which p99 TTFT and p99 TPOT stay within an SLO
  (SLO values fixed before the first measurement, recorded below)

## Fairness rules

1. Two configurations per engine: **defaults** (as shipped) and **tuned** (following that
   engine's own documentation). Both are reported.
2. Matched features: prefix caching, maximum context length and GPU memory fraction set the same
   way for all engines in each configuration.
3. Quality check: a fixed GSM8K subset on every engine and configuration, to show that no speedup
   comes from changed outputs.
4. One warm-up pass, then 3 measured repeats per point, interleaved across engines; GPU clocks and
   temperature logged during every run.

## Pre-registered prediction (single user, decode)

Committed before any engine was benchmarked (2026-11-20).

At concurrency 1 each new token requires reading every weight once from GPU memory (the input
embedding table is the exception: only one row is gathered). Decode speed is therefore bounded by
memory bandwidth.

| Input | Value | Source |
|---|---|---|
| Weights, total | 16.381 GB (16,381,470,720 bytes) | safetensors headers, Qwen/Qwen3-8B @ b968826 |
| Input embedding (not read per token) | 1.245 GB | same |
| **Bytes read per decoded token** | **15.137 GB** | difference; KV cache adds ~144 KiB per context token (<1% here) |
| L40S bandwidth, spec sheet | 864 GB/s | NVIDIA |
| L40S bandwidth, measured read | 746.7 GB/s (3 runs, ±0.1) | `bench/membw.py`, 4 GiB bf16 sum, torch 2.14 |
| L40S bandwidth, measured copy | 634.2 GB/s | same |

Ceilings at concurrency 1:

- Spec-sheet ceiling: 864 / 15.137 = **57.1 tokens/s (17.5 ms per token)**. No engine can beat this.
- Measured-bandwidth ceiling: 746.7 / 15.137 = **49.3 tokens/s (20.3 ms per token)**.

**Prediction:** at concurrency 1 on the `generate` workload, the fastest engine reaches 80-95% of
the measured-bandwidth ceiling: **39-47 tokens/s, i.e. 21-25 ms per output token**, and all three
engines are within 15% of each other. Single-user decode is a memory-read problem, so the engines
should differ little there; they should differ much more under load (batching, scheduling,
prefill). If an engine beats 49.3 tokens/s, its weight-reading kernels beat a plain PyTorch
reduction, which is a finding in itself, not an error.

## Latency targets (SLOs) used for goodput

Fixed before the first measurement. A request rate "meets the SLO" when p99 TTFT and p99 TPOT both
stay within the target over the whole run.

| Workload | p99 TTFT | p99 TPOT |
|---|---|---|
| chat | 2.0 s | 50 ms |
| rag | 3.0 s | 50 ms |
| generate | 1.0 s | 50 ms |

50 ms per token = 20 tokens/s, faster than people read.

## Order of work

1. Check g6e on-demand capacity; launch the instance; install the three engines.
2. Measure memory bandwidth; fill in the exact weight bytes; commit the prediction and SLOs.
3. Write the load generator; validate it end to end on one engine.
4. Full sweep: 3 engines x 2 configs x 3 workloads x concurrency and rate sweeps x 3 repeats.
5. Quality check, analysis, charts, write-up, LinkedIn post.

Estimated cost: 15-25 GPU-hours, about $45-75.
