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

At concurrency 1 each new token requires reading all model weights from GPU memory, so decode speed
is bounded by memory bandwidth:

- Weights: ~8.2B parameters x 2 bytes = ~16.4 GB (exact byte count to be read from the
  checkpoint and filled in before the first run)
- L40S memory bandwidth: 864 GB/s (spec sheet; measured value to be added)
- Upper bound: 864 / 16.4 = **~53 tokens/s, ~19 ms per token**

Prediction: every engine lands below this bound; the best reaches 80-90% of it. The final numbers
are committed here before any engine is benchmarked.

## Order of work

1. Check g6e on-demand capacity; launch the instance; install the three engines.
2. Measure memory bandwidth; fill in the exact weight bytes; commit the prediction and SLOs.
3. Write the load generator; validate it end to end on one engine.
4. Full sweep: 3 engines x 2 configs x 3 workloads x concurrency and rate sweeps x 3 repeats.
5. Quality check, analysis, charts, write-up, LinkedIn post.

Estimated cost: 15-25 GPU-hours, about $45-75.
