# Cross-check: our load generator vs NVIDIA AIPerf

Question: would the Step 1 numbers change if they had been measured with NVIDIA AIPerf instead of
our own client (`bench/loadgen.py`)? This run benchmarks the engines and the benchmarking tool at the
same time.

Numbers: [summary.md](summary.md) / [summary.json](summary.json). Code:
[bench/tool_compare.py](../../bench/tool_compare.py) (runs both tools),
[bench/tool_analyze.py](../../bench/tool_analyze.py) (compares them).

## Setup

- **Server and hardware:** same as Step 1. Qwen3-8B BF16 on 1x L40S (g6e.4xlarge), the same three
  engine images, default settings. Both clients run on the same host as the server, over localhost.
- **Tools:** AIPerf 0.14.0 vs loadgen.
- **Points:** chat (1k in / 256 out), RAG (4k / 128) and generate (128 / 1k), each at 1, 16 and 64
  concurrent users.
- **Repeats:** 3, with engine order shuffled each time.
- **Pairing:** for each point, both tools ran back to back against the same running server. Request
  count, warm-up count, prompt length and output length were the same (end-of-sequence ignored). Which
  tool went first alternated from point to point.
- **Stress test:** one per engine, generate at 256 users with 1,024 requests per tool, both tools
  capturing the raw stream.
- **Totals:** 11,982 measured requests per tool, run 2026-11-21 between 16:47 and 22:47 UTC.

Each tool was used as it ships, apart from matching the workload:

| | loadgen | AIPerf |
|---|---|---|
| Prompts | random tokens, a unique prefix per request | default synthetic text (Shakespeare) |
| Output token count | server's `usage` count | re-tokenizes the returned text |
| Warm-up | separate prompts, discarded | warm-up prompts are sent again in the measured phase |
| Time to first token | first chunk with text | first chunk with text |

## Results

**1. Throughput agrees.**
- AIPerf / loadgen output tokens/s ranges from 0.987 to 1.021, with a median of 0.997. 24 of the 27
  points are within 1%.
- Repeat-to-repeat spread was up to 2.0% for loadgen and up to 2.6% for AIPerf.
- Two small, consistent patterns:
  - **Long generation:** AIPerf reads 0.3-0.9% lower on every engine. Its median time per output token
    is 0.3-0.8% higher there.
  - **Chat at 16 users:** AIPerf reads 0.7-2.1% higher.

**2. The Step 1 conclusions do not change.**
- The engine ranking by throughput is the same under both tools at all 9 workload and user-count
  combinations.
- Pass/fail against the Step 1 latency targets (p99 first token 2 s chat / 3 s RAG / 1 s generate, p99
  per-token 50 ms) is the same for both tools at all 27 points.

**3. Median time to first token differs by up to 31% at 16 and 64 users.** p99 agrees much better:
within -9% to +5% at every point.

| | median first token, AIPerf vs loadgen |
|---|---|
| chat, 16 users | -31% TensorRT-LLM, -11% SGLang, -3% vLLM |
| generate, 16 and 64 users | -7% to -17% on every engine |
| chat, 64 users | -2% to +2% |
| RAG | -2% to +4% |
| single user | within a few ms |

Two causes found:
- **Re-used warm-up prompts.** AIPerf sends its 8 warm-up prompts again during the measured phase. The
  engine already has them in its prefix cache, so they get their first token sooner. vLLM's own
  counter confirms it: at generate / 16 users, 512 of the 8,192 prompt tokens were cache hits, which
  is the re-used warm-ups and nothing else. Dropping those requests ("adj" in summary.md) closes part of
  the gap, for example SGLang chat 16 users from -11% to -2%.
- **Request turnaround.** loadgen sends the next request about 0.01 ms after one finishes. AIPerf takes
  a median of 3 ms, and up to 43 ms at generate / 64 users. In a closed loop with equal output
  lengths, requests finish and restart in waves. loadgen restarts a wave almost at once. AIPerf
  spreads it out a little, so fewer prompts queue for prefill at the same moment.
  - The first wave of each run, which both tools launch within about 10 ms, has matching first-token
    times: TensorRT-LLM chat 16 users, 902 vs 894 ms.
  - The gap appears only in later waves: 897 vs 639 ms.
  - That this spreading causes the gap is the likely explanation, but it is not proven here.

Neither tool is wrong here: they measure slightly different arrival patterns. At mid-range
concurrency, a closed-loop first-token median is sensitive to client details, so compare such numbers
only when they come from the same tool.

**4. AIPerf's token count is right in total but not per request.**
- Re-tokenizing the returned text gave a different count from the server on 1,231 of 8,910 requests in
  the main grid.
- The total is off by -0.04% (median; range -0.31% to +0.16% per point).
- The worst case was in the stress test: a vLLM response of " KNEELS" repeated hundreds of times. The
  server generated 1,024 tokens, and re-tokenizing the text gave 778. Converting tokens to text and back
  is not one-to-one, especially for degenerate text that runs on past end-of-sequence.

**5. Streams with no text (Step 1 open issue): still not attributed.**
- In the main grid, loadgen recorded 1 in 11,982 (vLLM, generate, 64 users, repeat 2). AIPerf recorded
  0 in 11,982.
- In the stress test, where both tools were capturing the raw stream, there were none on any engine.
- Step 1 saw 18 in 110,808, so the expected count here was small either way. This run cannot decide
  between engine and client. The capture is in place for the next time it happens.

**6. Stress point, 256 users** (one run per engine): AIPerf / loadgen tokens/s.

| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| ratio | 0.975 | 0.974 | 0.991 |

AIPerf reads up to 2.6% lower here, the largest gap in the run, consistent with its slower turnaround.
AIPerf and loadgen share the host's 16 vCPUs with the engine. Whether client CPU load slows the engine
was not measured.

## What this means for Step 1

The Step 1 throughput numbers, rankings and latency-target verdicts hold under an independent
industry tool. Median first-token times at 16-64 users are tool-dependent by up to about 30%, which is
a reason to report p99 and to keep the tool fixed when comparing.

## Limits

- Clients ran on the same host as the server, over localhost. Neither tool's numbers include network
  time.
- Points were 1, 16 and 64 users only, plus one 256-user run per engine.
- Prompt text differs between the tools (random tokens vs Shakespeare). Neither gets prefix-cache hits
  apart from AIPerf's re-used warm-ups.
- AIPerf's raw stream captures (`stress/*/aiperf/profile_export_raw.jsonl.gz`, 42 MB) are kept out of
  the repo. Their checksums are in `stress/raw_md5.txt`.

## Files

`rep<N>/<engine>/<workload>_c<users>/`:
- `loadgen.json` and `loadgen.requests.jsonl`
- `aiperf/`, AIPerf's own output (`profile_export.jsonl` has per-request records, `inputs.json` has
  the prompts sent)

Also:
- `stress/<engine>/`, the 256-user runs
- `gpu.csv`, GPU clocks, power and temperature every second
- `env.txt`, image digests and versions
- `toolcmp.log`, the run log
- `extra/`, AIPerf smoke tests and first trial runs
