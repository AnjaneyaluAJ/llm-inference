# Step 1 results: vLLM vs SGLang vs TensorRT-LLM

Plan and pre-registered prediction: [docs/step1_engines_plan.md](../../docs/step1_engines_plan.md).
Numbers: [summary.md](summary.md) / [summary.json](summary.json). Charts: [charts/](charts/).

**Setup.** Qwen3-8B BF16 on 1x L40S 48 GB (AWS g6e.4xlarge, $3.00/h on-demand). vLLM v0.34.0,
SGLang v0.5.23, TensorRT-LLM 1.3.0 (PyTorch backend, `trtllm-serve`), each from its official image.
3 workloads x 9 concurrency levels (1-256) x 3 repeats x 2 configurations, engines in shuffled order
each repeat. 110,808 measured requests over about 19 GPU-hours of sweep, 2026-11-20/21.

## 1. Single user: the prediction held

| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| tokens/s (generate, concurrency 1) | 45.5 | 46.2 | 45.6 |
| % of measured-bandwidth limit (49.3) | 92.2 | 93.6 | 92.5 |

Prediction (committed before any engine ran): fastest engine at 39-47 tokens/s, all within 15%.
Result: 46.2 tokens/s, all within 1.5%. One user is a memory-read problem and every engine reads
the weights at about the same speed. Matched config: identical to 0.1 tokens/s.

## 2. Under load: engines track each other, traffic shape dominates

Throughput within the latency targets (p99 TTFT 2 s chat / 3 s RAG / 1 s generate, p99 TPOT 50 ms),
default settings:

| Workload | vLLM | SGLang | TensorRT-LLM | Users at that point |
|---|---|---|---|---|
| chat (1k in, 256 out) | **506** | 495 | 480 | 16 |
| RAG (4k in, 128 out) | **162** | 106* | 151 | 8 (*SGLang: 4) |
| generate (128 in, 1k out) | **1,866** | 1,788 | 1,747 | 64 |

\* SGLang missed the target at 8 users by 0.15 ms (p99 TPOT 50.15 ms vs 50; 49.5-50.3 ms across the
three repeats). With matched memory it passed at 8 users (155). This is a cutoff effect of testing
only powers of two, not a real gap.

- vLLM leads within targets by 2-7%. Its repeat spread on RAG is up to 5.8%, so the RAG lead over
  TensorRT-LLM (7%) is only just outside the noise.
- Peak throughput (ignoring latency): vLLM 3,185 tokens/s on generate vs 2,722 SGLang and 2,653
  TensorRT-LLM (+17%), but only at 256 users, where p99 first-token wait is 25 s. On chat and RAG the
  peaks are within 6% (chat 1,218-1,257; RAG 228-242).
- **The workload moves capacity far more than the engine does.** At $3.00/h, within targets, vLLM
  costs $0.45 per million output tokens on generate, $1.65 on chat and $5.14 on RAG: 11x between
  workloads, at most 7% between engines.
- Past saturation (chat ~128 users, RAG ~32; generate ~128 for SGLang and TensorRT-LLM, vLLM still
  climbing at 256) throughput stays flat and extra users only queue: RAG p99 first-token wait is
  127-143 s at 256 users on every engine.

## 3. Matching KV-cache memory changed almost nothing

Defaults give different KV capacities (vLLM 186k tokens, TensorRT-LLM 175k, SGLang 158k). In the
matched config every engine had exactly 147,456 tokens. Within-target throughput moved by about 1% or
less everywhere except the SGLang RAG cutoff above. At these sizes none of the workloads was limited by
KV capacity before it was limited by compute or latency.

## 4. Quality: same answers, not identical text

GSM8K, 250 questions, 8-shot, greedy, served at 32 concurrent requests:

- Accuracy 93.6% on every engine and config, except vLLM matched at 94.4% (2 more correct).
- Final numeric answer agrees between any two runs on 244-250 of 250 questions.
- Full text is identical on only 168-189 of 250 between engines, and vLLM defaults vs vLLM matched
  agree on only 189. SGLang defaults vs SGLang matched agreed on all 250. Greedy decoding under
  batching is not bit-exact on vLLM or TensorRT-LLM. Likely cause (not verified here): batch
  composition changes the arithmetic order, which occasionally flips a near-tie token and the text
  diverges from there.

## 5. Open issue: streams with no text

18 of 110,808 requests, at 15 measurement points (vLLM and TensorRT-LLM only, all at 64+ users, none on
SGLang), completed with
the right token count in the usage record but the load generator saw no text chunks. They are
counted as errors and excluded from all metrics (effect on throughput <0.5% at the affected points).
**Not yet attributed:** it could be the engines' streaming or my client's parsing. Next session will
capture the raw SSE stream before anything is claimed. (First seen as a client crash, fixed in e16dea3;
the crashed point was re-measured: `defaults/rep1/trtllm/generate_c256_rerun.json`.)

## Deviations from the plan

- **No H100.** No on-demand p5/p6 capacity in us-east-1/us-east-2/us-west-2 on 2026-11-19/20.
- **No Poisson rate sweep.** "Within target" throughput is read off the closed-loop concurrency
  sweep, so it moves in steps of 2x users. A rate sweep would give finer goodput numbers.
- **"Matched" instead of "tuned".** The second configuration matches KV-cache capacity across
  engines; no engine-specific tuning was done. Prompts are random tokens with unique prefixes, so
  prefix caching cannot help any engine here.
- **vLLM matched restart.** vLLM v0.34 auto-selects 64-token KV blocks; the first matched launch asked
  for 4x the memory and failed. Fixed (4641b73) and resumed; no measured point was affected.
- The quality check compares accuracy and text agreement, not token-level logprobs.
- GPU clocks: median 2,040 MHz under load, but at the heaviest points the SM clock dropped (1st
  percentile ~1,350 MHz, power draw up to 361 W, temperature up to 88 C), i.e. power/thermal limits
  of the L40S. This applies to every engine; engine order was shuffled each repeat.

## Files

`<config>/rep<N>/<engine>/<workload>_c<users>.json` summary per point, `.requests.jsonl` per-request
records, `server.log`, `meta.json` (image digest, command, startup time); `<config>/gpu.csv` GPU
clocks/power/temperature every second; `quality/` GSM8K outputs; `logs/` sweep logs;
`../hw/` bandwidth runs; `../smoke/` first smoke tests.
