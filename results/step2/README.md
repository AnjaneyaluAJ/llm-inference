# Step 2 results: Qwen3-32B on 8x L40S, tensor vs data parallel

Plan and predictions, written before the run: [docs/step2_parallel_plan.md](../../docs/step2_parallel_plan.md).
Every number below comes from [summary.md](summary.md) and [summary.json](summary.json), produced by
`python bench/step2_analyze.py results/step2`.

**Setup:**
- Qwen3-32B in BF16 on an AWS g6e.48xlarge (8x NVIDIA L40S 48 GB, us-west-2), 2026-11-22 to 2026-11-24.
- Three splits of the 8 GPUs:
  - tp8: one copy of the model, tensor-parallel over 8 GPUs;
  - tp4dp2: two copies, 4 GPUs each;
  - tp2dp4: four copies, 2 GPUs each.
- Three engines: vLLM v0.34.0, SGLang v0.5.23 and TensorRT-LLM 1.3.0, at default settings with only the context length and the parallelism set.
- Data parallelism uses each engine's own option where one exists: vLLM `--data-parallel-size` and SGLang `--dp-size`. TensorRT-LLM has no such option for dense models, so it runs separate servers behind nginx (least connections).
- Three workloads:
  - chat: 1k tokens in, 256 out;
  - RAG: 4k in, 128 out;
  - generate: 128 in, 1k out.
- 1, 16, 64 and 256 concurrent users, 2 repeats.
- Each point was measured by both this repo's `loadgen` and NVIDIA AIPerf 0.14.0, back to back on the same running server: 73,116 requests per tool, 146,232 in all.
- Numbers are medians of the 2 repeats.

## The hardware: no peer-to-peer between GPUs

Measured before the sweep ([step2_hw](../step2_hw)):

| | |
|---|---|
| Memory bandwidth | 746.6-746.8 GB/s on every GPU, the same as Step 1 |
| Topology ([topo.txt](../step2_hw/topo.txt)) | GPUs 0-3 on NUMA node 0, 4-7 on node 1; every pair connects through PCIe host bridges (NODE) or across the CPU interconnect (SYS) |
| Peer-to-peer | Not supported for any pair: TensorRT-LLM logs "P2P not supported between local devices" for all of them at start-up ([server.log](rep1/tp8/trtllm/server.log)), and `nvidia-smi topo -p2p r` showed NS for every pair. GPU-to-GPU traffic goes through host memory |
| NCCL all-reduce, 10 KB | 55 µs (2 GPUs), 60 µs (4), 86 µs (8) |
| NCCL all-reduce, 16-256 MB, bus bandwidth | 3.7-4.2 GB/s (2 GPUs), 4.4-4.6 GB/s (4), 2.3 GB/s (8) |

For scale, one NVLink-connected GPU pair moves hundreds of GB/s. Communication between GPUs is
expensive on this instance, and most expensive across all 8.

## Results

### Peak output tokens/s (best of 1/16/64/256 users)

| split | engine | chat | RAG* | generate |
|---|---|---|---|---|
| tp8 | vLLM | 183 | 28 | 723 |
| tp8 | SGLang | 248 | 39 | 964 |
| tp8 | TRT-LLM | 183 | 29 | 723 |
| tp4dp2 | vLLM | 654 | 105 | 2,119 |
| tp4dp2 | SGLang | 741 | 121 | 2,374 |
| tp4dp2 | TRT-LLM | 643 | 110 | 2,049 |
| tp2dp4 | vLLM | **1,113** | **214** | **2,672** |
| tp2dp4 | SGLang | 966 | 184 | 2,095 |
| tp2dp4 | TRT-LLM | 1,095 | 209 | 2,656 |

All loadgen numbers, except *RAG on tp8 at 256 users, which is AIPerf: loadgen's 600-second read
timeout dropped part of those requests (see Errors). AIPerf is not used for the other peaks because
at 1 and 16 users it re-uses warm-up prompts and reads high (see prediction 5). Full tables, both tools at every point:
[summary.md](summary.md).

**Four copies on 2 GPUs each beat one copy on 8 GPUs by 2.2-7.6x at peak.** On RAG at 256 users,
vLLM served 214 tokens/s as tp2dp4 and 28 as tp8.

### Cost per million output tokens, at peak throughput

At the on-demand price of $30.13 per hour (g6e.48xlarge, us-west-2, AWS Price List API, checked
2026-11-25), divided by the peak throughput above.

| split | engine | chat | RAG | generate |
|---|---|---|---|---|
| tp8 | vLLM | $45.74 | $298.92 | $11.58 |
| tp8 | SGLang | $33.75 | $214.61 | $8.68 |
| tp8 | TRT-LLM | $45.74 | $288.61 | $11.58 |
| tp4dp2 | vLLM | $12.80 | $79.71 | $3.95 |
| tp4dp2 | SGLang | $11.30 | $69.17 | $3.53 |
| tp4dp2 | TRT-LLM | $13.02 | $76.09 | $4.08 |
| tp2dp4 | vLLM | **$7.52** | **$39.11** | **$3.13** |
| tp2dp4 | SGLang | $8.66 | $45.49 | $4.00 |
| tp2dp4 | TRT-LLM | $7.64 | $40.05 | $3.15 |

These are at peak throughput with no latency target. At those peaks, users wait seconds to minutes
for the first token (see [summary.md](summary.md)). They are not comparable with Step 1's costs,
which were measured within latency targets. Going from tp8 to tp2dp4 cuts the cost 2.2-7.6x,
depending on engine and workload. At tp2dp4 the engines differ by 15-28%.

### One user (generate, tokens/s)

| split | ceiling | vLLM | SGLang | TRT-LLM | % of ceiling |
|---|---|---|---|---|---|
| tp2 | 22.8 | 20.9 | 21.3 | 21.2 | 92-93% |
| tp4 | 45.6 | 36.0 | 37.2 | 36.8 | 79-81% |
| tp8 | 91.2 | 51.0 | 53.6 | 52.5 | 56-59% |

The ceiling is the memory bandwidth of the replica's GPUs divided by 65.5 GB of weights. With one
user, tp8 is the fastest, at 2.5x tp2, but it reaches the smallest share of its ceiling.

### Correctness

GSM8K, 100 questions per server in repeat 1: 93-95% on all nine split/engine pairs, so none of the
parallel setups is broken.

| split | vLLM | SGLang | TRT-LLM |
|---|---|---|---|
| tp8 | 94% | 94% | 94% |
| tp4dp2 | 93% | 95% | 95% |
| tp2dp4 | 95% | 95% | 95% |

### Repeatability

The median difference between the two repeats is 0.3%, and 107 of 108 points are within 5%. The
largest is 7%, at tp4dp2/TRT-LLM/generate/64 users.

## Predictions vs results

| # | Prediction | Result | |
|---|---|---|---|
| 1 | One user: tp2 19-22, tp4 33-42, tp8 45-70 tok/s, and tp8 gets the lowest fraction of its ceiling | 9 of 9 inside their ranges; tp8 at 56-59% of its ceiling vs 92-93% for tp2 | **held** |
| 2 | Many users: tp2dp4 has the highest peak on chat and generate, at least 20% above tp8 | tp2dp4 is 3.7-6.1x tp8 for vLLM and TRT-LLM. For SGLang it is 3.9x on chat and 2.2x on generate, but SGLang's best generate split is tp4dp2 (2,374 vs 2,095) | **held for vLLM and TRT-LLM; for SGLang held against tp8 but tp4dp2 was higher on generate** |
| 3 | RAG at 256 users: tp2dp4 runs out of KV-cache room, so tp4dp2 or tp8 matches or beats it | tp2dp4 is the best split for all three engines: 184-214 tok/s vs 103-121 (tp4dp2) and 28-39 (tp8) | **wrong** |
| 4 | Engines stay within about 10% of each other at a given split | Peaks 15-16% apart at tp4dp2, 15-28% at tp2dp4, 33-39% at tp8 | **wrong** |
| 5 | AIPerf and loadgen throughput agree within 3% at every point | 85 of 108 points; median ratio 1.006 | **wrong as stated**; the 23 exceptions are explained below |

### Why prediction 3 was wrong: tp8 pays for every prompt in all-reduces

The KV-cache part of the prediction was right. Each tp2dp4 replica holds 51-81k tokens, and RAG at
256 users needs 64 users x 4.2k tokens = 270k per replica, so most requests wait in a queue.
What I underestimated was how much tp8 pays for prefill on this interconnect.

| KV cache per replica (tokens) | vLLM | SGLang | TRT-LLM |
|---|---|---|---|
| tp8 | 1,069,056 | 915,840 | 1,007,488 |
| tp4dp2 | 407,552 | 341,078 | 389,152 |
| tp2dp4 | 78,848 | ~50,700† | 80,800 |

†The SGLang tp2dp4 server logs were cut at 2 MB, which lost the start-up line. This is estimated from
the runtime "#token / token usage" lines; the same estimate gives 914,651 for tp8, where the log
says 915,840.

**A rough estimate (made after the run, not a prediction).** Qwen3-32B does 2 all-reduces per layer
over 64 layers. For one 4,096-token prompt each all-reduce carries 4,096 x 5,120 x 2 bytes = 42 MB. At
the measured 8-GPU speed that is about 32 ms each, or about **4.1 s of communication per prompt**. At
256 users, tp8 completed one RAG request every 4.6 s on vLLM and 4.4 s on TRT-LLM, so communication
alone accounts for most of the time. SGLang managed 3.3 s, faster than this estimate, so the estimate
is not the whole story for that engine. The same estimate gives 1.8 s per prompt per tp4 replica and
1.4 s per tp2 replica, and those replicas run side by side.

The same arithmetic roughly explains the one-user numbers (also made after the run). Each token
needs 128 all-reduces of 10 KB, at the measured 55/60/86 µs. Add the time to read the weights and the
predicted rates are 19.7 (tp2), 33.8 (tp4) and 45.4 (tp8) tok/s. Measured rates are 7-15% higher,
which is consistent with the engines overlapping some communication with computation, or with their
own all-reduce kernels being faster than the NCCL test.

### Why prediction 4 was wrong: no engine wins at every split

- **tp8:** SGLang leads all three workloads by 33-36% (generate 964 vs 723). vLLM and TRT-LLM are
  within 4% of each other.
- **tp4dp2:** SGLang leads by 10-13% over the next engine.
- **tp2dp4:** SGLang comes last. On generate it reaches 2,095 vs 2,672 for vLLM. Its KV cache per
  replica is the smallest, about 51k tokens against 79-81k. Generate at 256 users needs up to 64 x
  1,152 = 74k tokens per replica, which fits in vLLM's and TRT-LLM's but not in SGLang's. This is
  consistent with SGLang's lower number, but I did not test it directly.
- **Latency:** throughput is not the whole picture. At tp8 with 256 chat users, SGLang's median time
  to first token was 103 s against 9 s for vLLM, while SGLang served 40% more tokens/s (248 vs 177).
- **Data parallelism:** TensorRT-LLM's nginx-fronted replicas performed within 3% of vLLM's
  built-in data parallelism at tp2dp4.

### Prediction 5: where the tools disagree, and why

The 23 points outside 3% fall into three groups:
- **19 at 1 or 16 users, AIPerf always higher (3-14%).** AIPerf re-uses its warm-up prompts in the
  measured run, the issue found in the [Step 1 cross-check](../toolcmp/README.md). Those requests hit
  the engine's prefix cache and skip prefill. At 16 users that is 8 of 64 requests, and at 1 user 1
  of 10. On tp8, where prefill is the expensive part, this matters a lot. At tp8/RAG/16 users, 12.5%
  of requests skip a roughly 4-second prefill, and AIPerf reads 12-13% higher.
- **2 at RAG with 256 users on tp8, AIPerf higher (1.04 and 1.33).** loadgen's 600-second read
  timeout dropped requests that were still waiting for their first token (see Errors). AIPerf
  waited, and its numbers are the ones used for these points.
- **2 at tp4dp2/generate/64 users (0.962 and 1.034).** These are the two points that differ most
  between repeats (7% and 4%).

## Errors

| | requests | failed | detail |
|---|---|---|---|
| AIPerf | 73,116 | 0 | no failed requests and no text-less responses |
| loadgen | 73,116 | 2,019 | 1,740 read timeouts + 279 text-less streams |

**Timeouts:** all 1,740 are at RAG with 256 users on tp8: vLLM 1,130, SGLang 370 and TRT-LLM 240 of
2,048 each. Median time to first token there was 426-891 s (AIPerf), so many requests waited more
than loadgen's 600-second limit. The tp8 RAG/256 point is reported from AIPerf.

**Text-less streams: now traced to the engine's output, not the client.** This is the open issue
from [Step 1](../step1/README.md#5-open-issue-streams-with-no-text). Here 279 requests, all on
generate (1,024 output tokens) and all on vLLM or TensorRT-LLM, finished with
`finish_reason: length` and 1,024 tokens in the usage record. loadgen saved the raw server-sent
events for every one of them: every chunk's `text` field is `""` from the first to the last. The
engine sent no text, so the client did not lose it.

The failures depend on the prompt:
- the prompts are seeded per repeat, and are identical across splits and engines;
- at every split and load level, vLLM and TRT-LLM fail on the same or nearly the same number of
  requests (e.g. 4 on each of the six at 64 users in repeat 1, 15-19 at 256 users);
- SGLang never fails on the same prompts;
- AIPerf, with different prompts, never saw one.

**Likely cause (not yet verified):** loadgen sends `ignore_eos: true` so that every request produces
exactly 1,024 tokens. If the model's first token is end-of-sequence, generation continues anyway
with tokens that decode to empty text. Checking this needs the generated token IDs, which were not
saved. These requests are counted as errors and excluded from loadgen's metrics, a loss of
3-6% of requests at the affected points.

## Deviations from the plan

- **Duration:** the plan estimated 12-18 h. The smoke test (one engine per split) took 9.3 h, with
  about 6.5 h of it on tp8 because of the slow RAG points. The full sweep then took a further 45.9 h
  (2026-11-22 20:10 to 2026-11-24 18:01 UTC). The full plan was kept after seeing this.
- **Repeat 1:** it re-uses the three smoke-test combinations (tp8/vLLM, tp4dp2/SGLang,
  tp2dp4/TensorRT-LLM) instead of re-running them. They used the same seeds and settings, but their
  run order and which tool went first differ from a freshly shuffled repeat. Repeat-to-repeat
  differences are small (median 0.3%), so this does not change any conclusion.
- **SGLang tp2dp4 server logs** were cut at 2 MB (see the KV-cache note).
- **AIPerf `inputs.json`** prompt files (550 MB) are not committed. AIPerf regenerates them from the
  recorded seed. The raw run is kept as a checksum-verified archive outside git.

## Limits

- One model, one instance type, default engine settings.
- No NVLink. On GPUs with fast peer-to-peer links, tensor parallelism costs far less and the
  ranking of splits could change.
- Two repeats.
- The cost figures and the communication estimates were worked out after the run; they were not predictions.
