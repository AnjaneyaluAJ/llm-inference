# Step 2 results: Qwen3-32B on 8x L40S, tensor vs data parallel

Median of 2 repeats. Output tokens/s unless stated. Default engine settings.

## chat: output tokens/s (loadgen / AIPerf)

| split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| tp8 | vllm | 42 / 43 | 153 / 165 | 183 / 186 | 177 / 177 |
| tp8 | sglang | 46 / 46 | 188 / 202 | 234 / 238 | 248 / 247 |
| tp8 | trtllm | 43 / 44 | 155 / 168 | 183 / 188 | 183 / 183 |
| tp4dp2 | vllm | 34 / 34 | 291 / 305 | 535 / 538 | 654 / 653 |
| tp4dp2 | sglang | 35 / 35 | 327 / 327 | 577 / 580 | 741 / 739 |
| tp4dp2 | trtllm | 34 / 34 | 313 / 314 | 523 / 522 | 643 / 647 |
| tp2dp4 | vllm | 20 / 20 | 262 / 263 | 690 / 681 | 1113 / 1126 |
| tp2dp4 | sglang | 20 / 21 | 266 / 265 | 674 / 675 | 966 / 964 |
| tp2dp4 | trtllm | 20 / 20 | 270 / 268 | 673 / 677 | 1095 / 1104 |

## rag: output tokens/s (loadgen / AIPerf)

| split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| tp8 | vllm | 19 / 20 | 28 / 31 | 28 / 29 | 21 / 28 |
| tp8 | sglang | 23 / 24 | 37 / 42 | 38 / 40 | 37 / 39 |
| tp8 | trtllm | 19 / 20 | 28 / 31 | 29 / 29 | 28 / 29 |
| tp4dp2 | vllm | 22 / 23 | 92 / 97 | 105 / 105 | 103 / 103 |
| tp4dp2 | sglang | 24 / 24 | 101 / 107 | 117 / 117 | 121 / 121 |
| tp4dp2 | trtllm | 23 / 23 | 92 / 98 | 106 / 106 | 110 / 110 |
| tp2dp4 | vllm | 16 / 16 | 135 / 140 | 204 / 204 | 214 / 214 |
| tp2dp4 | sglang | 16 / 16 | 133 / 131 | 181 / 181 | 184 / 184 |
| tp2dp4 | trtllm | 16 / 16 | 134 / 137 | 202 / 204 | 209 / 211 |

## generate: output tokens/s (loadgen / AIPerf)

| split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| tp8 | vllm | 51 / 51 | 413 / 419 | 715 / 729 | 723 / 730 |
| tp8 | sglang | 54 / 53 | 433 / 434 | 778 / 776 | 964 / 951 |
| tp8 | trtllm | 52 / 52 | 410 / 420 | 720 / 733 | 723 / 732 |
| tp4dp2 | vllm | 36 / 36 | 417 / 442 | 1144 / 1183 | 2119 / 2134 |
| tp4dp2 | sglang | 37 / 37 | 469 / 469 | 1239 / 1237 | 2374 / 2348 |
| tp4dp2 | trtllm | 37 / 37 | 449 / 462 | 1147 / 1103 | 2049 / 2092 |
| tp2dp4 | vllm | 21 / 21 | 294 / 303 | 964 / 975 | 2672 / 2715 |
| tp2dp4 | sglang | 21 / 21 | 309 / 308 | 1034 / 1032 | 2095 / 2093 |
| tp2dp4 | trtllm | 21 / 21 | 298 / 304 | 974 / 994 | 2656 / 2690 |

## Predictions

### 1. One user (generate, loadgen)

| split | engine | tok/s | ceiling | % of ceiling | predicted | inside? |
|---|---|---|---|---|---|---|
| tp8 | vllm | 51.0 | 91.2 | 56% | 45-70 | yes |
| tp8 | sglang | 53.6 | 91.2 | 59% | 45-70 | yes |
| tp8 | trtllm | 52.5 | 91.2 | 58% | 45-70 | yes |
| tp4dp2 | vllm | 36.0 | 45.6 | 79% | 33-42 | yes |
| tp4dp2 | sglang | 37.2 | 45.6 | 81% | 33-42 | yes |
| tp4dp2 | trtllm | 36.8 | 45.6 | 81% | 33-42 | yes |
| tp2dp4 | vllm | 20.9 | 22.8 | 92% | 19-22 | yes |
| tp2dp4 | sglang | 21.3 | 22.8 | 93% | 19-22 | yes |
| tp2dp4 | trtllm | 21.2 | 22.8 | 93% | 19-22 | yes |

Inside the predicted range: 9 of 9.

### 2. Peak throughput, tp2dp4 vs tp8 (loadgen, best of 1/16/64/256 users)

| engine | workload | tp8 | tp4dp2 | tp2dp4 | tp2dp4 / tp8 |
|---|---|---|---|---|---|
| vllm | chat | 183 | 654 | 1113 | 6.10x |
| sglang | chat | 248 | 741 | 966 | 3.89x |
| trtllm | chat | 183 | 643 | 1095 | 5.97x |
| vllm | generate | 723 | 2119 | 2672 | 3.69x |
| sglang | generate | 964 | 2374 | 2095 | 2.17x |
| trtllm | generate | 723 | 2049 | 2656 | 3.67x |

### 3. RAG at 256 users (AIPerf; loadgen timed out on part of the tp8 requests)

| engine | tp8 | tp4dp2 | tp2dp4 | best |
|---|---|---|---|---|
| vllm | 28 | 103 | 214 | tp2dp4 |
| sglang | 39 | 121 | 184 | tp2dp4 |
| trtllm | 29 | 110 | 211 | tp2dp4 |

### 4. Engines at the same split: spread of peak throughput (max/min - 1)

| split | chat | rag | generate |
|---|---|---|---|
| tp8 | 36% | 38% | 33% |
| tp4dp2 | 15% | 15% | 16% |
| tp2dp4 | 15% | 16% | 28% |

### 5. Tools: AIPerf / loadgen throughput

Within 3%: 85 of 108 points. Range 0.962 (tp4dp2/trtllm/generate/64) to 1.334 (tp8/vllm/rag/256). Median 1.006.

Points outside 3%:

- tp4dp2/trtllm/generate/64: 0.962
- tp2dp4/vllm/generate/16: 1.031
- tp4dp2/sglang/rag/1: 1.032
- tp4dp2/vllm/generate/64: 1.034
- tp4dp2/vllm/rag/1: 1.035
- tp4dp2/trtllm/rag/1: 1.037
- tp8/sglang/rag/256: 1.042
- tp2dp4/vllm/rag/16: 1.044
- tp4dp2/vllm/chat/16: 1.048
- tp4dp2/sglang/rag/16: 1.051
- tp8/sglang/rag/1: 1.056
- tp4dp2/vllm/generate/16: 1.061
- tp4dp2/trtllm/rag/16: 1.062
- tp4dp2/vllm/rag/16: 1.062
- tp8/vllm/rag/1: 1.062
- tp8/trtllm/rag/1: 1.065
- tp8/sglang/chat/16: 1.073
- tp8/vllm/chat/16: 1.076
- tp8/trtllm/chat/16: 1.084
- tp8/vllm/rag/16: 1.123
- tp8/sglang/rag/16: 1.128
- tp8/trtllm/rag/16: 1.135
- tp8/vllm/rag/256: 1.334

## Time to first token, p50 (s, loadgen) at 256 users

| split | engine | chat | rag | generate |
|---|---|---|---|---|
| tp8 | vllm | 9.2 | 444.8 | 6.7 |
| tp8 | sglang | 102.6 | 270.7 | 14.7 |
| tp8 | trtllm | 23.5 | 578.7 | 19.2 |
| tp4dp2 | vllm | 4.6 | 164.6 | 3.3 |
| tp4dp2 | sglang | 31.7 | 175.2 | 2.3 |
| tp4dp2 | trtllm | 12.1 | 88.5 | 6.3 |
| tp2dp4 | vllm | 4.0 | 111.2 | 1.4 |
| tp2dp4 | sglang | 38.3 | 145.2 | 9.0 |
| tp2dp4 | trtllm | 10.2 | 111.7 | 3.4 |

## Repeatability (loadgen throughput, rep1 vs rep2)

Median spread 0.3%; 107 of 108 points within 5%. Largest: tp4dp2/trtllm/generate/64 7%, tp4dp2/vllm/generate/64 4%, tp2dp4/trtllm/generate/16 4%, tp2dp4/vllm/rag/16 4%, tp8/trtllm/generate/64 3%.

## Requests and errors (both repeats)

Requests: loadgen 73116, AIPerf 73116. AIPerf text-less responses: 0.

| point | aiperf_failed | loadgen_no_text | loadgen_timeout |
|---|---|---|---|
| tp2dp4/trtllm/generate/16 | 0 | 4 | 0 |
| tp2dp4/trtllm/generate/64 | 0 | 11 | 0 |
| tp2dp4/trtllm/generate/256 | 0 | 33 | 0 |
| tp2dp4/vllm/generate/16 | 0 | 3 | 0 |
| tp2dp4/vllm/generate/64 | 0 | 11 | 0 |
| tp2dp4/vllm/generate/256 | 0 | 32 | 0 |
| tp4dp2/trtllm/generate/16 | 0 | 4 | 0 |
| tp4dp2/trtllm/generate/64 | 0 | 10 | 0 |
| tp4dp2/trtllm/generate/256 | 0 | 34 | 0 |
| tp4dp2/vllm/generate/16 | 0 | 4 | 0 |
| tp4dp2/vllm/generate/64 | 0 | 10 | 0 |
| tp4dp2/vllm/generate/256 | 0 | 32 | 0 |
| tp8/sglang/rag/256 | 0 | 0 | 370 |
| tp8/trtllm/generate/16 | 0 | 3 | 0 |
| tp8/trtllm/generate/64 | 0 | 11 | 0 |
| tp8/trtllm/generate/256 | 0 | 32 | 0 |
| tp8/trtllm/rag/256 | 0 | 0 | 240 |
| tp8/vllm/generate/16 | 0 | 2 | 0 |
| tp8/vllm/generate/64 | 0 | 11 | 0 |
| tp8/vllm/generate/256 | 0 | 32 | 0 |
| tp8/vllm/rag/256 | 0 | 0 | 1130 |
| total | 0 | 279 | 1740 |

## GSM8K (100 questions, repeat 1)

| split | vllm | sglang | trtllm |
|---|---|---|---|
| tp8 | 94% | 94% | 94% |
| tp4dp2 | 93% | 95% | 95% |
| tp2dp4 | 95% | 95% | 95% |
