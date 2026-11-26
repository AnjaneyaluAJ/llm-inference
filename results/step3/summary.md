# Step 3 results: Qwen3-32B on 8x L40S, BF16 vs FP8 vs INT4

Median of 2 repeats, loadgen. Output tokens/s unless stated. Default engine settings. INT4 is the Red Hat GPTQ checkpoint (`RedHatAI/Qwen3-32B-quantized.w4a16`); see the README for why.

## chat: output tokens/s

| format/split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 20 | 262 | 690 | 1128 |
| bf16_tp2dp4 | sglang | 20 | 266 | 675 | 968 |
| bf16_tp2dp4 | trtllm | 20 | 268 | 679 | 1099 |
| fp8_tp1dp8 | vllm | 20 | 285 | 933 | 1591 |
| fp8_tp1dp8 | sglang | 15 | 270 | 930 | 1432 |
| fp8_tp2dp4 | vllm | 34 | 405 | 873 | 1246 |
| fp8_tp2dp4 | sglang | 25 | 373 | 853 | 1239 |
| fp8_tp2dp4 | trtllm | 32 | 407 | 874 | 1268 |
| int4_tp1dp8 | vllm | 35 | 488 | 1502 | 2794 |
| int4_tp1dp8 | sglang | 36 | 531 | 1495 | 2699 |
| int4_tp2dp4 | vllm | 52 | 558 | 1073 | 1403 |
| int4_tp2dp4 | sglang | 54 | 573 | 1033 | 1324 |

## rag: output tokens/s

| format/split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 16 | 137 | 205 | 214 |
| bf16_tp2dp4 | sglang | 16 | 130 | 182 | 185 |
| bf16_tp2dp4 | trtllm | 16 | 138 | 201 | 209 |
| fp8_tp1dp8 | vllm | 15 | 161 | 308 | 317 |
| fp8_tp1dp8 | sglang | 13 | 203 | 314 | 314 |
| fp8_tp2dp4 | vllm | 23 | 158 | 206 | 211 |
| fp8_tp2dp4 | sglang | 19 | 155 | 216 | 227 |
| fp8_tp2dp4 | trtllm | 22 | 164 | 209 | 233 |
| int4_tp1dp8 | vllm | 25 | 245 | 449 | 501 |
| int4_tp1dp8 | sglang | 25 | 270 | 430 | 474 |
| int4_tp2dp4 | vllm | 31 | 184 | 229 | 224 |
| int4_tp2dp4 | sglang | 31 | 182 | 219 | 227 |

## generate: output tokens/s

| format/split | engine | 1 users | 16 users | 64 users | 256 users |
|---|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 21 | 295 | 961 | 2678 |
| bf16_tp2dp4 | sglang | 21 | 309 | 1033 | 2098 |
| bf16_tp2dp4 | trtllm | 21 | 300 | 984 | 2664 |
| fp8_tp1dp8 | vllm | 21 | 326 | 1238 | 3610 |
| fp8_tp1dp8 | sglang | 16 | 281 | 988 | 2042 |
| fp8_tp2dp4 | vllm | 36 | 505 | 1537 | 3244 |
| fp8_tp2dp4 | sglang | 26 | 464 | 1463 | 2804 |
| fp8_tp2dp4 | trtllm | 34 | 496 | 1515 | 3549 |
| int4_tp1dp8 | vllm | 37 | 571 | 2162 | 6568 |
| int4_tp1dp8 | sglang | 38 | 595 | 2206 | 6530 |
| int4_tp2dp4 | vllm | 57 | 762 | 2121 | 4139 |
| int4_tp2dp4 | sglang | 60 | 827 | 2202 | 3701 |

## Predictions

### 1. One user (generate)

| format/split | engine | tok/s | ceiling | % of ceiling | predicted | inside? |
|---|---|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 21.0 | 22.8 | 92% | 20-22 | yes |
| bf16_tp2dp4 | sglang | 21.3 | 22.8 | 93% | 20-22 | yes |
| bf16_tp2dp4 | trtllm | 21.2 | 22.8 | 93% | 20-22 | yes |
| fp8_tp1dp8 | vllm | 21.0 | 21.8 | 96% | 15-20 | no |
| fp8_tp1dp8 | sglang | 15.7 | 21.8 | 72% | 15-20 | yes |
| fp8_tp2dp4 | vllm | 36.0 | 43.5 | 83% | 30-38 | yes |
| fp8_tp2dp4 | sglang | 26.4 | 43.5 | 61% | 30-38 | no |
| fp8_tp2dp4 | trtllm | 34.3 | 43.5 | 79% | 30-38 | yes |
| int4_tp1dp8 | vllm | 37.2 | 38.7 | 96% | 28-36 | no |
| int4_tp1dp8 | sglang | 38.0 | 38.7 | 98% | 28-36 | no |
| int4_tp2dp4 | vllm | 57.2 | 77.4 | 74% | 45-60 | yes |
| int4_tp2dp4 | sglang | 60.1 | 77.4 | 78% | 45-60 | no |

Inside the predicted range: 7 of 12. Fastest: int4_tp2dp4/sglang.

### 2-4. Peak throughput (best of 1/16/64/256 users) and users at the peak

| format/split | engine | chat | rag | generate |
|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 1128 (256) | 214 (256) | 2678 (256) |
| bf16_tp2dp4 | sglang | 968 (256) | 185 (256) | 2098 (256) |
| bf16_tp2dp4 | trtllm | 1099 (256) | 209 (256) | 2664 (256) |
| fp8_tp1dp8 | vllm | 1591 (256) | 317 (256) | 3610 (256) |
| fp8_tp1dp8 | sglang | 1432 (256) | 314 (64) | 2042 (256) |
| fp8_tp2dp4 | vllm | 1246 (256) | 211 (256) | 3244 (256) |
| fp8_tp2dp4 | sglang | 1239 (256) | 227 (256) | 2804 (256) |
| fp8_tp2dp4 | trtllm | 1268 (256) | 233 (256) | 3549 (256) |
| int4_tp1dp8 | vllm | 2794 (256) | 501 (256) | 6568 (256) |
| int4_tp1dp8 | sglang | 2699 (256) | 474 (256) | 6530 (256) |
| int4_tp2dp4 | vllm | 1403 (256) | 229 (64) | 4139 (256) |
| int4_tp2dp4 | sglang | 1324 (256) | 227 (256) | 3701 (256) |

Best quantized combination per engine against BF16 tp2dp4, same engine:

| engine | workload | BF16 tp2dp4 | best FP8 | x | best INT4 | x |
|---|---|---|---|---|---|---|
| vllm | chat | 1128 | 1591 fp8_tp1dp8 | 1.41 | 2794 int4_tp1dp8 | 2.48 |
| vllm | rag | 214 | 317 fp8_tp1dp8 | 1.48 | 501 int4_tp1dp8 | 2.34 |
| vllm | generate | 2678 | 3610 fp8_tp1dp8 | 1.35 | 6568 int4_tp1dp8 | 2.45 |
| sglang | chat | 968 | 1432 fp8_tp1dp8 | 1.48 | 2699 int4_tp1dp8 | 2.79 |
| sglang | rag | 185 | 314 fp8_tp1dp8 | 1.70 | 474 int4_tp1dp8 | 2.56 |
| sglang | generate | 2098 | 2804 fp8_tp2dp4 | 1.34 | 6530 int4_tp1dp8 | 3.11 |
| trtllm | chat | 1099 | 1268 fp8_tp2dp4 | 1.15 | - | - |
| trtllm | rag | 209 | 233 fp8_tp2dp4 | 1.12 | - | - |
| trtllm | generate | 2664 | 3549 fp8_tp2dp4 | 1.33 | - | - |

At 256 users, best FP8 vs best INT4 (same engine):

| engine | workload | FP8 | INT4 | FP8 / INT4 |
|---|---|---|---|---|
| vllm | chat | 1591 | 2794 | 0.57 |
| vllm | rag | 317 | 501 | 0.63 |
| vllm | generate | 3610 | 6568 | 0.55 |
| sglang | chat | 1432 | 2699 | 0.53 |
| sglang | rag | 314 | 474 | 0.66 |
| sglang | generate | 2804 | 6530 | 0.43 |

tp1dp8 / tp2dp4 at 256 users:

| format | engine | chat | rag | generate |
|---|---|---|---|---|
| fp8 | vllm | 1.28 | 1.50 | 1.11 |
| fp8 | sglang | 1.16 | 1.38 | 0.73 |
| int4 | vllm | 1.99 | 2.24 | 1.59 |
| int4 | sglang | 2.04 | 2.09 | 1.76 |

### 5. Quality: full GSM8K (1,319 questions), repeat 1

| format/split | engine | accuracy | 95% interval | vs BF16 (points) | same answer as BF16 | fixed | broke | McNemar p |
|---|---|---|---|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 93.0% | 91.5-94.3 | - | - | - | - | - |
| bf16_tp2dp4 | sglang | 93.3% | 91.8-94.5 | - | - | - | - | - |
| bf16_tp2dp4 | trtllm | 93.1% | 91.6-94.3 | - | - | - | - | - |
| fp8_tp1dp8 | vllm | 93.5% | 92.0-94.7 | +0.5 | 95.0% | 22 | 16 | 0.418 |
| fp8_tp1dp8 | sglang | 92.2% | 90.6-93.5 | -1.1 | 93.4% | 22 | 36 | 0.087 |
| fp8_tp2dp4 | vllm | 93.5% | 92.0-94.7 | +0.5 | 95.1% | 22 | 16 | 0.418 |
| fp8_tp2dp4 | sglang | 93.8% | 92.3-95.0 | +0.5 | 94.7% | 26 | 19 | 0.371 |
| fp8_tp2dp4 | trtllm | 93.0% | 91.5-94.3 | -0.1 | 94.0% | 25 | 26 | 1.000 |
| int4_tp1dp8 | vllm | 91.4% | 89.7-92.8 | -1.7 | 91.3% | 30 | 52 | 0.020 |
| int4_tp1dp8 | sglang | 92.2% | 90.6-93.5 | -1.1 | 91.4% | 32 | 46 | 0.141 |
| int4_tp2dp4 | vllm | 90.8% | 89.1-92.3 | -2.2 | 90.4% | 30 | 59 | 0.003 |
| int4_tp2dp4 | sglang | 91.3% | 89.6-92.7 | -2.0 | 90.7% | 31 | 57 | 0.007 |

BF16 engines against each other (the noise floor for agreement): vllm-sglang 96.4%, vllm-trtllm 97.0%, sglang-trtllm 97.0%.

## KV-cache room per replica (tokens, repeat 1 start-up)

| format/split | engine | tokens | source |
|---|---|---|---|
| bf16_tp2dp4 | sglang | 50,739 | SGLang log |
| bf16_tp2dp4 | trtllm | 80,800 | TRT-LLM log |
| bf16_tp2dp4 | vllm | 78,848 | vLLM log |
| fp8_tp1dp8 | sglang | 20,400 | estimated from SGLang #token / token usage |
| fp8_tp1dp8 | vllm | 30,208 | vLLM log |
| fp8_tp2dp4 | sglang | 168,467 | SGLang log |
| fp8_tp2dp4 | trtllm | 186,304 | TRT-LLM log |
| fp8_tp2dp4 | vllm | 193,536 | vLLM log |
| int4_tp1dp8 | sglang | 76,900 | estimated from SGLang #token / token usage |
| int4_tp1dp8 | vllm | 89,600 | vLLM log |
| int4_tp2dp4 | sglang | 225,411 | SGLang log |
| int4_tp2dp4 | vllm | 252,928 | vLLM log |

## Cost per million output tokens at peak ($30.13/h for the instance)

| format/split | engine | chat | rag | generate |
|---|---|---|---|---|
| bf16_tp2dp4 | vllm | $7.42 | $39.14 | $3.13 |
| bf16_tp2dp4 | sglang | $8.65 | $45.24 | $3.99 |
| bf16_tp2dp4 | trtllm | $7.62 | $40.05 | $3.14 |
| fp8_tp1dp8 | vllm | $5.26 | $26.38 | $2.32 |
| fp8_tp1dp8 | sglang | $5.84 | $26.63 | $4.10 |
| fp8_tp2dp4 | vllm | $6.72 | $39.68 | $2.58 |
| fp8_tp2dp4 | sglang | $6.75 | $36.92 | $2.99 |
| fp8_tp2dp4 | trtllm | $6.60 | $35.88 | $2.36 |
| int4_tp1dp8 | vllm | $3.00 | $16.72 | $1.27 |
| int4_tp1dp8 | sglang | $3.10 | $17.65 | $1.28 |
| int4_tp2dp4 | vllm | $5.97 | $36.60 | $2.02 |
| int4_tp2dp4 | sglang | $6.32 | $36.93 | $2.26 |

## BF16 control: this run vs Step 2 (tp2dp4, loadgen)

Median difference 0.2%; 36 of 36 points within 5%. Largest: trtllm/rag/16 +3%, sglang/rag/16 -2%, vllm/rag/16 +2%, vllm/chat/256 +1%.

## Time to first token, p50 (s) at 256 users

| format/split | engine | chat | rag | generate |
|---|---|---|---|---|
| bf16_tp2dp4 | vllm | 4.1 | 111.3 | 1.6 |
| bf16_tp2dp4 | sglang | 38.2 | 145.7 | 8.6 |
| bf16_tp2dp4 | trtllm | 10.3 | 112.1 | 3.8 |
| fp8_tp1dp8 | vllm | 7.5 | 79.4 | 2.2 |
| fp8_tp1dp8 | sglang | 23.8 | 93.3 | 63.0 |
| fp8_tp2dp4 | vllm | 4.6 | 48.5 | 1.7 |
| fp8_tp2dp4 | sglang | 3.1 | 87.4 | 0.6 |
| fp8_tp2dp4 | trtllm | 10.0 | 46.7 | 3.6 |
| int4_tp1dp8 | vllm | 3.2 | 25.7 | 0.8 |
| int4_tp1dp8 | sglang | 1.2 | 44.3 | 0.6 |
| int4_tp2dp4 | vllm | 4.2 | 14.5 | 1.3 |
| int4_tp2dp4 | sglang | 2.7 | 70.9 | 0.5 |

## Repeatability (rep1 vs rep2 throughput)

Median spread 0.3%; 143 of 144 points within 5%. Largest: int4_tp1dp8/vllm/rag/16 13%, int4_tp1dp8/vllm/chat/16 4%, int4_tp1dp8/sglang/rag/64 4%, fp8_tp1dp8/vllm/rag/16 4%, bf16_tp2dp4/vllm/chat/16 3%.

## Requests and errors (both repeats)

Requests: 97488.

| point | bad_streams_dumped | no_text |
|---|---|---|
| bf16_tp2dp4/trtllm/generate/16 | 3 | 3 |
| bf16_tp2dp4/trtllm/generate/64 | 11 | 11 |
| bf16_tp2dp4/trtllm/generate/256 | 31 | 31 |
| bf16_tp2dp4/vllm/generate/16 | 3 | 3 |
| bf16_tp2dp4/vllm/generate/64 | 11 | 11 |
| bf16_tp2dp4/vllm/generate/256 | 31 | 31 |
| fp8_tp1dp8/vllm/generate/16 | 2 | 2 |
| fp8_tp1dp8/vllm/generate/64 | 11 | 11 |
| fp8_tp1dp8/vllm/generate/256 | 31 | 31 |
| fp8_tp2dp4/trtllm/generate/16 | 2 | 2 |
| fp8_tp2dp4/trtllm/generate/64 | 11 | 11 |
| fp8_tp2dp4/trtllm/generate/256 | 31 | 31 |
| fp8_tp2dp4/vllm/generate/16 | 3 | 3 |
| fp8_tp2dp4/vllm/generate/64 | 12 | 12 |
| fp8_tp2dp4/vllm/generate/256 | 31 | 31 |
| int4_tp1dp8/vllm/generate/16 | 1 | 1 |
| int4_tp1dp8/vllm/generate/256 | 2 | 2 |
| int4_tp2dp4/vllm/generate/16 | 1 | 1 |
| int4_tp2dp4/vllm/generate/256 | 2 | 2 |
