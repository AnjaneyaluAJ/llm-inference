# Step 1 summary (median of repeats)

Measured-bandwidth ceiling at concurrency 1: 49.3 tokens/s

## Single user, generate workload (decode speed = 1000 / median TPOT)

| config/engine | tokens/s | % of measured ceiling |
|---|---|---|
| defaults/sglang | 46.17 | 93.6 |
| defaults/trtllm | 45.64 | 92.5 |
| defaults/vllm | 45.48 | 92.2 |
| matched/sglang | 46.19 | 93.6 |
| matched/trtllm | 45.64 | 92.5 |
| matched/vllm | 45.5 | 92.2 |

## Peak and SLO-constrained throughput (output tokens/s)

| config | engine | workload | peak | within SLO | at concurrency |
|---|---|---|---|---|---|
| defaults | sglang | chat | 1237 | 495 | 16 |
| defaults | sglang | generate | 2722 | 1788 | 64 |
| defaults | sglang | rag | 228 | 106 | 4 |
| defaults | trtllm | chat | 1218 | 480 | 16 |
| defaults | trtllm | generate | 2653 | 1747 | 64 |
| defaults | trtllm | rag | 242 | 151 | 8 |
| defaults | vllm | chat | 1257 | 506 | 16 |
| defaults | vllm | generate | 3185 | 1866 | 64 |
| defaults | vllm | rag | 242 | 162 | 8 |
| matched | sglang | chat | 1242 | 496 | 16 |
| matched | sglang | generate | 2736 | 1792 | 64 |
| matched | sglang | rag | 226 | 155 | 8 |
| matched | trtllm | chat | 1188 | 485 | 16 |
| matched | trtllm | generate | 2592 | 1745 | 64 |
| matched | trtllm | rag | 230 | 151 | 8 |
| matched | vllm | chat | 1197 | 507 | 16 |
| matched | vllm | generate | 2983 | 1870 | 64 |
| matched | vllm | rag | 234 | 161 | 8 |

Repeat spread (max-min)/median of throughput: median 0.7%, max 8.0%

Points with request errors: 15
