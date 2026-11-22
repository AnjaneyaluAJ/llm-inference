# loadgen vs AIPerf (median of repeats, default engine settings)

Ratio = AIPerf / loadgen. 'adj' = AIPerf recomputed without re-used warm-up prompts and with server token counts.

| engine | workload | users | tok/s loadgen | tok/s AIPerf | ratio | ratio adj | TTFT p50 lg / AIPerf / adj (ms) | TPOT p50 lg / AIPerf / adj (ms) | rep spread lg / AIPerf (%) | warm-up prompts re-used |
|---|---|---|---|---|---|---|---|---|---|---|
| sglang | chat | 1 | 45.3 | 45.2 | 0.997 | 0.998 | 87 / 85 / 86 | 21.78 / 21.85 / 21.85 | 0.0 / 0.1 | [1, 1, 1] |
| sglang | chat | 16 | 500.3 | 507.2 | 1.014 | 1.015 | 715 / 636 / 699 | 29.31 / 28.36 / 28.25 | 1.6 / 1.2 | [8, 8, 8] |
| sglang | chat | 64 | 1043.2 | 1047.5 | 1.004 | 1.005 | 2631 / 2675 / 2530 | 51.09 / 51.30 / 51.28 | 2.0 / 1.1 | [8, 8, 8] |
| sglang | generate | 1 | 46.1 | 46.0 | 0.996 | 0.997 | 35 / 34 / 34 | 21.66 / 21.73 / 21.73 | 0.0 / 0.1 | [1, 1, 1] |
| sglang | generate | 16 | 620.3 | 617.2 | 0.995 | 0.995 | 187 / 154 / 163 | 25.63 / 25.78 / 25.78 | 0.1 / 0.0 | [8, 8, 8] |
| sglang | generate | 64 | 1793.0 | 1778.5 | 0.992 | 0.992 | 529 / 491 / 491 | 35.22 / 35.50 / 35.48 | 0.2 / 0.2 | [8, 8, 8] |
| sglang | rag | 1 | 40.1 | 40.5 | 1.010 | 1.010 | 340 / 328 / 329 | 22.42 / 22.49 / 22.49 | 0.3 / 0.5 | [1, 1, 1] |
| sglang | rag | 16 | 203.1 | 204.8 | 1.008 | 1.009 | 2853 / 2799 / 2595 | 56.86 / 56.73 / 56.61 | 1.7 / 1.9 | [8, 8, 8] |
| sglang | rag | 64 | 237.1 | 236.7 | 0.998 | 0.999 | 22028 / 22044 / 22128 | 105.81 / 106.24 / 106.24 | 0.4 / 0.7 | [8, 8, 8] |
| trtllm | chat | 1 | 44.9 | 44.9 | 0.998 | 0.998 | 88 / 86 / 86 | 21.99 / 22.05 / 22.05 | 0.1 / 0.0 | [1, 1, 1] |
| trtllm | chat | 16 | 483.1 | 493.4 | 1.021 | 1.022 | 920 / 639 / 643 | 29.48 / 28.57 / 28.55 | 0.9 / 0.5 | [8, 8, 8] |
| trtllm | chat | 64 | 1011.1 | 1010.5 | 0.999 | 1.000 | 1689 / 1692 / 1693 | 56.72 / 56.72 / 56.71 | 2.0 / 2.3 | [8, 8, 8] |
| trtllm | generate | 1 | 45.6 | 45.5 | 0.997 | 0.997 | 34 / 35 / 35 | 21.91 / 21.97 / 21.96 | 0.0 / 0.1 | [1, 1, 1] |
| trtllm | generate | 16 | 609.0 | 606.6 | 0.996 | 0.997 | 181 / 151 / 153 | 26.12 / 26.22 / 26.22 | 0.1 / 0.2 | [8, 8, 8] |
| trtllm | generate | 64 | 1749.0 | 1739.2 | 0.994 | 0.995 | 575 / 484 / 482 | 36.06 / 36.36 / 36.35 | 0.1 / 0.1 | [8, 8, 8] |
| trtllm | rag | 1 | 39.8 | 40.1 | 1.006 | 1.005 | 335 / 341 / 342 | 22.67 / 22.73 / 22.72 | 0.3 / 0.5 | [1, 1, 1] |
| trtllm | rag | 16 | 200.5 | 198.9 | 0.992 | 0.993 | 1045 / 1059 / 1060 | 71.49 / 72.00 / 72.01 | 1.7 / 2.6 | [8, 8, 8] |
| trtllm | rag | 64 | 245.3 | 244.2 | 0.996 | 0.996 | 13772 / 13552 / 12102 | 157.89 / 159.40 / 159.40 | 0.9 / 0.5 | [8, 8, 8] |
| vllm | chat | 1 | 44.8 | 44.6 | 0.995 | 0.995 | 86 / 85 / 85 | 22.08 / 22.21 / 22.21 | 0.0 / 0.0 | [1, 1, 1] |
| vllm | chat | 16 | 509.0 | 512.6 | 1.007 | 1.007 | 479 / 464 / 480 | 29.44 / 29.61 / 29.61 | 1.6 / 0.5 | [8, 8, 8] |
| vllm | chat | 64 | 1096.7 | 1098.7 | 1.002 | 1.002 | 685 / 674 / 674 | 55.46 / 55.48 / 55.48 | 1.4 / 0.7 | [8, 8, 8] |
| vllm | generate | 1 | 45.5 | 45.2 | 0.994 | 0.994 | 34 / 33 / 33 | 21.98 / 22.10 / 22.10 | 0.0 / 0.1 | [1, 1, 1] |
| vllm | generate | 16 | 624.2 | 620.3 | 0.994 | 0.994 | 189 / 164 / 169 | 25.47 / 25.64 / 25.64 | 0.0 / 0.1 | [8, 8, 8] |
| vllm | generate | 64 | 1868.9 | 1852.7 | 0.991 | 0.992 | 406 / 351 / 350 | 33.91 / 34.18 / 34.18 | 0.3 / 0.4 | [8, 8, 8] |
| vllm | rag | 1 | 39.8 | 40.0 | 1.004 | 1.005 | 335 / 330 / 330 | 22.68 / 22.80 / 22.80 | 0.1 / 0.4 | [1, 1, 1] |
| vllm | rag | 16 | 213.1 | 212.4 | 0.997 | 0.997 | 899 / 897 / 897 | 67.25 / 67.53 / 67.53 | 1.6 / 0.9 | [8, 8, 8] |
| vllm | rag | 64 | 242.7 | 239.5 | 0.987 | 0.987 | 10315 / 10763 / 10777 | 174.04 / 176.98 / 176.96 | 0.9 / 0.5 | [8, 8, 8] |

AIPerf client token count vs server count: median -0.043%, range -0.312% to +0.156%
Requests where AIPerf's count differs from the server's: 1231 of 8910
Client turnaround (next request start - previous end), median of points: loadgen 0.01 ms, AIPerf 3.04 ms
Errors: loadgen 1, AIPerf zero-text 0

## Stress point: generate, 256 users, 1024 requests

```
{
 "sglang": {
  "loadgen": {
   "requests": 1024,
   "errors": 0,
   "error_sample": [],
   "bad_streams_saved": false
  },
  "aiperf": {
   "requests": 1024,
   "ok": 1024,
   "errors": [],
   "zero_text_records": 0
  }
 },
 "trtllm": {
  "loadgen": {
   "requests": 1024,
   "errors": 0,
   "error_sample": [],
   "bad_streams_saved": false
  },
  "aiperf": {
   "requests": 1024,
   "ok": 1024,
   "errors": [],
   "zero_text_records": 0
  }
 },
 "vllm": {
  "loadgen": {
   "requests": 1024,
   "errors": 0,
   "error_sample": [],
   "bad_streams_saved": false
  },
  "aiperf": {
   "requests": 1024,
   "ok": 1024,
   "errors": [],
   "zero_text_records": 0
  }
 }
}
```
