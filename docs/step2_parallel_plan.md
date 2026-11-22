# Step 2: a bigger model on 8 GPUs, tensor vs data parallel, three engines, two benchmark tools

**Question:** with 8 GPUs and a model that needs at least 2 of them, how should the GPUs be split
between tensor parallelism (TP: one copy of the model spread over several GPUs) and data parallelism
(DP: several independent copies)? Does the answer depend on the engine and the traffic, and do our
client and NVIDIA AIPerf still agree at this scale?

## Fixed setup

- **Model:** Qwen3-32B, BF16, about 65.5 GB of weights. It does not fit on one 48 GB GPU.
- **Hardware:** 8x NVIDIA L40S 48 GB on AWS g6e.48xlarge (192 vCPU, on-demand only). The GPUs talk
  over PCIe; there is no NVLink, which is what makes tensor parallelism expensive here.
- **Engines:** the same images as Step 1 (vLLM v0.34.0, SGLang v0.5.23, TensorRT-LLM 1.3.0). Default
  settings, with only the context length (8192) and the parallelism set.
- **Clients:** both `bench/loadgen.py` and AIPerf 0.14.0, run back to back on the same running server
  as in the [Step 1 cross-check](../results/toolcmp/README.md). Both run on the same host as the
  server.

## Splits (all use 8 GPUs)

| Split | Replicas | GPUs per replica | Weights per GPU | KV-cache room per replica (estimate) |
|---|---|---|---|---|
| tp8 | 1 | 8 | ~8 GB | ~1M tokens |
| tp4dp2 | 2 | 4 | ~16 GB | ~400k tokens |
| tp2dp4 | 4 | 2 | ~33 GB | ~60-80k tokens |

KV cache is about 256 KiB per token for this model (64 layers x 8 KV heads x 128 x 2 x 2 bytes; to be
confirmed from config.json).

**Data parallelism** uses each engine's own option where it exists: vLLM `--data-parallel-size` and
SGLang `--dp-size`. TensorRT-LLM has no data-parallel option for dense models, so it runs as separate
servers behind an nginx least-connections proxy. This difference is reported with the results.

## Workloads and points

- **Workloads:** chat (1k in / 256 out), RAG (4k / 128) and generate (128 / 1k), as in Step 1.
- **Users:** 1, 16, 64 and 256 concurrent users. The number of requests is max(4 x users, 10).
- **Repeats:** 2. The order of splits and engines is shuffled each repeat.
- **Correctness check:** in repeat 1 every server also answers 100 GSM8K questions, so a broken
  tensor-parallel setup cannot pass unnoticed.

**Before the sweep:**
- GPU memory bandwidth on each GPU.
- NCCL all-reduce time for 2, 4 and 8 GPUs (`bench/allreduce.py`).
- `nvidia-smi topo -m`.

## Predictions (written before any of this runs)

At one user, every token reads all the weights once, split across the replica's GPUs. With the Step 1
bandwidth of 746.7 GB/s per GPU, the ceilings are:

| Split | Ceiling | Prediction |
|---|---|---|
| tp2 | 2 x 746.7 / 65.5 = 22.8 tok/s | 19-22 tok/s |
| tp4 | 45.6 tok/s | 33-42 tok/s |
| tp8 | 91.2 tok/s | 45-70 tok/s |

1. **One user:** tp8 is fastest, but it gets the lowest fraction of its ceiling. Every token needs
   128 small all-reduces over PCIe, a fixed cost that grows in relative terms as the per-GPU work
   shrinks.
2. **Many users:** tp2dp4 gives the highest peak throughput on chat and generate, at least 20% above
   tp8, because its replicas never talk to each other.
3. **RAG at 256 users:** the exception. tp2dp4 runs out of KV-cache room (4k-token prompts, about
   60-80k tokens per replica), so tp4dp2 or tp8 matches or beats it.
4. **Engines:** at a given split they stay within about 10% of each other, as on one GPU.
5. **Tools:** AIPerf and loadgen throughput agree within 3% at every point.

## Budget and how it runs

- **Duration:** estimated 12-18 h of g6e.48xlarge at about $30/h. A smoke test comes first (one
  engine at each split, about 1.5 h). It measures how long each point takes, and the full plan is
  confirmed or trimmed before the sweep starts.
- **Data:** results are pulled to the laptop and checksum-verified after the run.
- **Commands:** `infra/host_setup_step2.sh`, then
  `python bench/multi_compare.py --reps 2 --out /data/results/step2`.
