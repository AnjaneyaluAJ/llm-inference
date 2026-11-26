# Step 4: agent-style multi-turn traffic, prefix caching and KV-cache offload on 8x L40S

**Question:** agents send the same, growing conversation back to the model every turn. How much does
reusing its KV cache save, on GPUs with no fast links between them? Specifically:
- what the GPU prefix cache saves;
- whether the router has to send a session back to the same copy of the model for that to work;
- what offloading the KV cache to CPU memory adds once the conversations no longer fit on the GPUs.

## Fixed setup

- **Hardware:** the same AWS g6e.48xlarge as Steps 2-3: 8x L40S 48 GB, PCIe, no peer-to-peer,
  1,536 GiB of host memory.
- **Model:** Qwen3-32B in BF16, as four copies of 2 GPUs each (tp2dp4, the best Step 2 split under load).
  The context length is 32,768 tokens. A token of KV cache takes 256 KiB (64 layers x 8 KV heads x 128 x 2
  x 2 bytes), and each copy has room for about 50-80k tokens on its GPUs (Step 3: 78,848 on vLLM, 50,739
  on SGLang).
- **Engines:** vLLM v0.34.0 and SGLang v0.5.23, the same images as before. TensorRT-LLM is left out to keep
  the run under a day.
- **Replicas:** four separate servers, one per GPU pair (ports 8001-8004). Routing is done by the load
  generator, so it is identical for both engines and can be varied. This replaces the engines' built-in
  data parallelism, which does not route by session.

## Workload: simulated agent sessions

Each run has its own random seed, so nothing is reused across runs. Every session:
- **Turn 1:** a 3,000-token shared prefix (system prompt and tool definitions, identical for all sessions in
  a run), plus a 500-token task.
- **Turns 2-12:** the previous prompt, plus the model's previous output, plus a 1,500-token tool result.
  The final prompt is 22,893 tokens (measured with the tokenizer: random text re-tokenizes about 5% longer
  than the nominal lengths).
- **Output:** 150 tokens each turn, greedy, end-of-sequence ignored (as in Steps 1-3).
- **Between turns:** a pause for the "tool", exponential with a mean of 2 s.
- **Prompt text:** random vocabulary tokens, as in `loadgen`. The model's actual output text is appended,
  so the next prompt contains what the model produced, re-tokenized, as a real client would send it.

**Load:** 8, 32, 64 and 128 concurrent sessions, closed loop. When a session finishes its 12 turns, a new
one starts in its place.

**Steady state:** each slot's first session starts at a random turn from 1 to 12, with a synthetic history
of the right length. So the mix of context lengths is close to steady state from the start, instead of
every session starting short. These first turns are cache misses in every configuration. Each run lasts
10 minutes, and only turns that start after the first 3 minutes are measured. The mean context length of
the measured turns is reported for every run, so the mixes can be compared.

## Configurations (per engine)

| Name | Prefix cache | Routing | CPU offload |
|---|---|---|---|
| `nocache` | off | least-loaded | no |
| `gpu-ll` | on (default) | least-loaded: the replica with the fewest requests in flight | no |
| `gpu-sticky` | on | sticky: a session always goes to the same replica | no |
| `cpu-ll` | on | least-loaded | 150 GiB per replica (600 GiB in all) |
| `cpu-sticky` | on | sticky | 150 GiB per replica |

**Engine options:**
- **Prefix cache off:** vLLM `--no-enable-prefix-caching`, SGLang `--disable-radix-cache`.
- **CPU offload:** vLLM's native offloading connector and SGLang's hierarchical cache (HiCache), each
  sized to 150 GiB with the engine's own option. The exact flags are confirmed from each image's `--help`
  in the smoke test and recorded in each run's metadata.
- **Everything else:** default settings.

150 GiB holds about 600k tokens per replica, or 2.4M in all. At 128 sessions the working set is about
128 x 13k (the average context) = 1.7M tokens, so it fits in CPU memory but not on the GPUs.

**Scale:** 2 engines x 5 configurations x 4 loads x 2 repeats = 80 runs of 10 minutes. Each
(engine, configuration) server set is started once per repeat, the four loads run in a shuffled order, and
the engine and configuration order is shuffled per repeat.

## Metrics

- **Time to first token per turn:** p50, p90, p99. This is what an agent waits for before acting.
- **Turn latency:** first token plus 150 output tokens.
- **Throughput:** completed turns per minute and output tokens/s.
- **Goodput:** turns per minute whose first token arrived within 2 s.
- **Cache hits:** the engines' own prefix-cache hit counters (GPU and offload tiers where reported), scraped
  from `/metrics` at the start and end of the measured window.
- **Energy:** GPU joules per output token, from power logged every second (gross, including idle draw).

## Checks before the sweep (smoke test)

1. **Flags:** every configuration starts on both engines, and the offload flags are confirmed.
2. **Correctness under offload:** 10 prompts of 16k tokens, each sent three times:
   - cold;
   - after other traffic has pushed it out of GPU memory (so it comes back from the CPU tier);
   - immediately again, as a GPU hit.

   The first 64 greedy tokens of each are compared. Exact equality is not guaranteed (batching changes the
   arithmetic slightly), so this is reported as an observation. A large mismatch would stop the sweep.
3. **GSM8K:** 100 questions on each engine's `cpu-sticky` configuration, against Step 3's BF16 numbers.
4. **Timing:** one full (engine, configuration) at all four loads, to confirm the estimate below.

## Predictions (written before any of this runs)

Rough arithmetic from Step 2:
- One 2-GPU copy reads a prompt at about 2,000 tokens/s, and generates at about 48 ms per token for one
  user.
- A measured turn has about 13k tokens of context on average. Without a cache, that is about 6 s of
  prefill. With the history cached, only about 1.7k new tokens need reading, under 1 s.

1. **Low load, cache works:** at 8 sessions, `gpu-sticky` cuts the median time to first token by at least
   3x against `nocache`, on both engines.
2. **Routing matters as much as the cache:** at 8 and 32 sessions, `gpu-ll` has at least 2x the median time
   to first token of `gpu-sticky`. With least-loaded routing, a turn lands on the replica that holds the
   session's history only about 1 time in 4.
3. **The GPU cache runs out:** at 64 and 128 sessions, the active sessions' contexts far exceed GPU memory,
   so `gpu-sticky` serves no more than 1.5x the turns per minute of `nocache`.
4. **Offload recovers it:** at 64 and 128 sessions, `cpu-sticky` serves at least 2x the turns per minute of
   `gpu-sticky`, on both engines.
5. **Offload costs nothing at low load:** at 8 sessions, `cpu-sticky`'s median time to first token is within
   10% of `gpu-sticky`'s.
6. **Engines:** at every load, each engine's best configuration is within 25% of the other's on turns per
   minute.
7. **Energy:** at 128 sessions, `cpu-sticky` uses at most half the GPU energy per output token of `nocache`.

## Budget and how it runs

- **Duration:** smoke test about 2 h. The sweep is 80 x 10 min plus about 20 server starts, about 15-16 h.
  In all about 17-18 h at $30.13/h, roughly $520-550, confirmed or trimmed after the timing check.
- **Data:** pulled as one archive, with the checksum compared against the instance's copy.
- **Command:** `python bench/agent_compare.py --reps 2 --out /data/results/step4`.

## Limits known in advance

- Synthetic sessions: random-token content with fixed lengths and pause times. Real agent traces vary more,
  and research shows prefix reuse differs a lot by agent type.
- One model, BF16 only. Two engines.
- Offload to NVMe and KV-aware routers (llm-d, NVIDIA Dynamo, SGLang's router) are left for later.
