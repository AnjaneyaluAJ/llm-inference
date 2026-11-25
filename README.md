# LLM inference

Measuring what it takes to serve a language model fast and cheaply. The follow-up to
[llm-pretraining](https://github.com/AnjaneyaluAJ/llm-pretraining), which trained models from scratch.

Each step is one self-contained experiment with its methods, raw results and limitations published
here, and a prediction written down before the measurements.

| Step | Question | Hardware | Status |
|---|---|---|---|
| 1 | vLLM vs SGLang vs TensorRT-LLM: which serves fastest, and does it depend on the traffic? | 1x L40S (g6e.4xlarge) | [done](results/step1/README.md), with an [AIPerf cross-check](results/toolcmp/README.md) |
| 2 | A 32B model on 8 GPUs: tensor vs data parallel, three engines, two benchmark tools | 8x L40S (g6e.48xlarge) | [done](results/step2/README.md) |
| 3 | Quantization (BF16, FP8, INT8, INT4): quality vs speed vs memory | L40S / H100 | later |
| 4 | Cost per million tokens across GPUs and traffic levels | all of the above | later |

Plans with the predictions, written before each run: [step 1](docs/step1_engines_plan.md),
[step 2](docs/step2_parallel_plan.md).

## Layout

- `bench/`: load generator and analysis scripts
- `infra/`: instance launch and setup scripts
- `results/`: raw results, charts and write-ups, one folder per step
- `docs/`: plans and pre-registered predictions
