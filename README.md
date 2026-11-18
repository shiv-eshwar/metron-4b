# ModelDistl

ModelDistl 4B Astra v2 is a 4,022,468,096-parameter offline assistant derived from Qwen3-4B-Instruct-2507. It was fine-tuned on 5,200 conversations with verified GPT-6.2 Astra target answers spanning Python, math, general assistance and reasoning. Another 166 conversations selected the checkpoint. The study compares the exported weights with Gemma 3 4B IT, Gemma 4 E4B IT, stock Qwen and the previous pilot on the same questions. [Model card and scores](MODEL_CARD.md) · [Evaluation protocol and reproduction](runs/astra-v2/PROTOCOL.md)

The Q4 student scored 193/256, tying Gemma 4 E4B and finishing nine answers ahead of Gemma 3 4B. Its strongest gain over stock Qwen was science-label instruction following. Stock Qwen solved more Python and math tasks. On an M5 MacBook Air, Q4 generated 23.0 tokens/s.

## Get the trained model

| Build | Download | File size |
| --- | --- | ---: |
| Q4_K_M | From an authenticated checkout, run `python3 scripts/model_file.py download --run astra-v2`. It verifies and joins the two parts into one GGUF. | 2.50 GB |
| Q3_K_S | Download the single GGUF while signed in, then import it into a compatible runtime. [Mobile GGUF](https://github.com/shiv-eshwar/modeldistl/releases/download/astra-v0.2.0/ModelDistl-4B-Q3_K_S.gguf) | 1.89 GB |
| MLX adapter | Unzip the adapter beside the pinned 4-bit MLX conversion and supply the directory as the adapter path. [MLX adapter](https://github.com/shiv-eshwar/modeldistl/releases/download/astra-v0.2.0/ModelDistl-4B-Astra-v2-MLX.zip) | 19.46 MB compressed |

The release includes checksums and the Qwen license. The CPU release workflow rebuilds both model files from the original BF16 weights and the trained adapter. [GitHub release](https://github.com/shiv-eshwar/modeldistl/releases/tag/astra-v0.2.0) · [Rebuild workflow](.github/workflows/astra-release.yml)

## Run Astra v2

Run this from the repository root on the training Mac.

```sh
.tools/llama-bin/llama-cli -m artifacts/astra-v2/export/ModelDistl-4B-Q4_K_M.gguf -c 2048
```

A fresh Mac checkout can install the pinned runner with `bash scripts/setup-llama.sh`. On Windows or Linux, point a compatible llama.cpp executable at the downloaded GGUF. For iOS or Android, transfer the single Q3 file or the reassembled Q4 file to a GGUF-capable app. A 2,048-token context was used for the laptop tests. [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) · [PocketPal](https://github.com/a-ghorbani/pocketpal-ai)

## Reproduce Astra v2

The repository includes training conversations, validation data, source revisions, run metrics, and the raw answers from each model. The supervised snapshot supports offline retraining. The protocol also gives the commands to regenerate teacher targets with an API credential, train with MLX-LM, export on CPU and run the comparative evaluations. [Run protocol](runs/astra-v2/PROTOCOL.md) · [Verified conversations](runs/astra-v2/data) · [Resolved training configuration](runs/astra-v2/training-config.json) · [Per-item responses](runs/astra-v2/results)

## Earlier pilot

The original release trained on 64 demonstrations and evaluated 24 local and 40 public tasks. Its weights and results are archived. [Pilot release](https://github.com/shiv-eshwar/modeldistl/releases/tag/pilot-v0.1.0) · [Pilot model card](https://github.com/shiv-eshwar/modeldistl/blob/pilot-v0.1.0/MODEL_CARD.md)

## Reproduce the pilot on Apple silicon

The pinned environment uses Python 3.12, uv, MLX-LM, and llama.cpp. The training host has 16 GB of unified memory. Allow about 40 GB of working disk space for the source weights, temporary BF16 conversion, and quantized models.

```sh
uv sync --locked --all-extras
source .venv/bin/activate
bash scripts/setup-llama.sh
python -m modeldistl.weights
python -m modeldistl.train --data runs/pilot/data
docker pull python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89
python -m modeldistl.calibrate
python -m modeldistl.export --quantization Q4_K_M Q3_K_S
```

The training command masks user tokens and padding, rejects truncated conversations, fixes both random seeds, and saves an adapter every 32 microbatches. It scores the saved checkpoints on all 12 validation demonstrations and puts the lowest-loss adapter in `artifacts/adapters/pilot/selected`.

Calibration scores an unadapted control and five adapter strengths on the 12 validation prompts and eight MBPP validation problems. If that grid misses the retention criterion, it tests six additional strengths. A candidate qualifies by matching or exceeding base coding accuracy, reasoning accuracy, and the average of the four domain accuracies. The highest average wins, with ties going to the smallest update. This run selected 1/128 strength. The exporter uses `artifacts/adapters/pilot/deploy`. Docker supplies isolated CPython execution for these checks. For another training run, use `--output` with the trainer, `--source` and `--out` with calibration, and `--adapter` with the exporter.

Training uses a 4-bit, group-size-64 MLX base. The rank-8 adapter updates the query and value projections in the final 16 of 36 layers. Export applies the learned delta to the pinned original BF16 weights, preserves the original tokenizer and chat template, and quantizes each deployment build directly from the merged BF16 weights.

## Run the pilot

```sh
.tools/llama-bin/llama-cli -m artifacts/export/ModelDistl-4B-Q4_K_M.gguf -c 2048
```

Windows and Linux can use the same GGUF with a current llama.cpp build. On iOS or Android, transfer the single GGUF to the device and import it into a Qwen3-compatible app such as PocketPal. Start with a 2,048-token context. At that context the uncompressed FP16 key/value cache is about 288 MiB, in addition to the model and runtime buffers. Actual headroom depends on the device, the app’s memory limit, and the context setting. [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) · [PocketPal](https://github.com/a-ghorbani/pocketpal-ai)

## Generate a pilot-format supervised dataset

Provide the project credential through the process environment. This shell example accepts it with terminal echo disabled. The checked-in demonstration snapshot can be used directly for the pilot.

```zsh
read -rs 'OPENAI_API_KEY?OpenAI API key: ' && export OPENAI_API_KEY && print
python -m modeldistl.teacher generate --max-new 76
unset OPENAI_API_KEY
python -m modeldistl.teacher prepare
python -m modeldistl.train --data artifacts/data --output artifacts/adapters/new-run
```

The generator sends only the prompt and teacher instructions to the Responses API. It caps output at 1,024 tokens per call, uses standard service and low reasoning effort, and records usage and response IDs. Re-running generation skips completed verified requests. A failed, incomplete, or interrupted request requires `--retry-failed` and counts against the next invocation’s `--max-new` limit. Regrading cached answers and preparing MLX data are offline operations. [OpenAI model and pricing](https://developers.openai.com/api/docs/models/gpt-6.2-astra)

Prompts and independent checks live in `curriculum/{train,valid,test}.jsonl`. Keep related variants in the same family and split. The loader checks normalized prompts, IDs, and family overlap. The teacher receives just the training and validation prompts. Coding examples run in Monty with isolated interpreter sessions, time and memory limits, and no host callbacks for filesystem or network access. Numerical answers use exact fractions. Label and JSON checks compare values. Short prose checks look for required content and enforce word limits. The prose checks are heuristic.

## Evaluate the pilot suites

```sh
python -m modeldistl.evaluate --out artifacts/eval/mlx-base.json
python -m modeldistl.evaluate --adapter artifacts/adapters/pilot/deploy --out artifacts/eval/mlx-student.json
python -m modeldistl.export --base-only
python -m modeldistl.evaluate --gguf artifacts/export/Qwen3-4B-base-Q4_K_M.gguf --jobs 4 --code-runtime cpython --out artifacts/eval/gguf-base.json
python -m modeldistl.evaluate --gguf artifacts/export/ModelDistl-4B-Q4_K_M.gguf --jobs 4 --code-runtime cpython --out artifacts/eval/gguf-student.json
python -m unittest discover -s tests -v
```

The local evaluation has six tasks per domain. For a second comparison, add `--suite public` and use a new output filename. That suite selects 16 GSM8K and 16 ARC-Challenge test rows at fixed dataset revisions and seed 20261118. It uses zero-shot prompts and a 320-token generation cap. These are sampled scores under this project’s prompting and formatting rules. Each report includes per-item answers, verdicts, and generation lengths. The GGUF evaluator starts and stops its own loopback server. [GSM8K](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · [ARC](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)

For independent Python tests use `--suite mbpp`. This samples eight sanitized MBPP test problems at a pinned revision. Each prompt includes one worked assertion, and scoring uses the complete assertion set in network-isolated CPython with resource limits. [MBPP, Google Research, CC BY 4.0](https://huggingface.co/datasets/google-research-datasets/mbpp/tree/4bb6404fdc6cacfda99d4ac4205087b89d32030c)

## Sources

The weights derive from Qwen3-4B-Instruct-2507 under Apache 2.0. ModelDistl adds a task-specific adapter and changes the weight representation for deployment. The original model license accompanies the export. [Pinned Qwen source](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [License](licenses/Qwen-Apache-2.0.txt) · [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/v0.32.2) · [Pinned llama.cpp](https://github.com/ggml-org/llama.cpp/tree/401563f10ec60754e688cd3165c1343cf4be9642)
