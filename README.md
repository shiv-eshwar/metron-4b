# ModelDistl

ModelDistl fine-tunes Qwen3-4B-Instruct-2507 on verified demonstrations from OpenAI’s GPT-6.2 Astra. The 4,022,468,096-parameter student runs locally. The teacher is needed only when generating new demonstrations.

## Download the trained model

| Build | File size | How to get it |
| --- | ---: | --- |
| Q4_K_M | 2.50 GB | Run `python3 scripts/model_file.py download` from this checkout with GitHub CLI signed in. The script downloads two release parts, verifies their SHA-256 hashes, and assembles the single GGUF in `artifacts/export/`. |
| Q3_K_S | 1.89 GB | Download the single GGUF from the private release while signed in. [Download Q3_K_S](https://github.com/shiv-eshwar/modeldistl/releases/download/pilot-v0.1.0/ModelDistl-4B-Q3_K_S.gguf) |
| MLX adapter | 4.86 MB compressed | Unzip alongside a local 4-bit MLX conversion of the pinned base. [Download adapter](https://github.com/shiv-eshwar/modeldistl/releases/download/pilot-v0.1.0/ModelDistl-4B-MLX-adapter.zip) |

[Release and checksums](https://github.com/shiv-eshwar/modeldistl/releases/tag/pilot-v0.1.0)

## Reproduce the pilot on Apple silicon

The pinned environment uses Python 3.12, uv, MLX-LM, and llama.cpp. The training host has 16 GB of unified memory. Allow about 40 GB of working disk space for the source weights, temporary BF16 conversion, and quantized models.

```sh
uv sync --locked --all-extras
source .venv/bin/activate
bash scripts/setup-llama.sh
python -m modeldistl.weights
python -m modeldistl.train --data runs/pilot/data
python -m modeldistl.export --quantization Q4_K_M Q3_K_S
```

The training command masks user tokens, rejects truncated conversations, fixes both random seeds, and saves an adapter every 32 microbatches. It scores the saved checkpoints on all 12 validation demonstrations and puts the lowest-loss adapter in `artifacts/adapters/pilot/selected`. Use `--output` for an additional training run and pass that run’s selected adapter to the exporter with `--adapter`.

Training uses a 4-bit, group-size-64 MLX base. The rank-8 adapter updates the query and value projections in the final 16 of 36 layers. Export applies the learned delta to the pinned original BF16 weights, preserves the original tokenizer and chat template, and quantizes each deployment build directly from the merged BF16 weights.

## Run the model

```sh
.tools/llama-bin/llama-cli -m artifacts/export/ModelDistl-4B-Q4_K_M.gguf -c 2048
```

Windows and Linux can use the same GGUF with a current llama.cpp build. On iOS or Android, transfer the single GGUF to the device and import it into a Qwen3-compatible app such as PocketPal. Start with a 2,048-token context. At that context the uncompressed FP16 key/value cache is about 288 MiB, in addition to the model and runtime buffers. Actual headroom depends on the device, the app’s memory limit, and the context setting. [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) · [PocketPal](https://github.com/a-ghorbani/pocketpal-ai)

## Generate a new supervised dataset

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

## Evaluate

```sh
python -m modeldistl.evaluate --out artifacts/eval/mlx-base.json
python -m modeldistl.evaluate --adapter artifacts/adapters/pilot/selected --out artifacts/eval/mlx-student.json
python -m modeldistl.export --base-only
python -m modeldistl.evaluate --gguf artifacts/export/Qwen3-4B-base-Q4_K_M.gguf --out artifacts/eval/gguf-base.json
python -m modeldistl.evaluate --gguf artifacts/export/ModelDistl-4B-Q4_K_M.gguf --out artifacts/eval/gguf-student.json
python -m unittest discover -s tests -v
```

The local evaluation has six tasks per domain. For a second comparison, add `--suite public` and use a new output filename. That suite selects 16 GSM8K and 16 ARC-Challenge test rows at fixed dataset revisions and seed 20261118. It uses zero-shot prompts and a 320-token generation cap. These are sampled scores under this project’s prompting and formatting rules. Each report includes per-item answers, verdicts, and generation lengths. The GGUF evaluator starts and stops its own loopback server. [GSM8K](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · [ARC](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)

## Sources

The weights derive from Qwen3-4B-Instruct-2507 under Apache 2.0. ModelDistl adds a task-specific adapter and changes the weight representation for deployment. The original model license accompanies the export. [Pinned Qwen source](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [License](licenses/Qwen-Apache-2.0.txt) · [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/v0.32.2) · [Pinned llama.cpp](https://github.com/ggml-org/llama.cpp/tree/401563f10ec60754e688cd3165c1343cf4be9642)
