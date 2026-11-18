# Astra response distillation and Gemma comparison

The comparison uses 256 English text questions selected before the expanded training run. There are 64 problems each from the sanitized MBPP test split (programs executed against the supplied tests), GSM8K test (exact final number), ARC-Challenge test (requested label), and SQuAD 2.0 validation (normalized exact answer span, including 16 unanswerable questions). The local pilot's sampled public test rows are excluded. These tests cover Python, word problems, science multiple-choice and grounded reading. The local pilot tests give a separate view of general writing and logic.

All models use their native chat templates with llama.cpp b12499, Q4_K_M, text-only prompts, greedy decoding, a 512-token output limit, a 2,048-token context per slot and four concurrent slots. Optional thinking is disabled for the direct-answer comparison. Archive each generated answer, its finish reason and its verdict. Use paired differences across identical questions, with an exact two-sided sign test on discordant scores and a stratified paired bootstrap interval. A score applies to these models and questions.

| Comparator | Pinned revision | File size |
| --- | --- | ---: |
| Qwen3-4B-Instruct-2507 | cdbee75f17c01a7cc42f958dc650907174af0554 | 2.50 GB |
| Gemma 3 4B IT | d0976223747697cb51e056d85c532013931fe52e | 2.49 GB |
| Gemma 4 E4B IT | bfc15c382204943c3a8fff0c750b94ae2364d7a3 | 4.98 GB |

Teacher prompts use only training or validation rows. Verify Python with isolated CPython, numerical and label answers against independent references, reading against human-annotated spans, and constrained writing against explicit rubrics. Train from the original Qwen model with completion-only loss. Select the checkpoint with the lowest supervised validation loss. Evaluate the resulting Q4 student at its selected checkpoint with the same fixed protocol. The blind suite is never used for checkpoint or multiplier selection.

The GSM8K material comes from OpenAI under MIT. MBPP comes from Google Research and Code Contests from DeepMind under CC BY 4.0. ARC comes from AI2 and SQuAD 2.0 from Rajpurkar et al. under CC BY-SA 4.0. Source records are adapted into prompts and independent checks. SQuAD contexts originate in Wikipedia. The dataset manifest pins all source revisions. [GSM8K](https://huggingface.co/datasets/openai/gsm8k) · [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) · [Code Contests](https://huggingface.co/datasets/deepmind/code_contests) · [ARC](https://huggingface.co/datasets/allenai/ai2_arc) · [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)

## Supervision and training

The expanded curriculum also draws on Hendrycks et al.’s MATH competition problems, AI2’s QuaRTz qualitative reasoning problems, and the QuAC conversational reading dataset. The mathematics verifier parses the final boxed human reference and selects text-only problems with an unambiguous rational answer. QuAC preserves an actual previous question and answer. The model must resolve a follow-up using the passage. Article titles are separated across supervised and evaluation material, including titles that occur in SQuAD. Wikipedia retains attribution for the source articles. The QuAC card specifies MIT, MATH specifies MIT and QuaRTz specifies CC BY 4.0. [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) · [QuaRTz](https://huggingface.co/datasets/allenai/quartz) · [QuAC](https://huggingface.co/datasets/allenai/quac)

Additional authored tasks cover Python execution traces, function specifications, forward deduction, instructions for public communications, and revisions to earlier messages. Code Contests examples run as separate programs against published inputs and hidden reference outputs. For QuAC, a response can also qualify by quoting the passage and reaching 0.85 token F1 with a human reference. Source-grounded abstentions require the correct abstention. Prose checks enforce vocabulary and length and provide a limited quality signal for writing.

Astra receives a task and any dialogue history. References and hidden tests stay local. A second request is permitted for failed or rejected demonstrations. The collector records each attempt, reserves a conservative cost for in-flight calls, and enforces a cumulative $35 cap for this run. Only verified final answers become supervision. Multi-turn loss applies only to the final assistant answer. The student also rehearses a small, recorded subset of the pilot's verified training examples.

The MLX recipe trains rank-16 LoRA on query, key, value and output projections in the last 16 layers. MLX applies the scale of 2 directly. The frozen 4-bit base uses groups of 64. AdamW uses a peak learning rate of 0.000025, 4% warmup and cosine decay. An effective batch contains eight chats, with completion-only loss, gradient checkpointing and one complete training epoch. The selected checkpoint is rescored over the entire validation split. Portable CPU fusion adds the adapter to the original BF16 weights with fixed rank summation order, followed by one llama.cpp quantization for each deployment build.

## Reproduction

Use the repository's pinned Python environment, source weights and llama.cpp setup. Docker supplies isolated CPython. For dataset regeneration, install the benchmark extra. Generated answers and provenance from the completed run can be used directly for offline student training.

```sh
uv sync --locked --all-extras
source .venv/bin/activate
bash scripts/setup-llama.sh
python -m modeldistl.weights
docker pull python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89
python -m modeldistl.expanded_data
python -m modeldistl.diverse_data
```

To collect new teacher labels, supply the API key through the process environment and run `python -m modeldistl.distill collect --budget-usd 35`. The budget applies to that local request ledger. Use the checked-in training snapshot to reproduce the student without teacher calls.

```sh
python -m modeldistl.train --config configs/astra-v2.yaml
python -m modeldistl.export --adapter artifacts/adapters/astra-v2/selected --directory artifacts/astra-v2/export --name ModelDistl-4B-Astra-v2 --portable-fusion --quantization Q4_K_M Q3_K_S
python -m modeldistl.export --base-only
python -m modeldistl.comparison score --model base
python -m modeldistl.comparison score --model gemma3
python -m modeldistl.comparison score --model gemma4
python -m modeldistl.comparison score --model student
```

The pilot comparator reads its release from `artifacts/export/ModelDistl-4B-Q4_K_M.gguf`. Run `python -m modeldistl.comparison score --model pilot` once it is present. Run `python -m modeldistl.comparison summarize` to compute paired results for all five builds. The Gemma files come from the pinned third-party GGUF conversions and are checked against their published SHA-256 hashes. [Gemma 3 conversion](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/tree/d0976223747697cb51e056d85c532013931fe52e) · [Gemma 4 conversion](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/bfc15c382204943c3a8fff0c750b94ae2364d7a3)
