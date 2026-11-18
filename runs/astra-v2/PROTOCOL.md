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
