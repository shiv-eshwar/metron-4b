# Retaining Python and math performance

The original, locked 256-question comparison found that full-strength distillation improved label following but reduced Python and math scores. This follow-up retains the original result and uses different questions to choose an operating point. It uses the same trained adapter with no additional teacher requests.

Before generating any candidate scores, fix five strengths at 0, 1/16, 1/4, 1/2 and 1. Zero is stock Qwen and one is the full Astra student. Each positive intermediate checkpoint scales only the second LoRA matrix, fuses into the original BF16 weights and is quantized to Q4_K_M. Compare the exact deployment files using the same native template, greedy generation, 512 output tokens, 2,048 context tokens per slot, and four llama.cpp slots.

Select using 96 development questions, 24 per original domain. An eligible model must match or exceed the stock result on Python and math combined and lose at most one item in either domain individually. Among eligible models maximize the total strict score, then Python and math combined, then the score with science labels constrained, then favor the smallest change. The zero control is eligible. Use a second, untouched set of 128 questions, 32 per domain, to measure the single selected result, the full student, stock Qwen and both Gemma models. The primary retention comparison uses executable Python plus exact GSM8K answers. Report the four-domain total under both strict and constrained science scoring. No score on the new confirmation set can choose a multiplier.

The new sets exclude prior curriculum questions, earlier benchmark items and matching MBPP callable names. Reading contexts are unique across the two sets and outside earlier supervision and evaluations. One quarter of each reading subset is unanswerable. The new confirmation set uses official held-out splits and has no shared question with the initial 256. Public benchmark exposure during model pretraining is unknown.

## Reproduction

After training or restoring the selected full-strength adapter and exporting the stock and full Q4 models, run the commands below in the project's pinned environment. The evaluator uses the committed question files and writes resumable per-model results. The summary command reads the completed reports offline, verifies their question fingerprints and computes paired differences for both the selected strength and the published full student. It requires neither the weights nor a teacher API key. [Original selection rule and frozen questions](https://github.com/shiv-eshwar/modeldistl/tree/e5a86e0/runs/astra-v2/retention) · [Model setup](../PROTOCOL.md)

```sh
python -m modeldistl.retention run
python -m modeldistl.retention summarize
```
