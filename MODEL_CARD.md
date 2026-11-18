# ModelDistl 4B Astra v2

The Q4 student answered 193 of 256 questions correctly. Gemma 4 E4B also answered 193 correctly and Gemma 3 4B answered 184. Gemma 4 used a 4.98 GB text model and the student used 2.50 GB. The student's nine-answer lead over Gemma 3 had a paired 95% bootstrap interval from −2.73 to +9.77 percentage points. Both models' strengths depended on the task. [Paired comparisons](runs/astra-v2/comparison.json) · [All responses](runs/astra-v2/results)

## Preselected head-to-head

Each domain has 64 questions. Python functions run against the MBPP reference assertions in isolated CPython. Math uses GSM8K exact final numbers. Reading uses normalized SQuAD 2.0 answer spans, including 16 unanswerable questions. Science uses the requested single ARC-Challenge option label. All models used their native text templates, greedy decoding, a 512-token output limit, a 2,048-token context per slot, four slots and optional thinking disabled.

| Model | GGUF GB | Python | Math | Reading | Science label | Total out of 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Astra student Q4 | 2.50 | 42 | 45 | 49 | 57 | **193** |
| Astra student Q3 | 1.89 | 39 | 30 | 46 | 55 | **170** |
| Stock Qwen Q4 | 2.50 | 52 | 55 | 47 | 18 | **172** |
| Previous pilot Q4 | 2.50 | 51 | 55 | 47 | 18 | **171** |
| Gemma 3 4B IT Q4 | 2.49 | 50 | 50 | 36 | 48 | **184** |
| Gemma 4 E4B IT Q4 | 4.98 | 45 | 55 | 37 | 56 | **193** |

## What changed

Against stock Qwen, the student gained 39 strict science answers and lost none. On Python it gained two cases and lost 12. Ten of those newly failing programs produced an incorrect result under the reference assertions, one had a syntax error and one raised a type error. On math it gained three and lost 13. All 13 newly missed math problems contained an incorrect final number. Reading gained five answers and lost three.

When every model could emit only an available science label, the student answered 57/64 correctly, stock Qwen 55/64, Gemma 3 51/64 and Gemma 4 56/64. Substituting those scores into the four-domain total gives the student 193, stock Qwen 209, Gemma 3 187 and Gemma 4 193. The original Q4 student therefore improved substantially at following the requested format, while stock Qwen retained stronger Python and math performance. A follow-up uses separate development and confirmation questions to test adapter strength. [Retention protocol](runs/astra-v2/retention/PROTOCOL.md)

| Model | Constrained science | Total with constrained science | Reading token F1 | Local tasks |
| --- | ---: | ---: | ---: | ---: |
| Astra student Q4 | 57/64 | 193/256 | 0.874 | 14/24 |
| Astra student Q3 | 55/64 | 170/256 | 0.802 | 14/24 |
| Stock Qwen Q4 | 55/64 | 209/256 | 0.838 | 16/24 |
| Previous pilot Q4 | 55/64 | 208/256 | 0.838 | 17/24 |
| Gemma 3 4B IT Q4 | 51/64 | 187/256 | 0.707 | 15/24 |
| Gemma 4 E4B IT Q4 | 56/64 | 193/256 | 0.677 | 17/24 |

The two totals share all questions and differ only in how the 64 science answers were generated. The local set contains six tasks per domain and informed the earlier pilot. The 256 primary questions were fixed before this training run. Public benchmarks may have appeared during pretraining of the original models. These measurements cover English text, a subset of Python, grade-school word problems, grounded reading and school science. [Protocol and source revisions](runs/astra-v2/PROTOCOL.md)

## Distillation

GPT-6.2 Astra supplied 5,139 new verified training targets and 166 separate verified validation targets. Training included 61 verified pilot examples, bringing the total to 5,200 chats. There were 308 multi-turn training chats. The 6,912 teacher requests used 1,619,791 input and 340,514 output tokens at an estimated cost of $26.58. Only the final, visible assistant answers were training targets. Tests and reference answers stayed local. Code was executed in a restricted, network-isolated CPython container. The numeric and logical checks grade final answers, and the constrained-writing rubrics provide partial checks on prose. [Data and accounting](runs/astra-v2/data) · [Curriculum and overlap checks](runs/astra-v2/curriculum)

MLX-LM trained 5,242,880 adapter parameters in the final 16 attention layers of Qwen3-4B-Instruct-2507. The underlying model has 4,022,468,096 parameters. One epoch covered all 256,629 answer tokens with 650 AdamW updates. Prompt, previous dialogue and padding tokens were masked. Peak MLX allocation was 5.08 GB. Rescoring the five saved weights on all 166 validation chats selected the last checkpoint at loss 0.5862, compared with 2.4304 for the unadapted model. The Q4 and Q3 exports each differ from the matching stock quantization in exactly 64 of 398 tensors, all belonging to trained projections. [Resolved configuration](runs/astra-v2/training-config.json) · [Training and checkpoint scores](runs/astra-v2/training.json) · [Tensor verification](runs/astra-v2/tensor-verification.json)

## Local deployment

| Build | File size | M5 prompt tokens/s | M5 generated tokens/s |
| --- | ---: | ---: | ---: |
| Q4_K_M, Metal | 2.50 GB | 717.1 | 23.0 |
| Q3_K_S, Metal | 1.89 GB | 660.7 | 22.8 |
| Q4_K_M, four CPU threads | 2.50 GB | 54.9 | 17.8 |

Measurements used a 16 GB M5 MacBook Air with an eight-core GPU and three llama-bench repetitions of 128 prompt and 64 generated tokens. A separate Q4 chat at 2,048 context tokens used 2.94 GB peak resident memory. Physical phones, Windows and Linux PCs were outside the hardware test. The private release supplies both GGUFs, a 19.46 MB compressed MLX adapter, the Qwen license and SHA-256 checksums. A second machine reproduced both complete model files from the published adapter and the original BF16 revision. [Local measurements](runs/astra-v2/performance.json) · [Independent rebuild](https://github.com/shiv-eshwar/modeldistl/actions/runs/44283437263) · [Download](https://github.com/shiv-eshwar/modeldistl/releases/tag/astra-v0.2.0)

The source weights are Qwen3-4B-Instruct-2507 under Apache 2.0. Gemma models served as text-only comparators. Their published GGUF revisions, data attribution and reproduction commands are recorded in the protocol. [Qwen revision](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [Earlier 64-example pilot](https://github.com/shiv-eshwar/modeldistl/blob/pilot-v0.1.0/MODEL_CARD.md)
