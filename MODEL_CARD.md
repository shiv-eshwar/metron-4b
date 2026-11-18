# ModelDistl 4B pilot

ModelDistl is an English-language, on-device instruction model derived from Qwen3-4B-Instruct-2507. The Q4_K_M build answered 14 of 16 sampled GSM8K problems, compared with 12 for the stock Q4_K_M model. The two models passed the same six local coding tasks and the same five of eight MBPP problems. On the local balance-scale question, the student reached the 320-token limit before giving its final answer. Stock finished in 307 tokens. [Item-by-item comparison](runs/pilot/comparison.json)

## Held-out task results

| Correct answers | Stock Q4_K_M | ModelDistl Q4_K_M | ModelDistl Q3_K_S |
| --- | ---: | ---: | ---: |
| Local Python functions | 6/6 | 6/6 | 3/6 |
| Local general assistance | 2/6 | 2/6 | 3/6 |
| Local math | 3/6 | 3/6 | 3/6 |
| Local reasoning | 5/6 | 4/6 | 2/6 |
| GSM8K math | 12/16 | 14/16 | 14/16 |
| MBPP Python functions | 5/8 | 5/8 | 4/8 |
| ARC-Challenge, requested label only | 8/16 | 8/16 | 2/16 |
| ARC-Challenge, correct option content | 14/16 | 14/16 | 15/16 |

The last two rows score the same 16 questions. The first checks whether the model followed the label-only instruction. The second also accepts the exact correct option text, with or without its matching label. The smaller build often added the option text. General assistance includes structured extraction and short writing tasks. Its prose checks use required concepts and word limits.

The 24 local prompts were held out of supervised training. An initial evaluation on them revealed a coding regression and informed the decision to calibrate adapter strength. The 40 public test items comprise fixed samples from GSM8K, ARC-Challenge, and sanitized MBPP. Calibration used separate validation items. MBPP exposes one assertion as an example and executes the complete test list in isolated CPython. All three deployment builds used identical prompts, greedy generation, a 320-token cap, and four llama.cpp slots with 2,048 tokens per slot. These small samples give limited evidence about performance on other workloads. [Raw responses and scores](runs/pilot/results) · [Public dataset revisions](modeldistl/benchmarks.py)

## Training and selection

- The source has 4,022,468,096 parameters. The 4-bit MLX training copy uses group size 64. A rank-8 LoRA adapter updates 1,310,720 parameters in the query and value projections of the last 16 layers. [Model source](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [Training config](configs/pilot.yaml)
- GPT-6.2 Astra generated 64 training and 12 validation demonstrations. Each split is balanced across coding, math, reasoning, and general assistance. All 76 passed the curriculum checks on the first attempt. All 19 Python demonstrations also passed the reference cases in CPython. The Responses API recorded 8,212 input and 2,938 output tokens, about $0.18 at the published standard rates. [Demonstrations and provenance](runs/pilot/data) · [Teacher pricing](https://developers.openai.com/api/docs/models/gpt-6.2-astra)
- One epoch comprised 64 microbatches and 32 AdamW updates at a learning rate of 0.00004. Prompt and padding tokens were excluded, giving 2,522 supervised tokens. Peak MLX allocation was 2.75 GB. Raw adapter validation loss fell from 2.901 to 0.637. At full strength the adapter scored 2/6 on local coding, following adjudication in CPython. [Training measurements](runs/pilot/training.json) · [Initial task scores](runs/pilot/initial)
- Calibration compared 12 strengths, including the zero-update control, on 12 local validation prompts and eight MBPP validation problems. The shipped multiplier is 1/128. It preserved 11/11 coding and 2/3 reasoning passes, raised math from 2/3 to 3/3, and kept general assistance at 2/3. Its measured validation loss was 2.809. The independently exported Q4 files differ in exactly the 32 trained projection tensors out of 398. [Calibration](runs/pilot/calibration.json) · [Deployed adapter check](runs/pilot/deployed-mlx.json) · [Tensor comparison](runs/pilot/tensor-verification.json)

## Deployment measurements

| Build and backend | GGUF size | Prompt tokens/s | Generated tokens/s |
| --- | ---: | ---: | ---: |
| Q4_K_M, Metal | 2.50 GB | 690.9 | 20.0 |
| Q4_K_M, CPU | 2.50 GB | 45.6 | 14.3 |
| Q3_K_S, Metal | 1.89 GB | 626.6 | 18.4 |

Measurements come from a 16 GB M5 MacBook Air with an eight-core GPU. llama-bench ran three repetitions with a 128-token prompt and 64-token generation benchmark. CPU used four threads. A separate Q4 chat process with a 2,048-token context reached 2.94 GB maximum resident memory. Phone speed and memory headroom depend on the device and runtime. Physical iOS, Android, Windows, and Linux devices were outside this hardware test. [Raw timing and memory measurements](runs/pilot/performance.json)

The release includes both standalone GGUF builds, the calibrated MLX adapter, checksums, and the original Apache 2.0 model license. The download helper joins the two Q4 release parts into one file. Q3 is available as a single download. [Run locally](README.md) · [Release](https://github.com/shiv-eshwar/modeldistl/releases/tag/pilot-v0.1.0)
