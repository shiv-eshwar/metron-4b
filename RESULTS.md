# ModelDistl 4B scored 91/128, ahead of both Gemma models on our fresh test

| Q4 model | First test / 256 | Fresh test / 128 | Fresh Python and math / 64 |
| --- | ---: | ---: | ---: |
| ModelDistl 4B | **193** | **91** | 37 |
| Gemma 4 E4B IT | **193** | 85 | 46 |
| Gemma 3 4B IT | 184 | 80 | 41 |
| Stock Qwen 4B | 172 | 79 | **52** |

Each test gave equal weight to Python, math, reading and science. Every model answered the same questions within each test. The paired 95% intervals for the overall differences against both Gemma models included a tie on both tests. [Model card](MODEL_CARD.md) · [Fresh answers](runs/astra-v2/retention/test)

## X

Our 2.50 GB ModelDistl 4B scored 91/128 on a fresh test of Python, math, reading and science. Gemma 4 E4B scored 85 and Gemma 3 4B scored 80. On the first test we tied Gemma 4 at 193/256. Both Gemmas scored higher on fresh Python + math. The overall margins are uncertain.

## LinkedIn

We trained a 4B offline assistant on 5,200 verified conversations using GPT-6.2 Astra as the teacher. On 128 fresh questions, ModelDistl scored 91, ahead of Gemma 4 E4B at 85 and Gemma 3 4B at 80. On the earlier 256-question test, ModelDistl tied Gemma 4 at 193 and finished nine answers ahead of Gemma 3.

Where did the lead come from? On fresh reading questions, ModelDistl scored 24/32, versus 13 for Gemma 4 and 15 for Gemma 3. It also followed the single-label science instruction more consistently. Coding and math tell a different story. Stock Qwen solved 52/64, Gemma 4 solved 46, Gemma 3 solved 41 and our student solved 37.

The overall Gemma margins are within sampling uncertainty. The practical result is a 2.50 GB model that generated 23 tokens per second on our M5 MacBook Air.
