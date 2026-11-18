# Metron 4B

Metron (Greek **μέτρον**, “measure”) is a 4.02B-parameter text model for local inference. It starts from Qwen3-4B-Instruct-2507 and learns from 5,200 verified conversations with GPT-6.2 Astra. The published weights, checksums and Python module retain the original ModelDistl names.

## Run locally

Download the 1.89 GB Q3 build and open it with llama.cpp. The commands below require curl and an installed llama-cli.

```sh
curl -fLO https://github.com/shiv-eshwar/metron-4b/releases/download/astra-v0.2.0/ModelDistl-4B-Q3_K_S.gguf
llama-cli -m ModelDistl-4B-Q3_K_S.gguf -c 2048
```

For the higher-scoring 2.50 GB Q4 build, run these commands from a clone. The join command checks both parts and the assembled model against the release manifest.

```sh
mkdir -p artifacts/astra-v2/release
for part in 01 02; do
  file="ModelDistl-4B-Q4_K_M.gguf.part-$part"
  curl -fL "https://github.com/shiv-eshwar/metron-4b/releases/download/astra-v0.2.0/$file" -o "artifacts/astra-v2/release/$file" || exit 1
done
python3 scripts/model_file.py join --run astra-v2
llama-cli -m artifacts/astra-v2/export/ModelDistl-4B-Q4_K_M.gguf -c 2048
```

On a 16 GB M5 MacBook Air, Q4 generated 23 tokens/s and used 2.94 GB peak resident memory in a 2,048-token chat. Phone performance is unmeasured. [Release and checksums](https://github.com/shiv-eshwar/metron-4b/releases/tag/astra-v0.2.0)

## Evaluation

| Q4 model | First test / 256 | Fresh test / 128 | Fresh Python and math / 64 |
| --- | ---: | ---: | ---: |
| Metron 4B | 193 | 91 | 37 |
| Gemma 4 E4B IT | 193 | 85 | 46 |
| Gemma 3 4B IT | 184 | 80 | 41 |
| Stock Qwen 4B | 172 | 79 | 52 |

Metron gained most on grounded reading and following the requested science answer format. Both Gemma comparisons included a tie within their paired 95% intervals on both tests. Stock Qwen solved every fresh Python and math problem Metron solved, plus 15 more. [Model card](MODEL_CARD.md) · [Per-question results](runs/astra-v2/retention/test)

## Reproduce and contribute

The training protocol includes the pinned environment, data sources, verification rules, MLX training commands and GGUF export procedure. The verified conversations are available for offline retraining. [Training protocol](runs/astra-v2/PROTOCOL.md) · [Training data](runs/astra-v2/data) · [Contribution guide](CONTRIBUTING.md)

## License

Original project code is licensed under Apache 2.0. The model derives from Apache-licensed Qwen weights. Third-party datasets retain their respective licenses and attribution requirements. [Code license](LICENSE) · [Model notice](runs/astra-v2/NOTICE.txt) · [Dataset sources and licenses](runs/astra-v2/PROTOCOL.md)
