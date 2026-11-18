"""Fetch the immutable source and quantize it locally for MLX training."""

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE = Path("artifacts/base-bf16")
QUANT = Path("artifacts/base-mlx-4bit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    snapshot_download(REPO, revision=REVISION, local_dir=BASE, max_workers=3)
    if not args.fetch_only:
        subprocess.run(
            ["mlx_lm.convert", "--hf-path", str(BASE), "--mlx-path", str(QUANT),
             "--quantize", "--q-bits", "4", "--q-group-size", "64"],
            check=True,
        )


if __name__ == "__main__":
    main()
