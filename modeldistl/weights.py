"""Fetch the immutable source and quantize it locally for MLX training."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .files import write_json

REPO = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE = Path("artifacts/base-bf16")
QUANT = Path("artifacts/base-mlx-4bit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    BASE.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(BASE.parent / ".weights.lock"):
        snapshot_download(REPO, revision=REVISION, local_dir=BASE, max_workers=3)
        if args.fetch_only:
            return
        expected = {"repo": REPO, "revision": REVISION, "bits": 4, "group_size": 64}
        if QUANT.exists():
            mark = QUANT / "source.json"
            if mark.exists() and json.loads(mark.read_text()) == expected:
                print(f"Using {QUANT}")
                return
            raise SystemExit(f"{QUANT} already exists with unverified provenance.")
        with tempfile.TemporaryDirectory(dir=QUANT.parent, prefix=".quant-") as tmp:
            stage = Path(tmp) / "mlx"
            subprocess.run(
                [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(BASE), "--mlx-path", str(stage),
                 "--quantize", "--q-bits", "4", "--q-group-size", "64"],
                check=True,
            )
            write_json(stage / "source.json", expected)
            stage.rename(QUANT)


if __name__ == "__main__":
    main()
