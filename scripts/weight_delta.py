"""Transfer a quantized adapter update as a compressed XOR against the pinned stock GGUF."""

import argparse
import gzip
import hashlib
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from model_file import ROOT


def transform(action, quant):
    models = ROOT / "artifacts/export"
    patches = ROOT / "artifacts/release"
    patches.mkdir(parents=True, exist_ok=True)
    base = models / f"Qwen3-4B-base-{quant}.gguf"
    model = models / f"ModelDistl-4B-{quant}.gguf"
    patch = patches / f"{quant}.xor.gz"
    base_spec = json.loads((ROOT / "runs/pilot" / base.with_suffix(".json").name).read_text())
    spec = json.loads((ROOT / "runs/pilot" / model.with_suffix(".json").name).read_text())
    writing = action == "make"
    destination = patch if writing else model
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / destination.name
        parent_hash, model_hash = hashlib.sha256(), hashlib.sha256()
        with base.open("rb") as left, ExitStack() as stack:
            if writing:
                right = stack.enter_context(model.open("rb"))
                target = stack.enter_context(gzip.GzipFile(filename=stage, mode="wb", compresslevel=3, mtime=0))
            else:
                right = stack.enter_context(gzip.open(patch, "rb"))
                target = stack.enter_context(stage.open("wb"))
            for block in iter(lambda: left.read(4 << 20), b""):
                other = right.read(len(block))
                if len(other) != len(block):
                    raise ValueError("Mismatched GGUF length")
                result = np.bitwise_xor(np.frombuffer(block, dtype="u1"), np.frombuffer(other, dtype="u1"))
                target.write(result)
                parent_hash.update(block)
                model_hash.update(other if writing else result)
            if right.read(1):
                raise ValueError("Unexpected trailing bytes")
        if parent_hash.hexdigest() != base_spec["sha256"] or model_hash.hexdigest() != spec["sha256"]:
            raise ValueError("GGUF checksum differs from the recorded build")
        stage.replace(destination)
    print(f"Verified {destination.name} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["make", "apply"])
    parser.add_argument("quantization", choices=["Q4_K_M", "Q3_K_S"])
    args = parser.parse_args()
    transform(args.action, args.quantization)
