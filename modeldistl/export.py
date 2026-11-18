"""Fuse the selected delta into the original BF16 source and quantize once for llama.cpp."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from filelock import FileLock

from .files import write_json
from .weights import BASE, REPO, REVISION

SOURCE = Path(".tools/llama.cpp")
COMMIT = "401563f10ec60754e688cd3165c1343cf4be9642"
DEFAULT_ADAPTER = Path("artifacts/adapters/pilot/selected")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export(adapter, targets):
    revision = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise SystemExit("Run bash scripts/setup-llama.sh to install the pinned converter.")
    metadata = {"base": REPO, "revision": REVISION, "llama_cpp": COMMIT,
                "adapter_sha256": digest(adapter / "adapters.safetensors") if adapter else None}
    pending = {}
    for quantization, output in targets.items():
        receipt = output.with_suffix(".json")
        spec = metadata | {"quantization": quantization}
        if output.exists() and receipt.exists():
            previous = json.loads(receipt.read_text())
            if all(previous.get(k) == v for k, v in spec.items()) and digest(output) == previous.get("sha256"):
                print(f"Verified existing export {output}")
                continue
            raise SystemExit(f"{output} belongs to a different run. Select another --out path.")
        pending[quantization] = output
    if not pending:
        return
    destination = next(iter(pending.values())).parent
    with tempfile.TemporaryDirectory(dir=destination, prefix=".convert-") as scratch:
        scratch = Path(scratch)
        source = BASE
        if adapter:
            fused = scratch / "fused"
            subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", BASE, "--adapter-path", adapter,
                            "--save-path", fused], check=True)
            # Preserve the original HF tokenizer and config across both exports;
            # only tensor weights and their index come from the fusion step.
            source = scratch / "hf"
            source.mkdir()
            for path in BASE.iterdir():
                if path.is_file() and ".safetensors" not in path.name:
                    (source / path.name).symlink_to(path.resolve())
            for path in fused.glob("*.safetensors*"):
                (source / path.name).symlink_to(path.resolve())
        bf16 = scratch / "bf16.gguf"
        quantized = scratch / "model.gguf"
        subprocess.run([sys.executable, SOURCE / "convert_hf_to_gguf.py", source, "--outfile", bf16,
                        "--outtype", "bf16", "--model-name", "ModelDistl-4B-Pilot" if adapter else "Qwen3-4B-Instruct-2507"],
                       check=True)
        for quantization, output in pending.items():
            subprocess.run([".tools/llama-bin/llama-quantize", bf16, quantized, quantization], check=True)
            spec = metadata | {"quantization": quantization, "bytes": quantized.stat().st_size, "sha256": digest(quantized)}
            quantized.replace(output)
            write_json(output.with_suffix(".json"), spec)
            print(f"{output} ({spec['bytes']:,} bytes, SHA-256 {spec['sha256']})", flush=True)
        shutil.copy2(BASE / "LICENSE", destination / "Qwen-LICENSE.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    group.add_argument("--base-only", action="store_true", help="Export the matching baseline")
    parser.add_argument("--quantization", nargs="+", choices=["Q4_K_M", "Q3_K_S"], default=["Q4_K_M"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stem = "Qwen3-4B-base" if args.base_only else "ModelDistl-4B"
    if args.out and len(args.quantization) != 1:
        parser.error("--out specifies a single file; choose one quantization")
    parent = args.out.parent if args.out else Path("artifacts/export")
    targets = {q: args.out or parent / f"{stem}-{q}.gguf" for q in args.quantization}
    parent.mkdir(parents=True, exist_ok=True)
    with FileLock(parent / ".export.lock", timeout=0):
        export(None if args.base_only else args.adapter, targets)


if __name__ == "__main__":
    main()
