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


def export(adapter, output):
    revision = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise SystemExit("Run bash scripts/setup-llama.sh to install the pinned converter.")
    metadata = {"base": REPO, "revision": REVISION, "llama_cpp": COMMIT, "quantization": "Q4_K_M",
                "adapter_sha256": digest(adapter / "adapters.safetensors") if adapter else None}
    receipt = output.with_suffix(".json")
    if output.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if all(previous.get(k) == v for k, v in metadata.items()) and digest(output) == previous.get("sha256"):
            print(f"Verified existing export {output}")
            return
        raise SystemExit(f"{output} belongs to a different run. Select another --out path.")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".convert-") as scratch:
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
        subprocess.run([".tools/llama-bin/llama-quantize", bf16, quantized, "Q4_K_M"], check=True)
        metadata.update(bytes=quantized.stat().st_size, sha256=digest(quantized))
        quantized.replace(output)
        write_json(receipt, metadata)
        shutil.copy2(BASE / "LICENSE", output.parent / "Qwen-LICENSE.txt")
        print(f"{output} ({metadata['bytes']:,} bytes, SHA-256 {metadata['sha256']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    group.add_argument("--base-only", action="store_true", help="Export the matching baseline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    name = "Qwen3-4B-base-Q4_K_M.gguf" if args.base_only else "ModelDistl-4B-Q4_K_M.gguf"
    out = args.out or Path("artifacts/export") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out) + ".lock", timeout=0):
        export(None if args.base_only else args.adapter, out)


if __name__ == "__main__":
    main()
