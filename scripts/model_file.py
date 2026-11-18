"""Publishable parts for GitHub's per-asset size limit, with verified single-file reconstruction."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/pilot/release.json"
MODEL = "ModelDistl-4B-Q4_K_M.gguf"
BLOCK = 4 * 1024 * 1024
PART_BYTES = 1_300_000_000


def sha256(file):
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(model, directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    full = hashlib.sha256()
    size = model.stat().st_size
    with model.open("rb") as source, tempfile.TemporaryDirectory(dir=directory) as tmp:
        if source.read(4) != b"GGUF":
            raise ValueError("Expected a GGUF model")
        source.seek(0)
        while source.tell() < size:
            part = Path(tmp) / f"{model.name}.part-{len(parts) + 1:02d}"
            h = hashlib.sha256()
            with part.open("wb") as target:
                remaining = min(size - source.tell(), PART_BYTES)
                while remaining:
                    chunk = source.read(min(BLOCK, remaining))
                    if not chunk:
                        raise EOFError("Source model changed while packing")
                    h.update(chunk)
                    full.update(chunk)
                    target.write(chunk)
                    remaining -= len(chunk)
            parts.append({"file": part.name, "bytes": part.stat().st_size, "sha256": h.hexdigest()})
        for part in parts:
            (Path(tmp) / part["file"]).replace(directory / part["file"])
    info = {"repo": "shiv-eshwar/modeldistl", "tag": "pilot-v0.1.0", "file": model.name,
            "bytes": size, "sha256": full.hexdigest(), "parts": parts}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    return info


def assemble(info, directory, target, download):
    if target.exists():
        if sha256(target) == info["sha256"]:
            print(f"Verified {target}")
            return
        raise SystemExit(f"A different file exists at {target}. Use --out to choose a new filename.")
    target.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        single = Path(work) / info["file"]
        with single.open("wb") as combined:
            for part in info["parts"]:
                source = directory / part["file"]
                if not source.exists() or sha256(source) != part["sha256"]:
                    if not download:
                        raise ValueError(f"Missing or damaged part {source}")
                    with tempfile.TemporaryDirectory(dir=directory) as incoming:
                        subprocess.run(["gh", "release", "download", info["tag"], "--repo", info["repo"],
                                        "--pattern", part["file"], "--dir", incoming], check=True)
                        fetched = Path(incoming) / part["file"]
                        if sha256(fetched) != part["sha256"]:
                            raise ValueError(f"Checksum mismatch for {fetched.name}")
                        fetched.replace(source)
                with source.open("rb") as chunk:
                    shutil.copyfileobj(chunk, combined, length=BLOCK)
        if single.stat().st_size != info["bytes"] or sha256(single) != info["sha256"]:
            raise ValueError("Reassembled model failed verification")
        single.replace(target)
    print(f"Verified {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "download", "join"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/export" / MODEL)
    parser.add_argument("--parts", type=Path, default=ROOT / "artifacts/release")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.out, args.parts, args.manifest)
    else:
        assemble(json.loads(args.manifest.read_text()), args.parts, args.out, args.action == "download")


if __name__ == "__main__":
    main()
