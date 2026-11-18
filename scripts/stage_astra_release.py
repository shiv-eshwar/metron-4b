"""Verify the expanded student and prepare the adapter, checksums, and Q4 release parts."""

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from model_file import ROOT, pack, sha256

RECORD = ROOT / "runs/astra-v2"
ARTIFACTS = ROOT / "artifacts/astra-v2"
ADAPTER = ROOT / "artifacts/adapters/astra-v2/selected"
TAG = "astra-v0.2.0"
QUANTS = ("Q4_K_M", "Q3_K_S")


def tensor_audit(quant):
    sys.path.insert(0, str(ROOT / ".tools/llama.cpp/gguf-py"))
    from gguf import GGUFReader

    def hashes(p):
        model = GGUFReader(p)
        return {t.name: hashlib.sha256(memoryview(t.data).cast("B")).digest() for t in model.tensors}

    base = hashes(ROOT / "artifacts/export" / f"Qwen3-4B-base-{quant}.gguf")
    student = hashes(ARTIFACTS / "export" / f"ModelDistl-4B-{quant}.gguf")
    if base.keys() != student.keys():
        raise ValueError(f"Unexpected tensor inventory in {quant}")
    changed = sorted(k for k in student if student[k] != base[k])
    expected = {f"blk.{i}.attn_{p}.weight" for i in range(20, 36) for p in ("q", "k", "v", "output")}
    if set(changed) != expected:
        raise ValueError(f"{quant} changes differ from the trained linear projections")
    return {"total_tensors": len(base), "changed": len(changed), "names": changed}


def main():
    release = ARTIFACTS / "release"
    release.mkdir(exist_ok=True, parents=True)
    metrics = json.loads((RECORD / "training.json").read_text())
    if sha256(ADAPTER / "adapters.safetensors") != metrics["adapter_sha256"]:
        raise ValueError("Adapter digest differs from the selected validation checkpoint")
    archive = release / "ModelDistl-4B-Astra-v2-MLX.zip"
    with tempfile.TemporaryDirectory(dir=release) as scratch:
        tmp = Path(scratch) / archive.name
        with zipfile.ZipFile(tmp, "w") as z:
            for name in ("adapter_config.json", "adapters.safetensors"):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                z.writestr(info, (ADAPTER / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
        tmp.replace(archive)
    (RECORD / "adapter.SHA256").write_text(f"{sha256(archive)}  {archive.name}\n")
    checks, audits = [], {}
    for quant in QUANTS:
        path = ARTIFACTS / "export" / f"ModelDistl-4B-{quant}.gguf"
        receipt = path.with_suffix(".json")
        meta = json.loads(receipt.read_text())
        digest = sha256(path)
        if digest != meta["sha256"] or meta["adapter_sha256"] != metrics["adapter_sha256"]:
            raise ValueError(f"Export receipt mismatch in {quant}")
        checks.append(f"{digest}  {path.name}\n")
        shutil.copy2(receipt, RECORD / receipt.name)
        audits[quant] = tensor_audit(quant)
        print(quant, meta["bytes"], "bytes,", audits[quant]["changed"], "changed tensors", flush=True)
    (RECORD / "models.SHA256").write_text("".join(checks))
    (RECORD / "tensor-verification.json").write_text(json.dumps(audits, indent=2) + "\n")
    pack(ARTIFACTS / "export/ModelDistl-4B-Q4_K_M.gguf", release, RECORD / "release.json", TAG)


if __name__ == "__main__":
    main()
