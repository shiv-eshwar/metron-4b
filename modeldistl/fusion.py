"""Fuse MLX linear LoRA tensors into the original BF16 shards on a CPU."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def merge_linear(base, a, b, scale):
    """Apply MLX's BF16 update using fixed-order FP32 sums for portable release hashes."""
    if base.dtype != torch.bfloat16 or base.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected a BF16 linear weight and two adapter matrices")
    if (a.shape[0], b.shape[1]) != (base.shape[1], base.shape[0]) or a.shape[1] != b.shape[0]:
        raise ValueError("LoRA dimensions do not match the base tensor")
    a = np.ascontiguousarray(a.float().numpy().T)
    b = b.float().numpy() * np.float32(scale)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite adapter")
    fused = torch.empty_like(base)
    for start in range(0, base.shape[0], 128):
        stop = min(start + 128, base.shape[0])
        delta = np.zeros((stop-start, base.shape[1]), dtype=np.float32)
        product = np.empty_like(delta)
        for rank in range(a.shape[0]):
            np.multiply(b[rank, start:stop, None], a[rank], out=product)
            np.add(delta, product, out=delta)
        fused[start:stop] = base[start:stop] + torch.from_numpy(delta).to(torch.bfloat16)
    return fused


def fuse_shards(base, adapter, output):
    torch.set_num_threads(1)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config["fine_tune_type"] != "lora":
        raise ValueError("Portable export supports linear LoRA")
    scale = config["lora_parameters"]["scale"]
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    output.mkdir()
    with safe_open(adapter / "adapters.safetensors", framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        targets = defaultdict(dict)
        for key in sorted(keys):
            if not key.endswith(".lora_a"):
                continue
            prefix = key.removesuffix(".lora_a")
            target = prefix + ".weight"
            if target not in index or prefix + ".lora_b" not in keys:
                raise ValueError(f"Unpaired adapter or missing base tensor at {prefix}")
            targets[index[target]][target] = prefix
        if 2 * sum(map(len, targets.values())) != len(keys) or not targets:
            raise ValueError("Unsupported or empty adapter")
        for file in base.iterdir():
            if not file.is_file():
                continue
            if file.name not in targets:
                (output / file.name).symlink_to(file.resolve())
                continue
            with safe_open(file, framework="pt", device="cpu") as source:
                shard = {}
                for key in source.keys():  # noqa: SIM118 - safetensors reader is not iterable.
                    tensor = source.get_tensor(key)
                    if key in targets[file.name]:
                        prefix = targets[file.name][key]
                        tensor = merge_linear(tensor, weights.get_tensor(prefix+".lora_a"),
                                              weights.get_tensor(prefix+".lora_b"), scale)
                    shard[key] = tensor
                save_file(shard, output / file.name, metadata=source.metadata())
            del shard
    print(f"Fused {sum(map(len, targets.values()))} linear weights into {len(targets)} shards", flush=True)
    return Path(output)
