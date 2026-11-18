"""Train with MLX-LM, then select a saved adapter by validation completion loss."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from filelock import FileLock
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, create_dataset
from mlx_lm.tuner.trainer import evaluate

from .files import write_json


class Metrics(TrainingCallback):
    def __init__(self, path, token_counts):
        self.path = path
        self.data = {"token_counts": token_counts, "train": [], "validation": []}

    def on_train_loss_report(self, info):
        self.data["train"].append(info)
        write_json(self.path, self.data)

    def on_val_loss_report(self, info):
        self.data["validation"].append(info)
        write_json(self.path, self.data)


def train(config, output):
    # Only the two supervised splits are opened by this process.
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config | {"adapter_path": str(output)}))
    if not args.mask_prompt or args.iters % args.grad_accumulation_steps:
        raise ValueError("Use prompt masking and complete gradient accumulation cycles.")
    np.random.seed(args.seed)
    model, tokenizer = load(args.model)
    datasets = {}
    counts = {}
    for split in ("train", "valid"):
        source = Path(args.data) / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        ds = create_dataset(rows, tokenizer, args)
        supervised = []
        lengths = []
        for row in rows:
            tokens, offset = ds.process(row)
            if not 0 < offset < len(tokens) <= args.max_seq_length:
                raise ValueError(f"{split} contains an empty target or would truncate a chat.")
            supervised.append(len(tokens) - offset)
            lengths.append(len(tokens))
        datasets[split] = ds
        counts[split] = {"rows": len(rows), "completion_tokens": sum(supervised),
                         "longest_chat": max(lengths), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    output.mkdir(parents=True)
    metrics = Metrics(output / "metrics.json", counts)
    train_model(args, model, datasets["train"], datasets["valid"], training_callback=metrics)
    metrics.data["trainable_parameters"] = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))

    # The trainer logs validation before the update at a reporting step. Score
    # the actual saved weights after training so the selected file is unambiguous.
    candidates = {int(p.name.split("_")[0]): p for p in output.glob("*_adapters.safetensors")}
    candidates[args.iters] = output / "adapters.safetensors"
    scores = []
    validation = CacheDataset(datasets["valid"])
    for step, path in sorted(candidates.items()):
        model.load_weights(str(path), strict=False)
        loss = evaluate(model, validation, batch_size=1, num_batches=-1, max_seq_length=args.max_seq_length)
        scores.append({"iteration": step, "loss": float(loss), "file": path.name})
    best = min(scores, key=lambda r: r["loss"])
    selected = output / "selected"
    selected.mkdir()
    shutil.copy2(output / best["file"], selected / "adapters.safetensors")
    shutil.copy2(output / "adapter_config.json", selected / "adapter_config.json")
    metrics.data.update(checkpoints=scores, selected_iteration=best["iteration"],
                        adapter_sha256=hashlib.sha256((selected / "adapters.safetensors").read_bytes()).hexdigest())
    write_json(metrics.path, metrics.data)
    print(f"Selected iteration {best['iteration']} at validation loss {best['loss']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--data", type=Path, help="Directory with verified train.jsonl and valid.jsonl")
    parser.add_argument("--output", type=Path, help="New run directory; defaults to adapter_path in the config")
    options = parser.parse_args()
    config = yaml.safe_load(options.config.read_text())
    if options.data:
        config["data"] = str(options.data)
    output = options.output or Path(config["adapter_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(output.parent / f".{output.name}.lock", timeout=0):
        if output.exists():
            raise SystemExit(f"{output} already exists. Choose --output for a new run.")
        train(config, output)


if __name__ == "__main__":
    main()
