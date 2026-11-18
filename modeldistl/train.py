"""Train with MLX-LM, then select a saved adapter by validation completion loss."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml
from filelock import FileLock
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, create_dataset
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, evaluate
from mlx_lm.tuner.trainer import train as train_steps
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

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


def completion_loss(model, batch, bounds):
    # MLX-LM 0.32.2 pads the batch by at least one position and returns
    # (first target index, sequence length). Its loss takes an inclusive end.
    return default_loss(model, batch, bounds - mx.array([0, 1], dtype=bounds.dtype))


def train(config, output):
    # Only the two supervised splits are opened by this process.
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config | {"adapter_path": str(output)}))
    if not args.mask_prompt or args.iters % args.grad_accumulation_steps:
        raise ValueError("Use prompt masking and complete gradient accumulation cycles.")
    if args.fine_tune_type != "lora" or args.optimizer != "adamw":
        raise ValueError("This recipe uses LoRA and AdamW.")
    np.random.seed(args.seed)
    model, tokenizer = load(args.model)
    mx.random.seed(args.seed)
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
    if getattr(args, "epochs", None):
        if counts["train"]["rows"] % (args.batch_size * args.grad_accumulation_steps):
            raise ValueError("The complete-epoch recipe needs a whole number of optimizer updates.")
        args.iters = counts["train"]["rows"] // args.batch_size * args.epochs
        updates = args.iters // args.grad_accumulation_steps
        interval = max(1, updates // 4) * args.grad_accumulation_steps
        args.save_every = args.steps_per_eval = interval
        if getattr(args, "cosine_warmup_fraction", None):
            warmup = max(1, round(updates * args.cosine_warmup_fraction))
            args.lr_schedule = {"name": "cosine_decay", "arguments": [args.learning_rate, max(1, updates-warmup),
                                 args.learning_rate / 10], "warmup": warmup, "warmup_init": args.learning_rate / 20}
    output.mkdir(parents=True)
    metrics = Metrics(output / "metrics.json", counts)
    if getattr(args, "epochs", None):
        metrics.data["base_validation_loss"] = float(evaluate(model, CacheDataset(datasets["valid"]), batch_size=1,
                                                      num_batches=-1, max_seq_length=args.max_seq_length,
                                                      loss=completion_loss))
    model.freeze()
    if not 1 <= args.num_layers <= len(model.layers):
        raise ValueError(f"num_layers must be in [1, {len(model.layers)}]")
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters)
    if args.resume_adapter_file:
        model.load_weights(args.resume_adapter_file, strict=False)
    write_json(output / "adapter_config.json", vars(args))
    schedule = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer = optim.AdamW(learning_rate=schedule, **args.optimizer_config.get("adamw", {}))
    training = TrainingArgs(
        batch_size=args.batch_size, iters=args.iters, val_batches=args.val_batches,
        steps_per_report=args.steps_per_report, steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every, max_seq_length=args.max_seq_length,
        adapter_file=output / "adapters.safetensors", grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    train_steps(model=model, optimizer=optimizer, args=training, loss=completion_loss,
                train_dataset=CacheDataset(datasets["train"]), val_dataset=CacheDataset(datasets["valid"]),
                training_callback=metrics)
    metrics.data["trainable_parameters"] = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    metrics.data["optimizer_updates"] = args.iters // args.grad_accumulation_steps

    # The trainer logs validation before the update at a reporting step. Score
    # the actual saved weights after training so the selected file is unambiguous.
    candidates = {int(p.name.split("_")[0]): p for p in output.glob("*_adapters.safetensors")}
    candidates[args.iters] = output / "adapters.safetensors"
    scores = []
    validation = CacheDataset(datasets["valid"])
    for step, path in sorted(candidates.items()):
        model.load_weights(str(path), strict=False)
        loss = evaluate(model, validation, batch_size=1, num_batches=-1, max_seq_length=args.max_seq_length,
                        loss=completion_loss)
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
