"""Select an adapter multiplier using validation generation and coding retention."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from filelock import FileLock
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from .benchmarks import mbpp
from .evaluate import MAX_TOKENS
from .files import write_json
from .tasks import DOMAINS, curriculum, grade
from .weights import QUANT

MULTIPLIERS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
REFINEMENT = (0.03125, 0.1875, 0.3125, 0.375, 0.015625, 0.0078125)


def scaled(weights, factor):
    return {k: v * factor if k.endswith(".lora_b") else v for k, v in weights.items()}


def qualified(candidates):
    baseline = candidates[0]
    return [c for c in candidates[1:] if c["scores"]["coding"] >= baseline["scores"]["coding"]
            and c["scores"]["reasoning"] >= baseline["scores"]["reasoning"] and c["macro"] >= baseline["macro"]]


def calibrate(source, dest):
    tasks = curriculum()["valid"] + mbpp("validation")
    weights_path = source / "adapters.safetensors"
    weights = mx.load(str(weights_path))
    model, tok = load(str(QUANT), adapter_path=str(source))
    prompts = [tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                       tokenize=True, return_dict=False) for t in tasks]
    report = {"source_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
              "multipliers": MULTIPLIERS + REFINEMENT, "candidates": []}
    path = dest.parent / "calibration.json"
    if path.exists():
        cache = json.loads(path.read_text())
        if all(cache.get(k) == report[k] for k in ("source_sha256", "validation_sha256")):
            report["candidates"] = cache["candidates"]
    done = {c["factor"] for c in report["candidates"]}
    for factor in MULTIPLIERS + REFINEMENT:
        if factor in REFINEMENT and qualified([c for c in report["candidates"] if c["factor"] in MULTIPLIERS]):
            break
        if factor in done:
            continue
        model.load_weights(list(scaled(weights, factor).items()), strict=False)
        mx.random.seed(42)
        output = batch_generate(model, tok, prompts, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0),
                                prefill_batch_size=4, completion_batch_size=8)
        results = []
        for task, answer in zip(tasks, output.texts, strict=True):
            passed, detail = grade(task, answer, None, cpython=True)
            results.append({"id": task["id"], "domain": task["domain"], "answer": answer,
                            "pass": passed, "check": detail})
        scores = {}
        for domain in DOMAINS:
            rows = [r for r in results if r["domain"] == domain]
            scores[domain] = sum(r["pass"] for r in rows) / len(rows)
        candidate = {"factor": factor, "scores": scores, "macro": sum(scores.values()) / len(scores),
                     "results": results}
        report["candidates"].append(candidate)
        write_json(path, report)
        print(f"Validation factor {factor}: {scores}", flush=True)
    eligible = qualified(report["candidates"])
    if not eligible:
        raise SystemExit("No tested adapter retained base coding, reasoning, and macro accuracy on validation.")
    # For ties keep the smallest nonzero update, closest to the pretrained weights.
    winner = max(eligible, key=lambda c: (c["macro"], -c["factor"]))
    dest.mkdir(exist_ok=True, parents=True)
    mx.save_safetensors(str(dest / "adapters.safetensors"), scaled(weights, winner["factor"]))
    shutil.copy2(source / "adapter_config.json", dest / "adapter_config.json")
    report["selected_multiplier"] = winner["factor"]
    report["adapter_sha256"] = hashlib.sha256((dest / "adapters.safetensors").read_bytes()).hexdigest()
    write_json(path, report)
    print(f"Deployment multiplier: {winner['factor']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/adapters/pilot/selected"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/adapters/pilot/deploy"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(args.out.parent / ".calibrate.lock", timeout=0):
        calibrate(args.source, args.out)


if __name__ == "__main__":
    main()
