"""Select adapter strength on independent development data and confirm on a fresh holdout."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from filelock import FileLock

from .comparison import GEMMA, fetch, paired
from .expanded_data import ROOT, gsm, mbpp, order, question_key, reading, science
from .files import write_json, write_jsonl
from .overlap import function_name
from .weights import REVISION

RECORD = ROOT / "retention"
WORK = Path("artifacts/astra-v2/retention")
FACTORS = {"base": 0, "one-sixteenth": 0.0625, "one-quarter": 0.25, "one-half": 0.5, "full": 1}
ADAPTER = Path("artifacts/adapters/astra-v2/selected")
STOCK = Path("artifacts/export/Qwen3-4B-base-Q4_K_M.gguf")
FULL = Path("artifacts/astra-v2/export/ModelDistl-4B-Q4_K_M.gguf")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def passage(t):
    if t["source"] == "squad":
        return " ".join(t["prompt"].partition("\n\nQuestion:")[0].removeprefix("Passage:").casefold().split())
    if t["source"] == "quac":
        return " ".join(t["check"]["context"].casefold().split())
    return None


def prepare():
    prior = [t for split in ("train", "valid", "test") for t in read_lines(ROOT / "curriculum" / f"{split}.jsonl")]
    seen = {t["id"] for t in prior}
    fingerprints = {question_key(t) for t in prior}
    functions = {s for t in prior if (s := function_name(t))}
    contexts = {s for t in prior if (s := passage(t))}
    for domain in ("public", "mbpp"):
        for r in json.loads(Path(f"runs/pilot/results/base-{domain}.json").read_text())["results"]:
            seen.add("mbpp-" + r["id"].split("-")[-1] if domain == "mbpp" else r["id"])
    from .benchmarks import mbpp as pilot_mbpp

    seen.update("mbpp-" + t["id"].split("-")[-1] for t in pilot_mbpp("validation"))
    for phase, n in (("valid", 24), ("test", 32)):
        official = "test" if phase == "test" else "validation"
        grouped = reading("train" if phase == "valid" else "validation")
        read_rows = [r for per_title in grouped.values() for bucket in per_title.values() for r in bucket]
        buckets = [
            (mbpp(official, full=phase == "valid"), n),
            (gsm("train" if phase == "valid" else "test"), n),
            (science(official), n),
            ([r for r in read_rows if r["check"]["answers"] != ["UNANSWERABLE"]], 3 * n // 4),
            ([r for r in read_rows if r["check"]["answers"] == ["UNANSWERABLE"]], n // 4),
        ]
        rows = []
        for candidates, wanted in buckets:
            count = 0
            for t in order(candidates, "retention-confirm-91844-" + phase):
                name, ctx, key = function_name(t), passage(t), question_key(t)
                if (
                    t["id"] in seen
                    or key in fingerprints
                    or (name and name in functions)
                    or (ctx and ctx in contexts)
                ):
                    continue
                rows.append(t)
                seen.add(t["id"])
                fingerprints.add(key)
                if name:
                    functions.add(name)
                if ctx:
                    contexts.add(ctx)
                count += 1
                if count == wanted:
                    break
            if count != wanted:
                raise ValueError(f"{phase} only has {count} of {wanted} eligible items")
        rows = order(rows, phase + "-retention-mix")
        write_jsonl(RECORD / f"{phase}.jsonl", rows)
        print(phase, dict(Counter(t["domain"] for t in rows)), flush=True)


def model_path(name):
    if name in GEMMA:
        return fetch(name)
    if name == "base":
        return STOCK
    if name == "full":
        return FULL
    return WORK / name / "ModelDistl-4B-Retention-Q4_K_M.gguf"


def export_scaled(name):
    import numpy as np
    from safetensors.numpy import load_file, save_file

    out = model_path(name)
    dest = out.parent / "adapter"
    dest.mkdir(parents=True, exist_ok=True)
    weights = load_file(ADAPTER / "adapters.safetensors")
    for k in weights:
        if k.endswith(".lora_b"):
            weights[k] = weights[k] * np.float32(FACTORS[name])
    source = dest / "adapters.safetensors"
    if not source.exists():
        save_file(weights, source)
    else:
        actual = load_file(source)
        if actual.keys() != weights.keys() or any(not np.array_equal(v, actual[k]) for k, v in weights.items()):
            raise ValueError(f"Existing adapter mismatch at {name}")
    shutil.copy2(ADAPTER / "adapter_config.json", dest / "adapter_config.json")
    with (out.parent / "export.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "modeldistl.export",
                "--adapter",
                dest,
                "--out",
                out,
                "--name",
                "ModelDistl-4B-Astra-Retention",
                "--portable-fusion",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate(name, phase):
    path = model_path(name)
    reports = RECORD / phase
    logs = WORK / phase
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for suite, suffix, limit in (("sealed", "", 512), ("sealed-choice", "-choice", 8)):
        out = logs / (name + suffix + ".json")
        with out.with_suffix(".runner.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "modeldistl.evaluate",
                    "--gguf",
                    path,
                    "--suite",
                    suite,
                    "--test-file",
                    RECORD / (phase + ".jsonl"),
                    "--reasoning",
                    "off",
                    "--max-tokens",
                    str(limit),
                    "--jobs",
                    "4",
                    "--code-runtime",
                    "cpython",
                    "--resume",
                    "--source-revision",
                    GEMMA[name][1] if name in GEMMA else REVISION,
                    "--out",
                    out,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        r = json.loads(out.read_text())
        if not r["complete"] or r["total"] != (24 if phase == "valid" else 32) * (4 if suite == "sealed" else 1):
            raise RuntimeError("Incomplete retention report")
        write_json(reports / out.name, r)
        results.append(r)
    regular, forced = results
    values = {k: v["correct"] for k, v in regular["domains"].items()}
    values.update(
        strict=regular["correct"],
        forced_science=forced["correct"],
        forced_total=regular["correct"] - values["reasoning"] + forced["correct"],
    )
    print(phase, name, values, flush=True)
    return values


def select(scores):
    base = scores["base"]
    eligible = [
        k
        for k, v in scores.items()
        if v["coding"] >= base["coding"] - 1
        and v["math"] >= base["math"] - 1
        and v["coding"] + v["math"] >= base["coding"] + base["math"]
    ]
    return max(
        eligible,
        key=lambda k: (
            scores[k]["strict"],
            scores[k]["coding"] + scores[k]["math"],
            scores[k]["forced_total"],
            -FACTORS[k],
        ),
    )


def report_pairs(phase, winner, names):
    path = RECORD / phase

    def load(name, subset):
        r = json.loads((path / (name + ".json")).read_text())
        if subset == "coding_math":
            r["results"] = [a for a in r["results"] if a["domain"] in ("coding", "math")]
        if subset == "forced":
            f = json.loads((path / (name + "-choice.json")).read_text())
            opts = {a["id"]: a["pass"] for a in f["results"]}
            r["results"] = [a | {"pass": opts.get(a["id"], a["pass"])} for a in r["results"]]
        r["total"] = len(r["results"])
        return r

    return {
        name: {sub: paired(load(winner, sub), load(name, sub)) for sub in ("strict", "coding_math", "forced")}
        for name in names
        if name != winner
    }


def experiment():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = Path("artifacts/astra-v2/metal.lock")
    scores = {}
    with FileLock(WORK / "experiment.lock", timeout=0):
        for name, factor in FACTORS.items():
            with FileLock(lock):
                if 0 < factor < 1:
                    export_scaled(name)
                scores[name] = evaluate(name, "valid")
        winner = select(scores)
        result = {
            "strengths": FACTORS,
            "development": scores,
            "selected": winner,
            "multiplier": FACTORS[winner],
            "validation_sha256": hashlib.sha256((RECORD / "valid.jsonl").read_bytes()).hexdigest(),
            "confirmation_sha256": hashlib.sha256((RECORD / "test.jsonl").read_bytes()).hexdigest(),
        }
        write_json(RECORD / "selection.json", result)
        print("Selected on development", winner, flush=True)
        names = list(dict.fromkeys([winner, "base", "full", "gemma3", "gemma4"]))
        result["confirmation"] = {}
        for name in names:
            with FileLock(lock):
                result["confirmation"][name] = evaluate(name, "test")
        result["paired"] = report_pairs("test", winner, names)
        write_json(RECORD / "comparison.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run"])
    args = p.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        experiment()


if __name__ == "__main__":
    main()
