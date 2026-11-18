"""Fetch pinned Gemma weights and compare paired, complete inference reports."""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download

from .expanded_data import ROOT
from .files import write_json
from .weights import REVISION

GEMMA = {
    "gemma3": ("ggml-org/gemma-3-4b-it-GGUF", "d0976223747697cb51e056d85c532013931fe52e",
               "gemma-3-4b-it-Q4_K_M.gguf", "882e8d2db44dc554fb0ea5077cb7e4bc49e7342a1f0da57901c0802ea21a0863"),
    "gemma4": ("unsloth/gemma-4-E4B-it-GGUF", "bfc15c382204943c3a8fff0c750b94ae2364d7a3",
               "gemma-4-E4B-it-Q4_K_M.gguf", "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87"),
}


def fetch(name):
    repo, rev, filename, expected = GEMMA[name]
    path = Path(hf_hub_download(repo, filename, revision=rev, local_dir="artifacts/comparators"))
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"Comparator digest mismatch for {name}")
    print(path, flush=True)
    return path


def score(name):
    work = Path("artifacts/astra-v2")
    work.mkdir(parents=True, exist_ok=True)
    rev = REVISION
    if name in GEMMA:
        rev = GEMMA[name][1]
        path = fetch(name)
    else:
        path = {"base": Path("artifacts/export/Qwen3-4B-base-Q4_K_M.gguf"),
                "pilot": Path("artifacts/export/ModelDistl-4B-Q4_K_M.gguf"),
                "student": work / "export/ModelDistl-4B-Q4_K_M.gguf"}[name]
    out = work / f"{name}.json"
    with FileLock(work / "metal.lock"), (work / f"{name}.runner.log").open("a") as log:
        subprocess.run([sys.executable, "-u", "-m", "modeldistl.evaluate", "--gguf", path, "--suite", "sealed",
                        "--max-tokens", "512", "--reasoning", "off", "--jobs", "4", "--resume",
                        "--source-revision", rev, "--out", out], stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(out.read_text())
        if not report["complete"] or report["total"] != 256:
            raise RuntimeError("Incomplete head-to-head")
        write_json(ROOT / "results" / out.name, report)
        print(name, report["correct"], "/", report["total"], flush=True)


def exact_sign_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def paired(student, baseline, n_boot=10000):
    for field in ("test_sha256", "temperature", "seed", "max_tokens", "reasoning", "suite", "jobs", "code_runtime"):
        if student[field] != baseline[field]:
            raise ValueError(f"Evaluation settings differ at {field}")
    ids = lambda rows: {r["id"]: r for r in rows["results"]}
    target, control = ids(student), ids(baseline)
    if not student["complete"] or not baseline["complete"] or not target or (
            target.keys() != control.keys() or len(target) != student["total"] or len(control) != baseline["total"]):
        raise ValueError("Both reports must contain exactly the same complete set of unique IDs")
    changes = defaultdict(list)
    wins, losses = [], []
    for i, r in target.items():
        if r["domain"] != control[i]["domain"]:
            raise ValueError(f"Domain mismatch at {i}")
        diff = int(r["pass"]) - int(control[i]["pass"])
        changes[r["domain"]].append(diff)
        if diff == 1:
            wins.append(i)
        if diff == -1:
            losses.append(i)
    rng = random.Random(621908)
    n = len(target)
    distribution = sorted(sum(sum(rng.choices(group, k=len(group))) for group in changes.values()) / n
                          for _ in range(n_boot))
    return {"student_only": len(wins), "baseline_only": len(losses), "ties": n-len(wins)-len(losses),
            "difference_pp": round(100*(len(wins)-len(losses))/n, 2),
            "paired_95pct_bootstrap_pp": [round(100*distribution[i], 2) for i in (int(.025*n_boot), int(.975*n_boot)-1)],
            "two_sided_exact_p": round(exact_sign_p(len(wins), len(losses)), 6),
            "student_only_ids": wins, "baseline_only_ids": losses}


def summarize(directory):
    reports = {name: json.loads((directory / f"{name}.json").read_text())
               for name in ("student", "base", "pilot", "gemma3", "gemma4")}
    scores = {}
    for k, r in reports.items():
        spans = [a["span_f1"] for a in r["results"] if "span_f1" in a]
        scores[k] = {"domains": r["domains"], "total_correct": r["correct"], "total": r["total"],
                     "arc_correct_content": r.get("arc_choice_content"), "reading_mean_f1": sum(spans)/len(spans),
                     "length_limit_finishes": sum(a["finish"] == "length" for a in r["results"]),
                     "gguf_sha256": r["gguf_file_sha256"]}
    result = {"scores": scores, "student_minus_comparator": {k: paired(reports["student"], r)
              for k, r in reports.items() if k != "student"}}
    write_json(ROOT / "comparison.json", result)
    print(json.dumps(scores, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["fetch", "score", "summarize"])
    p.add_argument("--model", choices=[*GEMMA, "base", "pilot", "student"], default="gemma3")
    p.add_argument("--reports", type=Path, default=ROOT / "results")
    a = p.parse_args()
    if a.action == "fetch":
        if a.model not in GEMMA:
            p.error("fetch accepts a Gemma comparator")
        fetch(a.model)
    elif a.action == "score":
        score(a.model)
    else:
        summarize(a.reports)


if __name__ == "__main__":
    main()
