"""Read the curriculum and grade visible answers without sending references to a model."""

import json
import re
from fractions import Fraction
from pathlib import Path

from pydantic_monty import MontyError

DOMAINS = ("coding", "general", "math", "reasoning")
SPLITS = ("train", "valid", "test")


def curriculum(directory=Path("curriculum")):
    splits = {}
    ids, prompts, families = set(), set(), {}
    for split in SPLITS:
        tasks = []
        for line in (directory / f"{split}.jsonl").read_text().splitlines():
            task = json.loads(line)
            normalized = " ".join(task["prompt"].casefold().split())
            family = task["family"]
            if (task["id"] in ids or normalized in prompts
                    or families.setdefault(family, split) != split):
                raise ValueError(f"Duplicate or split leakage at {task['id']}")
            if task["domain"] not in DOMAINS or not normalized or not family:
                raise ValueError(f"Incomplete curriculum row {task['id']}")
            ids.add(task["id"])
            prompts.add(normalized)
            tasks.append(task)
        if {t["domain"] for t in tasks} != set(DOMAINS):
            raise ValueError(f"{split} must have coverage in all four domains")
        splits[split] = tasks
    return splits


def same_value(actual, expected):
    # Distinguish JSON true from 1 and false from 0, including inside containers.
    return json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(
        expected, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def grade(task, answer, pool):
    spec = task["check"]
    answer = answer.strip()
    if not answer or len(answer) > 32_000:
        return False, "empty or excessive answer"
    kind = spec["kind"]
    if kind == "code":
        fence = re.fullmatch(r"\x60{3}(?:python|py)?\s*\n(.*?)\n\x60{3}", answer, re.S | re.I)
        code = fence[1] if fence else answer
        limits = {"max_feed_duration_secs": 0.5, "max_memory": 8_000_000,
                  "max_recursion_depth": 128, "max_total_sleep_secs": 0}
        try:
            with pool.checkout(limits=limits) as session:
                session.feed_run(code, print_callback=lambda *_: None)
                for i, case in enumerate(spec["cases"], 1):
                    result = session.feed_run("solve(*arguments)", inputs={"arguments": case["args"]},
                                              print_callback=lambda *_: None)
                    if not same_value(result, case["want"]):
                        return False, f"case {i}"
        except (MontyError, TypeError, ValueError, OverflowError) as error:
            return False, type(error).__name__
        return True, "all executions passed"
    if kind == "json":
        try:
            result = same_value(json.loads(answer), spec["answer"])
        except (ValueError, TypeError, RecursionError):
            result = False
        return result, "JSON equality"
    if kind == "rubric":
        okay = len(answer.split()) <= spec["max_words"] and all(
            re.search(pattern, answer, re.I) for pattern in spec["all"]
        )
        return okay, "required content and word limit"
    if kind == "choice":
        return answer.strip(" .()").casefold() == spec["answer"].casefold(), "choice label"
    # Commas, LaTeX and units have distinct meanings. Do not strip arbitrary text
    # from model answers to find a reference number somewhere in the explanation.
    last = answer.splitlines()[-1].replace("**", "").strip()
    match = re.fullmatch(r"Answer\s*:\s*(.+)", last, re.I)
    if not match:
        return False, "missing final answer line"
    value = match[1].strip().rstrip(".").strip()
    if kind == "label":
        return value.casefold() in {a.casefold() for a in spec["answers"]}, "final label"
    if kind == "number":
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
            value = value.replace(",", "")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?", value):
            return False, "final number format"
        try:
            return Fraction(value) == Fraction(spec["answer"]), "exact numerical value"
        except (ValueError, ZeroDivisionError):
            return False, "invalid numerical value"
    raise ValueError(f"Unknown verifier {kind}")
