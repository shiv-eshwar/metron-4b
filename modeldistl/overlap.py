"""Audit supervised sources against held-out questions without exposing them to the teacher."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .expanded_data import ROOT, question_key
from .files import write_json


def function_name(row):
    if row["source"] != "mbpp":
        return None
    example = row["prompt"].partition("Example: ")[2].splitlines()[0]
    m = re.search(r"\bassert\s+\(?\s*([A-Za-z_]\w*)\s*\(", example)
    return m[1] if m and m[1] not in ("len", "set", "list", "sorted", "round", "abs", "sum", "str", "tuple") else None


def shingles(text):
    w = text.split()
    return {tuple(w[i:i+5]) for i in range(len(w)-4)}


def audit(directory):
    groups = {}
    prior = {}
    by_shingle, by_fn = defaultdict(set), defaultdict(set)
    exclusions = []
    for split in ("test", "valid", "train"):
        rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
        kept = []
        for row in rows:
            parts = shingles(question_key(row))
            # Ignore shared template wording within the same synthetic task family; the source split already keeps families together.
            neighbors = set()
            if row["source"] not in ("logic", "service", "trace", "spec", "rules", "revision"):
                for s in parts:
                    neighbors.update(by_shingle[s])
            fn = function_name(row)
            duplicates = set(by_fn.get(fn, ())) if fn else set()
            for key in neighbors:
                old_parts = prior[key]
                if len(parts) >= 4 and len(parts & old_parts)/len(parts | old_parts) >= .68:
                    duplicates.add(key)
            if duplicates and split != "test":
                exclusions.append({"id": row["id"], "split": split, "matches": sorted(duplicates)})
                continue
            kept.append(row)
            key = row["id"]
            prior[key] = parts
            if fn:
                by_fn[fn].add(key)
            for s in parts:
                by_shingle[s].add(key)
        groups[split] = len(kept)
    return {"kept": groups, "excluded": exclusions, "criterion": "MBPP callable match or word five-gram Jaccard >=0.68"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=ROOT / "curriculum")
    args = parser.parse_args()
    result = audit(args.curriculum)
    write_json(args.curriculum / "overlap.json", result)
    print(result["kept"], "excluded", len(result["excluded"]))


if __name__ == "__main__":
    main()
