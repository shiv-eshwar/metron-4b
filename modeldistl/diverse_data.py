"""Extend supervised splits with harder mathematics, conversational reading, coding and deduction."""

import json
import random
import re
from collections import Counter
from fractions import Fraction

from .expanded_data import (
    EXCLUDE,
    ROOT,
    SEED,
    SOURCES,
    contests,
    order,
    records,
    safe,
    task,
)
from .files import write_json, write_jsonl
from .overlap import audit

SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
SHARDS = ("data/train-00001-of-00039-e092fe56fda18715.parquet", "data/train-00002-of-00039-9cea23812e920e41.parquet")


def numeric_box(solution):
    """Only accept an unambiguous integer, decimal or rational in the final balanced box."""
    prefix = r"\boxed{"
    p = solution.rfind(prefix)
    if p < 0:
        return None
    depth, start = 1, p + len(prefix)
    for end in range(start, len(solution)):
        depth += (solution[end] == "{") - (solution[end] == "}")
        if not depth:
            content = solution[start:end].strip().replace(r"\,", "").replace(" ", "")
            if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", content):
                content = content.replace(",", "")
            frac = re.fullmatch(r"(-?)\\(?:d?frac)\{(-?\d+)\}\{(\d+)\}", content)
            if frac and int(frac[3]):
                return str(Fraction((-1 if frac[1] else 1)*int(frac[2]), int(frac[3])))
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", content):
                return str(Fraction(content))
            return None
    return None


def olympiad():
    train, valid = [], []
    for subject in SUBJECTS:
        tasks = []
        for i, row in enumerate(records("competition_math", f"{subject}/train-00000-of-00001.parquet")):
            question = row["problem"].strip()
            target = numeric_box(row["solution"])
            if target is None or "[asy]" in question or len(question) > 1800 or EXCLUDE.search(question):
                continue
            prompt = (question + "\nGive a concise mathematical solution. Finish with 'Answer: <value>' using "
                      "an integer or a fraction a/b, without units or LaTeX on the final line.")
            t = task("competition_math", f"{subject}-{i}", "math", prompt, {"kind": "number", "answer": target})
            t["dedup_text"] = question
            tasks.append(t)
        tasks = order(tasks, "competition-math")
        valid.extend(tasks[:3])
        train.extend(tasks[3:115])
    return train, valid


def qualitative(split, count):
    tasks = []
    for r in records("quartz", f"data/{split}-00000-of-00001.parquet"):
        options = " / ".join(f"{k}. {v}" for k, v in zip(r["choices"]["label"], r["choices"]["text"], strict=True))
        prompt = (f"{r['para']}\n\n{r['question']}\nChoices: {options}\nExplain the relationship briefly, "
                  "then finish with 'Answer: <label>'.")
        t = task("quartz", r["id"], "reasoning", prompt, {"kind": "label", "answers": [r["answerKey"]]},
                 family="quartz-" + r["para_id"])
        t["dedup_text"] = r["question"] + " " + options
        tasks.append(t)
    return order(safe(tasks), "qualitative")[:count]


def rules(i, rng):
    bank = ["amber", "brisk", "calm", "tidy", "nimble", "tall", "curious", "swift", "patient", "warm",
            "gentle", "bright", "quiet", "steady", "alert", "bold", "silver", "blue", "merry", "kind"]
    a, b, c, d, e, f, g, h = rng.sample(bank, 8)
    name, other = rng.sample(["Ida", "Kai", "Noa", "Bea", "Sam", "Tess", "Vic", "Remy", "Lee", "Wren"], 2)
    mode = i % 3
    attrs = {a, c}
    implications = [({a}, b), ({b, c}, d), ({d}, e), ({f, g}, h)]
    changed = True
    while changed:
        old = len(attrs)
        attrs.update(z for needs, z in implications if needs <= attrs)
        changed = len(attrs) != old
    target = rng.choice(sorted(attrs - {a, c})) if mode == 0 else (f if mode == 1 else rng.choice([g, h]))
    answer = "entailed" if target in attrs else ("contradicted" if target == f else "unknown")
    rules_text = [f"Anyone who is {' and '.join(sorted(needs))} is {z}." for needs, z in implications]
    facts = [f"{name} is {a}.", f"{name} is {c}.", f"{name} is not {f}.", f"{other} is {g}."]
    rng.shuffle(rules_text)
    rng.shuffle(facts)
    prompt = ("Use only these facts and one-way rules. Missing facts are unknown. " + " ".join(rules_text + facts)
              + f" Is the claim '{name} is {target}' entailed, contradicted or unknown? "
                "Explain briefly and finish with 'Answer: entailed', 'Answer: contradicted', or 'Answer: unknown'.")
    return task("rules", i, "reasoning", prompt, {"kind": "label", "answers": [answer]}, family="rules-forward")


def synthesis(i, rng):
    mode, k = i % 8, rng.randint(2, 6)
    bound = rng.randint(-5, 11)
    arrays = [[], [bound], [bound]*k, list(range(-3, 5))]
    arrays += [[rng.randint(-8, 11) for _ in range(rng.randint(2, 13))] for _ in range(5)]
    if mode == 0:
        goal = f"Return the indices i for which the length-{k} window starting at i has sum greater than {bound}. Ignore incomplete windows."
        def solve(a):
            return [j for j in range(len(a)-k+1) if sum(a[j:j+k]) > bound]
    elif mode == 1:
        goal = f"Group the values by remainder modulo {k}. Return a list of {k} lists ordered by remainder, preserving input order inside each."
        def solve(a):
            return [[x for x in a if x % k == j] for j in range(k)]
    elif mode == 2:
        goal = f"Return how many pairs of indices i < j satisfy a[i] + a[j] == {bound}. Count duplicate occurrences separately."
        def solve(a):
            return sum(x+y == bound for j, x in enumerate(a) for y in a[j+1:])
    elif mode == 3:
        goal = f"Return the {k} distinct values nearest to {bound}, sorted by distance, then by smaller value. Return fewer if necessary."
        def solve(a):
            return sorted(set(a), key=lambda x: (abs(x-bound), x))[:k]
    elif mode == 4:
        goal = f"Split at values less than {bound}, discard separators, and return the nonempty runs with at least {k} elements."
        def solve(a):
            runs, run = [], []
            for x in a:
                if x < bound:
                    if len(run) >= k:
                        runs.append(run)
                    run = []
                else:
                    run.append(x)
            if len(run) >= k:
                runs.append(run)
            return runs
    elif mode == 5:
        goal = f"Starting with balance {bound}, add each value. Return the first zero-based index after which the balance is divisible by {k}; return -1 if none."
        def solve(a):
            bal = bound
            for j, x in enumerate(a):
                bal += x
                if bal % k == 0:
                    return j
            return -1
    elif mode == 6:
        goal = f"Partition into consecutive blocks of {k} entries, including a short last block. For each block return its largest value minus its smallest, in order."
        def solve(a):
            return [max(a[j:j+k])-min(a[j:j+k]) for j in range(0, len(a), k)]
    else:
        goal = f"Return the earliest starting index of a length-{k} strictly increasing consecutive run. Return -1 when there is no such run."
        def solve(a):
            for j in range(len(a)-k+1):
                if all(a[h]<a[h+1] for h in range(j, j+k-1)):
                    return j
            return -1
    # A random worked example plus edge cases separate the contract from a single example.
    sample = arrays[-1]
    prompt = (f"Implement solve(a) in Python for a list of integers. {goal} "
              f"Example: solve({sample}) returns {solve(sample)}. Return only the function in a Python code block.")
    return task("spec", i, "coding", prompt, {"kind": "code", "cases": [{"args": [a], "want": solve(a)} for a in arrays]},
                family=f"spec-{mode}")


def revised(i, rng):
    venue, alt = rng.sample(["Oak Hall", "River House", "Maple Room", "City Library", "Market Hall", "Union Center"], 2)
    d1, d2 = rng.sample(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], 2)
    old, new = (f"{h:02d}:{m:02d}" for h, m in rng.sample([(h, m) for h in range(8, 20) for m in (0, 15, 30, 45)], 2))
    role = rng.choice(["reading group", "gardening club", "choir", "makers group", "museum tour", "film club"])
    history = [{"role": "user", "content": f"Write a short reminder for the {role} at {venue} on {d1} at {old}."},
               {"role": "assistant", "content": f"The {role} meets at {venue} on {d1} at {old}."}]
    if i % 2:
        prompt = (f"One correction: move it to {alt}, {d2} at {new}. Keep the same group. Return only JSON "
                  "with keys group, venue, day, and time, preserving the names and 24-hour time exactly.")
        spec = {"kind": "json", "answer": {"group": role, "venue": alt, "day": d2, "time": new}}
    else:
        prompt = (f"The time has changed to {new}; keep the original day and venue. Write the revised reminder "
                  "in at most 30 words. Include the group, day, venue and new time. Omit the old time.")
        spec = {"kind": "rubric", "all": [re.escape(s) for s in (role, venue, d1, new)], "none": [re.escape(old)], "max_words": 30}
    t = task("revision", i, "general", prompt, spec, family="revision-public-reminder")
    t.update(history=history, dedup_text=history[0]["content"]+" "+prompt)
    return t


def wiki_title(title):
    return re.sub(r"[^a-z0-9]", "", title.casefold())


def conversations(base):
    # QuAC and SQuAD both draw on Wikipedia. Reserve whole titles, including across these two datasets.
    used = {wiki_title(t["family"].removeprefix("squad-")) for rows in base.values() for t in rows if t["source"] == "squad"}
    candidates = []
    for row in records("quac", "plain_text/train/0000.parquet"):
        title = row["wikipedia_page_title"]
        context = row["context"].removesuffix("CANNOTANSWER").strip()
        if wiki_title(title) in used or not 200 <= len(context) <= 2700 or EXCLUDE.search(context):
            continue
        questions, original = row["questions"], row["orig_answers"]["texts"]
        for turn in range(1, min(4, len(questions))):
            if original[turn-1] == "CANNOTANSWER" or EXCLUDE.search(" ".join(questions[:turn+1])):
                continue
            references = [a.strip() for a in row["answers"]["texts"][turn] if a.strip()]
            if not references or max(map(len, references)) > 300 or len(original[turn-1]) > 280:
                continue
            if "CANNOTANSWER" in references:
                references = ["UNANSWERABLE"]
            first = (f"Use this passage from the Wikipedia article on {title}, section {row['section_title']}:\n{context}\n\n"
                     f"{questions[turn-1]}")
            history = [{"role": "user", "content": first}, {"role": "assistant", "content": original[turn-1]}]
            prompt = (questions[turn] + "\nAnswer with a short quotation from the passage on one line. "
                      "If the passage does not answer this follow-up, reply UNANSWERABLE.")
            t = task("quac", row["turn_ids"][turn], "general", prompt,
                     {"kind": "grounded_span", "context": context, "answers": references}, family="quac-"+wiki_title(title))
            t.update(history=history, dedup_text=title + " " + questions[turn-1] + " " + questions[turn])
            candidates.append(t)
    selected = []
    titles = set()
    for t in order(candidates, "dialogues"):
        if t["family"] not in titles:
            selected.append(t)
            titles.add(t["family"])
        if len(selected) == 540:
            break
    return selected[20:], selected[:20]


def extend():
    base = {}
    for split in ("test", "valid", "train"):
        path = ROOT / "curriculum" / f"{split}.jsonl"
        base[split] = [r for line in path.read_text().splitlines() if (r := json.loads(line)).get("section") != "extension"]
    train, valid = olympiad()
    multi, multi_val = conversations(base)
    train.extend(multi)
    valid.extend(multi_val)
    train.extend(qualitative("train", 350))
    valid.extend(qualitative("validation", 16))
    rng = random.Random(SEED + 1)
    train.extend(rules(i, rng) for i in range(370))
    train.extend(revised(i, rng) for i in range(240))
    for shard in SHARDS:
        train.extend(contests(shard))
    train.extend(synthesis(i, rng) for i in range(432))
    # Reserve genuinely new coding questions for checkpoint selection as well.
    # The MBPP validation split remains available from the first phase.
    held = order([t for t in train if t["source"] == "contests"], "extra-code-valid")[:16]
    held_ids = {t["id"] for t in held}
    train = [t for t in train if t["id"] not in held_ids]
    valid.extend(held)
    used = {t["id"] for rows in base.values() for t in rows}
    used_families = {t["family"] for t in base["test"] + base["valid"]}
    for split, rows in (("valid", valid), ("train", train)):
        added = []
        seen_prompts = set()
        for t in order(rows, split+"-extension"):
            signature = json.dumps([t.get("history", []), t["prompt"]], sort_keys=True)
            if t["id"] in used or t["family"] in used_families or signature in seen_prompts:
                continue
            t["section"] = "extension"
            added.append(t)
            used.add(t["id"])
            seen_prompts.add(signature)
        if split == "valid":
            used_families.update(t["family"] for t in added)
        write_jsonl(ROOT / "curriculum" / f"{split}.jsonl", base[split] + added)
        print(split, "added", len(added), dict(Counter(t["source"] for t in added)), flush=True)
    dest = ROOT / "curriculum"
    write_json(dest / "sources.json", {"seed": SEED, "sources": SOURCES})
    result = audit(dest)
    write_json(dest / "overlap.json", result)
    print("After overlap audit", result["kept"], flush=True)


if __name__ == "__main__":
    extend()
