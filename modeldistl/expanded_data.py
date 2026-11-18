"""Build the reproducible Astra curriculum and a sealed, source-separated evaluation set."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from functools import cache
from itertools import pairwise
from pathlib import Path

from huggingface_hub import hf_hub_download

from .benchmarks import mbpp as pilot_mbpp
from .files import write_json, write_jsonl

ROOT = Path("runs/astra-v2")
SEED = 621907
SOURCES = {
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "CC BY 4.0"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "MIT"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "CC BY-SA 4.0"),
    "squad": ("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412", "CC BY-SA 4.0"),
    "contests": ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6", "CC BY 4.0"),
    "competition_math": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3", "MIT"),
    "quartz": ("allenai/quartz", "28c1dbb56caf81799296cb17892fa73402e23464", "CC BY 4.0"),
    "quac": ("allenai/quac", "768addbf2099da8efabd67d3a669f6483b813c1b", "MIT dataset card; Wikipedia context attribution"),
}
# Keep teacher requests focused on ordinary language, arithmetic and benign programs.
EXCLUDE = re.compile(r"\b(?:explosiv\w*|weapon\w*|missile\w*|firearm\w*|gunpowder|bomb\w*|"
                     r"virus\w*|viral|pathogen\w*|plasmid\w*|transfect\w*|bioweapon\w*|malware|"
                     r"exploit\w*|hacking|phishing|payload\w*|rifle\w*|pistol\w*)\b", re.IGNORECASE)
DOMAINS = ("coding", "general", "math", "reasoning")


@cache
def records(source, filename, columns=None):
    import pyarrow.parquet as pq

    repo, revision, _ = SOURCES[source]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def order(rows, namespace):
    def rank(row):
        return hashlib.sha256(f"{SEED}/{namespace}/{row['id']}".encode()).digest()
    return sorted(rows, key=rank)


def task(source, key, domain, prompt, check, family=None):
    return {"id": f"{source}-{key}", "source": source, "family": family or f"{source}-{key}",
            "domain": domain, "prompt": prompt, "check": check}


def safe(rows):
    return [r for r in rows if not EXCLUDE.search(r["prompt"])]


def mbpp(split, full=False):
    corpus = {}
    if full:
        for r in records("mbpp", f"full/{split}-00000-of-00001.parquet"):
            corpus[r["task_id"]] = (r["text"], [r["test_setup_code"]] if r["test_setup_code"] else [],
                                     r["test_list"], r["challenge_test_list"])
    for r in records("mbpp", f"sanitized/{split}-00000-of-00001.parquet"):
        corpus[r["task_id"]] = (r["prompt"], r["test_imports"], r["test_list"], [])
    tasks = []
    for key, (question, setup, tests, extra) in corpus.items():
        if not tests:
            continue
        prompt = question.strip() + "\nExample: " + tests[0] + (
            "\nUse the function name in the example. Return only Python code in a single code block.")
        tasks.append(task("mbpp", key, "coding", prompt, {"kind": "code", "assertions": setup + tests + extra}))
    return safe(tasks)


def gsm(split):
    tasks = []
    for i, row in enumerate(records("gsm8k", f"main/{split}-00000-of-00001.parquet")):
        expected = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        prompt = row["question"] + '\nShow a brief calculation and end with "Answer: <number>" without units.'
        tasks.append(task("gsm8k", f"{split}-{i}", "math", prompt, {"kind": "number", "answer": expected}))
    return safe(tasks)


def science(split, training=False):
    results = []
    for i, r in enumerate(records("arc", f"ARC-Challenge/{split}-00000-of-00001.parquet")):
        choices = "\n".join(f"{k}. {v}" for k, v in zip(r["choices"]["label"], r["choices"]["text"], strict=True))
        explain = training and i % 2 == 0
        fmt = "Give a brief explanation and end with 'Answer: <label>'." if explain else "Reply with the correct choice label only."
        check = ({"kind": "label", "answers": [r["answerKey"]]} if explain else {"kind": "choice", "answer": r["answerKey"]})
        results.append(task("arc", f"{split}-{i}", "reasoning", f"{r['question']}\n{choices}\n{fmt}", check,
                            family="arc-" + r["id"]))
    return safe(results)


def reading(split):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records("squad", f"squad_v2/{split}-00000-of-00001.parquet"):
        if len(r["context"]) > 2200 or EXCLUDE.search(r["title"] + " " + r["context"] + " " + r["question"]):
            continue
        refs = list(dict.fromkeys(r["answers"]["text"]))
        if any(len(ref.split()) > 16 for ref in refs):
            continue
        prompt = (f"Passage:\n{r['context'].strip()}\n\nQuestion: {r['question'].strip()}\n"
                  "Reply with a short exact phrase from the passage. If the passage does not answer the question, reply UNANSWERABLE.")
        row = task("squad", r["id"], "general", prompt, {"kind": "span", "answers": refs or ["UNANSWERABLE"]},
                   family="squad-" + r["title"])
        grouped[r["title"]][bool(refs)].append(row)
    return grouped


def choose_reading(grouped, n, namespace, titles):
    # Cycle across articles and take exactly one quarter unanswerable questions.
    result = []
    for answerable, count in [(True, n - n // 4), (False, n // 4)]:
        buckets = [order(grouped[k][answerable], namespace) for k in sorted(titles)]
        candidates = [b[i] for i in range(max(map(len, buckets))) for b in buckets if i < len(b)]
        if len(candidates) < count:
            raise ValueError("Insufficient reading questions")
        result.extend(candidates[:count])
    return order(result, namespace)


def contests(filename="data/train-00000-of-00039-e991a271dbfa9925.parquet"):
    rows = records("contests", filename, ("name", "description", "public_tests", "private_tests", "cf_rating", "difficulty"))
    tasks = []
    for row in rows:
        text, pub, hidden = row["description"].strip(), row["public_tests"], row["private_tests"]
        if not 100 <= len(text) <= 2700 or not pub["input"] or not hidden["input"] or (
                row["cf_rating"] and row["cf_rating"] > 1700):
            continue
        if re.search(r"(?i)interactive|special judge|any (?:valid|correct) (?:answer|solution)|absolute or relative error", text):
            continue
        pairs = [(i, o) for i, o in zip(hidden["input"], hidden["output"], strict=True) if len(i) + len(o) < 4000]
        if len(pairs) < 2:
            continue
        cases = [(pub["input"][0], pub["output"][0])] + pairs[:3]
        if len(cases[0][0]) + len(cases[0][1]) > 800:
            continue
        prompt = (f"{text}\n\nExample input:\n{cases[0][0]}\nExample output:\n{cases[0][1]}\n"
                  "Write a complete Python 3 program using standard input and output. Return only one Python code block.")
        tasks.append(task("contests", row["name"], "coding", prompt,
                          {"kind": "code", "stdin": [i for i, _ in cases], "stdout": [o for _, o in cases]}))
    return safe(tasks)


# These programs let the verifier derive results on new inputs using the actual semantics of ordinary Python.
TRACE_PROGRAMS = (
    ("window", "return sum(max(a[i:i+3]) for i in range(len(a)-2))"),
    ("stride", "return sum(x for i, x in enumerate(a) if i % 3 == 1)"),
    ("runs", "return sum(i == 0 or a[i-1] != x for i, x in enumerate(a))"),
    ("prefix", "s = best = 0\nfor x in a:\n    s = max(0, s + x)\n    best = max(s, best)\nreturn best"),
    ("inversions", "return sum(x > y for i, x in enumerate(a) for y in a[i+1:])"),
    ("windowxor", "v = 0\nfor x in a[1::2]:\n    v ^= abs(x)\nreturn v"),
    ("stack", "s = []\nfor x in a:\n    if s and s[-1] == x:\n        s.pop()\n    else:\n        s.append(x)\nreturn len(s)"),
    ("updates", "b = a.copy()\nfor i in range(1, len(b)):\n    b[i] = (b[i] - b[i-1]) % 7\nreturn sum(b)"),
    ("unique", "return sum(x for x in set(a) if a.count(x) == 1)"),
    ("recurrence", "u, v = 1, 2\nfor x in a:\n    u, v = v, (u + v + x) % 11\nreturn v"),
    ("index", "return sum(a.index(x) for x in a if x % 2 == 0)"),
    ("pairs", "return sum((x+y) % 4 == 0 for i, x in enumerate(a) for y in a[:i])"),
)


def trace(i, rng):
    name, body = TRACE_PROGRAMS[i % len(TRACE_PROGRAMS)]
    a = [rng.randint(-3, 8) for _ in range(rng.randint(5, 10))]
    code = "def f(a):\n" + "\n".join("    " + line for line in body.splitlines())
    trusted = {}
    exec(code, trusted)  # noqa: S102 - fixed, project-authored function body only.
    prompt = (f"Read this Python function. What does f({a}) return? Explain the decisive steps briefly "
              f"and finish with 'Answer: <integer>'.\n\n```python\n{code}\n```")
    return task("trace", i, "coding", prompt, {"kind": "number", "answer": str(trusted["f"](a))}, family="trace-" + name)


def symbolic(i, rng):
    names = rng.sample(["Ada", "Bo", "Chen", "Dev", "Eli", "Faye", "Gia", "Hugo", "Inez", "Jae", "Kit", "Luz"], 6)
    a, b, c, d, e, f = names
    mode = i % 5
    if mode == 0:
        index = rng.randrange(1, 5)
        pairs = [f"{v} finishes immediately after {u}" for u, v in pairwise(names)]
        rng.shuffle(pairs)
        prompt = f"Six runners finish with no ties. {'; '.join(pairs)}. Who finishes in position {index+1}?"
        answer = names[index]
    elif mode == 1:
        edges = [(a, b), (b, c), (a, d), (d, c), (c, e), (e, f), (d, f)]
        times = [rng.randrange(2, 18) for _ in edges]
        x, y, z, w, p, q, t = times
        answer = str(min(x+y+p+q, z+w+p+q, z+t))
        prompt = ("One-way routes and journey times in minutes: " +
                  ", ".join(f"{u} to {v} {n}" for (u, v), n in zip(edges, times, strict=True)) +
                  f". What is the shortest travel time from {a} to {f}?")
    elif mode == 2:
        start, step = rng.randrange(1, 15), rng.randrange(2, 9)
        for _ in range(rng.randrange(1, 7)):
            names = names[1:] + names[:1]
        turns = rng.randrange(13, 45)
        answer = names[(start - 1 + (turns-1)*step) % 6]
        prompt = (f"Seats in clockwise order are {', '.join(names)}. Number positions starting at 1 and wrap "
                  f"after the sixth. The first turn goes to position {start}, wrapping if needed. Jump {step} "
                  f"positions clockwise between turns. Who gets turn {turns}?")
    elif mode == 3:
        x, y, overlap, neither = [rng.randint(2, 18) for _ in range(4)]
        answer = str(x + y)
        prompt = (f"In a survey of {x+y+overlap+neither} people, {x+overlap} use the tram, {y+overlap} cycle, "
                  f"and {neither} do neither. How many use exactly one of these modes?")
    else:
        size = rng.randint(4, 8)
        shifts = [rng.randrange(-4, 6) for _ in range(size)]
        end = rng.randrange(3, 15)
        answer = str(end - sum(shifts))
        prompt = (f"A counter finishes at {end}. In order its changes were {', '.join(f'{v:+}' for v in shifts)}. "
                  "What value did it start at?")
    return task("logic", i, "reasoning", prompt + "\nExplain briefly and finish with 'Answer: <value>'.",
                {"kind": "label", "answers": [answer]}, family=f"logic-{mode}")


def general(i, rng):
    names = rng.sample(["Linden", "Cedar", "Elm", "Birch", "Maple", "Hazel", "Oak", "Willow", "Olive", "Larch",
                        "Alder", "Pine", "Juniper", "Aspen", "Laurel", "Beech"], 3)
    site, other, backup = names
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
    hour = f"{rng.randrange(9, 19):02}:{rng.choice(['10','20','30','40','50'])}"
    ref = f"R-{rng.randrange(1000,9999)}"
    count, second = rng.randint(12, 48), rng.randint(2, 9)
    mode = i % 7
    if mode == 0:
        prompt = (f"Draft a friendly public update in at most 45 words. The {site} library will reopen {day} at "
                  f"{hour}. Returns can go to {other} until then. Include all three details.")
        facts = [site, day, hour, other, "return"]
    elif mode == 1:
        prompt = (f"Extract the confirmed booking as JSON with exactly the keys site, day, time, reference. "
                  f"Notes: {other} is only a backup. Confirmed {day} at {hour} at {site}, ID {ref}.")
        return task("service", i, "general", prompt, {"kind": "json", "answer":
                    {"site": site, "day": day, "time": hour, "reference": ref}}, family=f"service-{mode}")
    elif mode == 2:
        prompt = (f"A customer asks whether their {site} pickup can move to {day}. You know the order number "
                  f"is {ref}, but you have no access to availability. Draft a reply of at most 45 words. "
                  "Mention the requested day and number, ask for the preferred time, and avoid promising a change.")
        facts = [day, ref, "time", r"\?"]
    elif mode == 3:
        prompt = (f"Write a two-sentence handover under 50 words. At {site}, {count} tickets arrived and "
                  f"{second} are waiting for review. {other} will review those by {hour}. Include all quantities and ownership.")
        facts = [site, str(count), str(second), other, hour]
    elif mode == 4:
        prompt = (f"Make an accessible reminder under 45 words. Workshop at {site}, {day} at {hour}. "
                  f"The step-free entrance is on {other} Street. Include time, day and accessible entrance.")
        facts = [site, day, hour, other, "step.free|accessible|level"]
    elif mode == 5:
        prompt = (f"Return only JSON mapping each location to the number of open tickets. At {site}, "
                  f"{count} tickets were open and {second} closed. At {other}, {second} remain. "
                  f"{backup} had no tickets.")
        return task("service", i, "general", prompt, {"kind": "json", "answer":
                    {site: count-second, other: second, backup: 0}}, family=f"service-{mode}")
    else:
        prompt = (f"Explain the limitation of this feedback in 50 words or fewer. {count} volunteers from the "
                  f"{site} branch reviewed a proposed {day} timetable. A manager wants to generalize to every "
                  f"branch, including {other}. Mention the sample and a better way to collect evidence.")
        facts = [str(count), site, "volunt|self.select", "random|represent|sample.*branch|survey.*branch|across.*branch"]
    return task("service", i, "general", prompt, {"kind": "rubric", "all": facts, "max_words": 50 if mode in (3, 6) else 45},
                family=f"service-{mode}")


def build_test():
    old = json.loads(Path("runs/pilot/results/base-public.json").read_text())["results"]
    old_ids = {r["id"] for r in old}
    old_ids.update("mbpp-" + r["id"].split("-")[-1] for r in
                   json.loads(Path("runs/pilot/results/base-mbpp.json").read_text())["results"])
    grouped = reading("validation")
    test = choose_reading(grouped, 64, "sealed", grouped)
    for rows in (mbpp("test"), gsm("test"), science("test")):
        test.extend(order([r for r in rows if r["id"] not in old_ids], "sealed")[:64])
    assert Counter(r["domain"] for r in test) == dict.fromkeys(DOMAINS, 64)
    return order(test, "sealed-mix")


def question_key(t):
    if t.get("section") == "extension":
        return " ".join(re.findall(r"[a-z0-9]+", t.get("dedup_text", t["prompt"]).lower()))
    text = t["prompt"].split("\n", 1)[0]
    if t["source"] == "squad":
        text = t["prompt"].split("\n\nQuestion: ", 1)[1].split("\n", 1)[0]
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build():
    test = build_test()
    grouped = reading("train")
    titles = sorted(grouped)
    random.Random(SEED).shuffle(titles)
    train = choose_reading(grouped, 480, "supervised", titles[32:])
    valid = choose_reading(grouped, 32, "valid", titles[:32])
    rng = random.Random(SEED)
    train.extend(general(i, rng) for i in range(288))
    math = order(gsm("train"), "math")
    valid += math[:32]
    train += math[32:800]
    valid += order(science("validation"), "select")[:32]
    train += order(science("train", training=True), "science")[:512]
    train.extend(symbolic(i, rng) for i in range(256))
    pilot_val = {int(r["id"].split("-")[-1]) for r in pilot_mbpp("validation")}
    coding_val = [r for r in mbpp("validation", full=True) if int(r["id"].split("-")[1]) not in pilot_val]
    valid += order(coding_val, "select")[:32]
    coding = order(mbpp("train", full=True), "code")[:360] + order(contests(), "contests")[:192]
    coding += [trace(i, rng) for i in range(768 - len(coding))]
    train += coding
    # Keep the blind items and validation first. Reject duplicate questions and titles across splits.
    def dedup(rows, seen_ids, seen_q, prior_families):
        kept = []
        for t in rows:
            q = question_key(t)
            if t["id"] in seen_ids or q in seen_q or t["family"] in prior_families:
                continue
            seen_ids.add(t["id"])
            seen_q.add(q)
            kept.append(t)
        return kept

    ids, questions, families = set(), set(), set()
    result = {}
    for split, rows in (("test", test), ("valid", valid), ("train", train)):
        kept = dedup(order(rows, split), ids, questions, families)
        families.update(t["family"] for t in kept)
        result[split] = kept
    if len(result["test"]) != 256:
        raise ValueError("A blind benchmark item is duplicated")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "curriculum")
    parser.add_argument("--test-only", action="store_true")
    a = parser.parse_args()
    splits = {"test": build_test()} if a.test_only else build()
    for name, rows in splits.items():
        file = a.out / f"{name}.jsonl"
        write_jsonl(file, rows if name != "test" else order(rows, "test"))
        print(name, len(rows), dict(Counter(r["source"] for r in rows)), hashlib.sha256(file.read_bytes()).hexdigest(), flush=True)
    if not a.test_only:
        write_json(a.out / "sources.json", {"seed": SEED, "sources": SOURCES})


if __name__ == "__main__":
    main()
