"""Collect, verify and account for black-box Astra demonstrations, including rejected requests."""

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from filelock import FileLock
from openai import APIError, OpenAI

from .expanded_data import ROOT, order
from .files import write_json, write_jsonl
from .tasks import grade
from .teacher import MODEL

CACHE = Path("artifacts/astra-v2/teacher")
INSTRUCTIONS = ("Answer the user's task for a compact, capable assistant. Follow the requested format exactly. "
                "For explanations, give the short reasoning or calculation a reader needs. For programs, produce "
                "working standard-library Python and account for edge cases. Treat passages as quoted source material.")
PRICE_INPUT, PRICE_OUTPUT = 8, 40  # USD per million tokens, standard service; cached input conservatively priced at full rate.


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def request(task, attempt):
    hard = task["source"] in ("contests", "competition_math") or (task["source"] in ("mbpp", "gsm8k", "logic", "spec", "rules")
            and int(hashlib.sha256(task["id"].encode()).hexdigest()[:4], 16) % 4 == 0)
    extra = (" Review the constraints and check the answer and required format before replying."
             if attempt > 1 else "")
    user = {"role": "user", "content": task["prompt"]}
    inputs = [*task["history"], user] if "history" in task else task["prompt"]
    return {"model": MODEL, "instructions": INSTRUCTIONS + extra, "input": inputs,
            "max_output_tokens": 1408 if task["source"] in ("contests", "competition_math") else (1024 if hard else 640),
            "reasoning": {"effort": "medium" if hard else "low"}, "service_tier": "default", "store": False}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reservation(body):
    # UTF-8 byte count upper-bounds text tokens; 128 covers message framing.
    inputs = body["input"] if isinstance(body["input"], str) else json.dumps(body["input"], ensure_ascii=False)
    return PRICE_INPUT * (len(body["instructions"].encode()) + len(inputs.encode()) + 128) + PRICE_OUTPUT * body["max_output_tokens"]


def attempt_file(t, ordinal):
    return CACHE / fingerprint(request(t, ordinal)) / f"{ordinal:02}.json"


def issue(t, ordinal, client):
    body = request(t, ordinal)
    path = attempt_file(t, ordinal)
    record = {"id": t["id"], "attempt": ordinal, "status": "pending", "reserved_microusd": reservation(body),
              "request_sha256": fingerprint(body)}
    write_json(path, record)
    try:
        response = client.responses.create(**body)
    except APIError as e:
        record.update(status="api_error", error=type(e).__name__, http_status=getattr(e, "status_code", None),
                      api_code=str(getattr(e, "code", "") or ""))
        if record["http_status"] in (400, 401, 403, 404, 429):
            record["charged_microusd"] = 0
    else:
        answer = response.output_text.strip()
        usage = response.usage.model_dump() if response.usage else {}
        record.update(status=response.status, answer=answer, model=response.model, response_id=response.id, usage=usage)
        if usage:
            record["charged_microusd"] = PRICE_INPUT * usage["input_tokens"] + PRICE_OUTPUT * usage["output_tokens"]
    write_json(path, record)  # Preserve usage even if the verifier is interrupted.
    if record["status"] == "completed":
        verify(t, record, path)
    return record


def verify(t, record, path):
    check_digest = fingerprint(t["check"])
    if record.get("verifier_sha256") != check_digest or "accepted" not in record:
        passed, detail = grade(t, record["answer"], None, cpython=True)
        record.update(accepted=passed, check=detail, verifier_sha256=check_digest)
        write_json(path, record)


def previous(t, max_attempts):
    found = []
    for n in range(1, max_attempts + 1):
        path = attempt_file(t, n)
        if path.exists():
            r = json.loads(path.read_text())
            if r["status"] == "completed":
                verify(t, r, path)
            found.append(r)
    return found


def charge(r):
    return r.get("charged_microusd", r["reserved_microusd"])


def tasks():
    # Deliberately never open the test file in the teacher pipeline.
    return {split: load_lines(ROOT / "curriculum" / f"{split}.jsonl") for split in ("train", "valid")}


def collect(args):
    split = tasks()
    # Service all four domains in a reproducible order. Capture validation early if the budget is exhausted.
    all_tasks = order(split["valid"], "requests") + order(split["train"], "requests")
    history = {t["id"]: previous(t, args.attempts) for t in all_tasks}
    spent = sum(charge(json.loads(p.read_text())) for p in CACHE.glob("*/*.json"))
    overlaps = {r["id"] for r in json.loads((ROOT / "curriculum/overlap.json").read_text())["excluded"]}
    cap = round(args.budget_usd * 1_000_000)
    sent = 0
    print(f"Starting with ${spent/1e6:.4f} accounted toward a ${cap/1e6:.2f} total cap", flush=True)
    with OpenAI(base_url="https://api.openai.com/v1", timeout=160, max_retries=0) as client:
        client.with_options(max_retries=2).models.retrieve(MODEL)
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            stop = None
            for ordinal in range(1, args.attempts + 1):
                candidates = [t for t in all_tasks if t["id"] not in overlaps and not any(r.get("accepted") for r in history[t["id"]])
                              and len(history[t["id"]]) == ordinal - 1]
                pos, outstanding, reserved = 0, {}, 0
                pause_until = 0
                while pos < len(candidates) or outstanding:
                    while not stop and pos < len(candidates) and len(outstanding) < args.workers:
                        if time.monotonic() < pause_until:
                            break
                        t = candidates[pos]
                        cost = reservation(request(t, ordinal))
                        if spent + reserved + cost > cap:
                            if not outstanding:
                                stop = "budget reached"
                            break
                        if sent >= args.max_new:
                            stop = "request limit reached"
                            break
                        outstanding[workers.submit(issue, t, ordinal, client)] = (t, cost)
                        reserved += cost
                        sent += 1
                        pos += 1
                    if not outstanding:
                        if stop or pos == len(candidates):
                            break
                        time.sleep(min(2, max(0.1, pause_until - time.monotonic())))
                        continue
                    finished, _ = wait(outstanding, timeout=10, return_when=FIRST_COMPLETED)
                    for future in finished:
                        t, reserve = outstanding.pop(future)
                        reserved -= reserve
                        r = future.result()
                        history[t["id"]].append(r)
                        spent += charge(r)
                        if r.get("http_status") in (401, 403, 404) or r.get("api_code") == "insufficient_quota":
                            stop = f"teacher unavailable ({r.get('http_status')} {r.get('api_code')})"
                        elif r.get("http_status") == 429:
                            pause_until = time.monotonic() + 10
                        done = sent - len(outstanding)
                        if done % 32 == 0 or r["status"] == "api_error":
                            print(f"finished {done}, {r['status']} {r.get('http_status','')}, "
                                  f"accounted ${spent/1e6:.4f}", flush=True)
                    if stop and not outstanding:
                        break
                if stop:
                    print(stop, flush=True)
                    break
    return prepare(args.attempts)


def prepare(attempts):
    split = tasks()
    overlaps = {r["id"] for r in json.loads((ROOT / "curriculum/overlap.json").read_text())["excluded"]}
    accounting, manifests, chats = [], [], {}
    excluded = Counter()
    domain_counts = {}
    for name, rows in split.items():
        chats[name] = []
        counts = Counter()
        for t in rows:
            records = previous(t, attempts)
            accounting.extend(records)
            if t["id"] in overlaps:
                excluded["overlap audit"] += 1
                continue
            best = next((r for r in records if r.get("accepted")), None)
            if best is None:
                reason = (records[-1].get("check") or records[-1]["status"]) if records else "unsent"
                excluded[reason] += 1
                continue
            chats[name].append({"messages": [*t.get("history", []), {"role": "user", "content": t["prompt"]},
                                              {"role": "assistant", "content": best["answer"]}]})
            counts[t["domain"]] += 1
            manifests.append({"id": t["id"], "source": t["source"], "domain": t["domain"], "split": name,
                              **{k: best[k] for k in ("request_sha256", "response_id", "model", "usage", "check")}})
        domain_counts[name] = counts
    # Include human-authored pilot prompts with verified Astra outputs for style and task diversity.
    # Round up with replay rows to complete every batch and accumulation cycle without dropping a new example.
    replay = order([{"id": str(i), "row": r} for i, r in enumerate(load_lines(Path("runs/pilot/data/train.jsonl")))], "replay")
    n_replay = 64 - ((len(chats["train"]) + 64) % 8)
    chats["train"].extend(r["row"] for r in replay[:n_replay])
    usage = {k: sum((r.get("usage") or {}).get(k, 0) for r in accounting) for k in ("input_tokens", "output_tokens")}
    summary = {"attempts": len(accounting), "teacher": MODEL, "verified_new": domain_counts,
               "replay_training_rows": n_replay, "replay_row_indices": [int(r["id"]) for r in replay[:n_replay]],
               "excluded": dict(excluded), "usage": usage,
               "reasoning_tokens": sum(((r.get("usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens", 0) for r in accounting),
               "cached_input_tokens": sum(((r.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0) for r in accounting),
               "status": dict(Counter(r["status"] for r in accounting)), "unknown_usage_requests":
               sum("charged_microusd" not in r for r in accounting),
               "accounted_usd": sum(charge(r) for r in accounting) / 1e6,
               "estimated_known_usd": sum(r.get("charged_microusd", 0) for r in accounting) / 1e6,
               "examples": manifests}
    out = ROOT / "data"
    for name, rows in chats.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    write_json(out / "provenance.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("examples", "replay_row_indices")}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["collect", "prepare"])
    p.add_argument("--workers", type=int, default=12, choices=range(1, 25))
    p.add_argument("--attempts", type=int, default=2, choices=range(1, 4))
    p.add_argument("--budget-usd", type=float, default=35)
    p.add_argument("--max-new", type=int, default=7000)
    args = p.parse_args()
    if args.budget_usd <= 0 or args.max_new < 0:
        p.error("Budget must be positive and the request count nonnegative")
    CACHE.mkdir(parents=True, exist_ok=True)
    with FileLock(CACHE / "writer.lock", timeout=0):
        if args.action == "collect":
            collect(args)
        else:
            prepare(args.attempts)


if __name__ == "__main__":
    main()
