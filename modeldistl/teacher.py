"""Generate bounded Astra demonstrations, or rebuild MLX data from the local response cache."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from filelock import FileLock
from openai import APIError, OpenAI
from pydantic_monty import Monty

from .files import write_json, write_jsonl
from .tasks import curriculum, grade

MODEL = "gpt-6.2-astra"
STATE = Path("artifacts/teacher")
DATA = Path("artifacts/data")
INSTRUCTIONS = (
    "You are writing a demonstration for an on-device assistant. Answer the user's task directly, "
    "accurately and concisely. Follow its output format exactly. For a requested justification, "
    "give a short explanation suitable for the reader. For code, provide a complete working function."
)


def request_for(task):
    return dict(model=MODEL, instructions=INSTRUCTIONS, input=task["prompt"],
                reasoning={"effort": "low"}, service_tier="default", max_output_tokens=1024, store=False)


def cache_dir(task):
    payload = json.dumps(request_for(task), sort_keys=True, ensure_ascii=False).encode()
    return STATE / hashlib.sha256(payload).hexdigest()


def records(task):
    return [json.loads(f.read_text()) for f in sorted(cache_dir(task).glob("*.json"))]


def accepted(task, attempts, pool):
    for attempt in reversed(attempts):
        if attempt["status"] == "completed" and grade(task, attempt["answer"], pool)[0]:
            return attempt
    return None


def generate(task, client):
    directory = cache_dir(task)
    count = len(list(directory.glob("*.json")))
    path = directory / f"{count + 1:06}.json"
    record = {"task_id": task["id"], "status": "pending", "teacher": MODEL}
    # A crash or ambiguous timeout leaves an explicit attempt. Another run must
    # opt in before retrying a request that might have been billed.
    write_json(path, record)
    try:
        response = client.responses.create(**request_for(task))
    except APIError as error:
        record.update(status="api_error", error=type(error).__name__,
                      http_status=getattr(error, "status_code", None), code=getattr(error, "code", None))
    else:
        record.update(status=response.status, resolved_model=response.model, response_id=response.id,
                      answer=response.output_text.strip(), usage=response.usage.model_dump() if response.usage else {})
    write_json(path, record)
    return record


def generate_missing(tasks, args, pool):
    pending = []
    for task in tasks:
        attempts = records(task)
        if not accepted(task, attempts, pool) and (not attempts or args.retry_failed):
            pending.append(task)
    pending = pending[:args.max_new]
    if not pending:
        print("0 new requests")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in this process before generating teacher answers.")
    print(f"{len(pending)} requests scheduled, at most {1024 * len(pending):,} output tokens", flush=True)
    # Resolve project-specific access before any paid call. API exceptions can
    # contain credential fragments, so only print a class and status on failure.
    with OpenAI(base_url="https://api.openai.com/v1", max_retries=0, timeout=90) as client:
        try:
            client.models.retrieve(MODEL)
        except APIError as error:
            raise SystemExit(f"Teacher access failed ({type(error).__name__}, "
                             f"HTTP {getattr(error, 'status_code', None)})") from None
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate, task, client): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                verdict = result["status"]
                if verdict == "completed":
                    passed, detail = grade(task, result["answer"], pool)
                    verdict = "accepted" if passed else f"rejected ({detail})"
                elif verdict == "api_error":
                    verdict += f" ({result['http_status']} {result['code']})"
                print(f"{task['id']} {verdict}", flush=True)


def prepare(splits, pool):
    data = {}
    manifest = []
    missing = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    calls = unknown_usage = 0
    for split in ("train", "valid"):
        data[split] = []
        for task in splits[split]:
            attempts = records(task)
            calls += len(attempts)
            for attempt in attempts:
                billed = attempt.get("usage")
                if not billed:
                    unknown_usage += 1
                    continue
                for key in ("input_tokens", "output_tokens"):
                    usage[key] += billed[key]
                usage["cached_input_tokens"] += (billed.get("input_tokens_details") or {}).get("cached_tokens", 0)
            best = accepted(task, attempts, pool)
            if best is None:
                missing.append(task["id"])
                continue
            row = {"messages": [{"role": "user", "content": task["prompt"]},
                                {"role": "assistant", "content": best["answer"]}]}
            data[split].append(row)
            manifest.append({"id": task["id"], "domain": task["domain"], "split": split,
                             "request_sha256": cache_dir(task).name, "response_id": best["response_id"],
                             "model": best["resolved_model"], "usage": best["usage"]})
    if missing:
        raise SystemExit(f"{len(missing)} tasks have no verified answer: " + ", ".join(missing))
    for split, rows in data.items():
        write_jsonl(DATA / f"{split}.jsonl", rows)
    write_json(DATA / "manifest.json", {"model": MODEL, "attempts": calls, "unknown_usage": unknown_usage,
                                       "usage": usage, "examples": manifest})
    print(f"Prepared {len(data['train'])} train and {len(data['valid'])} validation rows. {usage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "prepare"])
    parser.add_argument("--max-new", type=int, default=76, help="Maximum paid requests in this invocation")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Allow one further attempt per missing or rejected example within the cap")
    parser.add_argument("--curriculum", type=Path, default=Path("curriculum"))
    args = parser.parse_args()
    if args.max_new < 0:
        parser.error("--max-new must be nonnegative")
    splits = curriculum(args.curriculum)
    STATE.mkdir(parents=True, exist_ok=True)
    with FileLock(STATE / "writer.lock", timeout=0), Monty(max_processes=1, request_timeout=3) as pool:
        if args.action == "generate":
            generate_missing(splits["train"] + splits["valid"], args, pool)
        else:
            prepare(splits, pool)


if __name__ == "__main__":
    main()
