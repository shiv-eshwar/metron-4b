"""Run the fixed test prompts on the MLX base, an adapter, or a local GGUF."""

import argparse
import hashlib
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic_monty import Monty

from .files import write_json
from .tasks import curriculum, grade, span_score
from .weights import QUANT, REVISION

MAX_TOKENS = 320


def mlx_inference(adapter):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(42)
    model, tok = load(str(QUANT), adapter_path=str(adapter) if adapter else None)
    sampler = make_sampler(temp=0)

    def infer(prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=True, return_dict=False)
        pieces = []
        for chunk in stream_generate(model, tok, ids, max_tokens=MAX_TOKENS, sampler=sampler):
            pieces.append(chunk.text)
        return "".join(pieces), {"tokens": chunk.generation_tokens, "finish": chunk.finish_reason,
                                 "generation_tokens_per_sec": round(chunk.generation_tps, 2)}

    return infer


@contextmanager
def gguf_inference(model, binary, log, jobs=1, max_tokens=MAX_TOKENS, reasoning="auto"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(binary), "-m", str(model), "-c", str(2048 * jobs), "-np", str(jobs), "--jinja",
               "--host", "127.0.0.1", "--port", str(port), "--alias", "modeldistl"]
    if reasoning != "auto":
        command.extend(["--reasoning", reasoning])
    with log.open("w") as f, subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT) as server:
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited; see {log}")
                try:
                    with urlopen(base + "/health", timeout=1) as r:
                        if json.load(r).get("status") == "ok":
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.3)
            else:
                raise TimeoutError(f"llama-server startup timed out; see {log}")

            def infer(prompt):
                body = {"model": "modeldistl", "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "seed": 42, "max_tokens": max_tokens}
                req = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=300) as r:
                    data = json.load(r)
                choice = data["choices"][0]
                return choice["message"]["content"] or "", {"tokens": data["usage"]["completion_tokens"],
                                                      "finish": choice["finish_reason"]}
            yield infer
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def run(infer, tasks, report, out):
    results = report.setdefault("results", [])
    done = {r["id"] for r in results}
    pending = [t for t in tasks if t["id"] not in done]
    def predict(task):
        start = time.monotonic()
        answer, stats = infer(task["prompt"])
        return task, answer, stats, time.monotonic() - start

    with ExitStack() as stack:
        pool = stack.enter_context(Monty(max_processes=1, request_timeout=3))
        if report["jobs"] == 1:
            predictions = map(predict, pending)
        else:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=report["jobs"]))
            predictions = executor.map(predict, pending)
        for t, answer, stats, seconds in predictions:
            passed, detail = grade(t, answer, pool, cpython=report["code_runtime"] == "cpython")
            results.append({"id": t["id"], "domain": t["domain"], "pass": passed, "check": detail,
                            "answer": answer, "seconds": round(seconds, 3), **stats})
            if t["domain"] == "arc" or (t.get("source") == "arc" and t["check"]["kind"] == "choice"):
                from .benchmarks import matches_arc_option

                results[-1]["choice_content_pass"] = matches_arc_option(t, answer)
            if t["check"]["kind"] == "span":
                results[-1]["span_f1"] = span_score(answer, t["check"]["answers"])[1]
            print(f"{len(results)}/{len(tasks)} {t['id']} {'pass' if passed else 'fail'}", flush=True)
            write_json(out, report)
    report["domains"] = {}
    for domain in sorted({t["domain"] for t in tasks}):
        rows = [r for r in results if r["domain"] == domain]
        report["domains"][domain] = {"correct": sum(r["pass"] for r in rows), "total": len(rows)}
    report.update(complete=True, correct=sum(r["pass"] for r in results), total=len(results))
    choices = [r["choice_content_pass"] for r in results if "choice_content_pass" in r]
    if choices:
        report["arc_choice_content"] = {"correct": sum(choices), "total": len(choices)}
    write_json(out, report)
    print(f"{report['correct']}/{report['total']} passed. {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--adapter", type=Path, help="Selected MLX adapter directory")
    backend.add_argument("--gguf", type=Path, help="Single GGUF or first shard")
    parser.add_argument("--server-bin", type=Path, default=Path(".tools/llama-bin/llama-server"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, default=Path("curriculum"))
    parser.add_argument("--suite", choices=["local", "public", "mbpp", "sealed"], default="local")
    parser.add_argument("--test-file", type=Path, default=Path("runs/astra-v2/curriculum/test.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--reasoning", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--source-revision", default=REVISION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-runtime", choices=["monty", "cpython"], default="monty")
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=1, help="Concurrent 2,048-token GGUF slots")
    args = parser.parse_args()
    if not args.gguf and (args.jobs != 1 or args.max_tokens != MAX_TOKENS or args.reasoning != "auto"):
        parser.error("Custom parallelism, generation limits and reasoning mode require a GGUF server")
    if not 1 <= args.max_tokens < 2048:
        parser.error("The output budget must fit the inference context")
    if args.suite == "sealed":
        tasks = [json.loads(line) for line in args.test_file.read_text().splitlines()]
        args.code_runtime = "cpython"
    elif args.suite == "mbpp":
        from .benchmarks import mbpp

        args.code_runtime = "cpython"
        tasks = mbpp("test")
    elif args.suite == "public":
        from .benchmarks import public_tasks

        tasks = public_tasks()
    else:
        tasks = curriculum(args.curriculum)["test"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"backend": "llama.cpp" if args.gguf else "mlx", "source_revision": args.source_revision,
              "model": str(args.gguf or QUANT), "adapter": str(args.adapter) if args.adapter else None,
              "temperature": 0, "seed": 42, "max_tokens": args.max_tokens, "reasoning": args.reasoning, "complete": False,
              "suite": args.suite, "code_runtime": args.code_runtime, "jobs": args.jobs,
              "test_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()}
    if args.gguf:
        with args.gguf.open("rb") as f:
            report["gguf_file_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        if args.resume and args.out.exists():
            old = json.loads(args.out.read_text())
            if any(old.get(k) != v for k, v in report.items() if k != "complete"):
                raise ValueError("Existing report has different inputs or inference settings")
            if old.get("complete"):
                print(f"Already evaluated {args.out}")
                return
            report["results"] = old["results"]
        with gguf_inference(args.gguf, args.server_bin, args.out.with_suffix(".server.log"), args.jobs,
                            args.max_tokens, args.reasoning) as infer:
            run(infer, tasks, report, args.out)
    else:
        if args.adapter:
            report["adapter_sha256"] = hashlib.sha256((args.adapter / "adapters.safetensors").read_bytes()).hexdigest()
        run(mlx_inference(args.adapter), tasks, report, args.out)


if __name__ == "__main__":
    main()
