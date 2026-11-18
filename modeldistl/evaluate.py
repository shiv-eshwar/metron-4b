"""Run the fixed test prompts on the MLX base, an adapter, or a local GGUF."""

import argparse
import hashlib
import json
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic_monty import Monty

from .files import write_json
from .tasks import curriculum, grade
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
def gguf_inference(model, binary, log):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(binary), "-m", str(model), "-c", "2048", "-np", "1", "--jinja",
               "--host", "127.0.0.1", "--port", str(port), "--alias", "modeldistl"]
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
                        "temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS}
                req = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=120) as r:
                    data = json.load(r)
                choice = data["choices"][0]
                return choice["message"]["content"], {"tokens": data["usage"]["completion_tokens"],
                                                      "finish": choice["finish_reason"]}
            yield infer
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def run(infer, tasks, report, out):
    results = report["results"] = []
    with Monty(max_processes=1, request_timeout=3) as pool:
        for t in tasks:
            start = time.monotonic()
            answer, stats = infer(t["prompt"])
            seconds = time.monotonic() - start
            passed, detail = grade(t, answer, pool)
            results.append({"id": t["id"], "domain": t["domain"], "pass": passed, "check": detail,
                            "answer": answer, "seconds": round(seconds, 3), **stats})
            print(f"{len(results)}/{len(tasks)} {t['id']} {'pass' if passed else 'fail'}", flush=True)
            write_json(out, report)
    report["domains"] = {}
    for domain in sorted({t["domain"] for t in tasks}):
        rows = [r for r in results if r["domain"] == domain]
        report["domains"][domain] = {"correct": sum(r["pass"] for r in rows), "total": len(rows)}
    report.update(complete=True, correct=sum(r["pass"] for r in results), total=len(results))
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
    parser.add_argument("--suite", choices=["local", "public"], default="local")
    args = parser.parse_args()
    if args.suite == "public":
        from .benchmarks import public_tasks

        tasks = public_tasks()
    else:
        tasks = curriculum(args.curriculum)["test"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"backend": "llama.cpp" if args.gguf else "mlx", "source_revision": REVISION,
              "model": str(args.gguf or QUANT), "adapter": str(args.adapter) if args.adapter else None,
              "temperature": 0, "seed": 42, "max_tokens": MAX_TOKENS, "complete": False,
              "suite": args.suite,
              "test_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()}
    if args.gguf:
        with args.gguf.open("rb") as f:
            report["gguf_file_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        with gguf_inference(args.gguf, args.server_bin, args.out.with_suffix(".server.log")) as infer:
            run(infer, tasks, report, args.out)
    else:
        if args.adapter:
            report["adapter_sha256"] = hashlib.sha256((args.adapter / "adapters.safetensors").read_bytes()).hexdigest()
        run(mlx_inference(args.adapter), tasks, report, args.out)


if __name__ == "__main__":
    main()
