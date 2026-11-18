"""Run function tests with CPython inside a bounded, disposable Docker container."""

import json
import subprocess
import tempfile
import uuid

IMAGE = "python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89"
HARNESS = """
import contextlib, json, os, sys
job = json.load(sys.stdin)
scope = {"__name__": "submission"}
try:
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        exec(job["code"], scope)
        if "assertions" in job:
            for test in job["assertions"]:
                exec(test, scope)
            result = True
        else:
            result = [scope["solve"](*args) for args in job["args"]]
    print(json.dumps({"result": result}, allow_nan=False))
except BaseException as exc:
    print(json.dumps({"error": type(exc).__name__}))
"""


def check_code(code, spec):
    from .tasks import same_value

    name = f"modeldistl-check-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--pull=never", "-i", "--name", name, "--network", "none",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=12",
        "--memory=96m", "--memory-swap=96m", "--cpus=1", "--ulimit=cpu=2:3", "--user=65534:65534",
        IMAGE, "python", "-I", "-S", "-B", "-c", HARNESS,
    ]
    job = {"code": code}
    if "assertions" in spec:
        job["assertions"] = spec["assertions"]
    else:
        job["args"] = [case["args"] for case in spec["cases"]]
    with tempfile.TemporaryFile() as out:
        try:
            result = subprocess.run(command, input=json.dumps(job).encode(), stdout=out,
                                    stderr=subprocess.DEVNULL, timeout=12, check=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10, check=False)
            return False, "CPython wall timeout"
        if result.returncode == 125:
            raise RuntimeError(f"Start Docker and pull {IMAGE} to enable the CPython verifier.")
        if result.returncode or out.tell() > 64_000:
            return False, "CPython resource limit or process exit"
        out.seek(0)
        try:
            value = json.load(out)
        except (ValueError, UnicodeError):
            return False, "invalid execution result"
    if "error" in value:
        return False, value["error"]
    if "assertions" in spec:
        return value["result"] is True, "CPython assertions"
    expected = [case["want"] for case in spec["cases"]]
    return same_value(value["result"], expected), "CPython execution cases"
