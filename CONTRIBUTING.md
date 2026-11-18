# Contributing to Metron

Use GitHub issues for reproducible bugs and proposed changes. For inference issues, include the release, quantization, runtime version, hardware and a minimal prompt. Remove credentials and personal data before posting.

For a code change, open a pull request describing the behavior, how to reproduce it and which checks you ran. From a clone with uv and Docker installed, run the same offline checks as CI.

```sh
uv sync --locked
docker pull python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89
MODELDISTL_TEST_CPYTHON=1 uv run --no-sync python -m unittest discover -s tests -v
```

For benchmark changes, record the model revision, prompt template, decoding settings and per-question scores. Keep published test questions separate from training and model selection. Include provenance and license information for proposed datasets.
