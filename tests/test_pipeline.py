import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from pydantic_monty import Monty

from modeldistl import teacher
from modeldistl.benchmarks import matches_arc_option
from modeldistl.tasks import curriculum, grade
from scripts import model_file


class PipelineTests(unittest.TestCase):
    def test_arc_content_is_distinct_from_label_format(self):
        task = {"prompt": "Choose one.\nA. Hills\nB. Lakes", "check": {"kind": "choice", "answer": "A"}}
        self.assertFalse(grade(task, "A. Hills", None)[0])
        self.assertTrue(matches_arc_option(task, "A. Hills"))
        self.assertFalse(matches_arc_option(task, "B. Hills"))

    @unittest.skipUnless(os.environ.get("MODELDISTL_TEST_CPYTHON") == "1", "requires pinned Docker image")
    def test_cpython_generator_semantics(self):
        task = {"check": {"kind": "code", "cases": [{"args": [[3, 1]], "want": 2}, {"args": [[]], "want": 1}]}}
        code = "def solve(nums):\n    return next(i for i in range(1, len(nums)+2) if i not in nums)"
        self.assertTrue(grade(task, code, None, cpython=True)[0])
        self.assertFalse(grade(task, "def solve(nums):\n    return 0", None, cpython=True)[0])

    @unittest.skipUnless(find_spec("mlx") is not None, "MLX training extra")
    def test_completion_loss_excludes_prompt_and_all_padding(self):
        import mlx.core as mx

        from modeldistl.train import completion_loss

        # Real token positions are [0, 4) and [0, 3). Position 2 is the first
        # supervised token in both rows, so three predictions contribute.
        tokens = mx.array([[1, 2, 3, 4, 0, 0], [2, 1, 4, 0, 0, 0]])
        bounds = mx.array([[2, 4], [2, 3]])
        def logits(inputs):
            return mx.broadcast_to(mx.arange(5, dtype=mx.float32), (*inputs.shape, 5))
        loss, count = completion_loss(logits, tokens, bounds)
        changed = mx.array([[4, 0, 3, 4, 2, 1], [1, 2, 4, 2, 3, 4]])
        second, _ = completion_loss(logits, changed, bounds)
        self.assertEqual(count.item(), 3)
        self.assertAlmostEqual(loss.item(), second.item())

    def test_pilot_snapshot_matches_verified_prompts(self):
        splits = curriculum()
        provenance = json.loads(Path("runs/pilot/data/provenance.json").read_text())
        claims = {row["id"]: row for row in provenance["examples"]}
        with Monty(max_processes=1, request_timeout=3) as pool:
            for split in ("train", "valid"):
                chats = Path(f"runs/pilot/data/{split}.jsonl").read_text().splitlines()
                for task, line in zip(splits[split], chats, strict=True):
                    user, assistant = json.loads(line)["messages"]
                    self.assertEqual(user, {"role": "user", "content": task["prompt"]})
                    self.assertEqual(assistant["role"], "assistant")
                    self.assertTrue(grade(task, assistant["content"], pool)[0], task["id"])
                    self.assertEqual(claims[task["id"]]["request_sha256"], teacher.cache_dir(task).name)

    def test_model_parts_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(model_file, "PART_BYTES", 11):
            root = Path(tmp)
            model = root / "tiny.gguf"
            model.write_bytes(b"GGUF" + bytes(range(35)))
            parts = root / "parts"
            manifest = model_file.pack(model, parts, root / "manifest.json")
            output = root / "copy.gguf"
            model_file.assemble(manifest, parts, output, False)
            self.assertEqual(output.read_bytes(), model.read_bytes())
            (parts / manifest["parts"][0]["file"]).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "damaged"):
                model_file.assemble(manifest, parts, root / "broken.gguf", False)
            self.assertFalse((root / "broken.gguf").exists())

    def test_split_leakage(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            source = curriculum()
            source["train"][0]["family"] = source["test"][0]["family"]
            for name, rows in source.items():
                (root / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "leakage"):
                curriculum(root)

    def test_final_answer_only(self):
        task = {"check": {"kind": "number", "answer": "1/2"}}
        self.assertTrue(grade(task, "One of two cases.\nAnswer: 0.5", None)[0])
        self.assertFalse(grade(task, "The value is 1/2.\nAnswer: 2", None)[0])
        self.assertFalse(grade(task, "Answer: 1/0", None)[0])
        self.assertFalse(grade(task, "Answer: 1 or 1/2", None)[0])

    def test_json_types(self):
        task = {"check": {"kind": "json", "answer": {"count": 1}}}
        self.assertFalse(grade(task, '{"count": true}', None)[0])

    def test_isolated_code_cases(self):
        task = {"check": {"kind": "code", "cases": [{"args": [[3, 2]], "want": 5}]}}
        with Monty(max_processes=1, request_timeout=3) as pool:
            self.assertTrue(grade(task, "def solve(nums):\n    return sum(nums)", pool)[0])
            self.assertFalse(grade(task, "def solve(nums):\n    return 3", pool)[0])
            self.assertFalse(grade(task, "def solve(nums):\n    while True: pass", pool)[0])

    def test_teacher_cache_and_grader_are_independent(self):
        task = curriculum()["valid"][0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(teacher, "STATE", Path(tmp)):
            original = teacher.cache_dir(task)
            revised = {**task, "check": {"kind": "number", "answer": "99"}}
            self.assertEqual(original, teacher.cache_dir(revised))
            self.assertNotEqual(original, teacher.cache_dir({**task, "prompt": "A new question"}))
            request = teacher.request_for(task)
            self.assertEqual(request["input"], task["prompt"])
            self.assertFalse(any(k in request for k in ("check", "cases", "answer")))


if __name__ == "__main__":
    unittest.main()
