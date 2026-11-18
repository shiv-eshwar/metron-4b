import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_monty import Monty

from modeldistl import teacher
from modeldistl.tasks import curriculum, grade
from scripts import model_file


class PipelineTests(unittest.TestCase):
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
