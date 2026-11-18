import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_monty import Monty

from modeldistl import teacher
from modeldistl.tasks import curriculum, grade


class PipelineTests(unittest.TestCase):
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
