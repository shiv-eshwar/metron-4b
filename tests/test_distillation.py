import hashlib
import json
import os
import unittest
from collections import Counter
from importlib.util import find_spec

from modeldistl.comparison import exact_sign_p, paired
from modeldistl.distill import charge, request, reservation, tasks
from modeldistl.diverse_data import numeric_box
from modeldistl.expanded_data import ROOT
from modeldistl.overlap import audit
from modeldistl.retention import RECORD, select
from modeldistl.tasks import grade, span_score


class DistillationTests(unittest.TestCase):
    def test_retention_floors_and_fresh_holdout(self):
        stock = {"coding": 18, "math": 19, "strict": 70, "forced_total": 79}
        scores = {"base": stock, "one-sixteenth": stock | {"coding": 17, "math": 20, "strict": 75},
                  "one-quarter": stock | {"coding": 15, "math": 22, "strict": 80},
                  "one-half": stock | {"coding": 17}, "full": stock | {"math": 15, "strict": 84}}
        self.assertEqual(select(scores), "one-sixteenth")
        self.assertEqual(select({"base": stock, "full": stock}), "base")
        old = {json.loads(l)["id"] for l in (ROOT / "curriculum/test.jsonl").read_text().splitlines()}
        valid, test = ({json.loads(l)["id"] for l in (RECORD/f"{s}.jsonl").read_text().splitlines()}
                       for s in ("valid", "test"))
        self.assertEqual((len(valid), len(test)), (96, 128))
        self.assertTrue(old.isdisjoint(test) and valid.isdisjoint(test))

    def test_sealed_set_and_overlap_audit(self):
        file = ROOT / "curriculum/test.jsonl"
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),
                         "0c153bcfd1012a828f1ff522685254582070f7548d470b211d638d54d5e6ae50")
        held = [json.loads(line) for line in file.read_text().splitlines()]
        self.assertEqual(set(Counter(row["domain"] for row in held).values()), {64})
        self.assertEqual(len(held), 256)
        all_supervised = [t for rows in tasks().values() for t in rows]
        self.assertTrue({r["id"] for r in held}.isdisjoint(r["id"] for r in all_supervised))
        self.assertEqual(audit(file.parent), json.loads((file.parent / "overlap.json").read_text()))

    def test_reservations_and_prompt_boundaries(self):
        task = {"id": "math-4", "source": "gsm8k", "prompt": "How much is 7 × 11?", "check": {"answer": "77"}}
        body = request(task, 1)
        self.assertEqual(body["input"], task["prompt"])
        self.assertNotIn("77", json.dumps(body))
        self.assertGreater(reservation(body), 40 * body["max_output_tokens"])
        self.assertEqual(charge({"reserved_microusd": 200, "charged_microusd": 0}), 0)
        self.assertEqual(charge({"reserved_microusd": 200}), 200)

    def test_reading_scores(self):
        self.assertEqual(span_score("Answer: The River Thames.", ["River Thames"]), (True, 1.0))
        self.assertEqual(span_score("Thames", ["River Thames"]), (False, 2/3))
        self.assertEqual(span_score("UNANSWERABLE", ["UNANSWERABLE"]), (True, 1.0))
        self.assertFalse(grade({"check": {"kind": "span", "answers": ["blue"]}}, "", None)[0])

    def test_multiturn_and_numeric_references(self):
        self.assertEqual(numeric_box(r"Thus we have \boxed{-\frac{6}{8}}."), "-3/4")
        self.assertIsNone(numeric_box(r"Solutions are \boxed{1,2}."))
        prior = [{"role": "user", "content": "My train leaves at six."}, {"role": "assistant", "content": "At six."}]
        task = {"id": "follow", "source": "quac", "prompt": "When does it leave?", "history": prior}
        self.assertEqual(request(task, 1)["input"], [*prior, {"role": "user", "content": task["prompt"]}])
        self.assertGreater(reservation(request(task, 1)), 0)
        rubric = {"check": {"kind": "rubric", "all": ["six"], "none": ["five"], "max_words": 5}}
        self.assertTrue(grade(rubric, "At six.", None)[0])
        self.assertFalse(grade(rubric, "Five or six.", None)[0])

    @unittest.skipUnless(find_spec("torch") is not None, "requires the export extra")
    def test_portable_fusion_uses_mlx_orientation_and_rounding(self):
        import torch

        from modeldistl.fusion import merge_linear

        base = torch.ones((3, 2), dtype=torch.bfloat16)
        a = torch.tensor([[1., -1.], [2., 3.]])
        b = torch.tensor([[2., 3., -4.], [1., 0., 5.]])
        expected = base + ((2 * b.T) @ a.T).to(torch.bfloat16)
        self.assertTrue(torch.equal(merge_linear(base, a, b, 2), expected))

    def test_paired_stats_require_matched_runs(self):
        rows = [{"id": str(i), "domain": "math", "pass": i < 4} for i in range(8)]
        report = dict.fromkeys(("test_sha256", "temperature", "seed", "max_tokens", "reasoning", "suite", "jobs", "code_runtime"), 1)
        report.update(total=8, complete=True, results=rows)
        other = dict(report, results=[dict(r, **{"pass": False}) for r in rows])
        metrics = paired(report, other, 100)
        self.assertEqual((metrics["student_only"], metrics["baseline_only"], metrics["difference_pp"]), (4, 0, 50))
        self.assertEqual(exact_sign_p(4, 0), 0.125)
        with self.assertRaisesRegex(ValueError, "settings"):
            paired(report, other | {"max_tokens": 99})

    @unittest.skipUnless(os.environ.get("MODELDISTL_TEST_CPYTHON") == "1", "requires pinned Docker image")
    def test_stdio_program_in_fresh_cpython(self):
        task = {"check": {"kind": "code", "stdin": ["2\n", "4\n"], "stdout": ["4\n", "8\n"]}}
        self.assertTrue(grade(task, "print(2 * int(input()))", None, cpython=True)[0])
        self.assertFalse(grade(task, "print(4)", None, cpython=True)[0])
