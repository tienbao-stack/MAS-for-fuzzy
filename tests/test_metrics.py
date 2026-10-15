import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_fuzzy.metrics import answer_comparison, normalize_unit, to_numeric, evaluate_pair
from multimodal_fuzzy.parsing import parse_model_output


class MetricsTests(unittest.TestCase):
    def test_fraction_and_decimal(self):
        self.assertAlmostEqual(to_numeric("30/47"), 30 / 47, places=10)
        self.assertAlmostEqual(to_numeric("1,25"), 1.25, places=10)

    def test_numeric_tolerance(self):
        self.assertTrue(answer_comparison("1.71", "65/38")["answer_correct"])
        self.assertFalse(answer_comparison("393", "24/61")["answer_correct"])

    def test_unit_normalization(self):
        self.assertEqual(normalize_unit(" Ω "), "Ω")
        self.assertEqual(normalize_unit("—"), "")
        self.assertEqual(normalize_unit("A"), "A")

    def test_parse_json(self):
        parsed = parse_model_output('{"answer":"30/47", "unit":"A", "explanation":"calculated"}')
        self.assertEqual(parsed["predicted_answer"], "30/47")
        self.assertEqual(parsed["predicted_unit"], "A")
        self.assertEqual(parsed["parse_status"], "json")

    def test_parse_fenced_json_and_attached_unit(self):
        parsed = parse_model_output('```json\n{"answer":"3.33 A", "unit":"", "explanation":"ok"}\n```')
        self.assertEqual(parsed["predicted_answer"], "3.33")
        self.assertEqual(parsed["predicted_unit"], "A")

    def test_plain_scalar_fallback(self):
        parsed = parse_model_output("3.33 A")
        self.assertEqual(parsed["predicted_answer"], "3.33")
        self.assertEqual(parsed["predicted_unit"], "A")

    def test_joint_metric(self):
        rec = {"sample_id": "x", "reference": {"answer": "2.5", "unit": "V"}}
        pred = {"sample_id": "x", "predicted_answer": "2.5", "predicted_unit": "V", "status": "ok"}
        result = evaluate_pair(pred, rec)
        self.assertTrue(result["answer_correct"])
        self.assertTrue(result["unit_correct"])
        self.assertTrue(result["joint_correct"])


if __name__ == "__main__":
    unittest.main()
