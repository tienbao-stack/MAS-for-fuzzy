import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized_question(text):
    return " ".join((text or "").split()).casefold()


class SplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        obj = json.loads((ROOT / "data/processed/physics_dataset_candidate.json").read_text(encoding="utf-8"))
        cls.records = obj["records"]
        cls.assignments = json.loads((ROOT / "data/splits/split_assignments.json").read_text(encoding="utf-8"))["assignments"]
        cls.image_dir = ROOT / "data/images"

    def test_every_record_has_one_split(self):
        ids = {r["sample_id"] for r in self.records}
        self.assertEqual(ids, set(self.assignments))
        self.assertEqual(len(ids), len(self.records))

    def test_no_identical_image_content_crosses_splits(self):
        groups = {}
        for rec in self.records:
            ref = rec["input"]["image"]["image_ref"]
            digest = hashlib.sha256((self.image_dir / ref).read_bytes()).hexdigest()
            groups.setdefault(digest, set()).add(self.assignments[rec["sample_id"]])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_no_exact_normalized_question_crosses_splits(self):
        groups = {}
        for rec in self.records:
            q = normalized_question(rec["input"]["question"])
            groups.setdefault(q, set()).add(self.assignments[rec["sample_id"]])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_quarantined_groups_absent_from_candidate(self):
        bad = {"180.png", "181.png", "187.png", "188.png"}
        self.assertFalse(any(r["input"]["image"]["image_ref"] in bad for r in self.records))


if __name__ == "__main__":
    unittest.main()
