from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # audit still runs structural checks without Pillow
    Image = None
    UnidentifiedImageError = Exception

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal_fuzzy.dataset import image_ref, load_dataset, question  # noqa: E402


def norm_question(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a cleaned Physics Text+Image QA dataset.")
    ap.add_argument("--dataset", default=str(ROOT / "data/processed/physics_dataset_clean_all.json"))
    ap.add_argument("--image-dir", default=str(ROOT / "data/images"))
    ap.add_argument("--split-file", default=str(ROOT / "data/audit/legacy_split_assignments_by_image_ref.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/audit"))
    args = ap.parse_args()

    dataset_path, image_dir, out_dir = Path(args.dataset), Path(args.image_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata, records = load_dataset(dataset_path)
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()]) if image_dir.exists() else []
    image_names = {p.name for p in image_files}
    hash_by_name: dict[str, str] = {}
    unreadable_images: list[str] = []
    if Image is not None:
        for p in image_files:
            try:
                with Image.open(p) as im:
                    im.verify()
                hash_by_name[p.name] = sha256_file(p)
            except (OSError, UnidentifiedImageError):
                unreadable_images.append(p.name)
    else:
        for p in image_files:
            hash_by_name[p.name] = sha256_file(p)

    required_missing = collections.Counter()
    missing_refs = []
    rows_by_id = {}
    questions = collections.defaultdict(list)
    images_to_records = collections.defaultdict(list)
    for rec in records:
        sid = str(rec.get("sample_id", ""))
        rows_by_id[sid] = rec
        q = question(rec)
        img = image_ref(rec)
        ref = rec.get("reference", {})
        fields = {
            "sample_id": sid,
            "question": q,
            "solution_cot": ref.get("solution_cot"),
            "answer": ref.get("answer"),
            "unit": ref.get("unit"),
            "image_ref": img,
        }
        for k, v in fields.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                required_missing[k] += 1
        if img and img not in image_names:
            missing_refs.append({"sample_id": sid, "image_ref": img})
        questions[norm_question(q)].append(sid)
        images_to_records[img].append(sid)

    duplicate_hash_groups = collections.defaultdict(list)
    for name, h in hash_by_name.items():
        duplicate_hash_groups[h].append(name)
    duplicate_hash_groups = {h: names for h, names in duplicate_hash_groups.items() if len(names) > 1}
    duplicate_question_groups = {q: ids for q, ids in questions.items() if q and len(ids) > 1}

    split_leakage = {"cross_split_image_hash_groups": None, "cross_split_exact_question_groups": None}
    if Path(args.split_file).exists():
        sdata = json.load(open(args.split_file, encoding="utf-8"))
        assigns = sdata.get("assignments", {})
        img_splits = collections.defaultdict(set)
        q_splits = collections.defaultdict(set)
        for rec in records:
            sid = str(rec.get("sample_id", ""))
            split = assigns.get(sid)
            if not split:
                continue
            img = image_ref(rec)
            if img in hash_by_name:
                img_splits[hash_by_name[img]].add(split)
            q_splits[norm_question(question(rec))].add(split)
        split_leakage = {
            "cross_split_image_hash_groups": sum(1 for vals in img_splits.values() if len(vals) > 1),
            "cross_split_exact_question_groups": sum(1 for q, vals in q_splits.items() if q and len(vals) > 1),
        }

    summary = {
        "audit_version": "phase1-readiness-0.2",
        "dataset_path": str(dataset_path),
        "dataset_name": metadata.get("dataset", "unknown"),
        "records": len(records),
        "unique_sample_ids": len(rows_by_id),
        "duplicate_sample_ids": len(records) - len(rows_by_id),
        "required_field_missing_counts": dict(required_missing),
        "image_files": len(image_files),
        "unique_image_refs_in_records": len(images_to_records),
        "records_with_missing_image_ref": len(missing_refs),
        "unreadable_image_files": unreadable_images,
        "unique_image_content_hashes_in_image_dir": len(set(hash_by_name.values())),
        "unique_image_content_hashes_referenced": len({hash_by_name[image_ref(r)] for r in records if image_ref(r) in hash_by_name}),
        "duplicate_image_content_groups": len(duplicate_hash_groups),
        "image_files_in_duplicate_content_groups": sum(len(v) for v in duplicate_hash_groups.values()),
        "duplicate_exact_question_groups": len(duplicate_question_groups),
        "records_in_duplicate_exact_question_groups": sum(len(v) for v in duplicate_question_groups.values()),
        "split_file": str(Path(args.split_file)),
        "split_leakage": split_leakage,
        "metadata_review_status_counts": dict(collections.Counter(
            str(rec.get("metadata", {}).get("review_status", "unspecified")) for rec in records
        )),
        "known_visual_mismatch_quarantine_refs": ["180.png", "181.png", "187.png", "188.png"],
        "benchmark_verdict": "pilot_candidate_only_not_yet_fully_validated",
        "limitations": [
            "Completeness and image decodability do not prove image-question-answer semantic alignment.",
            "All retained records remain unreviewed for gold-label correctness unless separately adjudicated.",
            "A manual visual inspection identified four image groups with apparent circuit-topology mismatch; these are quarantined from the Phase 2 candidate.",
            "The legacy split grouped by image filename, so identical image pixels and exact question text could cross splits.",
            "The dataset has no native gold labels for multiple interpretations, evidence conflicts, or context-aware selection; those research claims need curated annotations/controlled subsets.",
        ],
    }
    (out_dir / "phase1_readiness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "duplicate_image_content_groups.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "count", "image_refs"])
        for h, names in sorted(duplicate_hash_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            writer.writerow([h, len(names), ";".join(sorted(names))])

    with (out_dir / "duplicate_exact_questions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_question", "count", "sample_ids", "image_refs", "answers_with_units"])
        for q, ids in sorted(duplicate_question_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rs = [rows_by_id[sid] for sid in ids]
            writer.writerow([
                q, len(ids), ";".join(ids),
                ";".join(sorted(set(image_ref(r) for r in rs))),
                ";".join(f"{r.get('reference', {}).get('answer')} {r.get('reference', {}).get('unit')}" for r in rs),
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
