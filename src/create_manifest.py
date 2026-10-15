from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    candidate_path = ROOT / "data/processed/physics_dataset_candidate.json"
    clean_path = ROOT / "data/processed/physics_dataset_clean_all.json"
    split_path = ROOT / "data/splits/split_assignments.json"
    config_path = ROOT / "configs/baseline.yaml"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    split_obj = json.loads(split_path.read_text(encoding="utf-8"))
    records = candidate["records"]
    ref_counts = collections.Counter(r["input"]["image"]["image_ref"] for r in records)
    image_rows = []
    content_hashes = set()
    image_dir = ROOT / "data/images"
    for ref, count in sorted(ref_counts.items()):
        path = image_dir / ref
        row = {"image_ref": ref, "records_referencing_image": count, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        content_hashes.add(row["sha256"])
        if Image is not None:
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
                row["format"] = img.format
        image_rows.append(row)
    with (ROOT / "data/audit/image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader(); writer.writerows(image_rows)

    manifest = {
        "manifest_version": "phase2-manifest-v1",
        "scope": "Physics Text + Image; no audio/video; ManyModalQA not used",
        "candidate_dataset": {
            "path": "data/processed/physics_dataset_candidate.json",
            "records": len(records),
            "sha256": sha256(candidate_path),
        },
        "phase1_clean_dataset": {
            "path": "data/processed/physics_dataset_clean_all.json",
            "records": len(json.loads(clean_path.read_text(encoding="utf-8"))["records"]),
            "sha256": sha256(clean_path),
        },
        "split_assignments": {
            "path": "data/splits/split_assignments.json",
            "sha256": sha256(split_path),
            "seed": split_obj["seed"],
            "counts": split_obj["counts"],
            "component_groups": split_obj["component_groups"],
            "leakage_checks": split_obj["leakage_checks"],
        },
        "baseline_config": {"path": "configs/baseline.yaml", "sha256": sha256(config_path)},
        "images": {
            "referenced_unique_files": len(image_rows),
            "unique_sha256_contents_referenced": len(content_hashes),
            "manifest_path": "data/audit/image_manifest.csv",
            "all_files_in_data_images": len([p for p in image_dir.iterdir() if p.is_file()]),
        },
        "known_quarantine": json.loads((ROOT / "data/audit/quarantine_image_refs.json").read_text(encoding="utf-8"))["image_refs"],
        "benchmark_status": "provisional_pilot_candidate_not_fully_human_validated",
        "created_by": "src/create_manifest.py",
    }
    out = ROOT / "data/audit/benchmark_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
