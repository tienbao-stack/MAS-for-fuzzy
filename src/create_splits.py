from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

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


def build_components(records: list[dict[str, Any]], image_dir: Path) -> list[list[dict[str, Any]]]:
    """Keep duplicate image bytes and exact normalized questions in one split."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    first_seen: dict[tuple[str, str], int] = {}
    for i, rec in enumerate(records):
        ref = image_ref(rec)
        image_path = image_dir / ref
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {rec.get('sample_id')}: {image_path}")
        keys = [
            ("image_sha256", sha256_file(image_path)),
            ("normalized_question", norm_question(question(rec))),
        ]
        for key in keys:
            if key in first_seen:
                union(i, first_seen[key])
            else:
                first_seen[key] = i

    by_root: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for i, rec in enumerate(records):
        by_root[find(i)].append(rec)
    return list(by_root.values())


def make_assignment(records: list[dict[str, Any]], groups: list[list[dict[str, Any]]], seed: int) -> dict[str, str]:
    total = len(records)
    targets = {"train": total * 0.70, "validation": total * 0.15, "test": total * 0.15}
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    counts = {k: 0 for k in targets}
    assignment: dict[str, str] = {}
    for group in shuffled:
        open_splits = [k for k in targets if counts[k] < targets[k]] or list(targets)
        chosen = max(open_splits, key=lambda k: (targets[k] - counts[k]) / targets[k])
        counts[chosen] += len(group)
        for rec in group:
            assignment[str(rec["sample_id"])] = chosen
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser(description="Create leakage-aware grouped splits for Physics QA.")
    ap.add_argument("--dataset", default=str(ROOT / "data/processed/physics_dataset_candidate.json"))
    ap.add_argument("--image-dir", default=str(ROOT / "data/images"))
    ap.add_argument("--output", default=str(ROOT / "data/splits/split_assignments.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, records = load_dataset(args.dataset)
    groups = build_components(records, Path(args.image_dir))
    assignments = make_assignment(records, groups, args.seed)
    rec_by_id = {str(r["sample_id"]): r for r in records}
    content_splits: dict[str, set[str]] = collections.defaultdict(set)
    question_splits: dict[str, set[str]] = collections.defaultdict(set)
    image_group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for sid, split in assignments.items():
        rec = rec_by_id[sid]
        image_path = Path(args.image_dir) / image_ref(rec)
        content_splits[sha256_file(image_path)].add(split)
        question_splits[norm_question(question(rec))].add(split)
        image_group_splits[image_ref(rec)].add(split)

    content_leaks = sum(len(s) > 1 for s in content_splits.values())
    question_leaks = sum(len(s) > 1 for s in question_splits.values())
    if content_leaks or question_leaks:
        raise RuntimeError(f"Leakage guard failed: image-content groups={content_leaks}, exact questions={question_leaks}")

    counts = dict(collections.Counter(assignments.values()))
    group_count = {k: sum(1 for g in groups if assignments[str(g[0]["sample_id"])] == k) for k in ("train", "validation", "test")}
    payload = {
        "version": "content-and-question-grouped-v1",
        "seed": args.seed,
        "grouping_policy": ["SHA-256 of image bytes", "exact whitespace-normalized, case-folded question text"],
        "ratios_requested": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "records": len(records),
        "component_groups": len(groups),
        "counts": counts,
        "group_counts": group_count,
        "leakage_checks": {
            "image_content_hash_groups_crossing_splits": content_leaks,
            "exact_question_groups_crossing_splits": question_leaks,
        },
        "note": "Group sizes make the ratios approximate. This split removes exact image-content and exact-question overlap, but does not detect semantically near-duplicate question templates.",
        "assignments": assignments,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "assignments"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
