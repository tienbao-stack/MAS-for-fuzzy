from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_prefix(sample_id: str) -> str:
    match = re.match(r"[A-Za-z]+", sample_id or "")
    return match.group(0) if match else "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a stratified human gold-label review queue.")
    ap.add_argument("--dataset", default=str(ROOT / "data/processed/physics_dataset_candidate.json"))
    ap.add_argument("--split-file", default=str(ROOT / "data/splits/split_assignments.json"))
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--output", default=str(ROOT / "data/audit/human_review_blind_sample_100.csv"))
    ap.add_argument("--gold-key-output", default=str(ROOT / "data/audit/human_review_gold_key_100.csv"))
    args = ap.parse_args()
    if args.sample_size < 3:
        raise SystemExit("sample-size must be at least 3 to cover all splits")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    assignments = json.loads(Path(args.split_file).read_text(encoding="utf-8"))["assignments"]
    by_split = collections.defaultdict(list)
    for rec in dataset["records"]:
        rec = dict(rec)
        sid = str(rec["sample_id"])
        rec["_split"] = assignments[sid]
        by_split[rec["_split"]].append(rec)

    split_order = ["train", "validation", "test"]
    base = args.sample_size // 3
    split_targets = {s: base for s in split_order}
    for s in split_order[:args.sample_size - base * 3]:
        split_targets[s] += 1
    rng = random.Random(args.seed)
    selected = []
    for split in split_order:
        groups = collections.defaultdict(list)
        for rec in by_split[split]:
            groups[get_prefix(str(rec["sample_id"]))].append(rec)
        n, total = split_targets[split], len(by_split[split])
        raw = {p: n * len(rows) / total for p, rows in groups.items()}
        alloc = {p: int(value) for p, value in raw.items()}
        left = n - sum(alloc.values())
        for p in sorted(groups, key=lambda p: (raw[p] - alloc[p], p), reverse=True)[:left]:
            alloc[p] += 1
        picked = 0
        for p, rows in groups.items():
            rng.shuffle(rows)
            take = min(alloc[p], len(rows))
            selected.extend(rows[:take])
            picked += take
        if picked < n:
            used = {r["sample_id"] for r in selected}
            remain = [r for r in by_split[split] if r["sample_id"] not in used]
            rng.shuffle(remain)
            selected.extend(remain[: n - picked])

    selected.sort(key=lambda r: (split_order.index(r["_split"]), str(r["sample_id"])))
    blind_fields = ["sample_id", "split", "id_prefix", "image_ref", "question", "independent_answer", "independent_unit", "image_matches_question", "reviewer_notes", "review_status"]
    key_fields = ["sample_id", "reference_answer", "reference_unit", "reference_solution_cot"]
    out = Path(args.output)
    key_out = Path(args.gold_key_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    key_out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f_blind, key_out.open("w", encoding="utf-8-sig", newline="") as f_key:
        blind_writer = csv.DictWriter(f_blind, fieldnames=blind_fields)
        key_writer = csv.DictWriter(f_key, fieldnames=key_fields)
        blind_writer.writeheader(); key_writer.writeheader()
        for rec in selected:
            blind_writer.writerow({
                "sample_id": rec.get("sample_id", ""),
                "split": rec["_split"],
                "id_prefix": get_prefix(str(rec.get("sample_id", ""))),
                "image_ref": rec.get("input", {}).get("image", {}).get("image_ref", ""),
                "question": rec.get("input", {}).get("question", ""),
                "independent_answer": "", "independent_unit": "", "image_matches_question": "",
                "reviewer_notes": "", "review_status": "not_reviewed",
            })
            key_writer.writerow({
                "sample_id": rec.get("sample_id", ""),
                "reference_answer": rec.get("reference", {}).get("answer", ""),
                "reference_unit": rec.get("reference", {}).get("unit", ""),
                "reference_solution_cot": rec.get("reference", {}).get("solution_cot", ""),
            })
    print(json.dumps({"sample_size": len(selected), "by_split": dict(collections.Counter(r["_split"] for r in selected)), "blind_output": str(out), "gold_key_output": str(key_out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
