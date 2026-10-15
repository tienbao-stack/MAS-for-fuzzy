from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a provisional Phase 2 candidate without known mismatched image groups.")
    ap.add_argument("--input", default=str(ROOT / "data/processed/physics_dataset_clean_all.json"))
    ap.add_argument("--quarantine", default=str(ROOT / "data/audit/quarantine_image_refs.json"))
    ap.add_argument("--output", default=str(ROOT / "data/processed/physics_dataset_candidate.json"))
    ap.add_argument("--jsonl", default=str(ROOT / "data/processed/physics_dataset_candidate.jsonl"))
    ap.add_argument("--review-csv", default=str(ROOT / "data/audit/manual_review_queue.csv"))
    args = ap.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    quarantine = json.loads(Path(args.quarantine).read_text(encoding="utf-8"))
    refs = set(quarantine["image_refs"])
    reasons = quarantine["reasons"]
    retained, flagged = [], []
    for rec in source["records"]:
        ref = str(rec.get("input", {}).get("image", {}).get("image_ref", ""))
        if ref in refs:
            flagged.append(rec)
        else:
            retained.append(rec)

    out_obj = {k: v for k, v in source.items() if k != "records"}
    out_obj["benchmark_candidate"] = {
        "status": "provisional_pilot_candidate_not_fully_human_validated",
        "clean_dataset_records": len(source["records"]),
        "quarantined_records": len(flagged),
        "candidate_records": len(retained),
        "quarantined_image_refs": sorted(refs),
        "exclusion_reason": "Manual visual audit found apparent diagram-question/solution mismatches; keep originals in physics_dataset_clean_all.json and manual_review_queue.csv.",
        "input_contract": "At inference use question + image only. Do not send cot, answer, or unit to the model.",
    }
    out_obj["records"] = retained
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.jsonl).open("w", encoding="utf-8") as f:
        for rec in retained:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fields = ["sample_id", "image_ref", "question", "answer", "unit", "review_reason", "status", "cot_excerpt"]
    with Path(args.review_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in flagged:
            ref = str(rec.get("input", {}).get("image", {}).get("image_ref", ""))
            cot = str(rec.get("reference", {}).get("solution_cot", ""))
            writer.writerow({
                "sample_id": rec.get("sample_id", ""),
                "image_ref": ref,
                "question": rec.get("input", {}).get("question", ""),
                "answer": rec.get("reference", {}).get("answer", ""),
                "unit": rec.get("reference", {}).get("unit", ""),
                "review_reason": reasons.get(ref, "Manual visual review required."),
                "status": "quarantined_pending_human_review",
                "cot_excerpt": cot[-500:].replace("\n", " "),
            })
    print(json.dumps({"input": len(source["records"]), "retained": len(retained), "quarantined": len(flagged), "quarantined_image_refs": sorted(refs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
