from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal_fuzzy.dataset import load_dataset  # noqa: E402
from multimodal_fuzzy.metrics import evaluate_pair, to_numeric  # noqa: E402


def wilson_ci(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate B0 predictions against Physics gold labels.")
    ap.add_argument("--dataset", default=str(ROOT / "data/processed/physics_dataset_candidate.json"))
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split-file", default=str(ROOT / "data/splits/split_assignments.json"))
    ap.add_argument("--split", choices=["train", "validation", "test"], default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "data/results"))
    ap.add_argument("--abs-tol", type=float, default=1e-3)
    ap.add_argument("--rel-tol", type=float, default=1e-2)
    ap.add_argument("--include-mock", action="store_true", help="Not recommended; includes synthetic test rows in metrics.")
    args = ap.parse_args()

    _, dataset_records = load_dataset(args.dataset)
    ref_by_id = {str(r["sample_id"]): r for r in dataset_records}
    split_assignments = json.loads(Path(args.split_file).read_text(encoding="utf-8")).get("assignments", {})
    pred_rows = read_jsonl(Path(args.predictions))
    duplicate_prediction_rows = max(0, len(pred_rows) - len({str(r.get("sample_id", "")) for r in pred_rows}))
    latest: dict[str, dict[str, Any]] = {}
    for pred in pred_rows:
        if pred.get("is_mock") and not args.include_mock:
            continue
        sid = str(pred.get("sample_id", ""))
        if sid in ref_by_id and (args.split is None or split_assignments.get(sid) == args.split):
            latest[sid] = pred

    if args.split:
        expected_ids = {sid for sid, split in split_assignments.items() if split == args.split and sid in ref_by_id}
    else:
        expected_ids = set(ref_by_id.keys())
    rows = []
    for sid, pred in sorted(latest.items()):
        result = evaluate_pair(pred, ref_by_id[sid], abs_tol=args.abs_tol, rel_tol=args.rel_tol)
        result["split"] = split_assignments.get(sid)
        result["parse_status"] = pred.get("parse_status", "unspecified")
        result["is_mock"] = bool(pred.get("is_mock", False))
        result["model"] = pred.get("model", "unknown")
        rows.append(result)

    n = len(rows)
    answer_ok = sum(r["answer_correct"] for r in rows)
    exact_ok = sum(r["answer_exact"] for r in rows)
    unit_ok = sum(r["unit_correct"] for r in rows)
    joint_ok = sum(r["joint_correct"] for r in rows)
    parse_ok = sum(r["parse_ok"] for r in rows)
    numeric_rows = [r for r in rows if r["numeric_comparable"]]
    abs_errors = [r["absolute_error"] for r in numeric_rows if r["absolute_error"] is not None]
    relative_errors = [r["relative_error"] for r in numeric_rows if r["relative_error"] is not None]
    summary = {
        "evaluation_version": "physics-qa-evaluator-0.1",
        "dataset": str(args.dataset),
        "predictions": str(args.predictions),
        "split": args.split or "all",
        "synthetic_predictions_excluded_by_default": not args.include_mock,
        "expected_samples_in_split": len(expected_ids),
        "prediction_rows_read": len(pred_rows),
        "unique_sample_predictions_evaluated": n,
        "prediction_coverage": len(set(latest) & expected_ids) / len(expected_ids) if expected_ids else None,
        "missing_predictions": len(expected_ids - set(latest)),
        "duplicate_prediction_rows": duplicate_prediction_rows,
        "parse_rate": parse_ok / n if n else None,
        "answer_accuracy_tolerance": answer_ok / n if n else None,
        "answer_accuracy_95pct_wilson_ci": wilson_ci(answer_ok, n),
        "answer_exact_accuracy": exact_ok / n if n else None,
        "unit_accuracy": unit_ok / n if n else None,
        "joint_answer_and_unit_accuracy": joint_ok / n if n else None,
        "joint_accuracy_95pct_wilson_ci": wilson_ci(joint_ok, n),
        "numeric_comparable_samples": len(numeric_rows),
        "numeric_mae": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "numeric_median_absolute_error": statistics.median(abs_errors) if abs_errors else None,
        "numeric_mean_relative_error": sum(relative_errors) / len(relative_errors) if relative_errors else None,
        "tolerance": {"absolute": args.abs_tol, "relative": args.rel_tol, "rule": "abs(pred-gold) <= max(abs_tol, rel_tol*abs(gold))"},
        "important_note": "solution_cot is not used as a gold string target and is not automatically graded. Human or separately validated reasoning evaluation is required.",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.split or "all"
    (out_dir / f"b0_{tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / f"b0_{tag}_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
