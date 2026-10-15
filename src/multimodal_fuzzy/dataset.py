from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load project JSON (object with records) or JSONL (one record per line)."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        return {"dataset": path.stem, "version": "jsonl"}, records
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return {"dataset": path.stem, "version": "list"}, obj
    if not isinstance(obj, dict) or not isinstance(obj.get("records"), list):
        raise ValueError("Expected a JSON object containing a 'records' list, a JSON array, or JSONL.")
    meta = {k: v for k, v in obj.items() if k != "records"}
    return meta, obj["records"]


def dump_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def image_ref(record: dict[str, Any]) -> str:
    return str(record.get("input", {}).get("image", {}).get("image_ref", ""))


def question(record: dict[str, Any]) -> str:
    return str(record.get("input", {}).get("question", ""))


def gold_answer(record: dict[str, Any]) -> str:
    return str(record.get("reference", {}).get("answer", ""))


def gold_unit(record: dict[str, Any]) -> str:
    return str(record.get("reference", {}).get("unit", ""))
