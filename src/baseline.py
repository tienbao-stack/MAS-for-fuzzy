from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
sys.path.insert(0, str(ROOT / "src"))
from multimodal_fuzzy.dataset import gold_answer, gold_unit, image_ref, load_dataset, question  # noqa: E402
from multimodal_fuzzy.parsing import parse_model_output  # noqa: E402
from multimodal_fuzzy.providers import (  # noqa: E402
    PROVIDER_CHOICES,
    create_chat_client,
    image_content_part,
    resolve_provider_config,
)

SYSTEM_PROMPT = """You are a careful physics question-answering model. Solve the given Vietnamese physics question using only the question text and the supplied image. Pay attention to whether the circuit/diagram agrees with the wording; do not invent components or values that are not shown. Return exactly one JSON object with keys: \"answer\" (short final scalar or text, without unit when possible), \"unit\" (unit symbol or empty string for dimensionless answers), and \"explanation\" (a concise verification summary, not a long derivation). If the text and image appear inconsistent, say so in the explanation and avoid pretending the data are certain."""


def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image file type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _read_existing_ids(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("sample_id"):
                    seen.add(str(row["sample_id"]))
            except json.JSONDecodeError:
                continue
    return seen


def _write_readable_predictions(jsonl_path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid prediction JSONL at {jsonl_path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {jsonl_path}:{line_no}")
                rows.append(row)

    readable_path = jsonl_path.with_name(f"{jsonl_path.stem}_readable.json")
    readable_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return readable_path


def _call_model(
    client: Any,
    provider: str,
    model: str,
    question_text: str,
    image_path: Path,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    data_url = _image_data_url(image_path)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question:\n" + question_text + "\n\nAnswer as the requested JSON object."},
                    image_content_part(provider, data_url),
                ],
            },
        ],
    )
    content = response.choices[0].message.content or ""
    usage: dict[str, Any] = {}
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }
    return content, usage


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the direct MLLM/VLM baseline B0 on Text+Image Physics QA.")
    ap.add_argument("--dataset", default=str(ROOT / "data/processed/physics_dataset_candidate.json"))
    ap.add_argument("--image-dir", default=str(ROOT / "data/images"))
    ap.add_argument("--split-file", default=str(ROOT / "data/splits/split_assignments.json"))
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help="openai, gemini, openrouter, or ollama (use --mode local|cloud).",
    )
    ap.add_argument(
        "--mode",
        choices=["local", "cloud"],
        default=None,
        help="Ollama mode; applies when --provider=ollama. Defaults to LLM_MODE or local.",
    )
    ap.add_argument("--model", default=None, help="Model name; overrides LLM_MODEL and provider-specific settings.")
    ap.add_argument("--base-url", default=None, help="Optional OpenAI-compatible API base URL override.")
    ap.add_argument("--limit", type=int, default=None, help="Optional pilot limit; not a full benchmark run.")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--resume", action="store_true", help="Skip sample IDs already present in the output JSONL.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output file; ignored together with --resume.")
    ap.add_argument("--smoke-test-from-gold", action="store_true", help="I/O-only smoke test; uses gold labels and is NEVER a model result.")
    args = ap.parse_args()

    dataset_path, image_dir = Path(args.dataset), Path(args.image_dir)
    _, records = load_dataset(dataset_path)
    split_obj = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    assignments = split_obj.get("assignments", {})
    chosen = [r for r in records if assignments.get(str(r.get("sample_id"))) == args.split]
    if not chosen:
        raise SystemExit(f"No dataset records assigned to split={args.split!r}. Check dataset and split file.")

    out = Path(args.output) if args.output else ROOT / f"data/predictions/b0_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.resume:
        if args.overwrite:
            out.unlink()
        else:
            raise SystemExit(f"Output already exists: {out}. Pass --resume to continue it or --overwrite to replace it.")
    already = _read_existing_ids(out) if args.resume else set()
    chosen = [r for r in chosen if str(r.get("sample_id")) not in already]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive.")
        chosen = chosen[:args.limit]
    if not chosen:
        readable_out = _write_readable_predictions(out)
        print(f"Nothing to run: all selected sample IDs are already in the output or the limit is zero.")
        print(f"Updated readable copy at {readable_out}")
        return 0

    client = None
    model = "SMOKE_TEST_NOT_A_MODEL"
    provider = "smoke_test_from_gold"
    if not args.smoke_test_from_gold:
        try:
            backend = resolve_provider_config(
                provider=args.provider,
                mode=args.mode,
                model=args.model,
                base_url=args.base_url,
            )
        except ValueError as exc:
            raise SystemExit(f"LLM configuration error: {exc}") from exc
        try:
            client = create_chat_client(backend)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        provider = backend.provider
        model = backend.model

    with out.open("a", encoding="utf-8") as f:
        for index, rec in enumerate(chosen, 1):
            sid = str(rec["sample_id"])
            ref = image_ref(rec)
            image_path = image_dir / ref
            start = time.perf_counter()
            base = {
                "sample_id": sid,
                "split": args.split,
                "model": model,
                "provider": provider,
                "is_mock": bool(args.smoke_test_from_gold),
                "image_ref": ref,
            }
            if not image_path.is_file():
                result = {**base, "status": "missing_image", "predicted_answer": None, "predicted_unit": "", "prediction_raw": "", "error": str(image_path)}
            elif args.smoke_test_from_gold:
                result = {
                    **base,
                    "status": "mock_test_only",
                    "predicted_answer": gold_answer(rec),
                    "predicted_unit": gold_unit(rec),
                    "predicted_explanation": "Synthetic I/O check: copied from gold label; not a model prediction.",
                    "prediction_raw": "SMOKE_TEST_FROM_GOLD",
                    "parse_status": "mock",
                    "latency_seconds": 0.0,
                }
            else:
                try:
                    prediction_raw, usage = _call_model(
                        client,
                        provider,
                        model,
                        question(rec),
                        image_path,
                        args.max_tokens,
                    )
                    parsed = parse_model_output(prediction_raw)
                    result = {
                        **base,
                        **parsed,
                        "status": "ok",
                        "prediction_raw": prediction_raw,
                        "latency_seconds": round(time.perf_counter() - start, 4),
                        "usage": usage,
                    }
                except Exception as exc:  # record failure so runs can be resumed/audited
                    result = {
                        **base,
                        "status": "api_error",
                        "predicted_answer": None,
                        "predicted_unit": "",
                        "predicted_explanation": "",
                        "prediction_raw": "",
                        "parse_status": "failed",
                        "latency_seconds": round(time.perf_counter() - start, 4),
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_cause": repr(exc.__cause__),
                    }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{index}/{len(chosen)}] {sid}: {result['status']} ({result.get('parse_status', 'n/a')})")

    readable_out = _write_readable_predictions(out)
    print(f"Wrote {len(chosen)} records to {out}")
    print(f"Wrote user-readable copy to {readable_out}")
    if args.smoke_test_from_gold:
        print("WARNING: This output is synthetic plumbing validation only; evaluator excludes it by default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
