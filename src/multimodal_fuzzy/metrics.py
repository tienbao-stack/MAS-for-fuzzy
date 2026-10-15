from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


def normalize_text(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\s.,;:!?]+$", "", s)
    return s


def _decimal_from_token(token: str) -> Decimal | None:
    token = token.strip().replace("−", "-").replace("–", "-").replace(" ", "")
    if "," in token and "." not in token:
        # Single comma with one or two trailing digits is treated as decimal comma;
        # a three-digit suffix is treated as a thousands separator.
        if token.count(",") == 1 and len(token.rsplit(",", 1)[1]) != 3:
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token and "." in token:
        token = token.replace(",", "")
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def to_numeric(value: Any) -> float | None:
    """Parse plain decimal, decimal-comma, or fractional scalar strings."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            n = float(value)
            return n if math.isfinite(n) else None
        except (TypeError, ValueError, OverflowError):
            return None
    s = str(value).strip().strip("$` ")
    s = re.sub(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"\1/\2", s)
    # The answer key may include an attached unit; accept only one leading scalar.
    frac = re.fullmatch(r"([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))\s*/\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))", s)
    if frac:
        a, b = _decimal_from_token(frac.group(1)), _decimal_from_token(frac.group(2))
        if a is None or b is None or b == 0:
            return None
        return float(a / b)
    scalar = re.fullmatch(r"([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*(?:[A-Za-zΩΩµμ°%$\\{}_^]+)?", s)
    if scalar:
        d = _decimal_from_token(scalar.group(1))
        if d is None:
            return None
        n = float(d)
        return n if math.isfinite(n) else None
    return None


def normalize_unit(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip()
    s = s.replace("Ω", "Ω").replace("ohms", "Ω").replace("ohm", "Ω")
    s = s.replace("µ", "μ")
    s = s.replace("$\\circ$", "°").replace("\\circ", "°")
    s = re.sub(r"\s+", "", s)
    if s.casefold() in {"", "-", "—", "–", "none", "dimensionless", "1"}:
        return ""
    return s


def answer_comparison(
    predicted: Any,
    gold: Any,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-2,
) -> dict[str, Any]:
    p_num, g_num = to_numeric(predicted), to_numeric(gold)
    if p_num is not None and g_num is not None:
        err = abs(p_num - g_num)
        tolerance = max(abs_tol, rel_tol * abs(g_num))
        return {
            "answer_correct": err <= tolerance,
            "answer_exact": math.isclose(p_num, g_num, rel_tol=1e-9, abs_tol=1e-9),
            "numeric_comparable": True,
            "absolute_error": err,
            "relative_error": err / abs(g_num) if g_num != 0 else (0.0 if err == 0 else None),
            "numeric_tolerance": tolerance,
        }
    text_correct = normalize_text(predicted) == normalize_text(gold) and normalize_text(gold) != ""
    return {
        "answer_correct": text_correct,
        "answer_exact": text_correct,
        "numeric_comparable": False,
        "absolute_error": None,
        "relative_error": None,
        "numeric_tolerance": None,
    }


def evaluate_pair(
    prediction: dict[str, Any], record: dict[str, Any],
    abs_tol: float = 1e-3, rel_tol: float = 1e-2,
) -> dict[str, Any]:
    reference = record.get("reference", {})
    ans = answer_comparison(
        prediction.get("predicted_answer"), reference.get("answer"), abs_tol, rel_tol
    )
    gold_unit = normalize_unit(reference.get("unit"))
    pred_unit = normalize_unit(prediction.get("predicted_unit"))
    parse_ok = prediction.get("predicted_answer") is not None and str(prediction.get("predicted_answer")).strip() != ""
    unit_ok = bool(parse_ok and pred_unit == gold_unit)
    return {
        "sample_id": record.get("sample_id"),
        "answer_correct": bool(ans["answer_correct"]),
        "answer_exact": bool(ans["answer_exact"]),
        "unit_correct": bool(unit_ok),
        "joint_correct": bool(ans["answer_correct"] and unit_ok),
        "parse_ok": bool(parse_ok),
        "numeric_comparable": bool(ans["numeric_comparable"]),
        "absolute_error": ans["absolute_error"],
        "relative_error": ans["relative_error"],
        "numeric_tolerance": ans["numeric_tolerance"],
        "gold_answer": reference.get("answer"),
        "predicted_answer": prediction.get("predicted_answer"),
        "gold_unit": reference.get("unit"),
        "predicted_unit": prediction.get("predicted_unit", ""),
        "status": prediction.get("status", "unknown"),
    }
