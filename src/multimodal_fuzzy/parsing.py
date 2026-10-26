"""Robust parser for B0 model outputs, including truncated JSON.

Drop-in replacement for src/multimodal_fuzzy/parsing.py.
A complete JSON parse is preferred. If JSON is truncated after the answer/unit
fields, the parser recovers those fields and marks the result as partial so the
run remains auditable.
"""
from __future__ import annotations

import json
import re
from typing import Any


# A JSON string token, including escaped quotes/backslashes, or a JSON primitive.
_JSON_VALUE = r'(?:"(?:\\.|[^"\\])*"|null|true|false|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'


def _field(raw: str, name: str) -> Any:
    match = re.search(r'"' + re.escape(name) + r'"\s*:\s*(' + _JSON_VALUE + r')', raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def split_trailing_unit(answer: str) -> tuple[str, str]:
    """Separate a scalar/fraction from a simple trailing physical unit."""
    s = (answer or "").strip().strip("`$ ")
    pattern = re.compile(
        r"^([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:\s*/\s*[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))?(?:[eE][+-]?\d+)?)\s*(Ω|Ω|ohms?|[A-Za-zµμ%°]+)?\s*$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(s)
    if match and match.group(2):
        return match.group(1).replace(" ", ""), match.group(2).strip()
    return s, ""


def parse_model_output(text: str) -> dict[str, Any]:
    """Parse complete JSON, recover fields from truncated JSON, or use safe fallbacks."""
    raw = (text or "").strip()
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # 1. Strict JSON parse first.
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(candidate[start:end + 1])
                if isinstance(obj, dict):
                    answer = obj.get("answer", obj.get("final_answer", obj.get("final")))
                    unit = obj.get("unit", "")
                    explanation = obj.get("explanation", obj.get("reasoning_summary", ""))
                    if answer is not None:
                        answer_text, extracted_unit = split_trailing_unit(str(answer).strip())
                        return {
                            "predicted_answer": answer_text,
                            "predicted_unit": str(unit or extracted_unit or "").strip(),
                            "predicted_explanation": str(explanation or "").strip(),
                            "parse_status": "json",
                        }
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # 2. Recover answer/unit if JSON was cut off after these fields, e.g. an
    # incomplete explanation due to the output-token limit. Keep a visible flag.
    answer = _field(raw, "answer")
    if answer is None:
        answer = _field(raw, "final_answer")
    if answer is not None:
        answer_text, extracted_unit = split_trailing_unit(str(answer).strip())
        unit = _field(raw, "unit")
        explanation = _field(raw, "explanation")
        return {
            "predicted_answer": answer_text,
            "predicted_unit": str(unit or extracted_unit or "").strip(),
            "predicted_explanation": str(explanation or "").strip() if explanation is not None else "",
            "parse_status": "partial_json_recovered",
        }

    # 3. Answer line fallback outside JSON.
    answer_line = re.search(
        r"(?:final\s+answer|answer|đáp\s*án|kết\s*quả)\s*[:：=]\s*([^\n]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if answer_line:
        answer_text, unit = split_trailing_unit(answer_line.group(1).strip().rstrip(".*"))
        return {
            "predicted_answer": answer_text,
            "predicted_unit": unit,
            "predicted_explanation": "",
            "parse_status": "answer_line_fallback",
        }

    # 4. A standalone scalar is safe to parse; don't guess from arbitrary prose.
    scalar_pattern = re.compile(
        r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:\s*/\s*[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))?(?:[eE][+-]?\d+)?(?:\s*(?:Ω|Ω|ohms?|[A-Za-zµμ%°]+))?$",
        flags=re.IGNORECASE,
    )
    if scalar_pattern.fullmatch(raw):
        answer_text, unit = split_trailing_unit(raw)
        return {
            "predicted_answer": answer_text,
            "predicted_unit": unit,
            "predicted_explanation": "",
            "parse_status": "scalar_fallback",
        }

    return {
        "predicted_answer": None,
        "predicted_unit": "",
        "predicted_explanation": "",
        "parse_status": "failed",
    }



# from __future__ import annotations

# import json
# import re
# from typing import Any


# def parse_model_output(text: str) -> dict[str, Any]:
#     """Parse a JSON response; fall back conservatively to a final-answer line."""
#     raw = (text or "").strip()
#     candidates = [raw]
#     fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
#     if fenced:
#         candidates.insert(0, fenced.group(1).strip())

#     for candidate in candidates:
#         start, end = candidate.find("{"), candidate.rfind("}")
#         if start >= 0 and end > start:
#             try:
#                 obj = json.loads(candidate[start:end + 1])
#                 if isinstance(obj, dict):
#                     answer = obj.get("answer", obj.get("final_answer", obj.get("final", None)))
#                     unit = obj.get("unit", "")
#                     explanation = obj.get("explanation", obj.get("reasoning_summary", ""))
#                     if answer is not None:
#                         answer_text = str(answer).strip()
#                         answer_text, extracted_unit = split_trailing_unit(answer_text)
#                         return {
#                             "predicted_answer": answer_text,
#                             "predicted_unit": str(unit or extracted_unit or "").strip(),
#                             "predicted_explanation": str(explanation or "").strip(),
#                             "parse_status": "json",
#                         }
#             except (json.JSONDecodeError, TypeError, ValueError):
#                 pass

#     # Fall back to an explicit answer/final-answer line to avoid parsing numbers
#     # from the middle of a long explanation whenever possible.
#     answer_line = re.search(
#         r"(?:final\s+answer|answer|đáp\s*án|kết\s*quả)\s*[:：=]\s*([^\n]+)",
#         raw,
#         flags=re.IGNORECASE,
#     )
#     if answer_line:
#         answer_text, unit = split_trailing_unit(answer_line.group(1).strip().rstrip(".*"))
#         return {
#             "predicted_answer": answer_text,
#             "predicted_unit": unit,
#             "predicted_explanation": "",
#             "parse_status": "answer_line_fallback",
#         }

#     # A standalone scalar is a safe fallback; do not search arbitrary explanations.
#     scalar_pattern = re.compile(
#         r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*/\s*[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))?(?:[eE][+-]?\d+)?(?:\s*(?:Ω|Ω|ohms?|[A-Za-zµμ%°]+))?$",
#         flags=re.IGNORECASE,
#     )
#     if scalar_pattern.fullmatch(raw):
#         answer_text, unit = split_trailing_unit(raw)
#         return {
#             "predicted_answer": answer_text,
#             "predicted_unit": unit,
#             "predicted_explanation": "",
#             "parse_status": "scalar_fallback",
#         }

#     # Do not silently guess a value from arbitrary prose: mark unparsed instead.
#     return {
#         "predicted_answer": None,
#         "predicted_unit": "",
#         "predicted_explanation": "",
#         "parse_status": "failed",
#     }


# def split_trailing_unit(answer: str) -> tuple[str, str]:
#     """Separate a simple scalar/fraction from a trailing physical unit."""
#     s = answer.strip().strip("`$ ")
#     pattern = re.compile(
#         r"^([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:\s*/\s*[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))?(?:[eE][+-]?\d+)?)\s*(Ω|Ω|ohms?|[A-Za-zµμ%°]+|\$\\circ\$)?\s*$",
#         flags=re.IGNORECASE,
#     )
#     match = pattern.match(s)
#     if match and match.group(2):
#         return match.group(1).replace(" ", ""), match.group(2).strip()
#     return s, ""
