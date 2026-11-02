"""Strict, provider-independent chat JSON parsing and output-field checks."""
from __future__ import annotations

import json
import math
from typing import Any

from review_bot.application.architecture_review import PermanentModelError

def _encode(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid model request JSON") from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _check_json_strings(value: Any) -> None:
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, dict):
        for key, child in value.items():
            key.encode("utf-8")
            _check_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            _check_json_strings(child)


def _json_object(value: bytes | str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        text.encode("utf-8")
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_float=_finite_float)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        _check_json_strings(result)
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid model response JSON") from None


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise PermanentModelError("Invalid model token count")
    return value


def _check_output_fields(value: Any, schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise PermanentModelError("Invalid model output schema")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise PermanentModelError("Unexpected model output fields")
        for key, child in value.items():
            _check_output_fields(child, properties.get(key))
    elif isinstance(value, list):
        for child in value:
            _check_output_fields(child, schema.get("items"))
