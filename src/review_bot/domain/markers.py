"""Trusted B marker syntax shared by configuration and rendering."""
from __future__ import annotations

DEFAULT_B_MARKER = "<!-- review-bot:intent:v1 -->"


def valid_intent_marker(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("<!-- ") or not value.endswith(" -->"):
        return False
    label = value[5:-4]
    alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return (1 <= len(label) <= 121 and label[0] in alphanumeric
        and all(char in alphanumeric + ":_-" for char in label)
        and value != "<!-- review-bot:architecture:v1 -->")