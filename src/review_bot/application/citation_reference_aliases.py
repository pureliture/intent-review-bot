"""Translate long source citation IDs to short IDs at the model boundary."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from review_bot.application.context import hash_json

CITATION_ALIAS_MIN_PROMPT_VERSION = 19

_REFERENCE_DERIVED_HASHES = {
    "checker_response_hash", "original_check_response_hash", "original_item_hash",
    "prior_response_hash", "raw_response_hash", "response_hash",
}


def uses_citation_aliases(prompt_version: str) -> bool:
    prefix = "partitioned-review/v"
    if not isinstance(prompt_version, str) or not prompt_version.startswith(prefix):
        return False
    version = prompt_version[len(prefix):].split("-", 1)[0]
    return version.isdigit() and int(version) >= CITATION_ALIAS_MIN_PROMPT_VERSION


def _payload_reference_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if (key in ("reference_id", "source_block_text_ref") or key.endswith("_reference_id")) and isinstance(child, str):
                result.add(child)
            elif key.endswith("_reference_ids") and isinstance(child, (list, tuple)):
                result.update(item for item in child if isinstance(item, str))
            else:
                result.update(_payload_reference_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_payload_reference_ids(child))
    return result


def _schema_reference_ids(value: Any, key: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        if key == "reference_id":
            enum = value.get("enum")
            if isinstance(enum, list):
                result.update(item for item in enum if isinstance(item, str))
        for child_key, child in value.items():
            result.update(_schema_reference_ids(child, child_key))
    elif isinstance(value, list):
        for child in value:
            result.update(_schema_reference_ids(child, key))
    return result


def _map_payload(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if (key in ("reference_id", "source_block_text_ref") or key.endswith("_reference_id")) and isinstance(child, str):
                result[key] = aliases.get(child, child)
            elif key == "validation_error" and isinstance(child, str):
                for original, alias in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
                    child = child.replace(original, alias)
                result[key] = child
            elif key.endswith("_reference_ids") and isinstance(child, (list, tuple)):
                result[key] = [aliases.get(item, item) if isinstance(item, str) else deepcopy(item)
                    for item in child]
            else:
                result[key] = _map_payload(child, aliases)
        for key in _REFERENCE_DERIVED_HASHES:
            if key in result and key[:-5] in result and isinstance(result[key], str):
                result[key] = hash_json(result[key[:-5]])
        return result
    if isinstance(value, list):
        return [_map_payload(child, aliases) for child in value]
    if isinstance(value, tuple):
        return tuple(_map_payload(child, aliases) for child in value)
    return deepcopy(value)


def _map_schema(value: Any, aliases: dict[str, str], parent_key: str = "") -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == "enum" and parent_key == "reference_id" and isinstance(child, list):
                result[key] = [aliases.get(item, item) if isinstance(item, str) else deepcopy(item)
                    for item in child]
            else:
                result[key] = _map_schema(child, aliases, key)
        return result
    if isinstance(value, list):
        return [_map_schema(child, aliases, parent_key) for child in value]
    if isinstance(value, str) and parent_key == "description":
        for original, alias in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(original, alias)
        return value
    return deepcopy(value)


def alias_model_references(payload: dict[str, Any], response_schema: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Return model-facing payload/schema and an alias-to-original map.

    Aliases are deterministic within one request. Source text and unrelated IDs
    remain unchanged; only fields explicitly naming source references are mapped.
    """
    identifiers = _payload_reference_ids(payload) | _schema_reference_ids(response_schema)
    if not identifiers:
        return deepcopy(payload), deepcopy(response_schema), {}

    allowed = _schema_reference_ids(response_schema)
    groups = (("R", sorted(allowed), allowed), ("X", sorted(identifiers - allowed), identifiers))
    alias_to_original: dict[str, str] = {}
    used_aliases: set[str] = set()
    for initial_prefix, originals, occupied in groups:
        prefix = initial_prefix
        while any(f"{prefix}{index}" in occupied or f"{prefix}{index}" in used_aliases
                for index in range(1, len(originals) + 1)):
            prefix = "C" + prefix
        for index, original in enumerate(originals, 1):
            alias = f"{prefix}{index}"
            alias_to_original[alias] = original
            used_aliases.add(alias)
    original_to_alias = {original: alias for alias, original in alias_to_original.items()}
    return (_map_payload(payload, original_to_alias),
        _map_schema(response_schema, original_to_alias), alias_to_original)


def restore_model_references(response: Any, alias_to_original: dict[str, str]) -> Any:
    """Restore model-selected short IDs before normal application validation."""
    return _map_payload(response, alias_to_original)
