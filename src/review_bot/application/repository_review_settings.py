"""Frozen review customization from one trusted default-branch commit.

The optional .review-bot/review-settings.json contains only judgment/explanation
text, inert presentation labels and declared comment sections. Modes, planning,
fact checking, schemas, citations and permissions remain program controlled.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from review_bot.ports import RepositoryReader

SETTINGS_PATH = ".review-bot/review-settings.json"
MAX_SETTINGS_BYTES = 65536
MAX_INSTRUCTION_BYTES = 16384
MAX_TEMPLATE_BYTES = 8192
PRESENTATION_KEYS = frozenset({
    "title_a", "title_b", "static_section", "llm_section", "details_label", "footer_notice",
})
TEMPLATE_SECTIONS = {
    "architecture": ("summary", "static_analysis", "llm_review"),
    "intent": ("summary", "llm_review", "perspectives"),
}
_MODES = frozenset(TEMPLATE_SECTIONS)
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_HTML = re.compile(r"<[/!?]?[A-Za-z][^>]*>")


class RepositorySettingsError(ValueError):
    """Safe diagnostic code; never includes repository content."""


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _pairs(mapping: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


@dataclass(frozen=True)
class RepositoryReviewSettings:
    review_instructions: tuple[tuple[str, str], ...] = ()
    presentation: tuple[tuple[str, str], ...] = ()
    comment_templates: tuple[tuple[str, str], ...] = ()
    source_sha: str | None = None

    def instructions_for(self, mode: str) -> str:
        return dict(self.review_instructions).get(mode, "")

    def comment_template(self, mode: str) -> str | None:
        return dict(self.comment_templates).get(mode)

    @property
    def generation_hash(self) -> str:
        return _hash({"review_instructions": dict(self.review_instructions)})

    @property
    def presentation_hash(self) -> str:
        return _hash({"presentation": dict(self.presentation),
            "comment_templates": dict(self.comment_templates)})

    @property
    def effective_version(self) -> str:
        return _hash({"generation_hash": self.generation_hash,
            "presentation_hash": self.presentation_hash})

    @property
    def is_builtin(self) -> bool:
        return not (self.review_instructions or self.presentation or self.comment_templates)

    def manifest(self) -> dict:
        return {"schema_version": 1, "review_instructions": dict(self.review_instructions),
            "presentation": dict(self.presentation), "comment_templates": dict(self.comment_templates)}


@dataclass(frozen=True)
class SettingsResolution:
    bundle: RepositoryReviewSettings
    diagnostic: str | None = None
    used_fallback: bool = False
    attempted_sha: str | None = None


class RepositoryReviewSettingsStore(Protocol):
    def transaction(self, installation_id: int, repository_id: int) -> AbstractContextManager: ...
    def load(self, installation_id: int, repository_id: int) -> RepositoryReviewSettings | None: ...
    def save(self, installation_id: int, repository_id: int, bundle: RepositoryReviewSettings) -> None: ...


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RepositorySettingsError("duplicate_key")
        result[key] = value
    return result


def _text(value: object, maximum: int, *, template: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepositorySettingsError("invalid_text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise RepositorySettingsError("invalid_text_encoding") from None
    if len(encoded) > maximum:
        raise RepositorySettingsError("text_size_limit")
    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in value):
        raise RepositorySettingsError("invalid_text_control")
    if _HTML.search(value) or "{%" in value or "{#" in value:
        raise RepositorySettingsError("non_plain_text")
    if not template and ("{{" in value or "}}" in value):
        raise RepositorySettingsError("unexpected_placeholder")
    return value


def _invalid_constant(_: str) -> None:
    raise RepositorySettingsError("invalid_json_constant")


def parse_repository_review_settings(raw: bytes, source_sha: str) -> RepositoryReviewSettings:
    if not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None:
        raise RepositorySettingsError("invalid_default_sha")
    if not isinstance(raw, bytes) or len(raw) > MAX_SETTINGS_BYTES:
        raise RepositorySettingsError("manifest_size_limit")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, RepositorySettingsError):
            raise
        raise RepositorySettingsError("invalid_manifest_encoding_or_json") from None
    if not isinstance(value, dict) or set(value) - {
        "schema_version", "review_instructions", "presentation", "comment_templates",
    } or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise RepositorySettingsError("invalid_manifest_slots")
    sections = {}
    for section, allowed, maximum in (
        ("review_instructions", _MODES, MAX_INSTRUCTION_BYTES),
        ("presentation", PRESENTATION_KEYS, 1024),
        ("comment_templates", _MODES, MAX_TEMPLATE_BYTES),
    ):
        supplied = value.get(section, {})
        if not isinstance(supplied, dict) or set(supplied) - allowed:
            raise RepositorySettingsError("invalid_manifest_slots")
        sections[section] = {}
        for key, text in supplied.items():
            text = _text(text, maximum, template=section == "comment_templates")
            if section == "comment_templates":
                for name in TEMPLATE_SECTIONS[key]:
                    token = "{{" + name + "}}"
                    if text.count(token) != 1:
                        raise RepositorySettingsError("invalid_template_sections")
                    text = text.replace(token, "")
                if "{" in text or "}" in text:
                    raise RepositorySettingsError("invalid_template_placeholder")
                text = supplied[key]
            sections[section][key] = text
    return RepositoryReviewSettings(**{key: _pairs(section) for key, section in sections.items()},
        source_sha=source_sha)


def _validate_identity(installation_id: int, repository_id: int) -> None:
    if any(type(value) is not int or value <= 0 for value in (installation_id, repository_id)):
        raise ValueError("invalid_repository_settings_identity")


class RepositoryReviewSettingsResolver:
    def __init__(self, store: RepositoryReviewSettingsStore) -> None:
        self.store = store

    def resolve(self, installation_id: int, repository_id: int,
        load_default: Callable[[], tuple[str, RepositoryReader]]) -> SettingsResolution:
        """Resolve once; invalid/unavailable settings use the whole last valid bundle.

        The loader authenticates and verifies the base repository identity. A missing
        file intentionally selects builtin defaults and persists that valid choice.
        """
        _validate_identity(installation_id, repository_id)
        attempted_sha = None
        try:
            with self.store.transaction(installation_id, repository_id):
                try:
                    attempted_sha, reader = load_default()
                    if not isinstance(attempted_sha, str) or _SHA.fullmatch(attempted_sha) is None:
                        raise RepositorySettingsError("invalid_default_sha")
                    blobs = reader.read_tree_blobs(attempted_sha, (SETTINGS_PATH,))
                    if not isinstance(blobs, dict) or set(blobs) - {SETTINGS_PATH}:
                        raise RepositorySettingsError("invalid_settings_path")
                    bundle = (parse_repository_review_settings(blobs[SETTINGS_PATH], attempted_sha)
                        if SETTINGS_PATH in blobs else RepositoryReviewSettings(source_sha=attempted_sha))
                    self.store.save(installation_id, repository_id, bundle)
                    return SettingsResolution(bundle, attempted_sha=attempted_sha)
                except Exception as exc:
                    diagnostic = (str(exc) if isinstance(exc, RepositorySettingsError)
                        else "settings_source_or_state_unavailable")
                    try:
                        previous = self.store.load(installation_id, repository_id)
                    except Exception:
                        previous = None
                    return SettingsResolution(previous or RepositoryReviewSettings(),
                        diagnostic=diagnostic, used_fallback=True, attempted_sha=attempted_sha)
        except Exception:
            return SettingsResolution(RepositoryReviewSettings(), diagnostic="settings_state_unavailable",
                used_fallback=True, attempted_sha=attempted_sha)
