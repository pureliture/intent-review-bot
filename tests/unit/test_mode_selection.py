"""Unit tests for Feature 2: Mode Selection for All PRs.

Requirements:
- requirements-a.md §A-01:
  - Every PR snapshot must evaluate mode selection.
  - Registers 'architecture' (A) with 'all_available' policy.
  - A-only availability does not select B; unknown modes are rejected.
- contracts.md §2:
  - ModeSelection schema: selection_id, snapshot_id, policy_version,
    available_modes, selected_modes, selection_status ('completed', 'failed', 'superseded').
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from review_bot.application.snapshot import assemble_pr_snapshot, assemble_review_snapshot, assemble_snapshot_bundle
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import ChangedFile, EnvelopeEvidence, ReviewSnapshot
from review_bot.domain.models_v1 import ModelResponse, PrSnapshot
from review_bot.ports import MergeBaseResult, ModelClient, RepositoryReader

try:
    from review_bot.application.selection import select_modes  # type: ignore
except ImportError:
    select_modes = None  # type: ignore


@pytest.fixture
def selector() -> Callable[..., Any]:
    if select_modes is None:
        pytest.skip("Mode selector review_bot.application.selection not yet implemented by Worker")
    return select_modes


# ==============================================================================
# Tier 1: Happy Path Tests
# ==============================================================================


class TestModeSelectionTier1:
    def test_default_all_available_selects_architecture(self, selector: Callable) -> None:
        result = selector(snapshot_id="snap-01", available_modes=["architecture"])
        assert "architecture" in getattr(result, "selected_modes", result.get("selected_modes"))
        status = getattr(result, "selection_status", result.get("selection_status"))
        assert status == "completed"

    def test_selection_record_fields(self, selector: Callable) -> None:
        result = selector(snapshot_id="snap-02", available_modes=["architecture"])
        assert getattr(result, "selection_id", result.get("selection_id")) is not None
        assert getattr(result, "snapshot_id", result.get("snapshot_id")) == "snap-02"
        assert getattr(result, "available_modes", result.get("available_modes")) == ["architecture"]

    def test_evaluates_for_both_refactoring_and_general_prs(self, selector: Callable) -> None:
        res_refactor = selector(snapshot_id="snap-refactor", available_modes=["architecture"])
        res_general = selector(snapshot_id="snap-general", available_modes=["architecture"])
        assert "architecture" in getattr(res_refactor, "selected_modes", res_refactor.get("selected_modes"))
        assert "architecture" in getattr(res_general, "selected_modes", res_general.get("selected_modes"))

    def test_mode_b_is_not_selected(self, selector: Callable) -> None:
        result = selector(snapshot_id="snap-03", available_modes=["architecture"])
        selected = getattr(result, "selected_modes", result.get("selected_modes"))
        assert "intent" not in selected
        assert "b" not in selected

    def test_intent_is_registered_independently(self, selector: Callable) -> None:
        assert selector("snap-b", ["intent"]).selected_modes == ("intent",)
        assert selector("snap-ab", ["architecture", "intent"]).selected_modes == ("architecture", "intent")


# ==============================================================================
# Tier 2: Boundary & Edge Cases
# ==============================================================================


class TestModeSelectionTier2:
    def test_empty_available_modes_fails_or_rejects(self, selector: Callable) -> None:
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            selector(snapshot_id="snap-empty", available_modes=[])

    def test_unregistered_mode_request_fails(self, selector: Callable) -> None:
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            selector(snapshot_id="snap-unreg", available_modes=["unknown_mode_xyz"])

    def test_idempotent_selection_for_identical_snapshot(self, selector: Callable) -> None:
        res1 = selector(snapshot_id="snap-stable", available_modes=["architecture"])
        res2 = selector(snapshot_id="snap-stable", available_modes=["architecture"])
        assert getattr(res1, "selected_modes", res1.get("selected_modes")) == getattr(
            res2, "selected_modes", res2.get("selected_modes")
        )


def selection_inputs(
    paths: tuple[str, ...] = (), title: str = "", body: str = "",
    renames: dict[str, str] | None = None,
) -> tuple[ReviewSnapshot, PrSnapshot]:
    changed_files = tuple(ChangedFile(path) for path in paths)
    snapshot = assemble_review_snapshot(
        "org/repo", 42, 1, 2, "base", "head", MergeBaseResult("base", "ancestor", "comparable"),
        changed_files=changed_files, title=title, body=body,
    )
    snapshot = replace(snapshot, evidence=(EnvelopeEvidence(
        "pr-text", "pr_text", "head", {"pr_number": 42},
        {"title": title, "body": body}, snapshot.pr_text_digest,
    ),))
    pr = assemble_pr_snapshot("org/repo", 42, "ancestor", "head", "main", changed_files=changed_files, rename_map=renames)
    return snapshot, pr


def test_local_selector_does_not_apply_a_docs_eligibility_to_b() -> None:
    from review_bot.application.selection import LightweightModeSelector

    snapshot, pr = selection_inputs(("README.md",), "Fix typos")
    assert LightweightModeSelector().select_modes(snapshot, pr, ("architecture", "intent")).selected_modes == ("intent",)


@pytest.mark.parametrize("selected", [[], ["intent"], ["architecture"], ["architecture", "intent"]])
def test_model_selector_accepts_only_available_independent_modes(selected) -> None:
    from review_bot.application.selection import ModelModeSelector

    snapshot, pr = selection_inputs(("custom.settings",))
    client = Mock(spec=ModelClient)
    client.complete.return_value = ModelResponse(json.dumps({"selected_modes": selected}))
    result = ModelModeSelector(client, model="selector").select_modes(snapshot, pr, ("architecture", "intent"))
    assert result.selected_modes == tuple(selected)
    request = client.complete.call_args.args[0]
    assert "intent" in request.system_prompt
    assert request.response_schema["properties"]["selected_modes"]["maxItems"] == 2


@pytest.mark.parametrize("paths", [(), ("README.md", "docs/guide.md"), ("assets/logo.png", "images/banner.svg")])
def test_lightweight_selector_skips_nonarchitectural_changes(paths: tuple[str, ...]) -> None:
    from review_bot.application.selection import LightweightModeSelector
    from review_bot.ports import ModeSelectorPort

    selector = LightweightModeSelector()
    snapshot, pr = selection_inputs(paths, title="Fix typos", body="Correct spelling")
    started = time.monotonic()

    result = selector.select_modes(snapshot, pr, ("architecture",))

    assert time.monotonic() - started < 1.0
    assert isinstance(selector, ModeSelectorPort)
    assert result.selected_modes == ()
    assert result.selection_status == "completed"
    assert result.snapshot_id == snapshot.snapshot_id
    assert result.available_modes == ("architecture",)
    assert result == selector.select_modes(snapshot, pr, ("architecture",))


@pytest.mark.parametrize("paths", [
    ("src/main/java/App.java",), ("src/main/kotlin/App.kt",), ("build.gradle.kts",),
    ("pom.xml",), ("README.md", "src/App.java"), ("config/custom.settings",),
    ("requirements.txt",), ("requirements-dev.txt",), ("config/requirements-prod.txt",),
    ("CMakeLists.txt",), ("pyproject.toml",), ("Pipfile",), ("Pipfile.lock",),
    ("setup.py",), ("setup.cfg",), ("settings.gradle.kts",), ("config/runtime.txt",),
    ("src/main/resources/META-INF/services/com.example.Plugin.txt",),
    ("src/main/resources/META-INF/services/com.example.Plugin",),
    ("src/main/resources/META-INF/manifest.md",),
    (r"src\main\resources\META-INF\services\com.example.Plugin.txt",),
    ("docs/examples/requirements-dev.txt",), ("docs/examples/CMakeLists.txt",),
])
def test_lightweight_selector_keeps_source_build_and_uncertain_changes(paths: tuple[str, ...]) -> None:
    from review_bot.application.selection import LightweightModeSelector

    snapshot, pr = selection_inputs(paths)

    assert LightweightModeSelector().select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)


@pytest.mark.parametrize("title, body", [("Refactor module boundaries", ""), ("Update guide", "Describe the new dependency injection architecture")])
def test_lightweight_selector_uses_title_and_body(title: str, body: str) -> None:
    from review_bot.application.selection import LightweightModeSelector

    snapshot, pr = selection_inputs(("README.md",), title, body)

    assert LightweightModeSelector().select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)


def test_lightweight_selector_keeps_structural_rename_from_source_to_docs() -> None:
    from review_bot.application.selection import LightweightModeSelector

    snapshot, pr = selection_inputs(("docs/old-code.md",), renames={"src/Old.java": "docs/old-code.md"})

    assert LightweightModeSelector().select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)


def test_model_selector_invokes_injected_model_with_metadata() -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)
    client.complete.return_value = ModelResponse('{"selected_modes": []}')
    snapshot, pr = selection_inputs(("config/appearance.settings",), "Adjust colours", "Cosmetic settings only")

    result = ModelModeSelector(client, model="selector-fast-v1").select_modes(snapshot, pr, ("architecture",))

    assert result.selected_modes == ()
    client.complete.assert_called_once()
    request = client.complete.call_args.args[0]
    assert request.model == "selector-fast-v1"
    metadata = json.loads(request.user_content)
    assert metadata["title"] == "Adjust colours"
    assert metadata["body"] == "Cosmetic settings only"
    assert metadata["changed_files"] == ["config/appearance.settings"]
    assert metadata["available_modes"] == ["architecture"]
    assert request.timeout_seconds <= 1
    assert request.max_output_tokens <= 64
    assert result.selection_status == "completed"


def test_model_selector_requires_explicit_model() -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)

    with pytest.raises(TypeError, match="model"):
        ModelModeSelector(client)

    client.complete.assert_not_called()


@pytest.mark.parametrize("model", ["", " ", None])
def test_model_selector_rejects_unconfigured_model(model: Any) -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)

    with pytest.raises(ValueError, match="Selector model"):
        ModelModeSelector(client, model=model)

    client.complete.assert_not_called()


@pytest.mark.parametrize("response", [
    "not JSON", '{"selected_modes": ["intent"]}', '{"selected_modes": "architecture"}',
    '{"selected_modes": ["architecture", "architecture"]}', '{}',
])
def test_model_selector_fails_safe_on_invalid_response(response: str) -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)
    client.complete.return_value = ModelResponse(response)
    snapshot, pr = selection_inputs(("custom.settings",))

    result = ModelModeSelector(client, model="selector-fast-v1").select_modes(snapshot, pr, ("architecture",))

    assert result.selected_modes == ("architecture",)
    assert result.selection_status == "completed"


@pytest.mark.parametrize("error", [httpx.ReadTimeout("selector timed out"), RuntimeError("model unavailable")])
def test_model_selector_fails_safe_on_model_error(error: Exception) -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)
    client.complete.side_effect = error
    snapshot, pr = selection_inputs(("custom.settings",))

    assert ModelModeSelector(client, model="selector-fast-v1").select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)


@pytest.mark.parametrize("timeout_seconds", [None, 0.05])
def test_model_selector_deadline_and_single_inflight_call(timeout_seconds: float | None) -> None:
    from review_bot.application.selection import ModelModeSelector

    release = threading.Event()
    finished = threading.Event()

    def blocked_model(request: Any) -> ModelResponse:
        try:
            assert release.wait(timeout=2.0)
            return ModelResponse('{"selected_modes": []}')
        finally:
            finished.set()

    client = Mock(spec=ModelClient)
    client.complete.side_effect = blocked_model
    if timeout_seconds is None:
        selector = ModelModeSelector(client, model="selector-fast-v1")
    else:
        selector = ModelModeSelector(client, model="selector-fast-v1", timeout_seconds=timeout_seconds)
    snapshot, pr = selection_inputs(("custom.settings",))
    try:
        started = time.monotonic()
        assert selector.select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)
        assert time.monotonic() - started < 1.0
        assert selector.select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)
        docs_snapshot, docs_pr = selection_inputs(("README.md",), "Fix typos")
        assert selector.select_modes(docs_snapshot, docs_pr, ("architecture",)).selected_modes == ()
        client.complete.assert_called_once()
    finally:
        release.set()
        assert finished.wait(timeout=1.0)


@pytest.mark.parametrize("fault", ["snapshot", "unregistered", "failed"])
def test_injected_selector_invalid_result_falls_back(fault: str) -> None:
    from review_bot.application.selection import DeadlineModeSelector
    from review_bot.ports import ModeSelectorPort

    snapshot, pr = selection_inputs(("README.md",))
    result = select_modes(snapshot.snapshot_id, ("architecture",))
    if fault == "snapshot":
        result = replace(result, snapshot_id="other-snapshot", selected_modes=())
    elif fault == "unregistered":
        result = replace(result, selected_modes=("intent",))
    else:
        result = replace(result, selection_status="failed", selected_modes=())
    selector = Mock(spec=ModeSelectorPort)
    selector.select_modes.return_value = result

    selected = DeadlineModeSelector(selector).select_modes(snapshot, pr, ("architecture",))

    assert selected.selected_modes == ("architecture",)
    assert selected.selection_status == "completed"


@pytest.mark.parametrize("uses_model", [False, True])
def test_snapshot_selection_respects_config_and_passes_metadata(uses_model: bool, valid_config_yaml: str) -> None:
    from review_bot.ports import ModeSelectorPort

    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "selector_uses_model: false", f"selector_uses_model: {str(uses_model).lower()}",
    ))
    assert config.execution.selector_uses_model is uses_model
    assert config.execution.selector_policy == "all_available"
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("base", "ancestor", "comparable")
    reader.get_changed_files.return_value = (ChangedFile("README.md"),)
    reader.get_rename_map.return_value = {}
    selector = Mock(spec=ModeSelectorPort)

    def select(snapshot: ReviewSnapshot, pr: PrSnapshot, modes: Any) -> Any:
        return replace(select_modes(snapshot.snapshot_id, modes), selected_modes=())

    selector.select_modes.side_effect = select
    snapshot, pr, _, selection = assemble_snapshot_bundle(
        reader, "org/repo", 42, 1, 2, "base", "head", "docs/typos",
        title="Fix typos", body="Correct spelling", selector_policy=config.execution.selector_policy,
        selector_uses_model=config.execution.selector_uses_model, mode_selector=selector,
    )

    if uses_model:
        selector.select_modes.assert_called_once_with(snapshot, pr, ("architecture",))
        pr_text = next(e for e in snapshot.evidence if e.kind == "pr_text")
        assert pr_text.content == {"title": "Fix typos", "body": "Correct spelling"}
        assert selection.selected_modes == ()
    else:
        selector.select_modes.assert_not_called()
        assert selection.selected_modes == ("architecture",)
        assert selection.policy_version == "all_available"


@pytest.mark.parametrize("path", [
    "src/App.java", "requirements.txt", "requirements-dev.txt", "CMakeLists.txt",
    "pyproject.toml", "Pipfile", "Pipfile.lock", "setup.py", "setup.cfg", "build.gradle.kts",
    "src/main/resources/META-INF/services/com.example.Plugin.txt",
    "src/main/resources/META-INF/services/com.example.Plugin",
    "src/main/resources/META-INF/spring.factories", "docs/examples/requirements-dev.txt",
    "docs/examples/CMakeLists.txt",
])
def test_model_selector_cannot_skip_architectural_paths_with_negative_model_prediction(path: str) -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)
    client.complete.return_value = ModelResponse('{"selected_modes": []}')
    snapshot, pr = selection_inputs((path,), "Small cleanup")

    assert ModelModeSelector(client, model="selector-fast-v1").select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)
    client.complete.assert_not_called()


def test_model_selector_oversized_metadata_is_not_truncated_or_sent() -> None:
    from review_bot.application.selection import ModelModeSelector

    client = Mock(spec=ModelClient)
    snapshot, pr = selection_inputs(("custom.settings",), body="x" * 40000)

    assert ModelModeSelector(client, model="selector-fast-v1").select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)
    client.complete.assert_not_called()
