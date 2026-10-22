"""Local Ollama HTTP contracts; never require a running model server."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from review_bot.application.architecture_review import PermanentModelError, TransientModelError, run_review_a_pipeline
from review_bot.application.selection import ModelModeSelector
from review_bot.domain.models_v1 import ModelRequest
from review_bot.ports import ModelClient
from tests.unit.test_mode_selection import selection_inputs

MAIN = "granite4.2:128k"
SELECTOR = "tev1:0.8b-128k"


def model_info(model: str) -> dict[str, Any]:
    if model == MAIN:
        return {
            "capabilities": ["completion", "thinking"], "parameters": "num_ctx 131072",
            "template": "<|im_start|>system\n{{ system }}<|im_end|>\n{{ messages }}",
            "model_info": {
                "general.architecture": "granite", "granite.context_length": 131072,
                "tokenizer.ggml.model": "gpt2",
            },
        }
    return {
        "capabilities": ["decision"], "parameters": "num_ctx 131072\ntemperature 0",
        "model_info": {"general.architecture": "qwen3_5", "qwen3_5.context_length": 262144},
    }


def load_response(**overrides: Any) -> dict[str, Any]:
    return {
        "model": MAIN, "done": True, "done_reason": "load", "response": "", **overrides,
    }


def chat_response(content: str = '{"ok":true}', **overrides: Any) -> dict[str, Any]:
    return {
        "model": MAIN, "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 42, "eval_count": 5, **overrides,
    }


def decision_response(choice: str = "not_applicable") -> dict[str, Any]:
    return {
        "model": SELECTOR,
        "answers": {"review_mode": {
            "type": "choice", "choice": choice,
            "probabilities": {"architecture": 0.99 if choice == "architecture" else 0.01,
                              "not_applicable": 0.01 if choice == "architecture" else 0.99},
            "confidence": 0.9,
        }},
        "usage": {"input_tokens": 179, "output_tokens": 0},
    }


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    client_type = httpx.Client

    def install(handler: Any) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert request.url.host == "127.0.0.1"
            if request.url.path == "/api/show":
                return httpx.Response(200, json=model_info(json.loads(request.content)["model"]))
            return handler(request)

        def client(**kwargs: Any) -> httpx.Client:
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            return client_type(transport=httpx.MockTransport(dispatch), **kwargs)

        monkeypatch.setattr(httpx, "Client", client)
        return requests

    return install


@pytest.fixture
def main_request() -> ModelRequest:
    return ModelRequest(
        1, MAIN, "Return only JSON.", "Synthetic connectivity request", 10, 32,
        {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )


@pytest.fixture
def decision_request() -> ModelRequest:
    return ModelRequest(
        1, SELECTOR, "Select architectural review.", json.dumps({
            "title": "Adjust colours", "body": "Only presentation settings changed.",
            "changed_files": ["config/appearance.settings"], "available_modes": ["architecture"],
        }), 1, 64, {"type": "object", "properties": {"selected_modes": {"type": "array"}}},
    )


def test_granite_warmup_uses_load_only_request(mock_http: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    requests = mock_http(lambda request: httpx.Response(200, json=load_response()))
    client = OllamaModelClient(timeout_seconds=17)
    assert requests == []
    client.warmup()

    assert [r.url.path for r in requests] == ["/api/show", "/api/generate"]
    assert requests[-1].method == "POST"
    payload = json.loads(requests[-1].content)
    assert payload == {"model": MAIN, "keep_alive": "5m", "options": {"num_ctx": 131072}}
    assert "prompt" not in payload and "stream" not in payload
    assert 0 < requests[-1].extensions["timeout"]["read"] <= 17


@pytest.mark.parametrize("response", [
    load_response(model=SELECTOR), load_response(done=False), load_response(done=1),
    load_response(done_reason="stop"), load_response(response="generated output"), load_response(response=None),
])
def test_granite_warmup_rejects_invalid_load_response(mock_http: Any, response: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    mock_http(lambda request: httpx.Response(200, json=response))
    with pytest.raises(PermanentModelError):
        OllamaModelClient().warmup()


def test_granite_port_posts_chat_schema_and_128k(mock_http: Any, main_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    requests = mock_http(lambda request: httpx.Response(200, json=chat_response()))
    client = OllamaModelClient()
    response = client.complete(main_request)

    assert isinstance(client, ModelClient)
    assert json.loads(response.text) == {"ok": True}
    assert response.usage == {"input_tokens": 42, "output_tokens": 5}
    payload = json.loads(requests[-1].content)
    assert requests[-1].url.path == "/api/chat"
    assert payload["model"] == MAIN
    assert payload["options"]["num_ctx"] == 131072
    assert payload["options"]["num_predict"] == 32
    assert payload["format"] == main_request.response_schema
    assert payload["think"] is False and payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": main_request.system_prompt},
        {"role": "user", "content": main_request.user_content},
    ]
    assert 0 < requests[-1].extensions["timeout"]["read"] <= 10


def test_no_count_only_probe_and_complete_uses_full_budget(mock_http: Any, main_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    requests = mock_http(lambda request: httpx.Response(200, json=chat_response()))
    client = OllamaModelClient()

    assert client.admission_bytes(main_request) is None
    assert requests == []
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")

    response = client.complete(main_request)
    assert json.loads(response.text) == {"ok": True}
    assert response.usage == {"input_tokens": 42, "output_tokens": 5}
    assert [r.url.path for r in requests] == ["/api/show", "/api/chat"]
    assert sum(r.url.path == "/api/chat" for r in requests) == 1
    payload = json.loads(requests[-1].content)
    assert payload["options"]["num_predict"] == 32
    assert payload["options"]["num_ctx"] == 131072
    assert payload["format"] == main_request.response_schema
    assert client.profile.max_input_tokens == 126976
    assert client.profile.max_output_tokens == 4096


def test_granite_pipeline_preserves_token_gate_and_validates_review(mock_http: Any, worker_review_inputs: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    task, expected = worker_review_inputs
    body = json.dumps({
        "base_sha": expected.base_sha, "head_sha": expected.head_sha, "review_mode": expected.review_mode,
        "summary": "Synthetic architecture review.", "findings": [], "questions": [],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(body, eval_count=100))

    requests = mock_http(handler)
    result = run_review_a_pipeline(task, OllamaModelClient())

    assert result.status == "completed", result.validation_errors
    assert result.model["provider"] == "ollama"
    assert result.model["model"] == MAIN
    assert [r.url.path for r in requests] == ["/api/show", "/api/chat"]
    assert sum(r.url.path == "/api/chat" for r in requests) == 1


def test_granite_oversized_input_never_reaches_http(mock_http: Any, main_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    requests = mock_http(lambda request: pytest.fail("Oversized input must not be sent"))
    client = OllamaModelClient()
    assert client.admission_bytes(replace(main_request, user_content="x")) is None
    assert requests == []
    oversized = replace(main_request, user_content="한" * 50000)
    with pytest.raises(PermanentModelError, match="byte budget"):
        client.complete(oversized)
    assert requests == []
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")


@pytest.mark.parametrize("response", [
    chat_response("not JSON"), chat_response("[]"), chat_response(done=False),
    chat_response(model=SELECTOR), chat_response(prompt_eval_count=True),
    chat_response(eval_count=-1), chat_response(done_reason="length"),
    chat_response(message={"role": "assistant", "content": None}),
])
def test_granite_rejects_incomplete_or_invalid_response(mock_http: Any, main_request: ModelRequest, response: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    mock_http(lambda request: httpx.Response(200, json=response))
    with pytest.raises(PermanentModelError):
        OllamaModelClient().complete(main_request)


@pytest.mark.parametrize("status,error", [(400, PermanentModelError), (307, PermanentModelError),
                                         (429, TransientModelError), (503, TransientModelError)])
def test_http_errors_use_pipeline_error_contract(mock_http: Any, main_request: ModelRequest, status: int, error: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    mock_http(lambda request: httpx.Response(status, json={"error": "synthetic failure"}))
    with pytest.raises(error):
        OllamaModelClient().complete(main_request)


def test_transport_timeout_becomes_timeout_error(mock_http: Any, main_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    mock_http(handler)
    with pytest.raises(TimeoutError):
        OllamaModelClient().complete(main_request)


@pytest.mark.parametrize("api_url", ["https://example.com", "http://127.0.0.1.example.com:11434",
                                   "http://user:secret@127.0.0.1:11434", "http://127.0.0.1:11434/api"])
def test_external_or_ambiguous_endpoint_is_rejected(api_url: str) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient, OllamaModelClient

    for client_type in (OllamaModelClient, OllamaDecisionClient):
        with pytest.raises(ValueError):
            client_type(api_url=api_url)
        with pytest.raises(ValueError):
            client_type(model="kimi-k2.6:cloud")


def test_tev_port_uses_systemone_not_chat(mock_http: Any, decision_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    requests = mock_http(lambda request: httpx.Response(200, json=decision_response()))
    client = OllamaDecisionClient()
    response = client.complete(decision_request)

    assert isinstance(client, ModelClient)
    assert json.loads(response.text) == {"selected_modes": []}
    payload = json.loads(requests[-1].content)
    assert requests[-1].url.path == "/v1/systemone"
    assert payload["model"] == SELECTOR
    assert payload["state"] == {
        "title": "Adjust colours", "body": "Only presentation settings changed.",
        "changed_files": ["config/appearance.settings"],
    }
    question = payload["questions"]["review_mode"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"architecture", "not_applicable"}
    assert "options" not in payload  # Context comes from the verified model parameters.
    assert requests[-1].extensions["timeout"]["read"] <= 1
    assert response.usage == {"input_tokens": 179, "output_tokens": 0}


@pytest.mark.parametrize("modes, choice, expected", [
    (["intent"], "intent", ["intent"]),
    (["intent"], "not_applicable", []),
    (["architecture", "intent"], "architecture", ["architecture"]),
    (["architecture", "intent"], "intent", ["intent"]),
    (["architecture", "intent"], "architecture_and_intent", ["architecture", "intent"]),
    (["architecture", "intent"], "not_applicable", []),
    (["intent", "architecture"], "architecture_and_intent", ["intent", "architecture"]),
])
def test_tev_selects_available_intent_subsets_in_one_existing_request(mock_http, decision_request, modes, choice, expected):
    from review_bot.adapters.ollama import OllamaDecisionClient

    choices = ["intent", "not_applicable"] if len(modes) == 1 else ["architecture", "intent", "architecture_and_intent", "not_applicable"]
    data = decision_response()
    data["answers"]["review_mode"].update(choice=choice, probabilities={key: 1.0 if key == choice else 0.0 for key in choices})
    requests = mock_http(lambda request: httpx.Response(200, json=data))
    metadata = json.loads(decision_request.user_content)
    metadata["available_modes"] = modes
    result = OllamaDecisionClient().complete(replace(decision_request, user_content=json.dumps(metadata)))
    assert json.loads(result.text) == {"selected_modes": expected}
    assert [request.url.path for request in requests] == ["/api/show", "/v1/systemone"]
    criteria = json.loads(requests[-1].content)["questions"]["review_mode"]["criteria"]
    assert set(criteria) == set(choices)
    assert "missing" in criteria["intent"].lower()


@pytest.mark.parametrize("modes", [[], ["intent", "intent"], ["intent", "unknown"], ["general"], ["architecture_and_intent"]])
def test_tev_invalid_mode_availability_is_rejected_without_http(mock_http, decision_request, modes):
    from review_bot.adapters.ollama import OllamaDecisionClient

    requests = mock_http(lambda request: pytest.fail("Invalid availability must not be sent"))
    metadata = json.loads(decision_request.user_content)
    metadata["available_modes"] = modes
    with pytest.raises(PermanentModelError):
        OllamaDecisionClient().complete(replace(decision_request, user_content=json.dumps(metadata)))
    assert requests == []


def test_tev_architecture_choice_and_local_prefilter_remain_separate(mock_http: Any) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    requests = mock_http(lambda request: httpx.Response(200, json=decision_response("architecture")))
    selector = ModelModeSelector(OllamaDecisionClient(), model=SELECTOR)
    uncertain, pr = selection_inputs(("config/custom.settings",))
    assert selector.select_modes(uncertain, pr, ("architecture",)).selected_modes == ("architecture",)
    count = len(requests)
    for path, expected in (("README.md", ()), ("src/App.java", ("architecture",))):
        snapshot, pr = selection_inputs((path,))
        assert selector.select_modes(snapshot, pr, ("architecture",)).selected_modes == expected
    assert len(requests) == count


def test_tev_invalid_choice_falls_back_to_architecture(mock_http: Any) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    mock_http(lambda request: httpx.Response(200, json=decision_response("unknown")))
    snapshot, pr = selection_inputs(("config/appearance.settings",))
    selector = ModelModeSelector(OllamaDecisionClient(), model=SELECTOR)
    result = selector.select_modes(snapshot, pr, ("architecture",))
    assert result.selected_modes == ("architecture",)
    assert result.policy_version == "all_available"


@pytest.mark.parametrize("probabilities", [
    {"architecture": 0.99, "not_applicable": 0.01},
    {"architecture": 0.2, "not_applicable": 0.2},
    {"architecture": -0.1, "not_applicable": 1.1},
])
def test_tev_inconsistent_probabilities_cannot_skip(mock_http: Any, probabilities: Any) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    response = decision_response()
    response["answers"]["review_mode"]["probabilities"] = probabilities
    mock_http(lambda request: httpx.Response(200, json=response))
    snapshot, pr = selection_inputs(("config/custom.settings",))
    selector = ModelModeSelector(OllamaDecisionClient(), model=SELECTOR)
    assert selector.select_modes(snapshot, pr, ("architecture",)).selected_modes == ("architecture",)


def test_tev_oversized_body_and_wrong_role_are_not_sent(mock_http: Any, decision_request: ModelRequest) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    requests = mock_http(lambda request: pytest.fail("Invalid state must not be sent"))
    client = OllamaDecisionClient()
    metadata = json.loads(decision_request.user_content)
    metadata["body"] = "x" * 65536
    for request in (replace(decision_request, user_content=json.dumps(metadata)),
                    replace(decision_request, model=MAIN)):
        with pytest.raises(PermanentModelError):
            client.complete(request)
    assert requests == []


def test_tev_warmup_is_explicit_and_outside_selector_deadline(mock_http: Any) -> None:
    from review_bot.adapters.ollama import OllamaDecisionClient

    requests = mock_http(lambda request: httpx.Response(200, json=decision_response()))
    client = OllamaDecisionClient()
    assert requests == []
    client.warmup()
    assert [r.url.path for r in requests] == ["/api/show", "/v1/systemone"]
    assert 0 < requests[-1].extensions["timeout"]["read"] <= 60


@pytest.mark.parametrize("fault", ["context", "parameters", "capability", "remote", "tokenizer"])
def test_model_verification_fails_closed(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from review_bot.adapters.ollama import OllamaModelClient

    info = model_info(MAIN)
    if fault == "context":
        info["model_info"]["granite.context_length"] = 32768
    elif fault == "parameters":
        info["parameters"] = "num_ctx 2048"
    elif fault == "capability":
        info["capabilities"] = ["decision"]
    elif fault == "remote":
        info["remote_host"] = "https://ollama.com"
    else:
        info["model_info"]["tokenizer.ggml.model"] = "unknown"
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=info)), **kwargs,
    ))
    with pytest.raises(PermanentModelError):
        OllamaModelClient().verify()


def test_review_profile_factory_is_explicit_and_uses_operator_limits(mock_http: Any) -> None:
    from review_bot.adapters.ollama import OllamaModelClient
    from review_bot.config import ModelConfig
    from review_bot.runtime.model_clients import build_review_client

    requests = mock_http(lambda request: pytest.fail("Construction must not call models"))
    assert build_review_client(None) is None
    client = build_review_client("ollama-local-128k", ModelConfig(call_timeout_seconds=17, max_output_bytes=12000))
    assert isinstance(client, OllamaModelClient)
    assert client.profile.model == MAIN
    assert client.profile.timeout_seconds == 17
    assert client.profile.max_output_bytes == 12000
    with pytest.raises(ValueError, match="Unsupported model profile"):
        build_review_client("unknown-profile")
    assert requests == []


def test_worker_profile_warms_models_before_claiming_work(mock_http: Any, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from review_bot.adapters.ollama import OllamaModelClient
    from review_bot.config import load_review_config_from_str
    from review_bot.ports import JobStore
    from review_bot.runtime.worker import WorkerLoop

    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: ollama-local-128k",
    ).replace("selector_uses_model: false", "selector_uses_model: true"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json=load_response())
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=decision_response())

    requests = mock_http(handler)
    store = Mock(spec=JobStore)

    def claim(*args: Any) -> None:
        assert [(r.url.path, json.loads(r.content)["model"]) for r in requests] == [
            ("/api/show", MAIN), ("/api/generate", MAIN),
            ("/api/show", SELECTOR), ("/v1/systemone", SELECTOR),
        ]

    store.claim_next_run.side_effect = claim
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    worker = WorkerLoop(config, store)
    assert requests == []
    assert isinstance(worker.model_client, OllamaModelClient)
    assert isinstance(worker.mode_selector, ModelModeSelector)
    assert worker.mode_selector.timeout_seconds == 0.75
    assert worker.claim_and_execute_once() is False
    assert worker.run_once() is False
    assert len(requests) == 4  # Preparation is idempotent, not repeated per job.


@pytest.mark.parametrize("failed_path", ["/api/generate", "/v1/systemone"])
def test_worker_failed_preparation_does_not_claim_jobs(
    mock_http: Any, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch, failed_path: str,
) -> None:
    from unittest.mock import Mock

    from review_bot.config import load_review_config_from_str
    from review_bot.ports import JobStore
    from review_bot.runtime.worker import WorkerLoop

    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: ollama-local-128k",
    ).replace("selector_uses_model: false", "selector_uses_model: true"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == failed_path:
            return httpx.Response(503, json={"error": "synthetic startup failure"})
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json=load_response())

    requests = mock_http(handler)
    store = Mock(spec=JobStore)
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    worker = WorkerLoop(config, store)
    with pytest.raises(TransientModelError):
        worker.run_once()
    store.claim_next_run.assert_not_called()
    assert requests[-1].url.path == failed_path


def test_worker_explicit_injection_overrides_profile_defaults(mock_http: Any, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from review_bot.application.selection import LightweightModeSelector
    from review_bot.config import load_review_config_from_str
    from review_bot.ports import JobStore
    from review_bot.runtime.worker import WorkerLoop

    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: ollama-local-128k",
    ).replace("selector_uses_model: false", "selector_uses_model: true"))
    requests = mock_http(lambda request: pytest.fail("Injected clients must be preserved"))
    client = Mock(spec=ModelClient)
    selector = LightweightModeSelector()
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    worker = WorkerLoop(config, Mock(spec=JobStore), model_client=client, mode_selector=selector)
    worker.prepare_models()
    assert worker.model_client is client
    assert worker.mode_selector.selector is selector
    assert requests == []


def test_review_profile_does_not_implicitly_enable_selector(mock_http: Any, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from review_bot.application.selection import LightweightModeSelector
    from review_bot.config import load_review_config_from_str
    from review_bot.ports import JobStore
    from review_bot.runtime.worker import WorkerLoop

    config, _ = load_review_config_from_str(valid_config_yaml.replace("profile: null", "profile: ollama-local-128k"))
    requests = mock_http(lambda request: pytest.fail("Construction must remain offline"))
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    worker = WorkerLoop(config, Mock(spec=JobStore))
    assert isinstance(worker.mode_selector.selector, LightweightModeSelector)
    assert requests == []


def test_configured_worker_systemone_skip_bypasses_heavy_pipeline(
    mock_http: Any, valid_config_yaml: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from review_bot.adapters.sqlite.store import SQLiteJobStore
    from review_bot.config import load_review_config_from_str
    from review_bot.domain.envelope_v2 import ChangedFile, PrRequestRecord
    from review_bot.ports import GitHubClient, MergeBaseResult, RepositoryReader, StaticAnalyzer
    from review_bot.runtime.worker import WorkerLoop

    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: ollama-local-128k",
    ).replace("selector_uses_model: false", "selector_uses_model: true"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/api/chat", "Skip must never invoke Granite chat"
        if request.url.path == "/api/generate":
            payload = json.loads(request.content)
            assert payload == {"model": MAIN, "keep_alive": "5m", "options": {"num_ctx": 131072}}
            assert "prompt" not in payload and "stream" not in payload
            return httpx.Response(200, json=load_response())
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=decision_response())

    requests = mock_http(handler)
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("a" * 40, "a" * 40, "comparable")
    reader.get_changed_files.return_value = (ChangedFile("config/appearance.settings"),)
    reader.get_rename_map.return_value = {}
    github = Mock(spec=GitHubClient)
    github.get_pull_request.return_value = {
        "state": "open", "head": {"sha": "b" * 40, "ref": "cosmetic"},
        "base": {"sha": "a" * 40, "repo": {"full_name": "org/repo"}},
        "title": "Adjust colours", "body": "Only presentation settings changed.",
    }
    analyzer = Mock(spec=StaticAnalyzer)
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    store = SQLiteJobStore(tmp_path / "ollama-worker.sqlite")
    try:
        worker = WorkerLoop(config, store, reader, analyzer, github_client=github, worker_id="ollama-worker")
        worker.prepare_models()
        store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
        run = store.claim_next_run("ollama-worker", 30)
        assert worker.process_run(run) is True
        state = store._get_connection().execute("SELECT status,reason FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
        assert tuple(state) == ("skipped", "not_applicable")
        assert sum(r.url.path == "/v1/systemone" for r in requests) == 2
        assert sum(r.url.path == "/api/generate" for r in requests) == 1
        assert all(r.url.path != "/api/chat" for r in requests)
        analyzer.run_analysis.assert_not_called()
        reader.collect_code_snapshot.assert_not_called()
        reader.read_tree_blobs.assert_not_called()
        github.post_comment.assert_not_called()
    finally:
        store.close()


def test_cli_exposes_explicit_local_profile_flag() -> None:
    from review_bot.runtime.cli import create_parser

    parser = create_parser()
    assert parser.parse_args(["review"]).model_profile is None
    assert parser.parse_args(["review", "--model-profile", "ollama-local-128k"]).model_profile == "ollama-local-128k"


@pytest.mark.parametrize("activation", ["flag", "config", "unset"])
def test_cli_profile_wires_client_without_changing_offline_default(
    mock_http: Any, worker_pipeline_ports: Any, worker_review_inputs: Any, tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch, capsys: Any, valid_config_yaml: str, activation: str,
) -> None:
    from types import SimpleNamespace

    from review_bot.runtime import cli

    task, expected = worker_review_inputs
    reader = worker_pipeline_ports["repo_reader"]
    reader.get_commit.side_effect = lambda ref: SimpleNamespace(
        sha="b" * 40 if ref == "HEAD" else "a" * 40, parents=("a" * 40,), message="Synthetic change",
    )
    monkeypatch.setattr(cli, "GitRepositoryReader", lambda path: reader)
    monkeypatch.setattr(cli, "PmdRunner", lambda: worker_pipeline_ports["analyzer"])
    monkeypatch.setattr(cli, "build_delta", lambda *args, **kwargs: task.delta)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    result_body = json.dumps({
        "base_sha": expected.base_sha, "head_sha": expected.head_sha, "review_mode": expected.review_mode,
        "summary": "Synthetic local review.", "findings": [], "questions": [],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(result_body, eval_count=100))

    requests = mock_http(handler)
    args = ["review", "--repo", str(repo), "--base", "a" * 40, "--format", "json"]
    if activation == "flag":
        args += ["--model-profile", "ollama-local-128k"]
    elif activation == "config":
        config_path = tmp_path / "operator.yaml"
        config_path.write_text(valid_config_yaml.replace("profile: null", "profile: ollama-local-128k"))
        args += ["--config", str(config_path)]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    if activation == "unset":
        assert result["status"] == "not_run" and result["reason"] == "model_profile_unset"
        assert requests == []
    else:
        assert result["status"] == "completed", result["validation_errors"]
        assert result["model"]["model"] == MAIN
        assert [r.url.path for r in requests] == ["/api/show", "/api/chat"]
        assert sum(r.url.path == "/api/chat" for r in requests) == 1