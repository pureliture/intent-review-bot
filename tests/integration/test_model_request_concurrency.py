"""Actual HTTP adapters share SQLite admission, including selection and metering."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import pytest

from review_bot.adapters.llamacpp import LlamaCppDecisionClient, LlamaCppModelClient
from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.adapters.sqlite.model_admission import ModelAdmissionCancelled, SQLiteModelAdmission
from review_bot.application.architecture_review import PermanentModelError, TransientModelError
from review_bot.application.selection import DeadlineModeSelector, ModelModeSelector
from review_bot.domain.models_v1 import ModelRequest
from review_bot.runtime import model_clients
from review_bot.config import ModelConfig
from tests.unit.test_ollama_client import selection_inputs


def _request(*, selector=False, timeout=5):
    metadata = {"title": "Optional", "body": "untrusted", "changed_files": ["a.py"],
        "available_modes": ["architecture", "intent"]} if selector else {}
    return ModelRequest(schema_version=1, system_prompt="trusted instructions",
        user_content=json.dumps(metadata) if selector else "fixed source",
        model="tev1-4b-q8_0" if selector else "bonsai2-27b", timeout_seconds=timeout,
        max_output_tokens=64 if selector else 4096, response_schema={"type": "object"})


def _response(request, *, content=None):
    selector = request.url.port == 11437
    model = "tev1-4b-q8_0" if selector else "bonsai2-27b"
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"object": "list", "data": [{"id": model, "object": "model"}]})
    if request.url.path == "/props":
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 131072 if selector else 131072},
            "chat_template": "fixture-chat-template", "total_slots": 2})
    if request.url.path == "/apply-template":
        return httpx.Response(200, json={"prompt": json.dumps(json.loads(request.content)["messages"])})
    if request.url.path == "/tokenize":
        return httpx.Response(200, json={"tokens": list(range(100))})
    assert request.url.path == "/v1/chat/completions"
    return httpx.Response(200, json={"model": model, "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": content if content is not None else
        ('{"selected_modes":["intent"]}' if selector else '{"ok":true}'),
    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}})


def _install_http(monkeypatch, handler):
    client_type = httpx.Client

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)


def test_review_selector_and_metering_share_exactly_two_http_slots(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    third_started = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum = 0
    requests = []

    def handler(request):
        nonlocal active, maximum
        with lock:
            requests.append(request)
            number = len(requests)
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                entered.set()
        try:
            if number <= 2:
                assert release.wait(timeout=5)
            return _response(request)
        finally:
            with lock:
                active -= 1

    _install_http(monkeypatch, handler)
    monkeypatch.setenv("ME707_BONSAI_BASE_URL", "http://ubuntu-ms-7e07:11435/v1")
    monkeypatch.setenv("ME707_TEV1_BASE_URL", "http://ubuntu-ms-7e07:11437/v1")
    settings = ModelConfig(max_concurrency=2)
    review = model_clients.build_review_client("me707-llamacpp", settings)
    selector = model_clients.build_selector_client("me707-llamacpp", settings)
    third = model_clients.build_review_client("me707-llamacpp", settings)

    def execute_third():
        third_started.set()
        return third.complete(_request())

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(review.complete, _request())
        second = pool.submit(selector.complete, _request(selector=True))
        try:
            assert entered.wait(timeout=5)
            waiting = pool.submit(execute_third)
            assert third_started.wait(timeout=5)
            assert not waiting.done()
            with lock:
                assert active == maximum == 2
                assert len(requests) == 2
        finally:
            release.set()
        assert json.loads(first.result(timeout=5).text) == {"ok": True}
        assert json.loads(second.result(timeout=5).text) == {"selected_modes": ["intent"]}
        assert json.loads(waiting.result(timeout=5).text) == {"ok": True}
    assert maximum == 2
    assert len([request for request in requests if request.url.path == "/v1/chat/completions"]) == 3
    assert len([request for request in requests if request.url.path == "/tokenize"]) == 3


@pytest.mark.parametrize("failure", ["timeout", "full_invalid_json", "http_error", "connect_error"])
def test_only_unknown_sent_generation_retains_capacity(monkeypatch, tmp_path, failure):
    requests = []
    fail = [True]

    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/chat/completions" and fail[0]:
            if failure == "timeout":
                raise httpx.ReadTimeout("private transport detail", request=request)
            if failure == "connect_error":
                raise httpx.ConnectError("private transport detail", request=request)
            if failure == "http_error":
                return httpx.Response(503, json={"error": "private transport detail"})
            return httpx.Response(200, content=b"not-json")
        return _response(request)

    _install_http(monkeypatch, handler)
    admission = SQLiteModelAdmission(tmp_path / "slots" / "requests.sqlite", max_concurrency=1)
    client = LlamaCppModelClient(api_url="http://ubuntu-ms-7e07:11435/v1", timeout_seconds=1)
    client.request_admission = admission
    error = TimeoutError if failure == "timeout" else (TransientModelError if failure in ("http_error", "connect_error") else PermanentModelError)
    with pytest.raises(error) as caught:
        client.complete(_request(timeout=1))
    assert "private transport detail" not in str(caught.value)
    after_failure = len(requests)
    fail[0] = False
    if failure == "timeout":
        with pytest.raises(TimeoutError):
            client.complete(_request(timeout=1))
        assert len(requests) == after_failure
    else:
        assert json.loads(client.complete(_request(timeout=1)).text) == {"ok": True}
        assert len(requests) > after_failure


def test_queued_selector_inherits_worker_fence_and_never_sends_after_cancellation(monkeypatch, tmp_path):
    admission = SQLiteModelAdmission(tmp_path / "slots" / "requests.sqlite", max_concurrency=1)
    entered = threading.Event()
    cancelled = threading.Event()
    requests = []

    class RecordingAdmission:
        @contextmanager
        def request(self, **options):
            entered.set()
            with admission.request(**options) as ticket:
                yield ticket

    _install_http(monkeypatch, lambda request: requests.append(request) or _response(request))
    client = LlamaCppDecisionClient(api_url="http://ubuntu-ms-7e07:11437/v1", timeout_seconds=2)
    client.request_admission = RecordingAdmission()
    selector = DeadlineModeSelector(ModelModeSelector(client, model="tev1-4b-q8_0"), timeout_seconds=0.75)
    snapshot, pr = selection_inputs(("config/custom.settings",))

    def select():
        token = set_model_request_context(cancelled.is_set, time.monotonic() + 5)
        try:
            return selector.select_modes(snapshot, pr, ("architecture", "intent"))
        finally:
            reset_model_request_context(token)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with admission.request(deadline=time.monotonic() + 5, generates=False):
            future = pool.submit(select)
            assert entered.wait(timeout=5)
            cancelled.set()
        selection = future.result(timeout=2)
    assert selection.selected_modes == ("architecture", "intent")
    assert requests == []


def test_slot_wait_reduces_the_actual_http_timeout(monkeypatch, tmp_path):
    admission = SQLiteModelAdmission(tmp_path / "slots" / "requests.sqlite", max_concurrency=1)
    entered = threading.Event()
    observed = []

    class RecordingAdmission:
        @contextmanager
        def request(self, **options):
            entered.set()
            with admission.request(**options) as ticket:
                yield ticket

    def handler(request):
        observed.append(request.extensions["timeout"]["read"])
        return _response(request)

    _install_http(monkeypatch, handler)
    client = LlamaCppModelClient(api_url="http://ubuntu-ms-7e07:11435/v1", timeout_seconds=2)
    client.request_admission = RecordingAdmission()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with admission.request(deadline=time.monotonic() + 5, generates=False):
            future = pool.submit(client.verify)
            assert entered.wait(timeout=5)
            time.sleep(0.2)
        future.result(timeout=3)
    assert observed and all(0 < timeout <= 1.85 for timeout in observed)


def test_cancel_at_send_fence_does_not_poison_capacity_without_an_http_send(monkeypatch, tmp_path):
    requests = []
    cancel = [True]
    _install_http(monkeypatch, lambda request: requests.append(request) or _response(request))
    admission = SQLiteModelAdmission(tmp_path / "slots" / "requests.sqlite", max_concurrency=1)
    original = admission._mark_sent

    def cancel_after_committed_send_fence(token, deadline, generates, cancelled):
        original(token, deadline, generates, cancelled)
        if generates and cancel[0]:
            raise ModelAdmissionCancelled("Model request cancelled before HTTP")

    monkeypatch.setattr(admission, "_mark_sent", cancel_after_committed_send_fence)
    client = LlamaCppModelClient(api_url="http://ubuntu-ms-7e07:11435/v1", timeout_seconds=1)
    client.request_admission = admission
    with pytest.raises(ModelAdmissionCancelled):
        client.complete(_request(timeout=1))
    assert not any(request.url.path == "/v1/chat/completions" for request in requests)
    cancel[0] = False
    assert json.loads(client.complete(_request(timeout=1)).text) == {"ok": True}
