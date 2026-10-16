"""Literal producer/caller fixtures; only model responses are simulated."""
from __future__ import annotations

import json
from dataclasses import replace

from review_bot.application.architecture_review import ModelProfile
from review_bot.application.intent_context import build_intent_task_input
from review_bot.domain.intent_perspectives import PERSPECTIVES
from review_bot.domain.models_v1 import ModelResponse
from tests.fixtures.code_first_intent_review import BASE_SOURCE, HEAD_SOURCE, make_task as make_code_task
from tests.fixtures.intent_review import digest


def make_task(*, repair=False, body="렌더링 값의 문자열 계약을 유지합니다."):
    task = make_code_task(body=body)
    if not repair:
        return task
    def swap(snapshot, content):
        files = tuple(replace(file, content=content, content_hash=digest(content)) if file.path == "src/value.py" else file
            for file in snapshot.files)
        inventory = digest(":".join(f"{file.path}:{file.content_hash}" for file in sorted(files, key=lambda file: file.path)))
        return replace(snapshot, files=files, inventory_hash=inventory)
    return build_intent_task_input(task.snapshot, swap(task.base, HEAD_SOURCE), swap(task.head, BASE_SOURCE),
        related_paths=("src/caller.py", "tests/test_label.py"))


def cites(payload):
    return [{"evidence_id": row["id"], "start_line": 2 if row["locator"]["path"] == "src/value.py" else 4,
        "end_line": 2 if row["locator"]["path"] == "src/value.py" else 4}
        for row in payload["evidence"] if row["locator"].get("version") == "head" and row["locator"].get("path") in ("src/value.py", "src/caller.py")]


def review_body(payload, *, repair=False):
    support = cites(payload)
    judgement_rows = []
    previous = {row["perspective_id"]: row for row in payload["previous_judgements"]}
    for view in PERSPECTIVES:
        concern = view.id == "outcome_connection" and not repair
        verdict = "concern" if concern else "aligned"
        failure = "label(7) calls strip on a dict in the head version." if concern else ""
        prior = previous.get(view.id)
        correction = "이전 해석을 고정 head 원문과 대조하여 정정했습니다." if prior and (prior["verdict"], prior["failure_condition"]) != (verdict, failure) else ""
        judgement_rows.append({"perspective_id": view.id, "verdict": verdict,
            "statement": "변경된 반환값이 호출부의 문자열 계약과 충돌합니다." if concern else "제공된 원문에서 이 관점의 추가 불일치는 확인하지 못했습니다.",
            "reasoning": "고정된 생산자와 호출부의 실제 반환값 및 사용 방식을 대조했습니다.",
            "failure_condition": failure, "citations": support, "correction_reason": correction})
    return {"narrative": {key: {"statement": text, "citations": support} for key, text in (
            ("intent", "문자열 렌더링 계약을 유지하려는 변경으로 설명되어 있습니다."),
            ("implementation", "변경 전후 반환값과 호출부의 문자열 연산을 함께 확인했습니다."),
            ("assessment", "head의 반환값과 호출부가 일치합니다." if repair else "head의 반환값과 호출부가 충돌합니다."))},
        "perspective_decisions": [{"perspective_id": view.id, "decision": "apply", "reasoning": "해당 목적과 구현의 연결을 원문으로 검토합니다.", "citations": support} for view in PERSPECTIVES],
        "judgements": judgement_rows, "questions": [], "context_gaps": [], "context_updates": [], "requested_paths": []}


class PerspectiveModelClient:
    def __init__(self, *, selector=False, choices=None, repair=False, transform=None, counter=None, profile=None):
        self.selector, self.choices, self.repair = selector, choices or {}, repair
        self.transform, self.counter = transform, counter
        self.profile = profile or ModelProfile("mock", "intent-choice-fixture" if selector else "intent-review-fixture",
            max_output_tokens=1024, is_mock=True)
        self.requests, self.count_requests = [], []

    def count_input_tokens(self, request):
        self.count_requests.append(request)
        return self.counter(request) if self.counter else 100

    def complete(self, request):
        self.requests.append(request)
        payload = json.loads(request.user_content)
        body = {"choice": self.choices.get(payload["perspective_id"], "A")} if self.selector else review_body(payload, repair=self.repair)
        value = self.transform(payload, body) if self.transform else body
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, ModelResponse) else ModelResponse(json.dumps(value, ensure_ascii=False),
            {"input_tokens": 100, "output_tokens": 5 if self.selector else 100})
