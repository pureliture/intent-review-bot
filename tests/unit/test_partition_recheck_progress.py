"""Issue #18: a split source recheck must make delivery progress, never re-send the same chunk."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from review_bot.application.intent_context import build_intent_task_input
from review_bot.application.partition_review import run_partition_review
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient

HEAD_PATHS = ("src/caller.py", "src/value.py")


def _large_task():
    text_a = "def call_value(x):\n" + "    x += 1\n" * 60 + "    return x\n"
    text_b = "def value(y):\n" + "    y += 2\n" * 60 + "    return y\n"
    task = make_task()
    files = []
    for f in task.head.files:
        if f.path in HEAD_PATHS:
            content = text_a if f.path == HEAD_PATHS[0] else text_b
            f = replace(f, content=content, content_hash=hashlib.sha256(content.encode()).hexdigest())
        files.append(f)
    inventory = hashlib.sha256(":".join(f"{f.path}:{f.content_hash}" for f in sorted(files, key=lambda x: x.path)).encode()).hexdigest()
    head = replace(task.head, files=tuple(files), inventory_hash=inventory)
    return build_intent_task_input(task.snapshot, task.base, head, related_paths=task.reviewed_paths)


def _client(delivered):
    def always_request_both(payload, body):
        if payload["stage"] != "final_review":
            return body
        test_ref = next(s["ranges"][0]["reference_id"] for s in payload["citation_catalog"] if s["path"] == "tests/test_label.py")
        for item in body["perspectives"]:
            item["citations"] = [{"reference_id": test_ref}]
        for key, review_item in body["target_reviews"].items():
            base = next((s for s in payload["citation_catalog"] if s["path"] == key and s["version"] == "base"), None)
            if base is not None:
                review_item["citations"] = [{"reference_id": base["ranges"][0]["reference_id"]}]
        heads = [row for row in payload["fixed_original_inventory"] if row["version"] == "head" and row["path"] in HEAD_PATHS]
        if "source_recheck" in payload:
            delivered.append(sorted((q["path"], q["start_byte"], q["end_byte"]) for q in payload["original_quotes"]
                if q["version"] == "head" and q["path"] in HEAD_PATHS and q["end_byte"] - q["start_byte"] > 500))
        # A reviewer that never clears its request is the worst case for call growth.
        body["source_requests"] = [{"evidence_id": row["evidence_id"], "reason": "Need full head."} for row in heads]
        return body

    def measured(request):
        payload = json.loads(request.user_content)
        if payload["stage"] == "final_review" and "source_recheck" in payload:
            quotes = [q for q in payload["original_quotes"] if q["path"] in HEAD_PATHS and q["end_byte"] - q["start_byte"] > 500]
            if len(quotes) > 1:
                return 100000
        return 100

    return PartitionModelClient(counter=measured, transform=always_request_both)


def test_split_recheck_delivers_each_chunk_once_and_stops_without_success():
    delivered = []
    review = _client(delivered)
    result = run_partition_review(_large_task(), "intent", review, PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))
    final_calls = [r for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]
    # One initial final review plus one recheck per distinct split chunk.
    assert len(delivered) == len(set(map(tuple, delivered))), delivered
    assert len(final_calls) <= 1 + 2, len(final_calls)
    assert result.status == "partial_review"
    assert result.partition["final"]["source_requests"], "open request must stay visible"
