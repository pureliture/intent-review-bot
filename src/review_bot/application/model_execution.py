"""Shared bounded model execution, native metering and delivery receipts.

Review pipelines own their prompts and validation. This module owns admission,
usage accounting, retry bounds and cancellation; it does not select review modes.
"""
from __future__ import annotations

import json
import re
import time
from hashlib import sha256
from copy import deepcopy
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from review_bot.application.architecture_review import ModelProfile, TransientModelError, ModelOutputLimitError, ModelInputCapacityError
from review_bot.application.context import hash_json
from review_bot.application.intent_context import MAX_INPUT_BYTES, IntentInputError, strict_json_loads
from review_bot.domain.intent_models import IntentModelAttempt, IntentStageRecord
from review_bot.domain.models_v1 import ModelRequest, ModelResponse
from review_bot.ports import ModelClient

MAX_OUTPUT_BYTES = 65536
MAX_ATTEMPTS = 3
MAX_REJECTED_OUTPUT_BYTES = 1024 * 1024
OUTPUT_RECOVERY_FINAL_RESERVE = 8192
_NATIVE_CAPACITY = re.compile(r"The prompt is too long: ([1-9][0-9]*), model maximum context length: ([1-9][0-9]*) \(ref: [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\)")


def native_capacity_count(body, context):
    """Recognize only the completed pinned provider's exact rejection protocol."""
    try:
        value = strict_json_loads(body, max_bytes=4096)
        if not isinstance(value, dict) or set(value) != {"error"} or not isinstance(value["error"], str):
            return None
        match = _NATIVE_CAPACITY.fullmatch(value["error"])
        if match is not None:
            count, maximum = map(int, match.groups())
            if context == maximum == 1048576 and maximum < count <= 10**9:
                return count
    except (ValueError, TypeError, OverflowError, RecursionError):
        pass
    return None


def capacity_input_hash(system, user, schema, model, output):
    return hash_json({"system": system, "user": user, "schema": schema, "model": model, "output": output})


def capacity_reservation(record, packet, profile=None, request=None):
    """Validate capacity evidence without converting its unknown usage to zero."""
    if not isinstance(packet, dict) or set(packet) != {"capacity_rejection", "budget_reservation", "witness_hash"}:
        raise ValueError("invalid_native_capacity_evidence")
    proof = packet["capacity_rejection"]
    keys = {"provider", "model", "status", "response_body", "response_sha256", "wire_sha256",
        "input_tokens", "context_tokens", "output_tokens_reserved", "completed", "request_hash", "source_input_hash"}
    if not isinstance(proof, dict) or set(proof) != keys:
        raise ValueError("invalid_native_capacity_evidence")
    count = native_capacity_count(proof["response_body"], proof["context_tokens"])
    if (count is None or proof["input_tokens"] != count or type(proof["input_tokens"]) is not int
            or proof["provider"] != "ollama-cloud" or proof["model"] != "beam:cloud"
            or proof["status"] != 400 or type(proof["status"]) is not int or proof["completed"] is not True
            or type(proof["context_tokens"]) is not int or proof["output_tokens_reserved"] != 1
            or type(proof["output_tokens_reserved"]) is not int
            or proof["response_sha256"] != sha256(proof["response_body"].encode("utf-8")).hexdigest()
            or not isinstance(proof["wire_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", proof["wire_sha256"]) is None
            or proof["request_hash"] != record.request_hash or record.request_hash != record.token_measurement_request_hash
            or not isinstance(proof["source_input_hash"], str) or re.fullmatch(r"[0-9a-f]{64}", proof["source_input_hash"]) is None
            or packet["witness_hash"] != hash_json(proof)
            or packet["budget_reservation"] != {"model_calls": 1, "total_tokens": count + 1}
            or any(type(value) is not int for value in packet["budget_reservation"].values())
            or record.measured_input_tokens != count or record.token_measurement != "measured"
            or record.attempts or record.token_measurement_usage is not None
            or record.token_measurement_delivery != "unknown" or record.token_measurement_model_calls is not None):
        raise ValueError("invalid_native_capacity_evidence")
    if profile is not None and (profile.provider != proof["provider"] or profile.model != proof["model"]
            or (profile.max_input_tokens or 0) + profile.max_output_tokens != proof["context_tokens"]):
        raise ValueError("invalid_native_capacity_profile")
    if request is not None:
        from review_bot.application.model_wire_contract import beam_meter_wire_sha256
        if (proof["wire_sha256"] != beam_meter_wire_sha256(request)
                or proof["source_input_hash"] != capacity_input_hash(request.system_prompt,
                    request.user_content, request.response_schema, request.model, request.max_output_tokens)):
            raise ValueError("invalid_native_capacity_wire")
    return packet["budget_reservation"]


class ModelLimits(Protocol):
    """Read-only structural contract shared by immutable review settings."""

    @property
    def max_input_tokens(self) -> int: ...

    @property
    def max_total_tokens(self) -> int: ...

    @property
    def task_timeout_seconds(self) -> int: ...


class ExecutionLimits(ModelLimits, Protocol):
    @property
    def max_calls(self) -> int: ...


def valid_profile(profile: Any, limits: ModelLimits) -> bool:
    return isinstance(profile, ModelProfile) and all(
        type(value) is int and value > 0 for value in (profile.max_input_bytes, profile.max_output_bytes,
            profile.timeout_seconds, profile.max_attempts, profile.max_output_tokens,
            limits.max_input_tokens, limits.max_total_tokens, limits.task_timeout_seconds)
    ) and profile.max_attempts <= MAX_ATTEMPTS and (
        profile.max_input_tokens is None or type(profile.max_input_tokens) is int and profile.max_input_tokens > 0
    ) and isinstance(profile.provider, str) and 0 < len(profile.provider) <= 128 and profile.provider.isprintable() and isinstance(profile.model, str) and 0 < len(profile.model) <= 256 and profile.model.isprintable() and type(profile.is_mock) is bool


def stage_gate(before_stage: Callable[[], bool] | None, deadline: float) -> str | None:
    if time.monotonic() >= deadline:
        return "task_deadline_exceeded"
    if before_stage is not None:
        try:
            if before_stage() is not True:
                return "stage_gate_rejected"
        except Exception:
            return "stage_gate_rejected"
    return None


def response_usage(response: ModelResponse) -> dict[str, int] | None:
    value = response.usage
    if not isinstance(value, dict) or not {"input_tokens", "output_tokens"} <= set(value) or set(value) - {"input_tokens", "output_tokens", "total_tokens"}:
        return None
    if not all(type(item) is int and 0 <= item <= 10**9 for item in value.values()):
        return None
    if "total_tokens" in value and value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        return None
    return {"input_tokens": value["input_tokens"], "output_tokens": value["output_tokens"]}


def measurement_receipt(record: IntentStageRecord, client: ModelClient, profile: ModelProfile) -> IntentStageRecord:
    try:
        receipt = getattr(client, "token_count_receipt", None)
    except Exception:
        receipt = None
    if receipt is None and profile.is_mock:
        return replace(record, token_measurement_delivery="not_attempted", token_measurement_model_calls=0,
            token_measurement_usage={"input_tokens": 0, "output_tokens": 0})
    if isinstance(receipt, dict) and set(receipt) == {"delivery", "model_calls", "usage"}:
        usage = response_usage(ModelResponse("", receipt["usage"]))
        if receipt["delivery"] in ("not_sent", "sent") and type(receipt["model_calls"]) is int and receipt["model_calls"] == 0 and usage == {"input_tokens": 0, "output_tokens": 0}:
            return replace(record, token_measurement_delivery=receipt["delivery"], token_measurement_model_calls=0, token_measurement_usage=usage)
        if receipt["delivery"] == "sent" and type(receipt["model_calls"]) is int and receipt["model_calls"] == 1 and usage is not None and usage["output_tokens"] <= 1:
            return replace(record, token_measurement_delivery="sent", token_measurement_model_calls=1, token_measurement_usage=usage)
    return replace(record, token_measurement_delivery="unknown", token_measurement_model_calls=None, token_measurement_usage=None)


def model_profile(client: ModelClient | None, limits: ModelLimits) -> ModelProfile | None:
    try:
        active = getattr(client, "profile", None)
        return active if valid_profile(active, limits) else None
    except Exception:
        return None


class ModelExecutionBudget:
    def __init__(self, limits: ExecutionLimits, deadline: float, fence: Callable[[], bool] | None):
        self.limits, self.deadline, self.fence = limits, deadline, fence
        self.records: list[IntentStageRecord] = []
        self.rejected_outputs: dict[int, list[dict]] = {}
        self.capacity_rejections: dict[int, dict] = {}
        self.output_recoveries: list[dict] = []

    def execute_bounded_output(self, request: ModelRequest, client: ModelClient, profile: ModelProfile, *,
        label: str, reserve_calls: int, reserve_tokens: int,
    ) -> tuple[Any | None, int]:
        """One known-limit recovery, keeping all input and the total output cap."""
        body, index = self.execute(request, client, profile, label=label,
            reserve_calls=reserve_calls, reserve_tokens=reserve_tokens)
        if (body is not None or self.records[index].reason != "model_output_limit"
                or not self.rejected_outputs.get(index)
                or getattr(client, "supports_bounded_reasoning", False) is not True
                or request.reasoning_budget_tokens is not None
                or request.max_output_tokens != profile.max_output_tokens
                or request.max_output_tokens <= OUTPUT_RECOVERY_FINAL_RESERVE):
            return body, index
        bounded = replace(request,
            reasoning_budget_tokens=request.max_output_tokens - OUTPUT_RECOVERY_FINAL_RESERVE)
        recovered, next_index = self.execute(bounded, client, profile, label=label + ":output_recovery",
            reserve_calls=reserve_calls, reserve_tokens=reserve_tokens)
        self.output_recoveries.append({"failed_stage_index": index, "recovery_stage_index": next_index,
            "reason": "known_native_output_limit", "output_limit": request.max_output_tokens,
            "reasoning_budget_tokens": bounded.reasoning_budget_tokens,
            "target_final_output_reserve": OUTPUT_RECOVERY_FINAL_RESERVE,
            "source_input_unchanged": (request.system_prompt, request.user_content, request.response_schema) ==
                (bounded.system_prompt, bounded.user_content, bounded.response_schema),
            "source_input_hash": hash_json({"system": request.system_prompt,
                "user": request.user_content, "schema": request.response_schema})})
        return recovered, next_index

    def totals(self) -> tuple[int, int, bool]:
        attempts = [attempt for record in self.records for attempt in record.attempts]
        measured = [record for record in self.records if record.token_measurement != "not_attempted"]
        known = [attempt.usage for attempt in attempts if attempt.usage is not None]
        known.extend(record.token_measurement_usage for record in measured if record.token_measurement_usage is not None)
        calls = len(attempts) + sum(record.token_measurement_model_calls or 0 for record in measured)
        unknown = any(attempt.usage is None or attempt.delivery == "unknown" for attempt in attempts) or any(
            record.token_measurement_usage is None or record.token_measurement_model_calls is None for record in measured)
        return calls, sum(usage["input_tokens"] + usage["output_tokens"] for usage in known), unknown

    def guard(self) -> str | None:
        try:
            return stage_gate(self.fence, self.deadline)
        except AsyncCancelledError:
            return "model_cancelled"

    def operational_totals(self) -> tuple[int, int, bool]:
        """Hard admission bounds; totals() remains actual, possibly unknown usage."""
        calls, tokens, _ = self.totals()
        unknown = any(attempt.usage is None or attempt.delivery == "unknown"
            for record in self.records for attempt in record.attempts)
        for index, record in enumerate(self.records):
            if record.token_measurement == "not_attempted":
                continue
            if record.token_measurement_usage is None or record.token_measurement_model_calls is None:
                try:
                    reservation = capacity_reservation(record, self.capacity_rejections.get(index))
                except (ValueError, TypeError, KeyError):
                    unknown = True
                else:
                    calls += reservation["model_calls"]
                    tokens += reservation["total_tokens"]
        return calls, tokens, unknown

    def measure(self, request: ModelRequest, client: ModelClient, profile: ModelProfile, *,
        label: str, reserve_calls: int, reserve_tokens: int, generate: bool = True,
    ) -> tuple[int | None, int]:
        record = IntentStageRecord(label, prompt_hash=hash_json({"system": request.system_prompt, "schema": request.response_schema}))
        index = len(self.records)
        self.records.append(record)
        reason = self.guard()
        calls, tokens, unknown = self.operational_totals()
        meter_reserve = 0 if profile.is_mock else 1
        if reason is None and unknown:
            reason = "usage_unavailable"
        if reason is None and calls + meter_reserve + int(generate) + reserve_calls > self.limits.max_calls:
            reason = "call_limit_exceeded"
        if reason is None and tokens + request.max_output_tokens + reserve_tokens > self.limits.max_total_tokens:
            reason = "task_token_limit_exceeded"
        remaining = self.deadline - time.monotonic()
        if reason is None and remaining < 1:
            reason = "task_deadline_exceeded"
        actual = replace(request, timeout_seconds=min(profile.timeout_seconds, max(1, int(remaining))))
        size = len(json.dumps(asdict(actual), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        record = replace(record, request_hash=hash_json(asdict(actual)), request_bytes=size)
        if reason is None and size > min(profile.max_input_bytes, MAX_INPUT_BYTES):
            reason = "input_limit_exceeded"
        if reason:
            self.records[index] = replace(record, status="not_run", reason=reason)
            return None, index
        record = replace(record, token_measurement_request_hash=hash_json(asdict(actual)))
        count = None
        count_reason = None
        capacity = None
        try:
            count = client.count_input_tokens(actual)
            if type(count) is not int or count < 0:
                count = None
        except (Exception, AsyncCancelledError) as error:
            if isinstance(error, (CancelledError, AsyncCancelledError)):
                count_reason = "model_cancelled"
            elif isinstance(error, ModelInputCapacityError):
                proof = deepcopy(error.capacity_rejection)
                if isinstance(proof, dict) and type(proof.get("input_tokens")) is int:
                    capacity = {"capacity_rejection": proof, "witness_hash": hash_json(proof),
                        "budget_reservation": {"model_calls": 1, "total_tokens": proof["input_tokens"] + 1}}
                    count = proof["input_tokens"]
        record = measurement_receipt(record, client, profile)
        self.records[index] = replace(record, measured_input_tokens=count,
            token_measurement="measured" if count is not None else "unknown")
        if capacity is not None:
            try:
                capacity_reservation(self.records[index], capacity, profile, actual)
            except (ValueError, TypeError, KeyError):
                count = None
                self.records[index] = replace(self.records[index], measured_input_tokens=None, token_measurement="unknown")
            else:
                self.capacity_rejections[index] = capacity
        reason = self.guard() or count_reason
        calls, tokens, unknown = self.operational_totals()
        if reason is None and unknown:
            reason = "usage_unavailable"
        if reason is None and count is None:
            reason = "token_count_unavailable"
        if index in self.capacity_rejections:
            # The rejected meter is charged its full bound before any fallback.
            reason = reason or ("call_limit_exceeded" if calls + reserve_calls > self.limits.max_calls else
                "task_token_limit_exceeded" if tokens + reserve_tokens > self.limits.max_total_tokens else
                "input_limit_exceeded")
        if reason is None and calls + int(generate) + reserve_calls > self.limits.max_calls:
            reason = "call_limit_exceeded"
        context_limit = min(profile.max_input_tokens or self.limits.max_input_tokens, self.limits.max_input_tokens)
        if reason is None and count > context_limit:
            reason = "input_limit_exceeded"
        if reason is None and tokens + count + request.max_output_tokens + reserve_tokens > self.limits.max_total_tokens:
            reason = "task_token_limit_exceeded"
        self.records[index] = replace(self.records[index], status="not_run" if reason else "completed", reason=reason or "input_measured")
        return (None if reason else count), index

    def execute(self, request: ModelRequest, client: ModelClient, profile: ModelProfile, *,
        label: str, reserve_calls: int, reserve_tokens: int,
    ) -> tuple[Any | None, int]:
        count, index = self.measure(request, client, profile, label=label,
            reserve_calls=reserve_calls, reserve_tokens=reserve_tokens)
        if count is None:
            return None, index
        for number in range(1, profile.max_attempts + 1):
            reason = self.guard()
            calls, tokens, unknown = self.operational_totals()
            if reason is None and unknown:
                reason = "usage_unavailable"
            if reason is None and calls + 1 + reserve_calls > self.limits.max_calls:
                reason = "call_limit_exceeded"
            if reason is None and tokens + count + request.max_output_tokens + reserve_tokens > self.limits.max_total_tokens:
                reason = "task_token_limit_exceeded"
            remaining = self.deadline - time.monotonic()
            if reason is None and remaining < 1:
                reason = "task_deadline_exceeded"
            if reason:
                self.records[index] = replace(self.records[index], status="not_run", reason=reason)
                return None, index
            actual = replace(request, timeout_seconds=min(profile.timeout_seconds, int(remaining)))
            start, started_at = time.monotonic(), datetime.now(UTC).isoformat()
            response, retry, usage, delivery = None, False, None, "unknown"
            reason = "received"
            try:
                response = client.complete(actual)
                if isinstance(response, ModelResponse):
                    delivery, usage = "sent", response_usage(response)
                else:
                    reason = "invalid_model_response"
            except (Exception, AsyncCancelledError) as error:
                reason = "model_cancelled" if isinstance(error, (CancelledError, AsyncCancelledError)) else (
                    "model_timeout" if isinstance(error, TimeoutError) else "model_error")
                try:
                    receipt_delivery = getattr(error, "delivery", None)
                    receipt_usage = response_usage(ModelResponse("", getattr(error, "usage", None)))
                except Exception:
                    receipt_delivery, receipt_usage = None, None
                if receipt_delivery == "not_sent" and receipt_usage in (None, {"input_tokens": 0, "output_tokens": 0}):
                    delivery, usage = "not_sent", {"input_tokens": 0, "output_tokens": 0}
                elif receipt_delivery == "sent" and receipt_usage is not None:
                    delivery, usage = "sent", receipt_usage
                if (isinstance(error, ModelOutputLimitError) and delivery == "sent" and usage is not None
                        and usage["output_tokens"] == actual.max_output_tokens):
                    reason = "model_output_limit"
                    output = getattr(error, "rejected_output", None)
                    if (isinstance(output, dict) and set(output) == {"finish_reason", "content", "reasoning_content"}
                            and output["finish_reason"] == "length"
                            and (output["content"] is None or isinstance(output["content"], str))
                            and (output["reasoning_content"] is None or isinstance(output["reasoning_content"], str))
                            and len(json.dumps(output, ensure_ascii=False).encode("utf-8")) <= MAX_REJECTED_OUTPUT_BYTES):
                        self.rejected_outputs.setdefault(index, []).append({"attempt": number,
                            "request_hash": hash_json(asdict(actual)), "usage": deepcopy(usage),
                            "output_limit": actual.max_output_tokens,
                            "output": deepcopy(output), "output_hash": hash_json(output)})
                retry = isinstance(error, (TimeoutError, TransientModelError)) and usage is not None and delivery != "unknown"
            if reason == "received" and usage is None:
                reason = "usage_unavailable"
            context_limit = min(profile.max_input_tokens or self.limits.max_input_tokens, self.limits.max_input_tokens)
            if usage is not None and (usage["input_tokens"] > context_limit
                    or usage["output_tokens"] > request.max_output_tokens):
                reason, retry = "usage_limit_exceeded", False
            attempt = IntentModelAttempt(number, "completed" if reason == "received" else "failed", delivery, reason,
                started_at, datetime.now(UTC).isoformat(), max(0, int((time.monotonic() - start) * 1000)), usage, hash_json(asdict(actual)))
            record = self.records[index]
            self.records[index] = replace(record, request_hash=attempt.request_hash, attempts=(*record.attempts, attempt))
            gate_reason = self.guard()
            _, used_tokens, unknown = self.operational_totals()
            if gate_reason or unknown or used_tokens + reserve_tokens > self.limits.max_total_tokens:
                reason = gate_reason or ("usage_unavailable" if unknown else "task_token_limit_exceeded")
                retry = False
            if reason == "received":
                try:
                    body = strict_json_loads(response.text, max_bytes=min(profile.max_output_bytes, MAX_OUTPUT_BYTES))
                    if not isinstance(body, dict):
                        raise IntentInputError("invalid_fields")
                    return body, index
                except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
                    reason = "invalid_model_response"
            self.records[index] = replace(self.records[index], status="not_run" if gate_reason else "failed", reason=reason)
            if not retry or number == profile.max_attempts:
                return None, index
        return None, index
