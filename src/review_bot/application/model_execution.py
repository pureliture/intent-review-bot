"""Shared bounded model execution and delivery receipts.

Review pipelines own their prompts and validation. This module owns admission,
usage accounting, retry bounds and cancellation; it does not select review modes.

There is no metering path: no measurement-only request is ever issued for
any client. Admission checks calls, deadlines and transport bytes only.
The input-token budget is proven pre-send by each adapter's exact-wire
byte-upper-bound gate; the response usage is recorded post-send and any
budget exceedance fails closed.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from review_bot.application.architecture_review import ModelProfile, TransientModelError, ModelOutputLimitError, ModelInputCapacityError, PermanentModelError
from review_bot.application.context import hash_json
from review_bot.application.intent_context import MAX_INPUT_BYTES, IntentInputError, strict_json_loads
from review_bot.domain.intent_models import IntentModelAttempt, IntentStageRecord
from review_bot.domain.models_v1 import ModelRequest, ModelResponse
from review_bot.ports import ModelClient

MAX_OUTPUT_BYTES = 65536
MAX_ATTEMPTS = 3
MAX_REJECTED_OUTPUT_BYTES = 1024 * 1024
OUTPUT_RECOVERY_FINAL_RESERVE = 8192


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
        known = [attempt.usage for attempt in attempts if attempt.usage is not None]
        calls = len(attempts)
        unknown = any(attempt.usage is None or attempt.delivery == "unknown" for attempt in attempts)
        return calls, sum(usage["input_tokens"] + usage["output_tokens"] for usage in known), unknown

    def guard(self) -> str | None:
        try:
            return stage_gate(self.fence, self.deadline)
        except AsyncCancelledError:
            return "model_cancelled"

    def operational_totals(self) -> tuple[int, int, bool]:
        """Hard admission bounds over generation attempts only.

        No metering calls exist, so there is no measurement accounting.
        Only the usage riding on real responses is recorded.
        """
        return self.totals()

    def measure(self, request: ModelRequest, client: ModelClient, profile: ModelProfile, *,
        label: str, reserve_calls: int, reserve_tokens: int, generate: bool = True,
    ) -> tuple[bool, int]:
        """Admission without any measurement call.

        Checks fence, call/deadline/transport budgets, then asks the
        client for its exact-wire admission bound (``admission_bytes``:
        wire bytes plus the client's framing reserve, a proven token
        upper bound). The input-token budget is therefore proven pre-send
        without any extra request; the response usage is recorded
        post-send. Returns (admitted, index).
        """
        record = IntentStageRecord(label, prompt_hash=hash_json({"system": request.system_prompt, "schema": request.response_schema}))
        index = len(self.records)
        self.records.append(record)
        reason = self.guard()
        calls, tokens, unknown = self.operational_totals()
        if reason is None and unknown:
            reason = "usage_unavailable"
        if reason is None and calls + int(generate) + reserve_calls > self.limits.max_calls:
            reason = "call_limit_exceeded"
        if reason is None and tokens + request.max_output_tokens + reserve_tokens > self.limits.max_total_tokens:
            reason = "task_token_limit_exceeded"
        remaining = self.deadline - time.monotonic()
        if reason is None and remaining < 1:
            reason = "task_deadline_exceeded"
        actual = replace(request, timeout_seconds=min(profile.timeout_seconds, max(1, int(remaining))))
        size = len(json.dumps(asdict(actual), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        record = replace(record, request_hash=hash_json(asdict(actual)), request_bytes=size,
            token_measurement="not_attempted", measured_input_tokens=None,
            token_measurement_request_hash=None, token_measurement_delivery="not_attempted",
            token_measurement_model_calls=0, token_measurement_usage=None)
        if reason is None and size > min(profile.max_input_bytes, MAX_INPUT_BYTES):
            reason = "input_limit_exceeded"
        bound: int | None = None
        if reason is None:
            try:
                admission = getattr(client, "admission_bytes", None)
                bound = admission(actual) if callable(admission) else None
            except Exception:
                bound = None
                reason = "token_count_unavailable"
            else:
                if bound is not None and (type(bound) is not int or bound < 0):
                    bound = None
                    reason = "token_count_unavailable"
        if reason is None and bound is not None:
            context_limit = min(profile.max_input_tokens or self.limits.max_input_tokens, self.limits.max_input_tokens)
            if bound > context_limit:
                reason = "input_limit_exceeded"
        if reason:
            self.records[index] = replace(record, status="not_run", reason=reason,
                # Keep the computed admission bound on bound-proven overflow
                # so split logic distinguishes it from unverified refusal.
                # No measurement call was made; this is a byte upper bound.
                measured_input_tokens=bound if reason == "input_limit_exceeded" else None)
            return False, index
        self.records[index] = replace(record, measured_input_tokens=bound,
            status="completed", reason="input_measured")
        return True, index

    def execute(self, request: ModelRequest, client: ModelClient, profile: ModelProfile, *,
        label: str, reserve_calls: int, reserve_tokens: int,
    ) -> tuple[Any | None, int]:
        admitted, index = self.measure(request, client, profile, label=label,
            reserve_calls=reserve_calls, reserve_tokens=reserve_tokens)
        if not admitted:
            return None, index
        for number in range(1, profile.max_attempts + 1):
            reason = self.guard()
            calls, tokens, unknown = self.operational_totals()
            if reason is None and unknown:
                reason = "usage_unavailable"
            if reason is None and calls + 1 + reserve_calls > self.limits.max_calls:
                reason = "call_limit_exceeded"
            if reason is None and tokens + request.max_output_tokens + reserve_tokens > self.limits.max_total_tokens:
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
                if isinstance(error, ModelInputCapacityError) or (
                        isinstance(error, PermanentModelError)
                        and "byte-upper-bound" in str(error)):
                    # Proven over-budget pre-send (adapter exact-wire gate)
                    # or provider-side context enforcement. No generation
                    # happened; treat as admission refusal, not a model error.
                    reason = "input_limit_exceeded"
                try:
                    receipt_delivery = getattr(error, "delivery", None)
                    receipt_usage = response_usage(ModelResponse("", getattr(error, "usage", None)))
                except Exception:
                    receipt_delivery, receipt_usage = None, None
                if receipt_delivery == "not_sent" and receipt_usage in (None, {"input_tokens": 0, "output_tokens": 0}):
                    delivery, usage = "not_sent", {"input_tokens": 0, "output_tokens": 0}
                elif receipt_delivery == "sent" and receipt_usage is not None:
                    delivery, usage = "sent", receipt_usage
                if reason == "input_limit_exceeded":
                    retry = False
                    self.records[index] = replace(self.records[index], status="not_run", reason=reason)
                    return None, index
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
            # Output is capped at the profile ceiling: adapters may floor tiny
            # per-request reservations (e.g. mandatory reasoning headroom), so
            # the request value alone is not the effective cap. The adapter
            # still enforces its own exact wire cap pre- and post-send.
            output_limit = max(request.max_output_tokens or 0, profile.max_output_tokens or 0)
            if usage is not None and (usage["input_tokens"] > context_limit
                    or usage["output_tokens"] > output_limit):
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
