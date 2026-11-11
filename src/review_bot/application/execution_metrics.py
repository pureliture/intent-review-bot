"""Request-scoped timings; no prompts, model text, credentials or session files."""
from __future__ import annotations

import contextvars
import copy
import threading
import time
from contextlib import contextmanager


class ExecutionMetrics:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.agent_sessions: list[dict] = []
        self.mode_spans: list[dict] = []

    def event(self, stage: str) -> None:
        with self.lock:
            self.events.append({"stage": stage, "review_mode": current_review_mode.get(),
                "elapsed_ms": self.elapsed_ms()})

    def mode_finished(self, mode: str, started: int) -> None:
        with self.lock:
            self.mode_spans.append({"mode": mode, "start_ms": started,
                "duration_ms": self.elapsed_ms() - started})

    def elapsed_ms(self) -> int:
        return max(0, round((time.monotonic() - self.started) * 1000))

    def agent_session(self, value: dict) -> None:
        with self.lock:
            self.agent_sessions.append(copy.deepcopy(value))

    def snapshot(self) -> dict:
        with self.lock:
            sessions = copy.deepcopy(self.agent_sessions)
            calls = [call for session in sessions for call in session["usage"].get("calls", [])]
            return {"schema_version": 1, "wall_ms": self.elapsed_ms(),
                "stage_events": list(self.events), "mode_spans": list(self.mode_spans), "agent_sessions": sessions,
                "model_calls": len(calls),
                # A+B overlap: summed API duration is work, not run wall time.
                "model_duration_ms": sum(call.get("duration_ms", 0) or 0 for call in calls),
                "known_input_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
                "known_output_tokens": sum(call.get("completion_tokens", 0) for call in calls),
                "unknown_usage_calls": sum("prompt_tokens" not in call or "completion_tokens" not in call
                    for call in calls)}


current_execution_metrics: contextvars.ContextVar[ExecutionMetrics | None] = contextvars.ContextVar(
    "execution_metrics", default=None)
current_agent_stage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("agent_stage", default=None)
current_review_mode: contextvars.ContextVar[str | None] = contextvars.ContextVar("review_mode", default=None)


@contextmanager
def agent_stage(mode: str, stage: str):
    token = current_agent_stage.set({"review_mode": mode, "stage": stage})
    try:
        yield
    finally:
        current_agent_stage.reset(token)
