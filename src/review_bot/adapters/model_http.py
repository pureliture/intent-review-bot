"""A shared local admission group for every model HTTP request."""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from review_bot.adapters.sqlite.model_admission import SQLiteModelAdmission
from review_bot.application.review_a import PermanentModelError

_REQUEST_CONTEXT: ContextVar[tuple[Callable[[], bool], float] | None] = ContextVar("model_request_context", default=None)


def default_model_admission(max_concurrency: int = 2) -> SQLiteModelAdmission:
    configured = os.environ.get("REVIEW_BOT_MODEL_STATE_DIR")
    if configured is not None:
        if not configured.strip():
            raise ValueError("REVIEW_BOT_MODEL_STATE_DIR must be an absolute private directory")
        root = Path(configured)
    else:
        root = Path.home() / ".local" / "state" / "github-review-bot" / "model-admission"
    if not root.is_absolute() or ".." in root.parts or root == Path("/"):
        raise ValueError("REVIEW_BOT_MODEL_STATE_DIR must be an absolute private directory")
    return SQLiteModelAdmission(root / "requests.sqlite", max_concurrency=max_concurrency)


def set_model_request_context(cancelled: Callable[[], bool], deadline: float) -> Token:
    return _REQUEST_CONTEXT.set((cancelled, deadline))


def reset_model_request_context(token: Token) -> None:
    _REQUEST_CONTEXT.reset(token)


def mark_model_request_sent(ticket: Any) -> None:
    try:
        ticket.mark_sent()
    except BaseException:
        # This function must return before the caller starts HTTP: no send occurred.
        try:
            ticket.finish()
        except Exception:
            pass  # Failed cleanup conservatively retains capacity; preserve the original error.
        raise


@contextmanager
def model_request_slot(owner: Any, deadline: float, *, generates: bool) -> Iterator[tuple[Any, float]]:
    scope = _REQUEST_CONTEXT.get()
    if scope is not None:
        deadline = min(deadline, scope[1])
    callbacks = [getattr(owner, "request_cancelled", None)]
    if scope is not None:
        callbacks.append(scope[0])

    def cancelled() -> bool:
        try:
            return any(callback() is not False for callback in callbacks if callback is not None)
        except Exception:
            raise PermanentModelError("Model request authorization unavailable") from None

    admission = getattr(owner, "request_admission", None)
    if admission is None:
        admission = default_model_admission()
    with admission.request(deadline=deadline, generates=generates, cancelled=cancelled) as ticket:
        if cancelled():
            raise PermanentModelError("Model request cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Model request admission timed out")
        yield ticket, deadline