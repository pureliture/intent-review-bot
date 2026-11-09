"""Runtime package for review_bot."""
from __future__ import annotations

__all__ = [
    "WorkerLoop",
    "create_webhook_app",
]


def __getattr__(name: str):
    if name == "WorkerLoop":
        from review_bot.runtime.worker import WorkerLoop

        return WorkerLoop
    if name == "create_webhook_app":
        from review_bot.runtime.webhook import create_webhook_app

        return create_webhook_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
