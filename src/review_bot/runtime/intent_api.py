"""Authenticated fixed-input review intake using the existing durable worker queue."""
from __future__ import annotations

import hmac
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from review_bot.application.intent_context import MAX_INPUT_BYTES, decode_intent_task_input
from review_bot.ports import JobStore
from review_bot.runtime.artifacts import read_private_intent_input


API_TOKEN_ENV = "REVIEW_BOT_API_TOKEN"


def attach_intent_api(app: FastAPI, config, store: JobStore, artifact_root: Path) -> None:
    """Opt-in private API; each accepted POST creates one durable review, never a subprocess."""
    if config.intent.pipeline != "perspectives-v1" or config.publish.mode != "artifact_only":
        raise ValueError("intent_api_requires_artifact_only_perspectives")
    if not os.environ.get(API_TOKEN_ENV, "").strip():
        raise ValueError("intent_api_requires_token")

    def authorize(request: Request) -> None:
        token = os.environ.get(API_TOKEN_ENV, "")
        supplied = request.headers.get("Authorization", "")
        if not token or not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    router = APIRouter(prefix="/reviews/intent", dependencies=[Depends(authorize)])
    limit = min(MAX_INPUT_BYTES, config.service.webhook_max_bytes)

    def state(review_id: str) -> dict:
        if not re.fullmatch(r"run-[0-9a-f]{32}", review_id):
            raise HTTPException(404, "Review not found")
        try:
            record = store.get_fixed_intent_status(review_id)
        except sqlite3.Error:
            raise HTTPException(503, "Review storage unavailable") from None
        if record is None:
            raise HTTPException(404, "Review not found")
        return record

    @router.post("", status_code=202)
    async def submit(request: Request, language: Literal["ko", "en"] = "ko"):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "Input too large")
            chunks.append(chunk)
        try:
            text = b"".join(chunks).decode("utf-8")
            task = decode_intent_task_input(text)
        except (ValueError, TypeError, KeyError, RecursionError):
            raise HTTPException(422, "Invalid fixed intent input") from None
        try:
            review_id = store.enqueue_fixed_intent(text, task.snapshot.repository_id, task.snapshot.pr_number, language)
        except sqlite3.Error:
            raise HTTPException(503, "Review storage unavailable") from None
        url = f"/reviews/intent/{review_id}"
        return JSONResponse(status_code=202, headers={"Location": url}, content={
            "review_id": review_id, "status": "queued", "status_url": url, "result_url": url + "/result"})

    @router.get("/{review_id}")
    def get_status(review_id: str):
        return state(review_id)

    @router.get("/{review_id}/result")
    def get_result(review_id: str):
        record = state(review_id)
        if record["status"] not in ("succeeded", "failed"):
            raise HTTPException(409, "Review is not finished")
        path = artifact_root / "fixed" / review_id / f"attempt-{record['attempts']}" / "B/result.json"
        try:
            return JSONResponse(json.loads(read_private_intent_input(path, max_bytes=16 * 1024 * 1024)))
        except (OSError, ValueError):
            raise HTTPException(404, "Review result unavailable") from None

    app.include_router(router)


def create_intent_app(config, store: JobStore, artifact_root: Path) -> FastAPI:
    app = FastAPI(title="review-bot", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    attach_intent_api(app, config, store, artifact_root)
    return app
