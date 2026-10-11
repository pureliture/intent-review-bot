"""FastAPI Webhook Receiver server implementation with raw body HMAC and size gating."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from review_bot.adapters.github.webhook import (
    PullRequestEvent,
    is_installation_allowed,
    is_repository_allowed,
    verify_github_signature,
)
from review_bot.config import ConfigValidationError, ReviewBotConfig
from review_bot.domain.envelope_v2 import DeliveryRecord, PrRequestRecord
from review_bot.ports import JobStore


def create_webhook_app(
    config: ReviewBotConfig | Any,
    job_store: JobStore,
    secret: str | None = None,
) -> FastAPI:
    """Factory creating configured FastAPI webhook application."""
    app = FastAPI(title="review-bot Webhook Receiver", version="0.1.0")

    max_bytes: int = getattr(config.service, "webhook_max_bytes", 10485760)
    secret_env_name: str = getattr(
        config.service, "webhook_secret_env", "REVIEW_BOT_WEBHOOK_SECRET"
    )

    def _get_secret() -> str:
        if secret is not None:
            return secret
        if hasattr(config.service, "webhook_secret") and config.service.webhook_secret:
            return str(config.service.webhook_secret)
        return os.environ.get(secret_env_name, "")

    active = getattr(getattr(config, "execution", None), "mode", "offline") == "github_app"
    if active and not _get_secret():
        raise ConfigValidationError("github_app receiver requires a webhook secret")

    def _finish(
        delivery: DeliveryRecord, response: Response, pr_request: PrRequestRecord | None = None,
        *, cancel: bool = False, revoke: bool = False, repository_ids: tuple[int, ...] | None = None,
    ) -> Response:
        try:
            if revoke:
                job_store.accept_revocation(delivery, repository_ids)
            elif cancel:
                job_store.accept_cancellation(delivery)
            elif pr_request is None:
                job_store.record_delivery(delivery)
            else:
                job_store.accept_delivery(delivery, pr_request)
        except sqlite3.IntegrityError:
            if job_store.is_delivery_processed(delivery.delivery_id):
                return JSONResponse(status_code=202, content={"status": "deduplicated"})
            return Response(content="Request storage unavailable", status_code=503)
        except sqlite3.Error:
            return Response(content="Request storage unavailable", status_code=503)
        return response

    def _payload_id(value: Any) -> int:
        identifier = value.get("id", 0) if isinstance(value, dict) else 0
        return identifier if type(identifier) is int and identifier > 0 else 0

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhook")
    async def handle_webhook(request: Request) -> Response:
        # 1. Content-Length preliminary check
        content_length_header = request.headers.get("Content-Length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
                if content_length > max_bytes:
                    return Response(
                        content="Payload Too Large",
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    )
            except ValueError:
                pass

        # 2. Stream consumption with byte count enforcement
        chunks: list[bytes] = []
        total_received = 0
        async for chunk in request.stream():
            total_received += len(chunk)
            if total_received > max_bytes:
                return Response(
                    content="Payload Too Large",
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
            chunks.append(chunk)
        raw_body = b"".join(chunks)

        # 3. Required Headers extraction
        event_name = request.headers.get("X-GitHub-Event")
        delivery_id = request.headers.get("X-GitHub-Delivery")
        signature_header = request.headers.get("X-Hub-Signature-256")

        if not event_name or not delivery_id:
            return Response(
                content="Missing required headers",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # 4. HMAC Verification
        configured_secret = _get_secret()
        if active and not configured_secret:
            return Response(content="Webhook secret unavailable", status_code=503)
        if configured_secret:
            if not signature_header or not verify_github_signature(
                raw_body, signature_header, configured_secret
            ):
                return Response(
                    content="Invalid HMAC signature",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
        elif signature_header:
            return Response(
                content="Invalid HMAC signature",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        # 5. Parse JSON payload
        try:
            payload_json = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(
                content="Malformed JSON payload",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(payload_json, dict):
            return Response(content="Invalid event payload", status_code=400)

        # 6. Ping Event Handling
        if event_name == "ping":
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "pong"})

        # 7. Prepare delivery metadata, without persisting raw private payloads.
        inst_id = _payload_id(payload_json.get("installation"))
        repo_id = _payload_id(payload_json.get("repository"))
        pr_num = payload_json.get("number", 0)
        if type(pr_num) is not int or pr_num < 0:
            pr_num = 0
        action_name = payload_json.get("action", "")
        if not isinstance(action_name, str):
            action_name = ""

        delivery = DeliveryRecord(
            delivery_id=delivery_id,
            payload_digest=hashlib.sha256(raw_body).hexdigest(),
            received_at=datetime.now(UTC).isoformat(),
            action=action_name,
            installation_id=inst_id,
            repository_id=repo_id,
            pr_number=pr_num,
            processed=False,
        )

        # 8. Signed revocation events stop only this installation's pending work.
        if event_name in ("installation", "installation_repositories"):
            if inst_id <= 0:
                return _finish(delivery, JSONResponse(status_code=202, content={"status": "ignored", "reason": "unlisted"}))
            if event_name == "installation" and action_name in ("deleted", "suspend"):
                return _finish(
                    delivery, JSONResponse(status_code=202, content={"status": "cancelled", "reason": "installation_revoked"}),
                    revoke=True,
                )
            if event_name == "installation_repositories" and action_name == "removed":
                removed = payload_json.get("repositories_removed", [])
                if not isinstance(removed, list) or any(_payload_id(repo) <= 0 for repo in removed):
                    return _finish(delivery, Response(content="Invalid removal event", status_code=400))
                return _finish(
                    delivery, JSONResponse(status_code=202, content={"status": "cancelled", "reason": "installation_revoked"}),
                    revoke=True, repository_ids=tuple(_payload_id(repo) for repo in removed),
                )

        if event_name != "pull_request":
            return _finish(delivery, JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "ignored", "reason": "unhandled_event"},
            ))

        # 9. Pull Request Event Parsing
        try:
            pr_event = PullRequestEvent.model_validate(payload_json)
        except ValidationError:
            return _finish(delivery, Response(
                content="Invalid PR event schema",
                status_code=status.HTTP_400_BAD_REQUEST,
            ))

        # 10. Installation eligibility, then optional operator restrictions.
        inst_id_val = pr_event.installation.id if pr_event.installation else None
        repo_allowlist = getattr(config.service, "repository_allowlist", [])
        inst_allowlist = getattr(config.service, "installation_allowlist", [])
        if (
            inst_id_val is None or inst_id_val <= 0 or pr_event.repository.id <= 0
            or not is_repository_allowed(pr_event.repository.full_name, repo_allowlist)
            or not is_installation_allowed(inst_id_val, inst_allowlist)
        ):
            return _finish(delivery, JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "ignored", "reason": "unlisted"},
            ))

        # 11. PR Action Filtering
        actions = getattr(config.service, "pr_actions", (
            "opened", "synchronize", "reopened", "edited", "labeled", "unlabeled", "ready_for_review",
        ))
        if pr_event.action != "closed" and pr_event.action in actions:
            pr_request = PrRequestRecord(
                installation_id=inst_id_val,
                repository_id=pr_event.repository.id,
                pr_number=pr_event.number,
                desired_generation=1,
            )
            return _finish(delivery, JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "accepted"},
            ), pr_request)

        if pr_event.action == "closed":
            return _finish(delivery, JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "cancelled", "reason": "closed"},
            ), cancel=True)

        return _finish(delivery, JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "ignored", "reason": "unhandled_action"},
        ))

    return app
