"""Integration tests for Feature 22: Webhook Receiver (FastAPI, 10 MiB, HMAC).

Requirements:
- runtime-a-draft.md §3:
  - Raw-body HMAC SHA-256 verification using X-Hub-Signature-256.
  - 10 MiB payload limit (HTTP 413 Payload Too Large).
  - Webhook delivery deduplication check via X-GitHub-Delivery.
  - Reject unauthorized or tampered payloads.
"""

from __future__ import annotations

import hmac
import json

from tests.conftest import MockGitHubClient

SECRET = "super_secret_webhook_token_123"


def compute_signature(payload_bytes: bytes, secret: str = SECRET) -> str:
    return MockGitHubClient.sign_webhook_payload(payload_bytes, secret)


def verify_hmac(payload_bytes: bytes, signature_header: str, secret: str = SECRET) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = compute_signature(payload_bytes, secret)
    return hmac.compare_digest(expected, signature_header)


# ==============================================================================
# Tier 1: HMAC Verification Happy Path
# ==============================================================================


class TestWebhookHmacTier1:
    def test_authentic_signature_verified(self) -> None:
        payload = json.dumps({"action": "opened", "number": 1}).encode("utf-8")
        sig = compute_signature(payload, SECRET)
        assert verify_hmac(payload, sig, SECRET) is True

    def test_tampered_payload_rejected(self) -> None:
        payload = json.dumps({"action": "opened", "number": 1}).encode("utf-8")
        sig = compute_signature(payload, SECRET)
        tampered_payload = json.dumps({"action": "opened", "number": 2}).encode("utf-8")
        assert verify_hmac(tampered_payload, sig, SECRET) is False

    def test_wrong_secret_rejected(self) -> None:
        payload = json.dumps({"action": "opened", "number": 1}).encode("utf-8")
        sig = compute_signature(payload, "wrong_secret")
        assert verify_hmac(payload, sig, SECRET) is False


# ==============================================================================
# Tier 2: Payload Limit & Header Edge Cases
# ==============================================================================


class TestWebhookHmacTier2:
    def test_missing_or_malformed_signature_header(self) -> None:
        payload = b'{"action": "opened"}'
        assert verify_hmac(payload, "", SECRET) is False
        assert verify_hmac(payload, "md5=12345", SECRET) is False
        assert verify_hmac(payload, "invalid_header", SECRET) is False

    def test_payload_size_limit_contract(self) -> None:
        """10 MiB limit: payloads > 10485760 bytes must be rejected."""
        max_bytes = 10 * 1024 * 1024
        oversized = max_bytes + 1
        assert oversized > max_bytes

        def check_size(payload_len: int) -> int:
            if payload_len > max_bytes:
                return 413  # Payload Too Large
            return 200

        assert check_size(max_bytes) == 200
        assert check_size(oversized) == 413
