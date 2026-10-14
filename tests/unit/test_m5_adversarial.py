"""Empirical adversarial stress-test suite for Milestone 5:
- review_bot.runtime.webhook (FastAPI Webhook Receiver, streaming limit, HMAC, allowlist)
- review_bot.runtime.worker (WorkerLoop daemon, lease heartbeat, generation fencing)
- review_bot.adapters.sqlite.store (SQLiteJobStore WAL, deduplication, coalescing, leases)
- review_bot.adapters.github.webhook (HMAC-SHA256, payload schemas, allowlist filtering)

Adversarial Dimensions:
1. Payload Streaming Limit & Memory Exhaustion Stress Testing (>10 MiB, boundary, chunking, deceptive headers)
2. HMAC Authentication & Cryptographic Integrity Stress Testing (missing, invalid, tampered byte, missing prefix, wrong secret)
3. Header Validation & Request Sanitization Stress Testing (missing event/delivery, malformed JSON, schema deviations)
4. Deduplication & Allowlist Policy Stress Testing (replay, unlisted repo/installation, prefix spoofing)
5. Worker Generation Fencing Race Conditions & Superseded Invalidation (advance desired_generation, freshness check)
6. Worker Lease Heartbeat & Slow Execution Stress Testing (slow stage lease extension, lease revocation, thread safety)
7. SQLite JobStore Concurrency & Mutual Exclusion Stress Testing (WAL concurrency, thread-safe claiming)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from review_bot.adapters.github.webhook import (
    is_installation_allowed,
    is_repository_allowed,
    verify_github_signature,
)
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import DeliveryRecord, PrRequestRecord
from review_bot.domain.status import RunStatus
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import MockGitHubClient

SECRET = "super_secret_m5_challenge_token_987"


@pytest.fixture(name="valid_config_yaml")
def comment_config_yaml(valid_config_yaml: str) -> str:
    """Publication stress scenarios deliberately opt into comment mode."""
    return valid_config_yaml.replace("mode: artifact_only", "mode: comment")


def compute_signature(payload_bytes: bytes, secret: str = SECRET) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_sample_pr_payload(
    pr_number: int = 101,
    repo_id: int = 1000,
    repo_name: str = "acme/review-repo",
    installation_id: int = 500,
    action: str = "opened",
    head_sha: str = "head111111111111111111111111111111111111",
    base_sha: str = "base000000000000000000000000000000000000",
) -> dict[str, Any]:
    return {
        "action": action,
        "number": pr_number,
        "pull_request": {
            "number": pr_number,
            "state": "open",
            "head": {"sha": head_sha, "ref": "feature/challenge"},
            "base": {"sha": base_sha, "ref": "main"},
        },
        "repository": {
            "id": repo_id,
            "name": repo_name.split("/")[-1],
            "full_name": repo_name,
        },
        "installation": {"id": installation_id},
    }


@pytest.fixture
def test_setup(
    tmp_path: Path, valid_config_yaml: str
) -> tuple[TestClient, SQLiteJobStore, Any]:
    db_file = tmp_path / "adversarial_m5.db"
    store = SQLiteJobStore(db_file)
    config, _ = load_review_config_from_str(valid_config_yaml)
    app = create_webhook_app(config, store, secret=SECRET)
    client = TestClient(app)
    return client, store, config


# ==============================================================================
# Dimension 1: Payload Streaming Limit & Memory Exhaustion Stress Testing
# ==============================================================================


class TestPayloadStreamingAndMemoryStress:
    """Stress-test streaming payload limits, memory consumption, and boundary conditions."""

    def test_streaming_payload_exceeding_10mib_returns_413(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        chunk_size = 64 * 1024  # 64 KiB
        num_chunks = 170  # 170 * 64 KiB = 11,141,120 bytes > 10 MiB (10,485,760 bytes)

        def stream_generator():
            for _ in range(num_chunks):
                yield b"A" * chunk_size

        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "stream-oversized-1",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=stream_generator(), headers=headers)
        assert resp.status_code == 413
        assert "Payload Too Large" in resp.text

    def test_streaming_boundary_exact_10mib_accepted_vs_10mib_plus_one_byte_rejected(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        max_bytes = 10485760  # Exactly 10 MiB

        # 1. Exactly max_bytes + 1 byte -> Must return HTTP 413
        def stream_over_boundary():
            # 10 MiB + 1 byte
            chunk = b"X" * (1024 * 1024)
            for _ in range(10):
                yield chunk
            yield b"X"

        headers_over = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "boundary-over-1",
            "Content-Type": "application/json",
        }
        resp_over = client.post("/webhook", content=stream_over_boundary(), headers=headers_over)
        assert resp_over.status_code == 413
        assert resp_over.text == "Payload Too Large"

        # 2. Exactly max_bytes -> Passes size check (proceeds to header/HMAC/JSON validation)
        # We supply an HMAC and valid headers so it doesn't fail early
        # len(b'{"zen":"') is 8, len(b'"}') is 2, total 10 bytes
        large_json_body = b'{"zen":"' + (b"A" * (max_bytes - 10)) + b'"}'
        assert len(large_json_body) == max_bytes

        headers_exact = {
            "X-GitHub-Event": "ping",
            "X-GitHub-Delivery": "boundary-exact-1",
            "X-Hub-Signature-256": compute_signature(large_json_body, SECRET),
            "Content-Type": "application/json",
        }
        resp_exact = client.post("/webhook", content=large_json_body, headers=headers_exact)
        # Should NOT be 413; passes streaming limit and ping returns 200
        assert resp_exact.status_code == 200
        assert resp_exact.json() == {"status": "pong"}

    def test_content_length_header_gating_immediate_rejection(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        # If Content-Length header is > 10 MiB, receiver must reject with 413 immediately
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "header-too-large",
            "Content-Length": str(10485761),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=b"minimal", headers=headers)
        assert resp.status_code == 413
        assert resp.text == "Payload Too Large"

    def test_deceptive_content_length_header_enforces_actual_stream_limit(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        # Attacker advertises small Content-Length (e.g. 50 bytes) but streams 11 MiB
        chunk_size = 64 * 1024
        num_chunks = 175

        def stream_generator():
            for _ in range(num_chunks):
                yield b"D" * chunk_size

        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deceptive-cl",
            "Content-Length": "50",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=stream_generator(), headers=headers)
        # Stream check detects total > 10 MiB and returns 413
        assert resp.status_code == 413

    def test_malformed_content_length_headers_fallback_to_stream_check(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        # Negative Content-Length
        headers_neg = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "neg-cl",
            "Content-Length": "-100",
            "Content-Type": "application/json",
        }
        resp_neg = client.post("/webhook", content=b"invalid json", headers=headers_neg)
        # Should not crash with 500; reaches json parser and returns 400
        assert resp_neg.status_code in (400, 401)

        # Non-numeric Content-Length
        headers_str = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "str-cl",
            "Content-Length": "ten-megabytes",
            "Content-Type": "application/json",
        }
        resp_str = client.post("/webhook", content=b"invalid json", headers=headers_str)
        assert resp_str.status_code in (400, 401)


# ==============================================================================
# Dimension 2: HMAC Authentication & Cryptographic Integrity Stress Testing
# ==============================================================================


class TestHmacAuthenticationAdversarial:
    """Stress-test HMAC authentication, header parsing, and cryptographic invariants."""

    def test_missing_signature_header_returns_401(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "hmac-missing",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 401
        assert "Invalid HMAC signature" in resp.text

    def test_empty_signature_header_returns_401(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "hmac-empty",
            "X-Hub-Signature-256": "",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 401
        assert "Invalid HMAC signature" in resp.text

    def test_missing_sha256_prefix_returns_401(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        raw_hex = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        # Raw hex digest without 'sha256=' prefix
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "hmac-no-prefix",
            "X-Hub-Signature-256": raw_hex,
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 401
        assert "Invalid HMAC signature" in resp.text

    @pytest.mark.parametrize(
        "invalid_prefix",
        [
            "sha1=",
            "sha512=",
            "md5=",
            "SHA256=",
            "sha256:",
            "bearer ",
            "hmac-sha256=",
        ],
    )
    def test_invalid_prefix_variants_rejected(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any], invalid_prefix: str
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        raw_hex = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": f"prefix-{invalid_prefix}",
            "X-Hub-Signature-256": f"{invalid_prefix}{raw_hex}",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 401

    def test_tampered_payload_single_byte_flip_returns_401(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        original_body = json.dumps(build_sample_pr_payload(pr_number=555)).encode()
        valid_sig = compute_signature(original_body, SECRET)

        # 1. Flip first byte
        tampered_1 = bytearray(original_body)
        tampered_1[0] = tampered_1[0] ^ 0x01
        headers_1 = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "tampered-1",
            "X-Hub-Signature-256": valid_sig,
            "Content-Type": "application/json",
        }
        resp_1 = client.post("/webhook", content=bytes(tampered_1), headers=headers_1)
        assert resp_1.status_code == 401

        # 2. Flip middle byte
        mid_idx = len(original_body) // 2
        tampered_2 = bytearray(original_body)
        tampered_2[mid_idx] = tampered_2[mid_idx] ^ 0x01
        headers_2 = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "tampered-2",
            "X-Hub-Signature-256": valid_sig,
            "Content-Type": "application/json",
        }
        resp_2 = client.post("/webhook", content=bytes(tampered_2), headers=headers_2)
        assert resp_2.status_code == 401

        # 3. Flip last byte
        tampered_3 = bytearray(original_body)
        tampered_3[-1] = tampered_3[-1] ^ 0x01
        headers_3 = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "tampered-3",
            "X-Hub-Signature-256": valid_sig,
            "Content-Type": "application/json",
        }
        resp_3 = client.post("/webhook", content=bytes(tampered_3), headers=headers_3)
        assert resp_3.status_code == 401

    def test_wrong_secret_signature_returns_401(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        wrong_sig = compute_signature(body, secret="wrong_token_entirely_123")
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "wrong-secret-1",
            "X-Hub-Signature-256": wrong_sig,
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 401
        assert "Invalid HMAC signature" in resp.text

    def test_verify_github_signature_unit_contract(self) -> None:
        body = b"hello github webhook security"
        sig = compute_signature(body, "my_secret")

        assert verify_github_signature(body, sig, "my_secret") is True
        assert verify_github_signature(body, sig, "different_secret") is False
        assert verify_github_signature(body, None, "my_secret") is False
        assert verify_github_signature(body, "", "my_secret") is False
        assert verify_github_signature(body, sig, "") is False
        assert verify_github_signature(b"", compute_signature(b"", "sec"), "sec") is True


# ==============================================================================
# Dimension 3: Header Validation & Request Sanitization Stress Testing
# ==============================================================================


class TestHeaderAndPayloadValidationStress:
    """Stress-test header presence, formatting, and malformed payload resilience."""

    def test_missing_event_header_returns_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Delivery": "delivery-has-no-event",
            "X-Hub-Signature-256": compute_signature(body, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 400
        assert "Missing required headers" in resp.text

    def test_empty_event_header_returns_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Event": "",
            "X-GitHub-Delivery": "delivery-empty-event",
            "X-Hub-Signature-256": compute_signature(body, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 400

    def test_missing_delivery_header_returns_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": compute_signature(body, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 400
        assert "Missing required headers" in resp.text

    def test_empty_delivery_header_returns_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload()).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "",
            "X-Hub-Signature-256": compute_signature(body, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        "malformed_content",
        [
            b'{"action": "open',  # Truncated
            b"{'action': 'opened'}",  # Single quotes (invalid JSON)
            b'{"action": opened}',  # Unquoted string
            b"",  # Empty body
            b"not json at all",  # Plain text
            b"\x80\x81\xff\xfe",  # Invalid UTF-8 bytes
        ],
    )
    def test_malformed_json_payloads_return_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any], malformed_content: bytes
    ) -> None:
        client, _, _ = test_setup
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "malformed-json-test",
            "X-Hub-Signature-256": compute_signature(malformed_content, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=malformed_content, headers=headers)
        assert resp.status_code == 400
        assert "Malformed JSON payload" in resp.text

    def test_invalid_pr_schema_returns_400(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        # Valid JSON but missing required PR model fields (action, number, repository, pull_request)
        invalid_schema = json.dumps({"foo": "bar", "unknown": 123}).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "invalid-schema-1",
            "X-Hub-Signature-256": compute_signature(invalid_schema, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=invalid_schema, headers=headers)
        assert resp.status_code == 400
        assert "Invalid PR event schema" in resp.text


# ==============================================================================
# Dimension 4: Deduplication & Allowlist Policy Stress Testing
# ==============================================================================


class TestDeduplicationAndAllowlistStress:
    """Stress-test delivery deduplication, replay prevention, and allowlists."""

    def test_exact_delivery_deduplication_returns_202_deduplicated(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, _, _ = test_setup
        body = json.dumps(build_sample_pr_payload(pr_number=202)).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "unique-delivery-202",
            "X-Hub-Signature-256": compute_signature(body, SECRET),
            "Content-Type": "application/json",
        }

        # First request accepted
        resp1 = client.post("/webhook", content=body, headers=headers)
        assert resp1.status_code == 202
        assert resp1.json() == {"status": "accepted"}

        # Duplicate delivery deduplicated
        resp2 = client.post("/webhook", content=body, headers=headers)
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "deduplicated"}

        # Triplicate delivery deduplicated
        resp3 = client.post("/webhook", content=body, headers=headers)
        assert resp3.status_code == 202
        assert resp3.json() == {"status": "deduplicated"}

    def test_duplicate_delivery_id_with_tampered_body_deduplicated(
        self, test_setup: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        client, store, _ = test_setup
        body1 = json.dumps(build_sample_pr_payload(pr_number=301)).encode()
        headers1 = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "replay-id-301",
            "X-Hub-Signature-256": compute_signature(body1, SECRET),
            "Content-Type": "application/json",
        }
        resp1 = client.post("/webhook", content=body1, headers=headers1)
        assert resp1.status_code == 202
        assert resp1.json() == {"status": "accepted"}

        # Attacker tries to reuse delivery ID with a different PR payload
        body2 = json.dumps(build_sample_pr_payload(pr_number=999)).encode()
        headers2 = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "replay-id-301",
            "X-Hub-Signature-256": compute_signature(body2, SECRET),
            "Content-Type": "application/json",
        }
        resp2 = client.post("/webhook", content=body2, headers=headers2)
        # Primary key on delivery_id prevents replay/processing
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "deduplicated"}

        # PR 999 was never enqueued
        assert store.get_pr_request(1000, 999) is None

    def test_repository_allowlist_filtering_and_prefix_spoof_defense(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        db_file = tmp_path / "allowlist_repo.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        config.service.repository_allowlist = ["org/authorized-repo"]

        app = create_webhook_app(config, store, secret=SECRET)
        client = TestClient(app)

        # 1. Unlisted repository -> Ignored unlisted
        body_unlisted = json.dumps(
            build_sample_pr_payload(repo_name="evil-org/evil-repo")
        ).encode()
        headers_unlisted = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "unlisted-deliv-1",
            "X-Hub-Signature-256": compute_signature(body_unlisted, SECRET),
            "Content-Type": "application/json",
        }
        resp1 = client.post("/webhook", content=body_unlisted, headers=headers_unlisted)
        assert resp1.status_code == 202
        assert resp1.json() == {"status": "ignored", "reason": "unlisted"}

        # 2. Prefix spoofing: 'org/authorized-repo-fake' -> Must be rejected as unlisted
        body_spoof = json.dumps(
            build_sample_pr_payload(repo_name="org/authorized-repo-fake")
        ).encode()
        headers_spoof = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "spoof-deliv-1",
            "X-Hub-Signature-256": compute_signature(body_spoof, SECRET),
            "Content-Type": "application/json",
        }
        resp2 = client.post("/webhook", content=body_spoof, headers=headers_spoof)
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "ignored", "reason": "unlisted"}

        # 3. Exact authorized repo -> Accepted
        body_auth = json.dumps(
            build_sample_pr_payload(repo_name="org/authorized-repo")
        ).encode()
        headers_auth = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "auth-deliv-1",
            "X-Hub-Signature-256": compute_signature(body_auth, SECRET),
            "Content-Type": "application/json",
        }
        resp3 = client.post("/webhook", content=body_auth, headers=headers_auth)
        assert resp3.status_code == 202
        assert resp3.json() == {"status": "accepted"}

        store.close()

    def test_installation_allowlist_filtering(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        db_file = tmp_path / "allowlist_inst.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        config.service.installation_allowlist = [7777]

        app = create_webhook_app(config, store, secret=SECRET)
        client = TestClient(app)

        # Unlisted installation 9999
        body_unlisted = json.dumps(
            build_sample_pr_payload(installation_id=9999)
        ).encode()
        headers_unlisted = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "unlisted-inst-1",
            "X-Hub-Signature-256": compute_signature(body_unlisted, SECRET),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body_unlisted, headers=headers_unlisted)
        assert resp.status_code == 202
        assert resp.json() == {"status": "ignored", "reason": "unlisted"}

        # Listed installation 7777
        body_listed = json.dumps(
            build_sample_pr_payload(installation_id=7777)
        ).encode()
        headers_listed = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "listed-inst-1",
            "X-Hub-Signature-256": compute_signature(body_listed, SECRET),
            "Content-Type": "application/json",
        }
        resp2 = client.post("/webhook", content=body_listed, headers=headers_listed)
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "accepted"}

        store.close()

    def test_allowlist_helper_pure_functions(self) -> None:
        assert is_repository_allowed("a/b", []) is True
        assert is_repository_allowed("a/b", ["a/b"]) is True
        assert is_repository_allowed("a/b-extra", ["a/b"]) is False

        assert is_installation_allowed(10, []) is True
        assert is_installation_allowed(10, [10, 20]) is True
        assert is_installation_allowed(30, [10, 20]) is False
        assert is_installation_allowed(None, [10, 20]) is False


# ==============================================================================
# Dimension 5: Worker Generation Fencing Race Conditions & Superseded Invalidation
# ==============================================================================


class TestWorkerGenerationFencingRaceStress:
    """Stress-test generation fencing when new commits arrive during execution."""

    def test_generation_fencing_aborts_during_prepare_stage(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        db_file = tmp_path / "fencing_prepare.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=1, head_sha="sha-1")

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="fencing-worker",
            lease_duration=10.0,
        )

        req = PrRequestRecord(installation_id=1, repository_id=10, pr_number=1, desired_generation=1)
        store.enqueue_request(req)

        # Worker claims run gen 1
        run = store.claim_next_run("fencing-worker", lease_seconds=10.0)
        assert run is not None
        assert run.generation == 1

        # While worker is starting, a new push advances desired_generation to 2
        store.enqueue_request(req)
        assert store.get_pr_request(10, 1).desired_generation == 2

        # Process the run
        result = worker.process_run(run)
        assert result is False

        # Run must be marked superseded in SQLite
        conn = store._get_connection()
        row = conn.execute("SELECT status, error, lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        assert row[0] == "superseded"
        assert row[2] is None  # Lease cleared

        # No comment published
        assert len(mock_gh.posted_comments) == 0

        # Publication receipts table must be empty
        cur = conn.execute("SELECT COUNT(*) FROM publication_receipts;")
        assert cur.fetchone()[0] == 0
        store.close()

    def test_generation_fencing_aborts_at_mid_pipeline_checkpoint(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        db_file = tmp_path / "fencing_mid.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=2, head_sha="sha-mid-1")

        req = PrRequestRecord(installation_id=1, repository_id=20, pr_number=2, desired_generation=1)
        store.enqueue_request(req)

        run = store.claim_next_run("mid-worker", lease_seconds=10.0)
        assert run is not None

        # Create a mock analyzer that simulates a new commit arriving during analysis
        mock_analyzer = MagicMock()

        def simulate_commit_during_analysis(*args, **kwargs):
            # Advance generation to 2 in JobStore while analyzer is running!
            store.enqueue_request(req)
            return MagicMock()

        mock_analyzer.run_analysis.side_effect = simulate_commit_during_analysis

        # Setup repo_reader mock to pass PREPARE and SELECT stages
        mock_repo = MagicMock()
        mock_repo.compute_merge_base.return_value = "base-sha"
        mock_repo.get_commit.return_value = MagicMock(sha="head-sha", message="msg", author_email="a@b.com", authored_at="")

        worker = WorkerLoop(
            config=config,
            job_store=store,
            repo_reader=mock_repo,
            analyzer=mock_analyzer,
            github_client=mock_gh,
            worker_id="mid-worker",
            lease_duration=10.0,
        )

        result = worker.process_run(run)
        assert result is False

        # Verify run marked superseded
        conn = store._get_connection()
        status = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()[0]
        assert status == "superseded"

        # Verify publication was NEVER reached
        assert len(mock_gh.posted_comments) == 0
        receipts = conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0]
        assert receipts == 0
        store.close()

    def test_pre_publication_freshness_check_detects_head_sha_change(
        self, tmp_path: Path, valid_config_yaml: str, worker_pipeline_ports: dict[str, Any],
    ) -> None:
        db_file = tmp_path / "freshness_head.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()

        # Initial PR head is fixed before publication
        mock_gh.set_pull_request(pr_number=3, head_sha="b" * 40, base_sha="c" * 40)

        req = PrRequestRecord(installation_id=1, repository_id=30, pr_number=3, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("fresh-worker", lease_seconds=10.0)
        assert run is not None

        # Simulate PR head advancing on GitHub right before publication
        # When get_pull_request is called in PREPARE, it sees the original head.
        # We hook get_pull_request so subsequent calls return sha-updated!
        call_count = 0

        def dynamic_get_pr(pr_num):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "number": pr_num,
                    "state": "open",
                    "head": {"sha": "b" * 40, "ref": "feat"},
                    "base": {"sha": "c" * 40, "ref": "main"},
                }
            # Later call at PUBLISH stage:
            return {
                "number": pr_num,
                "state": "open",
                "head": {"sha": "d" * 40, "ref": "feat"},
                "base": {"sha": "c" * 40, "ref": "main"},
            }

        mock_gh.get_pull_request = dynamic_get_pr

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="fresh-worker",
            lease_duration=10.0,
            **worker_pipeline_ports,
        )

        result = worker.process_run(run)
        assert result is False

        # Must be marked superseded due to freshness check failure
        conn = store._get_connection()
        row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        assert row[0] == "superseded"
        assert "Freshness check failed" in row[1]
        assert len(mock_gh.posted_comments) == 0
        store.close()

    def test_pre_publication_freshness_check_detects_closed_pr(
        self, tmp_path: Path, valid_config_yaml: str, worker_pipeline_ports: dict[str, Any],
    ) -> None:
        db_file = tmp_path / "freshness_closed.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()

        req = PrRequestRecord(installation_id=1, repository_id=35, pr_number=4, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("fresh-closed-worker", lease_seconds=10.0)
        assert run is not None

        call_count = 0

        def dynamic_get_pr(pr_num):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "number": pr_num,
                    "state": "open",
                    "head": {"sha": "b" * 40, "ref": "feat"},
                    "base": {"sha": "c" * 40, "ref": "main"},
                }
            return {
                "number": pr_num,
                "state": "closed",
                "head": {"sha": "b" * 40, "ref": "feat"},
                "base": {"sha": "c" * 40, "ref": "main"},
            }

        mock_gh.get_pull_request = dynamic_get_pr

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="fresh-closed-worker",
            lease_duration=10.0,
            **worker_pipeline_ports,
        )

        result = worker.process_run(run)
        assert result is False

        conn = store._get_connection()
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        assert row[0] == "superseded"
        store.close()

    def test_concurrent_worker_race_resolution(
        self, tmp_path: Path, valid_config_yaml: str, worker_pipeline_ports: dict[str, Any],
    ) -> None:
        """Fence stale gen 1 before admitting gen 2, and reject every later old-owner write."""
        db_file = tmp_path / "concurrent_fencing.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=5, head_sha="b" * 40, base_sha="c" * 40)

        req = PrRequestRecord(installation_id=1, repository_id=40, pr_number=5, desired_generation=1)
        store.enqueue_request(req)

        # Worker A claims gen 1
        worker_a = WorkerLoop(
            config, store, github_client=mock_gh, worker_id="worker-A", lease_duration=10.0,
            **worker_pipeline_ports,
        )
        run_a = store.claim_next_run("worker-A", lease_seconds=10.0)
        assert run_a is not None
        assert run_a.generation == 1

        # Advance generation to 2 (creates gen 2 queued run)
        store.enqueue_request(req)
        assert store.get_pr_request(40, 5).desired_generation == 2

        # Gen 2 waits while the stale owner still holds this PR's lease.
        worker_b = WorkerLoop(
            config, store, github_client=mock_gh, worker_id="worker-B", lease_duration=10.0,
            **worker_pipeline_ports,
        )
        assert store.claim_next_run("worker-B", lease_seconds=10.0) is None

        # Worker A tries to execute -> Fenced!
        result_a = worker_a.process_run(run_a)
        assert result_a is False

        run_b = store.claim_next_run("worker-B", lease_seconds=10.0)
        assert run_b is not None
        assert run_b.generation == 2
        assert store.update_run(
            run_a.run_id, RunStatus.SUCCEEDED, worker_id="worker-A", generation=run_a.generation,
        ) is False

        # Worker B tries to execute -> Succeeds!
        result_b = worker_b.process_run(run_b)
        assert result_b is True

        conn = store._get_connection()
        status_a = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run_a.run_id,)).fetchone()[0]
        status_b = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run_b.run_id,)).fetchone()[0]
        assert status_a == "superseded"
        assert status_b == "succeeded"

        # Exactly 1 publication receipt recorded
        receipts = conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0]
        assert receipts == 1
        assert conn.execute("SELECT comment_id FROM publication_receipts;").fetchone()[0] > 0
        assert len(mock_gh.posted_comments) == 1
        store.close()


# ==============================================================================
# Dimension 6: Worker Lease Heartbeat & Slow Execution Stress Testing
# ==============================================================================


class TestWorkerLeaseHeartbeatAndSlowExecutionStress:
    """Stress-test background lease renewal thread under slow execution and lease theft."""

    def test_heartbeat_renews_lease_under_slow_execution(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        db_file = tmp_path / "slow_exec.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=6, head_sha="sha-slow")

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="slow-worker",
            lease_duration=0.3,  # Very short 300ms lease
        )

        req = PrRequestRecord(installation_id=1, repository_id=50, pr_number=6, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("slow-worker", lease_seconds=0.3)
        assert run is not None

        # Start heartbeat
        heartbeat_stop, heartbeat_thread = worker._start_heartbeat(run.run_id)

        # Simulate slow step running for 0.7s (more than 2x initial lease duration)
        time.sleep(0.7)

        # Lease in database must still be valid (in the future) thanks to heartbeat!
        conn = store._get_connection()
        expires_at = conn.execute("SELECT lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()[0]
        now = time.time()
        assert expires_at > now, f"Lease expired: {expires_at} <= {now}"

        # Clean shutdown
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        assert not heartbeat_thread.is_alive()
        store.close()

    def test_heartbeat_thread_cleanly_terminated_after_run(
        self, tmp_path: Path, valid_config_yaml: str, worker_pipeline_ports: dict[str, Any],
    ) -> None:
        db_file = tmp_path / "heartbeat_cleanup.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=7, head_sha="b" * 40, base_sha="c" * 40)

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="cleanup-worker",
            lease_duration=0.5,
            **worker_pipeline_ports,
        )

        req = PrRequestRecord(installation_id=1, repository_id=60, pr_number=7, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("cleanup-worker", lease_seconds=0.5)

        result = worker.process_run(run)
        assert result is True

        # Ensure no zombie heartbeat threads remain
        time.sleep(0.05)
        heartbeat_threads = [t for t in threading.enumerate() if t.name.startswith("heartbeat-")]
        assert len(heartbeat_threads) == 0
        store.close()

    def test_lease_revocation_or_theft_aborts_worker(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        """Simulate another process stealing the lease while worker is running."""
        db_file = tmp_path / "lease_theft.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()
        mock_gh.set_pull_request(pr_number=8, head_sha="sha-theft")

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="theft-worker",
            lease_duration=0.3,
        )

        req = PrRequestRecord(installation_id=1, repository_id=70, pr_number=8, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("theft-worker", lease_seconds=0.3)
        assert run is not None

        # Start heartbeat
        heartbeat_stop, heartbeat_thread = worker._start_heartbeat(run.run_id)

        # Simulate lease theft: change worker_id to someone else in the DB
        conn = store._get_connection()
        conn.execute("UPDATE runs SET worker_id = 'thief-worker' WHERE run_id = ?;", (run.run_id,))

        # Wait for heartbeat cycle (interval = 0.1s) to attempt renewal
        time.sleep(0.2)

        # Heartbeat should fail to renew and mark _lease_lost = True
        assert worker._lease_lost is True

        # Now test that _check_fencing_and_lease aborts
        valid = worker._check_fencing_and_lease(run.run_id, run.generation)
        assert valid is False

        # A worker that no longer owns the lease cannot mutate the successor's state.
        row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        assert row == ("running", None)
        assert conn.execute(
            "SELECT worker_id FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone()[0] == "thief-worker"

        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        store.close()


# ==============================================================================
# Dimension 7: SQLite JobStore Concurrency & Mutual Exclusion Stress Testing
# ==============================================================================


class TestSqliteJobStoreConcurrencyAndResilience:
    """Stress-test concurrent access, WAL locking, and mutual exclusion."""

    def test_concurrent_deliveries_thread_safety(self, tmp_path: Path) -> None:
        db_file = tmp_path / "concurrent_deliveries.db"
        store = SQLiteJobStore(db_file)
        num_deliveries = 50

        def record_worker(index: int):
            deliv = DeliveryRecord(
                delivery_id=f"thread-deliv-{index}",
                payload_digest=f"digest-{index}",
                received_at="2026-10-07T05:00:00Z",
                action="opened",
                installation_id=10,
                repository_id=20,
                pr_number=index,
                processed=False,
            )
            store.record_delivery(deliv)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_worker, i) for i in range(num_deliveries)]
            for f in futures:
                f.result()

        conn = store._get_connection()
        count = conn.execute("SELECT COUNT(*) FROM deliveries;").fetchone()[0]
        assert count == num_deliveries
        store.close()

    def test_concurrent_claims_strictly_mutual_exclusive(self, tmp_path: Path) -> None:
        db_file = tmp_path / "concurrent_claims.db"
        store = SQLiteJobStore(db_file)
        num_jobs = 10

        # Enqueue 10 distinct PR requests
        for i in range(num_jobs):
            req = PrRequestRecord(installation_id=1, repository_id=100 + i, pr_number=1, desired_generation=1)
            store.enqueue_request(req)

        claimed_run_ids = []
        lock = threading.Lock()

        def claim_worker(worker_idx: int):
            # Try to claim runs
            while True:
                run = store.claim_next_run(f"worker-{worker_idx}", lease_seconds=5.0)
                if run is None:
                    break
                with lock:
                    claimed_run_ids.append(run.run_id)
                assert store.update_run(
                    run.run_id, RunStatus.SUCCEEDED, worker_id=f"worker-{worker_idx}", generation=run.generation,
                )

        # 6 threads competing for 10 jobs
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(claim_worker, i) for i in range(6)]
            for f in futures:
                f.result()

        # All 10 jobs must be claimed exactly once (zero duplicate claims!)
        assert len(claimed_run_ids) == num_jobs
        assert len(set(claimed_run_ids)) == num_jobs
        store.close()

    def test_database_pragmas_configured_correctly(self, tmp_path: Path) -> None:
        db_file = tmp_path / "pragmas_check.db"
        store = SQLiteJobStore(db_file)
        conn = store._get_connection()

        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1

        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert busy_timeout == 5000
        store.close()
