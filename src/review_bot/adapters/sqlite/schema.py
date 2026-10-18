"""SQLite schema definitions, DDL, and index declarations for JobStore."""
from __future__ import annotations

PRAGMA_SETUP_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;
"""

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    payload_digest TEXT NOT NULL,
    received_at TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT '',
    installation_id INTEGER NOT NULL DEFAULT 0,
    repository_id INTEGER NOT NULL DEFAULT 0,
    pr_number INTEGER NOT NULL DEFAULT 0,
    processed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'received'
);

CREATE TABLE IF NOT EXISTS pr_requests (
    installation_id INTEGER NOT NULL DEFAULT 0,
    repository_id INTEGER NOT NULL,
    pr_number INTEGER NOT NULL,
    repo_full_name TEXT NOT NULL DEFAULT '',
    desired_generation INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (repository_id, pr_number)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    installation_id INTEGER NOT NULL DEFAULT 0,
    repository_id INTEGER NOT NULL,
    repo_full_name TEXT NOT NULL DEFAULT '',
    pr_number INTEGER NOT NULL,
    generation INTEGER NOT NULL DEFAULT 1,
    fingerprint TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL, -- pending, queued, running, claimed, succeeded, skipped, failed, superseded, cancelled
    stage TEXT NOT NULL DEFAULT 'fetch',
    worker_id TEXT,
    lease_expires_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    reason TEXT,
    kind TEXT NOT NULL DEFAULT 'github_pr'
);

CREATE TABLE IF NOT EXISTS fixed_intent_inputs (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    input_json TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('ko', 'en'))
);

CREATE TABLE IF NOT EXISTS publication_receipts (
    repository_id INTEGER NOT NULL,
    pr_number INTEGER NOT NULL,
    marker TEXT NOT NULL,
    comment_id INTEGER,
    published_fingerprint TEXT NOT NULL,
    published_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unknown',
    run_id TEXT,
    input_fingerprint TEXT,
    PRIMARY KEY (repository_id, pr_number, marker)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_lease_expires_at ON runs (lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_runs_repo_full_name ON runs (repo_full_name);
CREATE INDEX IF NOT EXISTS idx_runs_repo_pr ON runs (repository_id, pr_number);
CREATE INDEX IF NOT EXISTS idx_runs_generation ON runs (generation);
CREATE INDEX IF NOT EXISTS idx_runs_claimable ON runs (status, lease_expires_at, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_fencing ON runs (repository_id, pr_number, generation);
CREATE INDEX IF NOT EXISTS idx_pr_requests_repo_full_name ON pr_requests (repo_full_name);
CREATE INDEX IF NOT EXISTS idx_pr_requests_lookup ON pr_requests (repository_id, pr_number, desired_generation);
CREATE INDEX IF NOT EXISTS idx_deliveries_repo_pr ON deliveries (repository_id, pr_number);
CREATE INDEX IF NOT EXISTS idx_deliveries_received_at ON deliveries (received_at);
CREATE INDEX IF NOT EXISTS idx_publication_receipts_published_at ON publication_receipts (published_at);
"""
