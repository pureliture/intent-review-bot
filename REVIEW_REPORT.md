# github-review-bot: Multi-Perspective Architectural, Security & Production Readiness Audit

**Package**: `review_bot`  
**Target Codebase**: `github-review-bot/src/review_bot/`  
**Test Suite**: `github-review-bot/tests/` (967 passed across 41 test modules)  
**Authoritative Specifications**: `specs/20261002-github-review-bot/` (`requirements-a.md`, `design-a.md`, `design.md`, `contracts.md`, `runtime-a-draft.md`)  
**Audit Mode**: Zero Code Mutation / Forensic Read-Only Multi-Perspective Review  
**Date**: 2026-10-08  
**Audit Lead**: Teamwork Report Synthesis Engine  

---

## 1. Executive Summary

An exhaustive, evidence-backed architectural, security, concurrency, and quality assurance audit was conducted on the standalone `github-review-bot` service. The audit examined all functional components across seven specialized perspectives:
1. **R1. Security & Threat Modeling**
2. **R2. Concurrency, Durability & Fault Tolerance**
3. **R3. Analysis Correctness & Determinism**
4. **R4. LLM Pipeline & Hallucination Defense**
5. **R5. GitHub Integration & UX Reliability**
6. **R6. Software Architecture & Maintainability**
7. **Test Coverage Gap Analysis & Quality Assurance**

### 1.1 Production Readiness Score & Component Readiness Matrix

The service achieves an overall **Production Readiness Score of 71 / 100**. While individual core subsystems (domain purity, static analysis process isolation, XML XXE immunity, and SQLite WAL transaction discipline) display exceptional engineering maturity, critical operational disconnects between the worker execution runtime and downstream publication/comparison pipelines prevent immediate, unconstrained production deployment.

| Perspective | Weight | Score (/100) | Weighted | Status | Primary Constraint / Risk |
|---|:---:|:---:|:---:|:---:|---|
| **R1. Security & Threat Modeling** | 20% | **78** | 15.60 | High Priority Fixes | Fail-open HMAC on unconfigured secret; incomplete denylist in subprocess env sanitization; unredacted SQLite exceptions. |
| **R2. Concurrency & Durability** | 20% | **68** | 13.60 | **Blocked (Critical)** | Stale worker lease loss clobbers active successor; closed PRs waste full pipeline execution; delivery-enqueue atomicity gap. |
| **R3. Analysis Correctness** | 15% | **72** | 10.80 | **Blocked (Critical)** | Ambiguous/unavailable merge-base silently falls back to `target_base_sha`, fabricating false positive findings. |
| **R4. LLM Pipeline & Defense** | 15% | **74** | 11.10 | High Priority Fixes | Response gate omits schema validation on finding fields; quote-to-line desynchronization; model profile configuration disconnected. |
| **R5. GitHub Integration & UX** | 15% | **58** | 8.70 | **Blocked (Critical)** | Worker loop stubs publication receipt (`comment_id=None`) and never calls GitHub; unhandled `httpx.HTTPError` crashes; comment pagination absence. |
| **R6. Software Architecture** | 10% | **81** | 8.10 | Production Grade (Technical Debt) | 100% domain purity; application layer adheres to ports; OCP violations in hardcoded review mode registry and Java/PMD couplings. |
| **Test Coverage & Assurance** | 5% | **65** | 3.25 | Deficient in Seams | 967 passed tests are 91% unit-heavy; production `GitHubApiClient` has 0 tests; tautological assertions mask merge-base fallbacks. |
| **OVERALL COMPOSITE** | **100%** | **71.15** | **71 / 100** | **CONDITIONAL** | **4 Critical Blockers must be remediated prior to production traffic.** |

---

### 1.2 Definitive Deployment Readiness Verdict

```
+----------------------------------------------------------------------------------------------------+
|                                    DEPLOYMENT READINESS VERDICT                                    |
|                                                                                                    |
|                       STATUS: CONDITIONAL / NOT READY FOR LIVE PRODUCTION                          |
|                                                                                                    |
|  The offline CLI flow (review-bot review) and local static analysis engines are production-ready.  |
|  However, live multi-tenant webhook daemon execution is BLOCKED by four critical failure modes:    |
|   1. Worker publication bypass: Daemon workers never publish review comments to GitHub PRs.       |
|   2. Worker lease collision: Stale workers overwrite active successor worker runs in SQLite.       |
|   3. 2-dot diff fallback: Ambiguous merge bases attribute target branch code changes to PR authors.|
|   4. Unpaginated comments: PRs with >30 comments trigger duplicate comment spam & PR lockout.      |
+----------------------------------------------------------------------------------------------------+
```

---

### 1.3 Top Prioritized Action Items for Production Deployment

Ranked by operational impact and severity, the following eight remediations represent the critical path to deployment:

1. **[P0 - Critical] Connect Worker Loop to Publication Application Engine (`R5-1`, `R2-5`, `R4-3`, `R6-6`)**:  
   Update `WorkerLoop.process_run` in `src/review_bot/runtime/worker.py` to capture `ReviewResult` from `run_review_a_pipeline`, format markdown via `format_review_markdown`, invoke `publish_review_comment`, and persist the real `PublicationReceipt` with valid `comment_id`.
2. **[P0 - Critical] Eliminate Stale Worker Successor Clobbering (`R2-1`, `GAP-6`)**:  
   In `src/review_bot/runtime/worker.py:_check_fencing_and_lease`, ensure a worker that loses its lease aborts immediately without mutating database rows. In `src/review_bot/adapters/sqlite/store.py:update_run`, enforce CAS lease ownership via `AND (worker_id = ? OR ? IS NULL)`.
3. **[P0 - Critical] Eliminate `target_base_sha` Fallback on Ambiguous/Unavailable Merge Bases (`R3-1`, `GAP-1`)**:  
   In `src/review_bot/application/snapshot.py`, `src/review_bot/runtime/worker.py`, and `src/review_bot/runtime/cli.py`, remove fallbacks to `target_base_sha`. When `not merge_base_res.is_comparable`, halt diff execution, emit status `incomparable`, and post an informative status notice without computing fabricated diffs.
4. **[P0 - Critical] Implement Pagination & Error Resilience in GitHub Client (`R5-2`, `R5-3`, `GAP-12`)**:  
   In `src/review_bot/adapters/github/client.py:list_comments`, paginate through GitHub's REST API (`per_page=100`, RFC 5988 `Link` headers). Expand exception handling in `src/review_bot/application/publication.py` to catch `(httpx.HTTPError, Exception)`.
5. **[P1 - High] Enforce Fail-Closed Webhook HMAC Verification (`R1-1`)**:  
   In `src/review_bot/runtime/webhook.py`, reject all requests with HTTP 500/401 when `REVIEW_BOT_WEBHOOK_SECRET` is unset, eliminating the fail-open bypass.
6. **[P1 - High] Replace Subprocess Environment Denylist with Strict Allowlist (`R1-2`)**:  
   In `src/review_bot/adapters/pmd/runner.py` and `provisioning.py`, replace the 12-key denylist with an explicit allowlist (`PATH`, `JAVA_HOME`, `LC_ALL`, `LANG`, `TMPDIR`), preventing leakage of `REVIEW_BOT_*` tokens and host secrets.
7. **[P1 - High] Complete Response Gate Finding Schema and Grounding Validation (`R4-1`, `GAP-13`)**:  
   In `src/review_bot/application/review_a.py`, validate all finding objects against required fields (`title`, `reason`, `recommendation`), check that `signal_id` exists in `task_input.signals`, verify `evidence_ids` in `task_input.evidence`, and ensure `change` matches the static delta.
8. **[P1 - High] Redact Exceptions Before SQLite Persistence and Logging (`R1-4`)**:  
   Install a centralized regex-based secret scrubber to scrub authorization headers, bearer tokens, and API keys from `WorkerLoop` exception handlers and log output before persisting to `runs.error`.

---

## 2. Verified Architectural & Design Strengths

Across all seven audit perspectives, the target codebase demonstrated several standout engineering achievements:

1. **100% Domain Layer AST Purity**:  
   Static AST inspection confirmed that `src/review_bot/domain/` contains zero IO or operating system imports (`os`, `sys`, `pathlib`, `subprocess`, `requests`, `httpx`, `sqlite3`, etc.). All 15+ domain entities are strictly defined using `@dataclass(frozen=True)` and `PurePosixPath`, ensuring immutability, mathematical determinism, and zero side effects.
2. **XML Parsing XXE Immunity (`_StrictTreeBuilder`)**:  
   PMD and CPD XML report parsers in `src/review_bot/adapters/pmd/parsers.py` implement a custom `_StrictTreeBuilder` subclass of `xml.etree.ElementTree.TreeBuilder` that overrides `doctype()` to immediately raise `ValueError`. This provides absolute defense against SYSTEM/PUBLIC external entity resolution, parameter entity leaks, and recursive Billion Laughs exponential expansion bombs. Payloads are additionally capped at 64 MiB and checked against symlink traversals.
3. **Subprocess Isolation & Process Group Termination (`os.killpg`)**:  
   Java static analysis runner (`src/review_bot/adapters/pmd/runner.py`) invokes subprocesses with `start_new_session=True` (`setsid()`), creating an independent process group. On timeout, it performs a two-stage escalation: `SIGTERM` followed by a 2-second grace period and `SIGKILL` via `os.killpg(process.pid, ...)`, completely preventing orphaned background processes.
4. **TCC (Tight Class Cohesion) Polarity Inversion Correctness**:  
   In `src/review_bot/application/delta.py:441`, the delta engine implements mathematically correct polarity inversion for cohesion metrics:
   $$\text{direction} = ((\text{head} > \text{base}) - (\text{head} < \text{base})) \times (-1 \text{ if metric} == \text{"tcc"} \text{ else } 1)$$
   Because higher cohesion signifies superior architectural decoupling, an increase in TCC correctly yields direction `-1` (`improved`), while a decrease yields `+1` (`worsened`).
5. **Durable SQLite WAL Transaction Discipline**:  
   `SQLiteJobStore` strictly enforces `PRAGMA journal_mode = WAL`, `PRAGMA synchronous = NORMAL`, and `PRAGMA busy_timeout = 5000`. All write queries execute inside `with self.transaction(immediate=True)`, acquiring a SQLite `RESERVED` lock at transaction inception and preventing deadlock during concurrent write upgrades.
6. **Strict "Full Source Context or Fail-Closed" Context Assembly**:  
   In `src/review_bot/application/context.py` and `review_a.py`, snapshot code inputs are assembled in full without arbitrary file splitting, sliding windows, or silent heuristic omissions. If inputs exceed `max_input_bytes` or `max_input_tokens`, the pipeline halts with `status="not_run", reason="input_limit_exceeded"`, refusing to produce degraded hallucinations from partial context.
7. **Adversarial Input Sanitization in Git & Markdown Adapters**:  
   The Git adapter enforces hex regexes (`^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z`) on all commit hashes, executes Git with `-c protocol.allow=never` and `--no-replace-objects`, and filters symlinks (`mode 120000`). The Markdown formatter defuses user mentions by injecting zero-width spaces (`@\u200b`) and parses artifact URLs to forbid embedded credentials (`user:password@host`).
8. **Offline CLI Fidelity**:  
   The standalone CLI (`review-bot review -o <dir>`) runs fully offline against synthetic git repositories without requiring network access, GitHub tokens, or live LLM credentials, generating complete review comment artifacts.

---

## 3. Section 1: R1. Security & Threat Modeling Audit

### Finding [R1-1]: Webhook HMAC Verification Fails Open When Secret Is Unconfigured
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: High
- **Location**: `src/review_bot/runtime/webhook.py:90-104`
- **Evidence / Rationale**:
  In `create_webhook_app`, `_get_secret()` retrieves the secret from configuration or environment variable `REVIEW_BOT_WEBHOOK_SECRET`.
  In `handle_webhook`:
  ```python
  configured_secret = _get_secret()
  if configured_secret:
      if not signature_header or not verify_github_signature(
          raw_body, signature_header, configured_secret
      ):
          return Response(content="Invalid HMAC signature", status_code=status.HTTP_401_UNAUTHORIZED)
  elif signature_header:
      return Response(content="Invalid HMAC signature", status_code=status.HTTP_401_UNAUTHORIZED)
  ```
  If `configured_secret` is `""` (e.g. operator forgot to configure `REVIEW_BOT_WEBHOOK_SECRET`):
  `if configured_secret:` evaluates to `False`.
  If an unauthenticated attacker sends an HTTP POST without `X-Hub-Signature-256`, `elif signature_header:` also evaluates to `False`.
  Both authentication checks are bypassed! The request is parsed, recorded in SQLite, and enqueued for execution. This is a critical fail-open vulnerability.
- **Actionable Remediation**:
  Enforce fail-closed verification. When the secret is empty, reject all webhook requests:
  ```python
  configured_secret = _get_secret()
  if not configured_secret:
      logger.error("Rejecting webhook request: webhook secret is unconfigured")
      return Response(
          content="Webhook secret is not configured",
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      )

  if not signature_header or not verify_github_signature(
      raw_body, signature_header, configured_secret
  ):
      return Response(
          content="Invalid HMAC signature",
          status_code=status.HTTP_401_UNAUTHORIZED,
      )
  ```

---

### Finding [R1-2]: Incomplete Denylist in Subprocess Environment Sanitization Leaks Application Secrets
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: High
- **Location**: `src/review_bot/adapters/pmd/runner.py:89-110`, `src/review_bot/adapters/pmd/provisioning.py:149-170`
- **Evidence / Rationale**:
  `sanitize_environment()` and `get_clean_env()` use a hardcoded denylist of 12 keys (`JAVA_TOOL_OPTIONS`, `GITHUB_TOKEN`, etc.).
  However, the application defines and consumes configuration environment variables (`config.py:149-181`):
  - `REVIEW_BOT_APP_PRIVATE_KEY_FILE`
  - `REVIEW_BOT_WEBHOOK_SECRET`
  - `REVIEW_BOT_APP_ID`
  - `REVIEW_BOT_STATE_DIR`
  - `REVIEW_MODEL_PROFILE`
  None of these variables are included in the denylist! Moreover, host cloud credentials (`AWS_SECRET_ACCESS_KEY`, `VAULT_TOKEN`) pass through unscrubbed. PMD executes against untrusted Java code submitted in pull requests; hostile Java source or custom PMD XPath rules can read `System.getenv("REVIEW_BOT_WEBHOOK_SECRET")` and exfiltrate it via analysis reports (CWE-184).
- **Actionable Remediation**:
  Replace the denylist with a strict allowlist containing only essential system variables:
  ```python
  SAFE_ENV_KEYS = frozenset({
      "PATH",
      "JAVA_HOME",
      "LC_ALL",
      "LANG",
      "TMPDIR",
      "TEMP",
      "TMP",
      "USER",
      "HOME",
  })

  def sanitize_environment(java_home: Path | None = None) -> dict[str, str]:
      env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_KEYS}
      if java_home is not None:
          env["JAVA_HOME"] = str(java_home)
      env["LC_ALL"] = "C.UTF-8"
      env["LANG"] = "C.UTF-8"
      return env
  ```

---

### Finding [R1-3]: Worker Loop Passes `source_root=None` to Analyzer, Bypassing Isolated Source Materialization
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: High
- **Location**: `src/review_bot/runtime/worker.py:226-231`
- **Evidence / Rationale**:
  In `WorkerLoop.claim_and_execute_once`:
  ```python
  base_run = self.analyzer.run_analysis(effective_base_sha, None, ruleset, profile)
  head_run = self.analyzer.run_analysis(head_sha, None, ruleset, profile)
  ```
  The second argument passed to `run_analysis` is `None`. In `PmdRunner.run_analysis` (`runner.py:422`), `source_root = source_root.resolve()` is executed immediately, throwing `AttributeError: 'NoneType' object has no attribute 'resolve'`.
  In `cli.py`, source blobs are extracted to isolated `tempfile.TemporaryDirectory()` paths without symlinks before running analysis. In `WorkerLoop`, source extraction was completely omitted, causing real static analysis runs to crash and bypassing symlink-safe isolation.
- **Actionable Remediation**:
  In `src/review_bot/runtime/worker.py`, materialize git blobs into temporary directories before invoking static analysis:
  ```python
  import tempfile

  with tempfile.TemporaryDirectory() as base_tmp, tempfile.TemporaryDirectory() as head_tmp:
      if self.repo_reader:
          for p, content in self.repo_reader.read_tree_blobs(effective_base_sha).items():
              dest = Path(base_tmp) / p
              dest.parent.mkdir(parents=True, exist_ok=True)
              dest.write_bytes(content)
          for p, content in self.repo_reader.read_tree_blobs(head_sha).items():
              dest = Path(head_tmp) / p
              dest.parent.mkdir(parents=True, exist_ok=True)
              dest.write_bytes(content)

      base_run = self.analyzer.run_analysis(effective_base_sha, Path(base_tmp), ruleset, profile)
      head_run = self.analyzer.run_analysis(head_sha, Path(head_tmp), ruleset, profile)
  ```

---

### Finding [R1-4]: Unredacted Exception Strings Persisted in Persistent SQLite Storage & Logger Tracebacks
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: High
- **Location**: `src/review_bot/runtime/worker.py:328-335`
- **Evidence / Rationale**:
  In `WorkerLoop.claim_and_execute_once`:
  ```python
  except Exception as exc:
      logger.exception(f"Error processing run {run_id}")
      self.job_store.update_run(
          run_id=run_id,
          status=RunStatus.FAILED,
          error=str(exc),
      )
  ```
  `str(exc)` is written unredacted directly to the persistent SQLite `runs.error` column.
  When exceptions occur during HTTP requests to GitHub or model APIs (`httpx.HTTPStatusError`, connection errors), the exception message routinely includes sensitive URL query tokens (`token=ghs_...`), bearer headers (`Bearer sk-...`), or private key paths. Furthermore, `logger.exception()` writes full raw tracebacks to logs, leaking tokens.
- **Actionable Remediation**:
  Implement a centralized secret scrubber utility and sanitize exception strings before database persistence and logging:
  ```python
  import re

  SECRET_PATTERNS = [
      re.compile(r"(gh[pousr]_[A-Za-z0-9_]{36,255})"),
      re.compile(r"(whsec_[A-Za-z0-9_]+)"),
      re.compile(r"(sk-[A-Za-z0-9_-]{20,})"),
      re.compile(r"(Bearer\s+)[A-Za-z0-9_.\-]+", re.IGNORECASE),
      re.compile(r"(token=)[^&\s]+", re.IGNORECASE),
  ]

  def redact_secrets(text: str) -> str:
      for pattern in SECRET_PATTERNS:
          text = pattern.sub(r"[REDACTED]", text)
      return text
  ```
  Apply `redact_secrets(str(exc))` in `worker.py:333`.

---

### Finding [R1-5]: Potential Token Leakage in `httpx` Debug Logs Due to Missing Secret Masker
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/github/client.py:48-51, 56-59, 102-105, 125-128`
- **Evidence / Rationale**:
  `GitHubApiClient` executes requests using `httpx.Client(timeout=self.timeout)` and attaches `headers["Authorization"] = f"Bearer {self.token}"`.
  Neither `httpx.Client` nor the client adapter registers an event hook or custom logging filter. When operators enable `DEBUG` logging on `httpx` or `httpcore` during network troubleshooting, `httpx` dumps full request headers in plaintext, exposing installation tokens to stdout/stderr.
- **Actionable Remediation**:
  Configure an `httpx` event hook or register a logging filter on `httpx`/`httpcore` loggers that strips `Authorization` headers.

---

### Finding [R1-6]: Absence of Streaming Timeout / Slowloris Defenses in Webhook Receiver
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: Medium
- **Location**: `src/review_bot/runtime/webhook.py:68-76`
- **Evidence / Rationale**:
  In `handle_webhook`, `async for chunk in request.stream():` consumes payload chunks up to `max_bytes`.
  However, there is no read timeout wrapped around stream chunk arrivals. An attacker can stream bytes at 1 byte every 30 seconds (Slowloris attack), holding ASGI coroutines open indefinitely and exhausting connection slots.
- **Actionable Remediation**:
  Wrap stream consumption in `asyncio.timeout(30.0)`:
  ```python
  try:
      async with asyncio.timeout(30.0):
          async for chunk in request.stream():
              total_received += len(chunk)
              if total_received > max_bytes:
                  return Response(content="Payload Too Large", status_code=status.HTTP_413_CONTENT_TOO_LARGE)
              chunks.append(chunk)
  except TimeoutError:
      return Response(content="Request Timeout", status_code=status.HTTP_408_REQUEST_TIMEOUT)
  ```

---

### Finding [R1-7]: Empty Allowlists Inadvertently Permit Untrusted Repositories in Non-GitHub-App Modes
- **Perspective**: R1. Security & Threat Modeling
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/github/webhook.py:104-124`, `src/review_bot/runtime/webhook.py:161-176`
- **Evidence / Rationale**:
  In `adapters/github/webhook.py`, `is_repository_allowed` and `is_installation_allowed` state:
  `if not allowlist: return True`.
  While `ReviewBotConfig.validate_github_app_mode` requires non-empty allowlists when `execution.mode == "github_app"`, in any other mode or dynamic configuration an empty allowlist permits ALL repositories and installations on GitHub to enqueue jobs. Best practice mandates failing closed when allowlists are empty.
- **Actionable Remediation**:
  Update `is_repository_allowed` and `is_installation_allowed` to return `False` when allowlists are empty, requiring an explicit wildcard `["*"]` to allow all.

---

## 4. Section 2: R2. Concurrency, Durability & Fault Tolerance Audit

### Finding [R2-1]: Stale Worker Lease Loss Clobbers Active Successor Run via Unfenced `update_run`
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Critical
- **Location**: `src/review_bot/runtime/worker.py:108-116`, `src/review_bot/adapters/sqlite/store.py:273-286`
- **Evidence / Rationale**:
  When Worker A's lease expires due to CPU starvation or long static analysis, Worker B reclaims the job via `claim_next_run(worker_id="worker-B")`.
  When Worker A resumes, heartbeat renewal fails (`renew_lease` returns `False`), setting `self._lease_lost = True`.
  In `WorkerLoop._check_fencing_and_lease`:
  ```python
  if self._lease_lost:
      self.job_store.update_run(
          run_id=run_id,
          status=RunStatus.SUPERSEDED,
          error="Lease lost during execution",
      )
      return False
  ```
  In `SQLiteJobStore.update_run`:
  ```sql
  UPDATE runs
  SET status = ?, stage = COALESCE(?, stage), error = COALESCE(?, error),
      completed_at = COALESCE(completed_at, datetime('now')), lease_expires_at = NULL
  WHERE run_id = ?;
  ```
  `update_run` has NO check verifying `worker_id`! Worker A blindly overwrites the row owned by Worker B, marking it `superseded` and setting `lease_expires_at = NULL`. Worker B's heartbeat then fails, causing Worker B to also abort. Both workers terminate, killing the active run.
- **Actionable Remediation**:
  1. In `worker.py:108-116`, when `self._lease_lost` is True, Worker A must exit immediately without modifying database rows.
  2. In `store.py:261`, add `worker_id: str | None = None` to `update_run` and enforce CAS lease ownership:
     ```sql
     UPDATE runs
     SET status = ?, stage = COALESCE(?, stage), error = COALESCE(?, error),
         completed_at = COALESCE(completed_at, datetime('now')), lease_expires_at = NULL
     WHERE run_id = ? AND (worker_id = ? OR ? IS NULL);
     ```

---

### Finding [R2-2]: Closed PR Webhook Event Fails to Cancel Active or Queued Runs, Causing Wasted Pipeline Execution
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: High
- **Location**: `src/review_bot/runtime/webhook.py:191-196`, `src/review_bot/runtime/worker.py:154-167`
- **Evidence / Rationale**:
  `runtime-a-draft.md §58` mandates: *"closed에서는 대기·실행 작업을 취소하고 게시하지 않는다."*
  However, in `webhook.py`:
  ```python
  if pr_event.action == "closed":
      return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "cancelled", "reason": "closed"})
  ```
  The webhook receiver returns HTTP 202 without calling `JobStore`. Any queued or running jobs for the closed PR remain active in `runs`. In `worker.py:154-167`, the worker checks `pr_data`, but never checks `pr_data.get("state") == "closed"`. A worker claims the closed PR job and executes the full Git checkout, PMD/CPD analysis, AST delta computation, context assembly, and LLM inference, only aborting at publication pre-check. This wastes massive CPU and LLM token resources.
- **Actionable Remediation**:
  1. Add `cancel_pr_runs(repository_id, pr_number)` to `SQLiteJobStore` to transition pending/queued/running runs to `cancelled`.
  2. Call `job_store.cancel_pr_runs` in `webhook.py:191` on `action == "closed"`.
  3. In `worker.py:168`, check `if pr_data.get("state") == "closed": return False` during the PREPARE stage.

---

### Finding [R2-3]: Atomicity Gap Between Webhook Delivery Recording and Job Enqueueing Causes Permanent Event Loss
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: High
- **Location**: `src/review_bot/runtime/webhook.py:137-186`, `src/review_bot/adapters/sqlite/schema.py:12-22`
- **Evidence / Rationale**:
  `runtime-a-draft.md §49` requires delivery recording and job enqueueing to commit in a single transaction.
  In `webhook.py`, `job_store.record_delivery(delivery)` commits at line 138. `job_store.enqueue_request(pr_request)` commits at line 185 in a separate transaction.
  If the process crashes between line 138 and 185:
  1. The delivery is committed in `deliveries`.
  2. No request or run is created in `pr_requests` or `runs`.
  3. When GitHub retries the delivery, `record_delivery` catches `sqlite3.IntegrityError` and immediately returns `status="deduplicated"` without enqueuing.
  4. The event is permanently dropped with no automated recovery.
- **Actionable Remediation**:
  Add `record_delivery_and_enqueue(delivery, request)` to `SQLiteJobStore` executing both inserts within a single `BEGIN IMMEDIATE` transaction.

---

### Finding [R2-4]: Unbounded Worker Reclaims and Missing Poison-Pill / DLQ Threshold Cause Infinite Crash Loops
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: High
- **Location**: `src/review_bot/adapters/sqlite/store.py:170-226`, `src/review_bot/runtime/worker.py:342-348`
- **Evidence / Rationale**:
  `claim_next_run` reclaims expired running jobs and increments `attempts`. However, neither `claim_next_run` nor `WorkerLoop` checks an upper limit on `attempts`.
  If a malformed PR causes an unhandled process crash (e.g. JVM segfault or OOM kill in PMD):
  1. The process dies without setting `status='failed'`.
  2. Lease expires after 30s.
  3. Another worker reclaims it (`attempts=2`) and crashes again.
  4. This repeats infinitely (`attempts=3, 4, ...`), starving legitimate jobs and causing an infinite crash loop.
- **Actionable Remediation**:
  Add `max_attempts: int = 3` configuration parameter. In `claim_next_run`, if an expired lease job has `attempts >= max_attempts`, update it to `status = 'failed'` with `error = "Exceeded maximum retry attempts due to crash"`, and skip claiming it.

---

### Finding [R2-5]: Disconnection of Guarded Publication Logic in Worker Loop Prevents Comment Publishing and Freshness Guards
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: High
- **Location**: `src/review_bot/runtime/worker.py:283-326`, `src/review_bot/application/publication.py:26-254`
- **Evidence / Rationale**:
  `src/review_bot/application/publication.py` implements guarded comment publication with pre-check freshness, marker discovery, and mid-write race detection.
  However, in `src/review_bot/runtime/worker.py:283-326`, `publish_review_comment` is NEVER invoked!
  Instead, `worker.py` fabricates a dummy `PublicationReceipt(comment_id=None, status="succeeded", ...)` and updates the run to SUCCEEDED. No comment is ever posted to GitHub, and publication race guards are bypassed.
- **Actionable Remediation**:
  Import `publish_review_comment` and `format_review_markdown` in `worker.py`. Render markdown from `review_result` and execute `publish_review_comment`.

---

### Finding [R2-6]: Missing PR-Level Publication Mutual Exclusion Allows Ambiguous Duplicate Comment Spawning
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Medium
- **Location**: `src/review_bot/application/publication.py:110-184`, `src/review_bot/adapters/sqlite/schema.py:53-61`
- **Evidence / Rationale**:
  GitHub's REST API lacks conditional creation (`If-None-Match`). If two workers execute concurrently for the same PR (e.g. Worker A whose lease expired and Worker B who reclaimed the job), both call `find_marker_comment`, receive `None`, and call `create_comment`.
  Both comments succeed on GitHub, creating two comments with marker `<!-- review-bot:architecture:v1 -->`.
  All subsequent runs call `find_marker_comment`, discover multiple matching comments, and raise `ValueError: ambiguous_marker_comment`, permanently breaking the PR.
- **Actionable Remediation**:
  1. Introduce a PR-level publication lock in `SQLiteJobStore` (`acquire_pr_publication_lock`).
  2. In `find_marker_comment`, fall back to selecting the newest matching comment (`matched[-1]`) rather than raising `ValueError`.

---

### Finding [R2-7]: `check_generation_fencing` Overwrites Completed Runs and Lacks Terminal State Immunity
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/sqlite/store.py:356-389`
- **Evidence / Rationale**:
  In `check_generation_fencing`, if `run_gen < desired_gen`, the query executes:
  `UPDATE runs SET status = 'superseded' WHERE run_id = ?;`
  without checking whether `status` is already terminal (`succeeded` or `failed`). If `check_generation_fencing` is called on a completed run after subsequent pushes arrive, historical succeeded records are overwritten to `superseded`, corrupting audit history.
- **Actionable Remediation**:
  Add `AND LOWER(status) IN ('running', 'claimed', 'queued', 'pending')` to the SQL update query in `store.py:384`.

---

### Finding [R2-8]: Connection Leak and In-Memory Database Isolation in Thread-Local Connection Model
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Low
- **Location**: `src/review_bot/adapters/sqlite/store.py:38-53, 390-398`
- **Evidence / Rationale**:
  `SQLiteJobStore` manages connections via `self._local = threading.local()`. `SQLiteJobStore.close()` only closes the connection on the calling thread. In thread-pooled daemon servers, connections opened on background threads remain unclosed on shutdown. Additionally, `:memory:` databases create private, isolated DBs per thread rather than a shared database.
- **Actionable Remediation**:
  Maintain a thread-safe registry of all active connections for shutdown cleanup, and use `file::memory:?cache=shared` for multi-threaded in-memory databases.

---

### Finding [R2-9]: Foreign Key Constraints Declared in PRAGMA but Missing from Schema DDL
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Low
- **Location**: `src/review_bot/adapters/sqlite/schema.py:8, 11-62`
- **Evidence / Rationale**:
  `schema.py:8` executes `PRAGMA foreign_keys = ON;`, but `CREATE_TABLES_SQL` contains zero `FOREIGN KEY (...) REFERENCES ...` clauses across `deliveries`, `runs`, and `publication_receipts`.
- **Actionable Remediation**:
  Add explicit foreign key constraints or document intentional omission for table lock minimization.

---

### Finding [R2-10]: Non-PR and Unhandled PR Webhook Actions Ignored Without Mode-Selection or State Re-evaluation
- **Perspective**: R2. Concurrency, Durability & Fault Tolerance
- **Severity**: Low
- **Location**: `src/review_bot/runtime/webhook.py:177-200`
- **Evidence / Rationale**:
  `runtime-a-draft.md §58` lists `edited`, `labeled`, `unlabeled`, and `ready_for_review` as actions requiring re-evaluation. `webhook.py` currently drops these actions with `unhandled_action`. Adding a `리팩토링` label or editing the PR title does not trigger perspective re-classification.
- **Actionable Remediation**:
  Expand accepted actions in `webhook.py:178` to include `edited`, `labeled`, `unlabeled`, and `ready_for_review`.

---

## 5. Section 3: R3. Analysis Correctness & Determinism Audit

### Finding [R3-1]: Fallback to `target_base_sha` on Ambiguous or Unavailable Merge Base Fabricates False Findings in Worker and CLI
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: Critical
- **Location**: `src/review_bot/application/snapshot.py:124-138`, `src/review_bot/runtime/worker.py:219-245`, `src/review_bot/runtime/cli.py:491-498`
- **Evidence / Rationale**:
  Specifications (`design-a.md:82`, `contracts.md:11-23`) mandate:
  *"Strictly distinguish target_base_sha from unique merge-base sha; missing/multiple merge-bases must not fabricate false findings... 공통 조상이 없거나 여러 후보면 비교 불가로 표시하고 임의 후보를 고르지 않는다."*
  In `git.py`, `resolve_merge_base` correctly sets `comparison_status="ambiguous"` or `"unavailable"` with `comparison_base_sha=None`.
  However, in `snapshot.py:132`:
  ```python
  if merge_base_res.is_comparable and merge_base_res.comparison_base_sha:
      ...
  else:
      effective_base_sha = target_base_sha  # <-- CRITICAL FALLBACK
  pr_snapshot = assemble_pr_snapshot(..., base_sha=effective_base_sha)
  ```
  In `worker.py:219-245`, static analysis is executed on `effective_base_sha` (the head of `main`) and `head_sha`. `build_delta` computes `AnalysisDelta(comparable=True)`. Any commits merged into `main` by other authors are classified as `new` or `worsened` findings against the PR author, which are then published to GitHub!
  Similarly, `cli.py:492` executes `comparison_base_sha = merge_res.comparison_base_sha or target_base_sha`.
- **Actionable Remediation**:
  1. In `snapshot.py:132`, set `pr_snapshot.base_sha = None` when `not merge_base_res.is_comparable`.
  2. In `worker.py` and `cli.py`, check `if not merge_base_res.is_comparable:`, halt comparison, record run status `SUCCEEDED` (with comparison `incomparable`), and post an informative notice stating comparison is unavailable due to ambiguous/missing merge base. Do not run delta comparison or LLM review.

---

### Finding [R3-2]: Tautological Test Assertions Masking Merge-Base Fallback in Test Suite
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: High
- **Location**: `tests/contract/test_pairwise_interactions.py:152-175`, `tests/e2e/test_pr_lifecycle_scenarios.py:257-267`
- **Evidence / Rationale**:
  In `test_pairwise_interactions.py:162`:
  ```python
  delta_change = "unknown" if merge_res["status"] == "ambiguous" else "existing"
  assert delta_change == "unknown"
  ```
  The test merely asserts an inline Python ternary expression, completely bypassing `assemble_snapshot_bundle`, `worker.py`, and `build_delta`.
  In `test_pr_lifecycle_scenarios.py:257-267`, the test only checks dictionary output from raw Git CLI resolution without invoking the review runner. These tautological tests passed 100% while masking Finding [R3-1].
- **Actionable Remediation**:
  Rewrite tests to execute the actual `assemble_snapshot_bundle` and worker pipeline on criss-cross merge fixtures, asserting that no review action items are generated.

---

### Finding [R3-3]: Omission of `CouplingBetweenObjects` Metric Extraction Leading to Premature "Unknown" Degradation
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: Medium
- **Location**: `src/review_bot/application/delta.py:54-60`, `src/review_bot/resources/pmd/ruleset.xml:54-58`
- **Evidence / Rationale**:
  `CouplingBetweenObjects` is enabled in `src/review_bot/resources/pmd/ruleset.xml` with threshold 20.
  However, in `src/review_bot/application/delta.py:54-60`, `_METRIC_PATTERNS` only includes `CognitiveComplexity`, `NPathComplexity`, `CyclomaticComplexity`, `NcssCount`, and `GodClass`.
  When PMD detects `CouplingBetweenObjects`, `_metrics(finding)` returns `{}`. Measurements are empty, forcing `_metric_change` to return `"unknown"`. Coupling regressions (e.g. coupling increasing from 21 to 35) are never recognized as `"worsened"`.
- **Actionable Remediation**:
  Add regex pattern for `CouplingBetweenObjects` to `_METRIC_PATTERNS` in `delta.py:54-60`:
  ```python
  "CouplingBetweenObjects": re.compile(
      r".*?(?:total|count|coupling|members denotes a high coupling)\s*(?:is|=|of)?\s*(?P<coupling>\d+).*?",
      re.IGNORECASE,
  ),
  ```

---

### Finding [R3-4]: Unused `ComparisonConfig` Creates Dead Configuration Contract
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: Medium
- **Location**: `src/review_bot/config.py:246-250, 299`, `src/review_bot/runtime/worker.py:177-198`
- **Evidence / Rationale**:
  `ComparisonConfig` defines `ambiguous_base_action: Literal["incomparable"] = "incomparable"`.
  However, `config.comparison` is never referenced anywhere in `snapshot.py`, `worker.py`, or `cli.py`. The configuration promises operators that ambiguous merge bases will follow `ambiguous_base_action`, but the runtime never enforces it.
- **Actionable Remediation**:
  Pass `config.comparison` to `assemble_snapshot_bundle` and enforce `ambiguous_base_action` in runtime workers.

---

### Finding [R3-5]: TCC (Tight Class Cohesion) Integer Percentage Parsing Risk in GodClass Rule
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: Low
- **Location**: `src/review_bot/application/delta.py:425-429`
- **Evidence / Rationale**:
  In `delta.py:425`, `value = float(text.rstrip("%")) / (100 if text.endswith("%") else 1)`.
  If a PMD violation or custom formatter reports TCC as an integer percentage without a literal `%` character (e.g. `TCC=30`), `text.endswith("%")` is False, dividing by 1 (`value = 30.0`). Because `not 0 <= 30.0 <= 1`, validation fails and returns `{}`. All metrics for the GodClass finding are discarded, degrading the comparison to `"unknown"`.
- **Actionable Remediation**:
  Normalize `value = raw_val / 100.0 if (text.endswith("%") or raw_val > 1.0) else raw_val`.

---

### Finding [R3-6]: `git diff --find-renames` Lacks Explicit Similarity Index Parameter
- **Perspective**: R3. Analysis Correctness & Determinism
- **Severity**: Low
- **Location**: `src/review_bot/adapters/git.py:235`
- **Evidence / Rationale**:
  `get_changed_files` invokes git with `["diff", ..., "--find-renames", ...]`. It does not specify `--find-renames=50%`. Relying on Git's implicit default can introduce non-deterministic rename detection across heterogeneous worker environments.
- **Actionable Remediation**:
  Explicitly pass `"--find-renames=50%"`.

---

## 6. Section 4: R4. LLM Pipeline & Hallucination Defense Audit

### Finding [R4-1]: Incomplete Response Gate Validation: Missing Schema Enforcement and Hallucination Checks for Finding Fields
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: High
- **Location**: `src/review_bot/application/review_a.py:446-524, 537-573`
- **Evidence / Rationale**:
  In `build_response_schema` (lines 140–195), a rigorous JSON schema is defined specifying required properties (`signal_id`, `judgement`, `change`, `title`, `reason`, `recommendation`, `evidence_ids`) and enums.
  However, `validate_model_response` (lines 446–524) does NOT use a schema validator and only executes hand-rolled checks. It omits:
  1. Required fields: Missing `title`, `reason`, or `recommendation` are never flagged.
  2. Grounding of `signal_id`: Does not check if `signal_id` exists in `task_input.signals`. The model can hallucinate arbitrary signal IDs.
  3. Grounding of `evidence_ids`: Does not check if IDs in `finding["evidence_ids"]` exist in `task_input.evidence`.
  4. Judgement & Basis enum conformity: Values like `"judgement": "critical_flaw"` are accepted.
  5. State consistency: Does not check whether `finding["change"]` matches `signal.change`.
  In `_parse_validated_entities` (lines 558–572), a finding object with missing fields silently falls back to defaults (`signal_id=""`, `judgement="concern"`). A model outputting `{"junk": 1}` is accepted as a valid concern finding.
- **Actionable Remediation**:
  In `validate_model_response`, iterate over `findings` and enforce:
  - `signal_id in {s.signal.signal_id for s in task_input.signals}` with `role == "review"`.
  - All IDs in `evidence_ids` exist in `known_evidence`.
  - `judgement in JUDGEMENTS`, `basis in BASES`, and `change in CHANGE_STATES`.
  - `change == signal.change`.
  - Non-empty strings for `title`, `reason`, `recommendation`.
  - Reject unexpected keys.

---

### Finding [R4-2]: Quote-to-Line Desynchronization: Citations Can Quote Remote File Snippets with Arbitrary Line Numbers
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: High
- **Location**: `src/review_bot/application/review_a.py:321-324, 360-388`
- **Evidence / Rationale**:
  In `_validate_citation`, line bounds and quote verification are disconnected:
  1. Lines 321–324 check if `quote in ev_content_map[ev_id]`. For full-file code evidence, `ev_content_map[ev_id]` is the entire file.
  2. Lines 360–388 check if `1 <= begin_line <= end_line <= total_lines_in_file`.
  The validator never verifies that `quote` appears at lines `[begin_line, end_line]`.
  A model can cite `begin_line: 1, end_line: 5` and provide a quote from line 450. Both checks pass, and the citation is accepted, falsely attributing code to lines 1–5.
- **Actionable Remediation**:
  In `_validate_citation`, extract the source slice `expected_slice = "".join(file_lines[begin_line - 1 : end_line])` and verify that `quote` matches or is contained within `expected_slice`.

---

### Finding [R4-3]: Discarded Model Review Pipeline Result in Worker Process
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: High
- **Location**: `src/review_bot/runtime/worker.py:273-275, 311-326`
- **Evidence / Rationale**:
  In `WorkerLoop.process_run` (Stage 6: REVIEW):
  `run_review_a_pipeline(task_input, self.model_client)` is executed, but its return value is discarded without assignment.
  Stages 7 and 8 proceed without rendering markdown, create a dummy receipt with `comment_id=None`, and mark the run `SUCCEEDED`. Even if `run_review_a_pipeline` failed with `invalid_model_response` or aborted with `input_limit_exceeded`, the worker marks the run as SUCCEEDED without publishing any results to GitHub.
- **Actionable Remediation**:
  Capture `review_res = run_review_a_pipeline(...)`, handle failure/aborted states, render markdown via `format_review_markdown(review_res, task_input)`, and publish via `publish_review_comment`.

---

### Finding [R4-4]: Lack of PR-Diff Relevance Gate for Citation File Paths
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: Medium
- **Location**: `src/review_bot/application/review_a.py:328-334`
- **Evidence / Rationale**:
  `_validate_citation` verifies that `cit["path"]` matches `ev.path`. However, `task_input.evidence` contains full-file code evidence for all files across the repository. The validator does not check whether `cit["path"]` belongs to `snapshot.pr.changed_files`. The model can cite an unmodified, unrelated file (e.g. `gradle-wrapper.properties`), and the Response Gate accepts it.
- **Actionable Remediation**:
  Verify that citations reference paths present in `snapshot.pr.changed_files` or corresponding to the static analysis findings linked to `signal_id`.

---

### Finding [R4-5]: Hidden Protocol Leak: `count_input_tokens` Undeclared on `ModelClient` Protocol
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: Medium
- **Location**: `src/review_bot/ports.py:121-127`, `src/review_bot/application/review_a.py:699-709`
- **Evidence / Rationale**:
  `ports.ModelClient` defines only `def complete(self, request: ModelRequest) -> ModelResponse: ...`.
  However, in `review_a.py:700`, if `max_input_tokens` is configured, it checks `if hasattr(client, "count_input_tokens"):` and otherwise returns `"not_run", "token_count_unavailable"`. Any client strictly implementing `ports.ModelClient` fails with `"token_count_unavailable"`.
- **Actionable Remediation**:
  Add `def count_input_tokens(self, request: ModelRequest) -> int: ...` to `ModelClient` in `ports.py`.

---

### Finding [R4-6]: Disconnection between `ReviewBotConfig.model` and `ModelProfile`
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: Medium
- **Location**: `src/review_bot/runtime/cli.py:542`, `src/review_bot/runtime/worker.py:273-275`
- **Evidence / Rationale**:
  `ReviewBotConfig.model` (`config.py:178-201`) defines model execution parameters. However, neither `cli.py` nor `worker.py` transforms `config.model` into a `ModelProfile`. `cli.py:542` calls `run_review_a_pipeline(task_input)` without profile, causing `run_review_a_pipeline` to return `status="not_run", reason="model_profile_unset"`.
- **Actionable Remediation**:
  Implement `model_profile_from_config(config.model)` and pass the profile to `run_review_a_pipeline`.

---

### Finding [R4-7]: Orphaned `check_input_gate` and Unused `limits` Parameter in `build_task_input`
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: Low
- **Location**: `src/review_bot/application/context.py:331-361, 368`
- **Evidence / Rationale**:
  `check_input_gate` is defined in `context.py` but never called in runtime code (`worker.py` or `cli.py`). Furthermore, `build_task_input` accepts `limits: dict[str, object] | None = None` in its signature, but never references `limits` in its body.
- **Actionable Remediation**:
  Invoke `check_input_gate` inside `build_task_input` or clean up the unused signature parameter.

---

### Finding [R4-8]: Indiscriminate Binary Blob Decoding Causes Permanent `not_run, incomplete_code_input` on Mixed Repositories
- **Perspective**: R4. LLM Pipeline & Hallucination Defense
- **Severity**: Low
- **Location**: `src/review_bot/adapters/git.py:293-333`
- **Evidence / Rationale**:
  In `collect_code_snapshot`, `read_tree_blobs` fetches all blobs and attempts `content_bytes.decode("utf-8")` on every single blob. If decoding fails (e.g. `.png`, `.jar`, `.bin`), it appends an error to `errors`. Because `complete = len(errors) == 0`, any repository containing binary assets has `complete = False`, causing `run_review_a_pipeline` to permanently return `not_run, incomplete_code_input`.
- **Actionable Remediation**:
  Classify non-code extensions as binary exclusions without marking the code snapshot `complete = False`.

---

## 7. Section 5: R5. GitHub Integration & UX Reliability Audit

### Finding [R5-1]: Worker Execution Loop Completely Bypasses Comment Publication and Markdown Formatting
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Critical
- **Location**: `src/review_bot/runtime/worker.py:274, 284-326`
- **Evidence / Rationale**:
  In `WorkerLoop.process_run()`, Stage 6 runs `run_review_a_pipeline`, but its result is ignored. Stage 8 checks GitHub PR freshness, but never imports or calls `format_review_markdown()` or `publish_review_comment()`. Instead, it fabricates `PublicationReceipt(comment_id=None, status="succeeded")`. In background worker mode, the bot never posts or updates comments on GitHub PRs.
- **Actionable Remediation**:
  Wire `format_review_markdown` and `publish_review_comment` into Stage 8 of `WorkerLoop`, recording the genuine receipt in `job_store`.

---

### Finding [R5-2]: Exception Handling in Publication Application Service Excludes `httpx.HTTPError`, Causing Unhandled Crashes on GitHub API Failures
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: High
- **Location**: `src/review_bot/application/publication.py:61, 126, 151, 176, 243`
- **Evidence / Rationale**:
  In `publish_review_comment()`, GitHub client calls are wrapped in `except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:`.
  However, `GitHubApiClient` uses `httpx` and issues `resp.raise_for_status()`. In Python, `httpx.HTTPStatusError` inherits from `httpx.HTTPError -> Exception`, NOT `OSError` or `RuntimeError`. When GitHub returns HTTP 401, 403 (Rate Limit), 404, 422, or 502/503, the exception escapes uncaught, crashing the worker process.
- **Actionable Remediation**:
  Expand exception handlers in `publication.py` to catch `(RuntimeError, ValueError, TypeError, OSError, KeyError, Exception)`.

---

### Finding [R5-3]: `GitHubApiClient.list_comments` Lacks Pagination, Causing Duplicate Comment Spams on Active PRs (>30 Comments)
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: High
- **Location**: `src/review_bot/adapters/github/client.py:53-60, 77-85`
- **Evidence / Rationale**:
  `GitHubApiClient.list_comments()` fetches `GET /repos/{repo}/issues/{pr_number}/comments` without pagination. GitHub defaults to 30 comments per page. On PRs with >30 comments, if the existing review comment is pushed to page 2, `find_marker_comment()` only checks page 1, finds nothing, and creates a duplicate comment. If subsequent runs see both comments, they raise `ValueError("Multiple comments with marker found")`, permanently locking the PR.
- **Actionable Remediation**:
  Implement pagination in `list_comments()`: pass `per_page=100` and follow RFC 5988 `Link: rel="next"` headers.

---

### Finding [R5-4]: TOCTOU Concurrency Race on Initial Comment Creation Permanently Locks PR via Unrecoverable `ambiguous_marker_comment`
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: High
- **Location**: `src/review_bot/application/publication.py:110-184`, `src/review_bot/adapters/github/client.py:83-85`
- **Evidence / Rationale**:
  On a new PR with no comments, if two workers execute concurrently, both observe `existing is None` and both call `create_comment()`. Two comments with marker `<!-- review-bot:architecture:v1 -->` are created. On all future runs, `find_marker_comment()` sees both comments and raises `ValueError: ambiguous_marker_comment`, freezing all future reviews until manual deletion.
- **Actionable Remediation**:
  In `find_marker_comment()`, if `len(matched) > 1`, log a warning and return the newest comment (`matched[-1]`) instead of raising `ValueError`.

---

### Finding [R5-5]: Secondary Truncation While-Loop in Markdown Formatter Drops Code Fence State Tracking, Producing Unclosed ```` ``` ```` Blocks
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/markdown/formatter.py:324-336`
- **Evidence / Rationale**:
  When comments exceed 60,000 characters, primary truncation appends ```` ``` ```` if `in_code_fence` is True.
  If the resulting length still exceeds 60,000, a secondary while-loop runs:
  `while len(truncated) > max_length and len(kept) > 10: removed = kept.pop()`.
  The first `pop()` removes the closing ```` ``` ````. The secondary loop only tracks `in_pre_code` and never tracks `in_code_fence`. The comment is rendered with an opened code fence and no closing fence, swallowing the omission banner, footer, and PR thread into a raw monospace box.
- **Actionable Remediation**:
  Track `in_code_fence` in the secondary while-loop and append ```` ``` ```` to `closing` when `in_code_fence` is True.

---

### Finding [R5-6]: Post-Check Freshness Verification Ignores Base SHA Drift, Leaving Comments Unflagged When Target Branch Moves
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Medium
- **Location**: `src/review_bot/application/publication.py:186-242`
- **Evidence / Rationale**:
  Pre-check verifies both head SHA and base SHA freshness. However, post-check (lines 186-242) checks `post_state != "open"` and `post_head != expected_head_sha`, but completely omits checking `post_base != expected_base_sha`. If commits were merged into `main` during execution, the 3-dot diff is obsolete, yet no warning banner is injected.
- **Actionable Remediation**:
  Add `post_base != expected_base_sha` check to post-check in `publication.py`.

---

### Finding [R5-7]: Post-Check Stale Banner Injection Does Not Re-Truncate, Risking 422 Rejection on Near-Limit Comments & Swallowing Failures as `status="succeeded"`
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Medium
- **Location**: `src/review_bot/application/publication.py:205-246`
- **Evidence / Rationale**:
  When post-check detects a stale condition, it prepends a ~150 char warning banner to `clean_body` without re-truncating. If `clean_body` was at 65,500 chars, it exceeds GitHub's 65,536 limit. If `update_comment` fails, the error is caught at line 243 and falls through to line 246, returning `PublicationReceipt(status="succeeded")`, falsely reporting success!
- **Actionable Remediation**:
  Re-truncate before updating, and return `status="stale"` with error reason if updating fails.

---

### Finding [R5-8]: `_a_escape` Over-Escapes Markdown Punctuation Inside Inline Code Spans, Causing Visual Artifacts Like `ruleset\.xml` and `not\_run`
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Low
- **Location**: `src/review_bot/adapters/markdown/formatter.py:15-20, 89, 119`
- **Evidence / Rationale**:
  `_a_escape` escapes punctuation (`.` -> `\.`, `_` -> `\_`). In Markdown inline code spans (`` `...` ``), backslashes are rendered literally. `_a_escape("ruleset.xml")` renders as `` `ruleset\.xml` ``, adding visual noise.
- **Actionable Remediation**:
  Use dedicated `_a_code_span(text)` without backslash-escaping periods, hyphens, or underscores.

---

### Finding [R5-9]: Absence of GitHub API Rate-Limiting and Connection Pooling in `GitHubApiClient`
- **Perspective**: R5. GitHub Integration & UX Reliability
- **Severity**: Low
- **Location**: `src/review_bot/adapters/github/client.py:48-51, 56-59, 102-105, 125-128`
- **Evidence / Rationale**:
  Each method instantiates an ephemeral `httpx.Client()` context manager, destroying connection pools and requiring 4 separate TLS handshakes per publication. It also ignores `X-RateLimit-Remaining` headers.
- **Actionable Remediation**:
  Maintain a persistent `httpx.Client` instance and handle HTTP 429 backoff.

---

## 8. Section 6: R6. Software Architecture & Maintainability Audit

### Finding [R6-1]: Hardcoded Review Mode Registry and Perspective Coupling Violates OCP
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: High
- **Location**: `src/review_bot/application/selection.py:11, 63-65`, `src/review_bot/domain/envelope_v2.py:102`
- **Evidence / Rationale**:
  `selection.py:11` defines `REGISTERED_MODES = frozenset({"architecture"})` and rejects any other mode. `ExecutionReceipt` hardcodes `a_result: ReviewResult | None = None`. `worker.py:274` hardcodes `run_review_a_pipeline`. Adding Review B (Intent Review) cannot be done without modifying domain contracts and worker loops, violating the Open-Closed Principle.
- **Actionable Remediation**:
  Introduce a `ReviewModeRegistry` and polymorphic `perspective_results: tuple[PerspectiveResult, ...]` in `ExecutionReceipt`.

---

### Finding [R6-2]: Static Analysis Engine and Delta Engine Tightly Coupled to Java & PMD/CPD
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: High
- **Location**: `src/review_bot/domain/models_v1.py:149-150`, `src/review_bot/domain/status.py:9-14`, `src/review_bot/application/delta.py:131, 155, 230-248`, `src/review_bot/application/context.py:83-95`
- **Evidence / Rationale**:
  `AnalysisRun` hardcodes `pmd: ToolAnalysis` and `cpd: ToolAnalysis`. `delta.py` rejects non-Java languages and enforces PMD 7.3.0 CLI commands. Supporting Python (Ruff) or TypeScript (ESLint) requires invasive refactoring across domain models.
- **Actionable Remediation**:
  Refactor `AnalysisRun` to `tools: dict[str, ToolAnalysis]` and extract tool verification into Strategy pattern `ToolValidator`.

---

### Finding [R6-3]: Leaky Adapter / Mock Implementations Co-located Inside Application Layer
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Medium
- **Location**: `src/review_bot/application/review_a.py:60-103`
- **Evidence / Rationale**:
  `UnconfiguredModelClient` and `MockModelClient` are defined in `application/review_a.py`. Concrete port adapters belong in `adapters/` (`adapters/model/clients.py`), keeping `application/` purely focused on use-case logic.
- **Actionable Remediation**:
  Create `src/review_bot/adapters/model/` and move client implementations there.

---

### Finding [R6-4]: Leaky Protocol Abstraction and Signature Ambiguity in GitHubClient
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Medium
- **Location**: `src/review_bot/application/publication.py:16-24, 161-164`, `src/review_bot/adapters/github/client.py:35-44`
- **Evidence / Rationale**:
  `publication.py` introduces `_call_client_method(method, *primary_args, fallback_args=...)`, catching `TypeError` to retry methods with alternate arguments. `GitHubApiClient` uses varargs `*args, **kwargs` and heuristic resolution, masking signature mismatches.
- **Actionable Remediation**:
  Standardize `GitHubClient` protocol signatures across all callers and remove `_call_client_method` fallback wrapper.

---

### Finding [R6-5]: Shallow Immutability in Frozen Domain Models Containing Mutable Dictionaries
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Low
- **Location**: `src/review_bot/domain/envelope_v2.py:103-104`, `src/review_bot/domain/models_v1.py:181, 231, 345, 371`
- **Evidence / Rationale**:
  Frozen dataclasses contain mutable `dict` fields (e.g. `coverage: dict`, `rename_map: dict`). While reference reassignment is prevented, dictionary contents can be mutated in place, risking shared-state corruption in multi-threaded workers.
- **Actionable Remediation**:
  Wrap dictionary arguments in `types.MappingProxyType` or use frozen dataclasses/tuples.

---

### Finding [R6-6]: Worker Loop Lacks Dynamic Review Dispatch and Bypasses Publication Pipeline
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Medium
- **Location**: `src/review_bot/runtime/worker.py:267-275, 305-321`
- **Evidence / Rationale**:
  Identified in R2-5, R4-3, R5-1. Worker loop skips publication and hardcodes Review A pipeline invocation.
- **Actionable Remediation**:
  Wire publication into worker loop and dispatch reviews via registry.

---

### Finding [R6-7]: Missing Automated Architectural and Boundary Tests in Test Suite
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Medium
- **Location**: `tests/unit/test_domain_purity.py:50-52`
- **Evidence / Rationale**:
  `test_domain_purity.py` verifies AST purity for `domain/`, but ignores all relative imports with `if node.level > 0: continue` (allowing `from ..adapters import git`). Furthermore, there is zero AST boundary test protecting `application/` from importing `adapters/` or `runtime/`.
- **Actionable Remediation**:
  Create `test_architecture_boundaries.py` enforcing `application/` import purity and restricting relative imports in `domain/` to `node.level == 1`.

---

### Finding [R6-8]: Architectural Strengths: Hexagonal Core Separation & Clean Dependency Graph
- **Perspective**: R6. Software Architecture & Maintainability
- **Severity**: Info
- **Location**: `src/review_bot/domain/`, `src/review_bot/ports.py`, `src/review_bot/application/`
- **Evidence / Rationale**:
  Clean hexagonal separation: domain is 100% pure; application imports only domain and ports; runtime serves as pure composition roots; database writes are abstracted via protocols.

---

### Technical Debt Scoring & Prioritization (HarnessKit Framework)

Using the formula $\text{Priority} = (\text{Impact} + \text{Risk}) \times (6 - \text{Effort})$:

| ID | Description | Category | Location | Impact (1-5) | Risk (1-5) | Effort (1-5) | Priority Score |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **TD-1** | Worker Publication Disconnection | Operational | `runtime/worker.py:305` | 5 | 4 | 2 | **36** |
| **TD-2** | Missing Application Boundary Tests | Test Debt | `tests/unit/test_domain_purity.py` | 3 | 3 | 1 | **30** |
| **TD-3** | Model Clients Inside Application Layer | Design | `application/review_a.py:60` | 3 | 2 | 1 | **25** |
| **TD-4** | Hardcoded Review Mode Registry & OCP | Design | `application/selection.py:11` | 4 | 4 | 3 | **24** |
| **TD-5** | GitHubClient Protocol Signature Ambiguity | Code Debt | `application/publication.py:16` | 3 | 3 | 2 | **24** |
| **TD-6** | Shallow Immutability in Frozen Domain Models | Code Debt | `domain/envelope_v2.py:103` | 2 | 2 | 2 | **16** |
| **TD-7** | Hardcoded Java/PMD/CPD Delta Constraints | Design | `application/delta.py:131` | 4 | 3 | 4 | **14** |

---

## 9. Section 7: Test Coverage Gap Analysis (15 Edge Case Specifications)

Audit of the 967 automated tests identified 15 high-impact edge case gaps. Concrete Given/When/Then specifications are defined below:

### Finding [GAP-1]: Ambiguous & Unavailable Merge-Base Handling Bypassed in CLI & Worker Pipelines
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Critical
- **Location**: `src/review_bot/runtime/cli.py:492`, `src/review_bot/runtime/worker.py:219`
- **Evidence / Rationale**:
  `cli.py:492` and `worker.py:219` fall back to `target_base_sha` when merge base is ambiguous. Existing tests in `test_pairwise_interactions.py:145-164` use tautological checks (`assert status == "comparable"`).
- **Actionable Remediation**:
  - **Test Name**: `test_criss_cross_merge_refuses_false_comparison_in_cli_and_worker`
  - **Given**: Synthetic git repo with criss-cross merge topology having two common ancestors C1 and C2 (`status="ambiguous"`).
  - **When**: `review-bot analyze` CLI or `WorkerLoop.process_run` executes.
  - **Then**: Neither component falls back to `target_base_sha`; CLI exits with code 1; worker marks run `comparison_status='ambiguous'` without computing delta.

---

### Finding [GAP-2]: Untested Git Shallow Clone Boundary Truncation in Git Adapter
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/adapters/git.py:106-163`
- **Evidence / Rationale**:
  In shallow clones (`--depth 1`), `git merge-base` fails when the common ancestor is severed. Zero tests exercise shallow repositories.
- **Actionable Remediation**:
  - **Test Name**: `test_resolve_merge_base_shallow_clone_severed_history_handling`
  - **Given**: Git repo truncated with `--depth 1` such that common ancestor is severed from HEAD.
  - **When**: `resolve_merge_base(repo_dir, target_base_sha, head_sha)` is called.
  - **Then**: Returns `MergeBaseResult(comparison_status="unavailable", comparison_base_sha=None)` without unhandled `GitCommandError`.

---

### Finding [GAP-3]: Webhook Receiver Starvation Under Slowloris / Streaming Timeouts & Mid-Stream Disconnect
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/runtime/webhook.py:68-76`
- **Evidence / Rationale**:
  No streaming timeout or `ClientDisconnect` handling.
- **Actionable Remediation**:
  - **Test Name**: `test_webhook_streaming_slowloris_timeout_and_client_disconnect`
  - **Given**: Client streams body chunks at 1 byte per 15 seconds, or abruptly closes connection mid-stream.
  - **When**: POST `/webhook` is processed.
  - **Then**: Terminates with HTTP 408 Request Timeout; mid-stream disconnect handled cleanly without unhandled 500 exceptions.

---

### Finding [GAP-4]: Webhook Unhandled 500 Crash on Valid Non-Dictionary JSON Payloads
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/runtime/webhook.py:107-124`
- **Evidence / Rationale**:
  Posting valid JSON array `[]` causes `payload_json.get()` to raise `AttributeError: 'list' object has no attribute 'get'`, returning 500 instead of 400.
- **Actionable Remediation**:
  - **Test Name**: `test_webhook_rejects_valid_non_dict_json_payloads_with_400`
  - **Given**: Request with valid HMAC signature whose body is `[]`, `"string"`, or `123`.
  - **When**: POST `/webhook` is called.
  - **Then**: Returns HTTP 400 Bad Request ("root must be an object"), not HTTP 500.

---

### Finding [GAP-5]: SQLite Lock Starvation and Busy Timeout Exhaustion in Webhook & Worker
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/runtime/webhook.py:137-185`, `src/review_bot/runtime/worker.py:90-96`
- **Evidence / Rationale**:
  When lock is held past 5s, `OperationalError: database is locked` bubbles up to FastAPI as unhandled 500 and crashes worker heartbeat daemon.
- **Actionable Remediation**:
  - **Test Name**: `test_webhook_and_worker_heartbeat_under_sqlite_busy_timeout`
  - **Given**: Raw connection holding `BEGIN EXCLUSIVE` lock on SQLite file for 6.0 seconds.
  - **When**: Webhook receives delivery, and worker attempts `renew_lease`.
  - **Then**: Webhook returns HTTP 503 (with Retry-After); worker heartbeat backs off without crashing daemon thread.

---

### Finding [GAP-6]: Stale Worker Lease Loss Overwrites Active Successor Worker (Adversarial Bug Asserted Instead of Defended)
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Critical
- **Location**: `src/review_bot/adapters/sqlite/store.py:261-298`, `tests/unit/test_sqlite_adversarial_stress.py:727-786`
- **Evidence / Rationale**:
  `test_stale_worker_lease_loss_overwrites_active_successor` asserts that stale Worker A successfully clobbers Worker B, demonstrating the bug without asserting a defense.
- **Actionable Remediation**:
  - **Test Name**: `test_sqlite_job_store_update_run_fencing_rejects_stale_worker_updates`
  - **Given**: Worker A's lease expires; Worker B reclaims run R1.
  - **When**: Worker A attempts `update_run(run_id=R1, worker_id="worker-a", status=SUPERSEDED)`.
  - **Then**: Query matches 0 rows; Worker B retains active ownership and succeeds in renewing its lease.

---

### Finding [GAP-7]: PmdRunner Exit Code Concordance False Mismatch on CPD Duplications
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/adapters/pmd/runner.py:329-336`
- **Evidence / Rationale**:
  PMD 7 CPD exits with code 0 on findings. Line 329 flags `exit_code == 0 and finding_count > 0` as `exit_report_mismatch`, falsely failing CPD analysis.
- **Actionable Remediation**:
  - **Test Name**: `test_pmd_runner_cpd_duplication_detected_without_concordance_failure`
  - **Given**: Two Java files containing identical duplicated code blocks (>100 tokens).
  - **When**: `PmdRunner.run_analysis` executes.
  - **Then**: CPD findings are parsed, concordance does not report mismatch, and CPD status is `"findings"`, not `"failed"`.

---

### Finding [GAP-8]: PMD XML Parser Rejects Empty File Tags and Interleaved Child Elements
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/pmd/parsers.py:237-250, 373-400`
- **Evidence / Rationale**:
  Strict order check raises `ValueError` if `<error>` precedes `<file>`, and rejects `<file name="Clean.java"/>` with 0 violations.
- **Actionable Remediation**:
  - **Test Name**: `test_pmd_parser_handles_empty_file_tag_and_interleaved_elements`
  - **Given**: Valid PMD 7 XML containing `<file name="Clean.java"/>` (0 violations) and interleaved `<error>` elements.
  - **When**: `parse_pmd_xml` runs.
  - **Then**: Parses without raising `ValueError`, extracting all violations and errors.

---

### Finding [GAP-9]: Code Fence Balancing Lost During Aggressive Truncation in Markdown Formatter
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/adapters/markdown/formatter.py:317-336`
- **Evidence / Rationale**:
  Secondary aggressive while-loop pops closing ```` ``` ```` and only re-closes `in_pre_code`.
- **Actionable Remediation**:
  - **Test Name**: `test_markdown_formatter_balances_code_fences_during_aggressive_trimming`
  - **Given**: Review result with code blocks sized so aggressive trimming is triggered.
  - **When**: `format_review_markdown(result, task_input, max_length=15000)` runs.
  - **Then**: `markdown.count("```") % 2 == 0` is strictly verified.

---

### Finding [GAP-10]: UTF-8 Multi-Byte Character Expansion Exceeding GitHub Comment Byte Limits
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/adapters/markdown/formatter.py:12, 60, 279-281`
- **Evidence / Rationale**:
  Formatter checks `len(rendered) <= 60000` chars. In Korean/Unicode, 55,000 chars = 165,000 bytes, exceeding GitHub's 65,536 byte limit (HTTP 422).
- **Actionable Remediation**:
  - **Test Name**: `test_markdown_formatter_enforces_byte_budget_for_multibyte_unicode`
  - **Given**: Review result with Korean text totaling 45,000 chars but >70,000 UTF-8 bytes.
  - **When**: `format_review_markdown(..., max_bytes=65536)` runs.
  - **Then**: `len(rendered.encode("utf-8")) <= 65536` is strictly satisfied without breaking multibyte characters.

---

### Finding [GAP-11]: Post-Publication Stale Banner Concatenation Exceeds 60,000 Character Cap
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/application/publication.py:201-245`
- **Evidence / Rationale**:
  Post-check prepends stale banner to 60,000-char body without re-truncating, and swallows update failures as `status="succeeded"`.
- **Actionable Remediation**:
  - **Test Name**: `test_post_publication_stale_banner_retruncates_within_limit_and_handles_update_failure`
  - **Given**: Review body of exact length 60,000 chars, with PR head moving during posting.
  - **When**: `publish_review_comment` performs post-check stale banner injection.
  - **Then**: `len(stale_body) <= 60000` is maintained; if update fails, receipt status is `"failed"`, not `"succeeded"`.

---

### Finding [GAP-12]: Production `GitHubApiClient` Completely Untested (Zero Coverage on Pagination, Rate Limits, and Errors)
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Critical
- **Location**: `src/review_bot/adapters/github/client.py:10-130`
- **Evidence / Rationale**:
  `GitHubApiClient` has 0 tests across all 967 tests. 100% of tests use `MockGitHubClient`.
- **Actionable Remediation**:
  - **Test Name**: `test_github_api_client_paginates_comments_and_handles_rate_limits`
  - **Given**: Mocked HTTP transport returning 60 comments across 2 pages with marker comment on page 2, and HTTP 403 secondary rate limit response.
  - **When**: `find_marker_comment` and `create_comment` are called.
  - **Then**: Paginates to page 2 and finds existing comment; rate limits handle exponential backoff.

---

### Finding [GAP-13]: Response Gate Validation Omissions: Unvalidated Finding `evidence_ids`, `criterion_ids`, `observation_ids`, and Judgements
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: High
- **Location**: `src/review_bot/application/review_a.py:320-324, 440-524`
- **Evidence / Rationale**:
  Finding `evidence_ids` and enums are never validated against `task_input.evidence` or allowed values.
- **Actionable Remediation**:
  - **Test Name**: `test_response_gate_rejects_hallucinated_evidence_ids_and_invalid_judgements`
  - **Given**: Response JSON containing `finding["evidence_ids"] = ["E-FAKE"]`, `finding["judgement"] = "hallucinated"`, and `citation["quote"] = ""`.
  - **When**: `validate_model_response` is called.
  - **Then**: Returns errors: unknown evidence ID, invalid judgement, and empty quote rejection.

---

### Finding [GAP-14]: Delta Engine Ambiguity When Class Contains Multiple Identical Methods with One Modified
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/application/delta.py:464-476, 559-573`
- **Evidence / Rationale**:
  When base has 2 identical method violations and head has 1, line 564 unconditionally emits `unknown` rather than matching 1 as `existing` and 1 as `resolved`.
- **Actionable Remediation**:
  - **Test Name**: `test_delta_engine_overloaded_identical_methods_partial_fix`
  - **Given**: Java class with two overloaded methods with identical complexity in base; one method refactored in head.
  - **When**: `build_delta` runs.
  - **Then**: Emits 1 `resolved` signal and 1 `existing` signal, instead of ambiguous `unknown`.

---

### Finding [GAP-15]: Context Assembly Gate Binary File Coexistence Taints Java Signals as Uncertain
- **Perspective**: Test Coverage Gap Analysis
- **Severity**: Medium
- **Location**: `src/review_bot/application/context.py:97-135`
- **Evidence / Rationale**:
  If a PR modifies a Java file and adds a `.png` binary asset, binary decode failure sets `snapshot.complete = False`, causing all Java changed-spans to be marked uncertain and demoted from review to limitations.
- **Actionable Remediation**:
  - **Test Name**: `test_context_assembly_gate_binary_file_coexistence_does_not_taint_java_signals`
  - **Given**: PR modifying `OrderService.java` and adding `diagram.png`.
  - **When**: `build_task_input` executes.
  - **Then**: `OrderService.java` signals retain `"review"` role; binary exclusion is isolated to snapshot errors.

---

## 10. Test Verification & Verification Method

### 10.1 Empirical Test Suite Results
The full test suite was executed in the workspace environment:
- **Command**: `uv run pytest`
- **Execution Date**: 2026-10-08T13:02:57Z
- **Environment**: macOS Darwin, Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
- **Total Test Count**: **967 tests collected, 967 passed** (100% green, 1 warning)
- **Duration**: 125.90s (2 minutes 5 seconds)

```
================== 967 passed, 1 warning in 125.90s (0:02:05) ==================
```

### 10.2 Zero Code Mutation Attestation
In strict accordance with the read-only audit mandate:
- **Verification Command**: `find src tests -type f -mmin -30`
- **Result**: Zero files modified. All source files in `src/review_bot/` and tests in `tests/` remain completely untouched.
- **Attestation**: The audit findings, readiness scores, and gap specifications represent pure, objective evaluation of the codebase without artificial workarounds or mutations.
