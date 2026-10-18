"""Shared test fixtures and test infrastructure for github-review-bot test suite.

Provides:
- SyntheticGitRepo: In-memory/temp-dir git repository factory with complex branch topologies
- java_fixtures: Synthetic Java code with specific metric and architectural profiles
- config_fixtures: Valid and invalid YAML configuration payloads
- mock_github_client: Mock GitHub API client and webhook signing utility
- pmd_xml_fixtures: Strict XML fixtures (valid, invalid, XXE defense)
- temp_sqlite_db: Temporary SQLite WAL test database
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import subprocess
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root and src are importable if present
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

REFERENCE_CONFIG_PATH = (
    PROJECT_ROOT.parent / "specs" / "20261002-github-review-bot" / "review-config.example.yaml"
)


# ==============================================================================
# Synthetic Git Repository Generator
# ==============================================================================


class SyntheticGitRepo:
    """Manages a temporary Git repository for deterministic testing of merge-bases,

    file histories, rename tracking, and tree blob reads.
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir
        self._git("init", "-b", "main")
        self._git("config", "user.name", "ReviewBotTester")
        self._git("config", "user.email", "tester@review-bot.local")
        self._git("config", "commit.gpgsign", "false")
        # Ensure main branch is established with an initial commit
        self.commit("Initialize repository")

    def _git(self, *args: str, input_data: bytes | None = None) -> str:
        cmd = ["git", "-C", str(self.repo_dir), *args]
        proc = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            check=True,
            timeout=15,
        )
        return proc.stdout.decode("utf-8", errors="replace").strip()

    def write_file(self, rel_path: str, content: str | bytes) -> Path:
        target = self.repo_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
        return target

    def delete_file(self, rel_path: str) -> None:
        target = self.repo_dir / rel_path
        if target.exists():
            target.unlink()

    def commit(self, message: str, paths: list[str] | None = None) -> str:
        if paths:
            for p in paths:
                self._git("add", p)
        else:
            self._git("add", "-A")
        self._git("commit", "-m", message, "--allow-empty")
        return self.get_head_sha()

    def create_branch(self, branch_name: str, start_point: str | None = None) -> None:
        if start_point:
            self._git("branch", branch_name, start_point)
        else:
            self._git("branch", branch_name)

    def checkout(self, branch_or_sha: str) -> None:
        self._git("checkout", branch_or_sha)

    def get_head_sha(self) -> str:
        return self._git("rev-parse", "HEAD")

    def get_commit_sha(self, rev: str) -> str:
        return self._git("rev-parse", rev)

    def compute_merge_base(self, rev1: str, rev2: str, all_bases: bool = False) -> list[str]:
        args = ["merge-base"]
        if all_bases:
            args.append("--all")
        args.extend([rev1, rev2])
        try:
            output = self._git(*args)
            return [line.strip() for line in output.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            return []

    def setup_standard_feature_pr(
        self,
        base_branch: str = "main",
        feature_branch: str = "feature/payment",
        advance_base: bool = True,
    ) -> dict[str, str]:
        """Creates a standard feature PR:

        1. Root commit on base (common ancestor = comparison_base_sha)
        2. Feature branch forks and adds commits (head_sha)
        3. If advance_base is True, base branch moves ahead (target_base_sha != comparison_base_sha)
        """
        self.checkout(base_branch)
        self.write_file("build.gradle", "plugins { id 'java' }\n")
        self.write_file("src/main/java/Root.java", "public class Root {}")
        comparison_base = self.commit("Initial root commit")

        # Create and advance feature branch
        self.checkout(base_branch)
        self.create_branch(feature_branch, comparison_base)
        self.checkout(feature_branch)
        self.write_file("src/main/java/PaymentService.java", "public class PaymentService {}")
        head_sha = self.commit("Add payment service")

        # Advance base branch if requested
        if advance_base:
            self.checkout(base_branch)
            self.write_file("docs/README.md", "# Project Documentation")
            target_base = self.commit("Update docs on main")
        else:
            target_base = comparison_base

        return {
            "comparison_base_sha": comparison_base,
            "target_base_sha": target_base,
            "head_sha": head_sha,
            "base_branch": base_branch,
            "feature_branch": feature_branch,
        }

    def setup_criss_cross_merge(self) -> dict[str, Any]:
        r"""Creates a criss-cross merge topology resulting in multiple merge-bases

        (ambiguous merge-base test case).
        Topology:
           A (root)
          / \
         B   C
         |\ /|
         | X |
         |/ \|
         D   E
        where merge-base(D, E) has two candidates: B and C.
        """
        self.checkout("main")
        self.write_file("root.txt", "root")
        root = self.commit("Root commit")

        # Branch 1
        self.create_branch("branch-1", root)
        self.checkout("branch-1")
        self.write_file("file-b.txt", "content B")
        sha_b = self.commit("Commit B")

        # Branch 2
        self.create_branch("branch-2", root)
        self.checkout("branch-2")
        self.write_file("file-c.txt", "content C")
        sha_c = self.commit("Commit C")

        # D merges B and C
        self.create_branch("branch-d", sha_b)
        self.checkout("branch-d")
        self._git("merge", "--no-ff", "-m", "Merge C into D", sha_c)
        sha_d = self.get_head_sha()

        # E merges C and B
        self.create_branch("branch-e", sha_c)
        self.checkout("branch-e")
        self._git("merge", "--no-ff", "-m", "Merge B into E", sha_b)
        sha_e = self.get_head_sha()

        bases = self.compute_merge_base(sha_d, sha_e, all_bases=True)
        return {
            "sha_d": sha_d,
            "sha_e": sha_e,
            "merge_bases": bases,
            "is_ambiguous": len(bases) > 1,
        }

    def setup_orphan_branches(self) -> tuple[str, str]:
        """Creates two independent orphan commits with no common ancestor."""
        self.checkout("main")
        self.write_file("main_file.txt", "main content")
        main_sha = self.commit("Main commit")

        self._git("checkout", "--orphan", "orphan-branch")
        self.write_file("orphan_file.txt", "orphan content")
        orphan_sha = self.commit("Orphan commit")
        return main_sha, orphan_sha


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> SyntheticGitRepo:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    return SyntheticGitRepo(repo_dir)


# ==============================================================================
# Synthetic Java Source Fixtures
# ==============================================================================


class JavaSourceFixtures:
    """Standard Java code fixtures designed to test PMD complexity, CPD duplication,

    God Class patterns, and refactoring diffs.
    """

    CLEAN_SERVICE = """package demo;

public class CleanService {
    public int add(int a, int b) {
        return a + b;
    }

    public boolean isValid(String name) {
        return name != null && !name.trim().isEmpty();
    }
}
"""

    COMPLEX_CALCULATOR = """package demo;

public class ComplexCalculator {
    // Cognitive complexity: ~26, Cyclomatic: ~14, NPath: >200
    public int calculate(int a, int b, int mode, boolean flag, String operation) {
        int result = 0;
        if (a > 0) {
            if (b > 0) {
                for (int i = 0; i < a; i++) {
                    if (flag && operation != null) {
                        switch (mode) {
                            case 1:
                                result += (i % 2 == 0) ? b : -b;
                                break;
                            case 2:
                                if (i > 5) {
                                    result *= 2;
                                } else {
                                    result += b;
                                }
                                break;
                            case 3:
                                while (result < 100) {
                                    result += (a + b);
                                }
                                break;
                            default:
                                result = 0;
                        }
                    } else {
                        result -= i;
                    }
                }
            } else if (b < -10) {
                result = a * b;
            }
        } else {
            result = -1;
        }
        return result;
    }
}
"""

    REFACTORED_CALCULATOR = """package demo;

public class ComplexCalculator {
    // Refactored: Extracted helper methods, lower complexity
    public int calculate(int a, int b, int mode, boolean flag, String operation) {
        if (a <= 0) {
            return -1;
        }
        if (b <= 0) {
            return (b < -10) ? a * b : 0;
        }
        return computePositive(a, b, mode, flag, operation);
    }

    private int computePositive(int a, int b, int mode, boolean flag, String operation) {
        int result = 0;
        for (int i = 0; i < a; i++) {
            result += step(i, b, mode, flag, operation);
        }
        return result;
    }

    private int step(int i, int b, int mode, boolean flag, String operation) {
        if (!flag || operation == null) {
            return -i;
        }
        return (mode == 1 && i % 2 == 0) ? b : -b;
    }
}
"""

    # Duplication fragment exceeding 100 PMD CPD tokens:
    CPD_DUPLICATE_FRAGMENT = """        int sum = 0;
        java.util.List<String> items = new java.util.ArrayList<>();
        for (int i = 0; i < 50; i++) {
            items.add("item_" + i);
            sum += (i * 3) + (i % 7);
            if (sum > 500) {
                sum = sum / 2;
            } else {
                sum += 15;
            }
        }
        System.out.println("Processing completed: " + sum + ", total items: " + items.size());
"""

    DUPLICATE_CLASS_A = f"""package demo;

public class DuplicateA {{
    public void processA() {{
{CPD_DUPLICATE_FRAGMENT}
    }}
}}
"""

    DUPLICATE_CLASS_B = f"""package demo;

public class DuplicateB {{
    public void processB() {{
{CPD_DUPLICATE_FRAGMENT}
    }}
}}
"""

    GOD_CLASS = """package demo;

// God Class: excessive methods, excessive public count, excessive parameters
public class GodManager {
    public void method01(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8) {}
    public void method02() {}
    public void method03() {}
    public void method04() {}
    public void method05() {}
    public void method06() {}
    public void method07() {}
    public void method08() {}
    public void method09() {}
    public void method10() {}
    public void method11() {}
    public void method12() {}
    public void method13() {}
    public void method14() {}
    public void method15() {}
    public void method16() {}
    public void method17() {}
    public void method18() {}
    public void method19() {}
    public void method20() {}
    public void method21() {}
    public void method22() {}
    public void method23() {}
    public void method24() {}
    public void method25() {}
    public void method26() {}
    public void method27() {}
    public void method28() {}
    public void method29() {}
    public void method30() {}
    public void method31() {}
    public void method32() {}
}
"""


@pytest.fixture
def java_fixtures() -> JavaSourceFixtures:
    return JavaSourceFixtures()


# ==============================================================================
# Configuration YAML Fixtures
# ==============================================================================


@pytest.fixture
def valid_config_yaml() -> str:
    if REFERENCE_CONFIG_PATH.exists():
        return REFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
    # Fallback minimal valid YAML
    return """schema_version: 2
codebase:
  product: github-review-bot
  python_package: review_bot
  python_min_version: "3.11"
  layout: src
execution:
  mode: offline
  backend: github_app
  available_modes: [architecture]
  select_modes_for_all_prs: true
  run_only_selected_modes: true
  selector_policy: all_available
  selector_uses_model: false
  worker_concurrency: 1
  task_timeout_seconds: 900
  pr_actions: [opened, synchronize, reopened, edited, labeled, unlabeled, ready_for_review, closed]
service:
  receiver: fastapi_uvicorn
  worker: synchronous
  job_store: sqlite
  sqlite_wal: true
  lease_and_generation_fencing: true
  webhook_max_bytes: 10485760
  state_root_env: REVIEW_BOT_STATE_DIR
  app_id_env: REVIEW_BOT_APP_ID
  app_private_key_file_env: REVIEW_BOT_APP_PRIVATE_KEY_FILE
  webhook_secret_env: REVIEW_BOT_WEBHOOK_SECRET
  installation_allowlist: []
  repository_allowlist: []
publish:
  mode: artifact_only
  separate_comments_per_task: true
  a_marker: "<!-- review-bot:architecture:v1 -->"
  draft_comments: false
  automatic_approval: false
  automatic_request_changes: false
  recheck_before_and_after_write: true
  legacy_migration:
    enabled: false
    marker: "<!-- seulgi-review:A -->"
    exact_title: "## 리팩토링 신호 — 이번 PR 변경 파일 기준"
model:
  profile: null
  profile_env: REVIEW_MODEL_PROFILE
  unconfigured_action: not_run
  input_policy: full_or_not_run
  overflow_action: not_run
  require_token_count: true
  allow_summarization: false
  allow_truncation: false
  allow_automatic_splitting: false
  max_concurrency: 1
  call_timeout_seconds: 60
  max_attempts: 1
  max_output_bytes: 32768
  required_profile_limits: [max_input_bytes, max_input_tokens, max_output_tokens, max_total_tokens]
architecture:
  review_type: A
  analysis_profile: java21-jdk-only-v1
  tool_version: "7.3.0"
  runtime_java_version: "21"
  language_version: "21"
  cpd_minimum_tokens: 100
  encoding: UTF-8
  type_resolution: jdk_only
  ruleset_resource: pmd/ruleset.xml
  tools_lock_path: tooling/tools.lock.json
  tool_image_digest: null
  run_target_build_scripts: false
  tool_timeout_seconds: 120
  isolation:
    runtime: rootless_docker
    network: none
    read_only_root: true
    no_new_privileges: true
    cpus: 2
    memory_mib: 2048
    pids: 128
repository_settings:
  analysis_profile: java21-jdk-only-v1
  source_roots: [src/main/java]
  architecture_rule_paths: []
comparison:
  base_strategy: unique_merge_base
  record_target_base: true
  ambiguous_base_action: incomparable
code_input:
  versions: [comparison_base, head]
  include_all_source_tests_and_related_config: true
  exclusion_policy: trusted_builtin
  incomplete_required_input_action: not_run
artifacts:
  visibility: private
  retention_days: 30
  max_archive_bytes: 104857600
  max_unpacked_bytes: 402653184
  max_file_count: 20000
  publish_private_data_to_public_ci: false
output:
  language: ko
  max_findings_per_task: 20
  max_questions_per_task: 20
  max_comment_characters: 60000
  include_coverage: true
  distinguish_prepared_sent_and_opinion_scope: true
  evidence_ids_required: true
"""


@pytest.fixture
def duplicate_keys_yaml() -> str:
    return """schema_version: 2
codebase:
  product: github-review-bot
  product: duplicate-bot
"""


@pytest.fixture
def unknown_keys_yaml(valid_config_yaml: str) -> str:
    return valid_config_yaml + "\nextra_unrecognized_key: true\n"


@pytest.fixture
def invalid_type_yaml() -> str:
    return """schema_version: "two"
codebase:
  product: 12345
"""


# ==============================================================================
# Mock GitHub Client & Webhook Payload Factory
# ==============================================================================


@dataclass
class MockGitHubComment:
    id: int
    body: str
    created_at: str
    updated_at: str


class MockGitHubClient:
    """Mock GitHub client tracking comments, issues, and PR states."""

    def __init__(self) -> None:
        self.comments: dict[int, MockGitHubComment] = {}
        self._next_comment_id = 1
        self.pull_requests: dict[int, dict[str, Any]] = {}
        self.posted_comments: list[dict[str, Any]] = []
        self.updated_comments: list[dict[str, Any]] = []

    def set_pull_request(
        self,
        pr_number: int,
        state: str = "open",
        head_sha: str = "b" * 40,
        base_sha: str = "a" * 40,
        branch: str = "feature/test",
        labels: list[str] | None = None,
    ) -> None:
        self.pull_requests[pr_number] = {
            "number": pr_number,
            "state": state,
            "head": {"sha": head_sha, "ref": branch},
            "base": {"sha": base_sha, "ref": "main"},
            "labels": [{"name": name} for name in (labels or [])],
        }

    def get_pull_request(self, pr_number: int) -> dict[str, Any]:
        return self.pull_requests.get(pr_number, {"number": pr_number, "state": "open"})

    def find_marker_comment(
        self, pr_number: int, marker: str = "<!-- review-bot:architecture:v1 -->"
    ) -> MockGitHubComment | None:
        for comment in self.comments.values():
            if marker in comment.body:
                return comment
        return None

    def post_comment(self, pr_number: int, body: str) -> MockGitHubComment:
        cid = self._next_comment_id
        self._next_comment_id += 1
        comment = MockGitHubComment(cid, body, "2026-10-06T12:00:00Z", "2026-10-06T12:00:00Z")
        self.comments[cid] = comment
        self.posted_comments.append({"pr_number": pr_number, "id": cid, "body": body})
        return comment

    def update_comment(self, comment_id: int, body: str) -> MockGitHubComment:
        if comment_id not in self.comments:
            raise KeyError(f"Comment {comment_id} not found")
        comment = MockGitHubComment(
            comment_id, body, self.comments[comment_id].created_at, "2026-10-06T12:05:00Z"
        )
        self.comments[comment_id] = comment
        self.updated_comments.append({"id": comment_id, "body": body})
        return comment

    @staticmethod
    def sign_webhook_payload(payload_bytes: bytes, secret: str) -> str:
        sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return f"sha256={sig}"


@pytest.fixture
def mock_github() -> MockGitHubClient:
    return MockGitHubClient()


@pytest.fixture
def worker_review_inputs() -> tuple[Any, Any]:
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.application.architecture_review import MockModelClient, ModelProfile, run_review_a_pipeline
    from review_bot.domain.models_v1 import (
        Classification, CodeFile, CodeSnapshot, ModelResponse, PrSnapshot, Snapshot,
    )

    path = "src/main/java/demo/Service.java"
    base_source = "public class Service {\n    void run() {}\n}\n"
    head_source = "public class Service {\n    void run() {\n        int x = 1;\n    }\n}\n"
    base_hash = hashlib.sha256(base_source.encode()).hexdigest()
    head_hash = hashlib.sha256(head_source.encode()).hexdigest()
    snapshot = Snapshot(
        pr=PrSnapshot("org/repo", 1, "a" * 40, "b" * 40, "feature/test", changed_files=(path,)),
        base=CodeSnapshot(
            "a" * 40, (CodeFile(path, base_source, base_hash, "source"),),
            inventory_hash=hashlib.sha256(f"{path}:{base_hash}".encode()).hexdigest(),
        ),
        head=CodeSnapshot(
            "b" * 40, (CodeFile(path, head_source, head_hash, "source"),),
            inventory_hash=hashlib.sha256(f"{path}:{head_hash}".encode()).hexdigest(),
        ),
    )
    delta = compute_delta(
        base_findings=[],
        head_findings=[{"rule": "CognitiveComplexity", "path": path, "metric": 20, "method": "run"}],
        base_sha="a" * 40, head_sha="b" * 40,
    )
    task = build_task_input(snapshot, delta, Classification("general"))
    model = MockModelClient([ModelResponse(json.dumps({
        "base_sha": "a" * 40, "head_sha": "b" * 40, "review_mode": "general",
        "summary": "Worker architecture review completed.", "findings": [], "questions": [],
    }))])
    result = run_review_a_pipeline(task, model, profile=ModelProfile(provider="mock", model="review-test"))
    assert result.status == "completed", result.validation_errors
    return task, result


@pytest.fixture
def worker_pipeline_ports(worker_review_inputs: tuple[Any, Any]) -> dict[str, Any]:
    from unittest.mock import Mock

    from review_bot.application.architecture_review import MockModelClient
    from review_bot.domain.envelope_v2 import ChangedFile
    from review_bot.domain.models_v1 import ModelResponse
    from review_bot.ports import MergeBaseResult, RepositoryReader, StaticAnalyzer
    from tests.unit.test_delta_adversarial import make_pmd_violation, make_test_run

    task, result = worker_review_inputs
    path = task.snapshot.head.files[0].path
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("c" * 40, "a" * 40, "comparable")
    reader.get_changed_files.return_value = (ChangedFile(path),)
    reader.get_rename_map.return_value = {}
    code = {"a" * 40: task.snapshot.base, "b" * 40: task.snapshot.head}
    reader.collect_code_snapshot.side_effect = code.__getitem__
    def read_blobs(sha, paths=None):
        blobs = {"build.gradle": b"plugins { id 'java' }\n", **{
            file.path: file.content.encode("utf-8") for file in code[sha].files}}
        return blobs if paths is None else {path: content for path, content in blobs.items() if path in paths}

    reader.read_tree_blobs.side_effect = read_blobs
    runs = {
        "a" * 40: make_test_run("a" * 40, source_files=(path,)),
        "b" * 40: make_test_run("b" * 40, source_files=(path,), pmd_violations=(
            make_pmd_violation(path=path, begin_line=2, end_line=4, klass="Service", method="run"),
        )),
    }

    def analyze(sha: str, source_root: Path, *args: Any, **kwargs: Any) -> Any:
        assert isinstance(source_root, Path) and source_root.is_dir()
        for file in code[sha].files:
            assert (source_root / file.path).read_text(encoding="utf-8") == file.content
        return runs[sha]

    analyzer = Mock(spec=StaticAnalyzer)
    analyzer.run_analysis.side_effect = analyze
    model = MockModelClient([ModelResponse(json.dumps({
        "base_sha": result.base_sha, "head_sha": result.head_sha, "review_mode": result.review_mode,
        "summary": result.summary, "findings": [], "questions": [],
    }))])
    return {"repo_reader": reader, "analyzer": analyzer, "model_client": model}


# ==============================================================================
# Temporary SQLite Database (WAL mode)
# ==============================================================================


@pytest.fixture
def temp_sqlite_db(tmp_path: Path) -> Generator[Path, None, None]:
    db_file = tmp_path / "test_jobs.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()
    yield db_file
    if db_file.exists():
        db_file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def private_model_admission_state(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("model-admission")
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(root))
