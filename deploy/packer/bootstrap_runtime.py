#!/usr/bin/env python3
import json
import os
import pwd
import sys
from pathlib import Path

import boto3


def write_private(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        if not value.endswith("\n"):
            stream.write("\n")
    os.chmod(path, 0o600)
    os.chown(path, pwd.getpwnam("reviewbot").pw_uid, pwd.getpwnam("reviewbot").pw_gid)


def secret(client, secret_arn: str) -> str:
    response = client.get_secret_value(SecretId=secret_arn)
    value = response.get("SecretString")
    if not isinstance(value, str) or not value:
        raise ValueError("required Secrets Manager value is empty")
    return value


def main() -> None:
    github_arn, openrouter_arn, region = sys.argv[1:4]
    client = boto3.client("secretsmanager", region_name=region)
    github_credentials = json.loads(secret(client, github_arn))
    app_id = str(github_credentials["app_id"])
    private_key = github_credentials["private_key"]
    if not app_id or not isinstance(private_key, str) or not private_key:
        raise ValueError("GitHub App secret must contain app_id and private_key")
    openrouter_key = secret(client, openrouter_arn)

    directory = Path("/run/review-bot")
    reviewbot = pwd.getpwnam("reviewbot")
    os.chown(directory, reviewbot.pw_uid, reviewbot.pw_gid)
    os.chmod(directory, 0o700)
    write_private(directory / "github-app-id", app_id)
    write_private(directory / "github-app-private-key.pem", private_key)
    write_private(directory / "openrouter-api-key", openrouter_key)

    pin = Path("/etc/review-bot/pmd-image-pin").read_text(encoding="utf-8").strip()
    if not pin.startswith("sha256:") or len(pin) != 71:
        raise ValueError("PMD image is not pinned by a SHA-256 image ID")
    config = f"""schema_version: 2
codebase:
  product: github-review-bot
  python_package: review_bot
  python_min_version: "3.11"
  layout: src
execution:
  mode: github_app
  backend: github_app
  available_modes: [architecture, intent]
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
  state_root_env: REVIEW_BOT_STATE_DIR
  app_id_env: REVIEW_BOT_APP_ID
  app_private_key_file_env: REVIEW_BOT_APP_PRIVATE_KEY_FILE
  webhook_secret_env: REVIEW_BOT_WEBHOOK_SECRET
publish:
  mode: comment
  separate_comments_per_task: true
  b_comments_enabled: true
  draft_comments: false
  automatic_approval: false
  automatic_request_changes: false
  recheck_before_and_after_write: true
  legacy_migration:
    enabled: false
model:
  profile: null
  unconfigured_action: not_run
  input_policy: full_or_not_run
  overflow_action: not_run
  max_concurrency: 1
  call_timeout_seconds: 60
  max_attempts: 1
review_engine: agent
agent_engine:
  image_tag: review-bot-pi-agent:0.73.1
  model_id: stealth/crouton-alpha
  timeout_seconds: 900
  max_proxy_calls: 40
  max_output_tokens: 4096
  session_store_dir: /var/lib/review-bot/agent-sessions
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
  tool_image_digest: {pin}
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
  language: en
  max_findings_per_task: 20
  max_questions_per_task: 20
  max_comment_characters: 60000
  include_coverage: true
  distinguish_prepared_sent_and_opinion_scope: true
  evidence_ids_required: true
"""
    path = Path("/etc/review-bot/operator.yaml")
    path.write_text(config, encoding="utf-8")
    os.chown(path, reviewbot.pw_uid, reviewbot.pw_gid)
    os.chmod(path, 0o640)
    state_dir = Path("/var/lib/review-bot")
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chown(state_dir, reviewbot.pw_uid, reviewbot.pw_gid)
    os.chmod(state_dir, 0o700)


if __name__ == "__main__":
    main()
