# On-demand review: Lambda intake → one-shot Linux VM

Terraform in `deploy/terraform/` is the recommended deployment path. The CloudFormation template at
`deploy/aws/template.yaml` is retained as an equivalent reference. LocalStack is used for free service emulation;
no real AWS account, EC2 worker, or GitHub delivery has been used to verify this deployment.

## Flow

```text
            ┌────────────── current (persistent host) ───────────────┐
GitHub PR ─▶│ FastAPI webhook ─▶ SQLite runs/lease ─▶ WorkerLoop ─▶ A/B comment │
 webhook    └─────────────────────────────────────────────────────────┘
            ┌────────────── on-demand (this design) ─────────────────────────────────────────┐
            │ Lambda intake ──▶ DynamoDB job (conditional writes) ──▶ EC2 RunInstances          │
            │  · HMAC verify     · delivery ledger (replay = 409)      · launch template, IMDSv2 │
            │  · allowlists      · job_id = H(inst,repo,pr,base,head)  · shutdown -h +TTL,       │
            │  · closed → cancel · newer head → superseded               shutdown = terminate    │
            │                                                                                   │
            │ one-shot VM: review-bot-ondemand ─▶ claim ─▶ heartbeat ─▶ review (private SQLite)│
            │   ─▶ is_current() gate before publish ─▶ record_trace ─▶ complete ─▶ terminate     │
            │                                                                                   │
            │ EventBridge 5 min ─▶ recovery: lost lease / stuck launch → requeue+relaunch       │
            │                           terminal/missing/detached managed VMs → terminate       │
            └───────────────────────────────────────────────────────────────────────────────────┘
The VM entrypoint reads its own identity using IMDSv2, verifies the fetched PR head still matches the claimed
job's head SHA, and executes one WorkerLoop run. Its GitHub App
credentials and model keys are supplied by the operator's AMI/secret bootstrap.
```

## Contract (`review_bot.application.ondemand`)

| Concern | Rule |
|---|---|
| Duplicate delivery | `record_delivery` insert-if-absent. Same digest → `duplicate_delivery`, different digest → `replay_mismatch` (HTTP 409). |
| Duplicate job | `job_id` hashes installation, repository, PR, base and head. `put_new` is insert-if-absent, so at most one VM per head is started. |
| PR update | A new head/base marks active jobs of the PR `superseded`; their `is_current()` turns false before any publication, then their VM is terminated. |
| PR closed | Active jobs become `cancelled` and their VMs are terminated. A termination error is retained without reverting the state; recovery retries it. |
| Lease | `claim` sets `running` with an owner and expiry. Only the owner may `heartbeat`, `record_trace`, or `complete`. A different VM can take over only after the lease expires. |
| Durability | `complete` is rejected until the metadata-only trace is stored, so the trace outlives the VM. |
| A/B state | `mode_results` are kept per mode. All `completed` → `succeeded`; only failures → `failed`; otherwise `partial`. |
| VM loss | `recover()` requeues an expired `running` job or a `launching` job past its timeout, terminates the detached VM, and dead-letters the job after `max_attempts`. It also terminates managed instances whose job is missing or terminal, and instances no longer referenced by an active job. |
| Failed claim | `review-bot ondemand-run --job-id <id>` terminates its own instance when it cannot claim; it never releases a different job's recorded VM. |
| Launch idempotency | `launch_token = job_id[:40]-version` is used as the EC2 `ClientToken`. |
| Concurrency | Every transition is a version-conditional write. No shared SQLite. A VM may keep a private SQLite for its own run. |

GCP fits behind the same ports: Firestore/Spanner conditional writes for `JobStorePort`, and Compute Engine
instances with `instance_termination_action=DELETE` plus `max_run_duration` for `VmLauncherPort`. No GCP adapter is
included.

## Security boundaries

- The Lambda URL is `AuthType: NONE` because GitHub cannot sign SigV4. Every request needs a valid
  `X-Hub-Signature-256`, has a 1 MiB limit, and must pass the installation/repository allowlists.
- Stored data is metadata only (`ReviewJob`, trace counts, model IDs, usage). Raw payloads, prompts, responses and
  source code are not stored.
- IAM: intake/recovery may run instances only from the given launch template, pass only the worker role, describe
  instances for orphan reconciliation, and terminate only instances tagged `review-bot:managed=true`. The worker
  policy covers only the job table, its own managed-instance termination, the GitHub App and OpenRouter secrets,
  session-bucket `sessions/` objects, and its CloudWatch log group.
- Terraform creates secret containers only. It does not create secret versions or store values in state. The operator
  writes the webhook secret, GitHub App JSON (`app_id` and `private_key`), and OpenRouter key with
  `aws secretsmanager put-secret-value`.
- The VM receives only the job ID in user data. GitHub App credentials and models belong in the operator-owned AMI or
  secret store. OpenRouter is a supported production model profile; its API key is obtained from the operator's
  secret store and is not placed in user data.

## Terraform deployment

Prerequisites: Terraform 1.6 or newer, Docker, Packer with the Amazon plugin, an AWS account/region, an existing VPC
and subnet with outbound HTTPS access, an operator-built Ubuntu 24.04 AMI, and a GitHub App with the required
repository permissions. The Terraform module does not create a VPC or subnet. Its worker security group has no
ingress and permits only outbound TCP 443. The launch template requires IMDSv2, a hop limit of one, encrypted gp3
root storage, and terminate-on-shutdown behavior.
The module creates the Lambda and worker CloudWatch log groups by default, with 30-day retention; set
`create_log_groups = false` only when testing against an emulator without the Logs service. The LocalStack example
uses that opt-out because the reused LocalStack instance has Logs disabled.

1. Build the x86_64 Python 3.11 Lambda package. The script uses the Lambda base image, installs only the application
   wheel plus `boto3` and `pydantic`, and proves `review_bot.runtime.aws_lambda` imports inside that image:

   ```bash
   deploy/scripts/build_lambda_zip.sh
   ```

   This produces `deploy/terraform/dist/lambda.zip` and the original-named application wheel under
   `deploy/terraform/dist/wheel/` for the Packer build.

2. Initialize Packer and build the AMI in the target AWS environment. Limit the temporary SSH rule to the operator's
   source IP. The Packer source filter is restricted to Canonical owner `099720109477`.

   ```bash
   packer init deploy/packer/review-bot.pkr.hcl
   packer build \
     -var 'subnet_id=subnet-REPLACE' \
     -var 'ssh_source_cidr=YOUR_PUBLIC_IP/32' \
     -var 'repo_source_path=.' \
     -var 'review_bot_wheel_dir=deploy/terraform/dist/wheel/' \
     deploy/packer/review-bot.pkr.hcl
   ```

   The Packer provisioner downloads any missing PMD artifacts from their locked HTTPS Maven Central URLs and verifies
   both downloaded and pre-existing files against `tooling/tools.lock.json` before building the PMD image. The image
   installs the project wheel, JDK 21, the locked PMD 7.3.0 tool bundle, a dedicated rootless Docker daemon for
   PMD/CPD, and the agent/proxy images. The proxy image tag is generated from the proxy Dockerfile and its copied
   source files, so source changes require a correspondingly tagged image. Agent/proxy containers use the separate
   Docker daemon with the runner's
   read-only root, internal network, capability, PID, memory, and no-new-privileges restrictions; neither container
   receives a Docker socket. It enables a systemd oneshot bootstrap. The worker reads its job ID and secret ARNs
   from IMDSv2 user data and instance tags, writes secret files with mode `0600` under `/run/review-bot`, and runs
   `review-bot-ondemand --job-id` as the rootless worker user. The bootstrap always requests immediate shutdown in
   an exit trap; the launcher's independent `MaxVmSeconds` shutdown timer remains a backstop. The OpenRouter key stays
   in its file until the Docker proxy launch and is never placed in the agent container or user data.

3. Copy `deploy/terraform/terraform.tfvars.example` to `deploy/terraform/terraform.tfvars`, then set the AMI ID, VPC
   ID, subnet ID, Lambda ZIP path, and GitHub allowlists. Run Terraform once to create the secret containers and
   infrastructure:

   ```bash
   cp deploy/terraform/terraform.tfvars.example deploy/terraform/terraform.tfvars
   terraform -chdir=deploy/terraform init
   terraform -chdir=deploy/terraform apply
   ```

   Secret values cannot be set before their Terraform-created containers exist. After the first apply, supply values
   directly through Secrets Manager (not Terraform variables or state):

   ```bash
   aws secretsmanager put-secret-value --secret-id "$(terraform -chdir=deploy/terraform output -raw webhook_secret_arn)" --secret-string file://webhook-secret.txt
   aws secretsmanager put-secret-value --secret-id "$(terraform -chdir=deploy/terraform output -raw github_app_secret_arn)" --secret-string file://github-app.json
   aws secretsmanager put-secret-value --secret-id "$(terraform -chdir=deploy/terraform output -raw openrouter_secret_arn)" --secret-string file://openrouter-key.txt
   ```

   `github-app.json` must contain `{"app_id":"…","private_key":"-----BEGIN RSA PRIVATE KEY-----\n…"}`
   (the private key's newlines must be JSON-escaped). Keep these local input files private. Secret values are fetched
   at worker boot and are not included in EC2 user data or Terraform state.

4. Register the output URL as the GitHub App webhook URL, use `application/json`, subscribe to pull-request events,
   and configure the same webhook secret value. The URL is public by design; the Lambda validates GitHub's HMAC
   signature before accepting a request.

`deploy/terraform/examples/localstack/` shows provider endpoint overrides and uses LocalStack's existing default VPC
and subnet. It is for emulation only and must not be used as a production variable file.

## Verification

```bash
uv run pytest tests/unit/test_ondemand_contract.py                    # in-memory contract
uv run --group aws pytest tests/unit/test_aws_ondemand.py             # moto DynamoDB/EC2 emulation, no network
uvx --from "cfn-lint>=1.20,<2" cfn-lint deploy/aws/template.yaml      # template lint
```

Not verified: a real AMI build or boot, a real AWS Terraform apply, or actual GitHub webhook delivery. LocalStack does
not boot EC2 or run the worker bootstrap; Lambda invocation, EC2 `RunInstances`, lifecycle calls, and AWS APIs are
emulated there.
