# On-demand review: Lambda intake → one-shot Linux VM (AWS reference)

This is a **reference design with code and local emulation tests**. It has not been deployed: no AWS account,
GitHub App webhook, or real EC2 instance was used to verify it. `deploy/aws/template.yaml` passes `cfn-lint`.

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
  policy covers only the job table plus self-termination.
- The VM receives only the job ID in user data. GitHub App credentials and models belong in the operator-owned AMI or
  secret store. OpenRouter is a supported production model profile; its API key is obtained from the operator's
  secret store and is not placed in user data.

## Verification

```bash
uv run pytest tests/unit/test_ondemand_contract.py                    # in-memory contract
uv run --group aws pytest tests/unit/test_aws_ondemand.py             # moto DynamoDB/EC2 emulation, no network
uvx --from "cfn-lint>=1.20,<2" cfn-lint deploy/aws/template.yaml      # template lint
```

Not verified: a real AWS deployment, Lambda cold start/packaging, EC2 boot time, the AMI contents, GitHub
redelivery timing, and real GitHub/model calls from an EC2 worker. The AMI must install the `review-bot-ondemand`
console script, operator configuration (default `/etc/review-bot/operator.yaml`), and private state directory
(`/var/lib/review-bot` by default).
