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
            │ one-shot VM: claim (lease) ─▶ heartbeat ─▶ review (local SQLite, local models)    │
            │   ─▶ is_current() gate before publish ─▶ record_trace ─▶ complete ─▶ terminate     │
            │                                                                                   │
            │ EventBridge 5 min ─▶ recovery Lambda: lost lease / stuck launch → requeue+relaunch │
            │                                        attempts ≥ max → dead_letter               │
            └───────────────────────────────────────────────────────────────────────────────────┘
local model path (tested): VM-local Ollama on 127.0.0.1 → tev1:0.8b-128k (decision) + granite4.2:128k (review)
```

## Contract (`review_bot.application.ondemand`)

| Concern | Rule |
|---|---|
| Duplicate delivery | `record_delivery` insert-if-absent. Same digest → `duplicate_delivery`, different digest → `replay_mismatch` (HTTP 409). |
| Duplicate job | `job_id` hashes installation, repository, PR, base and head. `put_new` is insert-if-absent, so at most one VM per head is started. |
| PR update | A new head/base marks active jobs of the PR `superseded`; their `is_current()` turns false before any publication. |
| PR closed | Active jobs become `cancelled`. |
| Lease | `claim` sets `running` with an owner and expiry. Only the owner may `heartbeat`, `record_trace`, or `complete`. A different VM can take over only after the lease expires. |
| Durability | `complete` is rejected until the metadata-only trace is stored, so the trace outlives the VM. |
| A/B state | `mode_results` are kept per mode. All `completed` → `succeeded`; only failures → `failed`; otherwise `partial`. |
| VM loss | `recover()` requeues an expired `running` job or a `launching` job past its timeout, terminates the old VM, and dead-letters the job after `max_attempts`. |
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
- IAM: intake may run instances only from the given launch template, pass only the worker role, and terminate only
  instances tagged `review-bot:managed=true`. The worker policy covers only the job table plus self-termination.
- The VM receives only the job ID in user data. GitHub App credentials and models belong in the operator-owned AMI or
  secret store. External model APIs are not used.

## Verification

```bash
uv run pytest tests/unit/test_ondemand_contract.py                    # in-memory contract
uv run --group aws pytest tests/unit/test_aws_ondemand.py             # moto DynamoDB/EC2 emulation, no network
uvx --from "cfn-lint>=1.20,<2" cfn-lint deploy/aws/template.yaml      # template lint
```

Not verified: a real AWS deployment, Lambda cold start/packaging, EC2 boot time, the AMI contents, GitHub
redelivery timing, and integrating `runtime.ondemand_vm.run_job` with `WorkerLoop.process_run` on a real VM.
