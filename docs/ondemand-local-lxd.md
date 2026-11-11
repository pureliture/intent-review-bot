# Local on-demand review with LXD VMs

The local LXD backend runs the same on-demand coordinator, durable job-store contract, and worker bootstrap as the EC2 backend. Only the VM launcher changes: a host process launches ephemeral LXD virtual machines, while the worker uses the LXD guest launcher to terminate itself. Jobs still use `DynamoDbJobStore`; point boto3 at a local DynamoDB- and Secrets Manager-compatible service with `AWS_ENDPOINT_URL`.

The backend was exercised with LXD 5.0.11 and Ubuntu 24.04 VMs. The GitHub App verification/publication path has not been verified locally, and this guide does not claim an end-to-end review against GitHub.

## Host prerequisites

- Linux with LXD 5.0 or later, a configured LXD daemon, and KVM support.
- Initialize LXD and add the operator to the LXD group. Re-login so the group change takes effect:

  ```bash
  sudo lxd init
  sudo usermod -aG lxd "$USER"
  ```

- When Docker is also running with a `FORWARD` policy of `DROP`, allow LXD bridge traffic through Docker's user chain:

  ```bash
  sudo iptables -I DOCKER-USER 1 -i lxdbr0 -j ACCEPT
  sudo iptables -I DOCKER-USER 2 -o lxdbr0 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  ```

  These rules enable guest egress; make them persistent using the host's normal firewall-management mechanism if needed.
- The VM needs outbound HTTPS access to GitHub, the model provider, and the local AWS-compatible endpoint. A host-only LocalStack listener can be forwarded to the LXD bridge address without changing LocalStack's loopback binding:

  ```bash
  socat TCP-LISTEN:4566,bind=10.31.133.1,reuseaddr,fork TCP:127.0.0.1:4566
  ```

  Configure the worker endpoint as `http://10.31.133.1:4566` (replace the bridge address and port as appropriate). Confirm the listener is reachable from a VM before running a review. Do not expose the compatibility endpoint to untrusted networks.

## Worker profile and image

The Packer LXD builder uses a named profile. Create it from the default profile so it retains the bridge NIC and root device, then size it for the worker:

```bash
sudo lxc profile copy default review-bot-worker
sudo lxc profile set review-bot-worker limits.cpu 4
sudo lxc profile set review-bot-worker limits.memory 8GiB
sudo lxc profile device set review-bot-worker root size 40GiB
```

The LXD plugin's `launch_config` sets the build VM to 4 CPUs and 8 GiB; LXD v1.0.3 does not expose a root-disk-size launch option, so the profile supplies a root disk of at least 40 GiB. The runtime launcher should use both `default` and `review-bot-worker` profiles, in that order.

Build the wheel bundle and the local LXD image:

```bash
deploy/scripts/build_lambda_zip.sh
packer init deploy/packer/review-bot.pkr.hcl
packer build \
  -only=lxd.review_bot \
  -var 'repo_source_path=.' \
  -var 'review_bot_wheel_dir=deploy/terraform/dist/wheel/' \
  deploy/packer/review-bot.pkr.hcl
```

The default source image is `ubuntu:24.04`, output image name is `review-bot-worker`, and profile is `review-bot-worker`. Override `lxd_source_image`, `lxd_output_image`, or `lxd_profile` when appropriate. This build provisions the same worker software as the Amazon source; it does not require an EC2 subnet.

## Host configuration

Select the host LXD launcher in the process that runs the on-demand coordinator:

```bash
export REVIEW_BOT_VM_PROVIDER=lxd
export REVIEW_BOT_LXD_IMAGE=review-bot-worker
export REVIEW_BOT_LXD_PROFILES=default,review-bot-worker
export REVIEW_BOT_INSTANCE_TAGS='{"user.review-bot.job-table":"review-jobs","user.review-bot.github-app-secret-arn":"arn:aws:secretsmanager:local:000000000000:secret:github-app","user.review-bot.openrouter-secret-arn":"arn:aws:secretsmanager:local:000000000000:secret:model","user.review-bot.region":"us-east-1","user.review-bot.aws-endpoint-url":"http://10.31.133.1:4566"}'
export AWS_ENDPOINT_URL=http://10.31.133.1:4566
export AWS_REGION=us-east-1
export REVIEW_BOT_JOB_TABLE=review-jobs
```

`REVIEW_BOT_LXD_PROFILES` defaults to `default`; `REVIEW_BOT_LXD_PROJECT` is optional. For LXD, the JSON keys must use the `user.review-bot.*` config namespace (unlike EC2 tags with `review-bot:` names). `REVIEW_BOT_INSTANCE_TAGS` must be a JSON object with string keys and values. The host coordinator stores and recovers jobs through the configured DynamoDB-compatible endpoint. The endpoint is also passed to the guest in LXD metadata so its Secrets Manager client can retrieve credentials.

The worker bootstrap reads these `user.review-bot.*` keys from LXD devlxd metadata:

- Required: `job-id`, `vm-id`, `max-lifetime-seconds`, `region`, `github-app-secret-arn`, `openrouter-secret-arn`, and `job-table`.
- Optional: `aws-endpoint-url`, `agent-provider`, `agent-model-id`, `agent-context-window-tokens`, and `agent-profile` (`fast` by default; `deep` for broader review).
- The host launcher supplies `managed`, `job-id`, `vm-id`, and `max-lifetime-seconds`; configure the remaining values through `REVIEW_BOT_INSTANCE_TAGS`.

To reconcile lost jobs from the local host, run `review-bot ondemand-recover` with the same DynamoDB endpoint and table configuration as the coordinator.

## Execution diagnostics

Each review run writes `execution-metrics.json` beside its manifest and includes it in the on-demand job's `trace.metrics`. It contains worker stage events, per-mode wall spans, and agent session setup/invocation timings. Invocation stages distinguish generation, output repair, verification, revision, and reverification. A model call belongs to an invocation when its number is greater than `call_start` and at most `call_end` within that session.

The numeric proxy ledger includes attempted calls, HTTP status, duration, response-header latency (`first_byte_ms`), input/output tokens, optional reasoning tokens, and finish reason. An `in_flight` row is merged with its completion row by call number; an interrupted attempt with no usage remains counted with unknown usage. Summed model duration includes overlapping modes and is not run wall time. Prompts, responses, API keys, and raw agent transcripts are excluded from these metrics.

When the review deadline expires, completed parallel mode receipts and publication receipts are persisted before the run is marked failed with `task_deadline_exceeded`. A published mode can therefore remain completed inside a failed overall run. This persistence still requires the current lease and fencing ownership. Normal completion/deadline cleanup preserves the metrics in the job trace even after the ephemeral VM is deleted; an abrupt VM loss before finalization can still leave incomplete diagnostics.

## Security

LXD guest root can read `user.*` instance config through `/dev/lxd/sock`. Never put secret values, API keys, private keys, or other credentials in LXD config or `REVIEW_BOT_INSTANCE_TAGS`. Put secret values in Secrets Manager and pass only their ARNs as metadata; use a trusted Secrets Manager-compatible endpoint reachable from the bridge. The endpoint and LocalStack should remain restricted to trusted host/guest traffic.
