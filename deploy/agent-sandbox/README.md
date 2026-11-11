# pi-agent sandbox images

Build the pinned images from the repository root:

```sh
docker build -f deploy/agent-sandbox/Dockerfile.agent -t review-bot-pi-agent:0.73.1 .
docker build -f deploy/agent-sandbox/Dockerfile.proxy -t "$(review-bot agent-proxy-image)" deploy/agent-sandbox
```

The worker supplies the selected OpenRouter or OpenCode Go key only to the
proxy container. The agent receives a dummy provider key and can reach only the
per-run internal network.

## Review profiles

The agent engine defaults to `agent_engine.profile: deep` to prioritize review
coverage and quality. Deep reviews retain generation, independent verification,
revision and reverification, with medium thinking and no three-item fast cap.
Omitting the profile selects deep, including in generated worker configuration.

An explicitly selected `agent_engine.profile: fast` review
prioritizes changed code and direct dependencies, selects at most three findings
or three intent assessments/questions per mode, and independently verifies the
summary and selected items once. Unverified claims remain a partial review;
there is no revision/reverification loop. All required intent perspective IDs
remain in the result, including empty perspectives.

| Default budget | fast | deep |
| --- | ---: | ---: |
| Mode wall time (generation + verification + format repair) | 240 s | Remaining task time |
| Model calls per agent session, including format repair | 16 | 40 |
| Output tokens per model call | 4,096 | Provider/pi ceiling |
| Cumulative output tokens per agent session | 12,000 | No additional cap |
| Thinking level | off | medium |
| Semantic revision rounds | 0 | 1 |

Generation and verification use separate sessions. With A/B both selected, the
fast defaults therefore allow at most 64 model calls and 48,000 output tokens in
total. The proxy reserves each call's output allowance before forwarding it,
refunds unused output only when usage is known, and retains the reservation if
usage is missing. These budgets rely on the provider honoring its requested token
ceiling. Each session may make multiple tool/model calls. Each initial output can
receive one format-repair invocation within the same session budget. A/B still
run concurrently by default and share the outer task deadline. Setup/static
analysis/publication time is additional; 240 seconds is not an end-to-end SLA.

Select the bounded first pass explicitly in the trusted operator configuration:

```yaml
review_engine: agent
agent_engine:
  profile: fast
```

`deep` retains the previous generation, independent verification, revision and
reverification policy. Profile defaults can be tuned by trusted operators through
`thinking_level`, `max_mode_seconds`, `max_proxy_calls`, `max_output_tokens`,
`max_total_output_tokens`, and `max_revisions`. Fast always disables semantic
revisions and caps result items at three, even if a revision override is supplied.
Explicit operator values override profile defaults. Deployments with an explicit
`profile: fast` remain fast; select deep or remove that setting to use the default.
Remove any old fast budget overrides when switching to deep if its defaults are
desired. The legacy partition engine is unaffected.

The published comment identifies the profile and bounded scope. Artifacts record
the policy in result identity, and generation cache keys include profile/budget
settings. Switching profiles cannot reuse a result generated under another policy.
Fast workspaces include a bounded change index and diff excerpts (up to eight
paths and 24,000 diff characters), with the full frozen files still available to
bind exact citations. The initial prompt requests batched evidence reads rather
than repository-wide discovery. Truncated or omitted context does not establish
absence of behavior.
When the context bundle fits within 96,000 characters, fast supplies the contracts,
selected skill, change excerpts, PR text and (for the verifier) claims in the
initial prompt. This avoids model calls spent rediscovering the input files.
OpenCode Go MiMo uses `reasoning_effort: none` for fast; that gateway ignored
native `thinking.type: disabled` in a comparison probe. The adapter preserves
reasoning history in multi-turn deep reviews.
Fast review is an opt-in worker configuration; a deep review is not automatically
queued after it. There is no separate PR command for switching profiles in this change.

For on-demand EC2 workers, set Terraform `agent_profile` (`deep` by default).
For LXD workers, set `user.review-bot.agent-profile` in `REVIEW_BOT_INSTANCE_TAGS`.
Bootstrap maps this metadata to `REVIEW_BOT_AGENT_PROFILE` and renders the selected
operator profile. Rebuild the worker image to include these changes; rebuild the
content-addressed proxy image after changing proxy source. No deployment is
performed by editing these settings.
