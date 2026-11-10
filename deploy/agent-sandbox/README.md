# pi-agent sandbox images

Build the pinned images from the repository root:

```sh
docker build -f deploy/agent-sandbox/Dockerfile.agent -t review-bot-pi-agent:0.73.1 .
docker build -f deploy/agent-sandbox/Dockerfile.proxy -t review-bot-agent-proxy:1 deploy/agent-sandbox
```

The worker supplies the selected OpenRouter or OpenCode Go key only to the
proxy container. The agent receives a dummy provider key and can reach only the
per-run internal network.
