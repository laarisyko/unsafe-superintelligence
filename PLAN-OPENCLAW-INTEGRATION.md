# SSSI (Super Safe Super Intelligence) -- OpenClaw Agent Integration Plan

## Context

[OpenClaw](https://github.com/openclaw/openclaw) is an open-source autonomous AI
agent platform (145K+ GitHub stars). Agents extend their capabilities through three
mechanisms:

| Mechanism   | Effort   | What it does                                              |
|-------------|----------|-----------------------------------------------------------|
| **Skill**   | Lowest   | A `SKILL.md` file teaches the agent to use CLI/APIs       |
| **Plugin**  | Medium   | TypeScript module registers native tools in the Gateway    |
| **Webhook** | Low      | External events push notifications to the agent            |

Our goal: **any OpenClaw agent should be able to join the decentralized LLM network
with zero code changes to OpenClaw itself.**

---

## 1. OpenClaw Skill Package (Primary Integration)

The Skill is the lowest-friction path. An OpenClaw user runs
`clawhub install openclaw-network` (or drops the folder into
`~/.openclaw/skills/`) and their agent immediately understands how to join the
P2P network, contribute compute, train models, run inference, and participate in
architecture evolution.

### Files to create

```
openclaw-skill/
├── SKILL.md                    # Skill definition (required)
├── scripts/
│   ├── setup.sh                # Install dependencies (pip, docker)
│   ├── node-start.sh           # Start a local P2P node (Docker or binary)
│   ├── node-stop.sh            # Stop the local node
│   ├── join.sh                 # Join network + advertise capacity
│   ├── status.sh               # Show node/network/peer status (JSON)
│   ├── infer.sh                # Run inference on a model
│   ├── train.sh                # Join a training round
│   ├── evolve.sh               # Propose an architecture mutation
│   └── vote.sh                 # Vote on an architecture proposal
└── references/
    ├── protocol-summary.md     # Condensed protocol docs for the agent
    └── cli-reference.md        # Full CLI reference
```

### SKILL.md content (draft)

```yaml
---
name: openclaw-network
description: >
  Join a decentralized peer-to-peer LLM training and inference network.
  Use when the user wants to contribute GPU/CPU compute to collective AI
  training, run inference on community models, or participate in
  collaborative architecture evolution. No central server required.
tools: Bash, Read, Write
requires:
  bins:
    - docker
    - python3
    - pip
---

# OpenClaw Decentralized LLM Network

## Setup

Run `bash scripts/setup.sh` to install the openclaw-network SDK and pull the
node Docker image. This only needs to happen once.

## Starting a Node

Run `bash scripts/node-start.sh [--bootstrap <multiaddr>]` to start a local
P2P node in a Docker container. The node listens on port 9000 (P2P) and
50051 (API). If no bootstrap address is given, the node starts in standalone
mode and waits for peers to connect.

## Joining the Network

Run `bash scripts/join.sh --gpu-memory <size> --accelerator <type>` to
advertise this machine's compute capacity. Example:
  bash scripts/join.sh --gpu-memory 8GB --accelerator cuda

## Checking Status

Run `bash scripts/status.sh` to see JSON output of node health, connected
peers, active training rounds, and current shard assignments.

## Running Inference

Run `bash scripts/infer.sh --model <model-id> --prompt "<text>"` to run
inference on a model hosted by the network.

## Training

Run `bash scripts/train.sh --model <model-id> --rounds <n>` to participate
in decentralized training rounds.

## Architecture Evolution

Run `bash scripts/evolve.sh --model <model-id> --mutation <type> --position <n>`
to propose a mutation. Types: add_layer, remove_layer, widen_layer,
swap_activation, insert_skip.

Run `bash scripts/vote.sh --proposal <id> --decision <approve|reject>`
to vote on a peer's architecture proposal.
```

### What this requires from our codebase

The skill scripts shell out to our `openclaw` CLI. The CLI needs to be
complete and produce machine-readable (JSON) output. Changes needed:

**a) CLI enhancements (`agent-sdk/openclaw_sdk/cli.py`)**

Add these missing commands:
- `openclaw node start` -- Start a P2P node (wraps Docker or binary)
- `openclaw node stop` -- Stop the local node
- `openclaw evolve` -- Propose an architecture mutation
- `openclaw vote` -- Vote on an architecture proposal
- `openclaw models` -- List available models on the network
- `openclaw rounds` -- List active training rounds

Add a global `--json` flag so all commands output structured JSON (critical
for agent parsing).

Add a global `--quiet` flag to suppress human-readable decoration.

**b) Node management (`agent-sdk/openclaw_sdk/node_manager.py` -- new file)**

A helper that manages the local P2P node lifecycle:
- `start(port, api_port, bootstrap, docker=True)` -- Start node via Docker
  or direct binary
- `stop()` -- Stop the node container/process
- `is_running()` -- Check if a node is alive
- `logs(tail=50)` -- Fetch recent node logs

This lets the OpenClaw agent autonomously start/stop nodes without the user
manually running Docker commands.

**c) Auto-detection of compute resources**

Add `openclaw detect` command that auto-detects:
- Available GPU memory (via `nvidia-smi` or `torch.cuda`)
- CPU cores and RAM
- Network bandwidth estimate

This lets the agent self-configure without asking the user for specs.

---

## 2. OpenClaw Plugin (Deeper Integration)

For users who want tighter integration, we provide a TypeScript plugin that
registers native tools in the OpenClaw Gateway. The agent can then use
`openclaw_network_join`, `openclaw_network_train`, etc. without needing bash.

### Files to create

```
openclaw-plugin/
├── package.json
├── openclaw.plugin.json
└── src/
    └── index.ts                # register(api) entry point
```

### Registered tools

| Tool name                    | Description                                    |
|------------------------------|------------------------------------------------|
| `openclaw_network_status`    | Check node health and peer count               |
| `openclaw_network_join`      | Join the P2P network with compute capacity     |
| `openclaw_network_infer`     | Run inference on a decentralized model          |
| `openclaw_network_train`     | Join a training round                          |
| `openclaw_network_evolve`    | Propose an architecture mutation               |
| `openclaw_network_vote`      | Vote on an architecture proposal               |
| `openclaw_network_models`    | List available models                          |
| `openclaw_network_peers`     | List connected peers                           |

Each tool calls our node's HTTP API (`http://127.0.0.1:50051/...`) directly,
avoiding the need for Python/pip on the host. The plugin reads the node URL
from OpenClaw's config:

```json5
// openclaw.json
{
  "openclaw_network": {
    "node_url": "http://127.0.0.1:50051",
    "auto_start_node": true,
    "default_accelerator": "cuda"
  }
}
```

The plugin also provides a Zod config schema that gets merged into
OpenClaw's validated config.

---

## 3. Webhook Bridge (Event-Driven Participation)

Our P2P network generates events that should proactively notify OpenClaw
agents. We add a webhook dispatcher to the Rust node that POSTs to
configured endpoints when events occur.

### Events to dispatch

| Event                        | When                                           |
|------------------------------|-------------------------------------------------|
| `training.round.proposed`    | A peer proposes a new training round           |
| `training.round.started`     | Enough peers joined; round begins              |
| `training.round.completed`   | Round finished, weights updated                |
| `architecture.proposal.new`  | A peer proposes an architecture mutation        |
| `architecture.vote.needed`   | A proposal needs more votes to reach quorum    |
| `architecture.accepted`      | A mutation was accepted by the network         |
| `peer.joined`                | A new peer joined the network                  |
| `peer.left`                  | A peer left or timed out                       |

### Implementation

**a) Rust node changes (`node/src/api/webhooks.rs` -- new file)**

- Config: list of webhook URLs + optional auth tokens
- Dispatcher: background task that POSTs JSON payloads to registered URLs
- Retry: 3 attempts with exponential backoff
- Config via CLI flags: `--webhook-url`, `--webhook-token`

**b) OpenClaw webhook config**

Users configure an OpenClaw webhook to receive these events:

```json5
// openclaw.json
{
  "hooks": {
    "enabled": true,
    "mappings": [
      {
        "match": { "path": "/hooks/openclaw-network" },
        "action": "agent",
        "template": {
          "message": "OpenClaw Network: {{body.event}} - {{body.summary}}",
          "agentId": "main"
        }
      }
    ]
  }
}
```

This means the OpenClaw agent gets proactively notified when training rounds
start, votes are needed, etc. -- enabling truly autonomous participation.

---

## 4. Node HTTP API Enhancements

The current node API (`node/src/api/grpc_server.rs`) needs new endpoints
to support the skill scripts and plugin:

| Method | Path                     | Purpose                                |
|--------|--------------------------|----------------------------------------|
| GET    | `/health`                | Already exists                         |
| GET    | `/peers`                 | Already exists                         |
| GET    | `/shards`                | Already exists                         |
| POST   | `/publish`               | Already exists                         |
| POST   | `/infer`                 | Already exists                         |
| GET    | `/models`                | **NEW** - List models on the network   |
| GET    | `/rounds`                | **NEW** - List active training rounds  |
| GET    | `/proposals`             | **NEW** - List architecture proposals  |
| POST   | `/train/join`            | **NEW** - Join a training round        |
| POST   | `/evolve/propose`        | **NEW** - Propose architecture mutation|
| POST   | `/evolve/vote`           | **NEW** - Vote on a proposal           |
| GET    | `/detect`                | **NEW** - Auto-detect compute capacity |
| POST   | `/webhooks/register`     | **NEW** - Register a webhook URL       |
| DELETE | `/webhooks/:id`          | **NEW** - Unregister a webhook         |

---

## 5. Workspace Integration Files

Provide ready-made files that OpenClaw users can drop into their workspace:

### `AGENTS.md` snippet

```markdown
## OpenClaw Network Participation

This agent participates in the OpenClaw decentralized LLM network.

- The local P2P node runs at http://127.0.0.1:50051
- Use the `openclaw` CLI for network operations
- Prefer JSON output (`--json`) when parsing responses
- When asked about training, check `openclaw status --json` first
- When a training round is proposed, evaluate if resources allow joining
- When an architecture vote is needed, evaluate the proposal before voting
```

### `TOOLS.md` snippet

```markdown
## openclaw CLI

The `openclaw` command manages participation in the decentralized LLM network.

Key commands:
- `openclaw status --json` - Check health and peers
- `openclaw join --gpu-memory 8GB --accelerator cuda` - Contribute compute
- `openclaw infer -m llama-7b -p "prompt" --json` - Run inference
- `openclaw train -m llama-7b -r 5` - Join training
- `openclaw evolve -m llama-7b --mutation add_layer --position 3` - Propose mutation
- `openclaw vote --proposal <id> --decision approve` - Vote
```

---

## 6. One-Command Setup

An OpenClaw agent (or user) should be able to get running with a single command.

### Option A: pip install + Docker (recommended)

```bash
pip install openclaw-network && openclaw node start
```

This installs the SDK/CLI and starts a P2P node in Docker.

### Option B: Full Docker

```bash
docker run -d --name openclaw-node -p 9000:9000 -p 50051:50051 \
  ghcr.io/openclaw/network-node:latest \
  --bootstrap /ip4/203.0.113.1/tcp/9000/p2p/QmBootstrap...
```

### Option C: Skill auto-setup

The skill's `setup.sh` handles everything:
1. Checks if Python 3.10+ is available
2. Installs `openclaw-network` via pip
3. Pulls the Docker image
4. Starts the node
5. Verifies connectivity

---

## 7. Implementation Order

| Step | What                                        | Priority |
|------|---------------------------------------------|----------|
| 1    | CLI enhancements (--json, new commands)     | Critical |
| 2    | Node manager (start/stop/detect)            | Critical |
| 3    | New HTTP API endpoints on Rust node         | Critical |
| 4    | OpenClaw Skill package (SKILL.md + scripts) | Critical |
| 5    | Workspace files (AGENTS.md, TOOLS.md)       | High     |
| 6    | Webhook dispatcher in Rust node             | High     |
| 7    | OpenClaw Plugin (TypeScript)                | Medium   |
| 8    | ClawHub publishing                          | Medium   |
| 9    | Setup script / one-liner install            | Medium   |

Steps 1-4 are the minimum viable integration. An OpenClaw agent with just the
skill installed can autonomously join the network, contribute compute, train
models, and participate in architecture evolution.

---

## 8. Testing Plan

- **Unit tests**: CLI JSON output parsing, node manager lifecycle
- **Integration test**: Simulated OpenClaw skill invocation (bash scripts
  calling CLI, verifying JSON output)
- **End-to-end**: OpenClaw agent with skill installed, given the prompt
  "Join the decentralized training network and contribute my GPU", verifying
  it successfully calls the right scripts and joins
- **Webhook test**: Node event → webhook POST → verify payload format

---

## 9. Package Naming

To avoid confusion with the OpenClaw agent platform itself:

| Component           | Package name              | Why                              |
|---------------------|---------------------------|----------------------------------|
| Python SDK/CLI      | `openclaw-network`        | Distinguishes from `openclaw`    |
| CLI binary          | `openclaw-net`            | Short, no conflict               |
| Docker image        | `openclaw/network-node`   | Clear scope                      |
| OpenClaw Skill      | `openclaw-network`        | ClawHub skill name               |
| OpenClaw Plugin     | `@openclaw/network`       | npm scoped package               |

The CLI command stays `openclaw` for now since it's contextually clear when
used within the skill scripts. If conflicts arise, rename to `ocnet`.
