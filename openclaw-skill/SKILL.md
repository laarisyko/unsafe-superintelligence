---
name: unsafesuperintelligence
description: >
  Join the USSI decentralized peer-to-peer LLM training and inference network.
  Use when the user wants to contribute GPU/CPU compute to collective AI training,
  run inference on community models, participate in collaborative architecture
  evolution, or manage their local P2P node. No central server required.
  Two tiers: anyone can use it for free (rate-limited), contributors get unlimited access.
tools: Bash, Read, Write
requires:
  bins:
    - python3
    - pip
---

# Unsafe Superintelligence (USSI) -- Decentralized LLM Network

## Access Tiers

USSI has two access tiers:

| Tier            | How to get it                          | Limits                                      |
|-----------------|----------------------------------------|---------------------------------------------|
| **Free**        | Just connect (`ussi use`)              | 10 infer/min, 5000 tokens/hr, 2 train/day   |
| **Contributor** | Contribute compute (`ussi join`)       | UNLIMITED everything                         |

**To earn contributor status:** contribute GPU/CPU via `ussi join`, or earn credits
by training, hosting model shards, and voting on architecture proposals.

## First-Time Setup

If the `ussi` CLI is not installed, run:

```bash
pip install unsafesuperintelligence
```

Then start a local P2P node (requires Docker):

```bash
ussi node start --accelerator cpu
```

Or with GPU:

```bash
ussi node start --accelerator cuda --gpu-memory-mb 8192
```

To connect to an existing network, add a bootstrap address:

```bash
ussi node start --bootstrap /ip4/203.0.113.1/tcp/9000/p2p/12D3KooW...
```

## Joining as Contributor (Unlimited Access)

Contribute compute to get unlimited access to inference, training, and evolution:

```bash
ussi join --gpu-memory 8GB --accelerator cuda --json
```

This advertises your compute capacity and immediately unlocks contributor tier.

## Connecting as Free User (Rate-Limited)

If the user just wants to use the network without contributing compute:

```bash
ussi use --json
```

Free tier limits: 10 inference requests/minute, 5000 tokens/hour, 2 training
rounds/day, 3 architecture proposals/day. Voting is always unlimited.

## Check Your Quota

```bash
ussi quota --json
```

Returns your current tier, remaining rate limits, and contribution credits.

## Auto-Detect Resources

```bash
ussi detect --json
```

Returns JSON like `{"accelerator": "cuda", "gpu_memory_mb": 8192, "cpu_cores": 8}`.

## Checking Status

```bash
ussi status --json
```

Returns JSON with `agent_id`, `connected`, `contributing`, `tier`, and `node_health`.

## Listing Peers

```bash
ussi peers --json
```

## Listing Models

```bash
ussi models --json
```

## Running Inference

```bash
ussi infer --model llama-7b --prompt "Your prompt here" --max-tokens 256 --json
```

If rate-limited, the error includes a hint to contribute compute.

## Training

Join or propose decentralized training rounds:

```bash
ussi train --model llama-7b --rounds 5 --lr 0.0001 --batch-size 8 --json
```

Training earns contribution credits (10 credits per round).

## Architecture Evolution

Propose a mutation to a model's architecture:

```bash
ussi evolve --model llama-7b --mutation add_layer --position 3 --dim 256 --json
```

Mutation types: `add_layer`, `remove_layer`, `widen_layer`, `swap_activation`, `insert_skip`.

## Voting on Proposals

Vote on another peer's architecture proposal (always free, earns credits):

```bash
ussi vote --proposal arch-abc123 --decision approve --fitness 0.85 --json
```

Decisions: `approve`, `reject`, `abstain`. Each vote earns 1 contribution credit.

## Listing Training Rounds

```bash
ussi rounds --json
```

## Node Management

```bash
ussi node start                    # Start node (Docker)
ussi node start --no-docker        # Start node (local binary)
ussi node stop                     # Stop node
ussi node logs                     # View node logs
```

## Earning Credits (Path to Contributor Tier)

Even without contributing GPU, you can earn credits to reach contributor tier:
- Training rounds: 10 credits each
- Hosting model shards: 1 credit/minute
- Voting on proposals: 1 credit each
- Serving inference: 1 credit per request

Contributor threshold: 5 credits. Credits decay with a 24-hour half-life.

## JSON Output

All commands support `--json` for machine-readable output. Always use
`--json` when parsing results programmatically.

## Rate Limit Errors

If a rate limit is hit, the CLI returns a JSON error with a `hint` field
explaining how to get unlimited access. Example:

```json
{
  "error": "rate_limit_exceeded",
  "resource": "inference requests",
  "limit": 10,
  "window": "minute",
  "tier": "free",
  "hint": "Contribute compute (ussi join --gpu-memory ...) to unlock unlimited access."
}
```

## OpenAI-Compatible Server (Drop-In Replacement)

Start an OpenAI-compatible API server so any OpenAI client, LangChain,
LlamaIndex, or other tool can use USSI as a drop-in replacement:

```bash
ussi serve --json
```

Or with custom port and contributor tier:

```bash
ussi serve --port 11434 --contribute --gpu-memory 8GB --accelerator cuda
```

Then use any OpenAI-compatible client against `http://localhost:8000/v1`.

Endpoints served:
- `GET  /v1/models` -- List available models
- `POST /v1/chat/completions` -- Chat completions (streaming supported)
- `POST /v1/completions` -- Text completions (streaming supported)
- `GET  /health` -- Health check

## Python SDK

For more control, use the Python API directly:

```python
from ussi import Agent

# Free tier (rate-limited)
agent = Agent(node_api_url="http://127.0.0.1:50051")
result = agent.infer(model="llama-7b", prompt="Hello")
print(agent.quota())  # check remaining limits

# Contributor tier (unlimited)
agent = Agent(bootstrap="/ip4/.../tcp/9000/p2p/12D3KooW...")
agent.contribute(gpu_memory="8GB", accelerator="cuda")
# Now everything is unlimited
result = agent.infer(model="llama-7b", prompt="Hello")
agent.train(model="llama-7b", rounds=5)
agent.leave()
```

### Built-In OpenAI Client (No `openai` Package Needed)

```python
from ussi import OpenAI

client = OpenAI()  # connects to local ussi serve on port 8000
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```
