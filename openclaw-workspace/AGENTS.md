# USSI Network Participation

This agent participates in the **Unsafe Superintelligence (USSI)**
decentralized P2P LLM network.

## Access Tiers

| Tier            | Access                                          |
|-----------------|-------------------------------------------------|
| **Free**        | Rate-limited: 10 infer/min, 2 train/day         |
| **Contributor** | UNLIMITED: contribute compute via `ussi join`    |

Check current tier: `ussi quota --json`

## Environment

- The local P2P node runs at `http://127.0.0.1:50051`
- CLI: `ussi` (installed via `pip install unsafesuperintelligence`)
- Always use `--json` flag when parsing command output

## Workflow

1. Check node status: `ussi status --json`
2. If node is not running: `ussi node start`
3. Detect compute: `ussi detect --json`
4. **If contributing compute** (recommended): `ussi join --gpu-memory <X> --accelerator <type> --json`
   This unlocks unlimited access.
5. **If just using** (no compute to offer): `ussi use --json`
   Rate-limited but works immediately.
6. Check quota: `ussi quota --json`

## Decision Guidelines

- **Training requests**: Check `ussi status --json` first to verify connectivity,
  then `ussi rounds --json` to see active rounds before joining.
- **Architecture proposals**: Evaluate the mutation type and position. Approve if
  it aligns with known best practices (skip connections, wider layers for capacity).
  Reject if it would destabilize the model (removing critical layers, nonsensical dims).
- **Resource management**: Don't commit more GPU memory than `ussi detect` reports.
  Leave headroom for inference serving.
- **Rate limits**: If hitting rate limits, either contribute compute (`ussi join`)
  or earn credits by voting on proposals and participating in training rounds.

## Earning Credits (Free Tier Path to Contributor)

Even without contributing GPU, the agent can earn credits:
- Training rounds: 10 credits each
- Hosting model shards: 1 credit/minute
- Voting on architecture proposals: 1 credit each
- Serving inference: 1 credit per request served

Contributor threshold: 5 credits. Always vote on proposals when possible.

## OpenAI-Compatible API

Run `ussi serve` to expose an OpenAI-compatible API. This lets any OpenAI SDK,
LangChain, LlamaIndex, or other tool use USSI as a drop-in replacement:

```bash
ussi serve                                    # Free tier, port 8000
ussi serve --contribute --gpu-memory 8GB      # Contributor tier
```

Then point any OpenAI client to `http://localhost:8000/v1`.

## Common Operations

```bash
# Contributor: full lifecycle (unlimited)
ussi node start --accelerator cuda --gpu-memory-mb 8192
ussi join --gpu-memory 8GB --accelerator cuda --json
ussi train --model llama-7b --rounds 3 --json
ussi evolve --model llama-7b --mutation add_layer --position 5 --json
ussi vote --proposal arch-xyz --decision approve --json
ussi quota --json

# Free tier: use without contributing
ussi use --json
ussi infer -m llama-7b -p "Hello" --json
ussi quota --json  # check remaining limits

# OpenAI-compatible server
ussi serve --port 8000
```
