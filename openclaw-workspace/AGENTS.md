# SSSI Network Participation

This agent participates in the **Super Safe Super Intelligence (SSSI)**
decentralized P2P LLM network.

## Access Tiers

| Tier            | Access                                          |
|-----------------|-------------------------------------------------|
| **Free**        | Rate-limited: 10 infer/min, 2 train/day         |
| **Contributor** | UNLIMITED: contribute compute via `sssi join`    |

Check current tier: `sssi quota --json`

## Environment

- The local P2P node runs at `http://127.0.0.1:50051`
- CLI: `sssi` (installed via `pip install supersafesuperintelligence`)
- Always use `--json` flag when parsing command output

## Workflow

1. Check node status: `sssi status --json`
2. If node is not running: `sssi node start`
3. Detect compute: `sssi detect --json`
4. **If contributing compute** (recommended): `sssi join --gpu-memory <X> --accelerator <type> --json`
   This unlocks unlimited access.
5. **If just using** (no compute to offer): `sssi use --json`
   Rate-limited but works immediately.
6. Check quota: `sssi quota --json`

## Decision Guidelines

- **Training requests**: Check `sssi status --json` first to verify connectivity,
  then `sssi rounds --json` to see active rounds before joining.
- **Architecture proposals**: Evaluate the mutation type and position. Approve if
  it aligns with known best practices (skip connections, wider layers for capacity).
  Reject if it would destabilize the model (removing critical layers, nonsensical dims).
- **Resource management**: Don't commit more GPU memory than `sssi detect` reports.
  Leave headroom for inference serving.
- **Rate limits**: If hitting rate limits, either contribute compute (`sssi join`)
  or earn credits by voting on proposals and participating in training rounds.

## Earning Credits (Free Tier Path to Contributor)

Even without contributing GPU, the agent can earn credits:
- Training rounds: 10 credits each
- Hosting model shards: 1 credit/minute
- Voting on architecture proposals: 1 credit each
- Serving inference: 1 credit per request served

Contributor threshold: 5 credits. Always vote on proposals when possible.

## OpenAI-Compatible API

Run `sssi serve` to expose an OpenAI-compatible API. This lets any OpenAI SDK,
LangChain, LlamaIndex, or other tool use SSSI as a drop-in replacement:

```bash
sssi serve                                    # Free tier, port 8000
sssi serve --contribute --gpu-memory 8GB      # Contributor tier
```

Then point any OpenAI client to `http://localhost:8000/v1`.

## Common Operations

```bash
# Contributor: full lifecycle (unlimited)
sssi node start --accelerator cuda --gpu-memory-mb 8192
sssi join --gpu-memory 8GB --accelerator cuda --json
sssi train --model llama-7b --rounds 3 --json
sssi evolve --model llama-7b --mutation add_layer --position 5 --json
sssi vote --proposal arch-xyz --decision approve --json
sssi quota --json

# Free tier: use without contributing
sssi use --json
sssi infer -m llama-7b -p "Hello" --json
sssi quota --json  # check remaining limits

# OpenAI-compatible server
sssi serve --port 8000
```
