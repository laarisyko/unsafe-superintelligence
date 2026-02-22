# CLAUDE.md — Project Context for AI Assistants

## Project Overview

**USSI (Unsafe Superintelligence)** / **USSI** — A fully decentralized, peer-to-peer network for training and running large language models. "BitTorrent for AI training: a million volunteers training one model, owned by everyone, controlled by no one."

## Architecture

```
User / Agent SDK (ussi CLI)
      │ HTTP :50051
      ▼
Rust P2P Node (libp2p) ◄──► Other Peers (gossipsub, Kademlia DHT)
      │ HTTP :50052
      ▼
Python ML Engine (PyTorch transformer, training, inference)
```

Three-layer stack: Rust networking node, Python ML engine, Python agent SDK/CLI.

## Directory Structure

| Directory | Purpose | Tech |
|---|---|---|
| `node/` | P2P networking node (libp2p swarm, gossipsub, Kademlia, mDNS, scheduler, API) | Rust, libp2p 0.53, tokio, tonic/prost |
| `engine/` | ML engine — model architecture, training loop, inference, gradient aggregation, architecture evolution | Python, PyTorch >= 2.0, package name `ussi` |
| `agent-sdk/` | User-facing SDK + CLI (`ussi` command), OpenAI-compatible server | Python, httpx, pydantic, package name `unsafesuperintelligence` |
| `proto/` | Protobuf wire format (messages.proto, inference.proto, training.proto) | Protobuf |
| `docker/` | Dockerfile.node (multi-stage) + docker-compose.swarm.yml (5-peer local dev) | Docker |
| `ussi-skill/` | SKILL.md for USSI autonomous agent platform | Markdown |
| `ussi-workspace/` | AGENTS.md + TOOLS.md for drop-in workspace integration | Markdown |
| `tests/` | 13 integration test files + `simulation/swarm_sim.py` | Python/pytest |
| `docs/` | `protocol.md` (protocol spec), `threat_model.md` (security analysis) | Markdown |
| `site/` | Landing page / dashboard HTML | HTML |

## Key Entry Points

- **Rust node:** `node/src/main.rs` → builds SwarmDriver, starts scheduler + API server
- **Python engine CLI:** `engine/ussi_engine/cli.py` → `ussi join|status|generate|dataset|dashboard`
- **Agent SDK CLI:** `agent-sdk/ussi/cli.py` → `ussi join|use|status|infer|train|evolve|vote|serve|node`
- **Bridge (Rust→Python):** `node/src/bridge.rs` (client) ↔ `engine/ussi_engine/bridge.py` (server) on port 50052
- **Node API:** `node/src/api/grpc_server.rs` — HTTP/JSON on port 50051 (`/health`, `/peers`, `/shards`, `/publish`, `/infer`)

## Key Design Decisions

- **Gradient aggregation:** Ring all-reduce (small nets) + hierarchical tree-of-rings (1M+ agents)
- **Byzantine resilience:** Krum + coordinate-wise trimmed mean + Merkle verification of weights
- **Sybil resistance:** Proof-of-work admission controller (`engine/ussi_engine/training/sybil.py`)
- **Consensus:** CRDTs for shard maps (`node/src/consensus/shard_map.rs`), VRF for deterministic work assignment (`node/src/consensus/vrf.rs`)
- **Identity:** Ed25519 keypairs, libp2p Noise protocol
- **Model architecture:** Custom transformer with byte-level tokenizer (vocab 260), configurable via LMConfig
- **Architecture evolution:** Genome-based (`engine/ussi_engine/architecture/genome.py`), mutations + proposal/voting
- **Credits:** Free tier (rate-limited: 10 infer/min, 5000 tokens/hr) vs. contributor tier (unlimited)
- **API compat:** OpenAI-compatible server at `ussi serve` with streaming SSE

## Rust Node Internals (`node/src/`)

- `network/mod.rs` — SwarmDriver: libp2p event loop
- `network/gossip.rs` — Topic definitions: `ussi/heartbeat`, `ussi/training`, `ussi/gradient`, `ussi/checkpoint`, `ussi/architecture`
- `network/discovery.rs` — Kademlia + mDNS
- `network/transport.rs` — TCP + Noise + Yamux
- `consensus/vrf.rs` — Deterministic Fisher-Yates permutation for shard/ring/cluster assignment
- `consensus/shard_map.rs` — CRDT shard map with Lamport timestamps
- `consensus/merkle.rs` — Merkle tree over weight tensors
- `scheduler/mod.rs` — Training round state machine (Proposed → Joining → Computing → Aggregating → Checkpointing → Complete)

## Python Engine Internals (`engine/ussi_engine/`)

- `model/lm.py` — `LanguageModel` (nn.Module), `LMConfig`
- `model/shard.py` — `ModelShard` (layer subset for pipeline parallelism)
- `training/trainer.py` — `LocalTrainer` (forward/backward, gradient accumulation)
- `training/allreduce.py` — `RingAllReduce`
- `training/hierarchical.py` — `HierarchicalAllReduce` (tree-of-rings, configurable depth)
- `training/byzantine.py` — `robust_aggregate()` (Krum, trimmed mean)
- `training/compression.py` — Top-K sparsification + FP16 quantization
- `training/reputation.py` — Per-peer reputation scoring
- `training/sybil.py` — PoW admission controller
- `network.py` — `TrainingNetwork` (main controller for a participating peer)
- `genesis.py` — `GenesisTracker` (milestone tracking for dashboard wow-factor)
- `credits.py` — `CreditLedger` + `InferenceGate`
- `dashboard.py` — Live WebSocket dashboard server

## Current State (as of initial commit)

### What works:
- Complete Rust P2P node with full libp2p stack
- Complete Python ML engine with training, inference, architecture evolution
- Both CLIs (`ussi` and `ussi`)
- OpenAI-compatible server with streaming
- Docker multi-stage build + 5-peer local swarm
- VRF implementation with tests
- Hierarchical all-reduce for million-agent scale
- Byzantine aggregation, sybil defense, reputation, credits

### Known gaps (P0 blockers from PLAN-VIRAL-STRATEGY.md):
1. Rust node `/infer` endpoint is a stub — not connected to Python engine
2. Full kickstart flow not wired through gossipsub for live network training
3. No seed VPS nodes deployed
4. Not published to PyPI under final package name
5. No checkpoint distribution via DHT
6. No live dashboard at ussi.org

## Build & Run

```bash
# Rust node
cd node && cargo build --release

# Python engine
cd engine && pip install -e .

# Agent SDK
cd agent-sdk && pip install -e .

# Local 5-peer swarm
cd docker && docker compose -f docker-compose.swarm.yml up

# Run tests
cd tests && pytest
```

## Plan Documents

- `PLAN.md` — Core architecture (6 phases)
- `PLAN-OPENCLAW-INTEGRATION.md` — USSI platform integration (skill, plugin, webhook)
- `PLAN-VIRAL-STRATEGY.md` — 4-phase go-to-market (target: 1M+ volunteer peers)
