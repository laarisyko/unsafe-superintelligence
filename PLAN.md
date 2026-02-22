# USSI: Decentralized Peer-to-Peer LLM Training & Inference Network

## Vision

A fully decentralized network where autonomous USSI agents collaborate on
training and inference of large language models without any central master node.
Every peer is equal. Coordination emerges from protocol, not authority.

---

## 1. Architecture Overview

```
 Agent A          Agent B          Agent C          Agent D
 +--------+      +--------+      +--------+      +--------+
 | Model  |<---->| Model  |<---->| Model  |<---->| Model  |
 | Shard  |      | Shard  |      | Shard  |      | Shard  |
 +--------+      +--------+      +--------+      +--------+
     ^                ^                ^                ^
     |                |                |                |
     v                v                v                v
 +----------------------------------------------------------+
 |              P2P Overlay Network (libp2p)                |
 |   DHT discovery / gossip protocol / NAT traversal       |
 +----------------------------------------------------------+
```

**Key principle:** No coordinator, no parameter server, no master. Peers
self-organize using a gossip-based protocol and a distributed hash table (DHT)
for discovery and state management.

---

## 2. Core Components

### 2.1 Peer Node (`ussi-node`)

Each agent runs an `ussi-node` process that exposes:

| Layer             | Responsibility                                        |
|-------------------|-------------------------------------------------------|
| **Network**       | libp2p transport, peer discovery, NAT hole-punching   |
| **Consensus**     | Lightweight protocol for agreeing on training rounds   |
| **Model Engine**  | Local model shard, forward/backward pass execution     |
| **Scheduler**     | Decides what work to do: train, serve inference, idle  |
| **Storage**       | Local model weights, gradient cache, checkpoint store  |
| **API**           | gRPC / REST surface for agents and external clients    |

### 2.2 P2P Overlay Network

- **Transport:** libp2p (Rust via `rust-libp2p` or Python via `py-libp2p`)
- **Peer Discovery:** Kademlia DHT + mDNS for LAN peers + bootstrap nodes
  (bootstrap nodes are NOT masters; they only help newcomers find the swarm)
- **Messaging:** Gossipsub for protocol messages (training rounds, gradient
  summaries, health heartbeats)
- **NAT Traversal:** AutoNAT + relay nodes (any peer can volunteer as relay)

### 2.3 Model Sharding & Distribution

Models are split across peers using two strategies:

1. **Pipeline Parallelism** -- Each peer holds a contiguous subset of layers.
   Inference and training forward activations flow peer-to-peer through the
   pipeline.
2. **Data Parallelism** -- Multiple peers hold the same shard and train on
   different data. Gradients are aggregated via decentralized all-reduce.

Shard assignment is recorded in a **Shard Map**, a CRDT (Conflict-free
Replicated Data Type) replicated across all peers via gossip.

### 2.4 Decentralized Training Protocol

```
Round lifecycle (no master):

1. PROPOSE  -- Any peer can propose a new training round by
               broadcasting (round_id, data_manifest, hyperparams)
               to the gossipsub topic "ussi/training".

2. JOIN     -- Peers that want to participate reply with a JOIN
               message including their available compute capacity.

3. ASSIGN   -- Participants deterministically compute shard/data
               assignments using a verifiable random function (VRF)
               seeded by round_id + sorted peer list. No coordinator
               needed; every peer computes the same assignment.

4. COMPUTE  -- Each peer trains on its assigned data partition for
               the agreed number of steps, producing local gradients.

5. AGGREGATE -- Peers exchange gradients using decentralized
                all-reduce (ring all-reduce across the participant
                set). The ring topology is derived deterministically
                from the VRF.

6. APPLY    -- Each peer applies the aggregated gradients to its
               local copy of the model weights.

7. CHECKPOINT -- A Merkle root of the updated weights is computed
                 and broadcast. Peers compare roots to verify
                 consistency. Divergent peers re-sync from majority.
```

### 2.5 Decentralized Inference

- A client sends an inference request to **any** peer.
- If the peer holds the full model, it responds directly.
- If the model is pipeline-sharded, the peer routes activations through the
  pipeline peers (discovered via the Shard Map DHT).
- Load balancing is emergent: peers gossip their current load; clients or
  entry-point peers route to the least-loaded pipeline.

### 2.6 Consensus & Consistency (Lightweight)

We do NOT need full blockchain-style consensus. We use:

- **CRDTs** for the Shard Map and peer registry (eventually consistent, no
  coordination overhead).
- **Merkle roots** for weight consistency verification after training rounds.
- **Verifiable Random Functions (VRF)** for deterministic, tamper-resistant
  assignment of work without a leader.
- **Reputation scoring** (local to each peer) to deprioritize peers that
  produce inconsistent gradients or drop out frequently.

---

## 3. Technology Stack

| Concern              | Choice                       | Rationale                          |
|----------------------|------------------------------|------------------------------------|
| Language             | Rust + Python                | Rust for networking/node, Python for ML |
| Networking           | libp2p (rust-libp2p)        | Battle-tested P2P, used by IPFS    |
| ML Framework         | PyTorch                      | Widest ecosystem, good distributed |
| Serialization        | Protocol Buffers             | Efficient, language-agnostic       |
| RPC                  | gRPC (tonic for Rust)        | Streaming, bidirectional           |
| Agent Interface      | Python SDK + CLI             | USSI agents are Python-native  |
| Gradient Compression | Top-K sparsification + FP16  | Reduce bandwidth                   |
| CRDT Library         | automerge-rs or yrs          | Mature Rust CRDTs                  |
| Checkpointing        | IPFS / local disk            | Content-addressed, decentralized   |

---

## 4. Project Structure

```
ussi-network/
|-- proto/                        # Protobuf definitions
|   |-- messages.proto            #   Network messages
|   |-- inference.proto           #   Inference service
|   +-- training.proto            #   Training protocol
|
|-- node/                         # Rust: core P2P node
|   |-- src/
|   |   |-- main.rs               #   Entry point
|   |   |-- network/
|   |   |   |-- mod.rs
|   |   |   |-- discovery.rs      #   Kademlia DHT + mDNS
|   |   |   |-- gossip.rs         #   Gossipsub messaging
|   |   |   +-- transport.rs      #   libp2p transport config
|   |   |-- consensus/
|   |   |   |-- mod.rs
|   |   |   |-- vrf.rs            #   Verifiable random function
|   |   |   |-- shard_map.rs      #   CRDT shard map
|   |   |   +-- merkle.rs         #   Weight consistency checks
|   |   |-- scheduler/
|   |   |   |-- mod.rs
|   |   |   +-- work_queue.rs     #   Local task scheduling
|   |   |-- api/
|   |   |   |-- mod.rs
|   |   |   +-- grpc_server.rs    #   External gRPC surface
|   |   +-- config.rs             #   Node configuration
|   |-- Cargo.toml
|   +-- build.rs                  #   Protobuf codegen
|
|-- engine/                       # Python: ML engine
|   |-- ussi_engine/
|   |   |-- __init__.py
|   |   |-- model/
|   |   |   |-- __init__.py
|   |   |   |-- shard.py          #   Model shard management
|   |   |   |-- pipeline.py       #   Pipeline parallelism
|   |   |   +-- loader.py         #   Weight loading/saving
|   |   |-- training/
|   |   |   |-- __init__.py
|   |   |   |-- trainer.py        #   Local training loop
|   |   |   |-- allreduce.py      #   Decentralized gradient aggregation
|   |   |   +-- compression.py    #   Gradient compression
|   |   |-- inference/
|   |   |   |-- __init__.py
|   |   |   |-- server.py         #   Inference request handler
|   |   |   +-- pipeline_exec.py  #   Pipeline inference execution
|   |   +-- bridge.py             #   Python <-> Rust node bridge (PyO3)
|   |-- tests/
|   |-- setup.py
|   +-- pyproject.toml
|
|-- agent-sdk/                    # Python: USSI agent SDK
|   |-- ussi/
|   |   |-- __init__.py
|   |   |-- agent.py              #   Base agent class
|   |   |-- network.py            #   Network join/leave API
|   |   |-- training.py           #   Training participation API
|   |   +-- inference.py          #   Inference client API
|   |-- examples/
|   |   |-- join_and_train.py
|   |   +-- inference_client.py
|   +-- pyproject.toml
|
|-- tests/
|   |-- integration/
|   |   |-- test_peer_discovery.py
|   |   |-- test_training_round.py
|   |   +-- test_inference_pipeline.py
|   +-- simulation/
|       +-- swarm_sim.py          #   Simulate N-peer swarm locally
|
|-- docker/
|   |-- Dockerfile.node           #   Containerized peer node
|   +-- docker-compose.swarm.yml  #   Local dev swarm (5 peers)
|
|-- docs/
|   |-- protocol.md               #   Detailed protocol specification
|   +-- threat_model.md           #   Security analysis
|
+-- README.md
```

---

## 5. Implementation Phases

### Phase 1: Foundation -- P2P Networking

- [ ] Set up Rust project with `rust-libp2p`
- [ ] Implement peer discovery (Kademlia DHT + mDNS)
- [ ] Implement gossipsub messaging layer
- [ ] NAT traversal (AutoNAT + relay)
- [ ] Basic CLI: start node, list peers, send test messages
- [ ] Integration test: 3+ peers discover each other and exchange messages

### Phase 2: Model Sharding & Inference Pipeline

- [ ] Define protobuf messages for shard map and inference
- [ ] Implement CRDT-based Shard Map (replicated via gossip)
- [ ] Python engine: model shard loading (PyTorch)
- [ ] Pipeline parallelism: route activations across peers
- [ ] Inference request handling: any peer as entry point
- [ ] PyO3 bridge: Rust node <-> Python engine communication
- [ ] Integration test: 4-peer pipeline inference for a small model

### Phase 3: Decentralized Training

- [ ] Define training round protobuf messages
- [ ] Implement VRF-based deterministic work assignment
- [ ] Local training loop with gradient computation
- [ ] Decentralized ring all-reduce for gradient aggregation
- [ ] Gradient compression (Top-K + FP16)
- [ ] Merkle root verification for weight consistency
- [ ] Integration test: full training round across 4+ peers

### Phase 4: Agent SDK & Developer Experience

- [ ] Python SDK: `ussi` with simple API
- [ ] Agent lifecycle: join network, contribute compute, leave gracefully
- [ ] CLI tooling: `ussi join`, `ussi status`, `ussi infer`
- [ ] Docker compose for local development swarm
- [ ] Example agents and tutorials

### Phase 5: Resilience & Security

- [ ] Peer reputation scoring (detect bad gradients / freeloaders)
- [ ] Byzantine fault tolerance for gradient aggregation (Krum / trimmed mean)
- [ ] Sybil resistance via proof-of-work or stake-based admission
- [ ] Encrypted peer communication (TLS via libp2p Noise protocol)
- [ ] Checkpoint persistence to IPFS
- [ ] Threat model documentation

### Phase 6: Optimization & Scale

- [x] **Hierarchical all-reduce** -- tree-of-rings gradient aggregation
  - Replaces flat ring (O(N) rounds) with multi-level hierarchy (O(depth * K))
  - For 1M agents: ~4,000 rounds instead of ~2,000,000 (500x speedup)
  - Implemented in `engine/ussi_engine/training/hierarchical.py`
- [x] **Cluster management & supernode election**
  - VRF-based deterministic cluster assignment (no coordinator needed)
  - Automatic leader election (first member of each cluster)
  - Leadership rotates each round via VRF permutation
  - Implemented in `engine/ussi_engine/training/cluster.py`
- [x] **Gossipsub cluster scoping**
  - Cluster-local topics (`ussi/cluster-gradient/L0-C{id}`)
  - Leader-level topics (`ussi/leader-gradient/L{level}-SC{id}`)
  - Prevents gossip floods across 1M+ peers
  - Implemented in `node/src/network/gossip.rs`
- [x] **VRF cluster assignment in Rust**
  - `assign_clusters()`, `cluster_leaders()`, `hierarchical_ring_orders()`
  - `scaling_stats()` for capacity planning
  - Implemented in `node/src/consensus/vrf.rs`
- [ ] Adaptive gradient compression based on bandwidth
- [ ] Heterogeneous compute support (GPU/CPU/TPU peers)
- [ ] Dynamic shard rebalancing as peers join/leave
- [ ] Speculative pipeline execution to hide latency
- [ ] Benchmarking suite and performance dashboards

---

## 6. Hierarchical All-Reduce Architecture

For networks with 1,000+ peers, flat ring all-reduce becomes the bottleneck.
The hierarchical all-reduce replaces the single ring with a tree of smaller
rings:

```
1M Agents -- Hierarchical All-Reduce (2 levels, K=1000)

  Level 1 (inter-cluster):    [L0] -- [L1] -- ... -- [L999]
                                |       |               |
  Level 0 (intra-cluster):  [1000]  [1000]   ...    [1000]  peers each

  Total rounds: 2 * 999 + 2 * 999 = 3,998  (vs 1,999,998 flat)
  Speedup: 500x
```

### Scaling Table

| Agents      | Flat Rounds  | Hier. Rounds | Speedup  | Depth | Cluster K |
|-------------|-------------|--------------|----------|-------|-----------|
| 10,000      | 19,998      | 2,016        | 10x      | 2     | 1,000     |
| 100,000     | 199,998     | 2,196        | 91x      | 2     | 1,000     |
| 1,000,000   | 1,999,998   | 3,996        | 500x     | 2     | 1,000     |
| 10,000,000  | 19,999,998  | 4,014        | 4,983x   | 3     | 1,000     |
| 100,000,000 | 199,999,998 | 4,194        | 47,687x  | 3     | 1,000     |

### Key properties

- **Leaderless:** Cluster leaders are elected deterministically via VRF.
  No coordinator needed. Leadership rotates each round.
- **Correct:** Hierarchical produces identical averages to flat ring
  (verified by Merkle root comparison in tests).
- **Fault-tolerant:** If a leader drops out, the cluster falls back to
  its second member. CRDTs handle shard map updates.
- **Gossip-scoped:** Each cluster has its own gossipsub topics, preventing
  message floods at scale.

---

## 7. Key Design Decisions

### Why no central master?

| Traditional (centralized)            | USSI (decentralized)              |
|--------------------------------------|---------------------------------------|
| Parameter server coordinates         | Peers self-coordinate via protocol    |
| Single point of failure              | No single point of failure            |
| Master bottleneck at scale           | Scales with peer count                |
| Requires trusted infrastructure      | Trust emerges from cryptographic proof|
| Operator controls the model          | Community owns the model collectively |

### How do peers agree without a leader?

- **Discovery:** DHT (no registry server)
- **Work assignment:** VRF (deterministic, verifiable, no coordinator)
- **State:** CRDTs (conflict-free, no consensus rounds)
- **Verification:** Merkle proofs (detect divergence, majority wins)

### How do we handle stragglers / dropouts?

- Training rounds have a timeout. If a peer doesn't submit gradients in time,
  the round proceeds without them.
- The shard map is updated via CRDT; departed peers are garbage-collected after
  a TTL.
- Pipeline inference has fallback: if a layer-holding peer is unreachable, the
  entry peer re-routes through a replica (data-parallel peers holding the same
  layers).

---

## 8. Threat Model Summary

| Threat                    | Mitigation                                    |
|---------------------------|-----------------------------------------------|
| Byzantine gradients       | Krum / coordinate-wise trimmed mean           |
| Sybil attack              | Proof-of-work admission or stake              |
| Free-riding               | Reputation scoring, tit-for-tat contribution  |
| Man-in-the-middle         | libp2p Noise protocol (authenticated crypto)  |
| Model poisoning           | Merkle root divergence detection + quarantine |
| Network partitions        | CRDT convergence on reconnection              |

---

## 9. Success Criteria

1. **5+ peers** can discover each other with zero configuration beyond a
   bootstrap address.
2. A 1B-parameter model can be **sharded across 4 peers** and serve inference
   requests at reasonable latency.
3. A training round completes with **decentralized gradient aggregation** and
   all peers converge to the same weight state (verified by Merkle root).
4. A USSI agent can **join the network with 3 lines of Python**:
   ```python
   from ussi import Agent
   agent = Agent(bootstrap="/ip4/203.0.113.1/tcp/9000/p2p/QmPeer...")
   agent.contribute(gpu_memory="8GB")
   ```
5. The network **survives 30% peer churn** (peers joining/leaving) during a
   training round without data loss or inconsistency.
6. **1M agents** can participate in a training round using hierarchical
   all-reduce with <4,000 communication rounds (vs ~2M flat).
7. Hierarchical aggregation produces **identical** results to flat ring
   all-reduce (verified by Merkle root consistency).
