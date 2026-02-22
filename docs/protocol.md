# USSI Network Protocol Specification

## Overview

The USSI protocol enables fully decentralized training and inference of
large language models. There is no master node. All coordination emerges from
the protocol rules, cryptographic primitives, and CRDTs.

## 1. Transport Layer

- **Library:** libp2p
- **Transport:** TCP with Noise encryption
- **Multiplexing:** Yamux
- **Identification:** Identify protocol (`/ussi/0.1.0`)

## 2. Peer Discovery

### 2.1 Kademlia DHT

Peers maintain a Kademlia distributed hash table for long-range discovery.
The DHT is used to:
- Find peers by their PeerId
- Store and retrieve shard map entries
- Bootstrap new peers into the network

### 2.2 mDNS

For LAN environments, mDNS enables zero-configuration peer discovery.
Peers broadcast their presence on the local network.

### 2.3 Bootstrap Nodes

Bootstrap nodes are ordinary peers that are well-known and long-lived.
They serve only as entry points -- a new peer dials a bootstrap node to
learn about other peers, then operates independently.

Bootstrap nodes have **no special authority**. Any peer can serve as a
bootstrap node.

## 3. Gossipsub Topics

All protocol messages are broadcast via gossipsub on well-known topics:

| Topic                    | Purpose                           |
|--------------------------|-----------------------------------|
| `ussi/heartbeat`     | Peer liveness and capacity ads    |
| `ussi/shard-map`     | CRDT shard map updates            |
| `ussi/training`      | Training round proposals/joins    |
| `ussi/gradient`      | Gradient readiness announcements  |
| `ussi/checkpoint`    | Checkpoint completion notices     |

## 4. Shard Map (CRDT)

The shard map tracks which peer holds which model layers. It is a
**Last-Writer-Wins Element Set** (LWW-Element-Set) CRDT:

- **Key:** `(model_id, layer_start, layer_end)`
- **Value:** `(peer_id, lamport_timestamp)`
- **Merge rule:** Higher timestamp wins

The shard map is replicated across all peers via the `ussi/shard-map`
gossipsub topic. Because it is a CRDT, it converges without coordination.

## 5. Training Protocol

### 5.1 Round Lifecycle

```
PROPOSE -> JOIN -> ASSIGN -> COMPUTE -> AGGREGATE -> APPLY -> CHECKPOINT
```

1. **PROPOSE:** Any peer broadcasts a `TrainingProposal` containing:
   - `round_id` (unique identifier)
   - `model_id`
   - `hyper_params` (learning rate, batch size, steps)
   - `data_manifest_cid` (content address of training data manifest)
   - `deadline_ms` (how long to wait for JOINs)

2. **JOIN:** Interested peers respond with `TrainingJoin`:
   - `round_id`
   - `peer_id`
   - `capacity` (GPU mem, RAM, cores)

3. **ASSIGN:** After the deadline, each peer independently computes:
   - Sort all JOIN peer_ids lexicographically
   - Compute VRF: `SHA-256("ussi-vrf-v1:" || round_id || ":" || peers)`
   - Derive a permutation from the VRF hash
   - Assign data partitions and ring positions from the permutation

   **No coordinator needed** -- every peer computes the same assignment.

4. **COMPUTE:** Each peer trains on its assigned data partition for the
   agreed number of steps, producing local gradients.

5. **AGGREGATE:** Ring all-reduce:
   - The ring topology is the VRF-derived permutation
   - Phase 1 (scatter-reduce): N-1 steps, each peer sends 1/N of gradients
   - Phase 2 (all-gather): N-1 steps, distribute fully-reduced slices
   - Optional gradient compression (Top-K + FP16)

6. **APPLY:** Each peer applies the averaged gradients via its optimizer.

7. **CHECKPOINT:** Each peer:
   - Computes the Merkle root of updated weights
   - Broadcasts `CheckpointAnnounce` with the root
   - Compares roots -- if all match, the round is successful
   - Divergent peers re-sync from the majority

### 5.2 Straggler Handling

- Each round has a configurable timeout
- If a peer doesn't submit gradients in time, the round proceeds without them
- Timed-out peers receive a reputation penalty

## 6. Inference Protocol

1. Client sends `InferRequest` to any peer
2. If the peer holds the full model, respond directly
3. If pipeline-sharded:
   a. Consult the shard map to find the pipeline
   b. Tokenize and create initial activations
   c. Forward activations to the first pipeline stage
   d. Each stage runs its layers and forwards to the next
   e. The last stage detokenizes and returns the result

### 6.1 Load Balancing

- Peers gossip their `current_load` in heartbeats
- Entry-point peers route to the least-loaded pipeline replica
- This is emergent -- no central load balancer

## 7. Verifiable Random Function (VRF)

The VRF ensures deterministic, tamper-resistant work assignment:

```
VRF(round_id, peers) = SHA-256("ussi-vrf-v1:" || round_id || ":" || join(peers, ","))
```

Properties:
- **Deterministic:** Same inputs always produce same output
- **Verifiable:** Any peer can verify the assignment
- **Unpredictable:** The permutation depends on which peers join, preventing manipulation

## 8. Message Formats

All messages use Protocol Buffers (see `proto/` directory):
- `messages.proto` -- Core types (PeerId, Heartbeat, ShardMap, etc.)
- `inference.proto` -- InferenceService RPC definitions
- `training.proto` -- TrainingService RPC definitions
