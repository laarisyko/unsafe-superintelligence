# OpenClaw Threat Model

## 1. Threat Categories

### 1.1 Byzantine Gradient Attacks

**Threat:** A malicious peer sends corrupt gradients during all-reduce to
poison the model.

**Mitigations:**
- **Krum aggregation:** Select the gradient closest to the geometric median,
  rejecting outliers.
- **Coordinate-wise trimmed mean:** For each parameter, discard the highest
  and lowest values before averaging.
- **Merkle root verification:** After aggregation, all peers compare weight
  Merkle roots. Divergent peers are flagged and quarantined.
- **Reputation scoring:** Peers that produce inconsistent gradients accumulate
  negative reputation.

### 1.2 Sybil Attacks

**Threat:** An attacker creates many fake identities to gain disproportionate
influence over training or outvote honest peers in Merkle verification.

**Mitigations:**
- **Proof-of-work admission:** New peers must solve a computational puzzle
  before joining the network.
- **Stake-based admission:** Peers deposit a stake that is slashed for
  misbehavior (requires an on-chain component).
- **Reputation weighting:** New peers start with low reputation and have
  limited influence until they prove reliable.

### 1.3 Free-Riding

**Threat:** A peer participates in the network to receive inference services
but does not contribute compute to training.

**Mitigations:**
- **Tit-for-tat contribution tracking:** Peers preferentially serve inference
  to peers that have contributed training compute.
- **Reputation-gated access:** Only peers above a reputation threshold can
  submit inference requests.

### 1.4 Man-in-the-Middle

**Threat:** An attacker intercepts communication between peers to steal model
weights, gradients, or inference inputs.

**Mitigations:**
- **Noise protocol:** All libp2p connections are encrypted and authenticated
  using the Noise framework.
- **Peer identity verification:** Each peer has a cryptographic identity
  (Ed25519 keypair). Connections are authenticated via the Identify protocol.

### 1.5 Model Poisoning

**Threat:** A coordinated attack where multiple peers inject carefully crafted
gradients to embed a backdoor in the model.

**Mitigations:**
- **Merkle root consensus:** Weight consistency is verified after every round.
  A coordinated poisoning attack would require controlling >50% of peers.
- **Gradient clipping:** Local gradient norms are clipped, limiting the
  magnitude of any single peer's contribution.
- **Audit logging:** All training rounds, participants, and Merkle roots are
  logged for forensic analysis.

### 1.6 Network Partitions

**Threat:** The network splits into disconnected subgroups that train
independently, leading to model divergence.

**Mitigations:**
- **CRDT convergence:** The shard map and peer registry use CRDTs that
  automatically converge when partitions heal.
- **Round quorum:** Training rounds require a minimum number of participants.
  Partitioned sub-groups below quorum pause training.
- **Merkle reconciliation:** When partitions reconnect, peers compare Merkle
  roots and the majority state wins.

### 1.7 Denial of Service

**Threat:** An attacker floods the gossipsub network with messages or
repeatedly proposes training rounds.

**Mitigations:**
- **Gossipsub rate limiting:** The gossipsub protocol has built-in message
  scoring and rate limiting.
- **Peer banning:** Peers that exceed message rate limits are temporarily
  banned.
- **Proposal rate limiting:** Only peers with sufficient reputation can
  propose training rounds more than once per time window.

## 2. Trust Assumptions

| Assumption | Justification |
|---|---|
| Majority of peers are honest | Standard assumption for decentralized systems |
| Cryptographic primitives are secure | SHA-256, Ed25519, Noise are well-studied |
| libp2p transport is reliable | Battle-tested in IPFS, Filecoin, Ethereum |
| Peers have correct local clocks (±minutes) | Used for heartbeat TTLs, not consensus |

## 3. Out of Scope

The following are explicitly out of scope for the current threat model:
- Side-channel attacks on model inference (timing, cache)
- Physical attacks on peer hardware
- Legal/regulatory compliance (data privacy, model licensing)
- Attacks on the training data distribution mechanism
