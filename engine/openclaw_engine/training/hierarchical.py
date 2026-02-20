"""Hierarchical all-reduce for million-agent gradient aggregation.

Standard ring all-reduce requires 2(N-1) communication rounds, which breaks
at scale (1M agents = ~2M rounds per gradient sync). This module implements
a tree-of-rings approach that reduces total rounds from O(N) to O(depth * K),
where K is the cluster size at each level.

Architecture:
    Level 0 (leaf):  Peers within a cluster run ring all-reduce (K peers)
    Level 1:         Cluster leaders run ring all-reduce (K groups)
    Level 2:         Super-cluster leaders run ring all-reduce (K groups)
    ...

For 1M agents with cluster_size=1000:
    Level 0: 1000 clusters x 1000 peers = 1M peers, 2*999 rounds
    Level 1: 1000 cluster leaders, 2*999 rounds
    Total: ~3998 rounds (vs ~2,000,000 flat)

For 1M agents with 3-level hierarchy (cluster_size=100):
    Level 0: 10,000 clusters x 100 peers, 2*99 rounds
    Level 1: 100 super-clusters x 100 leaders, 2*99 rounds
    Level 2: 100 super-leaders, 2*99 rounds
    Total: ~594 rounds (vs ~2,000,000 flat)
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .allreduce import RingAllReduce, RingPeer

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for hierarchical clustering."""

    # Max number of peers per cluster at each level.
    cluster_size: int = 1000

    # Number of hierarchy levels. 0 = auto-detect from total peer count.
    depth: int = 0

    # Minimum peers to trigger hierarchical mode. Below this, flat ring is used.
    hierarchical_threshold: int = 64

    @staticmethod
    def auto(n_peers: int, target_cluster_size: int = 1000) -> "ClusterConfig":
        """Auto-configure hierarchy based on total peer count.

        Each level of hierarchy adds a synchronization barrier, so we prefer
        fewer levels with larger clusters (up to target_cluster_size), rather
        than deep trees with tiny clusters.

        Strategy: use the target_cluster_size and compute the minimum depth
        needed to cover all peers: depth = ceil(log(n) / log(K)).
        """
        if n_peers <= 64:
            return ClusterConfig(cluster_size=n_peers, depth=1)

        k = min(target_cluster_size, n_peers)
        if k <= 1:
            return ClusterConfig(cluster_size=n_peers, depth=1)

        depth = math.ceil(math.log(n_peers) / math.log(k))
        depth = max(depth, 1)

        return ClusterConfig(cluster_size=k, depth=depth)


@dataclass
class PeerClusterAssignment:
    """A peer's position within the hierarchy."""

    peer_id: str
    global_index: int
    # Cluster IDs at each level. E.g. [cluster_at_level_0, cluster_at_level_1, ...]
    cluster_path: List[int] = field(default_factory=list)
    # Position within the leaf cluster (level 0).
    position_in_cluster: int = 0
    # Whether this peer is the leader (index 0) at each level.
    is_leader: List[bool] = field(default_factory=list)


@dataclass
class ClusterTopology:
    """Complete topology for hierarchical all-reduce."""

    config: ClusterConfig
    n_peers: int
    assignments: List[PeerClusterAssignment] = field(default_factory=list)

    # Indexed by level, then by cluster_id: list of peer global indices.
    clusters: List[Dict[int, List[int]]] = field(default_factory=list)

    # Leaders at each level: cluster_id -> peer global index.
    leaders: List[Dict[int, int]] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return self.config.depth

    @property
    def total_communication_rounds(self) -> int:
        """Total communication rounds across all hierarchy levels."""
        total = 0
        for level in range(self.depth):
            if level < len(self.clusters) and self.clusters[level]:
                # All clusters at this level have the same size (within 1).
                sizes = [len(members) for members in self.clusters[level].values()]
                max_size = max(sizes) if sizes else 1
                total += 2 * (max_size - 1)
        return total


def assign_clusters_vrf(
    peer_ids: List[str],
    round_id: str,
    config: ClusterConfig,
) -> ClusterTopology:
    """Deterministically assign peers to a hierarchical cluster topology.

    Uses a VRF-style deterministic hash to assign peers to clusters,
    ensuring all peers independently compute the same topology without
    coordination.

    Args:
        peer_ids: Sorted list of participating peer IDs.
        round_id: Training round identifier (for VRF seed).
        config: Cluster configuration.

    Returns:
        Complete ClusterTopology with peer assignments.
    """
    n_peers = len(peer_ids)
    if config.depth == 0:
        config = ClusterConfig.auto(n_peers, config.cluster_size)

    # Generate deterministic permutation (mirrors Rust VRF).
    seed = _vrf_hash(round_id, peer_ids)
    permutation = _deterministic_permutation(seed, n_peers)

    # Build hierarchy bottom-up.
    topology = ClusterTopology(config=config, n_peers=n_peers)
    topology.clusters = [{} for _ in range(config.depth)]
    topology.leaders = [{} for _ in range(config.depth)]

    # Level 0: assign permuted peers to leaf clusters.
    k = config.cluster_size
    for idx, perm_pos in enumerate(permutation):
        cluster_id = perm_pos // k
        pos_in_cluster = perm_pos % k
        if cluster_id not in topology.clusters[0]:
            topology.clusters[0][cluster_id] = []
        topology.clusters[0][cluster_id].append(idx)

    # Sort members within each cluster for determinism.
    for cid in topology.clusters[0]:
        topology.clusters[0][cid].sort()

    # Elect leaders at level 0: first member of each cluster.
    for cid, members in topology.clusters[0].items():
        topology.leaders[0][cid] = members[0]

    # Build higher levels: group leaders of level L into clusters of level L+1.
    for level in range(1, config.depth):
        prev_leaders = sorted(topology.leaders[level - 1].items())
        leader_indices = [gi for _, gi in prev_leaders]

        for i, gi in enumerate(leader_indices):
            cluster_id = i // k
            if cluster_id not in topology.clusters[level]:
                topology.clusters[level][cluster_id] = []
            topology.clusters[level][cluster_id].append(gi)

        for cid, members in topology.clusters[level].items():
            members.sort()
            topology.leaders[level][cid] = members[0]

    # Build per-peer assignments.
    for idx, peer_id in enumerate(peer_ids):
        assignment = PeerClusterAssignment(
            peer_id=peer_id,
            global_index=idx,
        )

        # Find cluster path: which cluster is this peer in at each level?
        current_gi = idx
        for level in range(config.depth):
            for cid, members in topology.clusters[level].items():
                if current_gi in members:
                    assignment.cluster_path.append(cid)
                    is_leader = topology.leaders[level].get(cid) == current_gi
                    assignment.is_leader.append(is_leader)
                    break
            else:
                # Peer is not a leader at this level, stop climbing.
                break

        assignment.position_in_cluster = _find_position(
            topology.clusters[0], idx
        )
        topology.assignments.append(assignment)

    return topology


class HierarchicalAllReduce:
    """Hierarchical gradient aggregation using a tree of rings.

    Each level of the hierarchy runs independent ring all-reduce operations.
    Results are propagated up the tree (leaders aggregate across clusters)
    and then back down (leaders broadcast to cluster members).

    This reduces communication rounds from O(N) to O(depth * K).
    """

    def __init__(self, topology: ClusterTopology, local_index: int):
        self.topology = topology
        self.local_index = local_index

    @staticmethod
    def reduce_all(
        topology: ClusterTopology,
        all_gradients: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Run hierarchical all-reduce across all peers in lockstep simulation.

        This is the local simulation version for testing. In production,
        each peer runs reduce_async() with actual network communication.

        Algorithm:
            1. Level 0 (bottom-up): Each leaf cluster runs ring all-reduce.
               Result: each peer in a cluster has the cluster-averaged gradient.

            2. Level 1..depth-1 (bottom-up): Cluster leaders run ring all-reduce
               with leaders of sibling clusters.
               Result: each leader has the globally-averaged gradient for their
               super-cluster.

            3. Top-down broadcast: Leaders distribute the final result back
               to their cluster members.

        Args:
            topology: The hierarchical cluster topology.
            all_gradients: List of gradient dicts, one per peer.

        Returns:
            List of averaged gradient dicts (one per peer, all identical).
        """
        n = topology.n_peers
        assert len(all_gradients) == n

        if n == 0:
            return []

        if n == 1:
            return [all_gradients[0]]

        # Use flat ring for small peer counts.
        if n <= topology.config.hierarchical_threshold:
            rings = RingAllReduce.local_ring(n)
            return RingAllReduce.reduce_all(rings, all_gradients)

        # Working copy of gradients -- will be modified in-place at each level.
        working = [
            {k: v.clone() for k, v in g.items()} for g in all_gradients
        ]

        # === PHASE 1: Bottom-up aggregation ===
        for level in range(topology.depth):
            clusters_at_level = topology.clusters[level]

            for cid, members in clusters_at_level.items():
                if len(members) <= 1:
                    continue

                # Collect gradients from cluster members.
                member_grads = [working[gi] for gi in members]

                # Run ring all-reduce within this cluster.
                rings = RingAllReduce.local_ring(len(members))
                reduced = RingAllReduce.reduce_all(rings, member_grads)

                # Write back reduced gradients to cluster members.
                for i, gi in enumerate(members):
                    working[gi] = reduced[i]

        # === PHASE 2: Top-down broadcast ===
        # After bottom-up, leaders at each level have the aggregated result
        # for their portion of the tree. But non-leader peers at higher levels
        # only have their cluster-level aggregate. We need to push the global
        # aggregate from the top leaders down to all peers.
        for level in range(topology.depth - 1, -1, -1):
            clusters_at_level = topology.clusters[level]

            for cid, members in clusters_at_level.items():
                if len(members) <= 1:
                    continue

                leader_gi = topology.leaders[level][cid]
                leader_grads = working[leader_gi]

                # Broadcast leader's result to all members.
                for gi in members:
                    if gi != leader_gi:
                        working[gi] = {
                            k: v.clone() for k, v in leader_grads.items()
                        }

        return working

    async def reduce_async(
        self,
        local_gradients: Dict[str, torch.Tensor],
        send_fn: Callable,
        recv_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Async hierarchical all-reduce using network communication.

        Each peer participates in ring all-reduce at level 0 (its leaf cluster).
        If the peer is a cluster leader, it additionally participates in
        higher-level rings.

        Args:
            local_gradients: This peer's gradient dict.
            send_fn: async fn(data_bytes, target_peer_id) -> None
            recv_fn: async fn(source_peer_id) -> data_bytes

        Returns:
            Globally averaged gradient dict.
        """
        assignment = self.topology.assignments[self.local_index]
        result = local_gradients

        # Phase 1: Bottom-up -- participate in rings at each level where
        # this peer is a member.
        for level in range(self.topology.depth):
            if level >= len(assignment.cluster_path):
                break

            cid = assignment.cluster_path[level]
            members = self.topology.clusters[level][cid]

            if len(members) <= 1:
                continue

            # Only leaders participate in levels > 0.
            if level > 0 and not assignment.is_leader[level - 1]:
                break

            # Build ring peers for this cluster.
            my_pos = members.index(self.local_index)
            ring_peers = [
                RingPeer(
                    peer_id=self.topology.assignments[gi].peer_id,
                    ring_position=pos,
                    send_fn=send_fn,
                    recv_fn=recv_fn,
                )
                for pos, gi in enumerate(members)
            ]

            ring = RingAllReduce(ring=ring_peers, local_position=my_pos)
            result = await ring.reduce_async(result)

        # Phase 2: Top-down -- leaders broadcast to non-leader members.
        # In the async version, leaders send the final result to their
        # cluster members via direct messages.
        for level in range(self.topology.depth - 1, -1, -1):
            if level >= len(assignment.cluster_path):
                continue

            cid = assignment.cluster_path[level]
            leader_gi = self.topology.leaders[level][cid]

            if self.local_index == leader_gi:
                # I'm the leader -- send result to all members.
                members = self.topology.clusters[level][cid]
                for gi in members:
                    if gi != self.local_index:
                        peer_id = self.topology.assignments[gi].peer_id
                        data = _serialize_gradients(result)
                        await send_fn(data, peer_id)
            else:
                # I'm not the leader -- receive from leader.
                leader_peer_id = self.topology.assignments[leader_gi].peer_id
                data = await recv_fn(leader_peer_id)
                if data:
                    result = _deserialize_gradients(data, result)

        return result


def compute_scaling_stats(n_peers: int, config: Optional[ClusterConfig] = None) -> Dict:
    """Compute scaling statistics for a given peer count and config.

    Useful for capacity planning and understanding the efficiency gains
    of hierarchical vs flat all-reduce.
    """
    if config is None:
        config = ClusterConfig.auto(n_peers)

    flat_rounds = 2 * (n_peers - 1) if n_peers > 1 else 0

    # Simulate hierarchy to count actual rounds.
    k = config.cluster_size
    depth = config.depth

    hierarchical_rounds = 0
    peers_at_level = n_peers
    for level in range(depth):
        n_clusters = math.ceil(peers_at_level / k)
        cluster_sz = min(k, peers_at_level)
        hierarchical_rounds += 2 * (cluster_sz - 1)
        peers_at_level = n_clusters

    speedup = flat_rounds / hierarchical_rounds if hierarchical_rounds > 0 else float("inf")

    return {
        "n_peers": n_peers,
        "flat_rounds": flat_rounds,
        "hierarchical_rounds": hierarchical_rounds,
        "speedup": speedup,
        "depth": depth,
        "cluster_size": k,
        "n_clusters_level_0": math.ceil(n_peers / k),
        "config": config,
    }


# --- Internal helpers ---


def _vrf_hash(round_id: str, peer_ids: List[str]) -> bytes:
    """Compute deterministic VRF hash matching the Rust implementation."""
    h = hashlib.sha256()
    h.update(b"openclaw-vrf-v1:")
    h.update(round_id.encode())
    h.update(b":")
    for pid in peer_ids:
        h.update(pid.encode())
        h.update(b",")
    return h.digest()


def _deterministic_permutation(seed: bytes, n: int) -> List[int]:
    """Fisher-Yates shuffle matching the Rust VRF permutation."""
    if n == 0:
        return []

    indices = list(range(n))
    current_seed = seed

    for i in range(n - 1, 0, -1):
        h = hashlib.sha256()
        h.update(current_seed)
        h.update(i.to_bytes(8, byteorder="little"))
        current_seed = h.digest()

        rand_val = int.from_bytes(current_seed[:8], byteorder="little")
        j = rand_val % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    return indices


def _find_position(level0_clusters: Dict[int, List[int]], global_index: int) -> int:
    """Find a peer's position within its level-0 cluster."""
    for cid, members in level0_clusters.items():
        if global_index in members:
            return members.index(global_index)
    return 0


def _serialize_gradients(grads: Dict[str, torch.Tensor]) -> bytes:
    """Serialize gradient dict to bytes for network transfer."""
    import io
    buf = io.BytesIO()
    torch.save(grads, buf)
    return buf.getvalue()


def _deserialize_gradients(
    data: bytes, template: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Deserialize gradient bytes back to dict."""
    import io
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)
