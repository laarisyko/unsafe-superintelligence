"""Cluster management and supernode election for hierarchical aggregation.

This module handles the lifecycle of cluster membership during training rounds:
1. Peers join a training round and advertise their capacity.
2. The VRF deterministically assigns peers to clusters.
3. Cluster leaders (supernodes) are elected as the first member of each cluster.
4. Leaders manage inter-cluster aggregation at higher hierarchy levels.
5. After aggregation, leaders broadcast results back to cluster members.

Supernodes are NOT special infrastructure -- any peer can be elected as a leader
based purely on the deterministic VRF output. Leadership rotates each round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from .hierarchical import (
    ClusterConfig,
    ClusterTopology,
    PeerClusterAssignment,
    assign_clusters_vrf,
    compute_scaling_stats,
)

logger = logging.getLogger(__name__)


class PeerRole(Enum):
    """Role of a peer within the hierarchical aggregation."""
    MEMBER = auto()       # Regular cluster member
    LEADER_L0 = auto()    # Leader at level 0 (leaf cluster leader)
    LEADER_L1 = auto()    # Leader at level 1 (super-cluster leader)
    LEADER_L2 = auto()    # Leader at level 2 (super-super-cluster leader)
    TOP_LEADER = auto()   # Leader at the highest level


@dataclass
class PeerCapacity:
    """A peer's advertised compute capacity."""
    peer_id: str
    gpu_memory_mb: int = 0
    ram_mb: int = 4096
    cpu_cores: int = 1
    accelerator: str = "cpu"
    bandwidth_mbps: float = 100.0


@dataclass
class ClusterMembership:
    """A peer's membership info within the cluster hierarchy."""
    peer_id: str
    assignment: Optional[PeerClusterAssignment] = None
    role: PeerRole = PeerRole.MEMBER
    cluster_peers: List[str] = field(default_factory=list)
    leader_peer_id: Optional[str] = None
    # For leaders: list of peer IDs in sibling clusters at the next level.
    sibling_leaders: List[str] = field(default_factory=list)


class ClusterManager:
    """Manages cluster formation and supernode election for a training round.

    Each training round creates a new ClusterManager. The VRF ensures all
    peers independently compute the same cluster assignments without
    coordination.
    """

    def __init__(
        self,
        round_id: str,
        config: Optional[ClusterConfig] = None,
    ):
        self.round_id = round_id
        self.config = config or ClusterConfig()
        self._participants: Dict[str, PeerCapacity] = {}
        self._topology: Optional[ClusterTopology] = None
        self._memberships: Dict[str, ClusterMembership] = {}
        self._finalized = False

    @property
    def topology(self) -> Optional[ClusterTopology]:
        return self._topology

    @property
    def n_participants(self) -> int:
        return len(self._participants)

    def register_peer(self, capacity: PeerCapacity):
        """Register a peer that wants to participate in this training round."""
        if self._finalized:
            raise RuntimeError("Cannot register peers after finalization")
        self._participants[capacity.peer_id] = capacity

    def finalize(self) -> ClusterTopology:
        """Finalize cluster assignments using VRF.

        After this call, all peers will have identical cluster assignments.
        This MUST be called with the same participant list on all peers
        (ensured by the gossip-based JOIN phase with a deadline).
        """
        if self._finalized:
            return self._topology

        peer_ids = sorted(self._participants.keys())
        n = len(peer_ids)

        if n == 0:
            raise ValueError("No participants registered")

        # Auto-configure if depth is 0.
        if self.config.depth == 0:
            self.config = ClusterConfig.auto(n, self.config.cluster_size)

        self._topology = assign_clusters_vrf(peer_ids, self.round_id, self.config)

        # Build membership records.
        for assignment in self._topology.assignments:
            membership = ClusterMembership(
                peer_id=assignment.peer_id,
                assignment=assignment,
            )

            # Determine role.
            max_leader_level = -1
            for lvl, is_ldr in enumerate(assignment.is_leader):
                if is_ldr:
                    max_leader_level = lvl

            if max_leader_level == -1:
                membership.role = PeerRole.MEMBER
            elif max_leader_level == 0:
                membership.role = PeerRole.LEADER_L0
            elif max_leader_level == 1:
                membership.role = PeerRole.LEADER_L1
            elif max_leader_level == 2:
                membership.role = PeerRole.LEADER_L2
            else:
                membership.role = PeerRole.TOP_LEADER

            # Find cluster peers and leader at level 0.
            if assignment.cluster_path:
                cid = assignment.cluster_path[0]
                members = self._topology.clusters[0].get(cid, [])
                membership.cluster_peers = [
                    peer_ids[gi] for gi in members
                    if gi != assignment.global_index
                ]
                leader_gi = self._topology.leaders[0].get(cid)
                if leader_gi is not None:
                    membership.leader_peer_id = peer_ids[leader_gi]

            # For leaders: find sibling leaders at the next level.
            if membership.role != PeerRole.MEMBER and len(assignment.cluster_path) > 1:
                next_cid = assignment.cluster_path[1]
                next_members = self._topology.clusters[1].get(next_cid, [])
                membership.sibling_leaders = [
                    peer_ids[gi] for gi in next_members
                    if gi != assignment.global_index
                ]

            self._memberships[assignment.peer_id] = membership

        self._finalized = True

        # Log scaling stats.
        stats = compute_scaling_stats(n, self.config)
        logger.info(
            "Cluster topology finalized: %d peers, %d levels, "
            "cluster_size=%d, %d hierarchical rounds (vs %d flat, %.1fx speedup)",
            n,
            self.config.depth,
            self.config.cluster_size,
            stats["hierarchical_rounds"],
            stats["flat_rounds"],
            stats["speedup"],
        )

        return self._topology

    def get_membership(self, peer_id: str) -> ClusterMembership:
        """Get a peer's cluster membership info."""
        if not self._finalized:
            raise RuntimeError("Topology not finalized yet")
        return self._memberships[peer_id]

    def get_cluster_members(self, level: int, cluster_id: int) -> List[str]:
        """Get peer IDs of all members in a specific cluster."""
        if not self._finalized:
            raise RuntimeError("Topology not finalized yet")
        peer_ids = sorted(self._participants.keys())
        members = self._topology.clusters[level].get(cluster_id, [])
        return [peer_ids[gi] for gi in members]

    def get_leader(self, level: int, cluster_id: int) -> str:
        """Get the leader peer ID for a specific cluster."""
        if not self._finalized:
            raise RuntimeError("Topology not finalized yet")
        peer_ids = sorted(self._participants.keys())
        leader_gi = self._topology.leaders[level][cluster_id]
        return peer_ids[leader_gi]

    def is_leader(self, peer_id: str, level: int) -> bool:
        """Check if a peer is a leader at a given level."""
        membership = self._memberships.get(peer_id)
        if not membership or not membership.assignment:
            return False
        if level >= len(membership.assignment.is_leader):
            return False
        return membership.assignment.is_leader[level]

    def summary(self) -> Dict:
        """Return a summary of the cluster topology for logging/debugging."""
        if not self._finalized:
            return {"status": "not_finalized", "n_participants": self.n_participants}

        stats = compute_scaling_stats(self.n_participants, self.config)
        role_counts = {}
        for m in self._memberships.values():
            role_counts[m.role.name] = role_counts.get(m.role.name, 0) + 1

        return {
            "status": "finalized",
            "n_participants": self.n_participants,
            "depth": self.config.depth,
            "cluster_size": self.config.cluster_size,
            "hierarchical_rounds": stats["hierarchical_rounds"],
            "flat_rounds": stats["flat_rounds"],
            "speedup": stats["speedup"],
            "role_distribution": role_counts,
            "n_clusters_level_0": stats["n_clusters_level_0"],
        }
