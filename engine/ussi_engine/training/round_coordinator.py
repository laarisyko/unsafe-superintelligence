"""Training round coordinator with straggler/dropout tolerance.

Manages the lifecycle of a decentralized training round:
    PROPOSE -> JOIN (deadline) -> ASSIGN -> COMPUTE -> AGGREGATE -> VERIFY

Key resilience features:
    - Join deadline: peers must register before the cutoff
    - Gradient deadline: stragglers are excluded after timeout
    - Minimum quorum: round proceeds only if enough peers participate
    - Dropout recovery: mid-round departures reduce the participant set
      but don't abort the round (as long as quorum is maintained)
    - Leader failover: if a cluster leader drops, the next member takes over
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

import torch

from .byzantine import AggregationMethod, ByzantineConfig, robust_aggregate, score_gradients
from .cluster import ClusterManager, PeerCapacity
from .hierarchical import ClusterConfig, HierarchicalAllReduce, assign_clusters_vrf

logger = logging.getLogger(__name__)


class RoundPhase(Enum):
    """Phases of a training round."""
    PROPOSED = auto()
    JOINING = auto()
    ASSIGNED = auto()
    COMPUTING = auto()
    AGGREGATING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class RoundConfig:
    """Configuration for a training round."""

    round_id: str
    model_id: str

    # Timing.
    join_deadline_secs: float = 30.0
    gradient_deadline_secs: float = 120.0

    # Quorum.
    min_peers: int = 3
    max_peers: int = 0  # 0 = unlimited

    # Aggregation.
    cluster_size: int = 1000
    byzantine_config: ByzantineConfig = field(
        default_factory=lambda: ByzantineConfig(method=AggregationMethod.TRIMMED_MEAN)
    )

    # Dropout tolerance.
    min_quorum_ratio: float = 0.67  # Round fails if <67% of peers submit gradients


@dataclass
class PeerStatus:
    """Track a single peer's progress through the round."""
    peer_id: str
    joined_at: float = 0.0
    gradient_submitted: bool = False
    gradient_submitted_at: float = 0.0
    merkle_root: str = ""
    dropped: bool = False
    byzantine_score: float = 0.0


class RoundCoordinator:
    """Coordinates a single training round with fault tolerance.

    This coordinator is deterministic: every peer runs the same logic
    with the same inputs and arrives at the same decisions. No central
    authority needed.

    Usage:
        coord = RoundCoordinator(config)
        coord.register_peer(peer_capacity)   # during JOIN phase
        coord.close_registration()            # after deadline
        coord.submit_gradient(peer_id, grads) # during COMPUTE phase
        result = coord.finalize()             # aggregate + verify
    """

    def __init__(self, config: RoundConfig):
        self.config = config
        self.phase = RoundPhase.PROPOSED
        self._peers: Dict[str, PeerStatus] = {}
        self._capacities: Dict[str, PeerCapacity] = {}
        self._gradients: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cluster_manager: Optional[ClusterManager] = None
        self._result: Optional[Dict[str, torch.Tensor]] = None
        self._start_time = time.monotonic()
        self._registration_closed = False

    @property
    def n_registered(self) -> int:
        return len(self._peers)

    @property
    def n_submitted(self) -> int:
        return sum(1 for p in self._peers.values() if p.gradient_submitted)

    @property
    def active_peers(self) -> List[str]:
        return [pid for pid, ps in self._peers.items() if not ps.dropped]

    def register_peer(self, capacity: PeerCapacity) -> bool:
        """Register a peer for this round. Returns False if registration is closed."""
        if self._registration_closed:
            logger.warning("Peer %s tried to join after registration closed", capacity.peer_id)
            return False

        if self.config.max_peers > 0 and len(self._peers) >= self.config.max_peers:
            logger.warning("Round %s is full (%d peers)", self.config.round_id, self.config.max_peers)
            return False

        self._peers[capacity.peer_id] = PeerStatus(
            peer_id=capacity.peer_id,
            joined_at=time.monotonic(),
        )
        self._capacities[capacity.peer_id] = capacity
        self.phase = RoundPhase.JOINING
        return True

    def close_registration(self) -> bool:
        """Close registration and assign clusters.

        Returns False if quorum not met.
        """
        if self._registration_closed:
            return True

        n = len(self._peers)
        if n < self.config.min_peers:
            logger.error(
                "Round %s: only %d peers (need %d). Round FAILED.",
                self.config.round_id, n, self.config.min_peers,
            )
            self.phase = RoundPhase.FAILED
            return False

        self._registration_closed = True

        # Build cluster topology.
        self._cluster_manager = ClusterManager(
            self.config.round_id,
            ClusterConfig(cluster_size=self.config.cluster_size),
        )
        for pid, cap in self._capacities.items():
            self._cluster_manager.register_peer(cap)
        self._cluster_manager.finalize()

        self.phase = RoundPhase.ASSIGNED
        logger.info(
            "Round %s: registration closed. %d peers assigned to clusters.",
            self.config.round_id, n,
        )
        return True

    def mark_dropout(self, peer_id: str):
        """Mark a peer as dropped out (disconnected, timed out, etc.)."""
        if peer_id in self._peers:
            self._peers[peer_id].dropped = True
            logger.warning("Peer %s dropped from round %s", peer_id, self.config.round_id)

    def submit_gradient(
        self,
        peer_id: str,
        gradients: Dict[str, torch.Tensor],
        merkle_root: str = "",
    ) -> bool:
        """Submit a peer's computed gradients.

        Returns False if the peer is not registered or already submitted.
        """
        if peer_id not in self._peers:
            logger.warning("Unknown peer %s submitted gradients", peer_id)
            return False

        ps = self._peers[peer_id]
        if ps.dropped:
            logger.warning("Dropped peer %s tried to submit gradients", peer_id)
            return False
        if ps.gradient_submitted:
            logger.warning("Peer %s already submitted gradients", peer_id)
            return False

        ps.gradient_submitted = True
        ps.gradient_submitted_at = time.monotonic()
        ps.merkle_root = merkle_root
        self._gradients[peer_id] = gradients

        self.phase = RoundPhase.COMPUTING
        return True

    def check_deadline(self) -> List[str]:
        """Check gradient deadline and mark stragglers as dropped.

        Returns list of peer IDs that were marked as stragglers.
        """
        now = time.monotonic()
        deadline = self._start_time + self.config.gradient_deadline_secs
        stragglers = []

        if now < deadline:
            return stragglers

        for pid, ps in self._peers.items():
            if not ps.gradient_submitted and not ps.dropped:
                ps.dropped = True
                stragglers.append(pid)

        if stragglers:
            logger.warning(
                "Round %s: %d stragglers timed out: %s",
                self.config.round_id, len(stragglers),
                stragglers[:5],
            )

        return stragglers

    def can_finalize(self) -> bool:
        """Check if we have enough gradients to finalize."""
        n_registered = len([p for p in self._peers.values() if not p.dropped])
        n_submitted = self.n_submitted
        if n_registered == 0:
            return False
        quorum = int(n_registered * self.config.min_quorum_ratio)
        return n_submitted >= max(quorum, self.config.min_peers)

    def finalize(self) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregate submitted gradients and return the result.

        Uses Byzantine-resilient aggregation to filter out poisoned gradients.
        Returns None if quorum not met.
        """
        if not self.can_finalize():
            active = len(self.active_peers)
            submitted = self.n_submitted
            logger.error(
                "Round %s: cannot finalize. %d active, %d submitted, need %.0f%%",
                self.config.round_id, active, submitted,
                self.config.min_quorum_ratio * 100,
            )
            self.phase = RoundPhase.FAILED
            return None

        self.phase = RoundPhase.AGGREGATING

        # Collect gradients from non-dropped peers who submitted.
        submitted_peers = [
            pid for pid, ps in self._peers.items()
            if ps.gradient_submitted and not ps.dropped
        ]
        grad_list = [self._gradients[pid] for pid in submitted_peers]

        logger.info(
            "Round %s: aggregating %d/%d peer gradients (method: %s)",
            self.config.round_id,
            len(grad_list),
            len(self._peers),
            self.config.byzantine_config.method.name,
        )

        # Byzantine-resilient aggregation.
        result = robust_aggregate(grad_list, self.config.byzantine_config)

        # Score peers for reputation tracking.
        scores = score_gradients(grad_list, self.config.byzantine_config.max_byzantine)
        for pid, score in zip(submitted_peers, scores):
            self._peers[pid].byzantine_score = score

        self._result = result
        self.phase = RoundPhase.VERIFYING

        return result

    def verify_consistency(self, expected_merkle_root: str) -> List[str]:
        """Verify which peers have consistent weights after aggregation.

        Returns list of peer IDs with mismatched Merkle roots.
        """
        mismatched = []
        for pid, ps in self._peers.items():
            if ps.gradient_submitted and ps.merkle_root and ps.merkle_root != expected_merkle_root:
                mismatched.append(pid)

        if mismatched:
            logger.warning(
                "Round %s: %d peers have divergent Merkle roots: %s",
                self.config.round_id, len(mismatched), mismatched[:5],
            )

        self.phase = RoundPhase.COMPLETED
        return mismatched

    def summary(self) -> Dict:
        """Round summary for logging and monitoring."""
        n_total = len(self._peers)
        n_dropped = sum(1 for p in self._peers.values() if p.dropped)
        n_submitted = self.n_submitted

        # Top 5 most suspicious peers (highest Byzantine score).
        scored = [(ps.byzantine_score, ps.peer_id) for ps in self._peers.values() if ps.gradient_submitted]
        scored.sort(reverse=True)
        suspicious = [(pid, f"{score:.2f}") for score, pid in scored[:5]]

        return {
            "round_id": self.config.round_id,
            "phase": self.phase.name,
            "total_peers": n_total,
            "dropped": n_dropped,
            "submitted": n_submitted,
            "dropout_rate": n_dropped / n_total if n_total > 0 else 0,
            "submission_rate": n_submitted / n_total if n_total > 0 else 0,
            "aggregation_method": self.config.byzantine_config.method.name,
            "most_suspicious": suspicious,
        }
