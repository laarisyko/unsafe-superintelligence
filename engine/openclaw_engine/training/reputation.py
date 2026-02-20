"""Peer reputation system for decentralized trust.

Tracks peer behavior across training rounds to build a reputation score.
Byzantine detection scores from each round feed into a long-term reputation
that determines:
    - Whether a peer is allowed to participate in future rounds
    - How much weight a peer's gradient gets during aggregation
    - Whether a peer is flagged for investigation / banning

Reputation updates are local -- each node maintains its own view. Over time,
gossip spreads reputation signals so the network converges on a shared view
of trustworthiness.

Scoring model:
    - Base score: 0.5 (neutral)
    - Good round completion: +0.05 (capped at 1.0)
    - Straggler/dropout: -0.1
    - Byzantine detection (high Krum score): -0.15
    - Consistent honest behavior: bonus decay toward 1.0
    - Score < 0.1: peer is banned from future rounds
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PeerRecord:
    """Long-term record for a single peer."""

    peer_id: str
    score: float = 0.5
    rounds_participated: int = 0
    rounds_completed: int = 0
    rounds_dropped: int = 0
    rounds_byzantine: int = 0  # Flagged as suspicious by Krum scoring
    total_byzantine_score: float = 0.0  # Cumulative Krum distance score
    last_seen: float = 0.0
    first_seen: float = 0.0
    banned: bool = False
    ban_reason: str = ""

    @property
    def completion_rate(self) -> float:
        if self.rounds_participated == 0:
            return 0.0
        return self.rounds_completed / self.rounds_participated

    @property
    def avg_byzantine_score(self) -> float:
        if self.rounds_completed == 0:
            return 0.0
        return self.total_byzantine_score / self.rounds_completed


# Thresholds.
BAN_THRESHOLD = 0.1
SUSPECT_THRESHOLD = 0.25
GOOD_THRESHOLD = 0.7

# Score deltas.
REWARD_COMPLETE = 0.05
PENALTY_DROPOUT = 0.1
PENALTY_BYZANTINE = 0.15
PENALTY_REPEATED_BYZANTINE = 0.25


class ReputationTracker:
    """Tracks peer reputation across training rounds.

    Thread-safe: all mutations go through methods that can be wrapped with locks
    by the caller.
    """

    def __init__(self):
        self._peers: Dict[str, PeerRecord] = {}

    def _get_or_create(self, peer_id: str) -> PeerRecord:
        if peer_id not in self._peers:
            now = time.time()
            self._peers[peer_id] = PeerRecord(
                peer_id=peer_id, first_seen=now, last_seen=now
            )
        return self._peers[peer_id]

    def record_round_result(
        self,
        peer_id: str,
        completed: bool,
        byzantine_score: float = 0.0,
        is_byzantine_outlier: bool = False,
    ):
        """Record a peer's behavior in a training round.

        Args:
            peer_id: The peer's identifier.
            completed: Whether the peer submitted gradients before the deadline.
            byzantine_score: The peer's Krum distance score (lower = more honest).
            is_byzantine_outlier: Whether this peer was flagged as an outlier.
        """
        record = self._get_or_create(peer_id)
        record.rounds_participated += 1
        record.last_seen = time.time()

        if not completed:
            # Dropout / straggler.
            record.rounds_dropped += 1
            record.score = max(0.0, record.score - PENALTY_DROPOUT)
            logger.debug("Peer %s dropped out, score: %.2f", peer_id, record.score)
        elif is_byzantine_outlier:
            # Flagged as Byzantine.
            record.rounds_byzantine += 1
            record.rounds_completed += 1
            record.total_byzantine_score += byzantine_score

            # Harsher penalty for repeated Byzantine behavior.
            if record.rounds_byzantine >= 3:
                penalty = PENALTY_REPEATED_BYZANTINE
            else:
                penalty = PENALTY_BYZANTINE
            record.score = max(0.0, record.score - penalty)
            logger.warning(
                "Peer %s flagged as Byzantine (count=%d), score: %.2f",
                peer_id, record.rounds_byzantine, record.score,
            )
        else:
            # Honest completion.
            record.rounds_completed += 1
            record.total_byzantine_score += byzantine_score
            record.score = min(1.0, record.score + REWARD_COMPLETE)

        # Auto-ban if score drops below threshold.
        if record.score < BAN_THRESHOLD and not record.banned:
            record.banned = True
            record.ban_reason = (
                f"Score {record.score:.2f} below threshold {BAN_THRESHOLD}. "
                f"Byzantine count: {record.rounds_byzantine}, "
                f"Dropout count: {record.rounds_dropped}"
            )
            logger.warning("BANNED peer %s: %s", peer_id, record.ban_reason)

    def record_round_batch(
        self,
        submitted_peers: List[str],
        dropped_peers: List[str],
        byzantine_scores: Dict[str, float],
        outlier_peer_ids: List[str],
    ):
        """Batch-update reputation after a training round.

        Args:
            submitted_peers: Peers that submitted gradients.
            dropped_peers: Peers that dropped out / were stragglers.
            byzantine_scores: Krum distance scores for submitted peers.
            outlier_peer_ids: Peers flagged as Byzantine outliers.
        """
        outlier_set = set(outlier_peer_ids)

        for pid in submitted_peers:
            self.record_round_result(
                pid,
                completed=True,
                byzantine_score=byzantine_scores.get(pid, 0.0),
                is_byzantine_outlier=pid in outlier_set,
            )

        for pid in dropped_peers:
            self.record_round_result(pid, completed=False)

    def is_allowed(self, peer_id: str) -> bool:
        """Check if a peer is allowed to participate in training rounds."""
        record = self._peers.get(peer_id)
        if record is None:
            return True  # New peers are allowed by default.
        return not record.banned

    def get_score(self, peer_id: str) -> float:
        """Get a peer's current reputation score."""
        record = self._peers.get(peer_id)
        return record.score if record else 0.5

    def get_record(self, peer_id: str) -> Optional[PeerRecord]:
        """Get a peer's full reputation record."""
        return self._peers.get(peer_id)

    def ranked_peers(self) -> List[Tuple[str, float]]:
        """Get all peers ranked by reputation score (highest first)."""
        peers = [(r.peer_id, r.score) for r in self._peers.values()]
        peers.sort(key=lambda x: x[1], reverse=True)
        return peers

    def suspect_peers(self) -> List[PeerRecord]:
        """Get peers with scores below the suspect threshold."""
        return [
            r for r in self._peers.values()
            if r.score < SUSPECT_THRESHOLD and not r.banned
        ]

    def banned_peers(self) -> List[PeerRecord]:
        """Get all banned peers."""
        return [r for r in self._peers.values() if r.banned]

    def detect_outliers(
        self,
        peer_ids: List[str],
        byzantine_scores: List[float],
        threshold_ratio: float = 3.0,
    ) -> List[str]:
        """Detect Byzantine outliers based on Krum scores.

        Peers with Krum scores > threshold_ratio * median are flagged.

        Args:
            peer_ids: List of peer IDs.
            byzantine_scores: Corresponding Krum scores.
            threshold_ratio: How many times the median score to flag.

        Returns:
            List of peer IDs flagged as outliers.
        """
        if len(byzantine_scores) < 3:
            return []

        sorted_scores = sorted(byzantine_scores)
        median = sorted_scores[len(sorted_scores) // 2]
        threshold = median * threshold_ratio

        outliers = []
        for pid, score in zip(peer_ids, byzantine_scores):
            if score > threshold:
                outliers.append(pid)

        return outliers

    def summary(self) -> Dict:
        """Summary statistics for the reputation table."""
        total = len(self._peers)
        if total == 0:
            return {"total_peers": 0}

        scores = [r.score for r in self._peers.values()]
        banned = sum(1 for r in self._peers.values() if r.banned)
        suspect = sum(1 for r in self._peers.values()
                      if r.score < SUSPECT_THRESHOLD and not r.banned)
        good = sum(1 for r in self._peers.values() if r.score >= GOOD_THRESHOLD)

        return {
            "total_peers": total,
            "avg_score": sum(scores) / total,
            "min_score": min(scores),
            "max_score": max(scores),
            "good_peers": good,
            "suspect_peers": suspect,
            "banned_peers": banned,
        }
