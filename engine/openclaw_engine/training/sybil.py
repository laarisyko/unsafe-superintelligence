"""Sybil resistance via proof-of-work admission control.

In an open network, an attacker can spin up thousands of fake peers to:
    1. Overwhelm Byzantine defenses (need >50% honest for safety guarantees)
    2. Slow down training by flooding with garbage gradients
    3. DoS the gossip layer with fake messages

This module implements a lightweight proof-of-work challenge that each peer
must solve before being admitted to a training round. The difficulty adjusts
based on network size to maintain a target join time.

Protocol:
    1. Round proposer broadcasts: round_id, model_id, difficulty
    2. Each peer computes: SHA256(round_id || peer_id || nonce) < target
    3. Peer broadcasts the solution nonce as part of its join message
    4. All peers verify the solution deterministically (O(1) per peer)

The work is intentionally lightweight (~100ms-1s per join) -- enough to
make Sybil attacks expensive at scale (1000 fake peers = 1000x the work)
without burdening honest participants.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PowChallenge:
    """A proof-of-work challenge for round admission."""

    round_id: str
    difficulty: int  # Number of leading zero bits required.
    timestamp: float = 0.0

    @property
    def target(self) -> int:
        """The hash must be below this target value."""
        if self.difficulty <= 0:
            return (1 << 256) - 1  # Any hash is valid.
        return (1 << 256) >> self.difficulty


@dataclass
class PowSolution:
    """A solved proof-of-work challenge."""

    round_id: str
    peer_id: str
    nonce: int
    hash_hex: str
    attempts: int = 0
    solve_time_ms: float = 0.0


# Difficulty presets.
DIFFICULTY_TRIVIAL = 8    # ~256 hashes, ~0.1ms
DIFFICULTY_EASY = 16      # ~65K hashes, ~30ms
DIFFICULTY_MEDIUM = 20    # ~1M hashes, ~500ms
DIFFICULTY_HARD = 24      # ~16M hashes, ~8s


def compute_pow_hash(round_id: str, peer_id: str, nonce: int) -> bytes:
    """Compute the PoW hash for a given nonce.

    Hash = SHA256("openclaw-pow-v1:" || round_id || ":" || peer_id || ":" || nonce_le_bytes)
    """
    h = hashlib.sha256()
    h.update(b"openclaw-pow-v1:")
    h.update(round_id.encode())
    h.update(b":")
    h.update(peer_id.encode())
    h.update(b":")
    h.update(struct.pack("<Q", nonce))
    return h.digest()


def hash_to_int(hash_bytes: bytes) -> int:
    """Convert a 32-byte hash to an integer for comparison with target."""
    return int.from_bytes(hash_bytes, byteorder="big")


def solve(challenge: PowChallenge, peer_id: str) -> PowSolution:
    """Solve a proof-of-work challenge by brute-force nonce search.

    Args:
        challenge: The PoW challenge to solve.
        peer_id: This peer's identifier.

    Returns:
        PowSolution with the valid nonce.
    """
    target = challenge.target
    start = time.monotonic()
    nonce = 0

    # Start from a random offset to avoid all peers trying the same nonces.
    nonce = int.from_bytes(os.urandom(4), byteorder="little")
    start_nonce = nonce

    while True:
        h = compute_pow_hash(challenge.round_id, peer_id, nonce)
        if hash_to_int(h) < target:
            elapsed = (time.monotonic() - start) * 1000
            attempts = nonce - start_nonce + 1
            return PowSolution(
                round_id=challenge.round_id,
                peer_id=peer_id,
                nonce=nonce,
                hash_hex=h.hex(),
                attempts=attempts,
                solve_time_ms=elapsed,
            )
        nonce += 1


def verify(challenge: PowChallenge, solution: PowSolution) -> bool:
    """Verify a proof-of-work solution. O(1) -- single hash computation.

    Args:
        challenge: The original challenge.
        solution: The claimed solution.

    Returns:
        True if the solution is valid.
    """
    if solution.round_id != challenge.round_id:
        return False

    h = compute_pow_hash(challenge.round_id, solution.peer_id, solution.nonce)
    return hash_to_int(h) < challenge.target


def auto_difficulty(n_peers: int, target_solve_ms: float = 500.0) -> int:
    """Auto-compute difficulty based on network size and target solve time.

    Larger networks use higher difficulty to make Sybil attacks proportionally
    more expensive. The target solve time is what an honest peer should expect.

    Heuristic:
        - <10 peers: trivial (testing/development)
        - 10-100: easy (~30ms)
        - 100-1000: medium (~500ms)
        - 1000+: hard (~8s)
    """
    if n_peers < 10:
        return DIFFICULTY_TRIVIAL
    elif n_peers < 100:
        return DIFFICULTY_EASY
    elif n_peers < 1000:
        return DIFFICULTY_MEDIUM
    else:
        return DIFFICULTY_HARD


class AdmissionController:
    """Manages proof-of-work admission for training rounds.

    Works alongside the reputation system: peers with high reputation
    get reduced difficulty (they've already proven they're honest).
    """

    def __init__(
        self,
        base_difficulty: int = DIFFICULTY_EASY,
        reputation_discount: float = 0.5,
    ):
        """
        Args:
            base_difficulty: Default difficulty for new/unknown peers.
            reputation_discount: Score threshold above which difficulty is halved.
        """
        self.base_difficulty = base_difficulty
        self.reputation_discount = reputation_discount
        self._challenges: dict[str, PowChallenge] = {}

    def create_challenge(
        self,
        round_id: str,
        n_peers: int = 0,
    ) -> PowChallenge:
        """Create a PoW challenge for a training round."""
        difficulty = auto_difficulty(n_peers) if n_peers > 0 else self.base_difficulty
        challenge = PowChallenge(
            round_id=round_id,
            difficulty=difficulty,
            timestamp=time.time(),
        )
        self._challenges[round_id] = challenge
        return challenge

    def get_challenge(self, round_id: str) -> Optional[PowChallenge]:
        """Get the challenge for a round."""
        return self._challenges.get(round_id)

    def verify_admission(
        self,
        round_id: str,
        solution: PowSolution,
        reputation_score: float = 0.5,
    ) -> bool:
        """Verify a peer's admission to a round.

        High-reputation peers get a reduced difficulty check.
        """
        challenge = self._challenges.get(round_id)
        if challenge is None:
            logger.warning("No challenge found for round %s", round_id)
            return False

        # High-rep peers get easier challenge.
        if reputation_score >= self.reputation_discount:
            easier = PowChallenge(
                round_id=challenge.round_id,
                difficulty=max(0, challenge.difficulty - 4),  # 4 fewer bits
                timestamp=challenge.timestamp,
            )
            return verify(easier, solution)

        return verify(challenge, solution)

    def solve_challenge(self, round_id: str, peer_id: str) -> Optional[PowSolution]:
        """Solve the challenge for a round (convenience method for local peer)."""
        challenge = self._challenges.get(round_id)
        if challenge is None:
            return None
        return solve(challenge, peer_id)
