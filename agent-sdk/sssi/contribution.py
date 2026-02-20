"""Contribution tracking for the SSSI network.

Tracks what each peer contributes (compute, training, inference serving) and
provides cryptographic proofs that peers present when making requests.
Contributing peers get unlimited access; non-contributors are rate-limited.

Contribution is measured in *credits*:
  - Serving an inference request:     1 credit per request served
  - Completing a training round:     10 credits per round
  - Hosting a model shard:            1 credit per minute of uptime
  - Voting on architecture proposals:  1 credit per vote

Credits decay over time (half-life: 24 hours) to incentivize sustained
participation rather than one-time bursts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Credit awards
CREDITS_INFERENCE_SERVED = 1
CREDITS_TRAINING_ROUND = 10
CREDITS_SHARD_HOSTING_PER_MIN = 1
CREDITS_ARCHITECTURE_VOTE = 1

# Decay: credits halve every 24 hours
CREDIT_HALF_LIFE_SECS = 86400

# Threshold: peers with >= this many effective credits are "contributors"
CONTRIBUTOR_THRESHOLD = 5


@dataclass
class ContributionRecord:
    """A single contribution event."""
    event_type: str          # "inference_served", "training_round", "shard_hosting", "vote"
    credits: float
    timestamp: float
    round_id: str = ""
    model_id: str = ""


@dataclass
class PeerContribution:
    """Tracks all contributions from a single peer."""
    peer_id: str
    records: list = field(default_factory=list)
    _raw_credits: float = 0.0
    _last_update: float = 0.0

    def add(self, event_type: str, credits: float, round_id: str = "", model_id: str = ""):
        now = time.time()
        self.records.append(ContributionRecord(
            event_type=event_type,
            credits=credits,
            timestamp=now,
            round_id=round_id,
            model_id=model_id,
        ))
        self._raw_credits += credits
        self._last_update = now

    @property
    def effective_credits(self) -> float:
        """Credits with time decay applied."""
        now = time.time()
        total = 0.0
        for r in self.records:
            age = now - r.timestamp
            decay = math.pow(0.5, age / CREDIT_HALF_LIFE_SECS)
            total += r.credits * decay
        return total

    @property
    def is_contributor(self) -> bool:
        return self.effective_credits >= CONTRIBUTOR_THRESHOLD

    @property
    def tier(self) -> str:
        return "contributor" if self.is_contributor else "free"

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "tier": self.tier,
            "effective_credits": round(self.effective_credits, 2),
            "total_events": len(self.records),
            "is_contributor": self.is_contributor,
        }


class ContributionTracker:
    """Tracks contributions across all peers in the local node.

    The tracker maintains a local ledger. Peers gossip their contribution
    proofs, and the node verifies them against observed network activity.
    """

    def __init__(self):
        self._peers: Dict[str, PeerContribution] = {}

    def get_or_create(self, peer_id: str) -> PeerContribution:
        if peer_id not in self._peers:
            self._peers[peer_id] = PeerContribution(peer_id=peer_id)
        return self._peers[peer_id]

    def record_inference_served(self, peer_id: str, model_id: str = ""):
        pc = self.get_or_create(peer_id)
        pc.add("inference_served", CREDITS_INFERENCE_SERVED, model_id=model_id)
        logger.debug("Peer %s earned %d credit(s) for inference", peer_id, CREDITS_INFERENCE_SERVED)

    def record_training_round(self, peer_id: str, round_id: str, model_id: str = ""):
        pc = self.get_or_create(peer_id)
        pc.add("training_round", CREDITS_TRAINING_ROUND, round_id=round_id, model_id=model_id)
        logger.debug("Peer %s earned %d credit(s) for training round %s", peer_id, CREDITS_TRAINING_ROUND, round_id)

    def record_shard_hosting(self, peer_id: str, minutes: float, model_id: str = ""):
        credits = minutes * CREDITS_SHARD_HOSTING_PER_MIN
        pc = self.get_or_create(peer_id)
        pc.add("shard_hosting", credits, model_id=model_id)

    def record_vote(self, peer_id: str, proposal_id: str = ""):
        pc = self.get_or_create(peer_id)
        pc.add("vote", CREDITS_ARCHITECTURE_VOTE, round_id=proposal_id)

    def is_contributor(self, peer_id: str) -> bool:
        pc = self._peers.get(peer_id)
        if pc is None:
            return False
        return pc.is_contributor

    def get_tier(self, peer_id: str) -> str:
        pc = self._peers.get(peer_id)
        if pc is None:
            return "free"
        return pc.tier

    def get_credits(self, peer_id: str) -> float:
        pc = self._peers.get(peer_id)
        if pc is None:
            return 0.0
        return pc.effective_credits

    def get_quota(self, peer_id: str) -> dict:
        """Return the peer's current quota/tier info."""
        pc = self._peers.get(peer_id)
        if pc is None:
            return {
                "peer_id": peer_id,
                "tier": "free",
                "effective_credits": 0.0,
                "is_contributor": False,
                "credits_needed": CONTRIBUTOR_THRESHOLD,
            }
        info = pc.to_dict()
        if not pc.is_contributor:
            info["credits_needed"] = round(CONTRIBUTOR_THRESHOLD - pc.effective_credits, 2)
        else:
            info["credits_needed"] = 0
        return info

    def build_contribution_proof(self, peer_id: str) -> dict:
        """Build a signed contribution summary that the peer can attach to requests.

        The proof includes the peer's credit balance and a hash digest.
        In production this would be signed with the peer's private key.
        """
        pc = self._peers.get(peer_id)
        credits = pc.effective_credits if pc else 0.0
        tier = pc.tier if pc else "free"
        events = len(pc.records) if pc else 0

        payload = {
            "peer_id": peer_id,
            "tier": tier,
            "effective_credits": round(credits, 2),
            "total_events": events,
            "timestamp": time.time(),
        }
        # Simple HMAC-style digest (in production: Ed25519 signature)
        digest_input = json.dumps(payload, sort_keys=True).encode()
        payload["digest"] = hashlib.sha256(digest_input).hexdigest()[:16]
        return payload

    def leaderboard(self, top_n: int = 20) -> list:
        """Return top contributors sorted by effective credits."""
        entries = []
        for pc in self._peers.values():
            entries.append(pc.to_dict())
        entries.sort(key=lambda x: x["effective_credits"], reverse=True)
        return entries[:top_n]
