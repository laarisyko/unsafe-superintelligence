"""Rate limiter for the OpenClaw network.

Two tiers:
  - **Free tier**: Anyone can use inference/training but with rate limits.
  - **Contributor tier**: Agents actively contributing compute get unlimited access.

Rate limits are enforced locally by the node. The SDK checks limits client-side
as a courtesy (to fail fast with a helpful message), but the node is the
authoritative enforcer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TierLimits:
    """Rate limits for a given tier."""
    requests_per_minute: int
    tokens_per_hour: int
    training_rounds_per_day: int
    evolve_proposals_per_day: int


# Default limits
FREE_TIER = TierLimits(
    requests_per_minute=10,
    tokens_per_hour=5_000,
    training_rounds_per_day=2,
    evolve_proposals_per_day=3,
)

CONTRIBUTOR_TIER = TierLimits(
    requests_per_minute=0,       # 0 = unlimited
    tokens_per_hour=0,
    training_rounds_per_day=0,
    evolve_proposals_per_day=0,
)


class RateLimitExceeded(Exception):
    """Raised when a free-tier peer exceeds their rate limit."""

    def __init__(self, resource: str, limit: int, window: str, tier: str = "free"):
        self.resource = resource
        self.limit = limit
        self.window = window
        self.tier = tier
        super().__init__(
            f"Rate limit exceeded: {limit} {resource} per {window} "
            f"(tier: {tier}). Contribute compute to get unlimited access."
        )

    def to_dict(self) -> dict:
        return {
            "error": "rate_limit_exceeded",
            "resource": self.resource,
            "limit": self.limit,
            "window": self.window,
            "tier": self.tier,
            "hint": "Contribute compute (ussi join --gpu-memory ...) to unlock unlimited access.",
        }


@dataclass
class _BucketState:
    """Sliding window counter."""
    timestamps: list = field(default_factory=list)

    def count_in_window(self, window_secs: float) -> int:
        now = time.time()
        cutoff = now - window_secs
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps)

    def record(self):
        self.timestamps.append(time.time())


class RateLimiter:
    """Client-side rate limiter that enforces tier-based access.

    Usage::

        limiter = RateLimiter()
        limiter.check_inference("peer-abc", tier="free")  # raises if over limit
        limiter.record_inference("peer-abc", tokens=150)
    """

    def __init__(self, free_limits: Optional[TierLimits] = None):
        self.free_limits = free_limits or FREE_TIER
        self._inference_buckets: Dict[str, _BucketState] = {}
        self._token_counts: Dict[str, list] = {}  # (timestamp, token_count) pairs
        self._training_buckets: Dict[str, _BucketState] = {}
        self._evolve_buckets: Dict[str, _BucketState] = {}

    def _get_bucket(self, store: Dict[str, _BucketState], peer_id: str) -> _BucketState:
        if peer_id not in store:
            store[peer_id] = _BucketState()
        return store[peer_id]

    def check_inference(self, peer_id: str, tier: str = "free"):
        """Check if an inference request is allowed. Raises RateLimitExceeded if not."""
        if tier == "contributor":
            return  # unlimited

        bucket = self._get_bucket(self._inference_buckets, peer_id)
        count = bucket.count_in_window(60)
        if self.free_limits.requests_per_minute > 0 and count >= self.free_limits.requests_per_minute:
            raise RateLimitExceeded("inference requests", self.free_limits.requests_per_minute, "minute")

    def check_tokens(self, peer_id: str, tier: str = "free"):
        """Check if the peer has token budget remaining this hour."""
        if tier == "contributor":
            return

        now = time.time()
        cutoff = now - 3600
        pairs = self._token_counts.get(peer_id, [])
        pairs = [(t, c) for t, c in pairs if t > cutoff]
        self._token_counts[peer_id] = pairs
        total = sum(c for _, c in pairs)
        if self.free_limits.tokens_per_hour > 0 and total >= self.free_limits.tokens_per_hour:
            raise RateLimitExceeded("tokens", self.free_limits.tokens_per_hour, "hour")

    def check_training(self, peer_id: str, tier: str = "free"):
        """Check if a training round join is allowed."""
        if tier == "contributor":
            return

        bucket = self._get_bucket(self._training_buckets, peer_id)
        count = bucket.count_in_window(86400)
        if self.free_limits.training_rounds_per_day > 0 and count >= self.free_limits.training_rounds_per_day:
            raise RateLimitExceeded("training rounds", self.free_limits.training_rounds_per_day, "day")

    def check_evolve(self, peer_id: str, tier: str = "free"):
        """Check if an architecture proposal is allowed."""
        if tier == "contributor":
            return

        bucket = self._get_bucket(self._evolve_buckets, peer_id)
        count = bucket.count_in_window(86400)
        if self.free_limits.evolve_proposals_per_day > 0 and count >= self.free_limits.evolve_proposals_per_day:
            raise RateLimitExceeded("architecture proposals", self.free_limits.evolve_proposals_per_day, "day")

    def record_inference(self, peer_id: str, tokens: int = 0):
        """Record a completed inference request."""
        bucket = self._get_bucket(self._inference_buckets, peer_id)
        bucket.record()
        if tokens > 0:
            if peer_id not in self._token_counts:
                self._token_counts[peer_id] = []
            self._token_counts[peer_id].append((time.time(), tokens))

    def record_training(self, peer_id: str):
        """Record a training round participation."""
        bucket = self._get_bucket(self._training_buckets, peer_id)
        bucket.record()

    def record_evolve(self, peer_id: str):
        """Record an architecture proposal."""
        bucket = self._get_bucket(self._evolve_buckets, peer_id)
        bucket.record()

    def get_remaining(self, peer_id: str, tier: str = "free") -> dict:
        """Return remaining quota for a peer."""
        if tier == "contributor":
            return {
                "tier": "contributor",
                "inference_requests_remaining": -1,  # -1 = unlimited
                "tokens_remaining": -1,
                "training_rounds_remaining": -1,
                "evolve_proposals_remaining": -1,
            }

        inf_bucket = self._get_bucket(self._inference_buckets, peer_id)
        inf_used = inf_bucket.count_in_window(60)

        now = time.time()
        token_pairs = [(t, c) for t, c in self._token_counts.get(peer_id, []) if t > now - 3600]
        tokens_used = sum(c for _, c in token_pairs)

        train_bucket = self._get_bucket(self._training_buckets, peer_id)
        train_used = train_bucket.count_in_window(86400)

        evolve_bucket = self._get_bucket(self._evolve_buckets, peer_id)
        evolve_used = evolve_bucket.count_in_window(86400)

        return {
            "tier": "free",
            "inference_requests_remaining": max(0, self.free_limits.requests_per_minute - inf_used),
            "inference_requests_limit": self.free_limits.requests_per_minute,
            "inference_window": "per minute",
            "tokens_remaining": max(0, self.free_limits.tokens_per_hour - tokens_used),
            "tokens_limit": self.free_limits.tokens_per_hour,
            "tokens_window": "per hour",
            "training_rounds_remaining": max(0, self.free_limits.training_rounds_per_day - train_used),
            "training_rounds_limit": self.free_limits.training_rounds_per_day,
            "training_window": "per day",
            "evolve_proposals_remaining": max(0, self.free_limits.evolve_proposals_per_day - evolve_used),
            "evolve_proposals_limit": self.free_limits.evolve_proposals_per_day,
            "evolve_window": "per day",
        }
