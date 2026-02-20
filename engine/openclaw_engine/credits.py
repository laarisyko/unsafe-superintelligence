"""Credit system: non-transferable incentive credits for the network.

Credits are the internal incentive mechanism for OpenClaw. They are:
- EARNED by contributing (compute, data, voting, uptime)
- SPENT on inference (priority access to the model)
- NON-TRANSFERABLE (tied to peer identity, can't be traded)
- DECAYING (recent contributors get priority over historical ones)

This is NOT a cryptocurrency. There is no blockchain, no token, no exchange.
Credits are simply a local accounting system that rewards participation and
gates access to scarce inference resources.

The reward for helping train the AI is... the AI itself.

Earning rates:
    - Training round completed: 10 credits (base)
    - Reputation bonus: up to 2x for high-reputation peers
    - Data contribution: 5 credits per MB of training data
    - Vote participation: 2 credits per architecture vote
    - Uptime streak: 1.5x multiplier after 24h continuous

Spending rates:
    - Free tier: 10 requests/hour (no credits needed)
    - Standard inference: 1 credit per request
    - Priority inference: 5 credits (skip the queue)
    - Long generation (>256 tokens): 2 credits

Decay:
    - Credits decay with a configurable half-life (default 7 days)
    - This prevents hoarding and rewards recent contribution
    - Decay is applied lazily (on balance check, not on a timer)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CreditAction(str, Enum):
    """Types of credit-earning actions."""
    TRAINING_ROUND = "training_round"
    DATA_CONTRIBUTION = "data_contribution"
    VOTE_CAST = "vote_cast"
    UPTIME_BONUS = "uptime_bonus"
    # Spending actions.
    INFERENCE_STANDARD = "inference_standard"
    INFERENCE_PRIORITY = "inference_priority"
    INFERENCE_LONG = "inference_long"


class InferenceTier(str, Enum):
    """Inference access tiers based on credit balance."""
    FREE = "free"            # No credits needed, rate-limited.
    CONTRIBUTOR = "contributor"  # Has credits, standard rate.
    PRIORITY = "priority"     # Spends extra for priority queue.


@dataclass
class CreditConfig:
    """Configuration for the credit system."""
    # Earning rates.
    earn_training_round: float = 10.0
    earn_data_per_mb: float = 5.0
    earn_vote: float = 2.0
    earn_uptime_bonus: float = 5.0

    # Reputation multiplier range.
    reputation_min_multiplier: float = 0.5  # Bad rep earns less.
    reputation_max_multiplier: float = 2.0  # Great rep earns more.

    # Uptime streak multiplier.
    uptime_streak_hours: float = 24.0  # Hours for streak bonus.
    uptime_streak_multiplier: float = 1.5

    # Spending rates.
    cost_inference_standard: float = 1.0
    cost_inference_priority: float = 5.0
    cost_inference_long: float = 2.0  # >256 tokens.

    # Free tier.
    free_tier_requests_per_hour: int = 10

    # Decay.
    decay_half_life_secs: float = 7 * 24 * 3600  # 7 days.
    decay_enabled: bool = True

    # Starting credits for new peers (welcome bonus).
    welcome_bonus: float = 50.0


@dataclass
class CreditTransaction:
    """A single credit transaction (earn or spend)."""
    timestamp: float
    action: CreditAction
    amount: float  # Positive = earn, negative = spend.
    balance_after: float
    description: str = ""
    round_id: str = ""


@dataclass
class PeerCreditAccount:
    """Credit account for a single peer."""
    peer_id: str
    raw_balance: float = 0.0  # Balance before decay.
    last_decay_time: float = 0.0  # When decay was last applied.
    total_earned: float = 0.0
    total_spent: float = 0.0
    total_decayed: float = 0.0
    # Earning stats.
    rounds_contributed: int = 0
    data_contributed_mb: float = 0.0
    votes_cast: int = 0
    # Uptime tracking.
    connected_since: float = 0.0  # Timestamp when peer connected.
    uptime_streak_secs: float = 0.0
    # Free tier tracking.
    free_requests_this_hour: int = 0
    free_hour_start: float = 0.0
    # Transaction history (last N).
    transactions: List[CreditTransaction] = field(default_factory=list)

    @property
    def is_contributor(self) -> bool:
        """Whether this peer has ever contributed."""
        return self.total_earned > 0


@dataclass
class SpendResult:
    """Result of attempting to spend credits."""
    allowed: bool
    tier: InferenceTier
    credits_spent: float = 0.0
    balance_after: float = 0.0
    reason: str = ""
    wait_secs: float = 0.0  # If rate-limited, how long to wait.


class CreditLedger:
    """The credit ledger: tracks all peer credit balances.

    This is a local data structure — each peer maintains its own ledger.
    In a full deployment, credit proofs would be signed and gossiped
    so peers can verify each other's balances.
    """

    MAX_TRANSACTION_HISTORY = 200

    def __init__(self, config: Optional[CreditConfig] = None):
        self.config = config or CreditConfig()
        self._accounts: Dict[str, PeerCreditAccount] = {}
        self._total_credits_minted: float = 0.0
        self._total_credits_spent: float = 0.0
        self._total_credits_decayed: float = 0.0

    def _get_or_create(self, peer_id: str) -> PeerCreditAccount:
        """Get or create a credit account for a peer."""
        if peer_id not in self._accounts:
            now = time.time()
            account = PeerCreditAccount(
                peer_id=peer_id,
                raw_balance=self.config.welcome_bonus,
                last_decay_time=now,
                total_earned=self.config.welcome_bonus,
                connected_since=now,
                free_hour_start=now,
            )
            self._accounts[peer_id] = account
            self._total_credits_minted += self.config.welcome_bonus

            self._record_transaction(
                account,
                CreditAction.UPTIME_BONUS,
                self.config.welcome_bonus,
                "Welcome bonus for joining the network",
            )
            logger.info(
                "New credit account for %s: %.1f welcome credits",
                peer_id, self.config.welcome_bonus,
            )
        return self._accounts[peer_id]

    def get_balance(self, peer_id: str) -> float:
        """Get a peer's current credit balance (after decay)."""
        account = self._get_or_create(peer_id)
        self._apply_decay(account)
        return account.raw_balance

    def get_account(self, peer_id: str) -> PeerCreditAccount:
        """Get a peer's full credit account."""
        account = self._get_or_create(peer_id)
        self._apply_decay(account)
        return account

    # --- Earning ---

    def earn_training_round(
        self,
        peer_id: str,
        round_id: str = "",
        reputation_score: float = 0.5,
    ) -> float:
        """Award credits for completing a training round.

        Returns credits earned.
        """
        account = self._get_or_create(peer_id)
        self._apply_decay(account)

        # Base earning.
        base = self.config.earn_training_round

        # Reputation multiplier: 0.5x at score=0, 2.0x at score=1.0.
        rep_mult = (
            self.config.reputation_min_multiplier
            + (self.config.reputation_max_multiplier - self.config.reputation_min_multiplier)
            * reputation_score
        )

        # Uptime streak multiplier.
        streak_mult = self._uptime_multiplier(account)

        earned = base * rep_mult * streak_mult
        account.raw_balance += earned
        account.total_earned += earned
        account.rounds_contributed += 1
        self._total_credits_minted += earned

        self._record_transaction(
            account,
            CreditAction.TRAINING_ROUND,
            earned,
            f"Training round {round_id} (rep={reputation_score:.2f}, streak={streak_mult:.1f}x)",
            round_id=round_id,
        )

        return earned

    def earn_data_contribution(
        self,
        peer_id: str,
        data_size_bytes: int,
    ) -> float:
        """Award credits for contributing training data.

        Returns credits earned.
        """
        account = self._get_or_create(peer_id)
        self._apply_decay(account)

        mb = data_size_bytes / (1024 * 1024)
        earned = mb * self.config.earn_data_per_mb

        account.raw_balance += earned
        account.total_earned += earned
        account.data_contributed_mb += mb
        self._total_credits_minted += earned

        self._record_transaction(
            account,
            CreditAction.DATA_CONTRIBUTION,
            earned,
            f"Data contribution: {mb:.1f} MB",
        )

        return earned

    def earn_vote(self, peer_id: str, proposal_id: str = "") -> float:
        """Award credits for voting on an architecture proposal.

        Returns credits earned.
        """
        account = self._get_or_create(peer_id)
        self._apply_decay(account)

        earned = self.config.earn_vote
        account.raw_balance += earned
        account.total_earned += earned
        account.votes_cast += 1
        self._total_credits_minted += earned

        self._record_transaction(
            account,
            CreditAction.VOTE_CAST,
            earned,
            f"Architecture vote: {proposal_id}" if proposal_id else "Architecture vote",
        )

        return earned

    # --- Spending ---

    def check_inference(
        self,
        peer_id: str,
        priority: bool = False,
        long_generation: bool = False,
    ) -> SpendResult:
        """Check if a peer can make an inference request.

        Does NOT spend credits — call spend_inference() to actually spend.
        Returns a SpendResult indicating whether the request is allowed.
        """
        account = self._get_or_create(peer_id)
        self._apply_decay(account)

        # Priority inference: must have credits.
        if priority:
            cost = self.config.cost_inference_priority
            if long_generation:
                cost += self.config.cost_inference_long
            if account.raw_balance >= cost:
                return SpendResult(
                    allowed=True,
                    tier=InferenceTier.PRIORITY,
                    credits_spent=cost,
                    balance_after=account.raw_balance - cost,
                )
            return SpendResult(
                allowed=False,
                tier=InferenceTier.PRIORITY,
                reason=f"Insufficient credits: need {cost:.1f}, have {account.raw_balance:.1f}",
            )

        # Standard inference with credits.
        cost = self.config.cost_inference_standard
        if long_generation:
            cost += self.config.cost_inference_long
        if account.raw_balance >= cost:
            return SpendResult(
                allowed=True,
                tier=InferenceTier.CONTRIBUTOR,
                credits_spent=cost,
                balance_after=account.raw_balance - cost,
            )

        # Fall back to free tier.
        return self._check_free_tier(account)

    def spend_inference(
        self,
        peer_id: str,
        priority: bool = False,
        long_generation: bool = False,
    ) -> SpendResult:
        """Spend credits for an inference request.

        Returns a SpendResult. If allowed, credits are deducted.
        """
        result = self.check_inference(peer_id, priority, long_generation)
        if not result.allowed:
            return result

        account = self._accounts[peer_id]

        if result.tier == InferenceTier.FREE:
            # Free tier: just increment counter.
            account.free_requests_this_hour += 1
            return result

        # Deduct credits.
        account.raw_balance -= result.credits_spent
        account.total_spent += result.credits_spent
        self._total_credits_spent += result.credits_spent

        action = (
            CreditAction.INFERENCE_PRIORITY
            if priority
            else CreditAction.INFERENCE_STANDARD
        )
        self._record_transaction(
            account,
            action,
            -result.credits_spent,
            f"Inference ({result.tier.value})",
        )

        result.balance_after = account.raw_balance
        return result

    def _check_free_tier(self, account: PeerCreditAccount) -> SpendResult:
        """Check free tier rate limit."""
        now = time.time()

        # Reset hourly counter if hour has passed.
        if now - account.free_hour_start >= 3600:
            account.free_requests_this_hour = 0
            account.free_hour_start = now

        if account.free_requests_this_hour < self.config.free_tier_requests_per_hour:
            return SpendResult(
                allowed=True,
                tier=InferenceTier.FREE,
                credits_spent=0,
                balance_after=account.raw_balance,
            )

        # Rate limited.
        wait = 3600 - (now - account.free_hour_start)
        return SpendResult(
            allowed=False,
            tier=InferenceTier.FREE,
            reason="Free tier rate limit exceeded. Contribute to earn credits!",
            wait_secs=max(0, wait),
        )

    # --- Decay ---

    def _apply_decay(self, account: PeerCreditAccount):
        """Apply exponential decay to a peer's balance.

        Decay is applied lazily — only when the balance is checked.
        This is equivalent to continuous decay but cheaper to compute.
        """
        if not self.config.decay_enabled or self.config.decay_half_life_secs <= 0:
            return

        now = time.time()
        elapsed = now - account.last_decay_time
        if elapsed < 60:  # Don't decay more often than once per minute.
            return

        # Exponential decay: balance * (0.5 ^ (elapsed / half_life))
        decay_factor = math.pow(0.5, elapsed / self.config.decay_half_life_secs)
        old_balance = account.raw_balance
        account.raw_balance *= decay_factor
        decayed = old_balance - account.raw_balance
        account.total_decayed += decayed
        self._total_credits_decayed += decayed
        account.last_decay_time = now

    # --- Uptime ---

    def record_connect(self, peer_id: str):
        """Record that a peer has connected."""
        account = self._get_or_create(peer_id)
        account.connected_since = time.time()

    def record_disconnect(self, peer_id: str):
        """Record that a peer has disconnected."""
        account = self._get_or_create(peer_id)
        if account.connected_since > 0:
            account.uptime_streak_secs += time.time() - account.connected_since
        account.connected_since = 0.0

    def _uptime_multiplier(self, account: PeerCreditAccount) -> float:
        """Calculate uptime streak multiplier."""
        if account.connected_since <= 0:
            return 1.0

        streak = time.time() - account.connected_since + account.uptime_streak_secs
        streak_hours = streak / 3600

        if streak_hours >= self.config.uptime_streak_hours:
            return self.config.uptime_streak_multiplier
        return 1.0

    # --- Transactions ---

    def _record_transaction(
        self,
        account: PeerCreditAccount,
        action: CreditAction,
        amount: float,
        description: str,
        round_id: str = "",
    ):
        """Record a transaction in the account history."""
        tx = CreditTransaction(
            timestamp=time.time(),
            action=action,
            amount=amount,
            balance_after=account.raw_balance,
            description=description,
            round_id=round_id,
        )
        account.transactions.append(tx)

        # Trim history.
        if len(account.transactions) > self.MAX_TRANSACTION_HISTORY:
            account.transactions = account.transactions[-self.MAX_TRANSACTION_HISTORY:]

    # --- Queries ---

    def top_contributors(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N contributors by current balance."""
        balances = []
        for pid, account in self._accounts.items():
            self._apply_decay(account)
            balances.append((pid, account.raw_balance))
        balances.sort(key=lambda x: x[1], reverse=True)
        return balances[:n]

    def network_stats(self) -> Dict:
        """Get network-wide credit statistics."""
        total_accounts = len(self._accounts)
        if total_accounts == 0:
            return {
                "total_accounts": 0,
                "total_minted": 0,
                "total_spent": 0,
                "total_decayed": 0,
            }

        balances = [self.get_balance(pid) for pid in self._accounts]
        contributors = sum(
            1 for a in self._accounts.values() if a.is_contributor
        )

        return {
            "total_accounts": total_accounts,
            "total_contributors": contributors,
            "total_minted": round(self._total_credits_minted, 1),
            "total_spent": round(self._total_credits_spent, 1),
            "total_decayed": round(self._total_credits_decayed, 1),
            "avg_balance": round(sum(balances) / total_accounts, 1),
            "max_balance": round(max(balances), 1),
            "total_rounds_contributed": sum(
                a.rounds_contributed for a in self._accounts.values()
            ),
            "total_data_contributed_mb": round(
                sum(a.data_contributed_mb for a in self._accounts.values()), 1
            ),
            "total_votes_cast": sum(
                a.votes_cast for a in self._accounts.values()
            ),
        }

    def get_peer_summary(self, peer_id: str) -> Dict:
        """Get a summary of a peer's credit account."""
        account = self.get_account(peer_id)
        return {
            "peer_id": peer_id,
            "balance": round(account.raw_balance, 1),
            "total_earned": round(account.total_earned, 1),
            "total_spent": round(account.total_spent, 1),
            "total_decayed": round(account.total_decayed, 1),
            "is_contributor": account.is_contributor,
            "rounds_contributed": account.rounds_contributed,
            "data_contributed_mb": round(account.data_contributed_mb, 1),
            "votes_cast": account.votes_cast,
            "recent_transactions": [
                {
                    "action": tx.action.value,
                    "amount": round(tx.amount, 2),
                    "description": tx.description,
                }
                for tx in account.transactions[-10:]
            ],
        }


class InferenceGate:
    """Gates inference requests based on credit balance.

    Sits between the inference server and incoming requests. Checks the
    peer's credit balance and either allows the request (deducting credits),
    rate-limits it to the free tier, or rejects it.

    Usage:
        gate = InferenceGate(ledger)

        # Before serving inference:
        result = gate.authorize(peer_id, priority=False, token_count=100)
        if result.allowed:
            # Serve the request.
            response = model.generate(...)
        else:
            # Return rate-limit error.
            return 429, result.reason
    """

    def __init__(self, ledger: CreditLedger):
        self.ledger = ledger
        self._total_authorized: int = 0
        self._total_rejected: int = 0
        self._total_free: int = 0
        self._total_paid: int = 0

    def authorize(
        self,
        peer_id: str,
        priority: bool = False,
        token_count: int = 0,
    ) -> SpendResult:
        """Authorize an inference request.

        Checks credits and deducts if authorized.
        """
        long_gen = token_count > 256
        result = self.ledger.spend_inference(
            peer_id, priority=priority, long_generation=long_gen,
        )

        if result.allowed:
            self._total_authorized += 1
            if result.tier == InferenceTier.FREE:
                self._total_free += 1
            else:
                self._total_paid += 1
        else:
            self._total_rejected += 1

        return result

    def stats(self) -> Dict:
        """Gate statistics."""
        total = self._total_authorized + self._total_rejected
        return {
            "total_requests": total,
            "authorized": self._total_authorized,
            "rejected": self._total_rejected,
            "free_tier": self._total_free,
            "paid_tier": self._total_paid,
            "rejection_rate": (
                round(self._total_rejected / total, 3) if total > 0 else 0
            ),
        }
