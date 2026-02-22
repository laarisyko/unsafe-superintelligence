"""Decentralized architecture evolution protocol.

No master decides what the architecture should be. Instead:
1. Any peer can PROPOSE a mutation to the current genome.
2. Peers EVALUATE the proposal by running a short validation trial.
3. Peers VOTE (approve/reject) based on fitness improvement.
4. If a quorum of peers approves, the mutation is APPLIED globally.
5. Weights are MIGRATED from the old architecture to the new one.

This is evolutionary neural architecture search (NAS) running on a
decentralized swarm.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .genome import ArchitectureGenome
from .mutations import Mutation

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureProposal:
    """A proposed mutation to the model architecture.

    Any peer can create one and broadcast it via gossipsub. Other peers
    evaluate and vote.
    """

    proposal_id: str
    proposer_id: str
    model_id: str
    current_genome_hash: str
    mutation: Mutation
    new_genome: ArchitectureGenome
    timestamp_ms: int = 0
    # Fitness score observed by the proposer (optional, advisory).
    proposer_fitness: float = 0.0
    # Deadline for votes (ms since epoch).
    vote_deadline_ms: int = 0

    def __post_init__(self):
        if not self.proposal_id:
            self.proposal_id = f"arch-{uuid.uuid4().hex[:12]}"
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)
        if self.vote_deadline_ms == 0:
            self.vote_deadline_ms = self.timestamp_ms + 60_000  # 60s default

    def to_dict(self) -> Dict:
        return {
            "type": "architecture_proposal",
            "proposal_id": self.proposal_id,
            "proposer_id": self.proposer_id,
            "model_id": self.model_id,
            "current_genome_hash": self.current_genome_hash,
            "mutation": self.mutation.to_dict(),
            "new_genome": self.new_genome.to_dict(),
            "timestamp_ms": self.timestamp_ms,
            "proposer_fitness": self.proposer_fitness,
            "vote_deadline_ms": self.vote_deadline_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ArchitectureProposal":
        mutation = Mutation.from_dict(d["mutation"])
        new_genome = ArchitectureGenome.from_dict(d["new_genome"])
        return cls(
            proposal_id=d["proposal_id"],
            proposer_id=d["proposer_id"],
            model_id=d["model_id"],
            current_genome_hash=d["current_genome_hash"],
            mutation=mutation,
            new_genome=new_genome,
            timestamp_ms=d.get("timestamp_ms", 0),
            proposer_fitness=d.get("proposer_fitness", 0.0),
            vote_deadline_ms=d.get("vote_deadline_ms", 0),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ArchitectureProposal":
        return cls.from_dict(json.loads(data))


class VoteDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class ProposalVote:
    """A peer's vote on an architecture proposal."""

    proposal_id: str
    voter_id: str
    decision: VoteDecision
    # Fitness score the voter measured on the proposed architecture.
    measured_fitness: float = 0.0
    # How many validation samples the voter used.
    eval_samples: int = 0
    timestamp_ms: int = 0

    def __post_init__(self):
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> Dict:
        return {
            "type": "architecture_vote",
            "proposal_id": self.proposal_id,
            "voter_id": self.voter_id,
            "decision": self.decision.value,
            "measured_fitness": self.measured_fitness,
            "eval_samples": self.eval_samples,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ProposalVote":
        return cls(
            proposal_id=d["proposal_id"],
            voter_id=d["voter_id"],
            decision=VoteDecision(d["decision"]),
            measured_fitness=d.get("measured_fitness", 0.0),
            eval_samples=d.get("eval_samples", 0),
            timestamp_ms=d.get("timestamp_ms", 0),
        )


class FitnessEvaluator:
    """Evaluates a proposed architecture's fitness on local validation data.

    Each peer runs this independently to decide how to vote. The fitness
    function compares the new architecture against the current one.
    """

    def __init__(
        self,
        validation_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        loss_fn: Optional[nn.Module] = None,
        eval_steps: int = 10,
    ):
        self.validation_data = validation_data or []
        self.loss_fn = loss_fn or nn.MSELoss()
        self.eval_steps = eval_steps

    def evaluate(
        self,
        current_genome: ArchitectureGenome,
        proposed_genome: ArchitectureGenome,
    ) -> Tuple[float, float]:
        """Evaluate both architectures and return (current_fitness, proposed_fitness).

        Fitness is negative loss (higher is better).
        """
        current_model = current_genome.compile()
        proposed_model = proposed_genome.compile()

        current_fitness = self._evaluate_model(current_model, current_genome)
        proposed_fitness = self._evaluate_model(proposed_model, proposed_genome)

        return current_fitness, proposed_fitness

    def _evaluate_model(self, model: nn.Module, genome: ArchitectureGenome) -> float:
        """Run forward passes and compute fitness (negative loss).

        Returns float('-inf') for any crash, NaN output, or validation failure.
        """
        # Validate genome first.
        errors = genome.validate()
        if errors:
            return float("-inf")

        model.eval()
        total_loss = 0.0
        steps = 0

        if self.validation_data:
            with torch.no_grad():
                for x, y in self.validation_data[:self.eval_steps]:
                    try:
                        output = model(x)
                        # Check for NaN/Inf in output.
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            return float("-inf")
                        # Adjust output shape to match target if needed.
                        if output.shape != y.shape:
                            output = output[..., :y.shape[-1]]
                        loss = self.loss_fn(output, y)
                        loss_val = loss.item()
                        if not (loss_val == loss_val):  # NaN check
                            return float("-inf")
                        total_loss += loss_val
                        steps += 1
                    except (RuntimeError, Exception):
                        # Any crash = hard reject.
                        return float("-inf")
        else:
            # No validation data -- use parameter efficiency as proxy.
            param_count = genome.estimated_parameters()
            # Prefer smaller models (normalized).
            total_loss = param_count / 1e6
            steps = 1

        avg_loss = total_loss / max(steps, 1)
        return -avg_loss  # Fitness is negative loss (higher is better).

    def should_approve(
        self,
        current_fitness: float,
        proposed_fitness: float,
        threshold: float = 0.0,
    ) -> VoteDecision:
        """Decide whether to approve based on fitness improvement.

        Args:
            current_fitness: Fitness of current architecture.
            proposed_fitness: Fitness of proposed architecture.
            threshold: Minimum improvement required (default: any improvement).

        Returns:
            VoteDecision.
        """
        # Hard-reject broken architectures.
        if proposed_fitness == float("-inf"):
            return VoteDecision.REJECT

        improvement = proposed_fitness - current_fitness
        if improvement > threshold:
            return VoteDecision.APPROVE
        elif improvement < -threshold:
            return VoteDecision.REJECT
        else:
            return VoteDecision.ABSTAIN


@dataclass
class PendingOutcome:
    """Tracks an accepted proposal awaiting fitness evaluation."""

    proposal_id: str
    proposer_id: str
    voter_decisions: Dict[str, VoteDecision]
    baseline_fitness: float
    accepted_at_round: int
    settled: bool = False


class ProposalOutcomeTracker:
    """Tracks proposal outcomes and distributes rewards/penalties.

    After a mutation is accepted, waits `rounds_to_settle` training rounds
    to measure whether the mutation actually improved the model. Then:
    - Good mutation: proposer gets bonus credits, accurate approve-voters
      get their deposit refunded with bonus.
    - Bad mutation: proposer gets reputation penalty, accurate reject-voters
      get their deposit refunded with bonus.

    Voter accuracy: approved a good mutation OR rejected a bad mutation = accurate.
    """

    def __init__(
        self,
        reputation: object,  # ReputationTracker
        ledger: object,  # CreditLedger
        evaluator: FitnessEvaluator,
        rounds_to_settle: int = 5,
        proposer_reward_credits: float = 5.0,
        proposer_penalty_reputation: float = 0.05,
    ):
        self.reputation = reputation
        self.credit_ledger = ledger
        self.evaluator = evaluator
        self.rounds_to_settle = rounds_to_settle
        self.proposer_reward_credits = proposer_reward_credits
        self.proposer_penalty_reputation = proposer_penalty_reputation

        self._pending: List[PendingOutcome] = []
        self._current_round: int = 0

    def register_accepted(
        self,
        proposal: ArchitectureProposal,
        voter_decisions: Dict[str, VoteDecision],
        baseline_fitness: float,
    ):
        """Register an accepted proposal for future settlement."""
        outcome = PendingOutcome(
            proposal_id=proposal.proposal_id,
            proposer_id=proposal.proposer_id,
            voter_decisions=dict(voter_decisions),
            baseline_fitness=baseline_fitness,
            accepted_at_round=self._current_round,
        )
        self._pending.append(outcome)

    def tick_round(self, current_genome: ArchitectureGenome):
        """Called after each training round. Settles matured proposals."""
        self._current_round += 1
        self._settle_matured(current_genome)

    def _settle_matured(self, current_genome: ArchitectureGenome):
        """Settle all proposals that have matured (enough rounds elapsed)."""
        for outcome in self._pending:
            if outcome.settled:
                continue
            rounds_elapsed = self._current_round - outcome.accepted_at_round
            if rounds_elapsed >= self.rounds_to_settle:
                self._settle_one(outcome, current_genome)

    def _settle_one(
        self, outcome: PendingOutcome, current_genome: ArchitectureGenome
    ):
        """Settle a single pending outcome."""
        outcome.settled = True

        # Measure current fitness.
        current_fitness = self.evaluator._evaluate_model(
            current_genome.compile(), current_genome
        )
        is_good_mutation = current_fitness >= outcome.baseline_fitness

        # Reward/penalize the proposer.
        if is_good_mutation:
            # Good mutation: proposer gets bonus credits.
            if hasattr(self.credit_ledger, "earn_training_round"):
                proposer_account = self.credit_ledger._get_or_create(
                    outcome.proposer_id
                )
                proposer_account.raw_balance += self.proposer_reward_credits
                proposer_account.total_earned += self.proposer_reward_credits
                self.credit_ledger._total_credits_minted += (
                    self.proposer_reward_credits
                )
        else:
            # Bad mutation: proposer gets reputation penalty.
            if hasattr(self.reputation, "_peers"):
                record = getattr(self.reputation, "_get_or_create", lambda x: None)(
                    outcome.proposer_id
                )
                if record is not None:
                    record.score = max(
                        0.0, record.score - self.proposer_penalty_reputation
                    )

        # Settle voter deposits.
        for voter_id, decision in outcome.voter_decisions.items():
            if decision == VoteDecision.ABSTAIN:
                continue
            # Accurate = approved a good mutation or rejected a bad one.
            was_accurate = (
                decision == VoteDecision.APPROVE and is_good_mutation
            ) or (decision == VoteDecision.REJECT and not is_good_mutation)

            if hasattr(self.credit_ledger, "settle_vote"):
                self.credit_ledger.settle_vote(
                    voter_id, outcome.proposal_id, was_accurate
                )

    @property
    def pending(self) -> List[PendingOutcome]:
        return [p for p in self._pending if not p.settled]

    @property
    def settled(self) -> List[PendingOutcome]:
        return [p for p in self._pending if p.settled]


class EvolutionProtocol:
    """Orchestrates decentralized architecture evolution on a single peer.

    Manages the lifecycle of proposals: creation, broadcasting, vote
    collection, and decision-making. Works with the node's gossip layer
    to communicate with other peers.
    """

    def __init__(
        self,
        peer_id: str,
        current_genome: ArchitectureGenome,
        evaluator: Optional[FitnessEvaluator] = None,
        approval_quorum: float = 0.6,
        min_voters: int = 2,
        publish_fn: Optional[Callable] = None,
        get_reputation: Optional[Callable[[str], float]] = None,
        outcome_tracker: Optional["ProposalOutcomeTracker"] = None,
        ledger: Optional["EvolutionLedger"] = None,
    ):
        """
        Args:
            peer_id: This peer's identifier.
            current_genome: The currently active architecture genome.
            evaluator: Local fitness evaluator.
            approval_quorum: Fraction of votes that must approve (0.0 - 1.0).
            min_voters: Minimum number of votes required for a decision.
            publish_fn: Callable to publish messages to gossipsub.
            get_reputation: Callable returning reputation score (0.0-1.0) for a peer.
                If None, all peers get weight 0.5 (backward compatible).
            outcome_tracker: Tracks proposal outcomes for reward/penalty distribution.
            ledger: Signed evolution ledger for audit trail.
        """
        self.peer_id = peer_id
        self.current_genome = current_genome
        self.evaluator = evaluator or FitnessEvaluator()
        self.approval_quorum = approval_quorum
        self.min_voters = min_voters
        self.publish_fn = publish_fn
        self.get_reputation = get_reputation
        self.outcome_tracker = outcome_tracker
        self.ledger = ledger

        # Pending proposals and their votes.
        self._proposals: Dict[str, ArchitectureProposal] = {}
        self._votes: Dict[str, List[ProposalVote]] = {}
        # History of accepted genomes.
        self._history: List[ArchitectureGenome] = [current_genome]

    def propose_mutation(self, mutation: Mutation) -> ArchitectureProposal:
        """Create and broadcast an architecture mutation proposal."""
        new_genome = mutation.apply(self.current_genome)

        # Evaluate locally before proposing.
        _, proposed_fitness = self.evaluator.evaluate(
            self.current_genome, new_genome
        )

        proposal = ArchitectureProposal(
            proposal_id="",
            proposer_id=self.peer_id,
            model_id=self.current_genome.model_id,
            current_genome_hash=self.current_genome.hash(),
            mutation=mutation,
            new_genome=new_genome,
            proposer_fitness=proposed_fitness,
        )

        self._proposals[proposal.proposal_id] = proposal
        self._votes[proposal.proposal_id] = []

        # Broadcast to the swarm.
        if self.publish_fn:
            self.publish_fn("ussi/architecture", proposal.to_dict())

        logger.info(
            "Proposed architecture mutation: %s (fitness=%.4f) -> %s",
            mutation.describe(),
            proposed_fitness,
            new_genome.hash(),
        )
        return proposal

    def receive_proposal(self, proposal: ArchitectureProposal) -> ProposalVote:
        """Handle an incoming proposal from another peer.

        Evaluates the proposed architecture locally and casts a vote.
        """
        self._proposals[proposal.proposal_id] = proposal

        # Verify the proposal is against the current genome.
        if proposal.current_genome_hash != self.current_genome.hash():
            logger.warning(
                "Proposal %s targets genome %s but we have %s -- abstaining",
                proposal.proposal_id,
                proposal.current_genome_hash,
                self.current_genome.hash(),
            )
            vote = ProposalVote(
                proposal_id=proposal.proposal_id,
                voter_id=self.peer_id,
                decision=VoteDecision.ABSTAIN,
            )
        else:
            # Evaluate locally.
            current_fit, proposed_fit = self.evaluator.evaluate(
                self.current_genome, proposal.new_genome
            )
            decision = self.evaluator.should_approve(current_fit, proposed_fit)

            vote = ProposalVote(
                proposal_id=proposal.proposal_id,
                voter_id=self.peer_id,
                decision=decision,
                measured_fitness=proposed_fit,
                eval_samples=self.evaluator.eval_steps,
            )
            logger.info(
                "Voted %s on proposal %s (current=%.4f, proposed=%.4f)",
                decision.value,
                proposal.proposal_id,
                current_fit,
                proposed_fit,
            )

        self._votes.setdefault(proposal.proposal_id, []).append(vote)

        # Broadcast vote.
        if self.publish_fn:
            self.publish_fn("ussi/architecture", vote.to_dict())

        return vote

    def receive_vote(self, vote: ProposalVote):
        """Record a vote from another peer."""
        self._votes.setdefault(vote.proposal_id, []).append(vote)
        logger.debug(
            "Received vote from %s on %s: %s",
            vote.voter_id,
            vote.proposal_id,
            vote.decision.value,
        )

    def tally_votes(self, proposal_id: str) -> Optional[ArchitectureGenome]:
        """Tally votes for a proposal. Returns the new genome if accepted, else None.

        The decision is deterministic: given the same set of votes, every peer
        reaches the same conclusion (no coordinator needed).

        When get_reputation is set, votes are weighted by reputation score.
        ABSTAIN votes contribute to total weight (diluting low-rep approvers).
        A banned peer (reputation 0.0) has zero vote weight.
        """
        if proposal_id not in self._proposals:
            return None

        votes = self._votes.get(proposal_id, [])
        # Deduplicate by voter_id (last vote wins).
        unique_votes: Dict[str, ProposalVote] = {}
        for v in votes:
            unique_votes[v.voter_id] = v

        total_voters = len(unique_votes)

        if total_voters < self.min_voters:
            logger.info(
                "Proposal %s: not enough voters (%d/%d)",
                proposal_id,
                total_voters,
                self.min_voters,
            )
            return None

        # Compute weighted approval rate.
        approve_weight = 0.0
        reject_weight = 0.0
        total_weight = 0.0

        for voter_id, vote in unique_votes.items():
            if self.get_reputation is not None:
                w = self.get_reputation(voter_id)
            else:
                w = 0.5  # Fallback: equal weight for all peers.

            total_weight += w
            if vote.decision == VoteDecision.APPROVE:
                approve_weight += w
            elif vote.decision == VoteDecision.REJECT:
                reject_weight += w
            # ABSTAIN: contributes to total_weight but not approve/reject.

        approval_rate = approve_weight / total_weight if total_weight > 0 else 0.0

        proposal = self._proposals[proposal_id]
        accepted = approval_rate >= self.approval_quorum

        # Record baseline fitness before potential genome swap.
        baseline_fitness = None
        if len(self._history) >= 1:
            baseline_fitness = self.evaluator._evaluate_model(
                self._history[-1].compile(), self._history[-1]
            )

        if accepted:
            self.current_genome = proposal.new_genome
            self._history.append(self.current_genome)
            logger.info(
                "Proposal %s ACCEPTED (%.0f%% weighted approval, %d voters). "
                "New genome: gen=%d, hash=%s",
                proposal_id,
                approval_rate * 100,
                total_voters,
                self.current_genome.generation,
                self.current_genome.hash(),
            )

            # Register with outcome tracker for future settlement.
            if self.outcome_tracker is not None and baseline_fitness is not None:
                voter_decisions = {
                    vid: v.decision for vid, v in unique_votes.items()
                }
                self.outcome_tracker.register_accepted(
                    proposal, voter_decisions, baseline_fitness
                )
        else:
            logger.info(
                "Proposal %s REJECTED (%.0f%% weighted approval, %d voters)",
                proposal_id,
                approval_rate * 100,
                total_voters,
            )

        # Append to signed ledger for audit trail.
        if self.ledger is not None:
            self.ledger.append(
                proposal_id=proposal_id,
                proposer_id=proposal.proposer_id,
                mutation_description=proposal.mutation.describe(),
                outcome="accepted" if accepted else "rejected",
                approve_weight=approve_weight,
                reject_weight=reject_weight,
                total_weight=total_weight,
                voter_count=total_voters,
                baseline_fitness=baseline_fitness,
            )

        return self.current_genome if accepted else None

    @property
    def genome_history(self) -> List[ArchitectureGenome]:
        return self._history

    @property
    def pending_proposals(self) -> Dict[str, ArchitectureProposal]:
        return self._proposals
