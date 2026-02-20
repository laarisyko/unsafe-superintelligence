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
    ):
        """
        Args:
            peer_id: This peer's identifier.
            current_genome: The currently active architecture genome.
            evaluator: Local fitness evaluator.
            approval_quorum: Fraction of votes that must approve (0.0 - 1.0).
            min_voters: Minimum number of votes required for a decision.
            publish_fn: Callable to publish messages to gossipsub.
        """
        self.peer_id = peer_id
        self.current_genome = current_genome
        self.evaluator = evaluator or FitnessEvaluator()
        self.approval_quorum = approval_quorum
        self.min_voters = min_voters
        self.publish_fn = publish_fn

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
            self.publish_fn("openclaw/architecture", proposal.to_dict())

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
            self.publish_fn("openclaw/architecture", vote.to_dict())

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
        """
        if proposal_id not in self._proposals:
            return None

        votes = self._votes.get(proposal_id, [])
        # Deduplicate by voter_id (last vote wins).
        unique_votes = {}
        for v in votes:
            unique_votes[v.voter_id] = v

        approvals = sum(
            1 for v in unique_votes.values() if v.decision == VoteDecision.APPROVE
        )
        rejections = sum(
            1 for v in unique_votes.values() if v.decision == VoteDecision.REJECT
        )
        total = len(unique_votes)

        if total < self.min_voters:
            logger.info(
                "Proposal %s: not enough voters (%d/%d)",
                proposal_id,
                total,
                self.min_voters,
            )
            return None

        approval_rate = approvals / total if total > 0 else 0.0

        if approval_rate >= self.approval_quorum:
            proposal = self._proposals[proposal_id]
            self.current_genome = proposal.new_genome
            self._history.append(self.current_genome)
            logger.info(
                "Proposal %s ACCEPTED (%.0f%% approval, %d voters). "
                "New genome: gen=%d, hash=%s",
                proposal_id,
                approval_rate * 100,
                total,
                self.current_genome.generation,
                self.current_genome.hash(),
            )
            return self.current_genome
        else:
            logger.info(
                "Proposal %s REJECTED (%.0f%% approval, %d voters)",
                proposal_id,
                approval_rate * 100,
                total,
            )
            return None

    @property
    def genome_history(self) -> List[ArchitectureGenome]:
        return self._history

    @property
    def pending_proposals(self) -> Dict[str, ArchitectureProposal]:
        return self._proposals
