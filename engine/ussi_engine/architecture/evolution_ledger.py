"""Signed evolution ledger: tamper-evident audit trail of governance decisions.

Every architecture proposal decision (accepted or rejected) is recorded as a
LedgerEntry with a SHA-256 hash chain. This provides:
- Audit trail: see exactly what was proposed, who voted, and what happened.
- Tamper detection: any modification to past entries breaks the hash chain.
- Post-hoc analysis: track fitness before/after mutations to evaluate quality.

The hash chain does NOT require a blockchain or consensus -- each peer maintains
its own ledger. The chain is a local integrity check, not a distributed one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GENESIS_HASH = "0" * 64


@dataclass
class LedgerEntry:
    """A single entry in the evolution ledger."""

    entry_index: int
    proposal_id: str
    proposer_id: str
    mutation_description: str
    outcome: str  # "accepted", "rejected", or "expired"
    approve_weight: float
    reject_weight: float
    total_weight: float
    voter_count: int
    baseline_fitness: Optional[float]
    post_fitness: Optional[float]  # Updated later, not part of hash chain.
    timestamp_ms: int
    previous_hash: str
    entry_hash: str = ""

    def canonical_bytes(self) -> bytes:
        """Deterministic JSON for hashing (excludes entry_hash and post_fitness)."""
        d = {
            "entry_index": self.entry_index,
            "proposal_id": self.proposal_id,
            "proposer_id": self.proposer_id,
            "mutation_description": self.mutation_description,
            "outcome": self.outcome,
            "approve_weight": self.approve_weight,
            "reject_weight": self.reject_weight,
            "total_weight": self.total_weight,
            "voter_count": self.voter_count,
            "baseline_fitness": self.baseline_fitness,
            "timestamp_ms": self.timestamp_ms,
            "previous_hash": self.previous_hash,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

    def compute_hash(self) -> str:
        """SHA-256 of canonical_bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "entry_index": self.entry_index,
            "proposal_id": self.proposal_id,
            "proposer_id": self.proposer_id,
            "mutation_description": self.mutation_description,
            "outcome": self.outcome,
            "approve_weight": self.approve_weight,
            "reject_weight": self.reject_weight,
            "total_weight": self.total_weight,
            "voter_count": self.voter_count,
            "baseline_fitness": self.baseline_fitness,
            "post_fitness": self.post_fitness,
            "timestamp_ms": self.timestamp_ms,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LedgerEntry":
        return cls(
            entry_index=d["entry_index"],
            proposal_id=d["proposal_id"],
            proposer_id=d["proposer_id"],
            mutation_description=d["mutation_description"],
            outcome=d["outcome"],
            approve_weight=d["approve_weight"],
            reject_weight=d["reject_weight"],
            total_weight=d["total_weight"],
            voter_count=d["voter_count"],
            baseline_fitness=d.get("baseline_fitness"),
            post_fitness=d.get("post_fitness"),
            timestamp_ms=d["timestamp_ms"],
            previous_hash=d["previous_hash"],
            entry_hash=d.get("entry_hash", ""),
        )


class EvolutionLedger:
    """Hash-chained ledger of architecture evolution decisions.

    Each entry is chained to the previous via SHA-256 hashes. This provides
    a tamper-evident audit trail without requiring distributed consensus.
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []
        self._head_hash: str = GENESIS_HASH

    def append(
        self,
        proposal_id: str,
        proposer_id: str,
        mutation_description: str,
        outcome: str,
        approve_weight: float,
        reject_weight: float,
        total_weight: float,
        voter_count: int,
        baseline_fitness: Optional[float] = None,
        timestamp_ms: Optional[int] = None,
    ) -> LedgerEntry:
        """Create and append a new ledger entry, chaining to the head."""
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        entry = LedgerEntry(
            entry_index=len(self._entries),
            proposal_id=proposal_id,
            proposer_id=proposer_id,
            mutation_description=mutation_description,
            outcome=outcome,
            approve_weight=approve_weight,
            reject_weight=reject_weight,
            total_weight=total_weight,
            voter_count=voter_count,
            baseline_fitness=baseline_fitness,
            post_fitness=None,
            timestamp_ms=timestamp_ms,
            previous_hash=self._head_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._head_hash = entry.entry_hash
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Validate the full hash chain integrity.

        Returns True if the chain is valid, False if tampered.
        """
        expected_prev = GENESIS_HASH
        for entry in self._entries:
            if entry.previous_hash != expected_prev:
                return False
            if entry.compute_hash() != entry.entry_hash:
                return False
            expected_prev = entry.entry_hash
        return True

    def update_post_fitness(self, proposal_id: str, post_fitness: float):
        """Update the post_fitness field for a settled proposal.

        This is an informational update -- post_fitness is not part of the
        hash chain, so updating it does not break integrity.
        """
        for entry in self._entries:
            if entry.proposal_id == proposal_id:
                entry.post_fitness = post_fitness
                return

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def __len__(self) -> int:
        return len(self._entries)

    def to_json(self) -> str:
        """Serialize the ledger to JSON."""
        return json.dumps(
            {"entries": [e.to_dict() for e in self._entries]},
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> "EvolutionLedger":
        """Deserialize a ledger from JSON, verifying chain integrity.

        Raises ValueError if the hash chain is invalid.
        """
        parsed = json.loads(data)
        ledger = cls()
        for entry_dict in parsed["entries"]:
            entry = LedgerEntry.from_dict(entry_dict)
            ledger._entries.append(entry)
        if ledger._entries:
            ledger._head_hash = ledger._entries[-1].entry_hash

        if not ledger.verify_chain():
            raise ValueError("Ledger hash chain verification failed: data is tampered")

        return ledger
