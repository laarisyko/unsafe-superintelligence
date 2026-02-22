"""Tests for the signed evolution ledger.

Proves that:
    1. Entries are appended with correct hash chaining
    2. Genesis hash is used for the first entry
    3. Tamper detection works (modifying an entry breaks the chain)
    4. JSON serialization roundtrip preserves integrity
    5. Loading tampered JSON raises ValueError
    6. post_fitness can be updated without breaking the chain
    7. Integration with EvolutionProtocol records decisions
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from ussi_engine.architecture.evolution_ledger import (
    EvolutionLedger,
    LedgerEntry,
    GENESIS_HASH,
)
from ussi_engine.architecture.evolution import (
    EvolutionProtocol,
    FitnessEvaluator,
    VoteDecision,
    ProposalVote,
)
from ussi_engine.architecture.genome import (
    ArchitectureGenome,
    LayerGene,
    LayerType,
)
from ussi_engine.architecture.mutations import AddLayerMutation


def test_ledger_append_and_hash_chain():
    """Appended entries form a valid hash chain."""
    ledger = EvolutionLedger()

    e1 = ledger.append(
        proposal_id="prop-1",
        proposer_id="peer-a",
        mutation_description="Add norm layer",
        outcome="accepted",
        approve_weight=1.5,
        reject_weight=0.3,
        total_weight=1.8,
        voter_count=3,
        baseline_fitness=-0.5,
    )
    e2 = ledger.append(
        proposal_id="prop-2",
        proposer_id="peer-b",
        mutation_description="Widen layer 0",
        outcome="rejected",
        approve_weight=0.2,
        reject_weight=1.0,
        total_weight=1.2,
        voter_count=2,
        baseline_fitness=-0.4,
    )

    assert len(ledger) == 2
    assert e1.entry_index == 0
    assert e2.entry_index == 1
    assert e2.previous_hash == e1.entry_hash
    assert ledger.head_hash == e2.entry_hash
    assert ledger.verify_chain()


def test_ledger_genesis_hash():
    """First entry's previous_hash is the genesis hash."""
    ledger = EvolutionLedger()
    entry = ledger.append(
        proposal_id="prop-0",
        proposer_id="peer-a",
        mutation_description="Initial",
        outcome="accepted",
        approve_weight=1.0,
        reject_weight=0.0,
        total_weight=1.0,
        voter_count=1,
    )
    assert entry.previous_hash == GENESIS_HASH
    assert ledger.verify_chain()


def test_ledger_tamper_detection():
    """Modifying an entry's data breaks the hash chain."""
    ledger = EvolutionLedger()
    ledger.append(
        proposal_id="prop-1",
        proposer_id="peer-a",
        mutation_description="Add layer",
        outcome="accepted",
        approve_weight=1.0,
        reject_weight=0.0,
        total_weight=1.0,
        voter_count=1,
    )
    ledger.append(
        proposal_id="prop-2",
        proposer_id="peer-b",
        mutation_description="Remove layer",
        outcome="rejected",
        approve_weight=0.0,
        reject_weight=1.0,
        total_weight=1.0,
        voter_count=1,
    )

    assert ledger.verify_chain()

    # Tamper with the first entry.
    ledger._entries[0].mutation_description = "TAMPERED"
    assert not ledger.verify_chain()


def test_ledger_json_roundtrip():
    """Ledger survives JSON serialization/deserialization."""
    ledger = EvolutionLedger()
    for i in range(5):
        ledger.append(
            proposal_id=f"prop-{i}",
            proposer_id=f"peer-{i % 3}",
            mutation_description=f"Mutation {i}",
            outcome="accepted" if i % 2 == 0 else "rejected",
            approve_weight=float(i + 1),
            reject_weight=float(5 - i),
            total_weight=6.0,
            voter_count=3,
            baseline_fitness=-0.5 + i * 0.1,
            timestamp_ms=1000000 + i * 1000,
        )

    json_str = ledger.to_json()
    restored = EvolutionLedger.from_json(json_str)

    assert len(restored) == 5
    assert restored.head_hash == ledger.head_hash
    assert restored.verify_chain()
    for orig, rest in zip(ledger.entries, restored.entries):
        assert orig.entry_hash == rest.entry_hash
        assert orig.proposal_id == rest.proposal_id


def test_ledger_from_json_invalid_chain():
    """Loading tampered JSON raises ValueError."""
    ledger = EvolutionLedger()
    ledger.append(
        proposal_id="prop-1",
        proposer_id="peer-a",
        mutation_description="Good entry",
        outcome="accepted",
        approve_weight=1.0,
        reject_weight=0.0,
        total_weight=1.0,
        voter_count=1,
    )

    json_str = ledger.to_json()
    # Tamper with the JSON.
    tampered = json_str.replace("Good entry", "Bad entry")

    try:
        EvolutionLedger.from_json(tampered)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "tampered" in str(e).lower()


def test_ledger_update_post_fitness():
    """post_fitness can be updated without breaking the hash chain."""
    ledger = EvolutionLedger()
    ledger.append(
        proposal_id="prop-1",
        proposer_id="peer-a",
        mutation_description="Add layer",
        outcome="accepted",
        approve_weight=1.0,
        reject_weight=0.0,
        total_weight=1.0,
        voter_count=1,
        baseline_fitness=-0.5,
    )
    ledger.append(
        proposal_id="prop-2",
        proposer_id="peer-b",
        mutation_description="Widen",
        outcome="accepted",
        approve_weight=0.8,
        reject_weight=0.2,
        total_weight=1.0,
        voter_count=2,
        baseline_fitness=-0.4,
    )

    # Update post_fitness.
    ledger.update_post_fitness("prop-1", -0.3)
    assert ledger.entries[0].post_fitness == -0.3
    assert ledger.entries[1].post_fitness is None

    # Chain should still be valid (post_fitness is not in the hash).
    assert ledger.verify_chain()


def test_ledger_evolution_protocol_integration():
    """EvolutionProtocol records decisions in the ledger."""
    genome = ArchitectureGenome(
        model_id="ledger-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.ACTIVATION, 0, 0, activation="relu"),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )

    ledger = EvolutionLedger()
    protocol = EvolutionProtocol(
        peer_id="peer-0",
        current_genome=genome.clone(),
        evaluator=FitnessEvaluator(),
        approval_quorum=0.5,
        min_voters=1,
        ledger=ledger,
    )

    # Accepted proposal.
    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.NORM, 64, 64),
    )
    proposal = protocol.propose_mutation(mutation)
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal.proposal_id,
        voter_id="peer-1",
        decision=VoteDecision.APPROVE,
    ))
    result = protocol.tally_votes(proposal.proposal_id)
    assert result is not None

    assert len(ledger) == 1
    entry = ledger.entries[0]
    assert entry.proposal_id == proposal.proposal_id
    assert entry.outcome == "accepted"
    assert entry.voter_count >= 1

    # Rejected proposal.
    mutation2 = AddLayerMutation(
        position=0,
        gene=LayerGene(LayerType.NORM, 32, 32),
    )
    proposal2 = protocol.propose_mutation(mutation2)
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal2.proposal_id,
        voter_id="peer-1",
        decision=VoteDecision.REJECT,
    ))
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal2.proposal_id,
        voter_id="peer-2",
        decision=VoteDecision.REJECT,
    ))
    result2 = protocol.tally_votes(proposal2.proposal_id)
    assert result2 is None

    assert len(ledger) == 2
    entry2 = ledger.entries[1]
    assert entry2.outcome == "rejected"
    assert ledger.verify_chain()


if __name__ == "__main__":
    tests = [
        test_ledger_append_and_hash_chain,
        test_ledger_genesis_hash,
        test_ledger_tamper_detection,
        test_ledger_json_roundtrip,
        test_ledger_from_json_invalid_chain,
        test_ledger_update_post_fitness,
        test_ledger_evolution_protocol_integration,
    ]
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"\nAll {len(tests)} evolution ledger tests passed!")
