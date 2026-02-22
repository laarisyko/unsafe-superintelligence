"""Collaborative architecture evolution: propose, vote, and apply model mutations."""

from .genome import ArchitectureGenome, LayerGene, LayerType
from .mutations import (
    Mutation,
    AddLayerMutation,
    RemoveLayerMutation,
    WidenLayerMutation,
    InsertSkipConnection,
    SwapActivation,
    MutationGenerator,
)
from .evolution import (
    ArchitectureProposal,
    ProposalVote,
    VoteDecision,
    EvolutionProtocol,
    FitnessEvaluator,
    PendingOutcome,
    ProposalOutcomeTracker,
)
from .evolution_ledger import (
    EvolutionLedger,
    LedgerEntry,
)
from .migration import WeightMigrator
