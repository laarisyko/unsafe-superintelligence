"""Async training orchestrator: end-to-end training round execution.

Ties together all components into a single async flow:
    Admission (PoW) -> Join -> Train -> Compress -> Exchange -> Aggregate -> Checkpoint

Each peer runs an Orchestrator instance. The orchestrator is deterministic:
given the same inputs, all peers produce the same outputs (except for their
local gradients, which are aggregated to produce a global result).

Usage:
    orch = TrainingOrchestrator(config, shard, checkpoint_store, reputation)
    result = await orch.run_round(round_id, peer_registry, send_fn, recv_fn)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ..model.shard import ModelShard
from ..model.checkpoint import CheckpointStore
from .byzantine import AggregationMethod, ByzantineConfig, robust_aggregate, score_gradients
from .cluster import PeerCapacity
from .compression import GradientCompressor, TopKCompressor, FP16Compressor, CompressorChain
from .reputation import ReputationTracker
from .round_coordinator import RoundCoordinator, RoundConfig, RoundPhase
from .sybil import AdmissionController, PowChallenge, solve, verify
from .trainer import LocalTrainer, TrainingConfig
from .wire import encode, decode, WireMessage, estimate_wire_size

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""

    # Model.
    model_id: str = "default"

    # Training.
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Compression: "none", "topk", "fp16", "topk+fp16"
    compression: str = "fp16"
    topk_ratio: float = 0.01

    # Byzantine resilience.
    byzantine_config: ByzantineConfig = field(
        default_factory=lambda: ByzantineConfig(method=AggregationMethod.TRIMMED_MEAN)
    )

    # Round settings.
    min_peers: int = 3
    min_quorum_ratio: float = 0.67
    join_deadline_secs: float = 30.0
    gradient_deadline_secs: float = 120.0

    # Sybil resistance.
    pow_enabled: bool = True
    pow_difficulty: int = 16

    # Checkpointing.
    checkpoint_every: int = 1  # Checkpoint every N rounds.
    checkpoint_gc_keep: int = 5


@dataclass
class RoundResult:
    """Result of a completed training round."""

    round_id: str
    success: bool
    aggregated_gradients: Optional[Dict[str, torch.Tensor]] = None
    merkle_root: str = ""
    n_participants: int = 0
    n_submitted: int = 0
    n_dropped: int = 0
    n_byzantine_detected: int = 0
    wire_bytes_sent: int = 0
    wire_bytes_received: int = 0
    compression_ratio: float = 1.0
    pow_solve_ms: float = 0.0
    train_ms: float = 0.0
    aggregate_ms: float = 0.0
    total_ms: float = 0.0
    error: str = ""


class TrainingOrchestrator:
    """Orchestrates a complete training round from start to finish.

    This is the top-level component that each peer runs. It coordinates
    all subsystems: admission, training, compression, aggregation,
    checkpointing, and reputation updates.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        shard: ModelShard,
        checkpoint_store: Optional[CheckpointStore] = None,
        reputation: Optional[ReputationTracker] = None,
    ):
        self.config = config
        self.shard = shard
        self.trainer = LocalTrainer(shard, config.training_config)
        self.compressor = self._build_compressor()
        self.checkpoint_store = checkpoint_store
        self.reputation = reputation or ReputationTracker()
        self.admission = AdmissionController(
            base_difficulty=config.pow_difficulty
        ) if config.pow_enabled else None
        self.round_count = 0

    def _build_compressor(self) -> Optional[GradientCompressor]:
        """Build the gradient compressor from config."""
        c = self.config.compression
        if c == "none":
            return None
        elif c == "topk":
            return TopKCompressor(ratio=self.config.topk_ratio)
        elif c == "fp16":
            return FP16Compressor()
        elif c == "topk+fp16":
            return CompressorChain([
                TopKCompressor(ratio=self.config.topk_ratio),
                FP16Compressor(),
            ])
        return None

    def run_round_sync(
        self,
        round_id: str,
        peer_capacities: List[PeerCapacity],
        all_peer_gradients: Dict[str, Dict[str, torch.Tensor]],
        local_peer_id: str,
    ) -> RoundResult:
        """Run a complete training round synchronously (for simulation/testing).

        In production, this would be async with real network I/O.
        The sync version accepts pre-computed gradients from all peers.

        Args:
            round_id: Unique round identifier.
            peer_capacities: All participating peers.
            all_peer_gradients: Gradients from every peer (keyed by peer_id).
            local_peer_id: This peer's ID.

        Returns:
            RoundResult with aggregated gradients and metrics.
        """
        start = time.monotonic()
        result = RoundResult(round_id=round_id, success=False)

        # --- Phase 1: Admission (PoW) ---
        if self.admission is not None:
            challenge = self.admission.create_challenge(round_id, len(peer_capacities))
            pow_start = time.monotonic()
            solution = self.admission.solve_challenge(round_id, local_peer_id)
            result.pow_solve_ms = (time.monotonic() - pow_start) * 1000

            if solution is None or not self.admission.verify_admission(
                round_id, solution, self.reputation.get_score(local_peer_id)
            ):
                result.error = "PoW admission failed"
                return result

        # --- Phase 2: Registration ---
        round_config = RoundConfig(
            round_id=round_id,
            model_id=self.config.model_id,
            min_peers=self.config.min_peers,
            min_quorum_ratio=self.config.min_quorum_ratio,
            join_deadline_secs=self.config.join_deadline_secs,
            gradient_deadline_secs=self.config.gradient_deadline_secs,
            byzantine_config=self.config.byzantine_config,
        )
        coordinator = RoundCoordinator(round_config)

        # Filter out banned peers.
        admitted = []
        for cap in peer_capacities:
            if self.reputation.is_allowed(cap.peer_id):
                coordinator.register_peer(cap)
                admitted.append(cap)

        if not coordinator.close_registration():
            result.error = "Registration failed (not enough peers)"
            result.total_ms = (time.monotonic() - start) * 1000
            return result

        result.n_participants = len(admitted)

        # --- Phase 3: Local training ---
        train_start = time.monotonic()
        x = torch.randn(self.config.training_config.batch_size, 16)
        self.trainer.train_step(x)
        local_grads = self.trainer.get_gradients()
        result.train_ms = (time.monotonic() - train_start) * 1000

        # --- Phase 4: Encode + estimate wire size ---
        wire_est = estimate_wire_size(local_grads, self.compressor)
        result.compression_ratio = wire_est["compression_ratio"]

        # Encode our own gradients (simulates network send).
        wire_msg = encode(
            local_grads,
            round_id=round_id,
            peer_id=local_peer_id,
            compressor=self.compressor,
        )
        result.wire_bytes_sent = wire_msg.size_bytes

        # --- Phase 5: Submit all gradients to coordinator ---
        for cap in admitted:
            pid = cap.peer_id
            if pid in all_peer_gradients:
                grads = all_peer_gradients[pid]
                coordinator.submit_gradient(pid, grads)

        # --- Phase 6: Aggregate ---
        agg_start = time.monotonic()
        aggregated = coordinator.finalize()
        result.aggregate_ms = (time.monotonic() - agg_start) * 1000

        if aggregated is None:
            result.error = "Aggregation failed (quorum not met)"
            result.total_ms = (time.monotonic() - start) * 1000
            return result

        result.aggregated_gradients = aggregated
        result.n_submitted = coordinator.n_submitted

        # --- Phase 7: Apply aggregated gradients ---
        self.trainer.set_gradients(aggregated)
        self.trainer.apply_gradients()

        # --- Phase 8: Score peers for reputation ---
        submitted_peers = [
            cap.peer_id for cap in admitted
            if cap.peer_id in all_peer_gradients
        ]
        grad_list = [all_peer_gradients[pid] for pid in submitted_peers]
        scores = score_gradients(grad_list, self.config.byzantine_config.max_byzantine)
        outliers = self.reputation.detect_outliers(submitted_peers, scores)
        result.n_byzantine_detected = len(outliers)

        dropped = [
            cap.peer_id for cap in admitted
            if cap.peer_id not in all_peer_gradients
        ]
        result.n_dropped = len(dropped)

        self.reputation.record_round_batch(
            submitted_peers=submitted_peers,
            dropped_peers=dropped,
            byzantine_scores={pid: s for pid, s in zip(submitted_peers, scores)},
            outlier_peer_ids=outliers,
        )

        # --- Phase 9: Checkpoint ---
        self.round_count += 1
        if (
            self.checkpoint_store is not None
            and self.round_count % self.config.checkpoint_every == 0
        ):
            merkle = self.checkpoint_store.save(
                self.shard,
                round_id,
                peer_ids=submitted_peers,
                aggregation_method=self.config.byzantine_config.method.name,
            )
            result.merkle_root = merkle

            # GC old checkpoints.
            self.checkpoint_store.gc(keep_latest=self.config.checkpoint_gc_keep)
        else:
            result.merkle_root = self.shard.merkle_root().hex()

        result.success = True
        result.total_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Round %s complete: %d peers, %d submitted, %d Byzantine, "
            "%.1fms total, %.1f KB wire, %.1fx compression",
            round_id,
            result.n_participants,
            result.n_submitted,
            result.n_byzantine_detected,
            result.total_ms,
            result.wire_bytes_sent / 1024,
            result.compression_ratio,
        )

        return result

    def get_stats(self) -> Dict:
        """Get current orchestrator stats."""
        return {
            "model_id": self.config.model_id,
            "round_count": self.round_count,
            "compression": self.config.compression,
            "byzantine_method": self.config.byzantine_config.method.name,
            "pow_enabled": self.config.pow_enabled,
            "reputation_summary": self.reputation.summary(),
        }
