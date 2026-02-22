"""P2P training network: the real decentralized training loop.

Wires together the kickstart system with peer discovery, gradient exchange,
and model synchronization. This is what runs when a peer joins the network
and starts contributing.

Flow:
    1. Peer joins the network (discovers other peers via seed nodes)
    2. Syncs to the latest checkpoint (downloads from DHT)
    3. Loads local training data
    4. Enters the training loop:
       a. Wait for round announcement (or propose one)
       b. Solve PoW for admission
       c. Train locally on own data
       d. Exchange gradients with peers
       e. Aggregate gradients (Byzantine-resilient)
       f. Apply aggregated update
       g. Checkpoint + announce
    5. Repeat forever
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from .kickstart import Kickstart, KickstartConfig, KickstartResult
from .genesis import GenesisTracker, assess_quality
from .credits import CreditLedger, CreditConfig, InferenceGate
from .model.checkpoint import CheckpointStore
from .training.byzantine import AggregationMethod, ByzantineConfig, robust_aggregate
from .training.reputation import ReputationTracker
from .training.sybil import AdmissionController, solve, verify
from .architecture.genome import ArchitectureGenome
from .architecture.evolution import EvolutionProtocol, FitnessEvaluator, ProposalOutcomeTracker
from .architecture.evolution_ledger import EvolutionLedger

logger = logging.getLogger(__name__)


# Seed nodes for initial network bootstrap.
SEED_NODES = [
    "/dns4/seed1.ussi.org/tcp/9000",
    "/dns4/seed2.ussi.org/tcp/9000",
    "/dns4/seed3.ussi.org/tcp/9000",
]

# Default model configs by size tier.
MODEL_CONFIGS = {
    "tiny": KickstartConfig(
        model_id="ussi-tiny",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=128,
        batch_size=4,
        steps_per_round=10,
    ),
    "small": KickstartConfig(
        model_id="ussi-small",
        hidden_dim=256,
        n_layers=6,
        n_heads=4,
        max_seq_length=256,
        batch_size=8,
        steps_per_round=20,
    ),
    "medium": KickstartConfig(
        model_id="ussi-medium",
        hidden_dim=512,
        n_layers=12,
        n_heads=8,
        max_seq_length=512,
        batch_size=8,
        steps_per_round=50,
        learning_rate=3e-4,
    ),
    "large": KickstartConfig(
        model_id="ussi-large",
        hidden_dim=1024,
        n_layers=24,
        n_heads=16,
        max_seq_length=1024,
        batch_size=4,
        steps_per_round=100,
        learning_rate=1e-4,
    ),
}


@dataclass
class PeerInfo:
    """Information about a connected peer."""
    peer_id: str
    address: str = ""
    compute_type: str = "cpu"
    gpu_memory_mb: int = 0
    last_seen: float = 0.0
    rounds_completed: int = 0
    reputation_score: float = 1.0


@dataclass
class NetworkConfig:
    """Configuration for the P2P training network."""

    # Identity.
    peer_id: str = ""

    # Network.
    listen_port: int = 9000
    seed_nodes: List[str] = field(default_factory=lambda: list(SEED_NODES))
    max_peers: int = 100

    # Model.
    model_size: str = "medium"  # tiny, small, medium, large
    model_config: Optional[KickstartConfig] = None  # Override model config.

    # Training.
    byzantine_method: AggregationMethod = AggregationMethod.TRIMMED_MEAN
    byzantine_tolerance: float = 0.2
    min_peers_per_round: int = 3
    round_timeout_secs: float = 120.0
    pow_difficulty: int = 16

    # Data.
    data_paths: List[str] = field(default_factory=list)
    auto_download_dataset: bool = True

    # Checkpoint.
    checkpoint_dir: str = ""
    checkpoint_interval: int = 1

    # Teacher model for distillation/DPO/synthetic data.
    teacher_config: Optional[object] = None  # TeacherConfig
    enable_distillation: bool = False
    enable_dpo: bool = False


@dataclass
class NetworkStats:
    """Live network statistics."""
    peer_id: str = ""
    connected_peers: int = 0
    total_rounds: int = 0
    total_steps: int = 0
    current_loss: float = float("inf")
    best_loss: float = float("inf")
    tokens_processed: int = 0
    compute_hours: float = 0.0
    model_params: int = 0
    model_size: str = ""
    uptime_secs: float = 0.0
    latest_sample: str = ""
    loss_history: List[float] = field(default_factory=list)


class TrainingNetwork:
    """The decentralized training network controller.

    This is the main entry point for a peer joining the network.
    It manages the complete lifecycle: join, sync, train, aggregate, repeat.
    """

    def __init__(self, config: NetworkConfig):
        self.config = config

        # Generate peer ID if not set.
        if not config.peer_id:
            config.peer_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

        # Initialize model (possibly with distillation or DPO wrapping).
        model_config = config.model_config or MODEL_CONFIGS.get(
            config.model_size, MODEL_CONFIGS["medium"]
        )
        if config.enable_distillation and config.teacher_config is not None:
            from .training.distillation import DistillationKickstart, DistillationConfig
            distill_config = DistillationConfig(teacher=config.teacher_config)
            self.kickstart = DistillationKickstart(model_config, distill_config)
        elif config.enable_dpo and config.teacher_config is not None:
            from .training.rl_from_ai import DPOKickstart, DPOConfig
            dpo_config = DPOConfig(teacher=config.teacher_config)
            self.kickstart = DPOKickstart(model_config, dpo_config)
        else:
            self.kickstart = Kickstart(model_config)

        # Subsystems.
        self.reputation = ReputationTracker()
        self.admission = AdmissionController(base_difficulty=config.pow_difficulty)
        checkpoint_dir = config.checkpoint_dir or os.path.join(
            os.path.expanduser("~"), ".ussi", "checkpoints"
        )
        self.checkpoints = CheckpointStore(checkpoint_dir)

        # Peer registry.
        self.peers: Dict[str, PeerInfo] = {}

        # Genesis tracker -- the model's life story.
        self.genesis = GenesisTracker(model_id=model_config.model_id)
        self.genesis.record_birth(
            model_params=self.kickstart.model.num_parameters,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        )

        # Credit system -- earn by contributing, spend on inference.
        self.credits = CreditLedger()
        self.inference_gate = InferenceGate(self.credits)
        self.credits.record_connect(config.peer_id)

        # Architecture governance -- stake-weighted voting, outcome tracking, audit ledger.
        self.evolution_ledger = EvolutionLedger()
        self.evolution_evaluator = FitnessEvaluator()
        self.outcome_tracker = ProposalOutcomeTracker(
            reputation=self.reputation,
            ledger=self.credits,
            evaluator=self.evolution_evaluator,
        )
        self.evolution = EvolutionProtocol(
            peer_id=config.peer_id,
            current_genome=ArchitectureGenome.simple_transformer(
                model_id=model_config.model_id,
                n_layers=model_config.n_layers,
                hidden_dim=model_config.hidden_dim,
            ),
            evaluator=self.evolution_evaluator,
            get_reputation=self.reputation.get_score,
            outcome_tracker=self.outcome_tracker,
            ledger=self.evolution_ledger,
        )

        # Stats tracking.
        self._stats = NetworkStats(
            peer_id=config.peer_id,
            model_size=config.model_size,
            model_params=self.kickstart.model.num_parameters,
        )
        self._start_time = time.monotonic()
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    def stats(self) -> NetworkStats:
        """Get current network statistics."""
        self._stats.uptime_secs = time.monotonic() - self._start_time
        self._stats.connected_peers = len(self.peers)
        return self._stats

    def on(self, event: str, callback: Callable):
        """Register a callback for network events.

        Events:
            - round_start: (round_id,)
            - round_complete: (round_id, result)
            - peer_joined: (peer_info,)
            - peer_left: (peer_id,)
            - checkpoint_saved: (merkle_root,)
            - loss_update: (round_id, loss)
        """
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, *args):
        """Emit an event to registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args)
            except Exception as e:
                logger.warning("Callback error for %s: %s", event, e)

    def load_data(self, paths: Optional[List[str]] = None):
        """Load training data from files/directories.

        Args:
            paths: List of file/directory paths. If None, uses config.data_paths.
        """
        data_paths = paths or self.config.data_paths
        for path in data_paths:
            p = Path(path)
            if p.is_file():
                self.kickstart.load_file(str(p))
            elif p.is_dir():
                self.kickstart.load_directory(str(p))
            else:
                logger.warning("Data path not found: %s", path)

        logger.info(
            "Loaded %d tokens of training data",
            self.kickstart.data.total_tokens,
        )

    def load_text(self, text: str):
        """Load raw text into the training pipeline."""
        self.kickstart.load_text(text)

    def synthetic_warmup(self, teacher_config, n_samples: int = 50):
        """Generate synthetic data before first training round.

        Args:
            teacher_config: TeacherConfig for the teacher model.
            n_samples: Number of synthetic samples to generate.
        """
        fed = self.kickstart.generate_synthetic_data(teacher_config, n_samples)
        logger.info("Synthetic warmup: %d samples generated", fed)

    def register_peer(self, peer: PeerInfo):
        """Register a newly discovered peer."""
        self.peers[peer.peer_id] = peer
        peer.last_seen = time.monotonic()
        self._emit("peer_joined", peer)
        logger.info("Peer joined: %s (%s)", peer.peer_id, peer.compute_type)

    def remove_peer(self, peer_id: str):
        """Remove a peer that has disconnected."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            self._emit("peer_left", peer_id)

    def run_training_round(
        self,
        round_id: Optional[str] = None,
        peer_gradients: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> KickstartResult:
        """Execute one training round.

        In a real P2P network, gradients come from other peers via gossipsub.
        For local/simulation mode, peer_gradients can be passed directly.

        Args:
            round_id: Round identifier. Auto-generated if None.
            peer_gradients: Gradients from other peers (for simulation).

        Returns:
            KickstartResult with training metrics and gradients.
        """
        if round_id is None:
            round_id = f"round-{self._stats.total_rounds}"

        self._emit("round_start", round_id)

        # Local training.
        result = self.kickstart.train_round(round_id, self.config.peer_id)

        if result.steps_completed == 0:
            logger.warning("Round %s: no training steps (insufficient data?)", round_id)
            return result

        # Aggregate with peer gradients if available.
        if peer_gradients and result.gradients:
            all_grads = [result.gradients]
            for pid, grads in peer_gradients.items():
                if self.reputation.is_allowed(pid):
                    all_grads.append(grads)

            if len(all_grads) > 1:
                byz_config = ByzantineConfig(
                    method=self.config.byzantine_method,
                    trim_ratio=self.config.byzantine_tolerance,
                )
                aggregated = robust_aggregate(all_grads, byz_config)
                self.kickstart.apply_aggregated_gradients(aggregated)
                logger.info(
                    "Round %s: aggregated gradients from %d peers",
                    round_id, len(all_grads),
                )

        # Update stats.
        self._stats.total_rounds += 1
        self._stats.total_steps += result.steps_completed
        self._stats.current_loss = result.avg_loss
        self._stats.tokens_processed += result.tokens_processed
        self._stats.latest_sample = result.sample_text
        self._stats.loss_history.append(result.avg_loss)
        if result.avg_loss < self._stats.best_loss:
            self._stats.best_loss = result.avg_loss
        self._stats.compute_hours = (
            (time.monotonic() - self._start_time) / 3600
        )

        # Record in genesis tracker -- this detects milestones.
        new_milestones_before = len(self.genesis.milestones)
        self.genesis.record_round(
            round_id=round_id,
            loss=result.avg_loss,
            sample_text=result.sample_text,
            tokens_processed=result.tokens_processed,
            peers=len(self.peers) + 1,
        )

        # Emit milestone events if any new ones were detected.
        for ms_event in self.genesis.milestones[new_milestones_before:]:
            self._emit("milestone", ms_event)
            logger.info(
                "MILESTONE: %s (age=%s, round=%s)",
                ms_event.description,
                self.genesis.age_str,
                round_id,
            )

        # Tick outcome tracker for governance settlement.
        self.outcome_tracker.tick_round(self.evolution.current_genome)

        # Award credits for this training round.
        rep_score = self.reputation.get_score(self.config.peer_id)
        earned = self.credits.earn_training_round(
            self.config.peer_id, round_id=round_id, reputation_score=rep_score,
        )
        self._emit("credits_earned", self.config.peer_id, earned)

        # Checkpoint if configured.
        if (
            self._stats.total_rounds % self.config.checkpoint_interval == 0
        ):
            self._save_checkpoint(round_id)

        self._emit("round_complete", round_id, result)
        self._emit("loss_update", round_id, result.avg_loss)

        logger.info(
            "Round %s: loss=%.4f, steps=%d, tokens=%d",
            round_id, result.avg_loss, result.steps_completed,
            result.tokens_processed,
        )

        return result

    def run_continuous(
        self,
        max_rounds: int = -1,
        callback: Optional[Callable[[KickstartResult], None]] = None,
    ):
        """Run continuous training rounds.

        Args:
            max_rounds: Maximum rounds to run (-1 = infinite).
            callback: Called after each round with the result.
        """
        self._running = True
        round_num = 0

        logger.info(
            "Starting continuous training (peer=%s, model=%s, %d params)",
            self.config.peer_id,
            self.kickstart.config.model_id,
            self.kickstart.model.num_parameters,
        )

        while self._running:
            if max_rounds > 0 and round_num >= max_rounds:
                break

            result = self.run_training_round()

            if callback:
                callback(result)

            round_num += 1

        logger.info("Training stopped after %d rounds", round_num)

    def stop(self):
        """Stop continuous training."""
        self._running = False

    def _save_checkpoint(self, round_id: str):
        """Save a checkpoint of the current model state."""
        state = self.kickstart.state_dict()
        ckpt_path = os.path.join(
            self.checkpoints.base_dir,
            f"{self.kickstart.config.model_id}_{round_id}.pt",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "model_state": state,
                "round_id": round_id,
                "peer_id": self.config.peer_id,
                "total_rounds": self._stats.total_rounds,
                "loss": self._stats.current_loss,
                "model_config": {
                    "model_id": self.kickstart.config.model_id,
                    "hidden_dim": self.kickstart.config.hidden_dim,
                    "n_layers": self.kickstart.config.n_layers,
                    "n_heads": self.kickstart.config.n_heads,
                    "vocab_size": self.kickstart.config.vocab_size,
                    "max_seq_length": self.kickstart.config.max_seq_length,
                },
            },
            ckpt_path,
        )
        self._emit("checkpoint_saved", ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

    def load_checkpoint(self, path: str):
        """Load model state from a checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.kickstart.load_state_dict(ckpt["model_state"])
        self._stats.total_rounds = ckpt.get("total_rounds", 0)
        logger.info(
            "Loaded checkpoint: %s (round %s, loss %.4f)",
            path,
            ckpt.get("round_id", "?"),
            ckpt.get("loss", float("inf")),
        )

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """Generate text from the current model."""
        return self.kickstart.generate(prompt, max_tokens, temperature)

    def get_stats_dict(self) -> dict:
        """Get stats as a plain dict (JSON-serializable)."""
        s = self.stats
        genesis_status = self.genesis.get_status()
        return {
            "peer_id": s.peer_id,
            "connected_peers": s.connected_peers,
            "total_rounds": s.total_rounds,
            "total_steps": s.total_steps,
            "current_loss": s.current_loss,
            "best_loss": s.best_loss,
            "tokens_processed": s.tokens_processed,
            "compute_hours": round(s.compute_hours, 2),
            "model_params": s.model_params,
            "model_size": s.model_size,
            "uptime_secs": round(s.uptime_secs, 1),
            "latest_sample": s.latest_sample[:200],
            "loss_history": s.loss_history[-100:],
            # Genesis data.
            "model_age": genesis_status["age_str"],
            "milestones_achieved": genesis_status["milestones_achieved"],
            "milestones": genesis_status["milestones"],
            "current_quality": genesis_status["current_quality"],
            "quality_history": genesis_status["quality_history"],
            "generation": genesis_status["generation"],
            "mutations": genesis_status["mutations"],
            # Credit data.
            "credit_balance": round(self.credits.get_balance(self.config.peer_id), 1),
            "credit_earned": round(
                self.credits.get_account(self.config.peer_id).total_earned, 1
            ),
            "credit_spent": round(
                self.credits.get_account(self.config.peer_id).total_spent, 1
            ),
            "credit_network": self.credits.network_stats(),
        }
