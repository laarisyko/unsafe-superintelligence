"""Training participation API -- lets agents join decentralized training rounds.

Supports both flat ring all-reduce (for small peer counts) and hierarchical
tree-of-rings aggregation (for large-scale training with 1000+ peers).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .network import NetworkClient

logger = logging.getLogger(__name__)


class TrainingParticipant:
    """Manages this agent's participation in decentralized training rounds."""

    def __init__(self, network: NetworkClient, agent_id: str):
        self.network = network
        self.agent_id = agent_id
        self._active_round: Optional[str] = None

    def propose_round(
        self,
        model_id: str,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_steps: int = 100,
        cluster_size: int = 1000,
        hierarchical: bool = True,
    ) -> str:
        """Propose a new training round to the network.

        Args:
            model_id: Model to train.
            learning_rate: Learning rate for the round.
            batch_size: Batch size per peer.
            num_steps: Training steps per round.
            cluster_size: Max peers per cluster in hierarchical mode.
            hierarchical: Use hierarchical aggregation (recommended for >64 peers).

        Returns:
            The round_id of the proposed round.
        """
        round_id = f"round-{uuid.uuid4().hex[:12]}"
        proposal = {
            "type": "proposal",
            "round_id": round_id,
            "model_id": model_id,
            "proposer": self.agent_id,
            "hyper_params": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_steps": num_steps,
            },
            "aggregation": {
                "mode": "hierarchical" if hierarchical else "flat",
                "cluster_size": cluster_size,
            },
        }
        self.network.publish("openclaw/training", proposal)
        self._active_round = round_id
        logger.info("Proposed training round %s for model %s", round_id, model_id)
        return round_id

    def join_round(self, round_id: str, gpu_memory_mb: int = 0, accelerator: str = "cpu"):
        """Join an existing training round.

        Args:
            round_id: Round to join.
            gpu_memory_mb: Advertised GPU memory.
            accelerator: Accelerator type (cpu, cuda, rocm, tpu).
        """
        join_msg = {
            "type": "join",
            "round_id": round_id,
            "peer_id": self.agent_id,
            "capacity": {
                "gpu_memory_mb": gpu_memory_mb,
                "accelerator": accelerator,
            },
        }
        self.network.publish("openclaw/training", join_msg)
        self._active_round = round_id
        logger.info("Joined training round %s", round_id)

    def join_training(
        self,
        model_id: str,
        num_rounds: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        cluster_size: int = 1000,
    ):
        """Convenience: propose and participate in training rounds."""
        for i in range(num_rounds):
            round_id = self.propose_round(
                model_id=model_id,
                learning_rate=learning_rate,
                batch_size=batch_size,
                cluster_size=cluster_size,
            )
            logger.info("Training round %d/%d: %s", i + 1, num_rounds, round_id)

    def announce_gradient_ready(self, round_id: str, merkle_root: str, cluster_id: int = -1):
        """Announce that local gradients are ready for aggregation.

        Args:
            round_id: Training round.
            merkle_root: Hash of gradient tensors.
            cluster_id: Cluster ID for scoped announcement (-1 for global).
        """
        msg = {
            "type": "gradient_ready",
            "round_id": round_id,
            "peer_id": self.agent_id,
            "merkle_root": merkle_root,
            "cluster_id": cluster_id,
        }
        # Use cluster-scoped topic if cluster_id is specified.
        if cluster_id >= 0:
            topic = f"openclaw/cluster-gradient/L0-C{cluster_id}"
        else:
            topic = "openclaw/gradient"
        self.network.publish(topic, msg)

    def announce_checkpoint(self, round_id: str, weights_merkle_root: str, cid: str = ""):
        """Announce a completed checkpoint after training."""
        msg = {
            "type": "checkpoint",
            "round_id": round_id,
            "peer_id": self.agent_id,
            "weights_merkle_root": weights_merkle_root,
            "checkpoint_cid": cid,
        }
        self.network.publish("openclaw/checkpoint", msg)

    @property
    def active_round(self) -> Optional[str]:
        return self._active_round
