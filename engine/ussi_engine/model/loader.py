"""Weight loading and saving -- checkpoint management for model shards."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from .shard import ModelShard, ShardConfig

logger = logging.getLogger(__name__)


class WeightLoader:
    """Load model weights from disk into a shard."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)

    def load_shard(self, config: ShardConfig, shard: ModelShard) -> ModelShard:
        """Load weights for the given shard configuration from a checkpoint."""
        path = self._shard_path(config)
        if not path.exists():
            logger.warning("No checkpoint found at %s", path)
            return shard

        state_dict = torch.load(path, weights_only=True)
        shard.load_state_dict(state_dict)
        logger.info(
            "Loaded shard %s layers [%d, %d) from %s",
            config.model_id,
            config.layer_start,
            config.layer_end,
            path,
        )
        return shard

    def load_metadata(self, model_id: str) -> Optional[Dict]:
        """Load checkpoint metadata (round_id, merkle_root, etc.)."""
        meta_path = self.checkpoint_dir / model_id / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def _shard_path(self, config: ShardConfig) -> Path:
        return (
            self.checkpoint_dir
            / config.model_id
            / f"shard_{config.layer_start}_{config.layer_end}.pt"
        )


class WeightSaver:
    """Save model weights from a shard to disk."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save_shard(
        self,
        shard: ModelShard,
        round_id: Optional[str] = None,
    ) -> Path:
        """Save a shard's weights to disk. Returns the path."""
        config = shard.config
        shard_dir = self.checkpoint_dir / config.model_id
        shard_dir.mkdir(parents=True, exist_ok=True)

        path = shard_dir / f"shard_{config.layer_start}_{config.layer_end}.pt"
        torch.save(shard.state_dict(), path)

        # Save metadata alongside.
        merkle = shard.merkle_root()
        metadata = {
            "model_id": config.model_id,
            "layer_start": config.layer_start,
            "layer_end": config.layer_end,
            "total_layers": config.total_layers,
            "num_parameters": shard.num_parameters(),
            "merkle_root": merkle.hex(),
            "round_id": round_id,
        }
        meta_path = shard_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved shard to %s (merkle: %s)", path, merkle.hex()[:16])
        return path

    def save_checkpoint(
        self,
        shards: list[ModelShard],
        round_id: str,
    ) -> Path:
        """Save all local shards as a single checkpoint."""
        for shard in shards:
            self.save_shard(shard, round_id=round_id)

        model_id = shards[0].config.model_id if shards else "unknown"
        return self.checkpoint_dir / model_id
