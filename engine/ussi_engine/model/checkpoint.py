"""Content-addressed checkpoint persistence.

Stores model checkpoints with content-addressed hashing so any peer can
verify checkpoint integrity without trusting the source. Checkpoints are
identified by their Merkle root, not by filename or path.

Storage layout:
    {base_dir}/
        cas/                      # Content-addressed store
            {merkle_root_hex}/    # One dir per checkpoint
                shard_0_4.pt      # Weight tensors
                metadata.json     # Round info, peer list, hashes
        latest/                   # Symlink to latest checkpoint per model
            {model_id} -> ../cas/{merkle_root_hex}

Future: replace local disk with IPFS for truly decentralized persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .shard import ModelShard

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside a checkpoint."""

    model_id: str
    round_id: str
    merkle_root: str
    timestamp: float
    n_peers: int = 0
    n_parameters: int = 0
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0
    aggregation_method: str = "mean"
    peer_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "round_id": self.round_id,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "n_peers": self.n_peers,
            "n_parameters": self.n_parameters,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "total_layers": self.total_layers,
            "aggregation_method": self.aggregation_method,
            "peer_ids": self.peer_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CheckpointStore:
    """Content-addressed checkpoint storage.

    Checkpoints are stored by their Merkle root hash, ensuring integrity
    verification is built into the storage layer itself.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.cas_dir = self.base_dir / "cas"
        self.latest_dir = self.base_dir / "latest"
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        shard: ModelShard,
        round_id: str,
        peer_ids: Optional[List[str]] = None,
        aggregation_method: str = "mean",
    ) -> str:
        """Save a shard checkpoint to the content-addressed store.

        Returns the Merkle root hex string (the checkpoint's content address).
        """
        merkle_root = shard.merkle_root().hex()
        ckpt_dir = self.cas_dir / merkle_root

        if ckpt_dir.exists():
            logger.debug("Checkpoint %s already exists (dedup)", merkle_root[:16])
            return merkle_root

        # Write to temp dir first, then atomic rename for crash safety.
        tmp_dir = self.cas_dir / f".tmp-{merkle_root[:16]}-{os.getpid()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save weights.
            config = shard.config
            weight_path = tmp_dir / f"shard_{config.layer_start}_{config.layer_end}.pt"
            torch.save(shard.state_dict(), weight_path)

            # Save metadata.
            metadata = CheckpointMetadata(
                model_id=config.model_id,
                round_id=round_id,
                merkle_root=merkle_root,
                timestamp=time.time(),
                n_peers=len(peer_ids) if peer_ids else 0,
                n_parameters=shard.num_parameters(),
                layer_start=config.layer_start,
                layer_end=config.layer_end,
                total_layers=config.total_layers,
                aggregation_method=aggregation_method,
                peer_ids=peer_ids or [],
            )
            meta_path = tmp_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Atomic rename.
            tmp_dir.rename(ckpt_dir)

            # Update latest symlink.
            latest_link = self.latest_dir / config.model_id
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(ckpt_dir)

            logger.info(
                "Saved checkpoint %s for model %s round %s (%d params)",
                merkle_root[:16], config.model_id, round_id, shard.num_parameters(),
            )

        except Exception:
            # Clean up temp dir on failure.
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return merkle_root

    def load(self, merkle_root: str, shard: ModelShard) -> Optional[ModelShard]:
        """Load a checkpoint by its Merkle root and apply to shard.

        Returns None if the checkpoint doesn't exist or verification fails.
        """
        ckpt_dir = self.cas_dir / merkle_root
        if not ckpt_dir.exists():
            logger.warning("Checkpoint %s not found", merkle_root[:16])
            return None

        config = shard.config
        weight_path = ckpt_dir / f"shard_{config.layer_start}_{config.layer_end}.pt"
        if not weight_path.exists():
            logger.warning("Weight file not found in checkpoint %s", merkle_root[:16])
            return None

        state_dict = torch.load(weight_path, weights_only=True)
        shard.load_state_dict(state_dict)

        # Verify integrity.
        actual_root = shard.merkle_root().hex()
        if actual_root != merkle_root:
            logger.error(
                "Checkpoint integrity FAILED: expected %s, got %s",
                merkle_root[:16], actual_root[:16],
            )
            return None

        logger.info("Loaded and verified checkpoint %s", merkle_root[:16])
        return shard

    def load_latest(self, model_id: str, shard: ModelShard) -> Optional[ModelShard]:
        """Load the latest checkpoint for a model."""
        latest_link = self.latest_dir / model_id
        if not latest_link.exists():
            return None

        ckpt_dir = latest_link.resolve()
        meta_path = ckpt_dir / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            meta = CheckpointMetadata.from_dict(json.load(f))

        return self.load(meta.merkle_root, shard)

    def get_metadata(self, merkle_root: str) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata without loading weights."""
        meta_path = self.cas_dir / merkle_root / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return CheckpointMetadata.from_dict(json.load(f))

    def list_checkpoints(self, model_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List all checkpoints, optionally filtered by model_id."""
        results = []
        if not self.cas_dir.exists():
            return results

        for ckpt_dir in self.cas_dir.iterdir():
            if not ckpt_dir.is_dir() or ckpt_dir.name.startswith("."):
                continue
            meta_path = ckpt_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = CheckpointMetadata.from_dict(json.load(f))
            if model_id is None or meta.model_id == model_id:
                results.append(meta)

        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results

    def verify(self, merkle_root: str, shard: ModelShard) -> bool:
        """Verify a checkpoint's integrity without modifying the shard.

        Loads weights into a temporary copy and checks the Merkle root.
        """
        ckpt_dir = self.cas_dir / merkle_root
        config = shard.config
        weight_path = ckpt_dir / f"shard_{config.layer_start}_{config.layer_end}.pt"
        if not weight_path.exists():
            return False

        state_dict = torch.load(weight_path, weights_only=True)

        # Compute Merkle root of loaded weights without modifying shard.
        leaf_hashes = []
        for name in sorted(state_dict.keys()):
            h = hashlib.sha256()
            h.update(name.encode())
            h.update(state_dict[name].detach().cpu().numpy().tobytes())
            leaf_hashes.append(h.digest())

        actual = _merkle_root_from_leaves(leaf_hashes).hex()
        return actual == merkle_root

    def gc(self, keep_latest: int = 5, model_id: Optional[str] = None):
        """Garbage-collect old checkpoints, keeping the N most recent."""
        checkpoints = self.list_checkpoints(model_id)
        if len(checkpoints) <= keep_latest:
            return

        to_remove = checkpoints[keep_latest:]
        for meta in to_remove:
            ckpt_dir = self.cas_dir / meta.merkle_root
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                logger.info("GC: removed checkpoint %s", meta.merkle_root[:16])


def _merkle_root_from_leaves(leaves: List[bytes]) -> bytes:
    """Compute binary Merkle tree root from leaf hashes."""
    if not leaves:
        return b"\x00" * 32

    layer = list(leaves)
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                combined = hashlib.sha256(b"\x01" + layer[i] + layer[i + 1]).digest()
            else:
                combined = layer[i]
            next_layer.append(combined)
        layer = next_layer
    return layer[0]
