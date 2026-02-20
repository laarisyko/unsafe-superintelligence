"""Model shard management -- splitting and holding subsets of a model's layers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ShardConfig:
    """Configuration for a single shard of a model."""

    model_id: str
    layer_start: int  # inclusive
    layer_end: int  # exclusive
    total_layers: int
    dtype: torch.dtype = torch.float32

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    @property
    def is_first(self) -> bool:
        return self.layer_start == 0

    @property
    def is_last(self) -> bool:
        return self.layer_end == self.total_layers


class ModelShard:
    """Holds a contiguous slice of a model's transformer layers.

    This is the fundamental unit of pipeline parallelism. Each peer in the
    network holds one or more shards and executes forward/backward passes
    on its local layers only.
    """

    def __init__(self, config: ShardConfig, layers: Optional[nn.ModuleList] = None):
        self.config = config
        self.layers = layers or nn.ModuleList()
        self._device = torch.device("cpu")

    @classmethod
    def from_model(cls, model: nn.Module, config: ShardConfig) -> "ModelShard":
        """Extract a shard from a full model by slicing its layer list."""
        all_layers = _extract_layers(model)
        shard_layers = nn.ModuleList(all_layers[config.layer_start : config.layer_end])
        return cls(config=config, layers=shard_layers)

    def to(self, device: torch.device) -> "ModelShard":
        """Move shard to a device (CPU/GPU)."""
        self._device = device
        self.layers = self.layers.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through this shard's layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Iterate over all parameters in this shard."""
        return self.layers.parameters()

    def named_parameters(self):
        return self.layers.named_parameters()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.layers.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.layers.load_state_dict(state_dict)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def merkle_root(self) -> bytes:
        """Compute a Merkle root over the shard's weight tensors.

        Used to verify weight consistency between peers after training.
        """
        leaves = []
        for name, param in sorted(self.named_parameters()):
            h = hashlib.sha256()
            h.update(name.encode())
            h.update(param.detach().cpu().numpy().tobytes())
            leaves.append(h.digest())

        return _merkle_root(leaves)


def split_model(model: nn.Module, model_id: str, n_shards: int) -> List[ModelShard]:
    """Split a model into `n_shards` roughly equal shards for pipeline parallelism."""
    all_layers = _extract_layers(model)
    total = len(all_layers)
    if n_shards > total:
        raise ValueError(f"Cannot split {total} layers into {n_shards} shards")

    chunk_size = total // n_shards
    remainder = total % n_shards
    shards = []
    start = 0

    for i in range(n_shards):
        end = start + chunk_size + (1 if i < remainder else 0)
        config = ShardConfig(
            model_id=model_id,
            layer_start=start,
            layer_end=end,
            total_layers=total,
        )
        shard_layers = nn.ModuleList(all_layers[start:end])
        shards.append(ModelShard(config=config, layers=shard_layers))
        start = end

    return shards


def _extract_layers(model: nn.Module) -> List[nn.Module]:
    """Extract the list of sequential layers from a model.

    Supports models with a `.layers` attribute, a `.transformer.h` attribute
    (GPT-style), or falls back to all direct children.
    """
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    return list(model.children())


def _merkle_root(leaves: List[bytes]) -> bytes:
    """Compute a binary Merkle tree root from leaf hashes."""
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
