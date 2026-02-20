"""Pipeline parallelism -- routing activations across peer-held shards."""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .shard import ModelShard, ShardConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents one stage in the inference/training pipeline.

    Each stage corresponds to one peer's model shard.
    """

    peer_id: str
    shard: ModelShard
    # Callable that sends activations to the next stage (over network).
    # Signature: send(tensor_bytes, shape, dtype, target_peer) -> response_bytes
    send_fn: Optional[Callable] = None
    # Callable that receives activations from the previous stage.
    recv_fn: Optional[Callable] = None


class PipelineExecutor:
    """Executes forward passes across a pipeline of shards.

    For local execution (all shards on one machine, e.g. testing), the
    executor chains forward() calls directly. For distributed execution,
    it uses send_fn/recv_fn to route activations to peer nodes.
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    @classmethod
    def local(cls, shards: List[ModelShard]) -> "PipelineExecutor":
        """Create a pipeline executor where all stages run locally."""
        stages = [
            PipelineStage(peer_id=f"local-{i}", shard=shard)
            for i, shard in enumerate(shards)
        ]
        return cls(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass through all pipeline stages.

        For local stages, this calls shard.forward() directly.
        For remote stages, this serializes the tensor and sends it.
        """
        activation = x
        for stage in self.stages:
            if stage.send_fn is not None and stage.recv_fn is not None:
                # Remote stage: serialize, send, receive result.
                tensor_bytes = serialize_tensor(activation)
                shape = list(activation.shape)
                dtype = str(activation.dtype)
                response = stage.send_fn(tensor_bytes, shape, dtype, stage.peer_id)
                activation = deserialize_tensor(response, shape, dtype)
            else:
                # Local stage: run forward directly.
                activation = stage.shard.forward(activation)
        return activation

    async def forward_async(self, x: torch.Tensor) -> torch.Tensor:
        """Async version that can await network sends."""
        activation = x
        for stage in self.stages:
            if stage.send_fn is not None:
                tensor_bytes = serialize_tensor(activation)
                shape = list(activation.shape)
                dtype = str(activation.dtype)
                if asyncio.iscoroutinefunction(stage.send_fn):
                    response = await stage.send_fn(
                        tensor_bytes, shape, dtype, stage.peer_id
                    )
                else:
                    response = stage.send_fn(
                        tensor_bytes, shape, dtype, stage.peer_id
                    )
                activation = deserialize_tensor(response, shape, dtype)
            else:
                activation = stage.shard.forward(activation)
        return activation

    @property
    def num_stages(self) -> int:
        return len(self.stages)


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a PyTorch tensor to bytes for network transmission."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def deserialize_tensor(
    data: bytes,
    shape: Optional[List[int]] = None,
    dtype: Optional[str] = None,
) -> torch.Tensor:
    """Deserialize a PyTorch tensor from bytes."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)
