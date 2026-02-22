"""Pipeline inference executor -- coordinates multi-peer inference."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional

import torch

from ..model.shard import ModelShard
from ..model.pipeline import serialize_tensor, deserialize_tensor

logger = logging.getLogger(__name__)


class PipelineInferenceExecutor:
    """Coordinates inference across a pipeline of peers.

    When a request arrives at any peer, this executor:
    1. Tokenizes the prompt (if this is the first stage).
    2. Runs the local shard forward pass.
    3. Sends activations to the next peer in the pipeline.
    4. Receives the final output from the last peer.
    """

    def __init__(
        self,
        local_shard: ModelShard,
        pipeline_order: List[str],  # ordered peer_ids
        local_peer_id: str,
        send_activation_fn: Optional[Callable] = None,
        recv_activation_fn: Optional[Callable] = None,
    ):
        self.local_shard = local_shard
        self.pipeline_order = pipeline_order
        self.local_peer_id = local_peer_id
        self.send_activation_fn = send_activation_fn
        self.recv_activation_fn = recv_activation_fn

    @property
    def local_stage_index(self) -> int:
        try:
            return self.pipeline_order.index(self.local_peer_id)
        except ValueError:
            return -1

    @property
    def is_first_stage(self) -> bool:
        return self.local_stage_index == 0

    @property
    def is_last_stage(self) -> bool:
        return self.local_stage_index == len(self.pipeline_order) - 1

    @property
    def next_peer(self) -> Optional[str]:
        idx = self.local_stage_index
        if idx < 0 or idx >= len(self.pipeline_order) - 1:
            return None
        return self.pipeline_order[idx + 1]

    @property
    def prev_peer(self) -> Optional[str]:
        idx = self.local_stage_index
        if idx <= 0:
            return None
        return self.pipeline_order[idx - 1]

    def run(self, prompt: str) -> str:
        """Synchronous pipeline inference.

        If this peer is the first stage, create input from prompt.
        If this peer is an intermediate stage, receive from prev and send to next.
        If this peer is the last stage, return the output text.
        """
        self.local_shard.layers.eval()

        with torch.no_grad():
            if self.is_first_stage:
                # Create input (placeholder tokenization).
                x = self._tokenize(prompt)
            elif self.recv_activation_fn:
                data = self.recv_activation_fn(self.prev_peer)
                x = deserialize_tensor(data)
            else:
                x = self._tokenize(prompt)

            # Forward through local shard.
            output = self.local_shard.forward(x)

            if self.is_last_stage:
                return self._detokenize(output)
            elif self.send_activation_fn and self.next_peer:
                data = serialize_tensor(output)
                self.send_activation_fn(data, self.next_peer)
                return "[forwarded to next stage]"
            else:
                return f"[stage {self.local_stage_index} output shape: {list(output.shape)}]"

    async def run_async(self, prompt: str) -> str:
        """Async pipeline inference with network awaits."""
        self.local_shard.layers.eval()

        with torch.no_grad():
            if self.is_first_stage:
                x = self._tokenize(prompt)
            elif self.recv_activation_fn:
                data = await _maybe_await(self.recv_activation_fn(self.prev_peer))
                x = deserialize_tensor(data)
            else:
                x = self._tokenize(prompt)

            output = self.local_shard.forward(x)

            if self.is_last_stage:
                return self._detokenize(output)
            elif self.send_activation_fn and self.next_peer:
                data = serialize_tensor(output)
                await _maybe_await(self.send_activation_fn(data, self.next_peer))
                return "[forwarded to next stage]"
            else:
                return f"[stage {self.local_stage_index} output]"

    def process_activation(self, tensor_bytes: bytes) -> bytes:
        """Process an incoming activation chunk from the previous stage.

        This is called when this peer receives activations over the network.
        It runs the local forward pass and returns the output bytes.
        """
        self.local_shard.layers.eval()
        with torch.no_grad():
            x = deserialize_tensor(tensor_bytes)
            output = self.local_shard.forward(x)
            return serialize_tensor(output)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Placeholder tokenization. In production, use a real tokenizer."""
        # Encode prompt characters as a simple embedding.
        token_ids = [ord(c) % 256 for c in prompt[:128]]
        while len(token_ids) < 128:
            token_ids.append(0)
        x = torch.tensor(token_ids, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # Expand to a reasonable hidden dimension.
        x = x.expand(-1, -1, 512)
        return x.to(self.local_shard.device)

    def _detokenize(self, output: torch.Tensor) -> str:
        """Placeholder detokenization."""
        # Take argmax of last token's output as placeholder.
        if output.dim() >= 2:
            last = output[0, -1]
            return f"[output dim={output.shape[-1]}, max={last.max().item():.4f}]"
        return f"[output: {output.shape}]"


async def _maybe_await(val):
    if asyncio.iscoroutine(val):
        return await val
    return val
