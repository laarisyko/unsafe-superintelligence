"""Pipeline inference executor -- coordinates multi-peer inference."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Callable, Dict, List, Optional

import torch

from ..data.tokenizer import BOS_TOKEN, Tokenizer, TokenizerConfig
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
        self.tokenizer = Tokenizer(TokenizerConfig(mode="byte", vocab_size=260, max_sequence_length=512))
        self._seed = int.from_bytes(
            hashlib.sha256(local_shard.merkle_root() + local_shard.config.model_id.encode("utf-8")).digest()[:8],
            "big",
        )
        self._embedding_cache: Dict[int, torch.Tensor] = {}
        self._projection_cache: Dict[int, torch.Tensor] = {}

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
                x = self._tokenize(prompt)
            elif self.recv_activation_fn:
                data = self.recv_activation_fn(self.prev_peer)
                x = deserialize_tensor(data)
            else:
                x = self._tokenize(prompt)

            output = self.local_shard.forward(x)

            if self.is_last_stage:
                return self._detokenize(output)
            elif self.send_activation_fn and self.next_peer:
                data = serialize_tensor(output)
                self.send_activation_fn(data, self.next_peer)
                return f"forwarded:{self.next_peer}:{len(data)}"
            else:
                return self._activation_summary(output)

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
                return f"forwarded:{self.next_peer}:{len(data)}"
            else:
                return self._activation_summary(output)

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
        """Encode prompt bytes into deterministic embeddings."""
        token_ids = self.tokenizer.encode(prompt, add_special=False)
        if not token_ids:
            token_ids = [BOS_TOKEN]
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.local_shard.device)
        input_dim = self._infer_input_dim()
        embedding = self._embedding_matrix(input_dim, self._parameter_dtype())
        return embedding[ids]

    def _detokenize(self, output: torch.Tensor) -> str:
        """Project final activations to token space and decode to UTF-8 text."""
        if output.dim() == 2:
            output = output.unsqueeze(0)
        if output.dim() < 3:
            return ""

        projection = self._projection_matrix(output.shape[-1], output.dtype)
        logits = output @ projection
        if logits.shape[-1] >= 4:
            logits[..., :4] = float("-inf")
        token_ids = torch.argmax(logits[0], dim=-1).tolist()
        text = self.tokenizer.decode(token_ids)
        if text:
            return text
        return "tokens:" + ",".join(str(t) for t in token_ids[:8])

    def _activation_summary(self, output: torch.Tensor) -> str:
        digest = hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return f"activation:{digest}:{list(output.shape)}"

    def _embedding_matrix(self, dim: int, dtype: torch.dtype) -> torch.Tensor:
        mat = self._embedding_cache.get(dim)
        if mat is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed ^ (dim << 1) ^ 0xA5A5A5A5)
            mat = torch.randn(
                self.tokenizer.vocab_size,
                dim,
                generator=generator,
                dtype=torch.float32,
            ) * 0.02
            self._embedding_cache[dim] = mat
        return mat.to(device=self.local_shard.device, dtype=dtype)

    def _projection_matrix(self, dim: int, dtype: torch.dtype) -> torch.Tensor:
        mat = self._projection_cache.get(dim)
        if mat is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed ^ (dim << 3) ^ 0x5A5A5A5A)
            mat = torch.randn(
                dim,
                self.tokenizer.vocab_size,
                generator=generator,
                dtype=torch.float32,
            ) * 0.02
            self._projection_cache[dim] = mat
        return mat.to(device=self.local_shard.device, dtype=dtype)

    def _parameter_dtype(self) -> torch.dtype:
        for param in self.local_shard.parameters():
            return param.dtype
        return torch.float32

    def _infer_input_dim(self) -> int:
        for layer in self.local_shard.layers:
            for module in layer.modules():
                if hasattr(module, "in_features"):
                    return int(getattr(module, "in_features"))
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return int(weight.shape[1])
        for param in self.local_shard.parameters():
            if param.dim() >= 2:
                return int(param.shape[-1])
            if param.dim() == 1:
                return int(param.shape[0])
        return 256


async def _maybe_await(val):
    if asyncio.iscoroutine(val):
        return await val
    return val
