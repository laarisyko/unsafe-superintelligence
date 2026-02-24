"""Inference request handler -- receives prompts and routes through the model."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..data.tokenizer import BOS_TOKEN, EOS_TOKEN, Tokenizer, TokenizerConfig
from ..model.shard import ModelShard
from .pipeline_exec import PipelineInferenceExecutor

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """An inference request from a client or another peer."""

    request_id: str = ""
    model_id: str = ""
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class InferenceResponse:
    """Response from inference."""

    request_id: str = ""
    model_id: str = ""
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class InferenceServer:
    """Handles inference requests for models hosted on this peer.

    If the peer holds the full model, it serves directly. If the model is
    pipeline-sharded across peers, it coordinates with other nodes via the
    shard map to route activations through the pipeline.
    """

    def __init__(self):
        self._models: Dict[str, ModelShard] = {}
        self._pipelines: Dict[str, PipelineInferenceExecutor] = {}
        self._request_count = 0
        self._active_requests: Dict[str, float] = {}
        self._tokenizer = Tokenizer(TokenizerConfig(mode="byte", vocab_size=260, max_sequence_length=512))
        self._embedding_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._projection_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._seed_cache: Dict[str, int] = {}

    def register_shard(self, model_id: str, shard: ModelShard):
        """Register a local model shard for inference."""
        self._models[model_id] = shard
        logger.info(
            "Registered shard for model %s (layers %d-%d, %d params)",
            model_id,
            shard.config.layer_start,
            shard.config.layer_end,
            shard.num_parameters(),
        )

    def register_pipeline(self, model_id: str, executor: PipelineInferenceExecutor):
        """Register a pipeline executor for distributed inference."""
        self._pipelines[model_id] = executor

    @property
    def current_load(self) -> float:
        """Normalized load from 0.0 (idle) to 1.0 (fully loaded)."""
        if not self._active_requests:
            return 0.0
        return min(1.0, len(self._active_requests) / 10.0)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Synchronous inference for a single request."""
        start = time.monotonic()
        self._request_count += 1
        self._active_requests[request.request_id] = start

        try:
            prompt_tokens = len(self._tokenizer.encode(request.prompt, add_special=False))
            completion_tokens = 0
            if request.model_id in self._pipelines:
                text = self._pipelines[request.model_id].run(request.prompt)
                completion_tokens = len(self._tokenizer.encode(text, add_special=False))
            elif request.model_id in self._models:
                shard = self._models[request.model_id]
                text, prompt_tokens, completion_tokens = self._infer_local(shard, request)
            else:
                text = f"[model {request.model_id} not found on this peer]"

            elapsed_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
            )
        finally:
            self._active_requests.pop(request.request_id, None)

    async def infer_async(self, request: InferenceRequest) -> InferenceResponse:
        """Async inference -- useful when pipeline stages involve network calls."""
        start = time.monotonic()
        self._request_count += 1
        self._active_requests[request.request_id] = start

        try:
            prompt_tokens = len(self._tokenizer.encode(request.prompt, add_special=False))
            completion_tokens = 0
            if request.model_id in self._pipelines:
                text = await self._pipelines[request.model_id].run_async(request.prompt)
                completion_tokens = len(self._tokenizer.encode(text, add_special=False))
            elif request.model_id in self._models:
                shard = self._models[request.model_id]
                text, prompt_tokens, completion_tokens = self._infer_local(shard, request)
            else:
                text = f"[model {request.model_id} not found on this peer]"

            elapsed_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
            )
        finally:
            self._active_requests.pop(request.request_id, None)

    def _infer_local(self, shard: ModelShard, request: InferenceRequest) -> tuple[str, int, int]:
        """Run autoregressive text generation on a local shard."""
        prompt_tokens = self._tokenizer.encode(request.prompt, add_special=False)
        if not prompt_tokens:
            prompt_tokens = [BOS_TOKEN]

        generated_tokens: List[int] = []
        sequence = list(prompt_tokens)
        input_dim = self._infer_input_dim(shard)
        seed = self._model_seed(request.model_id, shard)
        dtype = self._parameter_dtype(shard)

        shard.layers.eval()
        with torch.no_grad():
            embedding = self._embedding_matrix(
                request.model_id, input_dim, seed, shard.device, dtype
            )

            max_steps = max(0, int(request.max_tokens))
            for _ in range(max_steps):
                window = sequence[-self._tokenizer.max_length :]
                token_ids = torch.tensor([window], dtype=torch.long, device=shard.device)
                x = embedding[token_ids]
                hidden = shard.forward(x)
                logits = hidden[:, -1, :]
                projection = self._projection_matrix(
                    request.model_id, logits.shape[-1], seed, shard.device, logits.dtype
                )
                token_logits = logits @ projection
                next_token = self._sample_next_token(
                    token_logits[0],
                    request,
                    allow_eos=bool(generated_tokens),
                )
                if next_token == EOS_TOKEN:
                    break
                sequence.append(next_token)
                generated_tokens.append(next_token)

        text = self._tokenizer.decode(generated_tokens) if generated_tokens else ""
        return text, len(prompt_tokens), len(generated_tokens)

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        request: InferenceRequest,
        allow_eos: bool = True,
    ) -> int:
        scores = logits.clone()
        if scores.shape[-1] >= 4:
            scores[0] = float("-inf")  # PAD
            scores[1] = float("-inf")  # BOS
            scores[3] = float("-inf")  # UNK
        if not allow_eos and scores.shape[-1] > EOS_TOKEN:
            scores[EOS_TOKEN] = float("-inf")

        temperature = float(request.temperature)
        if temperature <= 0:
            return int(torch.argmax(scores).item())
        scores = scores / max(temperature, 1e-6)

        top_k = int(request.top_k)
        if 0 < top_k < scores.shape[-1]:
            top_values, _ = torch.topk(scores, k=top_k)
            threshold = top_values[-1]
            scores = torch.where(scores < threshold, torch.full_like(scores, float("-inf")), scores)

        probs = torch.softmax(scores, dim=-1)
        top_p = float(request.top_p)
        if 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative > top_p
            cutoff_mask[0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
            denom = sorted_probs.sum()
            if denom > 0:
                sorted_probs = sorted_probs / denom
                selected = torch.multinomial(sorted_probs, num_samples=1)
                token = sorted_idx[selected]
                return int(token.item())

        return int(torch.multinomial(probs, num_samples=1).item())

    def _model_seed(self, model_id: str, shard: ModelShard) -> int:
        if model_id in self._seed_cache:
            return self._seed_cache[model_id]
        digest = hashlib.sha256(shard.merkle_root() + model_id.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        self._seed_cache[model_id] = seed
        return seed

    def _embedding_matrix(
        self,
        model_id: str,
        dim: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (model_id, dim)
        mat = self._embedding_cache.get(key)
        if mat is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed ^ (dim << 1) ^ 0xA5A5A5A5)
            mat = torch.randn(
                self._tokenizer.vocab_size,
                dim,
                generator=generator,
                dtype=torch.float32,
            ) * 0.02
            self._embedding_cache[key] = mat
        return mat.to(device=device, dtype=dtype)

    def _projection_matrix(
        self,
        model_id: str,
        dim: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (model_id, dim)
        mat = self._projection_cache.get(key)
        if mat is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed ^ (dim << 3) ^ 0x5A5A5A5A)
            mat = torch.randn(
                dim,
                self._tokenizer.vocab_size,
                generator=generator,
                dtype=torch.float32,
            ) * 0.02
            self._projection_cache[key] = mat
        return mat.to(device=device, dtype=dtype)

    @staticmethod
    def _parameter_dtype(shard: ModelShard) -> torch.dtype:
        for param in shard.parameters():
            return param.dtype
        return torch.float32

    @staticmethod
    def _infer_input_dim(shard: ModelShard) -> int:
        for layer in shard.layers:
            for module in layer.modules():
                if hasattr(module, "in_features"):
                    return int(getattr(module, "in_features"))
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                    return int(weight.shape[1])
        for param in shard.parameters():
            if param.dim() >= 2:
                return int(param.shape[-1])
            if param.dim() == 1:
                return int(param.shape[0])
        return 256

    def list_models(self) -> List[str]:
        """List all model IDs available on this peer."""
        models = set(self._models.keys()) | set(self._pipelines.keys())
        return sorted(models)

    def stats(self) -> Dict:
        """Return server statistics."""
        return {
            "total_requests": self._request_count,
            "active_requests": len(self._active_requests),
            "registered_models": len(self._models),
            "registered_pipelines": len(self._pipelines),
            "current_load": self.current_load,
        }
