"""Inference request handler -- receives prompts and routes through the model."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

import torch

from ..model.shard import ModelShard
from ..model.pipeline import PipelineExecutor, PipelineStage
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
            if request.model_id in self._pipelines:
                text = self._pipelines[request.model_id].run(request.prompt)
            elif request.model_id in self._models:
                shard = self._models[request.model_id]
                text = self._infer_local(shard, request)
            else:
                text = f"[model {request.model_id} not found on this peer]"

            elapsed_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                text=text,
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
            if request.model_id in self._pipelines:
                text = await self._pipelines[request.model_id].run_async(request.prompt)
            elif request.model_id in self._models:
                shard = self._models[request.model_id]
                text = self._infer_local(shard, request)
            else:
                text = f"[model {request.model_id} not found on this peer]"

            elapsed_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                text=text,
                latency_ms=elapsed_ms,
            )
        finally:
            self._active_requests.pop(request.request_id, None)

    def _infer_local(self, shard: ModelShard, request: InferenceRequest) -> str:
        """Run inference on a local shard.

        This is a simplified version: in production, we'd have a proper
        tokenizer and generation loop. Here we just do a forward pass.
        """
        shard.layers.eval()
        with torch.no_grad():
            # Detect hidden dimension from the first parameter.
            hidden_dim = 512
            for p in shard.parameters():
                hidden_dim = p.shape[-1]
                break
            dummy_input = torch.randn(1, 16, hidden_dim).to(shard.device)
            output = shard.forward(dummy_input)
            return f"[shard output shape: {list(output.shape)}]"

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
