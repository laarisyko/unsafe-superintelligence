"""Inference client API -- send prompts to the decentralized network."""

from __future__ import annotations

import asyncio
import logging
import uuid

from .network import NetworkClient

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for running inference on models in the SSSI network.

    Rate limits are enforced by the Agent class (not here) so that
    InferenceClient stays a pure network client.
    """

    def __init__(self, network: NetworkClient):
        self.network = network

    def infer(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Run synchronous inference.

        Returns:
            Generated text.
        """
        request_id = str(uuid.uuid4())
        result = self.network.submit_inference(
            model_id=model_id,
            prompt=prompt,
            request_id=request_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if "error" in result:
            error = result["error"]
            # Surface rate limit errors with a helpful message
            if "rate_limit" in str(error).lower():
                logger.warning(
                    "Rate limited. Contribute compute to unlock unlimited access: "
                    "sssi join --gpu-memory 8GB --accelerator cuda"
                )
            else:
                logger.error("Inference failed: %s", error)
            return f"[error: {error}]"

        return result.get("text", "[no text in response]")

    async def infer_async(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Async inference (runs sync call in executor to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(model_id, prompt, max_tokens, temperature),
        )

    def list_models(self) -> list:
        """List available models on the network."""
        result = self.network.models()
        if isinstance(result, list):
            return result
        # Fallback: derive from shard map
        shards = self.network.shards()
        if isinstance(shards, dict) and "entries" in shards:
            model_ids = set()
            for entry in shards["entries"].values():
                if "model_id" in entry:
                    model_ids.add(entry["model_id"])
            return sorted(model_ids)
        return []
