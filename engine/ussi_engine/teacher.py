"""Teacher model interface for SOTA model integration.

Provides a unified interface for calling any SOTA model (Claude, GPT-4, local)
for synthetic data generation, knowledge distillation, and DPO training.

Each peer can use whatever API keys they have — the network benefits from
diverse teacher access since gradients are shared through aggregation.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for a teacher model."""

    provider: str = "local"  # "anthropic", "openai", "local"
    model: str = ""          # e.g. "claude-sonnet-4-20250514", "gpt-4"
    api_key: str = ""        # reads from env var if empty
    api_base: str = ""       # custom endpoint
    max_tokens: int = 1024
    temperature: float = 0.7
    requests_per_minute: int = 30


class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float):
        """rate: requests per second."""
        self.rate = rate
        self.tokens = rate
        self.last_refill = time.monotonic()

    def acquire(self):
        """Block until a token is available."""
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # Sleep until next token available.
            sleep_time = (1.0 - self.tokens) / self.rate
            time.sleep(sleep_time)


class TeacherModel(ABC):
    """Abstract interface for calling any SOTA model."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        rate = max(config.requests_per_minute / 60.0, 0.1)
        self._limiter = TokenBucket(rate)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def log_probs(self, prompt: str, completion: str) -> List[float]:
        """Get log probabilities for a completion given a prompt.

        Returns per-token log probs for the completion tokens.
        For providers without native log_probs support, uses sampling approximation.
        """
        ...

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        ...


class AnthropicTeacher(TeacherModel):
    """Teacher using Anthropic's API (Claude models)."""

    def __init__(self, config: TeacherConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_base = config.api_base or "https://api.anthropic.com"
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key in config."
            )

    def generate(self, prompt: str, **kwargs) -> str:
        import httpx

        self._limiter.acquire()
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        response = httpx.post(
            f"{self.api_base}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    def log_probs(self, prompt: str, completion: str) -> List[float]:
        # Anthropic API doesn't expose log_probs directly.
        # Use sampling-based approximation: generate N completions and
        # estimate per-byte probabilities from frequency.
        return self._sampling_log_probs(prompt, completion)

    def _sampling_log_probs(self, prompt: str, completion: str, n_samples: int = 5) -> List[float]:
        """Approximate log probs via sampling."""
        import math

        completion_bytes = completion.encode("utf-8")
        # Generate samples and count byte matches.
        samples = self.batch_generate([prompt] * n_samples, max_tokens=len(completion) * 2)

        log_probs = []
        for i, byte in enumerate(completion_bytes):
            match_count = 0
            for sample in samples:
                sample_bytes = sample.encode("utf-8")
                if i < len(sample_bytes) and sample_bytes[i] == byte:
                    match_count += 1
            # Laplace smoothing.
            prob = (match_count + 1) / (n_samples + 256)
            log_probs.append(math.log(prob))

        return log_probs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]


class OpenAITeacher(TeacherModel):
    """Teacher using OpenAI's API (GPT-4, etc.)."""

    def __init__(self, config: TeacherConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = config.api_base or "https://api.openai.com"
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key in config."
            )

    def generate(self, prompt: str, **kwargs) -> str:
        import httpx

        self._limiter.acquire()
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        response = httpx.post(
            f"{self.api_base}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def log_probs(self, prompt: str, completion: str) -> List[float]:
        # Use OpenAI's logprobs parameter if available, otherwise sampling.
        try:
            return self._native_log_probs(prompt, completion)
        except Exception:
            return self._sampling_log_probs(prompt, completion)

    def _native_log_probs(self, prompt: str, completion: str) -> List[float]:
        """Get log probs using OpenAI's native logprobs support."""
        import httpx
        import math

        self._limiter.acquire()
        response = httpx.post(
            f"{self.api_base}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "max_tokens": len(completion) * 2,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Extract per-token log probs from the response.
        token_logprobs = []
        choice = data["choices"][0]
        if "logprobs" in choice and choice["logprobs"]:
            for token_info in choice["logprobs"].get("content", []):
                token_logprobs.append(token_info.get("logprob", math.log(1e-10)))

        # Expand to per-byte log probs by distributing token probs across bytes.
        completion_bytes = completion.encode("utf-8")
        if not token_logprobs:
            return [math.log(1.0 / 256)] * len(completion_bytes)

        # Simple uniform distribution of token log prob across its bytes.
        byte_log_probs = []
        avg_lp = sum(token_logprobs) / len(token_logprobs) if token_logprobs else math.log(1e-10)
        for _ in completion_bytes:
            byte_log_probs.append(avg_lp)
        return byte_log_probs

    def _sampling_log_probs(self, prompt: str, completion: str, n_samples: int = 5) -> List[float]:
        """Approximate log probs via sampling."""
        import math

        completion_bytes = completion.encode("utf-8")
        samples = self.batch_generate([prompt] * n_samples, max_tokens=len(completion) * 2)

        log_probs = []
        for i, byte in enumerate(completion_bytes):
            match_count = 0
            for sample in samples:
                sample_bytes = sample.encode("utf-8")
                if i < len(sample_bytes) and sample_bytes[i] == byte:
                    match_count += 1
            prob = (match_count + 1) / (n_samples + 256)
            log_probs.append(math.log(prob))

        return log_probs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]


class LocalTeacher(TeacherModel):
    """Teacher that wraps the local USSI model.

    Used for testing and as a fallback when no API keys are available.
    The local model acts as both teacher and student (self-distillation).
    """

    def __init__(self, config: TeacherConfig, kickstart=None):
        super().__init__(config)
        self.kickstart = kickstart
        self._model = None

    def set_kickstart(self, kickstart):
        """Set the kickstart instance to use for generation."""
        self.kickstart = kickstart

    def generate(self, prompt: str, **kwargs) -> str:
        self._limiter.acquire()
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        if self.kickstart is not None:
            return self.kickstart.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        # Fallback: return a simple templated response for testing.
        return f"Generated text about: {prompt[:100]}. " * 3

    def log_probs(self, prompt: str, completion: str) -> List[float]:
        """Get log probs from local model's forward pass."""
        import math

        if self.kickstart is None:
            # Uniform distribution fallback.
            n_bytes = len(completion.encode("utf-8"))
            return [math.log(1.0 / 260)] * n_bytes

        import torch
        from .data.tokenizer import SPECIAL_TOKENS

        model = self.kickstart.model
        tokenizer = self.kickstart.tokenizer

        # Encode full sequence.
        full_text = prompt + completion
        tokens = tokenizer.encode(full_text, add_special=False)
        prompt_tokens = tokenizer.encode(prompt, add_special=False)
        prompt_len = len(prompt_tokens)

        if len(tokens) < 2:
            return [math.log(1.0 / 260)]

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits, _ = model(input_ids)

        # Extract log probs for completion tokens.
        log_softmax = torch.nn.functional.log_softmax(logits[0], dim=-1)
        result = []
        for i in range(prompt_len, len(tokens) - 1):
            if i < log_softmax.shape[0]:
                token_id = tokens[i + 1]
                if token_id < log_softmax.shape[1]:
                    result.append(log_softmax[i, token_id].item())
                else:
                    result.append(math.log(1e-10))
            else:
                result.append(math.log(1e-10))

        return result if result else [math.log(1.0 / 260)]

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]


def create_teacher(config: TeacherConfig, kickstart=None) -> TeacherModel:
    """Factory function to create a teacher model from config.

    Args:
        config: Teacher configuration specifying provider and model.
        kickstart: Optional Kickstart instance for LocalTeacher.

    Returns:
        A TeacherModel implementation.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        return AnthropicTeacher(config)
    elif provider == "openai":
        return OpenAITeacher(config)
    elif provider == "local":
        return LocalTeacher(config, kickstart=kickstart)
    else:
        raise ValueError(f"Unknown teacher provider: {provider}. Use 'anthropic', 'openai', or 'local'.")


def parse_teacher_string(teacher_str: str) -> TeacherConfig:
    """Parse a teacher string like 'anthropic:claude-sonnet-4-20250514' into a TeacherConfig.

    Format: provider:model
    """
    if ":" in teacher_str:
        provider, model = teacher_str.split(":", 1)
    else:
        provider = teacher_str
        model = ""
    return TeacherConfig(provider=provider, model=model)
