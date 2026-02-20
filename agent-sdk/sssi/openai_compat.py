"""OpenAI-compatible request/response types for the SSSI network.

This module provides builders that translate between the OpenAI API format
and the SSSI network's internal format, so SSSI can serve as a drop-in
replacement for OpenAI.

Supported endpoints:
  POST /v1/chat/completions   (ChatCompletion)
  POST /v1/completions        (Completion)
  GET  /v1/models             (Model list)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


def make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def make_model_id_list(model_ids: List[str]) -> Dict[str, Any]:
    """Build an OpenAI-compatible GET /v1/models response."""
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": "sssi-network",
                "permission": [],
                "root": mid,
                "parent": None,
            }
            for mid in model_ids
        ],
    }


def make_chat_completion(
    model: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Build an OpenAI-compatible POST /v1/chat/completions response."""
    return {
        "id": make_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_completion(
    model: str,
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Build an OpenAI-compatible POST /v1/completions response."""
    return {
        "id": make_completion_id(),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_chat_completion_chunk(
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single SSE chunk for streaming chat completions."""
    delta: Dict[str, Any] = {}
    if content:
        delta["content"] = content
    if finish_reason is None and not content:
        delta["role"] = "assistant"

    return {
        "id": chunk_id or make_completion_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def make_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    status: int = 400,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible error response."""
    err: Dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": None,
        "code": code,
    }
    return {"error": err}


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI chat messages to a single prompt string.

    This is a simple concatenation strategy. The actual model on the SSSI
    network may use its own chat template; this provides a reasonable default.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return max(1, len(text) // 4)
