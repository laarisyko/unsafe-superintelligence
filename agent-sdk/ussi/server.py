"""OpenAI-compatible HTTP server for the OpenClaw network.

Serves the same endpoints as the OpenAI API so that any OpenAI SDK,
LangChain, LlamaIndex, or other tool can use USSI as a drop-in replacement.

Usage:
    ussi serve                          # Start on port 8000
    ussi serve --port 11434             # Custom port

Then in any OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
    GET  /health
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .agent import Agent
from .openai_compat import (
    make_chat_completion,
    make_chat_completion_chunk,
    make_completion,
    make_error_response,
    make_model_id_list,
    messages_to_prompt,
    estimate_tokens,
)
from .rate_limit import RateLimitExceeded

logger = logging.getLogger(__name__)


class OpenAIHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing the OpenAI API contract."""

    # Attached by the server factory
    agent: Agent

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/v1/models/":
            self._handle_models()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, make_error_response("Not found", code="not_found"))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions":
            self._handle_completions()
        else:
            self._json_response(404, make_error_response("Not found", code="not_found"))

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---- Endpoint handlers ----

    def _handle_models(self):
        model_ids = self.agent.models()
        if not model_ids:
            model_ids = ["openclaw-default"]
        self._json_response(200, make_model_id_list(model_ids))

    def _handle_chat_completions(self):
        body = self._read_body()
        if body is None:
            return

        model = body.get("model", "openclaw-default")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 256
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        if not messages:
            self._json_response(400, make_error_response("messages is required"))
            return

        prompt = messages_to_prompt(messages)
        prompt_tokens = estimate_tokens(prompt)

        try:
            text = self.agent.infer(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RateLimitExceeded as e:
            self._json_response(429, make_error_response(
                str(e),
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            ))
            return

        completion_tokens = estimate_tokens(text)

        if stream:
            self._stream_chat_response(model, text, prompt_tokens, completion_tokens)
        else:
            resp = make_chat_completion(
                model=model,
                content=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self._json_response(200, resp)

    def _handle_completions(self):
        body = self._read_body()
        if body is None:
            return

        model = body.get("model", "openclaw-default")
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        if not prompt:
            self._json_response(400, make_error_response("prompt is required"))
            return

        # Handle prompt as list (OpenAI allows string or list)
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        prompt_tokens = estimate_tokens(prompt)

        try:
            text = self.agent.infer(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RateLimitExceeded as e:
            self._json_response(429, make_error_response(
                str(e),
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            ))
            return

        completion_tokens = estimate_tokens(text)

        if stream:
            self._stream_completion_response(model, text, prompt_tokens, completion_tokens)
        else:
            resp = make_completion(
                model=model,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self._json_response(200, resp)

    # ---- Streaming ----

    def _stream_chat_response(self, model: str, text: str, prompt_tokens: int, completion_tokens: int):
        """Send response as Server-Sent Events (SSE), matching OpenAI streaming format."""
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        chunk_id = make_chat_completion_chunk(model)["id"]

        # Initial chunk with role
        initial = make_chat_completion_chunk(model, chunk_id=chunk_id)
        self._write_sse(initial)

        # Content chunks (simulate word-by-word streaming)
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            chunk = make_chat_completion_chunk(model, content=token, chunk_id=chunk_id)
            self._write_sse(chunk)

        # Final chunk with finish_reason
        final = make_chat_completion_chunk(model, finish_reason="stop", chunk_id=chunk_id)
        self._write_sse(final)

        # Done signal
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_completion_response(self, model: str, text: str, prompt_tokens: int, completion_tokens: int):
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Send full text as one chunk then DONE (matching OpenAI behavior for completions)
        resp = make_completion(model=model, text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._write_sse(resp)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse(self, data: dict):
        line = f"data: {json.dumps(data)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()

    # ---- Helpers ----

    def _read_body(self) -> Optional[dict]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._json_response(400, make_error_response("Empty request body"))
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, make_error_response("Invalid JSON"))
            return None

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def log_message(self, format, *args):
        logger.info(format, *args)


def run_server(
    port: int = 8000,
    host: str = "0.0.0.0",
    node_url: str = "http://127.0.0.1:50051",
    contribute: bool = False,
    gpu_memory: str = "0",
    accelerator: str = "cpu",
):
    """Start the OpenAI-compatible HTTP server.

    Args:
        port: Port to listen on.
        host: Host to bind to.
        node_url: URL of the local OpenClaw P2P node.
        contribute: If True, contribute compute (contributor tier).
        gpu_memory: GPU memory to advertise.
        accelerator: Accelerator type.
    """
    agent = Agent(node_api_url=node_url)
    agent.connect()

    if contribute:
        agent.contribute(gpu_memory=gpu_memory, accelerator=accelerator)

    # Attach the agent to the handler class
    OpenAIHandler.agent = agent

    server = HTTPServer((host, port), OpenAIHandler)
    tier = agent.tier

    print(f"USSI OpenAI-compatible server running on http://{host}:{port}")
    print(f"  Tier: {tier.upper()}")
    print()
    print("Use with any OpenAI client:")
    print(f'  from openai import OpenAI')
    print(f'  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="ussi")')
    print(f'  response = client.chat.completions.create(')
    print(f'      model="llama-7b",')
    print(f'      messages=[{{"role": "user", "content": "Hello"}}],')
    print(f'  )')
    print()
    print("Endpoints:")
    print(f"  GET  http://localhost:{port}/v1/models")
    print(f"  POST http://localhost:{port}/v1/chat/completions")
    print(f"  POST http://localhost:{port}/v1/completions")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
        agent.leave()
