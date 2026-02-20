"""Drop-in OpenAI client replacement backed by the SSSI network.

Usage (identical to the openai package)::

    from sssi import OpenAI

    client = OpenAI()  # defaults to http://localhost:8000/v1

    # Chat completions
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=256,
    )
    print(response.choices[0].message.content)

    # Legacy completions
    response = client.completions.create(
        model="llama-7b",
        prompt="Once upon a time",
        max_tokens=256,
    )
    print(response.choices[0].text)

    # List models
    models = client.models.list()
    for m in models.data:
        print(m.id)

This client works WITHOUT the `openai` package installed. It speaks the
same protocol over HTTP. For full OpenAI SDK compatibility (types, async,
etc.), use the official `openai` package pointed at the SSSI server:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sssi")
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# ---- Response dataclasses (mirror openai types) ----

@dataclass
class Message:
    role: str = ""
    content: str = ""


@dataclass
class ChatChoice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: str = "stop"


@dataclass
class CompletionChoice:
    index: int = 0
    text: str = ""
    finish_reason: str = "stop"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class Completion:
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class Model:
    id: str = ""
    object: str = "model"
    created: int = 0
    owned_by: str = "sssi-network"


@dataclass
class ModelList:
    object: str = "list"
    data: List[Model] = field(default_factory=list)


# ---- HTTP helpers ----

def _http_request(base_url: str, method: str, path: str, body: str = "", api_key: str = "") -> dict:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    full_path = (parsed.path.rstrip("/") + "/" + path.lstrip("/")).replace("//", "/")

    headers = f"{method} {full_path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n"
    if api_key:
        headers += f"Authorization: Bearer {api_key}\r\n"
    if body:
        headers += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    headers += "\r\n"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)
        sock.connect((host, port))
        sock.sendall((headers + body).encode())

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

    response_str = response.decode(errors="replace")
    body_start = response_str.find("\r\n\r\n")
    if body_start >= 0:
        body_str = response_str[body_start + 4:]
        try:
            return json.loads(body_str)
        except json.JSONDecodeError:
            return {"error": {"message": body_str, "type": "parse_error"}}

    return {"error": {"message": "Empty response", "type": "connection_error"}}


# ---- API namespaces ----

class _ChatCompletions:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url
        self._api_key = api_key

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # Streaming not supported in this lightweight client
            **kwargs,
        }
        data = _http_request(self._base_url, "POST", "/chat/completions", json.dumps(payload), self._api_key)

        if "error" in data:
            raise RuntimeError(f"SSSI API error: {data['error'].get('message', data['error'])}")

        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(ChatChoice(
                index=c.get("index", 0),
                message=Message(role=msg.get("role", "assistant"), content=msg.get("content", "")),
                finish_reason=c.get("finish_reason", "stop"),
            ))

        usage_data = data.get("usage", {})
        return ChatCompletion(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", model),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )


class _Chat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _ChatCompletions(base_url, api_key)


class _Completions:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url
        self._api_key = api_key

    def create(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Completion:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        data = _http_request(self._base_url, "POST", "/completions", json.dumps(payload), self._api_key)

        if "error" in data:
            raise RuntimeError(f"SSSI API error: {data['error'].get('message', data['error'])}")

        choices = []
        for c in data.get("choices", []):
            choices.append(CompletionChoice(
                index=c.get("index", 0),
                text=c.get("text", ""),
                finish_reason=c.get("finish_reason", "stop"),
            ))

        usage_data = data.get("usage", {})
        return Completion(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", model),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )


class _Models:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url
        self._api_key = api_key

    def list(self) -> ModelList:
        data = _http_request(self._base_url, "GET", "/models", api_key=self._api_key)

        if "error" in data:
            raise RuntimeError(f"SSSI API error: {data['error'].get('message', data['error'])}")

        models = []
        for m in data.get("data", []):
            models.append(Model(
                id=m.get("id", ""),
                object=m.get("object", "model"),
                created=m.get("created", 0),
                owned_by=m.get("owned_by", "sssi-network"),
            ))
        return ModelList(data=models)


# ---- Main client ----

class OpenAI:
    """Drop-in replacement for openai.OpenAI backed by the SSSI network.

    Usage::

        from sssi import OpenAI
        client = OpenAI()  # or OpenAI(base_url="http://host:port/v1")

        response = client.chat.completions.create(
            model="llama-7b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "sssi",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat = _Chat(self.base_url, self.api_key)
        self.completions = _Completions(self.base_url, self.api_key)
        self.models = _Models(self.base_url, self.api_key)
