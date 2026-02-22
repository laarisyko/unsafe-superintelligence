"""Network client -- communicates with the local USSI node via its HTTP API."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class NetworkClient:
    """HTTP client for the local USSI P2P node API."""

    def __init__(self, base_url: str = "http://127.0.0.1:50051"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def peers(self) -> List[Dict[str, Any]]:
        result = self._get("/peers")
        if isinstance(result, list):
            return result
        return []

    def shards(self) -> Dict[str, Any]:
        return self._get("/shards")

    def models(self) -> List[Dict[str, Any]]:
        return self._get("/models")

    def rounds(self) -> List[Dict[str, Any]]:
        return self._get("/rounds")

    def proposals(self) -> List[Dict[str, Any]]:
        return self._get("/proposals")

    def publish(self, topic: str, data: Any):
        payload = {"topic": topic, "data": json.dumps(data)}
        return self._post("/publish", payload)

    def dial(self, multiaddr: str):
        return self._post("/dial", {"address": multiaddr})

    def submit_inference(
        self,
        model_id: str,
        prompt: str,
        request_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        payload = {
            "request_id": request_id,
            "model_id": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post("/infer", payload)

    def submit_train_join(self, model_id: str, rounds: int, lr: float, batch_size: int) -> Dict[str, Any]:
        return self._post("/train/join", {
            "model_id": model_id,
            "rounds": rounds,
            "learning_rate": lr,
            "batch_size": batch_size,
        })

    def submit_evolve(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/evolve/propose", proposal)

    def submit_vote(self, proposal_id: str, decision: str, fitness: float = 0.0) -> Dict[str, Any]:
        return self._post("/evolve/vote", {
            "proposal_id": proposal_id,
            "decision": decision,
            "measured_fitness": fitness,
        })

    def submit_data(self, text: str, source: str = "") -> Dict[str, Any]:
        return self._post("/data/submit", {
            "text": text,
            "source": source,
        })

    def detect_compute(self) -> Dict[str, Any]:
        return self._get("/detect")

    def _get(self, path: str) -> Any:
        try:
            return self._http_request("GET", path)
        except Exception as e:
            logger.debug("GET %s failed: %s", path, e)
            return {"error": str(e)}

    def _post(self, path: str, data: Any) -> Any:
        try:
            return self._http_request("POST", path, json.dumps(data))
        except Exception as e:
            logger.debug("POST %s failed: %s", path, e)
            return {"error": str(e)}

    def _http_request(self, method: str, path: str, body: str = "") -> Any:
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80

        headers = f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n"
        if body:
            headers += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        headers += "\r\n"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
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
                return {"raw": body_str}

        return {"raw": response_str}
