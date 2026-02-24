"""OpenClaw integration helpers for USSI bootstrap discovery."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_OPENCLAW_GATEWAY = "https://gateway.openclaw.ai"
_DISCOVERY_PATHS = (
    "/.well-known/ussi-bootstrap.json",
    "/v1/ussi/bootstrap",
    "/v1/networks/ussi/bootstrap",
    "/v1/skills/unsafesuperintelligence/bootstrap",
)


@dataclass
class BootstrapDiscoveryResult:
    peers: List[str] = field(default_factory=list)
    source: str = "none"
    gateway: str = DEFAULT_OPENCLAW_GATEWAY
    attempted_urls: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def normalize_multiaddr(addr: str) -> str:
    parts = [part.strip() for part in addr.strip().split("/") if part.strip()]
    if not parts:
        return ""
    return "/" + "/".join(parts)


def is_multiaddr(addr: str) -> bool:
    if not addr.startswith("/"):
        return False
    if "/p2p/" not in addr:
        return False
    parts = [part for part in addr.split("/") if part]
    return len(parts) >= 4


def parse_bootstrap_peers(raw: Optional[Sequence[str] | str]) -> List[str]:
    if raw is None:
        return []

    candidates: List[str] = []
    items = [raw] if isinstance(raw, str) else list(raw)
    for item in items:
        if not item:
            continue
        for value in str(item).split(","):
            value = value.strip()
            if value:
                candidates.append(value)

    return _dedupe_valid(candidates)


def resolve_env_bootstrap_peers() -> List[str]:
    peers: List[str] = []

    for key in ("USSI_BOOTSTRAP_PEERS", "USSI_BOOTSTRAP"):
        value = os.getenv(key, "")
        if value:
            peers.extend(parse_bootstrap_peers(value))

    bootstrap_file = os.getenv("USSI_BOOTSTRAP_FILE", "")
    if bootstrap_file and os.path.isfile(bootstrap_file):
        with open(bootstrap_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        peers.extend(parse_bootstrap_peers([line for line in lines if line and not line.startswith("#")]))

    return _dedupe_valid(peers)


class OpenClawBootstrapResolver:
    """Resolves bootstrap peers from explicit args, env, or OpenClaw gateway."""

    def __init__(self, gateway_url: Optional[str] = None, timeout_seconds: float = 2.5):
        self.gateway = (
            gateway_url
            or os.getenv("OPENCLAW_GATEWAY_URL")
            or os.getenv("OPENCLAW_GATEWAY")
            or DEFAULT_OPENCLAW_GATEWAY
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def resolve(self, explicit: Optional[Sequence[str] | str] = None) -> BootstrapDiscoveryResult:
        explicit_peers = parse_bootstrap_peers(explicit)
        if explicit_peers:
            return BootstrapDiscoveryResult(
                peers=explicit_peers,
                source="explicit",
                gateway=self.gateway,
            )

        env_peers = resolve_env_bootstrap_peers()
        if env_peers:
            return BootstrapDiscoveryResult(
                peers=env_peers,
                source="env",
                gateway=self.gateway,
            )

        result = BootstrapDiscoveryResult(source="gateway", gateway=self.gateway)
        for path in _DISCOVERY_PATHS:
            url = f"{self.gateway}{path}"
            result.attempted_urls.append(url)
            payload = self._fetch_json(url)
            if payload is None:
                continue
            peers = _dedupe_valid(_extract_candidates(payload))
            if peers:
                result.peers = peers
                result.source = f"gateway:{path}"
                return result

        result.source = "none"
        result.warnings.append(
            "No bootstrap peers discovered. Set USSI_BOOTSTRAP or pass --bootstrap."
        )
        return result

    def _fetch_json(self, url: str) -> Optional[Any]:
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ussi-sdk/0.1.0",
            },
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            logger.debug("OpenClaw bootstrap fetch failed for %s: %s", url, exc)
            return None


def _extract_candidates(payload: Any) -> List[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        out: List[str] = []
        for item in payload:
            out.extend(_extract_candidates(item))
        return out
    if isinstance(payload, dict):
        out = []
        for key in ("bootstrap_peers", "bootstrap", "peers", "addresses", "multiaddrs"):
            if key in payload:
                out.extend(_extract_candidates(payload[key]))
        if "data" in payload:
            out.extend(_extract_candidates(payload["data"]))
        if not out:
            for value in payload.values():
                out.extend(_extract_candidates(value))
        return out
    return []


def _dedupe_valid(values: Sequence[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        normalized = normalize_multiaddr(value)
        if not normalized or not is_multiaddr(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique
