"""Tests for OpenClaw bootstrap discovery and onboarding paths."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent-sdk"))

from ussi.agent import Agent
from ussi.node_manager import NodeManager
from ussi.openclaw import OpenClawBootstrapResolver, parse_bootstrap_peers


ADDR_A = "/ip4/203.0.113.10/tcp/9000/p2p/12D3KooWAAAA"
ADDR_B = "/ip4/203.0.113.11/tcp/9000/p2p/12D3KooWBBBB"


def test_parse_bootstrap_peers_filters_and_dedupes():
    peers = parse_bootstrap_peers([f"{ADDR_A},invalid", ADDR_B, ADDR_A])
    assert peers == [ADDR_A, ADDR_B]


def test_resolver_prefers_explicit_bootstrap():
    resolver = OpenClawBootstrapResolver(gateway_url="https://gateway.example")
    result = resolver.resolve(explicit=[ADDR_A, ADDR_B])

    assert result.source == "explicit"
    assert result.peers == [ADDR_A, ADDR_B]
    assert result.attempted_urls == []


def test_resolver_uses_env_bootstrap_when_present():
    old = os.environ.get("USSI_BOOTSTRAP")
    os.environ["USSI_BOOTSTRAP"] = ADDR_A
    try:
        resolver = OpenClawBootstrapResolver(gateway_url="https://gateway.example")
        result = resolver.resolve()
        assert result.source == "env"
        assert result.peers == [ADDR_A]
    finally:
        if old is None:
            os.environ.pop("USSI_BOOTSTRAP", None)
        else:
            os.environ["USSI_BOOTSTRAP"] = old


def test_agent_connect_dials_all_bootstrap_peers():
    agent = Agent(bootstrap=f"{ADDR_A},{ADDR_B}", node_api_url="http://127.0.0.1:50051")
    dialed = []
    agent.network.dial = lambda addr: dialed.append(addr) or {"dialed": True}
    agent.network.health = lambda: {"status": "ok"}

    agent.connect()

    assert dialed == [ADDR_A, ADDR_B]


def test_node_manager_passes_multiple_bootstrap_flags():
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""
        stdout = "container-id"

    def fake_run(cmd, capture_output=True, text=True, timeout=60):
        captured["cmd"] = cmd
        return _Result()

    with patch("ussi.node_manager.shutil.which", return_value="/usr/bin/docker"):
        with patch("ussi.node_manager.subprocess.run", side_effect=fake_run):
            with patch("ussi.node_manager.time.sleep", return_value=None):
                mgr = NodeManager(bootstrap=[ADDR_A, ADDR_B])
                result = mgr._start_docker()

    cmd = captured["cmd"]
    assert cmd.count("--bootstrap") == 2
    assert ADDR_A in cmd
    assert ADDR_B in cmd
    assert result["status"] == "started"
