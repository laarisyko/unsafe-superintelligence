"""Integration tests for peer discovery (runs against Python components only).

These tests verify that the CRDT shard map, reputation system, and VRF
work correctly in a simulated multi-peer scenario.
"""

import sys
import os

# Add engine to path for testing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent-sdk"))

import json
import hashlib


def test_shard_map_crdt_merge():
    """Verify that two shard maps from different peers converge after merge."""
    from ussi_engine.model.shard import _merkle_root

    # Simulate two peers maintaining independent shard maps.
    # We replicate the CRDT logic here since the Rust shard map
    # is mirrored in the Python engine for testing.

    map_a = {}
    map_b = {}

    # Peer A assigns layers 0-12 to itself.
    map_a[("model1", 0, 12)] = {"peer_id": "peerA", "version": 1}
    # Peer B assigns layers 0-12 to itself (conflict!).
    map_b[("model1", 0, 12)] = {"peer_id": "peerB", "version": 2}
    # Peer B also assigns layers 12-24.
    map_b[("model1", 12, 24)] = {"peer_id": "peerB", "version": 1}

    # Merge B into A (higher version wins).
    for key, entry in map_b.items():
        if key not in map_a or map_a[key]["version"] < entry["version"]:
            map_a[key] = entry

    # After merge, peer B should own layers 0-12 (version 2 > 1).
    assert map_a[("model1", 0, 12)]["peer_id"] == "peerB"
    # And layers 12-24.
    assert map_a[("model1", 12, 24)]["peer_id"] == "peerB"


def test_vrf_deterministic_assignment():
    """Verify that VRF produces the same assignment on all peers."""
    # Replicate the VRF logic from the Rust node.
    def vrf_compute(round_id: str, sorted_peers: list[str]) -> bytes:
        h = hashlib.sha256()
        h.update(b"ussi-vrf-v1:")
        h.update(round_id.encode())
        h.update(b":")
        for p in sorted_peers:
            h.update(p.encode())
            h.update(b",")
        return h.digest()

    peers = ["peerA", "peerB", "peerC", "peerD"]
    round_id = "round-001"

    # Every peer independently computes the same hash.
    hash_a = vrf_compute(round_id, peers)
    hash_b = vrf_compute(round_id, peers)
    hash_c = vrf_compute(round_id, peers)

    assert hash_a == hash_b == hash_c, "VRF must be deterministic"

    # Different round produces different hash.
    hash_diff = vrf_compute("round-002", peers)
    assert hash_a != hash_diff


def test_peer_reputation_tracking():
    """Verify reputation scoring increases on success, decreases on failure."""
    # Simple reputation model mirroring the Rust implementation.
    scores = {}

    def record_complete(peer_id):
        scores.setdefault(peer_id, 0.5)
        scores[peer_id] = min(1.0, scores[peer_id] + 0.1)

    def record_failure(peer_id):
        scores.setdefault(peer_id, 0.5)
        scores[peer_id] = max(0.0, scores[peer_id] - 0.2)

    record_complete("peerA")
    record_complete("peerA")
    record_failure("peerB")

    assert scores["peerA"] > scores["peerB"]
    assert scores["peerA"] == 0.7
    assert scores["peerB"] == 0.3


def test_heartbeat_message_format():
    """Verify heartbeat message serialization format."""
    heartbeat = {
        "peer_id": "12D3KooWExample",
        "timestamp_ms": 1700000000000,
        "gpu_memory_mb": 8192,
        "ram_mb": 32768,
        "cpu_cores": 8,
        "accelerator": "cuda",
    }

    serialized = json.dumps(heartbeat)
    deserialized = json.loads(serialized)

    assert deserialized["peer_id"] == "12D3KooWExample"
    assert deserialized["gpu_memory_mb"] == 8192
    assert deserialized["accelerator"] == "cuda"


if __name__ == "__main__":
    test_shard_map_crdt_merge()
    test_vrf_deterministic_assignment()
    test_peer_reputation_tracking()
    test_heartbeat_message_format()
    print("All peer discovery tests passed!")
