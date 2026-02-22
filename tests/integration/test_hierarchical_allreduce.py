"""Integration tests for hierarchical all-reduce at scale.

Tests the tree-of-rings gradient aggregation that enables training with
up to 1M+ agents. Verifies correctness at multiple scales and hierarchy
depths.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from ussi_engine.training.hierarchical import (
    HierarchicalAllReduce,
    ClusterConfig,
    ClusterTopology,
    assign_clusters_vrf,
    compute_scaling_stats,
)
from ussi_engine.training.cluster import (
    ClusterManager,
    PeerCapacity,
    PeerRole,
)
from ussi_engine.training.allreduce import RingAllReduce


def _make_gradients(n_peers: int, hidden_dim: int = 16) -> list:
    """Create distinct gradient dicts for N peers."""
    grads = []
    for i in range(n_peers):
        grads.append({
            "layer.weight": torch.full((hidden_dim, hidden_dim), float(i + 1)),
            "layer.bias": torch.full((hidden_dim,), float(i + 1) * 10),
        })
    return grads


def _expected_average(n_peers: int) -> tuple:
    """Compute expected average for sequential fill values."""
    avg_weight = sum(range(1, n_peers + 1)) / n_peers
    avg_bias = sum(i * 10 for i in range(1, n_peers + 1)) / n_peers
    return avg_weight, avg_bias


# === Cluster Assignment Tests ===


def test_cluster_assignment_small():
    """Test cluster assignment with a small number of peers."""
    peer_ids = [f"peer-{i}" for i in range(10)]
    config = ClusterConfig(cluster_size=5, depth=1)
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    assert topology.n_peers == 10
    assert len(topology.clusters) == 1  # 1 level
    assert len(topology.clusters[0]) == 2  # 2 clusters of 5

    # All peers assigned exactly once.
    all_indices = []
    for members in topology.clusters[0].values():
        all_indices.extend(members)
    all_indices.sort()
    assert all_indices == list(range(10))


def test_cluster_assignment_deterministic():
    """Verify cluster assignment is deterministic (VRF property)."""
    peer_ids = [f"peer-{i}" for i in range(100)]
    config = ClusterConfig(cluster_size=10, depth=2)

    t1 = assign_clusters_vrf(peer_ids, "round-42", config)
    t2 = assign_clusters_vrf(peer_ids, "round-42", config)

    for level in range(config.depth):
        for cid in t1.clusters[level]:
            assert t1.clusters[level][cid] == t2.clusters[level][cid]


def test_cluster_assignment_two_level():
    """Test 2-level hierarchy with 100 peers, cluster_size=10."""
    peer_ids = [f"peer-{i}" for i in range(100)]
    config = ClusterConfig(cluster_size=10, depth=2)
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    # Level 0: 10 clusters of 10.
    assert len(topology.clusters[0]) == 10
    for members in topology.clusters[0].values():
        assert len(members) == 10

    # Level 1: leaders grouped.
    assert len(topology.clusters[1]) >= 1
    total_leaders = sum(len(m) for m in topology.clusters[1].values())
    assert total_leaders == 10  # 10 cluster leaders


def test_cluster_assignment_three_level():
    """Test 3-level hierarchy with 1000 peers, cluster_size=10."""
    peer_ids = [f"peer-{i}" for i in range(1000)]
    config = ClusterConfig(cluster_size=10, depth=3)
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    # Level 0: 100 clusters of 10.
    assert len(topology.clusters[0]) == 100

    # Level 1: 10 clusters of 10 leaders.
    assert len(topology.clusters[1]) == 10

    # Level 2: 1 cluster of 10 super-leaders.
    assert len(topology.clusters[2]) == 1


def test_different_rounds_different_assignments():
    """Ensure different rounds produce different cluster assignments."""
    peer_ids = [f"peer-{i}" for i in range(50)]
    config = ClusterConfig(cluster_size=10, depth=1)

    t1 = assign_clusters_vrf(peer_ids, "round-A", config)
    t2 = assign_clusters_vrf(peer_ids, "round-B", config)

    # At least one cluster should differ.
    differ = False
    for cid in t1.clusters[0]:
        if cid in t2.clusters[0] and t1.clusters[0][cid] != t2.clusters[0][cid]:
            differ = True
            break
    # Note: clusters may differ in structure too.
    assert differ or t1.clusters != t2.clusters


# === Hierarchical All-Reduce Correctness Tests ===


def test_hierarchical_small_flat_fallback():
    """For small peer counts, hierarchical falls back to flat ring."""
    n = 8
    grads = _make_gradients(n)
    config = ClusterConfig(cluster_size=10, depth=1, hierarchical_threshold=64)

    peer_ids = [f"peer-{i}" for i in range(n)]
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    results = HierarchicalAllReduce.reduce_all(topology, grads)

    expected_w, expected_b = _expected_average(n)
    for i in range(n):
        assert torch.allclose(
            results[i]["layer.weight"],
            torch.full((16, 16), expected_w),
            atol=1e-4,
        ), f"Peer {i} weight mismatch (flat fallback)"
        assert torch.allclose(
            results[i]["layer.bias"],
            torch.full((16,), expected_b),
            atol=1e-4,
        ), f"Peer {i} bias mismatch (flat fallback)"


def test_hierarchical_100_peers_2_levels():
    """Test hierarchical all-reduce with 100 peers and 2-level hierarchy."""
    n = 100
    grads = _make_gradients(n)
    config = ClusterConfig(cluster_size=10, depth=2, hierarchical_threshold=16)

    peer_ids = [f"peer-{i}" for i in range(n)]
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    results = HierarchicalAllReduce.reduce_all(topology, grads)

    expected_w, expected_b = _expected_average(n)
    for i in range(n):
        assert torch.allclose(
            results[i]["layer.weight"],
            torch.full((16, 16), expected_w),
            atol=1e-3,
        ), f"Peer {i} weight mismatch"
        assert torch.allclose(
            results[i]["layer.bias"],
            torch.full((16,), expected_b),
            atol=1e-3,
        ), f"Peer {i} bias mismatch"


def test_hierarchical_1000_peers_3_levels():
    """Test hierarchical all-reduce with 1000 peers and 3-level hierarchy."""
    n = 1000
    grads = _make_gradients(n, hidden_dim=4)  # Smaller for speed
    config = ClusterConfig(cluster_size=10, depth=3, hierarchical_threshold=16)

    peer_ids = [f"peer-{i}" for i in range(n)]
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    results = HierarchicalAllReduce.reduce_all(topology, grads)

    expected_w, expected_b = _expected_average(n)
    # Check a sample of peers (all should be identical).
    for i in [0, 100, 500, 999]:
        assert torch.allclose(
            results[i]["layer.weight"],
            torch.full((4, 4), expected_w),
            atol=1e-1,
        ), f"Peer {i} weight mismatch"


def test_hierarchical_matches_flat():
    """Verify hierarchical produces same result as flat ring all-reduce."""
    n = 20
    grads = _make_gradients(n)

    # Flat ring.
    rings = RingAllReduce.local_ring(n)
    flat_results = RingAllReduce.reduce_all(rings, grads)

    # Hierarchical with threshold=1 so it doesn't fall back.
    config = ClusterConfig(cluster_size=5, depth=2, hierarchical_threshold=1)
    peer_ids = [f"peer-{i}" for i in range(n)]
    topology = assign_clusters_vrf(peer_ids, "round-1", config)
    hier_results = HierarchicalAllReduce.reduce_all(topology, grads)

    # Both should produce the same average.
    for i in range(n):
        assert torch.allclose(
            flat_results[i]["layer.weight"],
            hier_results[i]["layer.weight"],
            atol=1e-3,
        ), f"Peer {i}: hierarchical != flat"


def test_hierarchical_single_peer():
    """Edge case: single peer should return its own gradients."""
    grads = [{"w": torch.tensor([1.0, 2.0, 3.0])}]
    config = ClusterConfig(cluster_size=10, depth=1)
    topology = assign_clusters_vrf(["peer-0"], "round-1", config)
    results = HierarchicalAllReduce.reduce_all(topology, grads)
    assert torch.allclose(results[0]["w"], grads[0]["w"])


# === Cluster Manager Tests ===


def test_cluster_manager_basic():
    """Test ClusterManager end-to-end."""
    manager = ClusterManager("round-1", ClusterConfig(cluster_size=5))

    for i in range(20):
        manager.register_peer(PeerCapacity(
            peer_id=f"peer-{i}",
            gpu_memory_mb=8192,
            accelerator="cuda",
        ))

    topology = manager.finalize()
    assert topology.n_peers == 20

    summary = manager.summary()
    assert summary["status"] == "finalized"
    assert summary["n_participants"] == 20
    assert summary["speedup"] > 0


def test_cluster_manager_roles():
    """Verify role assignment (members vs leaders)."""
    manager = ClusterManager("round-1", ClusterConfig(cluster_size=5, depth=2))

    for i in range(25):
        manager.register_peer(PeerCapacity(peer_id=f"peer-{i}"))

    manager.finalize()

    leaders_l0 = 0
    members = 0
    for i in range(25):
        m = manager.get_membership(f"peer-{i}")
        if m.role == PeerRole.MEMBER:
            members += 1
        elif m.role in (PeerRole.LEADER_L0, PeerRole.LEADER_L1, PeerRole.TOP_LEADER):
            leaders_l0 += 1

    # Should have some leaders and some members.
    assert leaders_l0 > 0, "Should have at least one leader"
    assert members > 0, "Should have regular members"
    assert leaders_l0 + members == 25


def test_cluster_manager_deterministic():
    """Same peers + same round_id = same topology on every node."""
    managers = []
    for _ in range(3):
        m = ClusterManager("round-42", ClusterConfig(cluster_size=10, depth=2))
        for i in range(50):
            m.register_peer(PeerCapacity(peer_id=f"peer-{i}"))
        m.finalize()
        managers.append(m)

    # All three should produce identical topologies.
    for level in range(2):
        for cid in managers[0].topology.clusters[level]:
            for m in managers[1:]:
                assert managers[0].topology.clusters[level][cid] == m.topology.clusters[level][cid]


# === Scaling & Performance Tests ===


def test_scaling_stats_1m():
    """Verify scaling statistics for 1M agents."""
    stats = compute_scaling_stats(1_000_000, ClusterConfig(cluster_size=1000, depth=2))
    assert stats["flat_rounds"] == 2 * 999_999
    assert stats["hierarchical_rounds"] < 5000
    assert stats["speedup"] > 400


def test_scaling_stats_10k():
    """Verify scaling for 10K agents."""
    stats = compute_scaling_stats(10_000, ClusterConfig(cluster_size=100, depth=2))
    assert stats["flat_rounds"] == 2 * 9999
    assert stats["hierarchical_rounds"] < 500
    assert stats["speedup"] > 30


def test_auto_config():
    """Test auto-configuration for different peer counts."""
    # Small.
    cfg = ClusterConfig.auto(50)
    assert cfg.depth == 1
    assert cfg.cluster_size >= 50

    # Medium.
    cfg = ClusterConfig.auto(10_000)
    assert cfg.depth >= 2

    # Large.
    cfg = ClusterConfig.auto(1_000_000)
    assert cfg.depth >= 2

    # Very large.
    cfg = ClusterConfig.auto(100_000_000)
    assert cfg.depth >= 3


def test_communication_rounds_property():
    """Test ClusterTopology.total_communication_rounds."""
    peer_ids = [f"peer-{i}" for i in range(100)]
    config = ClusterConfig(cluster_size=10, depth=2, hierarchical_threshold=16)
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    rounds = topology.total_communication_rounds
    # 2 levels * 2 * (10-1) = 36
    assert rounds <= 36
    assert rounds > 0


# === End-to-End Training with Hierarchical All-Reduce ===


def test_full_hierarchical_training_round():
    """End-to-end: shard model, train, hierarchical aggregate, verify Merkle."""
    from ussi_engine.model.shard import split_model
    from ussi_engine.training.trainer import LocalTrainer, TrainingConfig

    # Create model and replicate across 50 peers (data parallelism).
    n_peers = 50
    layers = nn.Module()
    layers.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(16, 16), nn.ReLU()) for _ in range(4)]
    )
    shards = [split_model(layers, "test", 1)[0] for _ in range(n_peers)]

    config = TrainingConfig(learning_rate=1e-3, num_steps=1)
    trainers = [LocalTrainer(s, config) for s in shards]

    # Each peer trains on different data.
    all_grads = []
    for i, trainer in enumerate(trainers):
        x = torch.randn(4, 8, 16) * (i + 1)
        trainer.train_step(x)
        grads = trainer.get_gradients()
        all_grads.append(grads)

    # Hierarchical all-reduce.
    cluster_config = ClusterConfig(cluster_size=10, depth=2, hierarchical_threshold=1)
    peer_ids = [f"peer-{i}" for i in range(n_peers)]
    topology = assign_clusters_vrf(peer_ids, "round-1", cluster_config)
    aggregated = HierarchicalAllReduce.reduce_all(topology, all_grads)

    # Apply aggregated gradients.
    for i, trainer in enumerate(trainers):
        trainer.set_gradients(aggregated[i])
        trainer.apply_gradients()

    # Verify all peers converged (Merkle roots should match).
    roots = [s.merkle_root() for s in shards]
    assert len(set(roots)) == 1, f"Peers diverged: {len(set(roots))} distinct roots"


def test_performance_hierarchical_vs_flat():
    """Benchmark hierarchical vs flat at 500 peers (simulation only)."""
    n = 500
    grads = _make_gradients(n, hidden_dim=4)

    # Hierarchical timing.
    config = ClusterConfig(cluster_size=50, depth=2, hierarchical_threshold=1)
    peer_ids = [f"peer-{i}" for i in range(n)]
    topology = assign_clusters_vrf(peer_ids, "round-1", config)

    start = time.monotonic()
    HierarchicalAllReduce.reduce_all(topology, grads)
    hier_ms = (time.monotonic() - start) * 1000

    # Flat timing.
    start = time.monotonic()
    rings = RingAllReduce.local_ring(n)
    RingAllReduce.reduce_all(rings, grads)
    flat_ms = (time.monotonic() - start) * 1000

    print(f"\n  500 peers: hierarchical={hier_ms:.1f}ms, flat={flat_ms:.1f}ms")

    # Hierarchical should be faster for large N due to O(depth*K) vs O(N).
    # In simulation both are in-process, but hierarchical does less work.
    stats = compute_scaling_stats(n, config)
    print(f"  Communication rounds: hierarchical={stats['hierarchical_rounds']}, "
          f"flat={stats['flat_rounds']}, speedup={stats['speedup']:.1f}x")


if __name__ == "__main__":
    tests = [
        test_cluster_assignment_small,
        test_cluster_assignment_deterministic,
        test_cluster_assignment_two_level,
        test_cluster_assignment_three_level,
        test_different_rounds_different_assignments,
        test_hierarchical_small_flat_fallback,
        test_hierarchical_100_peers_2_levels,
        test_hierarchical_1000_peers_3_levels,
        test_hierarchical_matches_flat,
        test_hierarchical_single_peer,
        test_cluster_manager_basic,
        test_cluster_manager_roles,
        test_cluster_manager_deterministic,
        test_scaling_stats_1m,
        test_scaling_stats_10k,
        test_auto_config,
        test_communication_rounds_property,
        test_full_hierarchical_training_round,
        test_performance_hierarchical_vs_flat,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
    print("\nAll hierarchical all-reduce tests passed!")
