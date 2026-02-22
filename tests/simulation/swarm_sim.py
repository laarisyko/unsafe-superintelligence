#!/usr/bin/env python3
"""Swarm simulation -- simulate N peers doing training and inference locally.

This simulates the full decentralized training lifecycle without needing
actual networking:
1. N peers each hold a shard of the model (data parallelism).
2. Each peer does a local training step.
3. Peers aggregate gradients via ring all-reduce OR hierarchical all-reduce.
4. All peers verify weight consistency via Merkle roots.
5. Each peer serves inference requests.

Supports hierarchical mode for simulating large-scale training (1000+ peers).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from ussi_engine.model.shard import split_model, ModelShard, ShardConfig
from ussi_engine.model.pipeline import PipelineExecutor
from ussi_engine.training.trainer import LocalTrainer, TrainingConfig
from ussi_engine.training.allreduce import RingAllReduce
from ussi_engine.training.compression import TopKCompressor
from ussi_engine.training.hierarchical import (
    HierarchicalAllReduce,
    ClusterConfig,
    assign_clusters_vrf,
    compute_scaling_stats,
)
from ussi_engine.training.cluster import ClusterManager, PeerCapacity
from ussi_engine.inference.server import InferenceServer, InferenceRequest


def make_model(n_layers: int = 8, hidden_dim: int = 64) -> nn.Module:
    wrapper = nn.Module()
    wrapper.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers)]
    )
    return wrapper


def run_simulation(
    n_peers: int = 4,
    n_layers: int = 8,
    hidden_dim: int = 64,
    training_steps: int = 5,
    training_rounds: int = 3,
    hierarchical: bool = False,
    cluster_size: int = 1000,
):
    print(f"=== USSI Swarm Simulation ===")
    print(f"Peers: {n_peers}, Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Training: {training_rounds} rounds x {training_steps} steps")
    mode = "hierarchical" if hierarchical else "flat ring"
    print(f"Aggregation: {mode}")

    # Print scaling stats.
    if hierarchical and n_peers > 64:
        cfg = ClusterConfig.auto(n_peers, cluster_size)
        stats = compute_scaling_stats(n_peers, cfg)
        print(f"  Hierarchy depth: {stats['depth']}")
        print(f"  Cluster size: {stats['cluster_size']}")
        print(f"  Flat rounds: {stats['flat_rounds']:,}")
        print(f"  Hierarchical rounds: {stats['hierarchical_rounds']:,}")
        print(f"  Speedup: {stats['speedup']:.1f}x")
    print()

    # --- Phase 1: Model Creation & Sharding ---
    print("[1/5] Creating model and distributing shards...")
    model = make_model(n_layers, hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Data parallelism: each peer gets a full copy (same shard).
    peer_shards = [split_model(model, "sim-model", 1)[0] for _ in range(n_peers)]
    print(f"  Each peer holds {peer_shards[0].num_parameters():,} parameters")
    print()

    # --- Phase 2: Training Rounds ---
    config = TrainingConfig(
        learning_rate=1e-3,
        num_steps=training_steps,
        optimizer="adamw",
    )
    trainers = [LocalTrainer(shard, config) for shard in peer_shards]

    # Set up aggregation.
    if hierarchical and n_peers > 64:
        cluster_cfg = ClusterConfig.auto(n_peers, cluster_size)
        cluster_cfg.hierarchical_threshold = 1
        peer_ids = [f"peer-{i}" for i in range(n_peers)]
        topology = assign_clusters_vrf(peer_ids, "sim-round", cluster_cfg)
    else:
        rings = RingAllReduce.local_ring(n_peers)

    compressor = TopKCompressor(ratio=0.1)

    for round_idx in range(training_rounds):
        round_start = time.monotonic()
        print(f"[2/5] Training round {round_idx + 1}/{training_rounds}...")

        # Each peer trains on different data.
        all_grads = []
        for peer_idx, trainer in enumerate(trainers):
            for step in range(training_steps):
                x = torch.randn(4, 8, hidden_dim) * (peer_idx + 1)
                metrics = trainer.train_step(x)

            grads = trainer.get_gradients()
            all_grads.append(grads)
            if peer_idx < 4 or peer_idx == n_peers - 1:
                print(f"    Peer {peer_idx}: grad_norm={metrics['grad_norm']:.4f}")
            elif peer_idx == 4:
                print(f"    ... ({n_peers - 5} more peers) ...")

        # Aggregate gradients.
        agg_start = time.monotonic()
        if hierarchical and n_peers > 64:
            print(f"  [3/5] Hierarchical all-reduce across {n_peers} peers...")
            aggregated = HierarchicalAllReduce.reduce_all(topology, all_grads)
        else:
            print(f"  [3/5] Ring all-reduce across {n_peers} peers...")
            aggregated = RingAllReduce.reduce_all(rings, all_grads)
        agg_ms = (time.monotonic() - agg_start) * 1000

        # Apply aggregated gradients.
        for i, trainer in enumerate(trainers):
            trainer.set_gradients(aggregated[i])
            trainer.apply_gradients()

        # Verify Merkle roots.
        roots = [shard.merkle_root().hex()[:16] for shard in peer_shards]
        consistent = len(set(roots)) == 1
        round_ms = (time.monotonic() - round_start) * 1000
        print(f"  [4/5] Merkle verification: {'CONSISTENT' if consistent else 'DIVERGENT'}")
        print(f"    Root: {roots[0]}...")
        print(f"    Aggregation: {agg_ms:.1f}ms, Round total: {round_ms:.1f}ms")
        print()

    # --- Phase 3: Inference ---
    print("[5/5] Running inference on sample peers...")
    sample_peers = [0, n_peers // 2, n_peers - 1] if n_peers > 3 else range(n_peers)
    for peer_idx in sample_peers:
        shard = peer_shards[peer_idx]
        server = InferenceServer()
        server.register_shard("sim-model", shard)
        request = InferenceRequest(model_id="sim-model", prompt="Hello from simulation")
        response = server.infer(request)
        print(f"  Peer {peer_idx}: {response.text} (latency: {response.latency_ms:.2f}ms)")

    print()
    print("=== Simulation Complete ===")


def run_pipeline_simulation(n_peers: int = 4, n_layers: int = 8, hidden_dim: int = 64):
    """Simulate pipeline-parallel inference across peers."""
    print(f"\n=== Pipeline Parallelism Simulation ===")
    print(f"Peers: {n_peers}, Layers: {n_layers}")

    model = make_model(n_layers, hidden_dim)
    shards = split_model(model, "pipeline-model", n_peers)

    for i, shard in enumerate(shards):
        print(
            f"  Peer {i}: layers [{shard.config.layer_start}, {shard.config.layer_end})"
            f" ({shard.num_parameters():,} params)"
        )

    pipeline = PipelineExecutor.local(shards)
    x = torch.randn(1, 16, hidden_dim)

    start = time.monotonic()
    output = pipeline.forward(x)
    elapsed_ms = (time.monotonic() - start) * 1000

    print(f"\n  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Pipeline latency: {elapsed_ms:.2f}ms")
    print(f"  Stages: {pipeline.num_stages}")
    print("=== Pipeline Simulation Complete ===\n")


def run_scaling_analysis():
    """Print scaling analysis for various agent counts."""
    print("\n=== Hierarchical All-Reduce Scaling Analysis ===\n")
    print(f"{'Agents':>12} | {'Flat Rounds':>14} | {'Hier. Rounds':>14} | {'Speedup':>10} | {'Depth':>5} | {'Cluster K':>10}")
    print("-" * 80)

    for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        cfg = ClusterConfig.auto(n)
        stats = compute_scaling_stats(n, cfg)
        print(
            f"{n:>12,} | {stats['flat_rounds']:>14,} | "
            f"{stats['hierarchical_rounds']:>14,} | "
            f"{stats['speedup']:>9.1f}x | {stats['depth']:>5} | {stats['cluster_size']:>10,}"
        )

    print("\n=== Analysis Complete ===\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="USSI swarm simulation")
    parser.add_argument("--peers", type=int, default=4, help="Number of simulated peers")
    parser.add_argument("--layers", type=int, default=8, help="Number of model layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--steps", type=int, default=5, help="Training steps per round")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Use hierarchical all-reduce (recommended for >64 peers)",
    )
    parser.add_argument(
        "--cluster-size", type=int, default=1000,
        help="Max peers per cluster in hierarchical mode",
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="Also run pipeline parallelism demo"
    )
    parser.add_argument(
        "--scaling", action="store_true", help="Print scaling analysis table"
    )
    args = parser.parse_args()

    if args.scaling:
        run_scaling_analysis()

    run_simulation(
        n_peers=args.peers,
        n_layers=args.layers,
        hidden_dim=args.hidden_dim,
        training_steps=args.steps,
        training_rounds=args.rounds,
        hierarchical=args.hierarchical,
        cluster_size=args.cluster_size,
    )

    if args.pipeline:
        run_pipeline_simulation(
            n_peers=args.peers,
            n_layers=args.layers,
            hidden_dim=args.hidden_dim,
        )
