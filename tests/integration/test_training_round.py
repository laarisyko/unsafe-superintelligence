"""Integration tests for decentralized training rounds.

Tests the full training lifecycle: model sharding, local training,
gradient compression, ring all-reduce, and weight verification.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn


def _make_simple_model(n_layers=8, hidden_dim=64):
    """Create a simple sequential model for testing."""
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    # Wrap in a module with a .layers attribute so split_model can find them.
    wrapper = nn.Module()
    wrapper.layers = nn.ModuleList([nn.Sequential(layers[i], layers[i + 1]) for i in range(0, len(layers), 2)])
    return wrapper


def test_model_sharding():
    """Verify a model can be split into shards and reassembled."""
    from ussi_engine.model.shard import split_model

    model = _make_simple_model(n_layers=8, hidden_dim=64)
    shards = split_model(model, model_id="test-model", n_shards=4)

    assert len(shards) == 4
    total_params = sum(s.num_parameters() for s in shards)
    original_params = sum(p.numel() for p in model.parameters())
    assert total_params == original_params

    # Verify shard configs.
    assert shards[0].config.layer_start == 0
    assert shards[0].config.is_first
    assert shards[-1].config.is_last


def test_pipeline_forward():
    """Verify pipeline inference produces output through all stages."""
    from ussi_engine.model.shard import split_model
    from ussi_engine.model.pipeline import PipelineExecutor

    model = _make_simple_model(n_layers=8, hidden_dim=64)
    shards = split_model(model, model_id="test-model", n_shards=4)
    pipeline = PipelineExecutor.local(shards)

    x = torch.randn(2, 10, 64)  # batch=2, seq=10, dim=64
    output = pipeline.forward(x)

    assert output.shape[0] == 2  # batch preserved
    assert output.shape[-1] == 64  # hidden dim preserved


def test_local_training_step():
    """Verify a single training step produces gradients."""
    from ussi_engine.model.shard import split_model
    from ussi_engine.training.trainer import LocalTrainer, TrainingConfig

    model = _make_simple_model(n_layers=4, hidden_dim=32)
    shards = split_model(model, model_id="test", n_shards=2)

    config = TrainingConfig(learning_rate=1e-3, num_steps=1)
    trainer = LocalTrainer(shards[0], config)

    x = torch.randn(4, 8, 32)
    metrics = trainer.train_step(x)

    assert "loss" in metrics
    assert "grad_norm" in metrics

    grads = trainer.get_gradients()
    assert len(grads) > 0, "Should have gradients after training step"


def test_gradient_compression_topk():
    """Verify Top-K compression preserves the largest values."""
    from ussi_engine.training.compression import TopKCompressor

    compressor = TopKCompressor(ratio=0.1)
    original = torch.randn(100)

    compressed, metadata = compressor.compress(original)
    decompressed = compressor.decompress(compressed, metadata)

    # Decompressed should be sparse with only top 10% nonzero.
    nonzero = (decompressed != 0).sum().item()
    assert nonzero == 10  # 100 * 0.1

    # Top-K values should be preserved exactly.
    _, top_indices = torch.topk(original.abs(), 10)
    for idx in top_indices:
        assert torch.isclose(original[idx], decompressed[idx], atol=1e-5)


def test_gradient_compression_fp16():
    """Verify FP16 compression reduces size by ~50%."""
    from ussi_engine.training.compression import FP16Compressor

    compressor = FP16Compressor()
    original = torch.randn(1000)

    compressed, metadata = compressor.compress(original)
    decompressed = compressor.decompress(compressed, metadata)

    # FP16 should be approximately half the size.
    assert len(compressed) == original.numel() * 2  # 2 bytes per fp16

    # Values should be close (within FP16 precision).
    assert torch.allclose(original, decompressed, atol=1e-3)


def test_ring_allreduce_local():
    """Verify ring all-reduce produces correct average across 4 peers."""
    from ussi_engine.training.allreduce import RingAllReduce

    n_peers = 4
    rings = RingAllReduce.local_ring(n_peers)

    # Each peer has different gradients.
    peer_grads = []
    for i in range(n_peers):
        grads = {
            "layer.weight": torch.full((8, 8), float(i + 1)),
            "layer.bias": torch.full((8,), float(i + 1) * 10),
        }
        peer_grads.append(grads)

    # Run all-reduce across all peers in lockstep.
    results = RingAllReduce.reduce_all(rings, peer_grads)

    # All peers should get the same averaged result.
    expected_weight = (1 + 2 + 3 + 4) / 4.0  # = 2.5
    expected_bias = (10 + 20 + 30 + 40) / 4.0  # = 25.0

    for i in range(n_peers):
        assert torch.allclose(
            results[i]["layer.weight"],
            torch.full((8, 8), expected_weight),
            atol=1e-4,
        ), f"Peer {i} weight mismatch"
        assert torch.allclose(
            results[i]["layer.bias"],
            torch.full((8,), expected_bias),
            atol=1e-4,
        ), f"Peer {i} bias mismatch"


def test_merkle_root_consistency():
    """Verify two shards with identical weights produce the same Merkle root."""
    from ussi_engine.model.shard import ModelShard, ShardConfig

    config = ShardConfig(model_id="test", layer_start=0, layer_end=2, total_layers=4)

    layers_a = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 32)])
    layers_b = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 32)])

    # Copy weights from a to b.
    layers_b.load_state_dict(layers_a.state_dict())

    shard_a = ModelShard(config, layers_a)
    shard_b = ModelShard(config, layers_b)

    assert shard_a.merkle_root() == shard_b.merkle_root()

    # Modify one weight -- roots should diverge.
    with torch.no_grad():
        list(layers_b.parameters())[0][0, 0] += 1.0

    assert shard_a.merkle_root() != shard_b.merkle_root()


def test_full_training_round():
    """End-to-end test: shard model, train, aggregate, verify Merkle roots."""
    from ussi_engine.model.shard import split_model
    from ussi_engine.training.trainer import LocalTrainer, TrainingConfig
    from ussi_engine.training.allreduce import RingAllReduce

    # Create model and shard across 3 peers (data parallelism: same shard).
    model = _make_simple_model(n_layers=4, hidden_dim=32)
    shards = [split_model(model, "test", 1)[0] for _ in range(3)]

    config = TrainingConfig(learning_rate=1e-3, num_steps=1)
    trainers = [LocalTrainer(s, config) for s in shards]
    rings = RingAllReduce.local_ring(3)

    # Each peer does a local training step with different data.
    all_grads = []
    for i, trainer in enumerate(trainers):
        x = torch.randn(4, 8, 32) * (i + 1)
        trainer.train_step(x)
        grads = trainer.get_gradients()
        all_grads.append(grads)

    # All-reduce gradients across all peers in lockstep.
    aggregated = RingAllReduce.reduce_all(rings, all_grads)

    # Apply aggregated gradients.
    for i, trainer in enumerate(trainers):
        trainer.set_gradients(aggregated[i])
        trainer.apply_gradients()

    # Verify all peers converged (Merkle roots should match).
    roots = [s.merkle_root() for s in shards]
    assert roots[0] == roots[1] == roots[2], "All peers must converge after all-reduce"


if __name__ == "__main__":
    test_model_sharding()
    print("  [PASS] test_model_sharding")
    test_pipeline_forward()
    print("  [PASS] test_pipeline_forward")
    test_local_training_step()
    print("  [PASS] test_local_training_step")
    test_gradient_compression_topk()
    print("  [PASS] test_gradient_compression_topk")
    test_gradient_compression_fp16()
    print("  [PASS] test_gradient_compression_fp16")
    test_ring_allreduce_local()
    print("  [PASS] test_ring_allreduce_local")
    test_merkle_root_consistency()
    print("  [PASS] test_merkle_root_consistency")
    test_full_training_round()
    print("  [PASS] test_full_training_round")
    print("\nAll training round tests passed!")
