"""Integration tests for distributed inference pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn


def _make_model(n_layers=6, hidden_dim=64):
    wrapper = nn.Module()
    wrapper.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers)]
    )
    return wrapper


def test_single_shard_inference():
    """Test inference with a single shard (no pipeline)."""
    from openclaw_engine.model.shard import split_model
    from openclaw_engine.inference.server import InferenceServer, InferenceRequest

    model = _make_model(n_layers=4, hidden_dim=64)
    shards = split_model(model, "test-model", 1)

    server = InferenceServer()
    server.register_shard("test-model", shards[0])

    request = InferenceRequest(model_id="test-model", prompt="Hello world")
    response = server.infer(request)

    assert response.model_id == "test-model"
    assert response.latency_ms > 0
    assert "output shape" in response.text


def test_pipeline_inference():
    """Test inference across a 4-stage pipeline."""
    from openclaw_engine.model.shard import split_model
    from openclaw_engine.model.pipeline import PipelineExecutor

    model = _make_model(n_layers=8, hidden_dim=64)
    shards = split_model(model, "pipeline-model", 4)
    pipeline = PipelineExecutor.local(shards)

    x = torch.randn(1, 16, 64)
    output = pipeline.forward(x)

    assert output.shape == (1, 16, 64), f"Unexpected shape: {output.shape}"


def test_pipeline_inference_executor():
    """Test the PipelineInferenceExecutor with local shards."""
    from openclaw_engine.model.shard import split_model
    from openclaw_engine.inference.pipeline_exec import PipelineInferenceExecutor

    model = _make_model(n_layers=4, hidden_dim=512)
    shards = split_model(model, "exec-test", 2)

    # Test first stage.
    executor = PipelineInferenceExecutor(
        local_shard=shards[0],
        pipeline_order=["peer0", "peer1"],
        local_peer_id="peer0",
    )
    result = executor.run("test prompt")
    assert "stage 0" in result or "output" in result.lower()

    # Test last stage.
    executor_last = PipelineInferenceExecutor(
        local_shard=shards[1],
        pipeline_order=["peer0", "peer1"],
        local_peer_id="peer1",
    )
    result_last = executor_last.run("test prompt")
    assert "output" in result_last.lower()


def test_inference_server_load_tracking():
    """Verify the inference server tracks load correctly."""
    from openclaw_engine.inference.server import InferenceServer, InferenceRequest
    from openclaw_engine.model.shard import split_model

    model = _make_model(n_layers=2, hidden_dim=64)
    shards = split_model(model, "load-test", 1)

    server = InferenceServer()
    server.register_shard("load-test", shards[0])

    assert server.current_load == 0.0

    # Run inference and check stats.
    request = InferenceRequest(model_id="load-test", prompt="test")
    server.infer(request)

    stats = server.stats()
    assert stats["total_requests"] == 1
    assert stats["active_requests"] == 0  # completed
    assert "load-test" in server.list_models()


def test_activation_serialization_roundtrip():
    """Verify tensor serialization/deserialization for pipeline communication."""
    from openclaw_engine.model.pipeline import serialize_tensor, deserialize_tensor

    original = torch.randn(2, 128, 512)
    data = serialize_tensor(original)
    restored = deserialize_tensor(data)

    assert torch.allclose(original, restored)
    assert original.shape == restored.shape


if __name__ == "__main__":
    test_single_shard_inference()
    print("  [PASS] test_single_shard_inference")
    test_pipeline_inference()
    print("  [PASS] test_pipeline_inference")
    test_pipeline_inference_executor()
    print("  [PASS] test_pipeline_inference_executor")
    test_inference_server_load_tracking()
    print("  [PASS] test_inference_server_load_tracking")
    test_activation_serialization_roundtrip()
    print("  [PASS] test_activation_serialization_roundtrip")
    print("\nAll inference pipeline tests passed!")
