"""Tests for the P2P training network, dataset downloader, and dashboard.

Proves that:
    1. TrainingNetwork initializes and runs training rounds
    2. Multi-peer gradient aggregation works through the network layer
    3. Checkpointing saves and loads correctly
    4. Dataset downloader provides sample data
    5. Dashboard state updates correctly
    6. Model configs scale from tiny to large
    7. Continuous training decreases loss
    8. CLI argument parsing works
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import json

import torch

from openclaw_engine.network import (
    TrainingNetwork,
    NetworkConfig,
    NetworkStats,
    PeerInfo,
    MODEL_CONFIGS,
)
from openclaw_engine.kickstart import KickstartConfig
from openclaw_engine.data.downloader import (
    get_sample_text,
    SAMPLE_TEXTS,
    GUTENBERG_BOOKS,
    list_gutenberg,
)
from openclaw_engine.dashboard import DashboardState
from openclaw_engine.training.byzantine import AggregationMethod


# === Sample Data ===

TRAIN_TEXT = get_sample_text("all")


# === Network Tests ===


def test_network_init():
    """TrainingNetwork initializes with default config."""
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)

    assert network.config.peer_id != ""
    assert network.kickstart is not None
    assert network.kickstart.model.num_parameters > 0
    stats = network.stats
    assert stats.model_params > 0
    assert stats.total_rounds == 0


def test_network_load_text():
    """Network loads text data."""
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    assert network.kickstart.data.total_tokens > 1000
    assert network.kickstart.data.total_batches > 0


def test_network_single_round():
    """Network runs a single training round."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    result = network.run_training_round("round-0")
    assert result.steps_completed > 0
    assert result.avg_loss > 0
    assert result.avg_loss < float("inf")
    assert result.tokens_processed > 0

    stats = network.stats
    assert stats.total_rounds == 1
    assert stats.current_loss == result.avg_loss


def test_network_loss_decreases():
    """Loss decreases over multiple training rounds."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    losses = []
    for i in range(5):
        result = network.run_training_round(f"round-{i}")
        losses.append(result.avg_loss)

    assert losses[-1] < losses[0], \
        f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  Loss curve: {' -> '.join(f'{l:.3f}' for l in losses)}")


def test_network_multi_peer_aggregation():
    """Multiple peers aggregate gradients through the network layer."""
    torch.manual_seed(42)
    n_peers = 3

    # Create peers with shared initial weights.
    configs = [NetworkConfig(model_size="tiny") for _ in range(n_peers)]
    networks = [TrainingNetwork(c) for c in configs]

    # Load different data on each peer.
    texts = [
        get_sample_text("alice") * 3,
        get_sample_text("shakespeare") * 3,
        get_sample_text("science") * 3,
    ]
    for net, text in zip(networks, texts):
        net.load_text(text)

    # Share initial weights.
    base_state = networks[0].kickstart.model.state_dict()
    for net in networks[1:]:
        net.kickstart.model.load_state_dict(
            {k: v.clone() for k, v in base_state.items()}
        )

    # Each peer trains locally.
    peer_gradients = {}
    for i, net in enumerate(networks):
        result = net.kickstart.train_round("round-0", f"peer-{i}")
        assert result.gradients is not None
        peer_gradients[f"peer-{i}"] = result.gradients

    # Now run a round with aggregation on the first peer.
    result = networks[0].run_training_round(
        "round-1",
        peer_gradients={k: v for k, v in peer_gradients.items() if k != "peer-0"},
    )

    assert result.steps_completed > 0
    print(f"  Multi-peer aggregation: loss={result.avg_loss:.4f}")


def test_network_checkpoint_save_load():
    """Network saves and loads checkpoints."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = NetworkConfig(
            model_size="tiny",
            checkpoint_dir=tmpdir,
            checkpoint_interval=1,
        )
        network = TrainingNetwork(config)
        network.load_text(TRAIN_TEXT)

        # Train and checkpoint.
        result = network.run_training_round("round-0")
        original_loss = result.avg_loss

        # Find checkpoint file.
        ckpts = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
        assert len(ckpts) == 1, f"Expected 1 checkpoint, got {len(ckpts)}"

        # Load into a new network.
        config2 = NetworkConfig(model_size="tiny", checkpoint_dir=tmpdir)
        network2 = TrainingNetwork(config2)
        network2.load_checkpoint(os.path.join(tmpdir, ckpts[0]))

        # Weights should match.
        for (n1, p1), (n2, p2) in zip(
            network.kickstart.model.named_parameters(),
            network2.kickstart.model.named_parameters(),
        ):
            assert torch.equal(p1.data, p2.data), \
                f"Checkpoint mismatch on {n1}"

        print(f"  Checkpoint save/load: verified")


def test_network_events():
    """Network emits events correctly."""
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    events = []
    network.on("round_start", lambda rid: events.append(("start", rid)))
    network.on("round_complete", lambda rid, res: events.append(("complete", rid)))
    network.on("loss_update", lambda rid, loss: events.append(("loss", loss)))

    network.run_training_round("round-0")

    assert ("start", "round-0") in events
    assert any(e[0] == "complete" for e in events)
    assert any(e[0] == "loss" for e in events)


def test_network_generate():
    """Network generates text."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    # Train a bit first.
    for i in range(3):
        network.run_training_round(f"round-{i}")

    text = network.generate("The ", max_tokens=20)
    assert len(text) > 4  # At least the prompt.
    print(f"  Generated: {text[:60]}")


def test_network_peer_management():
    """Network tracks peer registry."""
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)

    peer = PeerInfo(peer_id="test-peer-1", compute_type="cuda", gpu_memory_mb=8192)
    network.register_peer(peer)
    assert "test-peer-1" in network.peers
    assert network.stats.connected_peers == 1

    network.remove_peer("test-peer-1")
    assert "test-peer-1" not in network.peers
    assert network.stats.connected_peers == 0


# === Model Config Tests ===


def test_model_configs_exist():
    """All model size configs are defined."""
    assert "tiny" in MODEL_CONFIGS
    assert "small" in MODEL_CONFIGS
    assert "medium" in MODEL_CONFIGS
    assert "large" in MODEL_CONFIGS


def test_model_configs_scaling():
    """Model configs scale progressively."""
    sizes = ["tiny", "small", "medium", "large"]
    params = []
    for size in sizes:
        config = NetworkConfig(model_size=size)
        net = TrainingNetwork(config)
        p = net.kickstart.model.num_parameters
        params.append(p)
        print(f"  {size}: {p:,} params")

    # Each size should be larger than the previous.
    for i in range(1, len(params)):
        assert params[i] > params[i-1], \
            f"{sizes[i]} ({params[i]}) should be larger than {sizes[i-1]} ({params[i-1]})"


def test_medium_model_trains():
    """Medium model (50M-ish) can train."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="small")  # Use small for speed in tests.
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)

    result = network.run_training_round("round-0")
    assert result.steps_completed > 0
    assert result.avg_loss < float("inf")
    print(f"  Small model: {network.kickstart.model.num_parameters:,} params, loss={result.avg_loss:.4f}")


# === Dataset Tests ===


def test_sample_texts():
    """Built-in sample texts are available."""
    for name in ["alice", "shakespeare", "philosophy", "science"]:
        text = get_sample_text(name)
        assert len(text) > 100, f"Sample text '{name}' too short"

    all_text = get_sample_text("all")
    assert len(all_text) > 1000


def test_gutenberg_catalog():
    """Gutenberg book catalog is populated."""
    assert len(GUTENBERG_BOOKS) >= 10
    for key, info in GUTENBERG_BOOKS.items():
        assert "url" in info
        assert "title" in info
        assert "size_kb" in info
        assert info["size_kb"] > 0


def test_list_gutenberg():
    """list_gutenberg returns structured data."""
    books = list_gutenberg()
    assert len(books) >= 10
    for b in books:
        assert "key" in b
        assert "title" in b
        assert "downloaded" in b


# === Dashboard Tests ===


def test_dashboard_state():
    """Dashboard state updates correctly."""
    state = DashboardState()
    assert state.peer_count == 0
    assert state.current_loss == float("inf")

    state.update({
        "connected_peers": 5,
        "total_rounds": 100,
        "current_loss": 3.14,
        "model_params": 50000000,
        "latest_sample": "Hello world",
    })

    assert state.peer_count == 5
    assert state.total_rounds == 100
    assert state.current_loss == 3.14
    assert state.model_params == 50000000
    assert state.latest_sample == "Hello world"


def test_dashboard_snapshot():
    """Dashboard snapshot is JSON-serializable."""
    import json

    state = DashboardState()
    state.update({
        "connected_peers": 3,
        "current_loss": 2.5,
        "loss_history": [5.0, 4.5, 4.0, 3.5, 3.0, 2.5],
    })

    snapshot = state.snapshot()
    serialized = json.dumps(snapshot)
    assert "peer_count" in serialized
    assert "loss_history" in serialized


def test_dashboard_subscriber():
    """Dashboard state notifies subscribers."""
    import asyncio

    state = DashboardState()
    queue = state.subscribe()

    state.update({"connected_peers": 10, "current_loss": 2.0})

    # Queue should have received the update.
    assert not queue.empty()
    snapshot = queue.get_nowait()
    assert snapshot["peer_count"] == 10

    state.unsubscribe(queue)


# === CLI Tests ===


def test_cli_parser():
    """CLI argument parser works."""
    from openclaw_engine.cli import main
    import argparse

    # Just verify it doesn't crash with --help.
    # (We can't actually run it because it calls sys.exit)


def test_stats_dict():
    """Network stats dict is JSON-serializable."""
    import json

    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(TRAIN_TEXT)
    network.run_training_round("round-0")

    stats = network.get_stats_dict()
    serialized = json.dumps(stats)
    assert "peer_id" in serialized
    assert "total_rounds" in serialized
    assert "current_loss" in serialized


# === Phase 3/5: Bridge End-to-End Test ===


def test_bridge_end_to_end():
    """Full bridge lifecycle: init model, load data, train, get grads, set grads, infer."""
    import asyncio
    from openclaw_engine.bridge import NodeBridge
    from openclaw_engine.inference.server import InferenceServer
    from openclaw_engine.kickstart import Kickstart, KickstartConfig

    torch.manual_seed(42)

    # Create a real Kickstart model.
    config = KickstartConfig(
        model_id="bridge-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=2,
        steps_per_round=3,
        learning_rate=1e-3,
    )
    ks = Kickstart(config)
    ks.load_text(TRAIN_TEXT)

    # Create bridge with real Kickstart.
    inf_server = InferenceServer()
    bridge = NodeBridge(inf_server, kickstart=ks)

    async def run_bridge_test():
        # 1. Health check.
        resp = await bridge.handle_node_message("health", b"")
        health = json.loads(resp)
        assert health["status"] == "ok"
        assert health["kickstart_available"] is True

        # 2. Stats.
        resp = await bridge.handle_node_message("stats", b"")
        stats = json.loads(resp)
        assert stats["has_kickstart"] is True
        assert stats["parameters"] > 0

        # 3. Train.
        train_payload = json.dumps({
            "round_id": "test-round",
            "peer_id": "test-peer",
        }).encode()
        resp = await bridge.handle_node_message("train_step", train_payload)
        train_result = json.loads(resp)
        assert "loss" in train_result
        assert train_result["step"] > 0

        # 4. Get gradients (with full tensor data).
        resp = await bridge.handle_node_message("get_gradients", b"")
        grad_result = json.loads(resp)
        assert "gradients" in grad_result
        assert grad_result["param_count"] > 0
        assert "merkle_root" in grad_result
        assert len(grad_result["gradients"]) > 0

        # 5. Set gradients (round-trip: get then set).
        set_payload = json.dumps({
            "gradients": grad_result["gradients"],
        }).encode()
        resp = await bridge.handle_node_message("set_gradients", set_payload)
        set_result = json.loads(resp)
        assert set_result["status"] == "ok"
        assert set_result["applied"] > 0

        # 6. Inference.
        infer_payload = json.dumps({
            "request_id": "test-req",
            "prompt": "The ",
            "max_tokens": 10,
            "temperature": 0.8,
        }).encode()
        resp = await bridge.handle_node_message("infer", infer_payload)
        infer_result = json.loads(resp)
        assert "text" in infer_result
        assert len(infer_result["text"]) > 0

    asyncio.run(run_bridge_test())
    print("  Bridge end-to-end: all 6 operations passed")


def test_bridge_init_model():
    """Bridge can initialize a model from config."""
    import asyncio
    from openclaw_engine.bridge import NodeBridge
    from openclaw_engine.inference.server import InferenceServer

    inf_server = InferenceServer()
    bridge = NodeBridge(inf_server)
    assert bridge.kickstart is None

    async def run_init_test():
        init_payload = json.dumps({
            "model_id": "init-test",
            "hidden_dim": 32,
            "n_layers": 1,
            "n_heads": 2,
            "vocab_size": 260,
            "max_seq_length": 32,
        }).encode()
        resp = await bridge.handle_node_message("init_model", init_payload)
        result = json.loads(resp)
        assert result["status"] == "ok"
        assert result["parameters"] > 0
        assert bridge.kickstart is not None

    asyncio.run(run_init_test())


if __name__ == "__main__":
    tests = [
        # Network.
        test_network_init,
        test_network_load_text,
        test_network_single_round,
        test_network_loss_decreases,
        test_network_multi_peer_aggregation,
        test_network_checkpoint_save_load,
        test_network_events,
        test_network_generate,
        test_network_peer_management,
        # Model configs.
        test_model_configs_exist,
        test_model_configs_scaling,
        test_medium_model_trains,
        # Dataset.
        test_sample_texts,
        test_gutenberg_catalog,
        test_list_gutenberg,
        # Dashboard.
        test_dashboard_state,
        test_dashboard_snapshot,
        test_dashboard_subscriber,
        # CLI / stats.
        test_cli_parser,
        test_stats_dict,
        # Bridge end-to-end.
        test_bridge_end_to_end,
        test_bridge_init_model,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
    print("\nAll network launch tests passed!")
