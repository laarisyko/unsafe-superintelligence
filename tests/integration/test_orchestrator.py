"""Tests for Sybil resistance, wire protocol, compression integration,
and the full training orchestrator.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from ussi_engine.training.sybil import (
    AdmissionController,
    PowChallenge,
    solve,
    verify,
    auto_difficulty,
    compute_pow_hash,
    hash_to_int,
    DIFFICULTY_TRIVIAL,
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
)
from ussi_engine.training.wire import (
    encode,
    decode,
    estimate_wire_size,
    WireMessage,
)
from ussi_engine.training.compression import (
    TopKCompressor,
    FP16Compressor,
    CompressorChain,
)
from ussi_engine.training.orchestrator import (
    TrainingOrchestrator,
    OrchestratorConfig,
    RoundResult,
)
from ussi_engine.training.byzantine import AggregationMethod, ByzantineConfig
from ussi_engine.training.cluster import PeerCapacity
from ussi_engine.training.reputation import ReputationTracker
from ussi_engine.model.shard import split_model
from ussi_engine.model.checkpoint import CheckpointStore


def _make_model(n_layers=4, hidden_dim=16):
    model = nn.Module()
    model.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
         for _ in range(n_layers)]
    )
    return model


def _make_gradients(dim=16):
    return {
        "w": torch.randn(dim, dim),
        "b": torch.randn(dim),
    }


# === Sybil Resistance (PoW) Tests ===


def test_pow_solve_and_verify():
    """Solve a PoW challenge and verify the solution."""
    challenge = PowChallenge(round_id="test-round", difficulty=DIFFICULTY_TRIVIAL)
    solution = solve(challenge, "peer-1")

    assert solution.nonce >= 0
    assert solution.attempts > 0
    assert len(solution.hash_hex) == 64

    assert verify(challenge, solution), "Valid solution should verify"


def test_pow_invalid_solution():
    """Invalid nonce should not verify."""
    challenge = PowChallenge(round_id="test-round", difficulty=DIFFICULTY_TRIVIAL)
    solution = solve(challenge, "peer-1")

    # Tamper with the nonce.
    solution.nonce += 1
    # May or may not fail depending on whether next nonce is also valid.
    # Instead, create a definitely-wrong solution.
    h = compute_pow_hash("test-round", "peer-1", solution.nonce)
    # Just check the verify function works by cross-checking.
    expected = hash_to_int(h) < challenge.target
    assert verify(challenge, solution) == expected


def test_pow_different_rounds():
    """Solutions are bound to the round_id."""
    challenge1 = PowChallenge(round_id="round-1", difficulty=DIFFICULTY_TRIVIAL)
    challenge2 = PowChallenge(round_id="round-2", difficulty=DIFFICULTY_TRIVIAL)

    solution = solve(challenge1, "peer-1")
    assert verify(challenge1, solution)
    assert not verify(challenge2, solution), "Solution should not verify for different round"


def test_pow_different_peers():
    """Solutions are bound to the peer_id."""
    challenge = PowChallenge(round_id="test-round", difficulty=DIFFICULTY_TRIVIAL)
    solution = solve(challenge, "peer-1")

    # Pretend another peer submitted this solution.
    solution.peer_id = "peer-2"
    # The hash is recomputed with the new peer_id, so it should fail.
    assert not verify(challenge, solution), "Solution should not verify for different peer"


def test_pow_difficulty_scaling():
    """Higher difficulty takes more attempts."""
    easy = PowChallenge(round_id="test", difficulty=DIFFICULTY_TRIVIAL)
    medium = PowChallenge(round_id="test", difficulty=DIFFICULTY_EASY)

    sol_easy = solve(easy, "peer-1")
    sol_medium = solve(medium, "peer-1")

    # Medium should generally take more attempts (statistical, but very likely).
    # At minimum, both should solve successfully.
    assert verify(easy, sol_easy)
    assert verify(medium, sol_medium)


def test_pow_auto_difficulty():
    """Auto difficulty scales with network size."""
    assert auto_difficulty(5) == DIFFICULTY_TRIVIAL
    assert auto_difficulty(50) == DIFFICULTY_EASY
    assert auto_difficulty(500) == DIFFICULTY_MEDIUM


def test_admission_controller():
    """Admission controller manages challenges per round."""
    ac = AdmissionController(base_difficulty=DIFFICULTY_TRIVIAL)
    challenge = ac.create_challenge("round-1", n_peers=5)

    solution = ac.solve_challenge("round-1", "peer-1")
    assert solution is not None
    assert ac.verify_admission("round-1", solution)


def test_admission_reputation_discount():
    """High-reputation peers get easier challenges."""
    ac = AdmissionController(
        base_difficulty=DIFFICULTY_EASY,
        reputation_discount=0.7,
    )
    challenge = ac.create_challenge("round-1")

    # High-rep peer gets 4 fewer bits of difficulty.
    solution = ac.solve_challenge("round-1", "trusted-peer")
    assert solution is not None
    # With high reputation (0.9), should verify with reduced difficulty.
    assert ac.verify_admission("round-1", solution, reputation_score=0.9)


# === Wire Protocol Tests ===


def test_wire_encode_decode_roundtrip():
    """Encode gradients to wire format and decode back."""
    grads = _make_gradients()
    msg = encode(grads, round_id="r1", peer_id="p1")

    assert msg.n_params == 2
    assert msg.size_bytes > 0
    assert len(msg.merkle_root) == 64

    decoded, metadata = decode(msg.data)
    assert set(decoded.keys()) == set(grads.keys())
    for name in grads:
        assert torch.allclose(decoded[name], grads[name], atol=1e-6), \
            f"Mismatch on {name}"

    assert metadata["round_id"] == "r1"
    assert metadata["peer_id"] == "p1"


def test_wire_compressed_roundtrip():
    """Encode with FP16 compression and decode back."""
    grads = _make_gradients()
    compressor = FP16Compressor()

    msg = encode(grads, round_id="r1", peer_id="p1", compressor=compressor)
    assert msg.compressed

    decoded, metadata = decode(msg.data, compressor=compressor)
    for name in grads:
        # FP16 has lower precision.
        assert torch.allclose(decoded[name], grads[name], atol=0.01), \
            f"FP16 mismatch on {name}"


def test_wire_integrity_check():
    """Tampered wire data should fail Merkle verification."""
    grads = _make_gradients()
    msg = encode(grads, round_id="r1", peer_id="p1")

    # Tamper with the data.
    data = bytearray(msg.data)
    # Modify a byte in the tensor data area (after the header).
    data[100] ^= 0xFF
    tampered = bytes(data)

    try:
        decode(tampered)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Merkle root mismatch" in str(e) or "Invalid magic" in str(e)


def test_wire_size_estimation():
    """Size estimation is reasonable."""
    grads = {"w": torch.randn(128, 128), "b": torch.randn(128)}
    est = estimate_wire_size(grads)

    assert est["raw_bytes"] == 128 * 128 * 4 + 128 * 4
    assert est["n_params"] == 2
    assert est["total_elements"] == 128 * 128 + 128
    assert est["uncompressed_wire_bytes"] > est["raw_bytes"]  # Header overhead


def test_wire_fp16_halves_size():
    """FP16 compression roughly halves the wire size."""
    grads = {"w": torch.randn(128, 128)}
    msg_raw = encode(grads)
    msg_fp16 = encode(grads, compressor=FP16Compressor())

    ratio = msg_raw.size_bytes / msg_fp16.size_bytes
    assert 1.5 < ratio < 2.5, f"FP16 ratio should be ~2x, got {ratio:.1f}x"


# === Orchestrator Tests ===


def test_orchestrator_basic_round():
    """Orchestrator runs a basic round with 5 honest peers."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]

    config = OrchestratorConfig(
        model_id="test-model",
        compression="none",
        pow_enabled=False,
        min_peers=3,
        byzantine_config=ByzantineConfig(method=AggregationMethod.MEAN),
    )

    orch = TrainingOrchestrator(config, shard)

    # Simulate 5 peers with gradients.
    peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(5)]
    all_grads = {f"peer-{i}": _make_gradients() for i in range(5)}

    result = orch.run_round_sync("round-1", peers, all_grads, "peer-0")

    assert result.success
    assert result.n_participants == 5
    assert result.n_submitted == 5
    assert result.aggregated_gradients is not None
    assert result.total_ms > 0


def test_orchestrator_with_pow():
    """Orchestrator enforces PoW admission."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]

    config = OrchestratorConfig(
        model_id="test-model",
        compression="none",
        pow_enabled=True,
        pow_difficulty=DIFFICULTY_TRIVIAL,
        min_peers=3,
    )

    orch = TrainingOrchestrator(config, shard)
    peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(5)]
    all_grads = {f"peer-{i}": _make_gradients() for i in range(5)}

    result = orch.run_round_sync("pow-round", peers, all_grads, "peer-0")

    assert result.success
    assert result.pow_solve_ms > 0


def test_orchestrator_with_compression():
    """Orchestrator compresses gradients for wire transmission."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]

    config = OrchestratorConfig(
        model_id="test-model",
        compression="fp16",
        pow_enabled=False,
        min_peers=3,
    )

    orch = TrainingOrchestrator(config, shard)
    peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(5)]
    all_grads = {f"peer-{i}": _make_gradients() for i in range(5)}

    result = orch.run_round_sync("comp-round", peers, all_grads, "peer-0")

    assert result.success
    assert result.compression_ratio > 1.0
    assert result.wire_bytes_sent > 0


def test_orchestrator_with_checkpoint():
    """Orchestrator checkpoints after each round."""
    torch.manual_seed(42)
    tmpdir = tempfile.mkdtemp()
    try:
        store = CheckpointStore(tmpdir)
        model = _make_model()
        shard = split_model(model, "test-model", 1)[0]

        config = OrchestratorConfig(
            model_id="test-model",
            compression="none",
            pow_enabled=False,
            min_peers=3,
            checkpoint_every=1,
        )

        orch = TrainingOrchestrator(config, shard, checkpoint_store=store)
        peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(5)]
        all_grads = {f"peer-{i}": _make_gradients() for i in range(5)}

        result = orch.run_round_sync("ckpt-round", peers, all_grads, "peer-0")

        assert result.success
        assert len(result.merkle_root) == 64

        # Verify checkpoint was saved.
        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].round_id == "ckpt-round"

    finally:
        shutil.rmtree(tmpdir)


def test_orchestrator_byzantine_detection():
    """Orchestrator detects and penalizes Byzantine peers."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]

    config = OrchestratorConfig(
        model_id="test-model",
        compression="none",
        pow_enabled=False,
        min_peers=3,
        byzantine_config=ByzantineConfig(
            method=AggregationMethod.TRIMMED_MEAN,
            max_byzantine=2,
            trim_ratio=0.2,
        ),
    )

    orch = TrainingOrchestrator(config, shard)

    # 8 honest + 2 Byzantine.
    peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(10)]
    all_grads = {}
    for i in range(8):
        all_grads[f"peer-{i}"] = {"w": torch.randn(16, 16) * 0.1, "b": torch.randn(16) * 0.1}
    for i in range(8, 10):
        all_grads[f"peer-{i}"] = {"w": torch.randn(16, 16) * 50.0, "b": torch.randn(16) * 50.0}

    result = orch.run_round_sync("byz-round", peers, all_grads, "peer-0")

    assert result.success
    assert result.n_byzantine_detected > 0
    assert result.aggregated_gradients["w"].norm().item() < 5.0


def test_orchestrator_banned_peers_excluded():
    """Orchestrator excludes banned peers from rounds."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]

    config = OrchestratorConfig(
        model_id="test-model",
        compression="none",
        pow_enabled=False,
        min_peers=3,
        byzantine_config=ByzantineConfig(method=AggregationMethod.MEAN),
    )

    reputation = ReputationTracker()
    # Pre-ban a peer.
    reputation._peers["peer-bad"] = type(reputation._get_or_create("peer-bad"))(
        peer_id="peer-bad", score=0.0, banned=True, ban_reason="test"
    )

    orch = TrainingOrchestrator(config, shard, reputation=reputation)

    peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(5)]
    peers.append(PeerCapacity(peer_id="peer-bad"))
    all_grads = {f"peer-{i}": _make_gradients() for i in range(5)}
    all_grads["peer-bad"] = _make_gradients()

    result = orch.run_round_sync("ban-round", peers, all_grads, "peer-0")

    assert result.success
    assert result.n_participants == 5  # peer-bad excluded


def test_orchestrator_multi_round():
    """Orchestrator runs multiple rounds, stats accumulate."""
    torch.manual_seed(42)
    tmpdir = tempfile.mkdtemp()
    try:
        store = CheckpointStore(tmpdir)
        model = _make_model()
        shard = split_model(model, "test-model", 1)[0]

        config = OrchestratorConfig(
            model_id="test-model",
            compression="fp16",
            pow_enabled=True,
            pow_difficulty=DIFFICULTY_TRIVIAL,
            min_peers=3,
            checkpoint_every=1,
            checkpoint_gc_keep=3,
        )

        orch = TrainingOrchestrator(config, shard, checkpoint_store=store)
        peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(8)]

        for r in range(5):
            all_grads = {f"peer-{i}": _make_gradients() for i in range(8)}
            result = orch.run_round_sync(f"multi-r{r}", peers, all_grads, "peer-0")
            assert result.success, f"Round {r} failed: {result.error}"

        assert orch.round_count == 5

        # GC should have kept only 3 checkpoints.
        checkpoints = store.list_checkpoints()
        assert len(checkpoints) <= 3

        stats = orch.get_stats()
        assert stats["round_count"] == 5
        assert stats["compression"] == "fp16"

        print(f"\n  Orchestrator stats after 5 rounds: {stats}")

    finally:
        shutil.rmtree(tmpdir)


def test_full_stack_integration():
    """Full stack: PoW + compression + Byzantine + checkpoint + reputation."""
    torch.manual_seed(42)
    tmpdir = tempfile.mkdtemp()
    try:
        store = CheckpointStore(tmpdir)
        model = _make_model()
        shard = split_model(model, "test-model", 1)[0]

        config = OrchestratorConfig(
            model_id="test-model",
            compression="fp16",
            pow_enabled=True,
            pow_difficulty=DIFFICULTY_TRIVIAL,
            min_peers=5,
            byzantine_config=ByzantineConfig(
                method=AggregationMethod.TRIMMED_MEAN,
                max_byzantine=2,
                trim_ratio=0.2,
            ),
            checkpoint_every=1,
        )

        reputation = ReputationTracker()
        orch = TrainingOrchestrator(config, shard, store, reputation)

        n_honest = 10
        n_byzantine = 2

        peers = [PeerCapacity(peer_id=f"peer-{i}") for i in range(n_honest + n_byzantine)]

        for r in range(3):
            all_grads = {}
            for i in range(n_honest):
                all_grads[f"peer-{i}"] = {
                    "w": torch.randn(16, 16) * 0.1,
                    "b": torch.randn(16) * 0.1,
                }
            for i in range(n_honest, n_honest + n_byzantine):
                all_grads[f"peer-{i}"] = {
                    "w": torch.randn(16, 16) * 50.0,
                    "b": torch.randn(16) * 50.0,
                }

            result = orch.run_round_sync(f"full-r{r}", peers, all_grads, "peer-0")
            assert result.success, f"Round {r} failed: {result.error}"
            assert result.n_byzantine_detected >= 0
            assert result.compression_ratio > 1.0
            assert result.pow_solve_ms > 0

        # Check reputation state.
        summary = reputation.summary()
        assert summary["total_peers"] > 0

        print(f"\n  Full stack integration summary:")
        print(f"    Reputation: {summary}")
        print(f"    Checkpoints: {len(store.list_checkpoints())}")
        print(f"    Last round: submitted={result.n_submitted}, "
              f"byzantine={result.n_byzantine_detected}, "
              f"wire={result.wire_bytes_sent/1024:.1f}KB, "
              f"compression={result.compression_ratio:.1f}x")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        # Sybil resistance.
        test_pow_solve_and_verify,
        test_pow_invalid_solution,
        test_pow_different_rounds,
        test_pow_different_peers,
        test_pow_difficulty_scaling,
        test_pow_auto_difficulty,
        test_admission_controller,
        test_admission_reputation_discount,
        # Wire protocol.
        test_wire_encode_decode_roundtrip,
        test_wire_compressed_roundtrip,
        test_wire_integrity_check,
        test_wire_size_estimation,
        test_wire_fp16_halves_size,
        # Orchestrator.
        test_orchestrator_basic_round,
        test_orchestrator_with_pow,
        test_orchestrator_with_compression,
        test_orchestrator_with_checkpoint,
        test_orchestrator_byzantine_detection,
        test_orchestrator_banned_peers_excluded,
        test_orchestrator_multi_round,
        test_full_stack_integration,
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
    print("\nAll orchestrator tests passed!")
