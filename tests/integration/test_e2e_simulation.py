"""End-to-end multi-peer simulation tests.

Simulates a complete decentralized training network where multiple peers
independently:
    1. Initialize model shards
    2. Run local training steps
    3. Exchange gradients through the coordinator
    4. Aggregate with Byzantine protection
    5. Apply aggregated gradients
    6. Checkpoint weights with integrity verification
    7. Update peer reputation

This is the closest thing to a real network test without actual P2P networking.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from openclaw_engine.bridge import DirectBridgeHandler
from openclaw_engine.model.shard import ModelShard, ShardConfig, split_model
from openclaw_engine.model.checkpoint import CheckpointStore
from openclaw_engine.training.trainer import LocalTrainer, TrainingConfig
from openclaw_engine.training.byzantine import (
    AggregationMethod,
    ByzantineConfig,
    robust_aggregate,
    score_gradients,
)
from openclaw_engine.training.round_coordinator import (
    RoundCoordinator,
    RoundConfig,
    RoundPhase,
)
from openclaw_engine.training.cluster import PeerCapacity
from openclaw_engine.training.reputation import (
    ReputationTracker,
    BAN_THRESHOLD,
    SUSPECT_THRESHOLD,
)


def _make_model(n_layers=4, hidden_dim=16):
    model = nn.Module()
    model.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
         for _ in range(n_layers)]
    )
    return model


class SimulatedPeer:
    """A simulated peer in the decentralized training network."""

    def __init__(self, peer_id: str, model: nn.Module, training_config: TrainingConfig = None):
        self.peer_id = peer_id
        self.shard = split_model(model, "sim-model", 1)[0]
        self.bridge = DirectBridgeHandler(
            self.shard,
            training_config=training_config or TrainingConfig(learning_rate=1e-3),
        )
        self.is_byzantine = False
        self.byzantine_scale = 1.0  # How much to scale gradients (>1 = attack)
        self.dropped = False

    def train_step(self):
        """Run a local training step."""
        return self.bridge.train_step(input_shape=[2, 16])

    def get_gradients(self):
        """Get gradients, optionally poisoned for Byzantine peers."""
        grads = self.bridge.get_gradients()
        if self.is_byzantine:
            # Scale gradients to poison the model.
            grads = {k: v * self.byzantine_scale for k, v in grads.items()}
        return grads

    def apply_aggregated(self, gradients):
        """Apply aggregated gradients from the network."""
        self.bridge.set_gradients(gradients)

    def merkle_root(self):
        return self.bridge.merkle_root()


# === Tests ===


def test_e2e_honest_network():
    """10 honest peers train for 3 rounds. Weights converge."""
    torch.manual_seed(42)
    n_peers = 10
    n_rounds = 3

    # All peers start with the same model (in a real network,
    # they'd sync weights from a checkpoint first).
    base_model = _make_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    peers = []
    for i in range(n_peers):
        model = _make_model()
        model.load_state_dict(base_state)
        peers.append(SimulatedPeer(f"peer-{i}", model))

    reputation = ReputationTracker()

    for round_idx in range(n_rounds):
        config = RoundConfig(
            round_id=f"honest-round-{round_idx}",
            model_id="sim-model",
            min_peers=3,
            byzantine_config=ByzantineConfig(method=AggregationMethod.TRIMMED_MEAN),
        )
        coord = RoundCoordinator(config)

        # Register all peers.
        for p in peers:
            coord.register_peer(PeerCapacity(peer_id=p.peer_id))
        coord.close_registration()

        # Each peer trains locally.
        for p in peers:
            p.train_step()

        # Each peer submits gradients.
        for p in peers:
            grads = p.get_gradients()
            coord.submit_gradient(p.peer_id, grads)

        # Aggregate.
        result = coord.finalize()
        assert result is not None, f"Round {round_idx} failed to aggregate"

        # Apply aggregated gradients to all peers.
        for p in peers:
            p.apply_aggregated(result)

        # Update reputation.
        submitted = [p.peer_id for p in peers]
        scores = score_gradients(
            [p.get_gradients() for p in peers],
        )
        for pid, score in zip(submitted, scores):
            reputation.record_round_result(pid, completed=True, byzantine_score=score)

    # After 3 rounds, all peers should have identical weights.
    roots = [p.merkle_root() for p in peers]
    assert len(set(roots)) == 1, f"Weights diverged: {len(set(roots))} unique roots"

    # All peers should have good reputation.
    for p in peers:
        assert reputation.get_score(p.peer_id) > 0.5, \
            f"Peer {p.peer_id} has unexpectedly low score: {reputation.get_score(p.peer_id)}"


def test_e2e_byzantine_attack_mitigated():
    """15 honest + 5 Byzantine peers. Byzantine are detected and reputation drops."""
    torch.manual_seed(42)
    n_honest = 15
    n_byzantine = 5

    base_model = _make_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    peers = []
    for i in range(n_honest + n_byzantine):
        model = _make_model()
        model.load_state_dict(base_state)
        p = SimulatedPeer(f"peer-{i}", model)
        if i >= n_honest:
            p.is_byzantine = True
            p.byzantine_scale = 50.0
        peers.append(p)

    reputation = ReputationTracker()

    # Run 5 rounds.
    for round_idx in range(5):
        config = RoundConfig(
            round_id=f"byz-round-{round_idx}",
            model_id="sim-model",
            min_peers=10,
            byzantine_config=ByzantineConfig(
                method=AggregationMethod.TRIMMED_MEAN,
                max_byzantine=5,
                trim_ratio=0.25,
            ),
        )
        coord = RoundCoordinator(config)

        for p in peers:
            coord.register_peer(PeerCapacity(peer_id=p.peer_id))
        coord.close_registration()

        for p in peers:
            p.train_step()
        for p in peers:
            grads = p.get_gradients()
            coord.submit_gradient(p.peer_id, grads)

        result = coord.finalize()
        assert result is not None

        # Apply to honest peers only (in a real network, all peers apply).
        for p in peers[:n_honest]:
            p.apply_aggregated(result)

        # Score and detect outliers.
        all_grads = [p.get_gradients() for p in peers]
        scores = score_gradients(all_grads, max_byzantine=5)
        peer_ids = [p.peer_id for p in peers]
        outliers = reputation.detect_outliers(peer_ids, scores, threshold_ratio=3.0)

        reputation.record_round_batch(
            submitted_peers=peer_ids,
            dropped_peers=[],
            byzantine_scores={pid: s for pid, s in zip(peer_ids, scores)},
            outlier_peer_ids=outliers,
        )

    # Byzantine peers should have much lower reputation.
    honest_scores = [reputation.get_score(f"peer-{i}") for i in range(n_honest)]
    byzantine_scores_rep = [reputation.get_score(f"peer-{i}") for i in range(n_honest, n_honest + n_byzantine)]

    avg_honest = sum(honest_scores) / len(honest_scores)
    avg_byzantine = sum(byzantine_scores_rep) / len(byzantine_scores_rep)

    assert avg_honest > avg_byzantine, \
        f"Honest avg {avg_honest} should be > Byzantine avg {avg_byzantine}"

    # Check aggregated result wasn't poisoned.
    result_norm = result[list(result.keys())[0]].norm().item()
    assert result_norm < 10.0, f"Model poisoned: norm={result_norm}"

    print(f"  Honest avg reputation: {avg_honest:.3f}")
    print(f"  Byzantine avg reputation: {avg_byzantine:.3f}")
    print(f"  Outliers detected per round: {len(outliers)}")


def test_e2e_straggler_recovery():
    """Some peers drop out mid-round; remaining peers still complete."""
    torch.manual_seed(42)
    n_peers = 12

    base_model = _make_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    peers = []
    for i in range(n_peers):
        model = _make_model()
        model.load_state_dict(base_state)
        p = SimulatedPeer(f"peer-{i}", model)
        if i >= 9:
            p.dropped = True  # 3 stragglers
        peers.append(p)

    reputation = ReputationTracker()

    config = RoundConfig(
        round_id="straggler-round",
        model_id="sim-model",
        min_peers=3,
        min_quorum_ratio=0.6,
        byzantine_config=ByzantineConfig(method=AggregationMethod.TRIMMED_MEAN),
    )
    coord = RoundCoordinator(config)

    for p in peers:
        coord.register_peer(PeerCapacity(peer_id=p.peer_id))
    coord.close_registration()

    # Stragglers are marked as dropped.
    for p in peers:
        if p.dropped:
            coord.mark_dropout(p.peer_id)

    # Non-dropped peers train and submit.
    active_peers = [p for p in peers if not p.dropped]
    for p in active_peers:
        p.train_step()
        grads = p.get_gradients()
        coord.submit_gradient(p.peer_id, grads)

    result = coord.finalize()
    assert result is not None, "Round should succeed with 9/12 peers"

    # Apply and check.
    for p in active_peers:
        p.apply_aggregated(result)

    # All active peers should have same weights.
    roots = [p.merkle_root() for p in active_peers]
    assert len(set(roots)) == 1, "Active peers should converge"

    # Update reputation.
    reputation.record_round_batch(
        submitted_peers=[p.peer_id for p in active_peers],
        dropped_peers=[p.peer_id for p in peers if p.dropped],
        byzantine_scores={},
        outlier_peer_ids=[],
    )

    # Stragglers should have lower reputation.
    for p in peers:
        if p.dropped:
            assert reputation.get_score(p.peer_id) < 0.5, \
                f"Dropped peer {p.peer_id} should have reduced reputation"


def test_e2e_checkpoint_after_training():
    """Full training round followed by checkpoint save/load/verify."""
    torch.manual_seed(42)
    tmpdir = tempfile.mkdtemp()
    try:
        store = CheckpointStore(tmpdir)
        n_peers = 8

        base_model = _make_model()
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

        peers = []
        for i in range(n_peers):
            model = _make_model()
            model.load_state_dict(base_state)
            peers.append(SimulatedPeer(f"peer-{i}", model))

        config = RoundConfig(
            round_id="ckpt-round",
            model_id="sim-model",
            min_peers=3,
            byzantine_config=ByzantineConfig(method=AggregationMethod.MEAN),
        )
        coord = RoundCoordinator(config)

        for p in peers:
            coord.register_peer(PeerCapacity(peer_id=p.peer_id))
        coord.close_registration()

        for p in peers:
            p.train_step()
            coord.submit_gradient(p.peer_id, p.get_gradients())

        result = coord.finalize()
        assert result is not None

        # Apply to first peer and checkpoint.
        peers[0].apply_aggregated(result)
        merkle = store.save(
            peers[0].shard,
            "ckpt-round",
            peer_ids=[p.peer_id for p in peers],
            aggregation_method="mean",
        )

        # Load checkpoint into a fresh model.
        fresh_model = _make_model()
        fresh_shard = split_model(fresh_model, "sim-model", 1)[0]
        loaded = store.load(merkle, fresh_shard)
        assert loaded is not None
        assert loaded.merkle_root().hex() == merkle

        # Verify metadata.
        meta = store.get_metadata(merkle)
        assert meta is not None
        assert meta.round_id == "ckpt-round"
        assert meta.n_peers == n_peers

    finally:
        shutil.rmtree(tmpdir)


def test_e2e_multi_round_reputation_banning():
    """Byzantine peer gets banned after repeated bad behavior."""
    torch.manual_seed(42)

    base_model = _make_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    n_honest = 10
    n_byzantine = 1  # Single persistent attacker

    peers = []
    for i in range(n_honest + n_byzantine):
        model = _make_model()
        model.load_state_dict(base_state)
        p = SimulatedPeer(f"peer-{i}", model)
        if i >= n_honest:
            p.is_byzantine = True
            p.byzantine_scale = 100.0
        peers.append(p)

    reputation = ReputationTracker()
    attacker_id = peers[-1].peer_id

    # Run many rounds until the attacker gets banned.
    for round_idx in range(15):
        config = RoundConfig(
            round_id=f"ban-round-{round_idx}",
            model_id="sim-model",
            min_peers=3,
            byzantine_config=ByzantineConfig(
                method=AggregationMethod.TRIMMED_MEAN,
                trim_ratio=0.1,
            ),
        )
        coord = RoundCoordinator(config)

        # Only allow peers that aren't banned.
        active = [p for p in peers if reputation.is_allowed(p.peer_id)]
        for p in active:
            coord.register_peer(PeerCapacity(peer_id=p.peer_id))
        coord.close_registration()

        for p in active:
            p.train_step()
            coord.submit_gradient(p.peer_id, p.get_gradients())

        result = coord.finalize()
        if result is None:
            continue

        # Score and detect.
        all_grads = [p.get_gradients() for p in active]
        scores = score_gradients(all_grads)
        peer_ids = [p.peer_id for p in active]
        outliers = reputation.detect_outliers(peer_ids, scores)

        reputation.record_round_batch(
            submitted_peers=peer_ids,
            dropped_peers=[],
            byzantine_scores={pid: s for pid, s in zip(peer_ids, scores)},
            outlier_peer_ids=outliers,
        )

        if not reputation.is_allowed(attacker_id):
            print(f"  Attacker banned after round {round_idx + 1}")
            break

    # The attacker should eventually be banned.
    assert not reputation.is_allowed(attacker_id), \
        f"Attacker should be banned. Score: {reputation.get_score(attacker_id)}"

    # Honest peers should still be allowed.
    for i in range(n_honest):
        assert reputation.is_allowed(f"peer-{i}"), \
            f"Honest peer {i} should not be banned"


def test_e2e_bridge_handler_direct():
    """Test DirectBridgeHandler lifecycle without coordinator."""
    torch.manual_seed(42)
    model = _make_model()
    shard = split_model(model, "test-model", 1)[0]
    handler = DirectBridgeHandler(shard)

    # Train.
    metrics = handler.train_step([2, 16])
    assert "loss" in metrics
    assert "grad_norm" in metrics

    # Get gradients.
    grads = handler.get_gradients()
    assert len(grads) > 0
    for name, tensor in grads.items():
        assert isinstance(tensor, torch.Tensor)

    # Set gradients (simulate aggregation result).
    handler.set_gradients(grads)

    # Merkle root.
    root = handler.merkle_root()
    assert len(root) == 64  # SHA256 hex


def test_e2e_full_simulation_summary():
    """Full simulation with summary statistics across 5 rounds."""
    torch.manual_seed(42)
    n_peers = 20
    n_byzantine = 3
    n_stragglers = 2
    n_rounds = 5

    base_model = _make_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    peers = []
    for i in range(n_peers):
        model = _make_model()
        model.load_state_dict(base_state)
        p = SimulatedPeer(f"peer-{i}", model)
        if i >= n_peers - n_byzantine:
            p.is_byzantine = True
            p.byzantine_scale = 30.0
        elif i >= n_peers - n_byzantine - n_stragglers:
            p.dropped = True
        peers.append(p)

    reputation = ReputationTracker()
    round_summaries = []

    for round_idx in range(n_rounds):
        config = RoundConfig(
            round_id=f"sim-round-{round_idx}",
            model_id="sim-model",
            min_peers=5,
            min_quorum_ratio=0.5,
            byzantine_config=ByzantineConfig(
                method=AggregationMethod.TRIMMED_MEAN,
                max_byzantine=n_byzantine,
                trim_ratio=0.2,
            ),
        )
        coord = RoundCoordinator(config)

        active = [p for p in peers if not p.dropped and reputation.is_allowed(p.peer_id)]

        for p in active:
            coord.register_peer(PeerCapacity(peer_id=p.peer_id))
        coord.close_registration()

        # Mark stragglers.
        dropped_peers = [p for p in peers if p.dropped]
        for p in dropped_peers:
            if p.peer_id in [pp.peer_id for pp in active]:
                coord.mark_dropout(p.peer_id)

        # Train and submit.
        submitters = [p for p in active if not p.dropped]
        for p in submitters:
            p.train_step()
            coord.submit_gradient(p.peer_id, p.get_gradients())

        result = coord.finalize()
        assert result is not None, f"Round {round_idx} failed"

        # Apply to honest peers.
        for p in submitters:
            if not p.is_byzantine:
                p.apply_aggregated(result)

        # Score and update reputation.
        all_grads = [p.get_gradients() for p in submitters]
        scores = score_gradients(all_grads, max_byzantine=n_byzantine)
        peer_ids = [p.peer_id for p in submitters]
        outliers = reputation.detect_outliers(peer_ids, scores)

        reputation.record_round_batch(
            submitted_peers=peer_ids,
            dropped_peers=[p.peer_id for p in dropped_peers],
            byzantine_scores={pid: s for pid, s in zip(peer_ids, scores)},
            outlier_peer_ids=outliers,
        )

        summary = coord.summary()
        summary["outliers"] = len(outliers)
        round_summaries.append(summary)

    # Print simulation report.
    rep_summary = reputation.summary()
    print(f"\n  === Simulation Report ===")
    print(f"  Peers: {n_peers} (honest={n_peers-n_byzantine-n_stragglers}, "
          f"byzantine={n_byzantine}, stragglers={n_stragglers})")
    print(f"  Rounds: {n_rounds}")
    print(f"  Reputation: {rep_summary}")

    for i, s in enumerate(round_summaries):
        print(f"  Round {i}: submitted={s['submitted']}, "
              f"dropped={s['dropped']}, outliers={s['outliers']}")

    # Assertions.
    assert rep_summary["total_peers"] > 0
    assert rep_summary["avg_score"] > 0.3, "Overall avg score too low"

    # Honest peers should converge.
    honest_peers = [p for p in peers if not p.is_byzantine and not p.dropped]
    roots = [p.merkle_root() for p in honest_peers]
    assert len(set(roots)) == 1, \
        f"Honest peers should converge but got {len(set(roots))} unique roots"


if __name__ == "__main__":
    tests = [
        test_e2e_honest_network,
        test_e2e_byzantine_attack_mitigated,
        test_e2e_straggler_recovery,
        test_e2e_checkpoint_after_training,
        test_e2e_multi_round_reputation_banning,
        test_e2e_bridge_handler_direct,
        test_e2e_full_simulation_summary,
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
    print("\nAll end-to-end simulation tests passed!")
