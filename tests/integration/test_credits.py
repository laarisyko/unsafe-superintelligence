"""Tests for the credit system: non-transferable incentive credits.

Proves that:
    1. Credit accounts are created with welcome bonus
    2. Credits earned for training rounds, data, and votes
    3. Reputation multiplier affects earning rate
    4. Credits can be spent on inference
    5. Free tier rate limiting works
    6. Credit decay reduces balances over time
    7. Inference gate authorizes/rejects correctly
    8. Non-transferable: no transfer method exists
    9. Network integration awards credits during training
    10. Uptime streak multiplier works
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch

from ussi_engine.credits import (
    CreditLedger,
    CreditConfig,
    InferenceGate,
    CreditAction,
    InferenceTier,
    SpendResult,
    PeerCreditAccount,
)
from ussi_engine.network import TrainingNetwork, NetworkConfig
from ussi_engine.data.downloader import get_sample_text


# === Credit Ledger Core Tests ===


def test_new_account_welcome_bonus():
    """New accounts receive a welcome bonus."""
    ledger = CreditLedger(CreditConfig(welcome_bonus=50.0))
    balance = ledger.get_balance("peer-1")
    assert balance == 50.0
    account = ledger.get_account("peer-1")
    assert account.total_earned == 50.0
    assert account.is_contributor
    print(f"  Welcome bonus: {balance}")


def test_earn_training_round():
    """Credits earned for training round completion."""
    config = CreditConfig(
        welcome_bonus=0,
        earn_training_round=10.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    earned = ledger.earn_training_round("peer-1", "round-0", reputation_score=0.5)
    # Base 10 * rep multiplier (0.5 + (2.0 - 0.5) * 0.5 = 1.25)
    expected = 10.0 * 1.25
    assert abs(earned - expected) < 0.01, f"Expected {expected}, got {earned}"

    balance = ledger.get_balance("peer-1")
    assert abs(balance - expected) < 0.01
    account = ledger.get_account("peer-1")
    assert account.rounds_contributed == 1
    print(f"  Training round earned: {earned:.1f}")


def test_reputation_multiplier():
    """Higher reputation = more credits earned."""
    config = CreditConfig(
        welcome_bonus=0,
        earn_training_round=10.0,
        reputation_min_multiplier=0.5,
        reputation_max_multiplier=2.0,
        decay_enabled=False,
    )

    # Low reputation.
    ledger_low = CreditLedger(config)
    earned_low = ledger_low.earn_training_round("peer-low", "r0", reputation_score=0.0)

    # High reputation.
    ledger_high = CreditLedger(config)
    earned_high = ledger_high.earn_training_round("peer-high", "r0", reputation_score=1.0)

    # High rep should earn 4x more than low rep (2.0 vs 0.5 multiplier).
    assert earned_high > earned_low
    assert abs(earned_high / earned_low - 4.0) < 0.01
    print(f"  Low rep earned: {earned_low:.1f}, high rep: {earned_high:.1f} "
          f"(ratio: {earned_high / earned_low:.1f}x)")


def test_earn_data_contribution():
    """Credits earned for data contribution."""
    config = CreditConfig(welcome_bonus=0, earn_data_per_mb=5.0, decay_enabled=False)
    ledger = CreditLedger(config)

    # 2 MB of data.
    earned = ledger.earn_data_contribution("peer-1", 2 * 1024 * 1024)
    assert abs(earned - 10.0) < 0.01
    account = ledger.get_account("peer-1")
    assert abs(account.data_contributed_mb - 2.0) < 0.01
    print(f"  Data contribution earned: {earned:.1f}")


def test_earn_vote():
    """Credits earned for architecture voting."""
    config = CreditConfig(welcome_bonus=0, earn_vote=2.0, decay_enabled=False)
    ledger = CreditLedger(config)

    earned = ledger.earn_vote("peer-1", "proposal-123")
    assert abs(earned - 2.0) < 0.01
    account = ledger.get_account("peer-1")
    assert account.votes_cast == 1
    print(f"  Vote earned: {earned:.1f}")


def test_multiple_earnings():
    """Credits accumulate across multiple earning actions."""
    config = CreditConfig(
        welcome_bonus=0,
        earn_training_round=10.0,
        earn_vote=2.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    for i in range(5):
        ledger.earn_training_round("peer-1", f"round-{i}", reputation_score=0.5)
    ledger.earn_vote("peer-1")
    ledger.earn_vote("peer-1")

    balance = ledger.get_balance("peer-1")
    # 5 rounds * 10 * 1.25 + 2 votes * 2 = 62.5 + 4 = 66.5
    expected = 5 * 10 * 1.25 + 2 * 2
    assert abs(balance - expected) < 0.1, f"Expected {expected}, got {balance}"
    print(f"  Accumulated balance: {balance:.1f}")


# === Spending Tests ===


def test_spend_inference_standard():
    """Standard inference deducts credits."""
    config = CreditConfig(
        welcome_bonus=100.0,
        cost_inference_standard=1.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    result = ledger.spend_inference("peer-1")
    assert result.allowed
    assert result.tier == InferenceTier.CONTRIBUTOR
    assert abs(result.credits_spent - 1.0) < 0.01
    assert abs(result.balance_after - 99.0) < 0.01
    assert abs(ledger.get_balance("peer-1") - 99.0) < 0.01
    print(f"  Standard inference: spent={result.credits_spent}, balance={result.balance_after}")


def test_spend_inference_priority():
    """Priority inference costs more."""
    config = CreditConfig(
        welcome_bonus=100.0,
        cost_inference_priority=5.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    result = ledger.spend_inference("peer-1", priority=True)
    assert result.allowed
    assert result.tier == InferenceTier.PRIORITY
    assert abs(result.credits_spent - 5.0) < 0.01
    print(f"  Priority inference: spent={result.credits_spent}")


def test_spend_inference_long_generation():
    """Long generation costs extra."""
    config = CreditConfig(
        welcome_bonus=100.0,
        cost_inference_standard=1.0,
        cost_inference_long=2.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    result = ledger.spend_inference("peer-1", long_generation=True)
    assert result.allowed
    assert abs(result.credits_spent - 3.0) < 0.01  # 1 + 2.
    print(f"  Long generation: spent={result.credits_spent}")


def test_spend_insufficient_credits_falls_to_free():
    """When credits run out, falls back to free tier."""
    config = CreditConfig(
        welcome_bonus=0.0,
        cost_inference_standard=1.0,
        free_tier_requests_per_hour=10,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    result = ledger.spend_inference("peer-1")
    assert result.allowed
    assert result.tier == InferenceTier.FREE
    assert result.credits_spent == 0
    print(f"  Free tier fallback: tier={result.tier.value}")


def test_free_tier_rate_limit():
    """Free tier is rate-limited per hour."""
    config = CreditConfig(
        welcome_bonus=0.0,
        free_tier_requests_per_hour=3,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    # First 3 should work.
    for i in range(3):
        result = ledger.spend_inference("peer-1")
        assert result.allowed, f"Request {i} should be allowed"

    # 4th should be rejected.
    result = ledger.spend_inference("peer-1")
    assert not result.allowed
    assert "rate limit" in result.reason.lower()
    assert result.wait_secs > 0
    print(f"  Rate limited after 3 free requests. Wait: {result.wait_secs:.0f}s")


def test_priority_insufficient_credits_rejected():
    """Priority inference is rejected if credits insufficient (no free fallback)."""
    config = CreditConfig(
        welcome_bonus=1.0,
        cost_inference_priority=5.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    result = ledger.spend_inference("peer-1", priority=True)
    assert not result.allowed
    assert "insufficient" in result.reason.lower()
    print(f"  Priority rejected: {result.reason}")


def test_check_inference_does_not_spend():
    """check_inference previews cost without deducting."""
    config = CreditConfig(welcome_bonus=100.0, decay_enabled=False)
    ledger = CreditLedger(config)

    result = ledger.check_inference("peer-1")
    assert result.allowed
    # Balance should still be 100.
    assert abs(ledger.get_balance("peer-1") - 100.0) < 0.01
    print(f"  Check did not spend: balance still {ledger.get_balance('peer-1'):.0f}")


# === Decay Tests ===


def test_credit_decay():
    """Credits decay over time (exponential with half-life)."""
    config = CreditConfig(
        welcome_bonus=100.0,
        decay_half_life_secs=100.0,  # Short half-life for testing.
        decay_enabled=True,
    )
    ledger = CreditLedger(config)

    # Create account by accessing balance.
    ledger.get_balance("peer-1")
    account = ledger._accounts["peer-1"]

    # Simulate time passing (100s = 1 half-life).
    account.last_decay_time -= 100.0
    balance = ledger.get_balance("peer-1")

    # After one half-life, balance should be ~50.
    assert 45 < balance < 55, f"After 1 half-life expected ~50, got {balance:.1f}"
    print(f"  After 1 half-life: {balance:.1f} (expected ~50)")


def test_decay_disabled():
    """Decay can be disabled."""
    config = CreditConfig(welcome_bonus=100.0, decay_enabled=False)
    ledger = CreditLedger(config)

    ledger.get_balance("peer-1")  # Create account.
    account = ledger._accounts["peer-1"]

    # Simulate time passing.
    account.last_decay_time -= 100000.0
    balance = ledger.get_balance("peer-1")

    assert abs(balance - 100.0) < 0.01
    print(f"  Decay disabled: balance still {balance:.0f}")


# === Non-transferable Tests ===


def test_no_transfer_method():
    """Credits are non-transferable: no transfer method exists."""
    ledger = CreditLedger()
    assert not hasattr(ledger, "transfer")
    assert not hasattr(ledger, "send")
    assert not hasattr(ledger, "transfer_credits")
    print("  Verified: no transfer/send methods exist")


# === Uptime Tests ===


def test_uptime_streak_multiplier():
    """Uptime streak multiplier rewards continuous participation."""
    config = CreditConfig(
        welcome_bonus=0,
        earn_training_round=10.0,
        uptime_streak_hours=1.0,  # 1 hour for testing.
        uptime_streak_multiplier=1.5,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    # Without streak (just connected).
    earned_no_streak = ledger.earn_training_round("peer-1", "r0", reputation_score=0.5)

    # With streak (connected for 2 hours).
    account = ledger._accounts["peer-1"]
    account.connected_since = time.time() - 7200  # 2 hours ago.
    earned_with_streak = ledger.earn_training_round("peer-1", "r1", reputation_score=0.5)

    assert earned_with_streak > earned_no_streak
    ratio = earned_with_streak / earned_no_streak
    assert abs(ratio - 1.5) < 0.01
    print(f"  No streak: {earned_no_streak:.1f}, with streak: {earned_with_streak:.1f} "
          f"(ratio: {ratio:.1f}x)")


def test_connect_disconnect():
    """Connect/disconnect tracking works."""
    config = CreditConfig(welcome_bonus=0, decay_enabled=False)
    ledger = CreditLedger(config)

    ledger.record_connect("peer-1")
    account = ledger.get_account("peer-1")
    assert account.connected_since > 0

    ledger.record_disconnect("peer-1")
    account = ledger.get_account("peer-1")
    assert account.connected_since == 0.0
    assert account.uptime_streak_secs > 0
    print(f"  Connect/disconnect: uptime tracked {account.uptime_streak_secs:.1f}s")


# === Inference Gate Tests ===


def test_inference_gate_authorize():
    """Inference gate authorizes requests correctly."""
    config = CreditConfig(welcome_bonus=100.0, decay_enabled=False)
    ledger = CreditLedger(config)
    gate = InferenceGate(ledger)

    result = gate.authorize("peer-1")
    assert result.allowed
    assert result.tier == InferenceTier.CONTRIBUTOR

    stats = gate.stats()
    assert stats["authorized"] == 1
    assert stats["paid_tier"] == 1
    print(f"  Gate authorized: tier={result.tier.value}, stats={stats}")


def test_inference_gate_free_tier():
    """Inference gate falls back to free tier."""
    config = CreditConfig(welcome_bonus=0.0, decay_enabled=False)
    ledger = CreditLedger(config)
    gate = InferenceGate(ledger)

    result = gate.authorize("peer-1")
    assert result.allowed
    assert result.tier == InferenceTier.FREE

    stats = gate.stats()
    assert stats["free_tier"] == 1
    print(f"  Gate free tier: stats={stats}")


def test_inference_gate_rejection():
    """Inference gate rejects when rate-limited."""
    config = CreditConfig(
        welcome_bonus=0.0,
        free_tier_requests_per_hour=1,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)
    gate = InferenceGate(ledger)

    # First works.
    assert gate.authorize("peer-1").allowed

    # Second rejected.
    result = gate.authorize("peer-1")
    assert not result.allowed

    stats = gate.stats()
    assert stats["rejected"] == 1
    assert stats["rejection_rate"] > 0
    print(f"  Gate rejection: stats={stats}")


# === Query Tests ===


def test_top_contributors():
    """Top contributors ranking works."""
    config = CreditConfig(welcome_bonus=0, decay_enabled=False)
    ledger = CreditLedger(config)

    # Create peers with different earnings.
    for i in range(5):
        for _ in range(i + 1):
            ledger.earn_training_round(f"peer-{i}", reputation_score=0.5)

    top = ledger.top_contributors(3)
    assert len(top) == 3
    assert top[0][0] == "peer-4"  # Most rounds.
    assert top[0][1] > top[1][1] > top[2][1]
    print(f"  Top 3: {[(pid, f'{b:.0f}') for pid, b in top]}")


def test_network_stats():
    """Network-wide credit stats work."""
    config = CreditConfig(welcome_bonus=10, decay_enabled=False)
    ledger = CreditLedger(config)

    for i in range(3):
        ledger.earn_training_round(f"peer-{i}", reputation_score=0.5)

    stats = ledger.network_stats()
    assert stats["total_accounts"] == 3
    assert stats["total_contributors"] == 3
    assert stats["total_rounds_contributed"] == 3
    print(f"  Network stats: {stats}")


def test_peer_summary():
    """Peer summary contains expected fields."""
    config = CreditConfig(welcome_bonus=50, decay_enabled=False)
    ledger = CreditLedger(config)

    ledger.earn_training_round("peer-1", "r0", reputation_score=0.8)
    ledger.earn_vote("peer-1", "prop-1")

    summary = ledger.get_peer_summary("peer-1")
    assert summary["peer_id"] == "peer-1"
    assert summary["balance"] > 50
    assert summary["rounds_contributed"] == 1
    assert summary["votes_cast"] == 1
    assert len(summary["recent_transactions"]) >= 3  # Welcome + round + vote.
    print(f"  Peer summary: balance={summary['balance']}, txns={len(summary['recent_transactions'])}")


def test_transaction_history():
    """Transaction history is recorded and trimmed."""
    config = CreditConfig(welcome_bonus=0, earn_vote=1.0, decay_enabled=False)
    ledger = CreditLedger(config)

    for i in range(250):
        ledger.earn_vote("peer-1")

    account = ledger.get_account("peer-1")
    assert len(account.transactions) <= CreditLedger.MAX_TRANSACTION_HISTORY
    print(f"  Transaction history trimmed to {len(account.transactions)} (max {CreditLedger.MAX_TRANSACTION_HISTORY})")


# === Network Integration Tests ===


# === Vote Deposit / Refund Tests ===


def test_vote_deposit_charged():
    """Vote deposit is charged from peer balance."""
    config = CreditConfig(
        welcome_bonus=100.0,
        vote_deposit=0.5,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    deposit = ledger.charge_vote_deposit("peer-1", "prop-1")
    assert abs(deposit - 0.5) < 0.01
    assert abs(ledger.get_balance("peer-1") - 99.5) < 0.01
    account = ledger.get_account("peer-1")
    assert "prop-1" in account.pending_vote_deposits
    assert abs(account.pending_vote_deposits["prop-1"] - 0.5) < 0.01
    print(f"  Vote deposit charged: {deposit}, balance={ledger.get_balance('peer-1'):.1f}")


def test_vote_deposit_zero_for_broke_peer():
    """Broke peer gets zero deposit (free-tier vote)."""
    config = CreditConfig(
        welcome_bonus=0.0,
        vote_deposit=0.5,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    deposit = ledger.charge_vote_deposit("broke-peer", "prop-1")
    assert deposit == 0.0
    assert ledger.get_balance("broke-peer") == 0.0
    account = ledger.get_account("broke-peer")
    assert account.pending_vote_deposits["prop-1"] == 0.0
    print(f"  Broke peer deposit: {deposit}")


def test_vote_settle_accurate_gets_bonus():
    """Accurate voter gets deposit refund + bonus."""
    config = CreditConfig(
        welcome_bonus=100.0,
        vote_deposit=0.5,
        vote_refund_accurate=1.5,
        vote_refund_inaccurate=0.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    ledger.charge_vote_deposit("peer-1", "prop-1")
    balance_after_deposit = ledger.get_balance("peer-1")

    refund = ledger.settle_vote("peer-1", "prop-1", was_accurate=True)
    assert abs(refund - 1.5) < 0.01
    balance_after_refund = ledger.get_balance("peer-1")
    # Net: -0.5 deposit + 1.5 refund = +1.0 profit
    assert abs(balance_after_refund - (balance_after_deposit + 1.5)) < 0.01
    print(f"  Accurate refund: {refund}, net profit: {balance_after_refund - 100.0:.1f}")


def test_vote_settle_inaccurate_forfeits():
    """Inaccurate voter forfeits deposit (no refund)."""
    config = CreditConfig(
        welcome_bonus=100.0,
        vote_deposit=0.5,
        vote_refund_accurate=1.5,
        vote_refund_inaccurate=0.0,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    ledger.charge_vote_deposit("peer-1", "prop-1")
    balance_after_deposit = ledger.get_balance("peer-1")  # 99.5

    refund = ledger.settle_vote("peer-1", "prop-1", was_accurate=False)
    assert refund == 0.0
    balance_after_refund = ledger.get_balance("peer-1")
    assert abs(balance_after_refund - balance_after_deposit) < 0.01  # No change
    print(f"  Inaccurate refund: {refund}, balance unchanged at {balance_after_refund:.1f}")


def test_vote_settle_no_deposit_noop():
    """Settling a vote with no deposit is a noop."""
    config = CreditConfig(welcome_bonus=100.0, decay_enabled=False)
    ledger = CreditLedger(config)

    refund = ledger.settle_vote("peer-1", "nonexistent-prop", was_accurate=True)
    assert refund == 0.0
    assert abs(ledger.get_balance("peer-1") - 100.0) < 0.01
    print("  No-deposit settle: noop")


def test_earn_vote_backward_compatibility():
    """earn_vote still works and charges deposit when proposal_id given."""
    config = CreditConfig(
        welcome_bonus=100.0,
        earn_vote=2.0,
        vote_deposit=0.5,
        decay_enabled=False,
    )
    ledger = CreditLedger(config)

    earned = ledger.earn_vote("peer-1", "prop-1")
    assert abs(earned - 2.0) < 0.01

    # Balance should be 100 + 2 (earned) - 0.5 (deposit) = 101.5
    balance = ledger.get_balance("peer-1")
    assert abs(balance - 101.5) < 0.01

    account = ledger.get_account("peer-1")
    assert "prop-1" in account.pending_vote_deposits
    assert account.votes_cast == 1
    print(f"  Backward compat: earned={earned}, balance={balance:.1f}")


# === Network Integration Tests ===


def test_network_credits_integration():
    """Credits are wired into the training network."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))

    assert network.credits is not None
    assert network.inference_gate is not None

    # Initial balance: welcome bonus.
    initial_balance = network.credits.get_balance(network.config.peer_id)
    assert initial_balance > 0

    # Run training rounds -- should earn credits.
    for i in range(3):
        network.run_training_round(f"round-{i}")

    final_balance = network.credits.get_balance(network.config.peer_id)
    assert final_balance > initial_balance
    print(f"  Network: initial={initial_balance:.0f}, after 3 rounds={final_balance:.0f}")


def test_network_stats_include_credits():
    """Network stats dict includes credit data."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))
    network.run_training_round("round-0")

    stats = network.get_stats_dict()
    assert "credit_balance" in stats
    assert "credit_earned" in stats
    assert "credit_spent" in stats
    assert "credit_network" in stats
    assert stats["credit_balance"] > 0
    assert stats["credit_earned"] > 0
    print(f"  Stats: balance={stats['credit_balance']}, earned={stats['credit_earned']}")


def test_network_credits_earned_event():
    """Network emits credits_earned events."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))

    earned_events = []
    network.on("credits_earned", lambda pid, amt: earned_events.append((pid, amt)))

    network.run_training_round("round-0")

    assert len(earned_events) == 1
    assert earned_events[0][0] == network.config.peer_id
    assert earned_events[0][1] > 0
    print(f"  Credit event: peer={earned_events[0][0][:8]}..., earned={earned_events[0][1]:.1f}")


def test_network_inference_gate():
    """Inference gate works through the network."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)

    # Should be authorized (has welcome bonus).
    result = network.inference_gate.authorize(network.config.peer_id)
    assert result.allowed
    print(f"  Inference gate: allowed={result.allowed}, tier={result.tier.value}")


if __name__ == "__main__":
    tests = [
        # Core ledger.
        test_new_account_welcome_bonus,
        test_earn_training_round,
        test_reputation_multiplier,
        test_earn_data_contribution,
        test_earn_vote,
        test_multiple_earnings,
        # Spending.
        test_spend_inference_standard,
        test_spend_inference_priority,
        test_spend_inference_long_generation,
        test_spend_insufficient_credits_falls_to_free,
        test_free_tier_rate_limit,
        test_priority_insufficient_credits_rejected,
        test_check_inference_does_not_spend,
        # Decay.
        test_credit_decay,
        test_decay_disabled,
        # Non-transferable.
        test_no_transfer_method,
        # Uptime.
        test_uptime_streak_multiplier,
        test_connect_disconnect,
        # Inference gate.
        test_inference_gate_authorize,
        test_inference_gate_free_tier,
        test_inference_gate_rejection,
        # Queries.
        test_top_contributors,
        test_network_stats,
        test_peer_summary,
        test_transaction_history,
        # Vote deposit / refund.
        test_vote_deposit_charged,
        test_vote_deposit_zero_for_broke_peer,
        test_vote_settle_accurate_gets_bonus,
        test_vote_settle_inaccurate_forfeits,
        test_vote_settle_no_deposit_noop,
        test_earn_vote_backward_compatibility,
        # Network integration.
        test_network_credits_integration,
        test_network_stats_include_credits,
        test_network_credits_earned_event,
        test_network_inference_gate,
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
    print("\nAll credit system tests passed!")
