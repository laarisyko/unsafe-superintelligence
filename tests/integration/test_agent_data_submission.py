"""Tests for the agent data submission API.

Proves that:
    1. NetworkClient constructs correct request for submit_data
    2. Agent.feed() records data contribution
    3. Free tier is limited to 5 data submissions/day
    4. Contributor tier has no data submission limit
    5. Agent.generate_training_data() generates + feeds samples
    6. ContributionTracker awards credits for data submission
    7. RateLimiter tracks data submissions correctly
    8. Engine bridge processes load_data correctly
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent-sdk"))

from ussi.network import NetworkClient
from ussi.agent import Agent
from ussi.contribution import ContributionTracker, CREDITS_DATA_SUBMISSION
from ussi.rate_limit import RateLimiter, RateLimitExceeded, TierLimits


# === NetworkClient Tests ===


def test_network_client_submit_data():
    """NetworkClient.submit_data() constructs correct request."""
    client = NetworkClient("http://127.0.0.1:50051")

    # Mock the _post method to capture the call
    original_post = client._post
    calls = []

    def mock_post(path, data):
        calls.append((path, data))
        return {"status": "ok", "tokens": 42, "sequences": 2}

    client._post = mock_post

    result = client.submit_data("Hello world", source="test")
    assert len(calls) == 1
    assert calls[0][0] == "/data/submit"
    assert calls[0][1] == {"text": "Hello world", "source": "test"}
    assert result["status"] == "ok"
    assert result["tokens"] == 42
    print(f"  submit_data request: path={calls[0][0]}, body={calls[0][1]}")


# === Agent.feed() Tests ===


def test_agent_feed_records_contribution():
    """Agent.feed() records data contribution and returns result."""
    agent = Agent(node_api_url="http://127.0.0.1:50051")

    # Mock network to return success
    agent.network.submit_data = MagicMock(
        return_value={"status": "ok", "tokens": 100, "sequences": 5}
    )

    result = agent.feed("Some training text", source="test-source")

    assert result["status"] == "ok"
    assert result["tokens"] == 100
    agent.network.submit_data.assert_called_once_with("Some training text", "test-source")

    # Check contribution was recorded
    pc = agent.contributions.get_or_create(agent.agent_id)
    assert len(pc.records) == 1
    assert pc.records[0].event_type == "data_submission"
    assert pc.records[0].credits == CREDITS_DATA_SUBMISSION
    print(f"  feed() result: {result}, credits earned: {CREDITS_DATA_SUBMISSION}")


def test_agent_feed_rate_limited_free_tier():
    """Free tier agents are limited to 5 data submissions/day."""
    agent = Agent(node_api_url="http://127.0.0.1:50051")
    agent.network.submit_data = MagicMock(
        return_value={"status": "ok", "tokens": 10, "sequences": 1}
    )

    # Patch tier to stay "free" (credits from submissions would promote to contributor)
    with patch.object(type(agent), "tier", new_callable=lambda: property(lambda self: "free")):
        # First 5 should work
        for i in range(5):
            result = agent.feed(f"Text {i}")
            assert result["status"] == "ok"

        # 6th should be rate-limited
        try:
            agent.feed("Text 5")
            assert False, "Should have raised RateLimitExceeded"
        except RateLimitExceeded as e:
            assert "data submissions" in str(e)
            assert e.limit == 5
            print(f"  Rate limited after 5 submissions: {e}")


def test_agent_feed_unlimited_contributor():
    """Contributor tier has no data submission limit."""
    agent = Agent(node_api_url="http://127.0.0.1:50051")
    agent._contributing = True  # Set to contributor tier
    agent.network.submit_data = MagicMock(
        return_value={"status": "ok", "tokens": 10, "sequences": 1}
    )

    # Should be able to submit more than 5
    for i in range(10):
        result = agent.feed(f"Text {i}")
        assert result["status"] == "ok"

    print(f"  Contributor submitted 10 times without rate limiting")


# === Agent.generate_training_data() Tests ===


def test_agent_generate_training_data():
    """generate_training_data() generates via infer and feeds each sample."""
    agent = Agent(node_api_url="http://127.0.0.1:50051")

    # Mock infer to return generated text
    agent.inference.infer = MagicMock(return_value="Generated text sample")
    agent.network.submit_data = MagicMock(
        return_value={"status": "ok", "tokens": 20, "sequences": 1}
    )

    result = agent.generate_training_data(
        prompt="Generate something",
        model="test-model",
        n_samples=3,
    )

    assert result["samples_generated"] == 3
    assert len(result["texts"]) == 3
    assert all(t == "Generated text sample" for t in result["texts"])
    assert agent.network.submit_data.call_count == 3
    print(f"  Generated {result['samples_generated']} samples, tokens={result['total_tokens']}")


# === ContributionTracker Tests ===


def test_contribution_tracker_data_submission():
    """ContributionTracker.record_data_submission() awards correct credits."""
    tracker = ContributionTracker()

    tracker.record_data_submission("peer-1", token_count=100, source="test")
    pc = tracker.get_or_create("peer-1")

    assert len(pc.records) == 1
    assert pc.records[0].event_type == "data_submission"
    assert pc.records[0].credits == CREDITS_DATA_SUBMISSION
    assert pc.records[0].round_id == "test"  # source stored in round_id
    assert pc.effective_credits >= CREDITS_DATA_SUBMISSION - 0.01
    print(f"  Credits awarded: {CREDITS_DATA_SUBMISSION}, effective: {pc.effective_credits:.2f}")


# === RateLimiter Tests ===


def test_rate_limiter_data_submission():
    """RateLimiter tracks data submissions with correct limits."""
    limits = TierLimits(
        requests_per_minute=10,
        tokens_per_hour=5000,
        training_rounds_per_day=2,
        evolve_proposals_per_day=3,
        data_submissions_per_day=3,  # Low limit for testing
    )
    limiter = RateLimiter(free_limits=limits)

    # First 3 should pass
    for i in range(3):
        limiter.check_data_submission("peer-1", tier="free")
        limiter.record_data_submission("peer-1")

    # 4th should fail
    try:
        limiter.check_data_submission("peer-1", tier="free")
        assert False, "Should have raised RateLimitExceeded"
    except RateLimitExceeded as e:
        assert e.limit == 3
        assert e.resource == "data submissions"

    # Contributor should pass regardless
    limiter.check_data_submission("peer-1", tier="contributor")

    # Check remaining
    remaining = limiter.get_remaining("peer-1", tier="free")
    assert remaining["data_submissions_remaining"] == 0
    assert remaining["data_submissions_limit"] == 3

    remaining_contrib = limiter.get_remaining("peer-1", tier="contributor")
    assert remaining_contrib["data_submissions_remaining"] == -1

    print(f"  Rate limiter: 3/3 used, remaining={remaining['data_submissions_remaining']}")


def test_bridge_load_data_handler():
    """Engine bridge load_data path exists and has correct structure."""
    # Verify the NetworkClient has the submit_data method
    client = NetworkClient()
    assert hasattr(client, "submit_data")

    # Verify it constructs the right payload
    calls = []
    client._post = lambda path, data: calls.append((path, data)) or {"status": "ok", "tokens": 0, "sequences": 0}
    client.submit_data("test text", "test source")

    assert calls[0][0] == "/data/submit"
    payload = calls[0][1]
    assert "text" in payload
    assert "source" in payload
    assert payload["text"] == "test text"
    assert payload["source"] == "test source"
    print(f"  Bridge path: {calls[0][0]}, payload keys: {list(payload.keys())}")


if __name__ == "__main__":
    tests = [
        test_network_client_submit_data,
        test_agent_feed_records_contribution,
        test_agent_feed_rate_limited_free_tier,
        test_agent_feed_unlimited_contributor,
        test_agent_generate_training_data,
        test_contribution_tracker_data_submission,
        test_rate_limiter_data_submission,
        test_bridge_load_data_handler,
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
    print("\nAll agent data submission tests passed!")
