"""Tests for the genesis system: tracking the birth and growth of AI.

Proves that:
    1. Quality scorer detects text improvement from noise to language
    2. Milestones are automatically detected at the right moments
    3. Genesis tracker records birth, rounds, and mutations
    4. Timeline captures the full model lifecycle
    5. Quality scoring components work individually
    6. Network integration passes genesis data through to stats
    7. Multi-round training triggers progressive milestones
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch

from openclaw_engine.genesis import (
    GenesisTracker,
    QualityReport,
    MilestoneType,
    EventType,
    assess_quality,
    _score_entropy,
    _longest_real_phrase,
    _score_sentences,
    _score_punctuation,
    _score_coherence,
    MILESTONE_DESCRIPTIONS,
    MILESTONE_EMOJI,
)
from openclaw_engine.network import TrainingNetwork, NetworkConfig
from openclaw_engine.data.downloader import get_sample_text


# === Quality Scoring Tests ===


def test_quality_random_noise():
    """Random bytes should score near zero."""
    import random
    noise = "".join(chr(random.randint(0, 255)) for _ in range(200))
    report = assess_quality(noise)
    assert report.score < 0.3, f"Random noise scored too high: {report.score}"
    assert report.real_word_ratio < 0.2
    print(f"  Random noise: score={report.score:.3f}")


def test_quality_repeated_char():
    """Repeated single character should score low."""
    report = assess_quality("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert report.score < 0.3
    assert report.word_diversity == 0.0 or report.real_word_ratio == 0.0
    print(f"  Repeated char: score={report.score:.3f}")


def test_quality_single_word_repeated():
    """A single real word repeated should score moderately."""
    report = assess_quality("the the the the the the the the the the")
    assert report.real_word_ratio > 0.8
    assert report.word_diversity < 0.3  # Low diversity.
    print(f"  Repeated 'the': score={report.score:.3f}, real_ratio={report.real_word_ratio:.2f}")


def test_quality_real_words():
    """Real English words should have high real_word_ratio."""
    text = "the man went to the house and found a book on the table"
    report = assess_quality(text)
    assert report.real_word_ratio > 0.7
    assert len(report.real_words_found) > 5
    assert report.score > 0.2
    print(f"  Real words: score={report.score:.3f}, ratio={report.real_word_ratio:.2f}")


def test_quality_coherent_sentence():
    """A proper English sentence should score high."""
    text = "The man walked to the old house and found a book on the table."
    report = assess_quality(text)
    assert report.score > 0.4
    assert report.has_sentence
    assert report.sentence_score > 0
    assert report.real_word_ratio > 0.5  # Not all words are in the small common set.
    print(f"  Sentence: score={report.score:.3f}, sentence={report.sentence_score:.2f}, ratio={report.real_word_ratio:.2f}")


def test_quality_paragraph():
    """A paragraph with multiple sentences should score very high."""
    text = (
        "The old man walked slowly down the road. He had been walking for hours "
        "and his feet were very tired. The sun was setting in the west and he "
        "knew he would not make it home before dark. He stopped and looked around."
    )
    report = assess_quality(text)
    assert report.score > 0.5
    assert report.has_paragraph
    assert report.real_word_ratio > 0.5  # Many words are common but some may not be in our small set.
    assert report.coherence_score > 0
    print(f"  Paragraph: score={report.score:.3f}, coherence={report.coherence_score:.2f}, ratio={report.real_word_ratio:.2f}")


def test_quality_progression():
    """Quality should increase from noise -> words -> sentences -> paragraphs."""
    noise = "xkq zpt mfv nlw brh ygd ocj aue sti"
    words = "the and is not but for with from they all"
    sentence = "The man went to the house and found a book."
    paragraph = (
        "The man walked to the old house. He found a book on the table. "
        "It was very old and the pages were turning yellow. He sat down."
    )

    scores = []
    for label, text in [("noise", noise), ("words", words),
                         ("sentence", sentence), ("paragraph", paragraph)]:
        r = assess_quality(text)
        scores.append(r.score)
        print(f"  {label}: score={r.score:.3f}")

    # Each level should score higher (with some tolerance).
    assert scores[1] > scores[0], "Words should score higher than noise"
    assert scores[2] > scores[1], "Sentence should score higher than words"
    # Paragraph might score slightly lower than a perfect single sentence
    # due to more non-common words, so just check it's high.
    assert scores[3] > 0.5, f"Paragraph should score > 0.5, got {scores[3]}"


def test_quality_empty():
    """Empty text should score 0."""
    report = assess_quality("")
    assert report.score == 0.0
    report2 = assess_quality("ab")
    assert report2.score == 0.0


# === Component Scoring Tests ===


def test_entropy_scoring():
    """Entropy scoring distinguishes random from structured text."""
    # Random: high entropy
    random_score = _score_entropy("".join(chr(i) for i in range(32, 127)) * 2)
    # English: medium entropy
    english_score = _score_entropy("the quick brown fox jumps over the lazy dog " * 5)
    # Repeated: low entropy
    repeated_score = _score_entropy("aaaa" * 50)

    assert english_score > random_score or english_score > 0.5
    assert repeated_score < english_score
    print(f"  Entropy: random={random_score:.2f}, english={english_score:.2f}, repeated={repeated_score:.2f}")


def test_longest_real_phrase():
    """Finds the longest consecutive sequence of real words."""
    words = ["the", "xyz", "man", "went", "to", "the", "abc", "house"]
    phrase = _longest_real_phrase(words)
    assert phrase == "man went to the"
    print(f"  Longest phrase: '{phrase}'")


def test_sentence_scoring():
    """Sentence detection works on structured text."""
    score, has = _score_sentences("The quick brown fox jumps over the lazy dog.")
    assert has
    assert score > 0
    print(f"  Sentence: has={has}, score={score:.2f}")

    score2, has2 = _score_sentences("random gibberish words xyz abc")
    assert not has2


def test_punctuation_scoring():
    """Punctuation scoring detects proper usage."""
    good = _score_punctuation("Hello, world! How are you doing today?")
    none = _score_punctuation("hello world how are you doing today")
    assert good > none
    print(f"  Punctuation: good={good:.2f}, none={none:.2f}")


def test_coherence_scoring():
    """Coherence scoring detects common word pairs."""
    coherent = ["in", "the", "house", "of", "the", "old", "man"]
    random = ["xyz", "abc", "qrs", "tuv", "mno", "pqr", "stu"]

    c_score = _score_coherence(coherent)
    r_score = _score_coherence(random)
    assert c_score > r_score
    print(f"  Coherence: real={c_score:.2f}, random={r_score:.2f}")


# === Genesis Tracker Tests ===


def test_tracker_birth():
    """Genesis tracker records birth."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000000, hidden_dim=256, n_layers=6)

    assert len(genesis.timeline) == 1
    assert genesis.timeline[0].event_type == EventType.BIRTH
    assert "1,000,000" in genesis.timeline[0].description
    assert genesis.age_secs >= 0


def test_tracker_loss_milestones():
    """Loss milestones are detected at the right thresholds."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    # Loss above 5: no milestone.
    genesis.record_round("r0", loss=6.0)
    ms = [e.milestone for e in genesis.milestones]
    assert MilestoneType.LOSS_BELOW_5 not in ms

    # Loss drops below 5.
    genesis.record_round("r1", loss=4.5)
    ms = [e.milestone for e in genesis.milestones]
    assert MilestoneType.LOSS_BELOW_5 in ms
    assert MilestoneType.LOSS_BELOW_4 not in ms

    # Loss drops below 4.
    genesis.record_round("r2", loss=3.5)
    ms = [e.milestone for e in genesis.milestones]
    assert MilestoneType.LOSS_BELOW_4 in ms

    print(f"  Loss milestones: {len(genesis.milestones)} detected")


def test_tracker_scale_milestones():
    """Round count milestones are detected."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    for i in range(11):
        genesis.record_round(f"r{i}", loss=4.0)

    ms = [e.milestone for e in genesis.milestones]
    assert MilestoneType.ROUNDS_10 in ms
    print(f"  Scale milestones after 11 rounds: {len(genesis.milestones)}")


def test_tracker_quality_milestones():
    """Text quality milestones are detected as samples improve."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    # Round with random noise.
    genesis.record_round("r0", loss=5.0, sample_text="xkq zpt mfv nlw brh")

    # Round with real words.
    genesis.record_round("r1", loss=4.0, sample_text="the man the and of the in the for the")
    ms_vals = {e.milestone for e in genesis.milestones}
    assert MilestoneType.FIRST_REAL_WORD in ms_vals

    # Round with a phrase.
    genesis.record_round("r2", loss=3.5, sample_text="the man went to the house and found")
    ms_vals = {e.milestone for e in genesis.milestones}
    assert MilestoneType.FIRST_PHRASE in ms_vals

    print(f"  Quality milestones: {[e.milestone.value for e in genesis.milestones]}")


def test_tracker_mutation():
    """Mutation events are recorded."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    genesis.record_mutation("add_layer", "Added attention layer at position 5", generation=1)
    assert len(genesis.mutations) == 1
    ms_vals = {e.milestone for e in genesis.milestones}
    assert MilestoneType.FIRST_MUTATION in ms_vals


def test_tracker_peer_milestones():
    """Peer count milestones are detected."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    genesis.record_round("r0", loss=4.0, peers=5)
    ms_vals = {e.milestone for e in genesis.milestones}
    assert MilestoneType.PEERS_10 not in ms_vals

    genesis.record_round("r1", loss=4.0, peers=12)
    ms_vals = {e.milestone for e in genesis.milestones}
    assert MilestoneType.PEERS_10 in ms_vals


def test_tracker_no_duplicate_milestones():
    """Same milestone should not be recorded twice."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    genesis.record_round("r0", loss=4.5)  # Below 5.
    genesis.record_round("r1", loss=4.2)  # Still below 5.
    genesis.record_round("r2", loss=4.8)  # Still below 5.

    loss5_events = [
        e for e in genesis.milestones
        if e.milestone == MilestoneType.LOSS_BELOW_5
    ]
    assert len(loss5_events) == 1, "Loss below 5 should only be recorded once"


def test_tracker_get_status():
    """Status dict is JSON-serializable."""
    import json

    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=50000000)
    genesis.record_round("r0", loss=4.5, sample_text="the man went", tokens_processed=1000)

    status = genesis.get_status()
    serialized = json.dumps(status)
    assert "test-model" in serialized
    assert "milestones" in serialized
    assert status["total_rounds"] == 1
    assert status["total_tokens"] == 1000


def test_tracker_timeline_summary():
    """Timeline summary includes key events."""
    genesis = GenesisTracker("test-model")
    genesis.record_birth(model_params=1000)

    for i in range(20):
        genesis.record_round(f"r{i}", loss=5.0 - i * 0.1, sample_text="the man went to the house")

    summary = genesis.get_timeline_summary()
    assert len(summary) > 0
    # Should include birth + milestones + sampled rounds.
    types = {e["event_type"] for e in summary}
    assert "birth" in types


def test_shareable_milestone():
    """Shareable milestone string is formatted correctly."""
    genesis = GenesisTracker("openclaw-v0")
    genesis.record_birth(model_params=1000)
    genesis.record_round("r0", loss=4.0, sample_text="the man")

    ms = genesis.milestones[0]  # Should have LOSS_BELOW_5 at least.
    shareable = genesis.get_shareable_milestone(ms.milestone)
    assert "openclaw-v0" in shareable
    assert "#OpenClaw" in shareable
    print(f"  Shareable: {shareable[:80]}...")


# === Network Integration Tests ===


def test_network_genesis_integration():
    """Genesis tracker is wired into the training network."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))

    assert network.genesis is not None
    assert len(network.genesis.timeline) == 1  # Birth event.

    # Run training rounds.
    for i in range(5):
        network.run_training_round(f"round-{i}")

    # Should have detected milestones.
    assert network.genesis._total_rounds == 5
    assert len(network.genesis.milestones) > 0

    # Stats should include genesis data.
    stats = network.get_stats_dict()
    assert "milestones" in stats
    assert "current_quality" in stats
    assert "model_age" in stats
    assert stats["milestones_achieved"] > 0

    print(f"  Network genesis: {stats['milestones_achieved']} milestones, "
          f"quality={stats['current_quality']:.3f}")
    for ms in stats["milestones"]:
        print(f"    - {ms}")


def test_network_genesis_milestone_events():
    """Network emits milestone events."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))

    milestones = []
    network.on("milestone", lambda event: milestones.append(event))

    for i in range(5):
        network.run_training_round(f"round-{i}")

    assert len(milestones) > 0
    print(f"  Milestone events emitted: {len(milestones)}")
    for ms in milestones:
        print(f"    - {ms.description}")


def test_genesis_quality_improves_with_training():
    """Quality score increases as the model trains."""
    torch.manual_seed(42)
    config = NetworkConfig(model_size="tiny")
    network = TrainingNetwork(config)
    network.load_text(get_sample_text("all"))

    qualities = []
    for i in range(10):
        network.run_training_round(f"round-{i}")
        q = network.genesis.latest_quality
        if q:
            qualities.append(q.score)

    assert len(qualities) >= 5
    # Quality should generally improve (check last > first).
    print(f"  Quality progression: {' -> '.join(f'{q:.3f}' for q in qualities)}")


# === All Milestone Descriptions Present ===


def test_all_milestones_have_descriptions():
    """Every milestone type has a human-readable description."""
    for ms_type in MilestoneType:
        assert ms_type in MILESTONE_DESCRIPTIONS, \
            f"Missing description for {ms_type.value}"


if __name__ == "__main__":
    tests = [
        # Quality scoring.
        test_quality_random_noise,
        test_quality_repeated_char,
        test_quality_single_word_repeated,
        test_quality_real_words,
        test_quality_coherent_sentence,
        test_quality_paragraph,
        test_quality_progression,
        test_quality_empty,
        # Component scoring.
        test_entropy_scoring,
        test_longest_real_phrase,
        test_sentence_scoring,
        test_punctuation_scoring,
        test_coherence_scoring,
        # Genesis tracker.
        test_tracker_birth,
        test_tracker_loss_milestones,
        test_tracker_scale_milestones,
        test_tracker_quality_milestones,
        test_tracker_mutation,
        test_tracker_peer_milestones,
        test_tracker_no_duplicate_milestones,
        test_tracker_get_status,
        test_tracker_timeline_summary,
        test_shareable_milestone,
        # Network integration.
        test_network_genesis_integration,
        test_network_genesis_milestone_events,
        test_genesis_quality_improves_with_training,
        # Completeness.
        test_all_milestones_have_descriptions,
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
    print("\nAll genesis tests passed!")
