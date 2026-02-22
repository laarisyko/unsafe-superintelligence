"""Tests for synthetic data generation using teacher models."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from ussi_engine.teacher import (
    TeacherConfig, LocalTeacher, create_teacher, parse_teacher_string, TokenBucket,
)
from ussi_engine.data.synthetic import (
    SyntheticConfig, SyntheticDataGenerator, TOPIC_REGISTRY, ngram_overlap,
)
from ussi_engine.data.pipeline import TextDataPipeline, DataConfig
from ussi_engine.data.tokenizer import Tokenizer, TokenizerConfig
from ussi_engine.kickstart import Kickstart, KickstartConfig

SAMPLE_TEXT = "Alice was beginning to get very tired of sitting by her sister. " * 20


def test_topic_registry_not_empty():
    assert len(TOPIC_REGISTRY) >= 20
    for t in TOPIC_REGISTRY:
        assert isinstance(t, str) and len(t.strip()) > 0


def test_create_local_teacher():
    teacher = create_teacher(TeacherConfig(provider="local"))
    assert isinstance(teacher, LocalTeacher)


def test_parse_teacher_string():
    c = parse_teacher_string("anthropic:claude-sonnet-4-20250514")
    assert c.provider == "anthropic"
    assert c.model == "claude-sonnet-4-20250514"


def test_generate_batch_with_local_teacher():
    """LocalTeacher (no kickstart = template fallback) generates text."""
    teacher = create_teacher(TeacherConfig(provider="local"))
    gen = SyntheticDataGenerator(SyntheticConfig(min_length=5, samples_per_topic=1), teacher)
    texts = gen.generate_batch(n=2)
    assert len(texts) > 0
    assert all(len(t) > 0 for t in texts)


def test_feed_to_pipeline():
    teacher = create_teacher(TeacherConfig(provider="local"))
    tok = Tokenizer(TokenizerConfig(max_sequence_length=64))
    pipe = TextDataPipeline(tok, DataConfig(seq_length=32, batch_size=2))
    gen = SyntheticDataGenerator(SyntheticConfig(min_length=5), teacher)
    fed = gen.feed_to_pipeline(pipe, n=2)
    assert fed > 0
    assert pipe.total_tokens > 0


def test_dedup_rejects_identical():
    teacher = create_teacher(TeacherConfig(provider="local"))
    gen = SyntheticDataGenerator(SyntheticConfig(dedup_threshold=0.5, min_length=5), teacher)
    texts = gen.generate_for_topic("test", n=1)
    assert len(texts) > 0
    assert gen._is_duplicate(texts[0])


def test_ngram_overlap_identical():
    assert ngram_overlap("The quick brown fox", "The quick brown fox") == 1.0


def test_ngram_overlap_different():
    assert ngram_overlap("The quick brown fox", "Quantum mechanics describes") < 0.5


def test_kickstart_generate_synthetic_data():
    ks = Kickstart(KickstartConfig(
        model_id="t", hidden_dim=32, n_layers=1, n_heads=2,
        max_seq_length=32, batch_size=2, steps_per_round=2,
    ))
    ks.load_text(SAMPLE_TEXT)
    before = ks.data.total_tokens
    fed = ks.generate_synthetic_data(TeacherConfig(provider="local"), n_samples=2)
    assert fed > 0
    assert ks.data.total_tokens > before


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  [PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    if failed: sys.exit(1)
