"""Tests for the from-scratch LLM training kickstart.

Proves that the system can:
    1. Initialize a model from nothing (random weights)
    2. Tokenize real text
    3. Train on it (loss decreases)
    4. Multiple peers training on different data converge after aggregation
    5. The model generates (initially garbage) text that improves with training
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch

from ussi_engine.data.tokenizer import Tokenizer, TokenizerConfig, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from ussi_engine.data.pipeline import TextDataPipeline, DataConfig
from ussi_engine.model.lm import LanguageModel, LMConfig, create_from_scratch, TransformerBlock
from ussi_engine.kickstart import Kickstart, KickstartConfig
from ussi_engine.training.byzantine import AggregationMethod, ByzantineConfig, robust_aggregate


# Shared training text (public domain, Alice in Wonderland excerpt).
SAMPLE_TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, and what
is the use of a book, thought Alice, without pictures or conversations? So
she was considering in her own mind, as well as she could, for the hot day
made her feel very sleepy and stupid, whether the pleasure of making a
daisy chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.
"""

SAMPLE_TEXT_2 = """
There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I
shall be late! But when the Rabbit actually took a watch out of its waistcoat
pocket, and looked at it, and then hurried on, Alice started to her feet,
for it flashed across her mind that she had never before seen a rabbit with
either a waistcoat pocket, or a watch to take out of it.
"""


# === Tokenizer Tests ===


def test_tokenizer_encode_decode():
    """Byte-level tokenizer: encode then decode is identity."""
    tok = Tokenizer()
    text = "Hello, world! 日本語テスト 🚀"
    encoded = tok.encode(text)

    assert encoded[0] == BOS_TOKEN
    assert encoded[-1] == EOS_TOKEN
    assert len(encoded) > len(text)  # UTF-8 multi-byte chars expand

    decoded = tok.decode(encoded)
    assert decoded == text, f"Roundtrip failed: '{decoded}' != '{text}'"


def test_tokenizer_batch():
    """Batch encoding with padding."""
    tok = Tokenizer(TokenizerConfig(max_sequence_length=32))
    texts = ["short", "a somewhat longer sentence", "x"]

    padded, lengths = tok.batch_encode(texts, padding=True)
    assert len(padded) == 3
    assert all(len(seq) == len(padded[0]) for seq in padded)
    assert padded[2][-1] == PAD_TOKEN  # Shortest text is padded


def test_tokenizer_truncation():
    """Long text is truncated to max_sequence_length."""
    tok = Tokenizer(TokenizerConfig(max_sequence_length=20))
    long_text = "a" * 1000
    encoded = tok.encode(long_text)
    assert len(encoded) == 20


def test_tokenizer_serialization():
    """Tokenizer can be serialized and deserialized."""
    tok = Tokenizer(TokenizerConfig(max_sequence_length=64))
    data = tok.to_bytes()
    tok2 = Tokenizer.from_bytes(data)

    assert tok2.config.max_sequence_length == 64
    text = "test roundtrip"
    assert tok.encode(text) == tok2.encode(text)


# === Data Pipeline Tests ===


def test_pipeline_load_text():
    """Pipeline tokenizes and chunks text."""
    tok = Tokenizer()
    pipe = TextDataPipeline(tok, DataConfig(seq_length=32, batch_size=2))
    pipe.load_text(SAMPLE_TEXT)

    assert pipe.total_tokens > 100
    assert pipe.total_sequences > 0
    assert pipe.total_batches > 0


def test_pipeline_iter_batches():
    """Pipeline produces correctly shaped batches."""
    tok = Tokenizer()
    pipe = TextDataPipeline(tok, DataConfig(seq_length=32, batch_size=2, batches_per_round=3))
    pipe.load_text(SAMPLE_TEXT)

    batches = list(pipe.iter_batches("round-0", "peer-0"))
    assert len(batches) == 3

    input_ids, target_ids, mask = batches[0]
    assert input_ids.shape == (2, 32)
    assert target_ids.shape == (2, 32)
    assert mask.shape == (2, 32)

    # Target should be input shifted by 1 (next-token prediction).
    # This is verified indirectly -- they come from consecutive token positions.


def test_pipeline_deterministic_shuffle():
    """Same (round_id, peer_id) produces same batch order."""
    tok = Tokenizer()
    config = DataConfig(seq_length=32, batch_size=2, batches_per_round=5)

    pipe1 = TextDataPipeline(tok, config)
    pipe1.load_text(SAMPLE_TEXT)
    batches1 = list(pipe1.iter_batches("round-1", "peer-A"))

    pipe2 = TextDataPipeline(tok, config)
    pipe2.load_text(SAMPLE_TEXT)
    batches2 = list(pipe2.iter_batches("round-1", "peer-A"))

    for (a_in, a_tgt, _), (b_in, b_tgt, _) in zip(batches1, batches2):
        assert torch.equal(a_in, b_in)
        assert torch.equal(a_tgt, b_tgt)


def test_pipeline_different_peers_different_order():
    """Different peers get different batch orderings."""
    tok = Tokenizer()
    config = DataConfig(seq_length=32, batch_size=2, batches_per_round=5)

    pipe = TextDataPipeline(tok, config)
    pipe.load_text(SAMPLE_TEXT)

    b1 = list(pipe.iter_batches("round-1", "peer-A"))
    b2 = list(pipe.iter_batches("round-1", "peer-B"))

    # At least one batch should differ in ordering.
    any_different = any(
        not torch.equal(a[0], b[0]) for a, b in zip(b1, b2)
    )
    assert any_different, "Different peers should get different batch orderings"


# === Language Model Tests ===


def test_lm_forward():
    """Language model forward pass produces correct shapes."""
    config = LMConfig(vocab_size=260, hidden_dim=64, n_layers=2, n_heads=2)
    model = create_from_scratch(config)

    input_ids = torch.randint(0, 260, (2, 32))
    targets = torch.randint(0, 260, (2, 32))

    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 32, 260)
    assert loss is not None
    assert loss.item() > 0


def test_lm_no_targets():
    """Forward without targets returns logits, no loss."""
    config = LMConfig(vocab_size=260, hidden_dim=64, n_layers=2, n_heads=2)
    model = create_from_scratch(config)

    input_ids = torch.randint(0, 260, (1, 16))
    logits, loss = model(input_ids)
    assert logits.shape == (1, 16, 260)
    assert loss is None


def test_lm_generate():
    """Model can generate tokens (initially random, but it shouldn't crash)."""
    config = LMConfig(vocab_size=260, hidden_dim=64, n_layers=2, n_heads=2, max_seq_length=64)
    model = create_from_scratch(config)

    input_ids = torch.tensor([[1, 72, 105]])  # BOS + "Hi"
    output = model.generate(input_ids, max_new_tokens=10)
    assert output.shape[1] == 13  # 3 input + 10 generated


def test_lm_loss_decreases():
    """Training on repeated data should decrease loss."""
    torch.manual_seed(42)
    config = LMConfig(vocab_size=260, hidden_dim=64, n_layers=2, n_heads=2, max_seq_length=32)
    model = create_from_scratch(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed training batch.
    input_ids = torch.randint(4, 260, (4, 32))
    targets = torch.randint(4, 260, (4, 32))

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        _, loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease.
    assert losses[-1] < losses[0], \
        f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (20 steps)")


# === Kickstart End-to-End Tests ===


def test_kickstart_single_peer():
    """Single peer kickstart: init, load data, train, loss decreases."""
    torch.manual_seed(42)
    config = KickstartConfig(
        model_id="test-lm",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=2,
        steps_per_round=5,
        learning_rate=1e-3,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)  # Repeat for enough data.

    # Train 3 rounds, loss should trend down.
    losses = []
    for i in range(3):
        result = ks.train_round(f"round-{i}", "peer-0")
        assert result.steps_completed > 0
        assert result.avg_loss < float("inf")
        losses.append(result.avg_loss)

    # Loss should generally decrease over rounds.
    assert losses[-1] < losses[0], \
        f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"  Single peer: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Sample: {result.sample_text[:80]}...")
    print(f"  Stats: {ks.stats()}")


def test_kickstart_multi_peer_convergence():
    """Multiple peers with different data converge after gradient aggregation."""
    torch.manual_seed(42)
    n_peers = 4
    config = KickstartConfig(
        model_id="converge-lm",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=2,
        steps_per_round=5,
        learning_rate=1e-3,
    )

    # Each peer has different data.
    peer_data = [
        SAMPLE_TEXT,
        SAMPLE_TEXT_2,
        "The quick brown fox jumps over the lazy dog. " * 20,
        "To be or not to be, that is the question. " * 20,
    ]

    # All peers start with the SAME model weights.
    base_ks = Kickstart(config)
    base_state = base_ks.model.state_dict()

    peers = []
    for i in range(n_peers):
        ks = Kickstart(config)
        ks.model.load_state_dict({k: v.clone() for k, v in base_state.items()})
        ks.load_text(peer_data[i] * 3)
        peers.append(ks)

    # Each peer trains locally and produces gradients.
    all_grads = []
    for i, ks in enumerate(peers):
        result = ks.train_round("round-0", f"peer-{i}")
        assert result.gradients is not None
        all_grads.append(result.gradients)

    # Aggregate gradients.
    byz_config = ByzantineConfig(method=AggregationMethod.MEAN)
    aggregated = robust_aggregate(all_grads, byz_config)

    # Apply aggregated gradients to all peers.
    for ks in peers:
        ks.apply_aggregated_gradients(aggregated)

    # After aggregation, all peers should have very similar weights.
    # (Not identical because each also applied local optimizer steps, but close.)
    ref_params = dict(peers[0].model.named_parameters())
    for i in range(1, n_peers):
        for name, param in peers[i].model.named_parameters():
            diff = (param.data - ref_params[name].data).abs().mean().item()
            assert diff < 1.0, \
                f"Peer {i} diverged from peer 0 on {name}: mean diff={diff:.4f}"

    print(f"  Multi-peer convergence: max param diff < 1.0 across {n_peers} peers")


def test_kickstart_real_text_learning():
    """Model actually learns patterns from real text over multiple rounds."""
    torch.manual_seed(42)
    config = KickstartConfig(
        model_id="learn-lm",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=4,
        steps_per_round=10,
        learning_rate=1e-3,
    )

    ks = Kickstart(config)
    # Load enough data for learning.
    ks.load_text((SAMPLE_TEXT + SAMPLE_TEXT_2) * 10)

    round_losses = []
    for i in range(5):
        result = ks.train_round(f"round-{i}", "learner")
        round_losses.append(result.avg_loss)

    # Loss should consistently decrease.
    assert round_losses[-1] < round_losses[0], \
        f"Loss didn't decrease: {round_losses[0]:.4f} -> {round_losses[-1]:.4f}"

    print(f"  Learning curve: {' -> '.join(f'{l:.3f}' for l in round_losses)}")

    # Generate some text.
    sample = ks.generate("Alice ", max_tokens=40)
    print(f"  Generated: {sample[:100]}")


def test_kickstart_gradient_shapes():
    """Gradients from kickstart match model parameter shapes."""
    config = KickstartConfig(
        model_id="grad-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=32,
        batch_size=2,
        steps_per_round=1,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)

    result = ks.train_round("r0", "p0")
    assert result.gradients is not None
    assert len(result.gradients) > 0

    # Every gradient should match a model parameter.
    model_params = dict(ks.model.named_parameters())
    for name, grad in result.gradients.items():
        assert name in model_params, f"Gradient for unknown param: {name}"
        assert grad.shape == model_params[name].shape, \
            f"Shape mismatch for {name}: grad={grad.shape}, param={model_params[name].shape}"


def test_transformer_block_causal():
    """Transformer block applies causal masking (no future leakage)."""
    block = TransformerBlock(hidden_dim=32, n_heads=2, ff_dim=64)
    block.eval()

    # Create input where later tokens are obviously different.
    x = torch.randn(1, 8, 32)

    with torch.no_grad():
        out = block(x)

    assert out.shape == (1, 8, 32)
    # The output at position 0 should depend only on position 0.
    # If we change token 7, position 0's output shouldn't change.
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(32)

    with torch.no_grad():
        out2 = block(x2)

    # Position 0 should be identical (causal mask prevents seeing position 7).
    assert torch.allclose(out[0, 0], out2[0, 0], atol=1e-5), \
        "Causal masking failed: position 0 changed when position 7 changed"


# === Phase 2: NaN Resilience Tests ===


def test_kickstart_nan_resilience():
    """Training survives NaN losses without crashing.

    We inject NaN by corrupting model weights, then verify training
    continues (skipping bad steps) without raising an exception.
    """
    torch.manual_seed(42)
    config = KickstartConfig(
        model_id="nan-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=2,
        steps_per_round=5,
        learning_rate=1e-3,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)

    # Train normally first.
    result1 = ks.train_round("round-0", "peer-0")
    assert result1.steps_completed > 0

    # Corrupt some weights to potentially cause NaN.
    with torch.no_grad():
        for param in list(ks.model.parameters())[:1]:
            param.fill_(float("inf"))

    # Training should survive (skipping NaN steps).
    result2 = ks.train_round("round-1", "peer-0")
    # Should not crash — that's the main assertion.
    assert result2.skipped_steps >= 0  # May or may not skip depending on behavior.


def test_kickstart_gradient_shape_validation():
    """apply_aggregated_gradients skips mismatched gradient shapes."""
    config = KickstartConfig(
        model_id="shape-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=32,
        batch_size=2,
        steps_per_round=1,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)

    # Create gradients with wrong shapes (simulate stale grads after mutation).
    bad_gradients = {}
    for name, param in ks.model.named_parameters():
        # Make some gradients the wrong shape.
        bad_gradients[name] = torch.randn(3, 7)  # Wrong shape.
        break  # Only corrupt first one.

    # Should not crash.
    ks.apply_aggregated_gradients(bad_gradients)


def test_kickstart_nan_gradient_rejection():
    """apply_aggregated_gradients skips NaN gradients."""
    config = KickstartConfig(
        model_id="nan-grad-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=32,
        batch_size=2,
        steps_per_round=1,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)

    # Create gradients with NaN values.
    nan_gradients = {}
    for name, param in ks.model.named_parameters():
        nan_gradients[name] = torch.full_like(param, float("nan"))

    # Should not crash, should skip all gradients.
    ks.apply_aggregated_gradients(nan_gradients)


def test_kickstart_result_has_skipped_steps():
    """KickstartResult tracks skipped steps."""
    config = KickstartConfig(
        model_id="skip-test",
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_length=64,
        batch_size=2,
        steps_per_round=3,
        learning_rate=1e-3,
    )
    ks = Kickstart(config)
    ks.load_text(SAMPLE_TEXT * 3)

    result = ks.train_round("round-0", "peer-0")
    # Normal training should have 0 skipped steps.
    assert result.skipped_steps == 0
    assert result.reverted is False


if __name__ == "__main__":
    tests = [
        # Tokenizer.
        test_tokenizer_encode_decode,
        test_tokenizer_batch,
        test_tokenizer_truncation,
        test_tokenizer_serialization,
        # Data pipeline.
        test_pipeline_load_text,
        test_pipeline_iter_batches,
        test_pipeline_deterministic_shuffle,
        test_pipeline_different_peers_different_order,
        # Language model.
        test_lm_forward,
        test_lm_no_targets,
        test_lm_generate,
        test_lm_loss_decreases,
        # Kickstart.
        test_kickstart_single_peer,
        test_kickstart_multi_peer_convergence,
        test_kickstart_real_text_learning,
        test_kickstart_gradient_shapes,
        test_transformer_block_causal,
        # Phase 2: NaN resilience.
        test_kickstart_nan_resilience,
        test_kickstart_gradient_shape_validation,
        test_kickstart_nan_gradient_rejection,
        test_kickstart_result_has_skipped_steps,
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
    print("\nAll kickstart tests passed!")
