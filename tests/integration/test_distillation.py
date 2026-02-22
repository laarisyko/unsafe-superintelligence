"""Tests for knowledge distillation from teacher models."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn.functional as F

from ussi_engine.teacher import TeacherConfig, LocalTeacher, create_teacher
from ussi_engine.training.distillation import (
    DistillationConfig, DistillationKickstart, TeacherLogitProvider, distillation_loss,
)
from ussi_engine.kickstart import KickstartConfig

SAMPLE_TEXT = "Alice was beginning to get very tired of sitting by her sister. " * 20
TINY = dict(model_id="t", hidden_dim=32, n_layers=1, n_heads=2,
            max_seq_length=32, batch_size=2, steps_per_round=2, learning_rate=1e-3)


def test_distillation_loss_shape():
    B, T, V = 2, 16, 260
    loss = distillation_loss(
        torch.randn(B, T, V),
        F.log_softmax(torch.randn(B, T, V), dim=-1),
        torch.randint(4, V, (B, T)),
    )
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0


def test_distillation_loss_alpha_zero():
    B, T, V = 2, 16, 260
    logits = torch.randn(B, T, V)
    targets = torch.randint(4, V, (B, T))
    loss0 = distillation_loss(logits, F.log_softmax(torch.randn(B, T, V), dim=-1), targets, alpha=0.0)
    ce = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=0)
    assert torch.allclose(loss0, ce, atol=1e-5)


def test_distillation_loss_gradient_flows():
    B, T, V = 2, 16, 260
    logits = torch.randn(B, T, V, requires_grad=True)
    loss = distillation_loss(logits, F.log_softmax(torch.randn(B, T, V), dim=-1),
                             torch.randint(4, V, (B, T)))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_vocab_marginalization():
    teacher = LocalTeacher(TeacherConfig(provider="local"))
    provider = TeacherLogitProvider(teacher, vocab_size=260)
    lp = provider.get_byte_level_probs(["Hello"], seq_length=8)
    assert lp.shape == (1, 8, 260)
    probs = torch.exp(lp[0])
    for t in range(8):
        assert abs(probs[t].sum().item() - 1.0) < 0.01


def test_distillation_kickstart_round():
    torch.manual_seed(42)
    cfg = KickstartConfig(**TINY)
    dc = DistillationConfig(teacher=TeacherConfig(provider="local"), alpha=0.5, temperature=2.0)
    ks = DistillationKickstart(cfg, dc)
    ks.load_text(SAMPLE_TEXT)
    ks.set_teacher(LocalTeacher(TeacherConfig(provider="local"), kickstart=ks))
    r = ks.train_round("r0", "p0")
    assert r.steps_completed > 0 and r.avg_loss < float("inf")


def test_distillation_kickstart_generates_gradients():
    torch.manual_seed(42)
    cfg = KickstartConfig(**TINY)
    dc = DistillationConfig(teacher=TeacherConfig(provider="local"))
    ks = DistillationKickstart(cfg, dc)
    ks.load_text(SAMPLE_TEXT)
    ks.set_teacher(LocalTeacher(TeacherConfig(provider="local"), kickstart=ks))
    r = ks.train_round("r0", "p0")
    assert r.gradients and len(r.gradients) > 0


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
