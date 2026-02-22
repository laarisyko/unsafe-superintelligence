"""Tests for DPO (Direct Preference Optimization) from AI feedback."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn.functional as F

from ussi_engine.teacher import TeacherConfig, LocalTeacher
from ussi_engine.training.rl_from_ai import (
    DPOConfig, DPOKickstart, PreferenceDataCollector, PreferencePair, dpo_loss,
)
from ussi_engine.kickstart import Kickstart, KickstartConfig

SAMPLE_TEXT = "Alice was beginning to get very tired of sitting by her sister. " * 20
TINY = dict(model_id="t", hidden_dim=32, n_layers=1, n_heads=2,
            max_seq_length=32, batch_size=2, steps_per_round=2, learning_rate=1e-3)


def test_dpo_loss_correct_preference():
    loss_ok = dpo_loss(torch.tensor(-1.0), torch.tensor(-5.0),
                       torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    loss_bad = dpo_loss(torch.tensor(-5.0), torch.tensor(-1.0),
                        torch.tensor(-2.0), torch.tensor(-2.0), beta=0.1)
    assert loss_ok.item() < loss_bad.item()


def test_dpo_loss_scalar_finite():
    loss = dpo_loss(torch.tensor(-2.0), torch.tensor(-3.0),
                    torch.tensor(-2.5), torch.tensor(-2.5), beta=0.1)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_dpo_loss_equal_preferences():
    loss = dpo_loss(torch.tensor(-2.0), torch.tensor(-3.0),
                    torch.tensor(-2.0), torch.tensor(-3.0), beta=0.1)
    expected = -F.logsigmoid(torch.tensor(0.0))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_preference_collector_with_local():
    ks = Kickstart(KickstartConfig(**TINY))
    ks.load_text(SAMPLE_TEXT)
    teacher = LocalTeacher(TeacherConfig(provider="local"), kickstart=ks)
    pairs = PreferenceDataCollector(teacher, ks).collect_pairs(n=2)
    assert len(pairs) > 0
    assert all(isinstance(p, PreferencePair) and p.prompt and p.chosen and p.rejected for p in pairs)


def test_dpo_kickstart_round():
    torch.manual_seed(42)
    cfg = KickstartConfig(**TINY)
    dpo_cfg = DPOConfig(teacher=TeacherConfig(provider="local"), beta=0.1, pairs_per_round=3)
    ks = DPOKickstart(cfg, dpo_cfg)
    ks.load_text(SAMPLE_TEXT)
    ks.set_teacher(LocalTeacher(TeacherConfig(provider="local"), kickstart=ks))
    r = ks.train_dpo_round("r0", "p0")
    assert r.avg_loss < float("inf")


def test_reference_model_frozen():
    torch.manual_seed(42)
    cfg = KickstartConfig(**TINY)
    dpo_cfg = DPOConfig(teacher=TeacherConfig(provider="local"), beta=0.1, pairs_per_round=2)
    ks = DPOKickstart(cfg, dpo_cfg)
    ks.load_text(SAMPLE_TEXT)
    ks.set_teacher(LocalTeacher(TeacherConfig(provider="local"), kickstart=ks))
    ks._ensure_ref_model()
    snap = {n: p.clone() for n, p in ks._ref_model.named_parameters()}
    ks.train_dpo_round("r0", "p0")
    for n, p in ks._ref_model.named_parameters():
        assert torch.equal(p, snap[n]), f"Ref param '{n}' changed!"
    for n, p in ks._ref_model.named_parameters():
        assert not p.requires_grad


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
