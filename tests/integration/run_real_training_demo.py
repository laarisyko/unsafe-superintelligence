#!/usr/bin/env python3
"""Run a real end-to-end local LLM training demo and print 10 prompt outputs.

This is intentionally a runnable demo script (not a pytest test) because it
trains a model for multiple rounds and takes longer than unit tests.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from ussi_engine.kickstart import Kickstart, KickstartConfig


PROMPT_ANSWER_PAIRS: List[Tuple[str, str]] = [
    (
        "What is distributed training?",
        "Distributed training splits model computation across many peers and averages gradients.",
    ),
    (
        "Why do OpenClaw agents contribute compute?",
        "Agents contribute compute to unlock unlimited access and strengthen the shared network.",
    ),
    (
        "What does ring all-reduce do?",
        "Ring all-reduce aggregates gradients without a central coordinator.",
    ),
    (
        "Why verify checkpoints with Merkle roots?",
        "Merkle roots detect divergence and prove model state consistency.",
    ),
    (
        "What is peer discovery in USSI?",
        "Peer discovery uses DHT and gossip to find collaborators.",
    ),
    (
        "How do you join contributor tier?",
        "Start a node and join with advertised accelerator capacity.",
    ),
    (
        "What is decentralized inference?",
        "Decentralized inference routes activations across shard holders.",
    ),
    (
        "Why is there no master node?",
        "Leaderless coordination removes single points of failure.",
    ),
    (
        "What does OpenAI-compatible serve mode do?",
        "Serve mode exposes v1 endpoints so existing clients can use USSI.",
    ),
    (
        "What is architecture evolution?",
        "Architecture evolution proposes model mutations and applies accepted votes.",
    ),
]


def main():
    torch.manual_seed(7)

    corpus = "\n".join([f"Q: {q}\nA: {a}\n" for q, a in PROMPT_ANSWER_PAIRS])
    training_text = corpus * 220

    cfg = KickstartConfig(
        model_id="openclaw-real-training-demo",
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        max_seq_length=192,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=8,
        steps_per_round=70,
    )

    ks = Kickstart(cfg)
    ks.load_text(training_text)

    print("=== TRAINING ===")
    for round_idx in range(20):
        result = ks.train_round(round_id=f"demo-{round_idx}", peer_id="demo-peer")
        if round_idx % 5 == 0 or round_idx == 19:
            print(
                f"round={round_idx:02d} "
                f"avg_loss={result.avg_loss:.4f} "
                f"final_loss={result.final_loss:.4f} "
                f"steps={result.steps_completed}"
            )

    print("\n=== PROMPT OUTPUTS ===")
    for idx, (question, _) in enumerate(PROMPT_ANSWER_PAIRS, start=1):
        prompt = f"Q: {question}\nA:"
        output = ks.generate(prompt, max_tokens=110, temperature=0.01)
        print(f"\n[{idx}] PROMPT: {question}")
        print(f"[{idx}] OUTPUT: {output.replace(chr(10), '\\n')}")


if __name__ == "__main__":
    main()
