"""Kickstart: bootstrap an LLM from scratch on a decentralized network.

This is the top-level entry point for "start from nothing." A peer that joins
the network for the first time runs this to:

    1. Create a fresh model (random weights, architecture from genome)
    2. Load local training data (any text files the peer has)
    3. Tokenize and prepare batches
    4. Run local training steps
    5. Collect gradients for decentralized aggregation

After aggregation, all peers converge to the same weights -- even though each
peer trained on different data and started with different random seeds. This
is the magic of averaging: E[gradient_i] converges to the true gradient.

Bootstrapping protocol:
    Round 0: All peers init from the same architecture genome (but random weights).
             After gradient averaging, weights converge to a shared starting point.
    Round 1+: Normal training. Each round, peers train on their local data,
              aggregate gradients, checkpoint, and repeat.

The model is intentionally small at first (e.g. 6 layers, 256 hidden dim, ~5M params).
As the network grows and the model improves, the architecture governance system
can propose mutations (add layers, increase width) and peers vote on them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .data.tokenizer import Tokenizer, TokenizerConfig
from .data.pipeline import TextDataPipeline, DataConfig
from .model.lm import LanguageModel, LMConfig, create_from_scratch

logger = logging.getLogger(__name__)


@dataclass
class KickstartConfig:
    """Configuration for bootstrapping a new LLM from scratch."""

    # Model.
    model_id: str = "ussi-v0"
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 4
    vocab_size: int = 260  # 256 bytes + 4 special tokens
    max_seq_length: int = 128
    dropout: float = 0.1

    # Training.
    learning_rate: float = 3e-4
    batch_size: int = 4
    steps_per_round: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data.
    data_paths: List[str] = field(default_factory=list)


@dataclass
class KickstartResult:
    """Result of a kickstart training round."""

    round_id: str
    steps_completed: int = 0
    avg_loss: float = 0.0
    final_loss: float = 0.0
    tokens_processed: int = 0
    time_ms: float = 0.0
    gradients: Optional[Dict[str, torch.Tensor]] = None
    sample_text: str = ""  # Generated sample to track progress
    skipped_steps: int = 0  # Steps skipped due to NaN/Inf
    reverted: bool = False  # Whether weights were reverted due to loss spike


class Kickstart:
    """Bootstrap an LLM from scratch.

    Usage:
        ks = Kickstart(config)
        ks.load_data("path/to/texts")     # or ks.load_text("raw text...")
        result = ks.train_round("round-0") # local training, returns gradients

        # After decentralized aggregation:
        ks.apply_aggregated_gradients(aggregated_grads)
        ks.checkpoint("round-0")

        # Generate sample text to track progress:
        print(ks.generate("The "))
    """

    def __init__(self, config: KickstartConfig):
        self.config = config

        # Tokenizer.
        self.tokenizer = Tokenizer(TokenizerConfig(
            mode="byte",
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_seq_length,
        ))

        # Model.
        lm_config = LMConfig(
            model_id=config.model_id,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
        )
        self.model = create_from_scratch(lm_config)

        # Optimizer.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Data pipeline.
        self.data = TextDataPipeline(
            tokenizer=self.tokenizer,
            config=DataConfig(
                seq_length=config.max_seq_length,
                batch_size=config.batch_size,
                batches_per_round=config.steps_per_round,
            ),
        )

        self.total_steps = 0
        self.round_count = 0

        logger.info(
            "Kickstart initialized: %s (%d params, %d layers, hidden=%d, vocab=%d)",
            config.model_id,
            self.model.num_parameters,
            config.n_layers,
            config.hidden_dim,
            config.vocab_size,
        )

    def load_text(self, text: str):
        """Load raw text into the training data pipeline."""
        self.data.load_text(text)

    def load_file(self, path: str):
        """Load a text file into the training data pipeline."""
        self.data.load_file(path)

    def load_directory(self, directory: str):
        """Load all text files from a directory."""
        self.data.load_directory(directory)

    def train_round(
        self,
        round_id: str = "",
        peer_id: str = "local",
    ) -> KickstartResult:
        """Run a local training round on the peer's data.

        This is one peer's contribution to a training round. The gradients
        returned should be sent to the decentralized aggregation system.

        Includes NaN/Inf detection and loss spike detection with revert.

        Returns:
            KickstartResult with gradients ready for aggregation.
        """
        if self.data.total_tokens < self.config.max_seq_length + 1:
            return KickstartResult(
                round_id=round_id,
                avg_loss=float("inf"),
            )

        start = time.monotonic()
        self.model.train()

        # Snapshot weights for loss spike detection.
        weight_snapshot = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        prev_round_loss = getattr(self, '_last_round_loss', None)

        losses = []
        tokens = 0
        skipped = 0

        for input_ids, target_ids, mask in self.data.iter_batches(round_id, peer_id):
            self.optimizer.zero_grad()

            logits, loss = self.model(input_ids, target_ids)
            loss_val = loss.item()

            # NaN/Inf detection on loss — skip step.
            if not (loss_val == loss_val) or loss_val == float("inf") or loss_val == float("-inf"):
                logger.warning("Round %s: NaN/Inf loss detected, skipping step", round_id)
                skipped += 1
                self.optimizer.zero_grad()
                continue

            loss.backward()

            # NaN/Inf detection on gradients — zero and skip.
            has_bad_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_bad_grad = True
                        break
            if has_bad_grad:
                logger.warning("Round %s: NaN/Inf gradients detected, skipping step", round_id)
                self.optimizer.zero_grad()
                skipped += 1
                continue

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()

            losses.append(loss_val)
            tokens += input_ids.numel()
            self.total_steps += 1

        elapsed = (time.monotonic() - start) * 1000
        self.round_count += 1

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        final_loss = losses[-1] if losses else float("inf")

        # Loss spike detection: revert if final loss > 5x previous round's loss.
        reverted = False
        if prev_round_loss is not None and prev_round_loss > 0:
            if final_loss > 5 * prev_round_loss:
                logger.warning(
                    "Round %s: loss spike detected (%.4f > 5x %.4f), reverting weights",
                    round_id, final_loss, prev_round_loss,
                )
                self.model.load_state_dict(weight_snapshot)
                reverted = True

        self._last_round_loss = avg_loss if not reverted else prev_round_loss

        # Collect gradients for aggregation.
        gradients = self._collect_gradients(round_id, peer_id)

        result = KickstartResult(
            round_id=round_id,
            steps_completed=len(losses),
            avg_loss=avg_loss,
            final_loss=final_loss,
            tokens_processed=tokens,
            time_ms=elapsed,
            gradients=gradients,
            skipped_steps=skipped,
            reverted=reverted,
        )

        # Generate a sample to track progress (may fail if model is corrupted).
        try:
            result.sample_text = self.generate("The ", max_tokens=30)
        except (RuntimeError, Exception) as e:
            result.sample_text = f"[generation failed: {e}]"
            logger.warning("Sample generation failed: %s", e)

        logger.info(
            "Round %s: %d steps (%d skipped), avg_loss=%.4f, %d tokens, %.0fms%s",
            round_id, result.steps_completed, skipped, avg_loss, tokens, elapsed,
            " [REVERTED]" if reverted else "",
        )

        return result

    def _collect_gradients(self, round_id: str, peer_id: str) -> Dict[str, torch.Tensor]:
        """Run one forward-backward pass to collect fresh gradients."""
        self.model.train()

        # Get one batch.
        batch_iter = self.data.iter_batches(round_id, peer_id)
        try:
            input_ids, target_ids, mask = next(batch_iter)
        except StopIteration:
            return {}

        self.optimizer.zero_grad()
        _, loss = self.model(input_ids, target_ids)
        loss.backward()

        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().detach()
        return grads

    def apply_aggregated_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Apply aggregated gradients from the decentralized network.

        After all peers submit their gradients and the coordinator aggregates
        them, each peer applies the result to converge on shared weights.

        Validates gradient shapes and skips mismatched or NaN/Inf gradients
        to prevent silent corruption from stale gradients after architecture mutation.
        """
        self.optimizer.zero_grad()
        applied = 0
        skipped = 0
        for name, param in self.model.named_parameters():
            if name in gradients:
                grad = gradients[name]
                # Shape validation.
                if grad.shape != param.shape:
                    logger.warning(
                        "Gradient shape mismatch for %s: grad=%s, param=%s — skipping",
                        name, grad.shape, param.shape,
                    )
                    skipped += 1
                    continue
                # NaN/Inf validation.
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    logger.warning(
                        "NaN/Inf in aggregated gradient for %s — skipping",
                        name,
                    )
                    skipped += 1
                    continue
                param.grad = grad.to(param.device)
                applied += 1

        if skipped > 0:
            logger.warning(
                "Applied %d gradients, skipped %d (shape/NaN mismatch)",
                applied, skipped,
            )
        self.optimizer.step()

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        """Generate text from a prompt."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(output_ids[0].tolist())

    def state_dict(self) -> Dict:
        """Get model state for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        """Load model state from checkpoint."""
        self.model.load_state_dict(state_dict)

    def generate_synthetic_data(self, teacher_config, n_samples: int = 50):
        """Generate synthetic training data using a SOTA teacher model.

        Args:
            teacher_config: TeacherConfig for the teacher model.
            n_samples: Number of synthetic samples to generate.
        """
        from .teacher import create_teacher, TeacherConfig
        from .data.synthetic import SyntheticDataGenerator, SyntheticConfig

        teacher = create_teacher(teacher_config, kickstart=self)
        synth_config = SyntheticConfig(teacher=teacher_config)
        generator = SyntheticDataGenerator(synth_config, teacher)
        fed = generator.feed_to_pipeline(self.data, n=n_samples)
        logger.info("Generated %d synthetic samples (%d total tokens)", fed, self.data.total_tokens)
        return fed

    def stats(self) -> Dict:
        return {
            "model_id": self.config.model_id,
            "parameters": self.model.num_parameters,
            "total_steps": self.total_steps,
            "rounds": self.round_count,
            "data_tokens": self.data.total_tokens,
            "data_sequences": self.data.total_sequences,
            "data_batches": self.data.total_batches,
        }
