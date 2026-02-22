"""Local training loop -- runs on each peer's model shard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ..model.shard import ModelShard

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for a training round."""

    learning_rate: float = 1e-4
    batch_size: int = 8
    num_steps: int = 100
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1


class LocalTrainer:
    """Runs the local training loop on a peer's model shard.

    This trainer handles:
    1. Forward pass on local layers
    2. Loss computation (if this is the last shard in the pipeline)
    3. Backward pass
    4. Gradient collection for decentralized aggregation
    """

    def __init__(self, shard: ModelShard, config: TrainingConfig):
        self.shard = shard
        self.config = config
        self.optimizer = self._build_optimizer()
        self.step_count = 0

    def _build_optimizer(self) -> optim.Optimizer:
        params = list(self.shard.parameters())
        if not params:
            # Return a dummy optimizer if the shard has no parameters.
            return optim.SGD([torch.zeros(1)], lr=self.config.learning_rate)

        if self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train_step(
        self,
        input_activations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            input_activations: Input tensor (from previous pipeline stage or data).
            target: Ground truth labels (only needed for the last pipeline stage).
            loss_fn: Loss function (only for the last pipeline stage).

        Returns:
            Dictionary with training metrics (loss, grad_norm, etc.).
        """
        self.shard.layers.train()
        input_activations = input_activations.to(self.shard.device)
        input_activations.requires_grad_(True)

        # Forward pass through local layers.
        output = self.shard.forward(input_activations)

        # Compute loss if this is the last stage.
        loss_value = 0.0
        if target is not None and loss_fn is not None:
            target = target.to(self.shard.device)
            loss = loss_fn(output, target)
            loss.backward()
            loss_value = loss.item()
        else:
            # For intermediate stages, the backward pass is driven by
            # gradients received from the next stage.
            output.sum().backward()

        # Gradient clipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.shard.parameters(), self.config.max_grad_norm
        )

        self.step_count += 1
        return {
            "loss": loss_value,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "step": self.step_count,
        }

    def apply_gradients(self):
        """Apply accumulated gradients via the optimizer."""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Collect current gradients from all shard parameters.

        Returns a dict mapping parameter names to gradient tensors.
        These are exchanged during decentralized all-reduce.
        """
        grads = {}
        for name, param in self.shard.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().detach()
        return grads

    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Replace local gradients with aggregated gradients from all-reduce."""
        for name, param in self.shard.named_parameters():
            if name in gradients:
                param.grad = gradients[name].to(self.shard.device)

    def train_epoch(
        self,
        data_iterator: Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        loss_fn: Optional[nn.Module] = None,
    ) -> List[Dict[str, float]]:
        """Run multiple training steps from a data iterator.

        Each item from the iterator is (input_activations, optional_target).
        """
        metrics = []
        for step, (inputs, targets) in enumerate(data_iterator):
            if step >= self.config.num_steps:
                break
            step_metrics = self.train_step(inputs, targets, loss_fn)
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.apply_gradients()
            metrics.append(step_metrics)

        return metrics
