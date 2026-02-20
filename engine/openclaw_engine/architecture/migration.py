"""Weight migration -- transfer weights from old architecture to new one.

When the swarm accepts an architecture mutation, peers need to migrate their
weights from the old model to the new one. This module implements:
- Function-preserving transforms (Net2Net style) for widening
- Weight copying for matching layers
- Random initialization for newly added layers
- Gradient-free adaptation for changed dimensions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .genome import ArchitectureGenome, LayerGene, LayerType

logger = logging.getLogger(__name__)


class WeightMigrator:
    """Migrates weights from an old model to a new (mutated) architecture.

    The migration strategy depends on the type of mutation:
    - Added layers: initialize with identity-like weights (preserve function)
    - Removed layers: drop the weights (no migration needed)
    - Widened layers: copy existing weights + add noise for new neurons
    - Changed activations: keep surrounding weights as-is
    - Skip connections: no weight changes needed (additive)

    The goal is to preserve as much learned behavior as possible so the
    network doesn't have to retrain from scratch after every mutation.
    """

    def __init__(self, noise_scale: float = 0.01):
        """
        Args:
            noise_scale: Standard deviation of noise added to duplicated weights
                during widening. Small values preserve the function closely.
        """
        self.noise_scale = noise_scale

    def migrate(
        self,
        old_model: nn.Module,
        old_genome: ArchitectureGenome,
        new_genome: ArchitectureGenome,
    ) -> nn.Module:
        """Migrate weights from old_model to a new model built from new_genome.

        Returns the new model with migrated weights.
        """
        new_model = new_genome.compile()

        old_modules = _collect_param_modules(old_model)
        new_modules = _collect_param_modules(new_model)

        old_genes = old_genome.genes
        new_genes = new_genome.genes

        # Match genes by gene_id for layers that exist in both architectures.
        old_gene_map = {g.gene_id: (i, g) for i, g in enumerate(old_genes)}
        matched = 0
        widened = 0
        initialized = 0

        new_param_idx = 0
        for new_gene_idx, new_gene in enumerate(new_genes):
            if new_gene.gene_id in old_gene_map:
                old_idx, old_gene = old_gene_map[new_gene.gene_id]
                # Find the corresponding modules by position.
                if old_idx < len(old_modules) and new_param_idx < len(new_modules):
                    old_mod = old_modules[old_idx]
                    new_mod = new_modules[new_param_idx]
                    if _same_shape(old_mod, new_mod):
                        _copy_weights(old_mod, new_mod)
                        matched += 1
                    elif _is_widened(old_mod, new_mod):
                        _widen_weights(old_mod, new_mod, self.noise_scale)
                        widened += 1
                    else:
                        _init_identity(new_mod)
                        initialized += 1
            else:
                # New layer -- initialize to approximate identity.
                if new_param_idx < len(new_modules):
                    _init_identity(new_modules[new_param_idx])
                    initialized += 1

            # Only advance new_param_idx for genes that produce parameters.
            if new_gene.layer_type in (
                LayerType.LINEAR,
                LayerType.ATTENTION,
                LayerType.NORM,
                LayerType.FEEDFORWARD,
            ):
                new_param_idx += 1

        logger.info(
            "Weight migration: %d matched, %d widened, %d initialized",
            matched,
            widened,
            initialized,
        )
        return new_model

    def migrate_state_dict(
        self,
        old_state_dict: Dict[str, torch.Tensor],
        old_genome: ArchitectureGenome,
        new_genome: ArchitectureGenome,
    ) -> Dict[str, torch.Tensor]:
        """Migrate a state dict from old architecture to new.

        Uses a name-matching strategy: parameters with matching names are
        copied directly. Mismatched shapes trigger widening or truncation.
        """
        new_model = new_genome.compile()
        new_state = new_model.state_dict()

        migrated = {}
        for name, new_param in new_state.items():
            if name in old_state_dict:
                old_param = old_state_dict[name]
                if old_param.shape == new_param.shape:
                    migrated[name] = old_param.clone()
                else:
                    migrated[name] = _resize_tensor(
                        old_param, new_param.shape, self.noise_scale
                    )
            else:
                # New parameter -- use the randomly initialized value.
                migrated[name] = new_param.clone()

        return migrated


def _collect_param_modules(model: nn.Module) -> List[nn.Module]:
    """Collect modules that have parameters (Linear, Attention, Norm, etc.)."""
    modules = []
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm)):
            modules.append(mod)
    return modules


def _same_shape(old: nn.Module, new: nn.Module) -> bool:
    """Check if two modules have the same parameter shapes."""
    old_shapes = [p.shape for p in old.parameters()]
    new_shapes = [p.shape for p in new.parameters()]
    return old_shapes == new_shapes


def _is_widened(old: nn.Module, new: nn.Module) -> bool:
    """Check if the new module is a wider version of the old one."""
    if isinstance(old, nn.Linear) and isinstance(new, nn.Linear):
        return (
            new.out_features >= old.out_features
            and new.in_features >= old.in_features
        )
    return False


def _copy_weights(src: nn.Module, dst: nn.Module):
    """Copy weights from src to dst (same shape)."""
    dst.load_state_dict(src.state_dict())


def _widen_weights(old: nn.Module, new: nn.Module, noise_scale: float):
    """Net2Net-style widening: copy old weights and add noise for new neurons."""
    if isinstance(old, nn.Linear) and isinstance(new, nn.Linear):
        with torch.no_grad():
            # Copy the existing weight block.
            min_out = min(old.out_features, new.out_features)
            min_in = min(old.in_features, new.in_features)
            new.weight[:min_out, :min_in] = old.weight[:min_out, :min_in]

            # New output neurons: copy random existing ones + noise.
            if new.out_features > old.out_features:
                extra = new.out_features - old.out_features
                indices = torch.randint(0, old.out_features, (extra,))
                new.weight[old.out_features:, :min_in] = (
                    old.weight[indices, :min_in]
                    + torch.randn(extra, min_in) * noise_scale
                )

            # New input dimensions: zero-initialize (safe for function preservation).
            if new.in_features > old.in_features:
                new.weight[:, old.in_features:] = (
                    torch.randn(new.out_features, new.in_features - old.in_features)
                    * noise_scale
                )

            # Bias.
            if old.bias is not None and new.bias is not None:
                new.bias[:min_out] = old.bias[:min_out]
                if new.out_features > old.out_features:
                    new.bias[old.out_features:] = (
                        old.bias[indices] + torch.randn(extra) * noise_scale
                    )


def _init_identity(module: nn.Module):
    """Initialize a module to approximate the identity function.

    For Linear layers, this means setting the weight to a scaled identity
    matrix and bias to zero, so the layer passes through its input.
    """
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            nn.init.eye_(module.weight[:min(module.out_features, module.in_features),
                                       :min(module.out_features, module.in_features)])
            if module.weight.shape[0] > module.weight.shape[1]:
                module.weight[module.weight.shape[1]:, :] = 0
            if module.bias is not None:
                module.bias.zero_()
    elif isinstance(module, nn.LayerNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()


def _resize_tensor(
    old: torch.Tensor, new_shape: torch.Size, noise_scale: float
) -> torch.Tensor:
    """Resize a tensor to a new shape, preserving overlapping data.

    Noise is scaled relative to the existing tensor's std to prevent
    gradient explosion when the existing weights have a very different
    magnitude than the default noise_scale.
    """
    # Scale noise relative to existing tensor's std.
    tensor_std = old.std().item() if old.numel() > 1 else 1.0
    effective_noise = noise_scale * max(tensor_std, 1e-6)
    result = torch.randn(new_shape) * effective_noise

    # Compute the overlap region.
    slices = tuple(
        slice(0, min(o, n)) for o, n in zip(old.shape, new_shape)
    )
    result[slices] = old[slices]
    return result
