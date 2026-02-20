"""Language model wrapper: embeddings + transformer body + LM head.

The genome system defines the transformer body (attention + FFN layers).
This module wraps it with the parts needed for actual language modeling:

    Token IDs -> Embedding -> [Transformer Body] -> LM Head -> Logits -> Loss

For training from scratch:
    1. ArchitectureGenome.simple_transformer() defines the body
    2. LanguageModel wraps it with embedding + LM head
    3. Weights are randomly initialized (Xavier uniform)
    4. Peers train on their local data and aggregate gradients

The LM head ties weights with the embedding layer (standard practice since
GPT-2) to reduce parameter count.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..architecture.genome import ArchitectureGenome

logger = logging.getLogger(__name__)


@dataclass
class LMConfig:
    """Configuration for a from-scratch language model."""

    model_id: str = "openclaw-lm"
    vocab_size: int = 260       # 256 bytes + 4 special tokens
    hidden_dim: int = 256       # Transformer hidden dimension
    n_layers: int = 6           # Number of transformer layers
    n_heads: int = 4            # Attention heads
    ff_dim: int = 0             # FFN intermediate dim (0 = 4 * hidden_dim)
    max_seq_length: int = 512   # Maximum sequence length
    dropout: float = 0.1
    tie_weights: bool = True    # Tie embedding and LM head weights


class LanguageModel(nn.Module):
    """Complete language model for training from scratch.

    Architecture:
        token_ids (B, T)
            -> token_embedding (B, T, D) + position_embedding (B, T, D)
            -> dropout
            -> transformer_body (N layers of attention + FFN)
            -> layer_norm
            -> lm_head (B, T, V) -- logits over vocabulary

    The transformer body is compiled from an ArchitectureGenome, so the
    exact architecture is determined by the decentralized governance system.
    """

    def __init__(self, config: LMConfig, genome: Optional[ArchitectureGenome] = None):
        super().__init__()
        self.config = config

        ff_dim = config.ff_dim or config.hidden_dim * 4

        # Build genome if not provided.
        if genome is None:
            genome = ArchitectureGenome.simple_transformer(
                model_id=config.model_id,
                n_layers=config.n_layers,
                hidden_dim=config.hidden_dim,
                num_heads=config.n_heads,
                ff_dim=ff_dim,
            )
        self.genome = genome

        # Token + position embeddings.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer body: compiled from genome.
        # We build it as a ModuleList of transformer blocks (each block =
        # attention + norm + FFN + norm from the genome).
        self.layers = nn.ModuleList()
        self._build_transformer_body(genome, config)

        # Final layer norm.
        self.ln_f = nn.LayerNorm(config.hidden_dim)

        # LM head: project hidden states to vocabulary logits.
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights.
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights.
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "Initialized %s: %d layers, %d params, vocab=%d, hidden=%d",
            config.model_id, config.n_layers, n_params,
            config.vocab_size, config.hidden_dim,
        )

    def _build_transformer_body(self, genome: ArchitectureGenome, config: LMConfig):
        """Build transformer blocks from genome genes."""
        # Each "block" in the genome is: attention, norm, linear_up, activation, linear_down, norm
        # We wrap each block as a TransformerBlock for proper residual connections.
        genes_per_block = 6  # attention + norm + up + act + down + norm
        n_blocks = len(genome.genes) // genes_per_block

        for i in range(n_blocks):
            block = TransformerBlock(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim or config.hidden_dim * 4,
                dropout=config.dropout,
            )
            self.layers.append(block)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for training from scratch."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_length) token IDs.
            targets: (batch_size, seq_length) target token IDs for loss.

        Returns:
            (logits, loss) where loss is None if targets not provided.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings.
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embedding(input_ids)        # (B, T, D)
        pos_emb = self.position_embedding(positions)     # (1, T, D)
        x = self.embed_dropout(tok_emb + pos_emb)       # (B, T, D)

        # Transformer body.
        for layer in self.layers:
            x = layer(x)

        # Final norm + LM head.
        x = self.ln_f(x)                                # (B, T, D)
        logits = self.lm_head(x)                         # (B, T, V)

        # Compute loss if targets provided.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # Ignore PAD_TOKEN.
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: (1, T) starting token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = normal, <1 = greedy, >1 = diverse).

        Returns:
            (1, T + max_new_tokens) generated token IDs.
        """
        self.eval()
        max_len = self.config.max_seq_length

        for _ in range(max_new_tokens):
            # Crop to max sequence length.
            input_cropped = input_ids[:, -max_len:]

            with torch.no_grad():
                logits, _ = self.forward(input_cropped)

            # Take logits at the last position.
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm residual connections.

    Architecture (pre-norm, like GPT-2):
        x -> LayerNorm -> MultiHeadAttention -> + residual
          -> LayerNorm -> FFN (up + GELU + down) -> + residual
    """

    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Causal attention mask: prevent attending to future tokens.
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Pre-norm attention with residual.
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.attn_dropout(attn_out)

        # Pre-norm FFN with residual.
        x = x + self.ffn(self.ln2(x))

        return x


def create_from_scratch(config: LMConfig) -> LanguageModel:
    """Create a fresh language model ready for training from scratch.

    This is the kickstart entry point. Every peer calls this with the same
    config to get an identically-architectured (but randomly-initialized) model.
    After one round of gradient aggregation, all peers converge to the same weights.

    Args:
        config: LM configuration.

    Returns:
        Freshly initialized LanguageModel.
    """
    return LanguageModel(config)
