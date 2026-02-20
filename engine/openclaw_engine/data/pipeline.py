"""Decentralized data pipeline for LLM training from scratch.

In a decentralized network, training data is NOT centralized. Each peer
contributes whatever text data it has. The pipeline:

    1. Peer loads local text files (any UTF-8 text: books, code, Wikipedia, etc.)
    2. Tokenizes using the shared byte-level tokenizer (deterministic, no coordination)
    3. Chunks into fixed-length sequences for the model
    4. Creates (input, target) pairs for next-token prediction
    5. Deterministic batching seeded by (round_id, peer_id) for reproducibility

Data sharding:
    Each peer trains on its own data shard. The VRF assigns data partitions
    to peers, but for kickstart each peer just uses whatever it has locally.
    The gradient aggregation across peers implicitly averages over all data.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch

from .tokenizer import Tokenizer, TokenizerConfig, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for the training data pipeline."""

    # Sequence length for the model (context window).
    seq_length: int = 128

    # Batch size.
    batch_size: int = 4

    # Number of batches to produce per training round (-1 = all data).
    batches_per_round: int = 10

    # Whether to shuffle data. Seed = hash(round_id + peer_id).
    shuffle: bool = True

    # Data sources: list of file paths or directories.
    data_paths: List[str] = field(default_factory=list)


class TextDataPipeline:
    """Produces training batches for next-token prediction.

    This is the complete data path from raw text to training tensors:
        text files -> tokenize -> chunk -> (input, target) pairs -> batches

    Each batch yields:
        input_ids:  (batch_size, seq_length)     -- token IDs for input
        target_ids: (batch_size, seq_length)     -- shifted by 1 for next-token prediction
        mask:       (batch_size, seq_length)     -- 1 for real tokens, 0 for padding
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: DataConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self._token_buffer: List[int] = []

    def load_text(self, text: str):
        """Load raw text into the token buffer.

        Bypasses tokenizer truncation -- we want ALL tokens from the text.
        The pipeline handles its own chunking into seq_length sequences.
        """
        from .tokenizer import SPECIAL_TOKENS
        raw_bytes = text.encode("utf-8", errors="replace")
        tokens = [b + SPECIAL_TOKENS for b in raw_bytes]
        self._token_buffer.extend(tokens)
        logger.debug("Loaded %d tokens (buffer: %d)", len(tokens), len(self._token_buffer))

    def load_file(self, path: str):
        """Load a text file into the token buffer."""
        from .tokenizer import SPECIAL_TOKENS
        with open(path, "rb") as f:
            raw_bytes = f.read()
        tokens = [b + SPECIAL_TOKENS for b in raw_bytes]
        self._token_buffer.extend(tokens)
        logger.info("Loaded file %s (%d bytes, buffer: %d tokens)", path, len(raw_bytes), len(self._token_buffer))

    def load_directory(self, directory: str, extensions: tuple = (".txt", ".md", ".py", ".json")):
        """Recursively load all text files from a directory."""
        count = 0
        for root, _dirs, files in os.walk(directory):
            for fname in sorted(files):
                if fname.endswith(extensions):
                    self.load_file(os.path.join(root, fname))
                    count += 1
        logger.info("Loaded %d files from %s (buffer: %d tokens)", count, directory, len(self._token_buffer))

    def load_from_config(self):
        """Load data from all paths specified in config."""
        for path in self.config.data_paths:
            p = Path(path)
            if p.is_file():
                self.load_file(str(p))
            elif p.is_dir():
                self.load_directory(str(p))
            else:
                logger.warning("Data path not found: %s", path)

    @property
    def total_tokens(self) -> int:
        return len(self._token_buffer)

    @property
    def total_sequences(self) -> int:
        """Number of full sequences available."""
        if self.total_tokens < self.config.seq_length + 1:
            return 0
        return (self.total_tokens - 1) // self.config.seq_length

    @property
    def total_batches(self) -> int:
        """Number of full batches available."""
        if self.total_sequences == 0:
            return 0
        return self.total_sequences // self.config.batch_size

    def make_sequences(self, round_id: str = "", peer_id: str = "") -> List[Tuple[List[int], List[int]]]:
        """Create (input, target) sequence pairs for next-token prediction.

        Each pair:
            input  = tokens[i : i + seq_length]
            target = tokens[i + 1 : i + seq_length + 1]

        This is standard causal language modeling: predict the next token.
        """
        seq_len = self.config.seq_length
        sequences = []

        for start in range(0, len(self._token_buffer) - seq_len, seq_len):
            input_ids = self._token_buffer[start : start + seq_len]
            target_ids = self._token_buffer[start + 1 : start + seq_len + 1]
            sequences.append((input_ids, target_ids))

        # Deterministic shuffle if configured.
        if self.config.shuffle and sequences:
            seed = hashlib.sha256(f"{round_id}:{peer_id}".encode()).digest()
            seed_int = int.from_bytes(seed[:4], "little")
            # Fisher-Yates with deterministic seed.
            import random
            rng = random.Random(seed_int)
            rng.shuffle(sequences)

        return sequences

    def iter_batches(
        self,
        round_id: str = "",
        peer_id: str = "",
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over training batches.

        Yields:
            (input_ids, target_ids, attention_mask) tensors.
            All shapes: (batch_size, seq_length)
        """
        sequences = self.make_sequences(round_id, peer_id)
        batch_size = self.config.batch_size
        n_batches = self.config.batches_per_round

        batch_count = 0
        for batch_start in range(0, len(sequences), batch_size):
            if n_batches > 0 and batch_count >= n_batches:
                break

            batch_seqs = sequences[batch_start : batch_start + batch_size]
            if len(batch_seqs) < batch_size:
                break  # Skip incomplete batches.

            input_batch = torch.tensor([s[0] for s in batch_seqs], dtype=torch.long)
            target_batch = torch.tensor([s[1] for s in batch_seqs], dtype=torch.long)
            # Mask: 1 for real tokens, 0 for padding.
            mask = (input_batch != PAD_TOKEN).float()

            yield input_batch, target_batch, mask
            batch_count += 1

    def stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "total_batches": self.total_batches,
            "seq_length": self.config.seq_length,
            "batch_size": self.config.batch_size,
            "vocab_size": self.tokenizer.vocab_size,
        }
