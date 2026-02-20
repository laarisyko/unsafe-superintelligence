"""Byte-level BPE tokenizer for decentralized training.

A tokenizer that every peer can deterministically construct from the same
vocabulary file. No external dependencies (no sentencepiece, no HuggingFace).

Supports two modes:
    1. ByteLevel: every byte is a token (vocab_size=256). Zero training needed.
       Good enough for bootstrapping -- the model learns its own representations.
    2. BPE: byte-pair encoding built from a text corpus. Peers vote on the vocab
       via the architecture governance system, ensuring everyone uses the same tokens.

For kickstart (training from scratch), ByteLevel is the right choice:
    - No vocab training needed (deterministic, identical on every peer)
    - Works with any language / encoding
    - Model learns subword representations internally
    - GPT-2 used byte-level BPE for the same reason
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Special tokens.
PAD_TOKEN = 0
BOS_TOKEN = 1   # Beginning of sequence.
EOS_TOKEN = 2   # End of sequence.
UNK_TOKEN = 3   # Unknown (shouldn't occur in byte-level).
SPECIAL_TOKENS = 4  # First N IDs reserved for special tokens.


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    mode: str = "byte"  # "byte" or "bpe"
    vocab_size: int = 260  # 256 bytes + 4 special tokens
    max_sequence_length: int = 512
    bpe_merges: List[Tuple[int, int]] = field(default_factory=list)


class Tokenizer:
    """Deterministic tokenizer for decentralized LLM training.

    Every peer constructs the same tokenizer from the same config,
    ensuring identical token sequences for identical text.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._merge_map: Dict[Tuple[int, int], int] = {}

        if self.config.mode == "bpe" and self.config.bpe_merges:
            self._build_bpe(self.config.bpe_merges)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def max_length(self) -> int:
        return self.config.max_sequence_length

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_special: Whether to add BOS/EOS tokens.

        Returns:
            List of integer token IDs.
        """
        # Convert to bytes, offset by SPECIAL_TOKENS to avoid collision.
        raw_bytes = text.encode("utf-8", errors="replace")
        tokens = [b + SPECIAL_TOKENS for b in raw_bytes]

        # Apply BPE merges if configured.
        if self._merge_map:
            tokens = self._apply_bpe(tokens)

        # Add special tokens.
        if add_special:
            tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]

        # Truncate to max length.
        if len(tokens) > self.config.max_sequence_length:
            tokens = tokens[: self.config.max_sequence_length]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        # Strip special tokens.
        filtered = [t for t in tokens if t >= SPECIAL_TOKENS]

        # Reverse BPE if needed.
        if self._merge_map:
            filtered = self._reverse_bpe(filtered)

        # Convert back to bytes.
        raw_bytes = bytes(t - SPECIAL_TOKENS for t in filtered if SPECIAL_TOKENS <= t < SPECIAL_TOKENS + 256)
        return raw_bytes.decode("utf-8", errors="replace")

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        add_special: bool = True,
    ) -> Tuple[List[List[int]], List[int]]:
        """Encode a batch of texts with optional padding.

        Returns:
            Tuple of (padded_token_ids, lengths).
        """
        encoded = [self.encode(t, add_special=add_special) for t in texts]
        lengths = [len(e) for e in encoded]

        if padding:
            max_len = min(max(lengths), self.config.max_sequence_length)
            padded = []
            for seq in encoded:
                if len(seq) < max_len:
                    seq = seq + [PAD_TOKEN] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]
                padded.append(seq)
            return padded, lengths
        return encoded, lengths

    def _build_bpe(self, merges: List[Tuple[int, int]]):
        """Build BPE merge table."""
        next_id = self.config.vocab_size
        for a, b in merges:
            self._merge_map[(a, b)] = next_id
            next_id += 1
        self.config.vocab_size = next_id

    def _apply_bpe(self, tokens: List[int]) -> List[int]:
        """Apply BPE merges greedily."""
        while True:
            best_pair = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merge_map:
                    best_pair = pair
                    best_idx = i
                    break
            if best_pair is None:
                break
            merged_id = self._merge_map[best_pair]
            tokens = tokens[:best_idx] + [merged_id] + tokens[best_idx + 2:]
        return tokens

    def _reverse_bpe(self, tokens: List[int]) -> List[int]:
        """Reverse BPE merges for decoding."""
        # Build reverse map.
        reverse = {v: (a, b) for (a, b), v in self._merge_map.items()}
        result = []
        for t in tokens:
            if t in reverse:
                a, b = reverse[t]
                result.extend([a, b])
            else:
                result.append(t)
        return result

    def to_dict(self) -> Dict:
        return {
            "mode": self.config.mode,
            "vocab_size": self.config.vocab_size,
            "max_sequence_length": self.config.max_sequence_length,
            "bpe_merges": self.config.bpe_merges,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Tokenizer":
        config = TokenizerConfig(
            mode=d["mode"],
            vocab_size=d["vocab_size"],
            max_sequence_length=d.get("max_sequence_length", 512),
            bpe_merges=[tuple(m) for m in d.get("bpe_merges", [])],
        )
        return cls(config)

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Tokenizer":
        return cls.from_dict(json.loads(data))
