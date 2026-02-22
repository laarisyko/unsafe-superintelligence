"""Gradient compression to reduce bandwidth during all-reduce.

Supports Top-K sparsification and FP16 quantization, individually or chained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


class GradientCompressor(ABC):
    """Base class for gradient compressors."""

    @abstractmethod
    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        """Compress a gradient tensor.

        Returns (compressed_bytes, metadata_needed_for_decompression).
        """
        ...

    @abstractmethod
    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        """Decompress gradient bytes back into a tensor."""
        ...


class TopKCompressor(GradientCompressor):
    """Top-K sparsification: keep only the K largest gradient values.

    Reduces communication volume proportionally to the compression ratio.
    Error feedback (residuals) should be maintained by the caller across rounds.
    """

    def __init__(self, ratio: float = 0.01):
        """
        Args:
            ratio: Fraction of elements to keep (0.01 = top 1%).
        """
        self.ratio = ratio

    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        flat = tensor.flatten()
        k = max(1, int(flat.numel() * self.ratio))

        # Find top-k by absolute value.
        _, indices = torch.topk(flat.abs(), k)
        values = flat[indices]

        metadata = {
            "shape": list(tensor.shape),
            "numel": flat.numel(),
            "k": k,
            "dtype": str(tensor.dtype),
        }

        # Pack indices and values together.
        packed = torch.stack([indices.float(), values])
        return packed.numpy().tobytes(), metadata

    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        k = metadata["k"]
        numel = metadata["numel"]
        shape = metadata["shape"]

        packed = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(2, k)
        indices = packed[0].long()
        values = packed[1]

        result = torch.zeros(numel)
        result[indices] = values
        return result.reshape(shape)


class FP16Compressor(GradientCompressor):
    """FP16 quantization: cast gradients from FP32 to FP16 (50% reduction)."""

    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        fp16 = tensor.half()
        metadata = {
            "shape": list(tensor.shape),
            "original_dtype": str(tensor.dtype),
        }
        return fp16.numpy().tobytes(), metadata

    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        shape = metadata["shape"]
        tensor = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(shape)
        return tensor.float()


class CompressorChain(GradientCompressor):
    """Chain multiple compressors together (e.g. Top-K then FP16)."""

    def __init__(self, compressors: List[GradientCompressor]):
        self.compressors = compressors

    def compress(self, tensor: torch.Tensor) -> Tuple[bytes, dict]:
        all_metadata = []
        current_data = None
        current_tensor = tensor

        for i, comp in enumerate(self.compressors):
            data, meta = comp.compress(current_tensor)
            all_metadata.append(meta)
            if i < len(self.compressors) - 1:
                # Decompress to feed into next compressor.
                current_tensor = comp.decompress(data, meta)
            current_data = data

        return current_data, {"chain": all_metadata}

    def decompress(self, data: bytes, metadata: dict) -> torch.Tensor:
        # Decompress in reverse order using the last compressor.
        chain_meta = metadata["chain"]
        last_comp = self.compressors[-1]
        return last_comp.decompress(data, chain_meta[-1])


def compress_gradients(
    gradients: Dict[str, torch.Tensor],
    compressor: GradientCompressor,
) -> Tuple[Dict[str, bytes], Dict[str, dict]]:
    """Compress a dict of named gradient tensors."""
    compressed = {}
    metadata = {}
    for name, tensor in gradients.items():
        data, meta = compressor.compress(tensor)
        compressed[name] = data
        metadata[name] = meta
    return compressed, metadata


def decompress_gradients(
    compressed: Dict[str, bytes],
    metadata: Dict[str, dict],
    compressor: GradientCompressor,
) -> Dict[str, torch.Tensor]:
    """Decompress a dict of compressed gradient data back to tensors."""
    result = {}
    for name, data in compressed.items():
        result[name] = compressor.decompress(data, metadata[name])
    return result
