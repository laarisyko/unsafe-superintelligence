"""Gradient wire protocol: serialization format for network transport.

Defines the binary format for sending gradient tensors between peers over
the P2P gossip or direct-stream channels. Supports optional compression
and integrity verification.

Wire format (v1):
    [4 bytes] magic: "OCGR" (USSI GRadient)
    [1 byte]  version: 0x01
    [1 byte]  flags: bit 0 = compressed, bit 1 = signed
    [32 bytes] merkle_root of the gradient data
    [4 bytes] n_params (number of named parameters)
    For each parameter:
        [2 bytes] name_len
        [name_len bytes] param name (UTF-8)
        [4 bytes] n_dims
        [4 * n_dims bytes] shape
        [4 bytes] data_len
        [data_len bytes] tensor data (float32 or compressed)
    [4 bytes] metadata_len
    [metadata_len bytes] JSON metadata (compression info, round_id, etc.)

The format is designed for:
    - Zero-copy reads where possible
    - Streaming: header + params can be read incrementally
    - Integrity: Merkle root covers all param data
    - Extensibility: metadata JSON for future fields
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .compression import (
    CompressorChain,
    FP16Compressor,
    GradientCompressor,
    TopKCompressor,
    compress_gradients,
    decompress_gradients,
)

logger = logging.getLogger(__name__)

# Wire format constants.
MAGIC = b"OCGR"
VERSION = 1
FLAG_COMPRESSED = 0x01
FLAG_SIGNED = 0x02


@dataclass
class WireMessage:
    """A gradient message ready for network transmission."""

    data: bytes
    merkle_root: str
    n_params: int
    compressed: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def size_kb(self) -> float:
        return len(self.data) / 1024


def encode(
    gradients: Dict[str, torch.Tensor],
    round_id: str = "",
    peer_id: str = "",
    compressor: Optional[GradientCompressor] = None,
) -> WireMessage:
    """Encode gradient tensors into wire format.

    Args:
        gradients: Named parameter gradients.
        round_id: Training round identifier.
        peer_id: Sending peer's identifier.
        compressor: Optional gradient compressor.

    Returns:
        WireMessage ready for network transmission.
    """
    buf = io.BytesIO()
    names = sorted(gradients.keys())

    # Compress if requested.
    compressed_data = None
    compression_meta = None
    if compressor is not None:
        compressed_data, compression_meta = compress_gradients(gradients, compressor)

    # Compute Merkle root over gradient data.
    leaf_hashes = []
    for name in names:
        h = hashlib.sha256()
        h.update(name.encode())
        if compressed_data is not None:
            h.update(compressed_data[name])
        else:
            h.update(gradients[name].detach().cpu().numpy().tobytes())
        leaf_hashes.append(h.digest())
    merkle_root = _merkle_root(leaf_hashes)

    # Write header.
    flags = 0
    if compressor is not None:
        flags |= FLAG_COMPRESSED

    buf.write(MAGIC)
    buf.write(struct.pack("<B", VERSION))
    buf.write(struct.pack("<B", flags))
    buf.write(merkle_root)
    buf.write(struct.pack("<I", len(names)))

    # Write each parameter.
    for name in names:
        name_bytes = name.encode("utf-8")
        buf.write(struct.pack("<H", len(name_bytes)))
        buf.write(name_bytes)

        tensor = gradients[name]
        buf.write(struct.pack("<I", len(tensor.shape)))
        for dim in tensor.shape:
            buf.write(struct.pack("<I", dim))

        if compressed_data is not None:
            param_bytes = compressed_data[name]
        else:
            param_bytes = tensor.detach().cpu().numpy().tobytes()

        buf.write(struct.pack("<I", len(param_bytes)))
        buf.write(param_bytes)

    # Write metadata.
    metadata = {
        "round_id": round_id,
        "peer_id": peer_id,
        "version": VERSION,
    }
    if compression_meta is not None:
        metadata["compression"] = compression_meta
    meta_bytes = json.dumps(metadata).encode("utf-8")
    buf.write(struct.pack("<I", len(meta_bytes)))
    buf.write(meta_bytes)

    return WireMessage(
        data=buf.getvalue(),
        merkle_root=merkle_root.hex(),
        n_params=len(names),
        compressed=compressor is not None,
        metadata=metadata,
    )


def decode(
    data: bytes,
    compressor: Optional[GradientCompressor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Decode wire format back into gradient tensors.

    Args:
        data: Raw wire bytes.
        compressor: Compressor to decompress (must match encoder).

    Returns:
        Tuple of (gradient_dict, metadata_dict).

    Raises:
        ValueError: If the wire format is invalid or integrity check fails.
    """
    buf = io.BytesIO(data)

    # Read header.
    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")

    version = struct.unpack("<B", buf.read(1))[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    flags = struct.unpack("<B", buf.read(1))[0]
    is_compressed = bool(flags & FLAG_COMPRESSED)

    expected_merkle = buf.read(32)
    n_params = struct.unpack("<I", buf.read(4))[0]

    # Read parameters.
    names = []
    raw_data = {}
    shapes = {}

    for _ in range(n_params):
        name_len = struct.unpack("<H", buf.read(2))[0]
        name = buf.read(name_len).decode("utf-8")
        names.append(name)

        n_dims = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(n_dims))
        shapes[name] = shape

        data_len = struct.unpack("<I", buf.read(4))[0]
        raw_data[name] = buf.read(data_len)

    # Read metadata.
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta_bytes = buf.read(meta_len)
    metadata = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}

    # Verify Merkle root.
    leaf_hashes = []
    for name in names:
        h = hashlib.sha256()
        h.update(name.encode())
        h.update(raw_data[name])
        leaf_hashes.append(h.digest())
    actual_merkle = _merkle_root(leaf_hashes)

    if actual_merkle != expected_merkle:
        raise ValueError(
            f"Merkle root mismatch: expected {expected_merkle.hex()[:16]}, "
            f"got {actual_merkle.hex()[:16]}"
        )

    # Decompress or deserialize.
    if is_compressed and compressor is not None:
        compression_meta = metadata.get("compression", {})
        gradients = decompress_gradients(raw_data, compression_meta, compressor)
    else:
        gradients = {}
        for name in names:
            tensor = torch.frombuffer(bytearray(raw_data[name]), dtype=torch.float32)
            gradients[name] = tensor.reshape(shapes[name]).clone()

    return gradients, metadata


def estimate_wire_size(
    gradients: Dict[str, torch.Tensor],
    compressor: Optional[GradientCompressor] = None,
) -> Dict[str, float]:
    """Estimate wire message size without actually encoding.

    Useful for bandwidth budgeting.
    """
    raw_bytes = sum(t.numel() * 4 for t in gradients.values())  # float32
    header_bytes = 4 + 1 + 1 + 32 + 4  # magic + version + flags + merkle + n_params
    name_bytes = sum(2 + len(name.encode()) for name in gradients.keys())
    shape_bytes = sum(4 + 4 * len(t.shape) + 4 for t in gradients.values())

    uncompressed = header_bytes + name_bytes + shape_bytes + raw_bytes

    compressed = uncompressed
    if compressor is not None:
        # Estimate compression ratio.
        if isinstance(compressor, TopKCompressor):
            compressed = int(uncompressed * compressor.ratio * 2)  # indices + values
        elif isinstance(compressor, FP16Compressor):
            compressed = int(uncompressed * 0.5)
        elif isinstance(compressor, CompressorChain):
            compressed = int(uncompressed * 0.01)  # Rough estimate for TopK+FP16

    return {
        "raw_bytes": raw_bytes,
        "uncompressed_wire_bytes": uncompressed,
        "compressed_wire_bytes": compressed,
        "compression_ratio": uncompressed / compressed if compressed > 0 else 1.0,
        "n_params": len(gradients),
        "total_elements": sum(t.numel() for t in gradients.values()),
    }


def _merkle_root(leaves: List[bytes]) -> bytes:
    """Compute binary Merkle tree root."""
    if not leaves:
        return b"\x00" * 32
    layer = list(leaves)
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                combined = hashlib.sha256(b"\x01" + layer[i] + layer[i + 1]).digest()
            else:
                combined = layer[i]
            next_layer.append(combined)
        layer = next_layer
    return layer[0]
