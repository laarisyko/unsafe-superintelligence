"""Decentralized ring all-reduce for gradient aggregation.

No parameter server. Peers form a ring and exchange gradient slices until
every peer holds the fully aggregated result. The ring topology is determined
by the VRF so all peers agree on it without coordination.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class RingPeer:
    """A peer in the all-reduce ring."""

    peer_id: str
    ring_position: int
    # Callable to send a gradient chunk: send(chunk_bytes, target_peer_id) -> ack
    send_fn: Optional[Callable] = None
    # Callable to receive a gradient chunk: recv(source_peer_id) -> chunk_bytes
    recv_fn: Optional[Callable] = None


class RingAllReduce:
    """Implements decentralized ring all-reduce for gradient aggregation.

    The algorithm works in two phases:
    1. Scatter-reduce: Each peer sends 1/N of its gradients around the ring.
       After N-1 steps, each peer holds the sum of one slice from all peers.
    2. All-gather: Each peer sends its fully-reduced slice around the ring.
       After N-1 steps, every peer holds the complete aggregated gradient.

    For the decentralized network, send/recv are implemented as network calls
    to neighboring peers. For local testing, they operate on in-memory buffers.
    """

    def __init__(self, ring: List[RingPeer], local_position: int):
        self.ring = ring
        self.local_position = local_position
        self.n_peers = len(ring)

    @classmethod
    def local_ring(cls, n_peers: int) -> List["RingAllReduce"]:
        """Create N RingAllReduce instances for local (in-process) testing.

        Returns one instance per simulated peer. Use the class method
        `reduce_all()` to run all peers in lockstep, or call `reduce()`
        on individual instances for standalone use.
        """
        peers = [
            RingPeer(peer_id=f"peer-{i}", ring_position=i) for i in range(n_peers)
        ]
        instances = []
        for i in range(n_peers):
            instances.append(cls(ring=peers, local_position=i))
        return instances

    def reduce(self, local_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run all-reduce for a single peer (standalone, simple averaging).

        For correct ring all-reduce simulation across multiple peers, use
        the `reduce_all()` class method instead. This method falls back to
        simple local averaging for standalone use.
        """
        names = sorted(local_gradients.keys())
        # Standalone: just return the local gradients (no peers to aggregate with).
        return {n: local_gradients[n].clone() for n in names}

    @staticmethod
    def reduce_all(
        instances: List["RingAllReduce"],
        all_gradients: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Run ring all-reduce across all peers in lockstep (local simulation).

        This correctly simulates the ring all-reduce protocol by executing
        all peers synchronously at each step.

        Args:
            instances: List of RingAllReduce instances (one per peer).
            all_gradients: List of gradient dicts (one per peer).

        Returns:
            List of averaged gradient dicts (one per peer, all identical).
        """
        n = len(instances)
        assert len(all_gradients) == n

        names = sorted(all_gradients[0].keys())

        # Flatten each peer's gradients.
        all_flat = []
        shapes = None
        sizes = None
        for grads in all_gradients:
            flat, s, sz = _flatten(grads, names)
            all_flat.append(flat)
            if shapes is None:
                shapes = s
                sizes = sz

        # Split each peer's flat tensor into N chunks.
        all_chunks = []
        for flat in all_flat:
            chunks = list(flat.chunk(n))
            while len(chunks) < n:
                chunks.append(torch.zeros_like(chunks[0]))
            all_chunks.append(chunks)

        # Phase 1: Scatter-reduce.
        # At each step, peer i sends chunk[send_idx] to peer (i+1)%n
        # and receives from peer (i-1)%n into chunk[recv_idx].
        for step in range(n - 1):
            # Collect what each peer sends.
            sent = {}
            for i in range(n):
                send_idx = (i - step) % n
                sent[i] = all_chunks[i][send_idx].clone()

            # Each peer receives from its left neighbor and accumulates.
            for i in range(n):
                recv_idx = (i - step - 1) % n
                left = (i - 1) % n
                all_chunks[i][recv_idx] = all_chunks[i][recv_idx] + sent[left]

        # Phase 2: All-gather.
        # At each step, peer i sends its fully-reduced chunk to peer (i+1)%n.
        for step in range(n - 1):
            sent = {}
            for i in range(n):
                send_idx = (i - step + 1) % n
                sent[i] = all_chunks[i][send_idx].clone()

            for i in range(n):
                recv_idx = (i - step) % n
                left = (i - 1) % n
                all_chunks[i][recv_idx] = sent[left]

        # Reconstruct and average for each peer.
        results = []
        for i in range(n):
            aggregated = torch.cat(all_chunks[i][:n])[: all_flat[0].numel()]
            aggregated = aggregated / n
            results.append(_unflatten(aggregated, names, shapes, sizes))

        return results

    async def reduce_async(
        self,
        local_gradients: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Async version of ring all-reduce using network send/recv.

        Uses the send_fn and recv_fn of ring peers for actual network communication.
        """
        names = sorted(local_gradients.keys())
        flat, shapes, sizes = _flatten(local_gradients, names)
        chunks = list(flat.chunk(self.n_peers))

        left_peer = self.ring[(self.local_position - 1) % self.n_peers]
        right_peer = self.ring[(self.local_position + 1) % self.n_peers]

        # Phase 1: Scatter-reduce.
        for step in range(self.n_peers - 1):
            send_idx = (self.local_position - step) % self.n_peers
            recv_idx = (self.local_position - step - 1) % self.n_peers

            if right_peer.send_fn:
                send_data = chunks[send_idx].numpy().tobytes()
                await _maybe_await(right_peer.send_fn(send_data, right_peer.peer_id))

            if left_peer.recv_fn:
                recv_data = await _maybe_await(left_peer.recv_fn(left_peer.peer_id))
                if recv_data:
                    recv_tensor = torch.frombuffer(bytearray(recv_data), dtype=chunks[0].dtype)
                    chunks[recv_idx] = chunks[recv_idx] + recv_tensor

        # Phase 2: All-gather.
        for step in range(self.n_peers - 1):
            send_idx = (self.local_position - step + 1) % self.n_peers
            recv_idx = (self.local_position - step) % self.n_peers

            if right_peer.send_fn:
                send_data = chunks[send_idx].numpy().tobytes()
                await _maybe_await(right_peer.send_fn(send_data, right_peer.peer_id))

            if left_peer.recv_fn:
                recv_data = await _maybe_await(left_peer.recv_fn(left_peer.peer_id))
                if recv_data:
                    recv_tensor = torch.frombuffer(bytearray(recv_data), dtype=chunks[0].dtype)
                    chunks[recv_idx] = recv_tensor

        aggregated_flat = torch.cat(chunks[: self.n_peers])[:flat.numel()]
        aggregated_flat = aggregated_flat / self.n_peers

        return _unflatten(aggregated_flat, names, shapes, sizes)


def _flatten(
    grads: Dict[str, torch.Tensor], names: List[str]
) -> tuple[torch.Tensor, List[torch.Size], List[int]]:
    """Flatten a dict of named tensors into a single 1-D tensor."""
    tensors = [grads[n].flatten() for n in names]
    shapes = [grads[n].shape for n in names]
    sizes = [t.numel() for t in tensors]
    return torch.cat(tensors), shapes, sizes


def _unflatten(
    flat: torch.Tensor, names: List[str], shapes: List[torch.Size], sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """Reconstruct named tensors from a flattened 1-D tensor."""
    result = {}
    offset = 0
    for name, shape, size in zip(names, shapes, sizes):
        result[name] = flat[offset : offset + size].reshape(shape)
        offset += size
    return result


async def _maybe_await(val):
    """Await if coroutine, otherwise return directly."""
    if asyncio.iscoroutine(val):
        return await val
    return val
