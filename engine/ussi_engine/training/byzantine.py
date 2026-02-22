"""Byzantine-resilient gradient aggregation.

In an open decentralized network, malicious peers can submit poisoned gradients
to corrupt the model. This module implements robust aggregation methods that
tolerate up to f < n/2 Byzantine peers.

Supported methods:
    - Krum: Selects the gradient closest to its neighbors (Blanchard et al., 2017)
    - Multi-Krum: Selects top-m closest gradients and averages them
    - Coordinate-wise Trimmed Mean: Trims extreme values per coordinate
    - Coordinate-wise Median: Takes the median per coordinate
    - Combined: Krum selection followed by trimmed mean on survivors

All methods plug into the existing all-reduce pipeline as a post-aggregation
filter that replaces simple averaging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Available Byzantine-resilient aggregation methods."""
    MEAN = auto()           # Simple average (no protection)
    KRUM = auto()           # Select single best gradient
    MULTI_KRUM = auto()     # Average top-m gradients by Krum score
    TRIMMED_MEAN = auto()   # Coordinate-wise trimmed mean
    MEDIAN = auto()         # Coordinate-wise median
    BULYAN = auto()          # Multi-Krum selection + trimmed mean on survivors


@dataclass
class ByzantineConfig:
    """Configuration for Byzantine-resilient aggregation."""

    method: AggregationMethod = AggregationMethod.TRIMMED_MEAN

    # Maximum number of Byzantine peers to tolerate.
    # Must satisfy: n >= 2*f + 3 for Krum, n >= 2*f + 1 for trimmed mean.
    max_byzantine: int = 0

    # Fraction of values to trim from each end (for TRIMMED_MEAN).
    # Default 0.1 = trim top and bottom 10%. Auto-computed from max_byzantine if 0.
    trim_ratio: float = 0.0

    # Number of gradients to select in Multi-Krum (0 = auto).
    multi_krum_m: int = 0


def robust_aggregate(
    gradients: List[Dict[str, torch.Tensor]],
    config: ByzantineConfig,
) -> Dict[str, torch.Tensor]:
    """Aggregate gradients using a Byzantine-resilient method.

    Args:
        gradients: List of gradient dicts, one per peer.
        config: Byzantine resilience configuration.

    Returns:
        Single aggregated gradient dict.
    """
    n = len(gradients)
    if n == 0:
        raise ValueError("No gradients to aggregate")
    if n == 1:
        return gradients[0]

    f = config.max_byzantine
    if f == 0:
        # Auto-detect: tolerate up to 20% Byzantine, minimum 1.
        f = max(1, n // 5)

    method = config.method

    if method == AggregationMethod.MEAN:
        return _simple_mean(gradients)
    elif method == AggregationMethod.KRUM:
        return _krum(gradients, f, multi_k=1)
    elif method == AggregationMethod.MULTI_KRUM:
        m = config.multi_krum_m or max(1, n - f)
        return _krum(gradients, f, multi_k=m)
    elif method == AggregationMethod.TRIMMED_MEAN:
        trim = config.trim_ratio or (f / n if n > 0 else 0.1)
        return _trimmed_mean(gradients, trim)
    elif method == AggregationMethod.MEDIAN:
        return _coordinate_median(gradients)
    elif method == AggregationMethod.BULYAN:
        m = max(1, n - 2 * f)
        survivors = _krum_select(gradients, f, m)
        trim = config.trim_ratio or 0.25
        return _trimmed_mean(survivors, trim)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def score_gradients(
    gradients: List[Dict[str, torch.Tensor]],
    max_byzantine: int = 0,
) -> List[float]:
    """Score each gradient by its Krum distance (lower = more trustworthy).

    Useful for reputation tracking: peers that consistently submit high-score
    (outlier) gradients are likely Byzantine.

    Returns:
        List of scores, one per gradient. Lower is better.
    """
    n = len(gradients)
    f = max_byzantine or max(1, n // 5)

    flat = _flatten_all(gradients)
    distances = _pairwise_distances(flat)

    scores = []
    n_neighbors = max(1, n - f - 2)
    for i in range(n):
        dists = [(distances[i][j], j) for j in range(n) if j != i]
        dists.sort()
        score = sum(d for d, _ in dists[:n_neighbors])
        scores.append(score)

    return scores


# === Implementation ===


def _simple_mean(gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Plain average (no Byzantine protection)."""
    n = len(gradients)
    names = sorted(gradients[0].keys())
    result = {}
    for name in names:
        stacked = torch.stack([g[name] for g in gradients])
        result[name] = stacked.mean(dim=0)
    return result


def _krum(
    gradients: List[Dict[str, torch.Tensor]],
    f: int,
    multi_k: int = 1,
) -> Dict[str, torch.Tensor]:
    """Krum / Multi-Krum aggregation.

    For each gradient, compute the sum of distances to its n-f-2 nearest
    neighbors. Select the gradient(s) with the smallest score.

    Krum (multi_k=1): Return the single best gradient.
    Multi-Krum (multi_k=m): Average the top-m gradients.
    """
    selected = _krum_select(gradients, f, multi_k)
    return _simple_mean(selected)


def _krum_select(
    gradients: List[Dict[str, torch.Tensor]],
    f: int,
    m: int,
) -> List[Dict[str, torch.Tensor]]:
    """Select top-m gradients by Krum score."""
    n = len(gradients)
    if n <= 2 * f + 2:
        logger.warning(
            "Krum requires n >= 2f+3, got n=%d, f=%d. Falling back to mean.", n, f
        )
        return gradients

    flat = _flatten_all(gradients)
    distances = _pairwise_distances(flat)

    # For each peer, sum distances to n-f-2 nearest neighbors.
    n_neighbors = n - f - 2
    scores = []
    for i in range(n):
        dists = sorted(distances[i][j] for j in range(n) if j != i)
        score = sum(dists[:n_neighbors])
        scores.append((score, i))

    scores.sort()
    selected_indices = [idx for _, idx in scores[:m]]

    logger.debug(
        "Krum selected %d/%d peers: %s (scores: %.2f - %.2f)",
        m, n, selected_indices, scores[0][0], scores[-1][0],
    )

    return [gradients[i] for i in selected_indices]


def _trimmed_mean(
    gradients: List[Dict[str, torch.Tensor]],
    trim_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean.

    For each coordinate (weight value), sort the n values, discard the
    top and bottom trim_ratio fraction, and average the rest.
    """
    n = len(gradients)
    trim_count = max(1, int(n * trim_ratio))
    keep_count = n - 2 * trim_count

    if keep_count < 1:
        # Can't trim that much; fall back to median.
        return _coordinate_median(gradients)

    names = sorted(gradients[0].keys())
    result = {}

    for name in names:
        stacked = torch.stack([g[name] for g in gradients])  # (n, *shape)
        # Flatten the parameter dimensions, sort along peer axis.
        flat = stacked.reshape(n, -1)
        sorted_vals, _ = flat.sort(dim=0)
        # Trim top and bottom.
        trimmed = sorted_vals[trim_count : n - trim_count]
        result[name] = trimmed.mean(dim=0).reshape(gradients[0][name].shape)

    return result


def _coordinate_median(
    gradients: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median aggregation."""
    names = sorted(gradients[0].keys())
    n = len(gradients)
    result = {}

    for name in names:
        stacked = torch.stack([g[name] for g in gradients])
        flat = stacked.reshape(n, -1)
        result[name] = flat.median(dim=0).values.reshape(gradients[0][name].shape)

    return result


def _flatten_all(gradients: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
    """Flatten each gradient dict into a single 1-D tensor."""
    names = sorted(gradients[0].keys())
    flat = []
    for g in gradients:
        tensors = [g[n].flatten() for n in names]
        flat.append(torch.cat(tensors))
    return flat


def _pairwise_distances(flat: List[torch.Tensor]) -> List[List[float]]:
    """Compute pairwise L2 distances between flattened gradient vectors."""
    n = len(flat)
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (flat[i] - flat[j]).norm().item()
            distances[i][j] = d
            distances[j][i] = d
    return distances
