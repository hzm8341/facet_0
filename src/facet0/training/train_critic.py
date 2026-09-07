"""Numerical helpers for the experimental distributional critic."""

from __future__ import annotations

import numpy as np


def scalar_to_two_hot(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Linearly project scalar returns onto an ordered categorical support."""
    support = np.asarray(support, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if support.ndim != 1 or len(support) < 2 or not np.all(np.diff(support) > 0):
        raise ValueError("support must be a strictly increasing 1D array")
    clipped = np.clip(values, support[0], support[-1])
    upper = np.searchsorted(support, clipped, side="right")
    upper = np.clip(upper, 1, len(support) - 1)
    lower = upper - 1
    fraction = (clipped - support[lower]) / (support[upper] - support[lower])
    output = np.zeros((*values.shape, len(support)), dtype=np.float64)
    np.put_along_axis(output, lower[..., None], (1.0 - fraction)[..., None], axis=-1)
    np.put_along_axis(output, upper[..., None], fraction[..., None], axis=-1)
    return output


def expected_return(logits: np.ndarray, support: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return (probability * np.asarray(support)).sum(axis=-1)


def reward_proxy(
    *,
    success: np.ndarray,
    duration_fraction: np.ndarray,
    wrench_violation: np.ndarray,
    success_weight: float = 1.0,
    efficiency_weight: float = 0.1,
    violation_weight: float = 1.0,
) -> np.ndarray:
    """Configurable offline proxy; coefficients are experimental, not paper-specified."""
    return (
        success_weight * np.asarray(success)
        - efficiency_weight * np.asarray(duration_fraction)
        - violation_weight * np.asarray(wrench_violation)
    )
