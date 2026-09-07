"""TD3+BC numerical targets for offline local-adaptation proxy experiments."""

from __future__ import annotations

import numpy as np


def td3_bootstrap_target(
    reward: np.ndarray,
    done: np.ndarray,
    target_q1: np.ndarray,
    target_q2: np.ndarray,
    *,
    discount: float,
) -> np.ndarray:
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    return np.asarray(reward) + discount * (1.0 - np.asarray(done)) * np.minimum(target_q1, target_q2)


def actor_td3_bc_loss(
    q_value: np.ndarray,
    predicted_action: np.ndarray,
    demonstrated_action: np.ndarray,
    *,
    bc_weight: float,
) -> float:
    bc = np.square(np.asarray(predicted_action) - np.asarray(demonstrated_action)).mean()
    return float(-np.asarray(q_value).mean() + bc_weight * bc)
