"""Hardware-independent safety filter for shadow/replay testing."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    workspace_low: tuple[float, float, float]
    workspace_high: tuple[float, float, float]
    max_translation_step: float
    max_rotation_step: float
    wrench_abs_limit: tuple[float, float, float, float, float, float]
    command_timeout_seconds: float = 0.25


@dataclasses.dataclass(frozen=True)
class SafetyResult:
    command: np.ndarray
    stopped: bool
    clipped: bool
    reasons: tuple[str, ...]


def filter_command(
    current_action: np.ndarray,
    proposed_action: np.ndarray,
    measured_wrench: np.ndarray,
    *,
    command_age_seconds: float,
    config: SafetyConfig,
) -> SafetyResult:
    current = np.asarray(current_action, dtype=np.float64)
    proposed = np.asarray(proposed_action, dtype=np.float64)
    wrench = np.asarray(measured_wrench, dtype=np.float64)
    if current.shape != (7,) or proposed.shape != (7,) or wrench.shape != (6,):
        raise ValueError("expected current/proposed shape (7,) and wrench shape (6,)")
    if not np.isfinite(current).all() or not np.isfinite(proposed).all() or not np.isfinite(wrench).all():
        return SafetyResult(current.copy(), True, False, ("non_finite_input",))
    if command_age_seconds > config.command_timeout_seconds:
        return SafetyResult(current.copy(), True, False, ("watchdog_timeout",))
    if np.any(np.abs(wrench) > np.asarray(config.wrench_abs_limit)):
        return SafetyResult(current.copy(), True, False, ("wrench_limit",))

    command = proposed.copy()
    reasons = []
    xyz_clipped = np.clip(command[:3], config.workspace_low, config.workspace_high)
    if not np.array_equal(xyz_clipped, command[:3]):
        command[:3] = xyz_clipped
        reasons.append("workspace")
    translation_delta = command[:3] - current[:3]
    translation_norm = np.linalg.norm(translation_delta)
    if translation_norm > config.max_translation_step:
        command[:3] = current[:3] + translation_delta * config.max_translation_step / translation_norm
        reasons.append("translation_step")
    rotation_delta = command[3:6] - current[3:6]
    rotation_norm = np.linalg.norm(rotation_delta)
    if rotation_norm > config.max_rotation_step:
        command[3:6] = current[3:6] + rotation_delta * config.max_rotation_step / rotation_norm
        reasons.append("rotation_step")
    clipped_gripper = np.clip(command[6], 0.0, 1.0)
    if clipped_gripper != command[6]:
        command[6] = clipped_gripper
        reasons.append("gripper")
    return SafetyResult(command, False, bool(reasons), tuple(reasons))
