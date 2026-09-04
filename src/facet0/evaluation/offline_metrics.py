"""Offline accuracy metrics for FACET action/wrench predictions.

Per FACET0_REPRODUCTION_PLAN.md Stage 4/5: these are pure, model-agnostic numpy functions
operating on already-computed prediction/ground-truth arrays -- they never load a checkpoint or
run inference themselves (see `scripts/benchmark_inference.py` for throughput/latency/memory,
which this module intentionally does not duplicate). "Offline error is not a substitute for
closed-loop success rate" (FACET0_REPRODUCTION_PLAN.md line 236) -- these numbers are a
regression signal during reproduction, not a claim about real-robot performance.

Action channel convention (matches `facet0.data.transforms`, i.e. after
`public_to_model_gripper`): ``[x, y, z, rx, ry, rz, gripper]``, translation = channels 0:3,
rotation = channels 3:6, gripper = channel 6.
"""

import dataclasses

import numpy as np

TRANSLATION_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 6)
GRIPPER_INDEX = 6


def _flatten_valid(values: np.ndarray, valid: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Flattens every leading (batch/horizon) dim. Returns (`values[N, channels]`, `valid[N]`)."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    if valid is None:
        return flat, np.ones(flat.shape[0], dtype=bool)
    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.shape != values.shape[:-1]:
        raise ValueError(f"valid shape {valid_array.shape} must match values' leading shape {values.shape[:-1]}")
    return flat, valid_array.reshape(-1)


def _masked_scalar_mean(values: np.ndarray, valid: np.ndarray | None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        return float(values.mean())
    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.shape != values.shape:
        raise ValueError(f"valid shape {valid_array.shape} must match values shape {values.shape}")
    if not np.any(valid_array):
        raise ValueError("no valid entries to compute a metric over")
    return float(values[valid_array].mean())


def per_channel_mae(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean absolute error per channel (last axis), averaged over every other axis."""
    pred_flat, valid_flat = _flatten_valid(pred, valid)
    target_flat, _ = _flatten_valid(target, valid)
    if not np.any(valid_flat):
        raise ValueError("no valid entries to compute a metric over")
    return np.abs(pred_flat - target_flat)[valid_flat].mean(axis=0)


def per_channel_rmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Root mean squared error per channel (last axis), averaged over every other axis."""
    pred_flat, valid_flat = _flatten_valid(pred, valid)
    target_flat, _ = _flatten_valid(target, valid)
    if not np.any(valid_flat):
        raise ValueError("no valid entries to compute a metric over")
    return np.sqrt(np.square(pred_flat - target_flat)[valid_flat].mean(axis=0))


def translation_error(pred_actions: np.ndarray, target_actions: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Mean per-timestep Euclidean distance between predicted and target `[x, y, z]`."""
    pred_xyz = np.asarray(pred_actions, dtype=np.float64)[..., TRANSLATION_SLICE]
    target_xyz = np.asarray(target_actions, dtype=np.float64)[..., TRANSLATION_SLICE]
    per_step = np.linalg.norm(pred_xyz - target_xyz, axis=-1)
    return _masked_scalar_mean(per_step, valid)


def rotation_error(pred_actions: np.ndarray, target_actions: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Mean per-timestep L2 distance between predicted and target `[rx, ry, rz]`.

    `experimental`: FACET0_REPRODUCTION_PLAN.md asks for a rotation error metric but does not
    specify the rotation representation or a geodesic distance; this treats the three rotation
    channels as a flat vector and reports Euclidean distance, which is only meaningful if
    `pred_actions`/`target_actions` are already in the same representation the checkpoint uses.
    """
    pred_rot = np.asarray(pred_actions, dtype=np.float64)[..., ROTATION_SLICE]
    target_rot = np.asarray(target_actions, dtype=np.float64)[..., ROTATION_SLICE]
    per_step = np.linalg.norm(pred_rot - target_rot, axis=-1)
    return _masked_scalar_mean(per_step, valid)


def gripper_accuracy(
    pred_actions: np.ndarray,
    target_actions: np.ndarray,
    *,
    threshold: float = 0.5,
    valid: np.ndarray | None = None,
) -> float:
    """Fraction of timesteps where thresholding pred/target gripper channels agree.

    `threshold` is `experimental`: the paper does not specify an open/closed decision boundary
    for offline evaluation.
    """
    pred_gripper = np.asarray(pred_actions, dtype=np.float64)[..., GRIPPER_INDEX]
    target_gripper = np.asarray(target_actions, dtype=np.float64)[..., GRIPPER_INDEX]
    matches = ((pred_gripper >= threshold) == (target_gripper >= threshold)).astype(np.float64)
    return _masked_scalar_mean(matches, valid)


def gripper_mae(pred_actions: np.ndarray, target_actions: np.ndarray, valid: np.ndarray | None = None) -> float:
    pred_gripper = np.asarray(pred_actions, dtype=np.float64)[..., GRIPPER_INDEX]
    target_gripper = np.asarray(target_actions, dtype=np.float64)[..., GRIPPER_INDEX]
    return _masked_scalar_mean(np.abs(pred_gripper - target_gripper), valid)


def action_chunk_smoothness(actions: np.ndarray, *, channels: slice = slice(0, 6)) -> float:
    """Mean L2 norm of second-order finite differences (discrete jerk) along the horizon axis
    (axis=-2), restricted to `channels` (translation+rotation by default -- gripper is excluded
    since a legitimate gripper profile can be a step function). Lower is smoother.

    Takes only a predicted chunk (no ground truth): this measures self-consistency of the
    prediction, not accuracy. If `actions` includes tail-padded (post-episode) steps, the
    padding's zero second-difference will bias this toward "smoother" -- pass only the valid
    prefix of the chunk to avoid that.
    """
    array = np.asarray(actions, dtype=np.float64)[..., channels]
    if array.shape[-2] < 3:
        raise ValueError("action_chunk_smoothness needs a horizon of at least 3 steps")
    second_diff = array[..., 2:, :] - 2 * array[..., 1:-1, :] + array[..., :-2, :]
    return float(np.linalg.norm(second_diff, axis=-1).mean())


def wrench_mae(pred_wrench: np.ndarray, target_wrench: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean absolute error per wrench channel `[fx, fy, fz, tx, ty, tz]`."""
    return per_channel_mae(pred_wrench, target_wrench, valid)


def wrench_peak_error(pred_wrench: np.ndarray, target_wrench: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Maximum absolute error per wrench channel."""
    pred_flat, valid_flat = _flatten_valid(pred_wrench, valid)
    target_flat, _ = _flatten_valid(target_wrench, valid)
    if not np.any(valid_flat):
        raise ValueError("no valid entries to compute a metric over")
    return np.abs(pred_flat - target_flat)[valid_flat].max(axis=0)


def wrench_exceedance_recall(
    pred_wrench: np.ndarray,
    target_wrench: np.ndarray,
    *,
    threshold: float,
    valid: np.ndarray | None = None,
) -> float:
    """Recall of "exceedance events": among (channel, timestep) entries where the target's
    magnitude exceeds `threshold`, the fraction where the prediction's magnitude also exceeds it.

    `threshold` is `experimental` and caller-supplied (e.g. a percentile of the training wrench
    magnitude distribution) -- the paper does not specify one for offline evaluation.
    """
    pred_flat, valid_flat = _flatten_valid(pred_wrench, valid)
    target_flat, _ = _flatten_valid(target_wrench, valid)
    pred_flat, target_flat = pred_flat[valid_flat], target_flat[valid_flat]
    target_exceeds = np.abs(target_flat) > threshold
    if not np.any(target_exceeds):
        raise ValueError("no exceedance events in target at the given threshold")
    pred_exceeds = np.abs(pred_flat) > threshold
    true_positive = np.logical_and(target_exceeds, pred_exceeds)
    return float(true_positive.sum() / target_exceeds.sum())


def wrench_constant_baseline(history: np.ndarray, history_valid: np.ndarray, horizon: int) -> np.ndarray:
    """The "hold current wrench constant" baseline from FACET0_REPRODUCTION_PLAN.md Stage 5's
    acceptance criterion (line 279): repeats the most recent valid history step forward for
    `horizon` steps.

    ``history``: float ``[..., k, wrench_dim]`` (e.g. `WrenchWindow.history`). ``history_valid``:
    bool ``[..., k]``. Returns float ``[..., horizon, wrench_dim]``.
    """
    history = np.asarray(history, dtype=np.float64)
    history_valid = np.asarray(history_valid, dtype=bool)
    if not np.all(history_valid.any(axis=-1)):
        raise ValueError("every history window must contain at least one valid step")
    last_valid_index = history_valid.shape[-1] - 1 - np.argmax(history_valid[..., ::-1], axis=-1)
    index_shape = (*last_valid_index.shape, 1, history.shape[-1])
    indices = np.broadcast_to(last_valid_index[..., None, None], index_shape)
    last_value = np.take_along_axis(history, indices, axis=-2)
    return np.repeat(last_value, horizon, axis=-2)


@dataclasses.dataclass(frozen=True)
class ActionMetrics:
    channel_mae: np.ndarray
    channel_rmse: np.ndarray
    translation_error_mean: float
    rotation_error_mean: float
    gripper_accuracy: float
    gripper_mae: float
    chunk_smoothness: float

    def to_dict(self) -> dict:
        return {
            "channel_mae": self.channel_mae.tolist(),
            "channel_rmse": self.channel_rmse.tolist(),
            "translation_error_mean": self.translation_error_mean,
            "rotation_error_mean": self.rotation_error_mean,
            "gripper_accuracy": self.gripper_accuracy,
            "gripper_mae": self.gripper_mae,
            "chunk_smoothness": self.chunk_smoothness,
        }


def compute_action_metrics(
    pred_actions: np.ndarray,
    target_actions: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    gripper_threshold: float = 0.5,
) -> ActionMetrics:
    return ActionMetrics(
        channel_mae=per_channel_mae(pred_actions, target_actions, valid),
        channel_rmse=per_channel_rmse(pred_actions, target_actions, valid),
        translation_error_mean=translation_error(pred_actions, target_actions, valid),
        rotation_error_mean=rotation_error(pred_actions, target_actions, valid),
        gripper_accuracy=gripper_accuracy(pred_actions, target_actions, threshold=gripper_threshold, valid=valid),
        gripper_mae=gripper_mae(pred_actions, target_actions, valid),
        chunk_smoothness=action_chunk_smoothness(pred_actions),
    )


@dataclasses.dataclass(frozen=True)
class WrenchMetrics:
    channel_mae: np.ndarray
    channel_peak_error: np.ndarray
    exceedance_recall: float | None

    def to_dict(self) -> dict:
        return {
            "channel_mae": self.channel_mae.tolist(),
            "channel_peak_error": self.channel_peak_error.tolist(),
            "exceedance_recall": self.exceedance_recall,
        }


def compute_wrench_metrics(
    pred_wrench: np.ndarray,
    target_wrench: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    exceedance_threshold: float | None = None,
) -> WrenchMetrics:
    exceedance = (
        wrench_exceedance_recall(pred_wrench, target_wrench, threshold=exceedance_threshold, valid=valid)
        if exceedance_threshold is not None
        else None
    )
    return WrenchMetrics(
        channel_mae=wrench_mae(pred_wrench, target_wrench, valid),
        channel_peak_error=wrench_peak_error(pred_wrench, target_wrench, valid),
        exceedance_recall=exceedance,
    )
