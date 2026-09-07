"""Causal wrench history and future wrench targets for action-wrench alignment.

FACET's 13-dim state is ``[x,y,z,rx,ry,rz,gripper,fx,fy,fz,tx,ty,tz]``
(see ``facet0.data.transforms.STATE_DIM``); the wrench channel is the last six.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow.parquet as pq

from facet0.data.manufacet import ManuFacetDataset


WRENCH_DIM = 6
_STATE_WRENCH_SLICE = slice(7, 13)


@dataclasses.dataclass(frozen=True)
class WrenchNormalizer:
    """Quantile normalization for the six FACET wrist force/torque channels."""

    q01: np.ndarray
    q99: np.ndarray

    @classmethod
    def from_openpi_norm_stats(cls, norm_stats: dict) -> "WrenchNormalizer":
        state_stats = norm_stats["state"]
        q01 = np.asarray(state_stats.q01, dtype=np.float32)[_STATE_WRENCH_SLICE]
        q99 = np.asarray(state_stats.q99, dtype=np.float32)[_STATE_WRENCH_SLICE]
        if q01.shape != (WRENCH_DIM,) or q99.shape != (WRENCH_DIM,):
            raise ValueError("Checkpoint state statistics do not contain six wrench channels at [7:13].")
        if np.any(q99 <= q01):
            raise ValueError("Every wrench q99 value must be greater than q01.")
        return cls(q01=q01, q99=q99)

    def normalize(self, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=np.float32)
        return ((wrench - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0).astype(
            np.float32
        )

    def unnormalize(self, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=np.float32)
        return (((wrench + 1.0) * 0.5) * (self.q99 - self.q01) + self.q01).astype(
            np.float32
        )

    def normalize_delta(self, delta: np.ndarray) -> np.ndarray:
        """Scale a wrench difference without applying the absolute-value offset."""
        delta = np.asarray(delta, dtype=np.float32)
        return (delta * 2.0 / (self.q99 - self.q01 + 1e-6)).astype(np.float32)

    def unnormalize_delta(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32)
        return (delta * 0.5 * (self.q99 - self.q01)).astype(np.float32)


def extract_wrench(state: np.ndarray) -> np.ndarray:
    """Slice the 6-dim wrench ``[fx, fy, fz, tx, ty, tz]`` out of a 13-dim FACET state."""
    array = np.asarray(state, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] < 13:
        raise ValueError(f"state must end in dimension >= 13, got {array.shape}")
    return array[..., _STATE_WRENCH_SLICE]


@dataclasses.dataclass(frozen=True)
class WrenchWindow:
    """A causal wrench history paired with a future wrench alignment target.

    ``history`` covers frames ``[frame_index - history_len + 1, frame_index]`` and is
    safe to feed to the model as observation context: every valid entry has a frame
    index <= frame_index. ``future`` covers frames ``[frame_index, frame_index + horizon)``,
    mirroring ``ManuFacetDataset.load_action_chunk``'s window so it lines up 1:1 with the
    action chunk target -- it is a training label, never model input.

    Padding repeats the nearest valid frame rather than zero-filling, matching
    ``load_action_chunk``'s tail-padding convention, so a consumer that ignores the
    ``*_valid`` masks still sees a physically plausible (if stale) wrench value.
    """

    history: np.ndarray
    history_valid: np.ndarray
    history_frame_indices: np.ndarray
    future: np.ndarray
    future_valid: np.ndarray
    future_frame_indices: np.ndarray


def has_future_leakage(window: WrenchWindow, frame_index: int) -> bool:
    """True if any valid history entry references a frame after ``frame_index``."""
    valid_indices = window.history_frame_indices[window.history_valid]
    return bool(valid_indices.size) and bool(valid_indices.max() > frame_index)


def causal_wrench_window(
    wrench: np.ndarray,
    frame_index: int,
    *,
    history_len: int = 10,
    horizon: int = 50,
) -> WrenchWindow:
    """Build a causal history window and an aligned future target for one episode.

    ``wrench`` holds the wrench channel for every frame of one episode, in chronological
    order, shape ``[T, 6]``.
    """
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    array = np.asarray(wrench, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != WRENCH_DIM:
        raise ValueError(f"wrench must have shape [T, {WRENCH_DIM}], got {array.shape}")
    num_frames = array.shape[0]
    if not 0 <= frame_index < num_frames:
        raise IndexError(f"frame_index={frame_index} outside [0, {num_frames})")

    history_indices = np.arange(frame_index - history_len + 1, frame_index + 1)
    history_valid = history_indices >= 0
    history = array[np.clip(history_indices, 0, num_frames - 1)]
    history_frame_indices = np.where(history_valid, history_indices, -1)

    future_indices = np.arange(frame_index, frame_index + horizon)
    future_valid = future_indices < num_frames
    future = array[np.clip(future_indices, 0, num_frames - 1)]
    future_frame_indices = np.where(future_valid, future_indices, -1)

    window = WrenchWindow(
        history=history,
        history_valid=history_valid,
        history_frame_indices=history_frame_indices,
        future=future,
        future_valid=future_valid,
        future_frame_indices=future_frame_indices,
    )
    if has_future_leakage(window, frame_index):
        raise AssertionError("causal_wrench_window produced a leaking history window")
    return window


def load_episode_wrench(dataset: ManuFacetDataset, episode_index: int) -> np.ndarray:
    """Read the full per-frame wrench channel for one episode, in chronological order."""
    table = pq.read_table(dataset.parquet_path(episode_index), columns=["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    return extract_wrench(states)


def build_wrench_window(
    dataset: ManuFacetDataset,
    episode_index: int,
    frame_index: int,
    *,
    history_len: int = 10,
    horizon: int = 50,
) -> WrenchWindow:
    """Causal wrench window for one ``(episode, frame)`` pair of a ManuFacet dataset."""
    wrench = load_episode_wrench(dataset, episode_index)
    return causal_wrench_window(wrench, frame_index, history_len=history_len, horizon=horizon)
