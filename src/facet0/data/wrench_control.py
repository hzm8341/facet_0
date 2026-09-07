"""In-memory data pool for deterministic future-wrench control experiments."""

from __future__ import annotations

import collections
import dataclasses

import numpy as np
import pyarrow.parquet as pq

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.transforms import public_to_model_gripper
from facet0.data.wrench_windows import WrenchNormalizer, causal_wrench_window, extract_wrench


@dataclasses.dataclass(frozen=True)
class WrenchControlPool:
    pairs: np.ndarray
    history: np.ndarray
    history_valid: np.ndarray
    actions: np.ndarray
    target: np.ndarray
    target_valid: np.ndarray
    change_score: np.ndarray

    def take(self, indices: np.ndarray) -> "WrenchControlPool":
        return dataclasses.replace(
            self,
            pairs=self.pairs[indices],
            history=self.history[indices],
            history_valid=self.history_valid[indices],
            actions=self.actions[indices],
            target=self.target[indices],
            target_valid=self.target_valid[indices],
            change_score=self.change_score[indices],
        )

    def __len__(self) -> int:
        return len(self.pairs)


def _padded_chunk(values: np.ndarray, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    stop = min(frame + horizon, len(values))
    chunk = values[frame:stop]
    valid = np.arange(horizon) < len(chunk)
    if len(chunk) < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)])
    return chunk, valid


def build_wrench_control_pool(
    dataset: ManuFacetDataset,
    pairs: list[tuple[int, int]],
    *,
    wrench_normalizer: WrenchNormalizer,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
    history_len: int = 10,
    horizon: int = 50,
) -> WrenchControlPool:
    """Build a pool while reading each distinct episode parquet only once."""
    grouped: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for output_index, (episode, frame) in enumerate(pairs):
        grouped[episode].append((output_index, frame))

    result: list[dict | None] = [None] * len(pairs)
    action_q01 = np.asarray(action_q01, dtype=np.float32)[:7]
    action_q99 = np.asarray(action_q99, dtype=np.float32)[:7]
    for episode, requests in grouped.items():
        table = pq.read_table(
            dataset.parquet_path(episode),
            columns=["observation.state", "action"],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        wrench = extract_wrench(states)
        for output_index, frame in requests:
            window = causal_wrench_window(
                wrench,
                frame,
                history_len=history_len,
                horizon=horizon,
            )
            action_chunk, action_valid = _padded_chunk(actions, frame, horizon)
            state_model = public_to_model_gripper(states[frame])
            action_model = public_to_model_gripper(action_chunk)
            action_model[..., :6] -= state_model[:6]
            action_normalized = (
                (action_model - action_q01) / (action_q99 - action_q01 + 1e-6) * 2.0 - 1.0
            ).astype(np.float32)
            target = wrench_normalizer.normalize_delta(window.future - window.history[-1])
            target_valid = np.logical_and(window.future_valid, action_valid)
            result[output_index] = {
                "pair": (episode, frame),
                "history": wrench_normalizer.normalize(window.history),
                "history_valid": window.history_valid,
                "actions": action_normalized,
                "target": target,
                "target_valid": target_valid,
                "change_score": float(np.abs(target[target_valid]).mean()),
            }

    examples = [item for item in result if item is not None]
    return WrenchControlPool(
        pairs=np.asarray([item["pair"] for item in examples], dtype=np.int32),
        history=np.stack([item["history"] for item in examples]),
        history_valid=np.stack([item["history_valid"] for item in examples]),
        actions=np.stack([item["actions"] for item in examples]),
        target=np.stack([item["target"] for item in examples]),
        target_valid=np.stack([item["target_valid"] for item in examples]),
        change_score=np.asarray([item["change_score"] for item in examples], dtype=np.float32),
    )


def stratify_by_change(
    candidates: WrenchControlPool,
    *,
    output_size: int,
    high_change_fraction: float,
    rng: np.random.Generator,
) -> WrenchControlPool:
    if not 0.0 <= high_change_fraction <= 1.0:
        raise ValueError("high_change_fraction must lie in [0, 1]")
    if output_size > len(candidates):
        raise ValueError("output_size cannot exceed the candidate pool")
    high_count = round(output_size * high_change_fraction)
    ranked = np.argsort(candidates.change_score)
    high = ranked[-high_count:] if high_count else np.empty(0, dtype=np.int64)
    remaining = np.setdiff1d(np.arange(len(candidates)), high, assume_unique=True)
    random_count = output_size - high_count
    random_indices = rng.choice(remaining, size=random_count, replace=False)
    selected = np.concatenate([high, random_indices])
    rng.shuffle(selected)
    return candidates.take(selected)
