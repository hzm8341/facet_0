"""Minimal, traceable reader for the public ManuFacet-1K Facet0-1 release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERA_KEYS = ("view1", "hand", "view2")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decode_video_frame(path: Path, frame_index: int) -> np.ndarray:
    if frame_index < 0:
        raise IndexError("frame_index must be non-negative")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise IndexError(f"Frame {frame_index} is outside or unreadable in video {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class ManuFacetDataset:
    """Access episode metadata, parquet signals, prompts, and synchronized RGB frames."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Not a ManuFacet release directory: {self.root}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        tasks = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.episodes = {int(item["episode_index"]): item for item in episodes}
        self.tasks = {int(item["task_index"]): item["task"] for item in tasks}
        self.chunk_size = int(self.info["chunks_size"])

    def __len__(self) -> int:
        return len(self.episodes)

    def _chunk(self, episode_index: int) -> int:
        if episode_index not in self.episodes:
            raise IndexError(f"Unknown episode_index={episode_index}")
        return episode_index // self.chunk_size

    def parquet_path(self, episode_index: int) -> Path:
        chunk = self._chunk(episode_index)
        return self.root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"

    def video_path(self, episode_index: int, camera: str) -> Path:
        if camera not in CAMERA_KEYS:
            raise KeyError(f"Unknown camera {camera!r}; expected one of {CAMERA_KEYS}")
        chunk = self._chunk(episode_index)
        video_key = f"observation.images.{camera}"
        return self.root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    def load_frame(
        self,
        episode_index: int,
        frame_index: int,
        *,
        include_images: bool = True,
    ) -> dict[str, Any]:
        table = pq.read_table(self.parquet_path(episode_index))
        if not 0 <= frame_index < table.num_rows:
            raise IndexError(f"frame_index={frame_index} outside [0, {table.num_rows})")

        task_index = int(table["task_index"][frame_index].as_py())
        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": float(table["timestamp"][frame_index].as_py()),
            "task_index": task_index,
            "subtask": int(table["subtask"][frame_index].as_py()),
            "observation.state": np.asarray(
                table["observation.state"][frame_index].as_py(), dtype=np.float32
            ),
            "action": np.asarray(table["action"][frame_index].as_py(), dtype=np.float32),
            "prompt": self.tasks[task_index],
        }
        if include_images:
            sample["observation.images"] = {
                camera: _decode_video_frame(self.video_path(episode_index, camera), frame_index)
                for camera in CAMERA_KEYS
            }
        return sample

    def load_action_chunk(
        self, episode_index: int, frame_index: int, horizon: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        table = pq.read_table(self.parquet_path(episode_index), columns=["action"])
        if not 0 <= frame_index < table.num_rows:
            raise IndexError(f"frame_index={frame_index} outside [0, {table.num_rows})")
        stop = min(frame_index + horizon, table.num_rows)
        values = np.asarray(table["action"][frame_index:stop].to_pylist(), dtype=np.float32)
        valid = np.zeros(horizon, dtype=bool)
        valid[: len(values)] = True
        if len(values) < horizon:
            values = np.concatenate(
                [values, np.repeat(values[-1:], horizon - len(values), axis=0)], axis=0
            )
        return values, valid
