#!/usr/bin/env python3
"""Audit every parquet file and expected video path in a ManuFacet release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from facet0.data import ManuFacetDataset
from facet0.data.manufacet import CAMERA_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "ManuFacet-1K/Facet0-1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/dataset_audit.json")
    args = parser.parse_args()

    dataset = ManuFacetDataset(args.dataset)
    errors: list[str] = []
    total_rows = 0
    task_frames: Counter[str] = Counter()
    subtask_frames: Counter[str] = Counter()
    state_min = np.full(13, np.inf)
    state_max = np.full(13, -np.inf)
    state_sum = np.zeros(13, dtype=np.float64)
    action_min = np.full(7, np.inf)
    action_max = np.full(7, -np.inf)
    action_sum = np.zeros(7, dtype=np.float64)
    nonfinite_state = 0
    nonfinite_action = 0
    missing_videos: list[str] = []
    video_bytes = 0

    for progress, episode_index in enumerate(sorted(dataset.episodes), start=1):
        episode = dataset.episodes[episode_index]
        parquet_path = dataset.parquet_path(episode_index)
        if not parquet_path.is_file():
            errors.append(f"missing parquet: {parquet_path}")
            continue
        try:
            table = pq.read_table(
                parquet_path,
                columns=["task_index", "subtask", "action", "observation.state"],
            )
            if table.num_rows != int(episode["length"]):
                errors.append(
                    f"episode {episode_index}: parquet rows {table.num_rows} != metadata length {episode['length']}"
                )
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if states.shape != (table.num_rows, 13):
                errors.append(f"episode {episode_index}: invalid state shape {states.shape}")
            if actions.shape != (table.num_rows, 7):
                errors.append(f"episode {episode_index}: invalid action shape {actions.shape}")
            state_finite = np.isfinite(states)
            action_finite = np.isfinite(actions)
            nonfinite_state += int((~state_finite).sum())
            nonfinite_action += int((~action_finite).sum())
            if state_finite.all():
                state_min = np.minimum(state_min, states.min(axis=0))
                state_max = np.maximum(state_max, states.max(axis=0))
                state_sum += states.sum(axis=0, dtype=np.float64)
            if action_finite.all():
                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))
                action_sum += actions.sum(axis=0, dtype=np.float64)
            task_frames.update(str(value) for value in table["task_index"].to_pylist())
            subtask_frames.update(str(value) for value in table["subtask"].to_pylist())
            total_rows += table.num_rows
        except Exception as error:
            errors.append(f"episode {episode_index}: {type(error).__name__}: {error}")

        for camera in CAMERA_KEYS:
            video_path = dataset.video_path(episode_index, camera)
            if not video_path.is_file():
                missing_videos.append(str(video_path))
            else:
                video_bytes += video_path.stat().st_size

        if progress % 250 == 0 or progress == len(dataset):
            print(f"audited {progress}/{len(dataset)} episodes")

    expected_rows = int(dataset.info["total_frames"])
    expected_videos = len(dataset) * len(CAMERA_KEYS)
    found_videos = expected_videos - len(missing_videos)
    if total_rows != expected_rows:
        errors.append(f"total rows {total_rows} != info.json total_frames {expected_rows}")
    if found_videos != int(dataset.info["total_videos"]):
        errors.append(f"found videos {found_videos} != info.json total_videos {dataset.info['total_videos']}")

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "summary": {
            "episodes": len(dataset),
            "frames": total_rows,
            "expected_frames": expected_rows,
            "videos_found": found_videos,
            "videos_expected": expected_videos,
            "video_bytes": video_bytes,
            "nonfinite_state_values": nonfinite_state,
            "nonfinite_action_values": nonfinite_action,
            "error_count": len(errors),
            "passed": not errors and not missing_videos,
        },
        "state": {
            "minimum": state_min.tolist(),
            "maximum": state_max.tolist(),
            "mean": (state_sum / total_rows).tolist(),
        },
        "action": {
            "minimum": action_min.tolist(),
            "maximum": action_max.tolist(),
            "mean": (action_sum / total_rows).tolist(),
        },
        "task_frame_counts": dict(task_frames),
        "subtask_frame_counts": dict(subtask_frames),
        "missing_videos": missing_videos,
        "errors": errors,
        "video_validation_scope": "existence and file size; frame-level decode is sampled separately",
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

