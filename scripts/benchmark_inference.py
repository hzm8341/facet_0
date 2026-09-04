#!/usr/bin/env python3
"""Benchmark one persistent FACET-0 policy on reproducibly sampled public frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from facet0.data import ManuFacetDataset
from facet0.data.transforms import public_to_model_gripper
from facet0.policies import create_facet_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "Facet-0/facet0-post-training")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "ManuFacet-1K/Facet0-1")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/inference_benchmark_100.json")
    return parser.parse_args()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def sample_hash(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(sample["observation.state"]).view(np.uint8))
    for camera in sorted(sample["observation.images"]):
        digest.update(camera.encode())
        digest.update(np.ascontiguousarray(sample["observation.images"][camera]).view(np.uint8))
    digest.update(sample["prompt"].encode("utf-8"))
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 3)


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    warm = successful[1:] if len(successful) > 1 else successful
    wall = [record["wall_ms"] for record in warm]
    model = [record["model_infer_ms"] for record in warm]
    return {
        "requested_samples": len(records),
        "successful_samples": len(successful),
        "failed_samples": len(records) - len(successful),
        "all_outputs_finite": all(record.get("all_finite", False) for record in successful),
        "warm_sample_count": len(warm),
        "wall_ms": {
            "mean": round(float(np.mean(wall)), 3) if wall else None,
            "p50": percentile(wall, 50) if wall else None,
            "p95": percentile(wall, 95) if wall else None,
            "max": round(max(wall), 3) if wall else None,
        },
        "model_infer_ms": {
            "mean": round(float(np.mean(model)), 3) if model else None,
            "p50": percentile(model, 50) if model else None,
            "p95": percentile(model, 95) if model else None,
            "max": round(max(model), 3) if model else None,
        },
        "first_action_mae": {
            "translation": round(float(np.mean([r["translation_mae"] for r in successful])), 6)
            if successful
            else None,
            "rotation": round(float(np.mean([r["rotation_mae"] for r in successful])), 6)
            if successful
            else None,
            "gripper": round(float(np.mean([r["gripper_mae"] for r in successful])), 6)
            if successful
            else None,
        },
        "task_index_counts": dict(Counter(str(r["task_index"]) for r in successful)),
        "subtask_counts": dict(Counter(str(r["subtask"]) for r in successful)),
    }


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    dataset = ManuFacetDataset(args.dataset)
    rng = np.random.default_rng(args.seed)
    episode_ids = np.asarray(sorted(dataset.episodes))
    if args.num_samples <= len(episode_ids):
        chosen_episodes = rng.choice(episode_ids, size=args.num_samples, replace=False)
    else:
        chosen_episodes = rng.choice(episode_ids, size=args.num_samples, replace=True)

    restore_started = time.perf_counter()
    policy = create_facet_policy(args.checkpoint, seed=args.seed, num_steps=args.num_steps)
    restore_seconds = time.perf_counter() - restore_started

    records: list[dict[str, Any]] = []
    for sample_index, episode_index_raw in enumerate(chosen_episodes):
        episode_index = int(episode_index_raw)
        length = int(dataset.episodes[episode_index]["length"])
        frame_index = int(rng.integers(0, length))
        record: dict[str, Any] = {
            "sample_index": sample_index,
            "episode": episode_index,
            "frame": frame_index,
            "success": False,
        }
        try:
            raw = dataset.load_frame(episode_index, frame_index, include_images=True)
            demonstration_action = public_to_model_gripper(raw.pop("action"))
            demonstration_chunk, valid = dataset.load_action_chunk(episode_index, frame_index)
            demonstration_chunk = public_to_model_gripper(demonstration_chunk)
            noise = rng.standard_normal((50, 32)).astype(np.float32)
            input_digest = sample_hash(raw)

            started = time.perf_counter()
            result = policy.infer(raw, noise=noise)
            wall_ms = (time.perf_counter() - started) * 1000.0
            actions = np.asarray(result["actions"], dtype=np.float32)
            valid_predictions = actions[valid]
            valid_targets = demonstration_chunk[valid]
            chunk_mae = np.mean(np.abs(valid_predictions - valid_targets), axis=0)
            first_error = np.abs(actions[0] - demonstration_action)

            record.update(
                {
                    "success": actions.shape == (50, 7) and bool(np.isfinite(actions).all()),
                    "task_index": int(raw["task_index"]),
                    "subtask": int(raw["subtask"]),
                    "valid_horizon": int(valid.sum()),
                    "wall_ms": round(wall_ms, 3),
                    "model_infer_ms": round(float(result["policy_timing"]["infer_ms"]), 3),
                    "all_finite": bool(np.isfinite(actions).all()),
                    "action_min": float(actions.min()),
                    "action_max": float(actions.max()),
                    "translation_mae": float(first_error[:3].mean()),
                    "rotation_mae": float(first_error[3:6].mean()),
                    "gripper_mae": float(first_error[6]),
                    "chunk_mae_by_channel": chunk_mae.tolist(),
                    "input_sha256": input_digest,
                    "noise_sha256": array_hash(noise),
                    "output_sha256": array_hash(actions),
                    "first_action": actions[0].tolist(),
                }
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        print(
            f"[{sample_index + 1:03d}/{args.num_samples:03d}] "
            f"episode={episode_index} frame={frame_index} success={record['success']}"
        )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "num_steps": args.num_steps,
        "restore_seconds": round(restore_seconds, 6),
        "sampling": "uniform episodes without replacement; uniform frame within episode",
        "camera_mapping_status": "inferred_from_public_names",
        "summary": aggregate_metrics(records),
        "records": records,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0 if report["summary"]["failed_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
