#!/usr/bin/env python3
"""Run the released FACET-0 checkpoint on one public ManuFacet frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facet0.data import ManuFacetDataset
from facet0.data.transforms import public_to_model_gripper
from facet0.policies import create_facet_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "Facet-0/facet0-post-training")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "ManuFacet-1K/Facet0-1")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="Use explicit deterministic 50x32 Gaussian noise for golden-output replay.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ManuFacetDataset(args.dataset)
    raw = dataset.load_frame(args.episode, args.frame, include_images=True)
    demonstration_action = public_to_model_gripper(raw.pop("action"))

    restore_started = time.perf_counter()
    policy = create_facet_policy(
        args.checkpoint,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    restore_seconds = time.perf_counter() - restore_started

    infer_started = time.perf_counter()
    noise = None
    if args.noise_seed is not None:
        noise = np.random.default_rng(args.noise_seed).standard_normal((50, 32)).astype(np.float32)
    result = policy.infer(raw, noise=noise)
    wall_seconds = time.perf_counter() - infer_started
    actions = np.asarray(result["actions"])

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "episode": args.episode,
        "frame": args.frame,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "noise_seed": args.noise_seed,
        "prompt": raw["prompt"],
        "demonstration_action": demonstration_action.tolist(),
        "actions": actions.tolist(),
        "summary": {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "all_finite": bool(np.isfinite(actions).all()),
            "minimum": float(actions.min()),
            "maximum": float(actions.max()),
            "restore_seconds": round(restore_seconds, 6),
            "inference_wall_seconds": round(wall_seconds, 6),
            "model_infer_ms": round(float(result["policy_timing"]["infer_ms"]), 3),
            "output_sha256": hashlib.sha256(np.ascontiguousarray(actions).view(np.uint8)).hexdigest(),
            "noise_sha256": (
                hashlib.sha256(np.ascontiguousarray(noise).view(np.uint8)).hexdigest()
                if noise is not None
                else None
            ),
        },
        "metadata": policy.metadata,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")

    success = actions.shape == (50, 7) and report["summary"]["all_finite"]
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
