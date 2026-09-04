#!/usr/bin/env python3
"""Compare two FACET inference reports using channel-aware numeric tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_ATOL = np.asarray([5e-4, 5e-4, 5e-4, 4e-3, 4e-3, 4e-3, 2e-3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference_report = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = np.asarray(reference_report["actions"], dtype=np.float64)
    candidate = np.asarray(candidate_report["actions"], dtype=np.float64)
    if reference.shape != (50, 7) or candidate.shape != (50, 7):
        print(f"FAIL: expected two (50, 7) arrays, got {reference.shape} and {candidate.shape}")
        return 1
    if reference_report.get("noise_seed") != candidate_report.get("noise_seed"):
        print("FAIL: noise seeds differ")
        return 1
    if reference_report["summary"]["noise_sha256"] != candidate_report["summary"]["noise_sha256"]:
        print("FAIL: noise hashes differ")
        return 1

    difference = np.abs(reference - candidate)
    per_channel_max = difference.max(axis=0)
    passed = bool(np.all(per_channel_max <= DEFAULT_ATOL))
    print(
        json.dumps(
            {
                "passed": passed,
                "mean_absolute_difference": float(difference.mean()),
                "max_absolute_difference_by_channel": per_channel_max.tolist(),
                "absolute_tolerance_by_channel": DEFAULT_ATOL.tolist(),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

