#!/usr/bin/env python3
"""Create deterministic episode-level ManuFacet train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facet0.data import ManuFacetDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "ManuFacet-1K/Facet0-1")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs/splits/facet0_1_seed20260903.json")
    args = parser.parse_args()

    dataset = ManuFacetDataset(args.dataset)
    rng = np.random.default_rng(args.seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for episode_index, episode in dataset.episodes.items():
        groups[str(episode["task_family"])].append(episode_index)

    splits = {"train": [], "validation": [], "test": []}
    family_counts: dict[str, dict[str, int]] = {}
    for family, episode_ids in sorted(groups.items()):
        shuffled = np.asarray(sorted(episode_ids))
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * 0.8)
        validation_end = train_end + int(len(shuffled) * 0.1)
        family_split = {
            "train": shuffled[:train_end].tolist(),
            "validation": shuffled[train_end:validation_end].tolist(),
            "test": shuffled[validation_end:].tolist(),
        }
        for name, values in family_split.items():
            splits[name].extend(values)
        family_counts[family] = {name: len(values) for name, values in family_split.items()}

    for values in splits.values():
        values.sort()
    all_ids = [episode for values in splits.values() for episode in values]
    assert len(all_ids) == len(dataset)
    assert len(set(all_ids)) == len(dataset)

    canonical = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "strategy": "episode-level 80/10/10, stratified by task_family",
        "split_sha256": hashlib.sha256(canonical).hexdigest(),
        "counts": {name: len(values) for name, values in splits.items()},
        "family_counts": family_counts,
        "splits": splits,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("seed", "split_sha256", "counts", "family_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

