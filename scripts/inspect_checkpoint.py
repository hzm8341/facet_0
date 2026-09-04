#!/usr/bin/env python3
"""Compare the FACET-0 Orbax checkpoint with an OpenPI pi0.5 model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "Facet-0" / "facet0-post-training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore all arrays and load them into the OpenPI model on the active JAX device.",
    )
    return parser.parse_args()


def normalize_checkpoint_key(key: tuple[str, ...]) -> tuple[str, ...]:
    # NNX checkpoints saved through OpenPI contain an implementation-level
    # trailing `value` node. OpenPI strips it after restore as well.
    return key[:-1] if key and key[-1] == "value" else key


def array_description(value: Any) -> dict[str, Any]:
    shape = tuple(int(item) for item in value.shape)
    return {
        "shape": list(shape),
        "dtype": str(value.dtype),
        "parameters": math.prod(shape),
    }


def inspect(checkpoint: Path, restore: bool) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    from flax import nnx, traverse_util
    import orbax.checkpoint as ocp

    from openpi.models import model as model_lib
    from openpi.models import pi0_config

    checkpoint = checkpoint.resolve()
    params_path = checkpoint / "params"
    config_path = checkpoint / "config.json"
    norm_stats_path = checkpoint / "assets" / "norm_stats.json"

    if not params_path.is_dir():
        raise FileNotFoundError(f"Orbax params directory not found: {params_path}")

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_path)
    checkpoint_flat_raw = traverse_util.flatten_dict(metadata["params"])
    checkpoint_flat = {
        normalize_checkpoint_key(tuple(str(part) for part in key)): value
        for key, value in checkpoint_flat_raw.items()
    }

    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    expected_state = nnx.state(abstract_model, nnx.Param).to_pure_dict()
    expected_flat = traverse_util.flatten_dict(expected_state)

    checkpoint_keys = set(checkpoint_flat)
    expected_keys = set(expected_flat)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    common = sorted(checkpoint_keys & expected_keys)
    shape_mismatches = []
    for key in common:
        checkpoint_shape = tuple(checkpoint_flat[key].shape)
        expected_shape = tuple(expected_flat[key].shape)
        if checkpoint_shape != expected_shape:
            shape_mismatches.append(
                {
                    "key": "/".join(key),
                    "checkpoint": list(checkpoint_shape),
                    "expected": list(expected_shape),
                }
            )

    inventory = []
    module_counts: Counter[str] = Counter()
    total_parameters = 0
    for key in sorted(checkpoint_flat):
        description = array_description(checkpoint_flat[key])
        total_parameters += description["parameters"]
        module_counts[key[0]] += description["parameters"]
        inventory.append({"key": "/".join(key), **description})

    restored = {
        "requested": restore,
        "success": None,
        "seconds": None,
        "leaf_count": None,
        "device": None,
        "error": None,
    }
    if restore:
        started = time.perf_counter()
        try:
            params = model_lib.restore_params(params_path)
            jax.block_until_ready(params)
            trained_model = config.load(params, remove_extra_params=False)
            restored.update(
                {
                    "success": True,
                    "seconds": round(time.perf_counter() - started, 6),
                    "leaf_count": len(jax.tree.leaves(nnx.state(trained_model, nnx.Param))),
                    "device": str(jax.devices()[0]),
                }
            )
        except Exception as error:  # Preserve a machine-readable failure report.
            restored.update(
                {
                    "success": False,
                    "seconds": round(time.perf_counter() - started, 6),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    forbidden_module_terms = ["critic", "td3", "bottleneck", "judge", "value_head"]
    named_facets = {
        term: sorted("/".join(key) for key in checkpoint_keys if term in "/".join(key).lower())
        for term in forbidden_module_terms
    }

    compatible = not missing and not unexpected and not shape_mismatches
    if restore:
        compatible = compatible and restored["success"] is True

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "files": {
            "config": str(config_path),
            "config_exists": config_path.is_file(),
            "norm_stats": str(norm_stats_path),
            "norm_stats_exists": norm_stats_path.is_file(),
        },
        "openpi_config": {
            "model_type": config.model_type.value,
            "pi05": config.pi05,
            "dtype": config.dtype,
            "paligemma_variant": config.paligemma_variant,
            "action_expert_variant": config.action_expert_variant,
            "action_dim": config.action_dim,
            "action_horizon": config.action_horizon,
            "max_token_len": config.max_token_len,
            "discrete_state_input": config.discrete_state_input,
        },
        "summary": {
            "checkpoint_leaf_count": len(checkpoint_flat),
            "expected_leaf_count": len(expected_flat),
            "total_parameters": total_parameters,
            "parameter_count_by_top_module": dict(module_counts),
            "missing_keys": ["/".join(key) for key in missing],
            "unexpected_keys": ["/".join(key) for key in unexpected],
            "shape_mismatches": shape_mismatches,
            "compatible": compatible,
        },
        "explicit_facets_module_search": named_facets,
        "restore_test": restored,
        "inventory": inventory,
    }


def main() -> int:
    args = parse_args()
    report = inspect(args.checkpoint, args.restore)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["summary"]["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
