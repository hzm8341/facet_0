#!/usr/bin/env python3
"""Collect the FACET-0 runtime environment and execute a JAX GPU smoke test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"
CHECKPOINT_ROOT = PROJECT_ROOT / "Facet-0" / "facet0-post-training"
DATASET_ROOT = PROJECT_ROOT / "ManuFacet-1K" / "Facet0-1"


def run_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_info() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], OPENPI_ROOT)
    commit_date = run_command(["git", "show", "-s", "--format=%cI", "HEAD"], OPENPI_ROOT)
    status = run_command(["git", "status", "--porcelain"], OPENPI_ROOT)
    submodules = run_command(["git", "submodule", "status"], OPENPI_ROOT)
    return {
        "root": str(OPENPI_ROOT),
        "commit": commit,
        "commit_date": commit_date,
        "clean": status == "" if status is not None else None,
        "submodules": submodules.splitlines() if submodules else [],
    }


def gpu_info() -> list[dict[str, str]]:
    fields = ["index", "name", "memory.total", "driver_version", "compute_cap"]
    output = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    devices = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        devices.append(dict(zip(fields, values, strict=False)))
    return devices


def jax_smoke_test(matrix_size: int) -> dict[str, Any]:
    # Keep JAX from reserving most GPU memory; later checkpoint tests control
    # allocation independently.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    devices = [str(device) for device in jax.devices()]
    started = time.perf_counter()
    x = jnp.arange(matrix_size * matrix_size, dtype=jnp.float32).reshape(
        matrix_size, matrix_size
    )
    result = (x @ x.T).block_until_ready()
    elapsed = time.perf_counter() - started
    all_finite = bool(jnp.isfinite(result).all().block_until_ready())

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
        "matrix_size": matrix_size,
        "result_shape": list(result.shape),
        "result_dtype": str(result.dtype),
        "all_finite": all_finite,
        "elapsed_seconds": round(elapsed, 6),
    }


def build_report(matrix_size: int) -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    smoke = jax_smoke_test(matrix_size)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "gpu": gpu_info(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "assets": {
            "checkpoint_exists": CHECKPOINT_ROOT.is_dir(),
            "dataset_exists": DATASET_ROOT.is_dir(),
        },
        "openpi": git_info(),
        "packages": package_versions(
            [
                "openpi",
                "jax",
                "jaxlib",
                "jax-cuda12-plugin",
                "flax",
                "orbax-checkpoint",
                "numpy",
                "torch",
                "transformers",
                "lerobot",
            ]
        ),
        "jax_smoke_test": smoke,
        "success": (
            smoke["backend"] == "gpu"
            and bool(smoke["devices"])
            and smoke["all_finite"]
            and CHECKPOINT_ROOT.is_dir()
            and DATASET_ROOT.is_dir()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Parent directories are created.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=1024,
        help="Square matrix size used by the JAX smoke test (default: 1024).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be positive")

    report = build_report(args.matrix_size)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

