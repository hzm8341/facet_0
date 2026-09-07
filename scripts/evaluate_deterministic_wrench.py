#!/usr/bin/env python3
"""Evaluate the deterministic residual-wrench control against the causal constant baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.shared import normalize

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.wrench_control import build_wrench_control_pool
from facet0.data.wrench_windows import WrenchNormalizer
from facet0.evaluation.offline_metrics import compute_wrench_metrics, wrench_constant_baseline
from facet0.models.deterministic_wrench import (
    DeterministicResidualWrenchModel,
    load_deterministic_wrench,
)
from facet0.training.train_alignment import load_split, sample_sequence_pool
from facet0.training.train_deterministic_wrench import (
    DeterministicTrainConfig,
    predict_policy_actions,
)


@nnx.jit
def _predict(model, history, history_valid, actions):
    return model(history, history_valid, actions)


def evaluate(
    config,
    head_path,
    *,
    split,
    num_sequences,
    batch_size,
    seed,
    action_source,
    policy_batch_size,
    policy_num_steps,
    residual_scales,
):
    dataset = ManuFacetDataset(config.dataset)
    rng = np.random.default_rng(seed)
    pairs = sample_sequence_pool(
        dataset,
        load_split(config.split_path)[split],
        num_sequences,
        rng,
    )
    stats = normalize.load(config.checkpoint / "assets")
    wrench_normalizer = WrenchNormalizer.from_openpi_norm_stats(stats)
    pool = build_wrench_control_pool(
        dataset,
        pairs,
        wrench_normalizer=wrench_normalizer,
        action_q01=stats["actions"].q01,
        action_q99=stats["actions"].q99,
        history_len=config.history_len,
        horizon=config.horizon,
    )
    model = DeterministicResidualWrenchModel(config.model_config(), rngs=nnx.Rngs(config.seed))
    load_deterministic_wrench(model, head_path)
    action_condition = pool.actions
    if action_source == "policy":
        action_condition = predict_policy_actions(
            config,
            dataset,
            pairs,
            wrench_normalizer,
            batch_size=policy_batch_size,
            num_steps=policy_num_steps,
            seed=seed,
        )
    elif action_source == "zero":
        action_condition = np.zeros_like(pool.actions)
    predictions = []
    for start in range(0, len(pool), batch_size):
        stop = start + batch_size
        predictions.append(
            np.asarray(
                _predict(
                    model,
                    jnp.asarray(pool.history[start:stop]),
                    jnp.asarray(pool.history_valid[start:stop]),
                    jnp.asarray(action_condition[start:stop]),
                )
            )
        )
    pred_delta = np.concatenate(predictions)
    physical_history = wrench_normalizer.unnormalize(pool.history)
    current = physical_history[:, -1:, :]
    target = current + wrench_normalizer.unnormalize_delta(pool.target)
    baseline = wrench_constant_baseline(physical_history, pool.history_valid, config.horizon)
    baseline_metrics = compute_wrench_metrics(baseline, target, valid=pool.target_valid)
    scale_metrics = []
    for scale in residual_scales:
        scaled_pred = current + wrench_normalizer.unnormalize_delta(pred_delta * scale)
        metrics = compute_wrench_metrics(scaled_pred, target, valid=pool.target_valid)
        scale_metrics.append(
            {
                "scale": scale,
                "mean_channel_mae": float(metrics.channel_mae.mean()),
                "channel_mae": metrics.channel_mae.tolist(),
            }
        )
    selected = min(scale_metrics, key=lambda item: item["mean_channel_mae"])
    selected_scale = selected["scale"]
    pred = current + wrench_normalizer.unnormalize_delta(pred_delta * selected_scale)
    model_metrics = compute_wrench_metrics(pred, target, valid=pool.target_valid)
    model_mae = model_metrics.channel_mae
    baseline_mae = baseline_metrics.channel_mae
    normalized_mae = np.abs(pred_delta * selected_scale - pool.target)[pool.target_valid].mean(axis=0)
    return {
        "split": split,
        "num_sequences": num_sequences,
        "seed": seed,
        "head_path": str(head_path),
        "action_condition": {
            "demonstration": "demonstration_action_chunk_offline_oracle",
            "policy": "frozen_facet_policy_prediction",
            "zero": "zero_action_ablation",
        }[action_source],
        "policy_num_steps": policy_num_steps if action_source == "policy" else None,
        "residual_scale": selected_scale,
        "residual_scale_sweep": scale_metrics,
        "model": model_metrics.to_dict(),
        "constant_baseline": baseline_metrics.to_dict(),
        "normalized_channel_mae": normalized_mae.tolist(),
        "force_mae_mean": float(model_mae[:3].mean()),
        "torque_mae_mean": float(model_mae[3:].mean()),
        "baseline_force_mae_mean": float(baseline_mae[:3].mean()),
        "baseline_torque_mae_mean": float(baseline_mae[3:].mean()),
        "mean_channel_mae": {
            "model": float(model_mae.mean()),
            "constant_baseline": float(baseline_mae.mean()),
            "relative_improvement": float(1.0 - model_mae.mean() / baseline_mae.mean()),
        },
        "beats_constant_baseline": bool(model_mae.mean() < baseline_mae.mean()),
        "beats_baseline_per_channel": (model_mae < baseline_mae).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--action-source",
        choices=["demonstration", "policy", "zero"],
        default="demonstration",
    )
    parser.add_argument("--policy-batch-size", type=int, default=1)
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument(
        "--residual-scales",
        default="1.0",
        help="comma-separated residual scales; lowest MAE is selected within this evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DeterministicTrainConfig.from_yaml(args.config)
    residual_scales = [float(value) for value in args.residual_scales.split(",")]
    report = evaluate(
        config,
        args.head,
        split=args.split,
        num_sequences=args.num_sequences,
        batch_size=args.batch_size,
        seed=args.seed,
        action_source=args.action_source,
        policy_batch_size=args.policy_batch_size,
        policy_num_steps=args.policy_num_steps,
        residual_scales=residual_scales,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
