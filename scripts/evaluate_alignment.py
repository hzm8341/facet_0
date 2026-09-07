#!/usr/bin/env python3
"""Evaluate a trained FACET wrench head against the hold-current-wrench baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

from facet0.data.manufacet import ManuFacetDataset
from facet0.evaluation.offline_metrics import compute_wrench_metrics, wrench_constant_baseline
from facet0.training.train_alignment import (
    AlignmentTrainConfig,
    build_example,
    build_model_and_transforms,
    load_split,
    load_wrench_head,
    sample_sequence_pool,
    stack_examples,
)


@nnx.jit
def _sample(model, rng, observation, history, history_valid, num_steps):
    return model.sample_action_and_wrench(
        rng,
        observation,
        history,
        history_valid,
        num_steps=num_steps,
    )


def _batches(items: list[tuple[int, int]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate(
    config: AlignmentTrainConfig,
    head_path: Path,
    *,
    split: str,
    num_sequences: int,
    batch_size: int,
    num_steps: int,
    num_samples: int,
    seed: int,
) -> dict:
    dataset = ManuFacetDataset(config.dataset)
    episode_indices = load_split(config.split_path)[split]
    pairs = sample_sequence_pool(
        dataset,
        episode_indices,
        size=num_sequences,
        rng=np.random.default_rng(seed),
    )
    (
        model,
        facet_inputs,
        delta_actions,
        normalize_transform,
        model_input_transforms,
        wrench_normalizer,
    ) = build_model_and_transforms(config)
    load_wrench_head(model, head_path)

    pred_physical, target_physical, history_physical, target_valid = [], [], [], []
    rng = jax.random.key(seed)
    for pairs_batch in _batches(pairs, batch_size):
        examples = [
            build_example(
                dataset,
                episode,
                frame,
                facet_inputs=facet_inputs,
                delta_actions=delta_actions,
                normalize_transform=normalize_transform,
                model_input_transforms=model_input_transforms,
                wrench_normalizer=wrench_normalizer,
                wrench_target_representation=config.wrench_target_representation,
                action_horizon=config.action_horizon,
                wrench_history_len=config.wrench_history_len,
            )
            for episode, frame in pairs_batch
        ]
        batch = stack_examples(examples)
        observation = _model.Observation.from_dict(batch)
        sample_predictions = []
        for _ in range(num_samples):
            rng, sample_rng = jax.random.split(rng)
            _actions, pred_sample = _sample(
                model,
                sample_rng,
                observation,
                jnp.asarray(batch["wrench_history"]),
                jnp.asarray(batch["wrench_history_valid"]),
                num_steps,
            )
            sample_predictions.append(np.asarray(pred_sample))
        pred_normalized = np.mean(sample_predictions, axis=0)
        physical_history = wrench_normalizer.unnormalize(batch["wrench_history"])
        history_physical.append(physical_history)
        if config.wrench_target_representation == "delta_from_current":
            current = physical_history[:, -1:, :]
            pred_physical.append(current + wrench_normalizer.unnormalize_delta(np.asarray(pred_normalized)))
            target_physical.append(current + wrench_normalizer.unnormalize_delta(batch["wrench_target"]))
        else:
            pred_physical.append(wrench_normalizer.unnormalize(np.asarray(pred_normalized)))
            target_physical.append(wrench_normalizer.unnormalize(batch["wrench_target"]))
        target_valid.append(batch["wrench_target_valid"])

    pred = np.concatenate(pred_physical)
    target = np.concatenate(target_physical)
    history = np.concatenate(history_physical)
    valid = np.concatenate(target_valid)
    baseline = wrench_constant_baseline(history, np.ones(history.shape[:-1], dtype=bool), config.action_horizon)
    model_metrics = compute_wrench_metrics(pred, target, valid=valid)
    baseline_metrics = compute_wrench_metrics(baseline, target, valid=valid)
    model_mae = model_metrics.channel_mae
    baseline_mae = baseline_metrics.channel_mae
    return {
        "split": split,
        "num_sequences": num_sequences,
        "num_sampling_steps": num_steps,
        "num_samples_per_prediction": num_samples,
        "seed": seed,
        "head_path": str(head_path),
        "model": model_metrics.to_dict(),
        "constant_baseline": baseline_metrics.to_dict(),
        "mean_channel_mae": {
            "model": float(model_mae.mean()),
            "constant_baseline": float(baseline_mae.mean()),
            "relative_improvement": float(1.0 - model_mae.mean() / baseline_mae.mean()),
        },
        "beats_constant_baseline": bool(model_mae.mean() < baseline_mae.mean()),
        "beats_baseline_per_channel": (model_mae < baseline_mae).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="average this many independent flow samples before point-estimate metrics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AlignmentTrainConfig.from_yaml(args.config)
    report = evaluate(
        config,
        args.head,
        split=args.split,
        num_sequences=args.num_sequences,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
