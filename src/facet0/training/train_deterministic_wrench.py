#!/usr/bin/env python3
"""Train the deterministic residual-wrench control head on an in-memory frame pool."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from openpi import transforms as openpi_transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config as training_config

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.transforms import EEF_DELTA_MASK, FacetInputs
from facet0.data.wrench_control import build_wrench_control_pool, stratify_by_change
from facet0.data.wrench_windows import WrenchNormalizer
from facet0.models.deterministic_wrench import (
    DeterministicResidualWrenchModel,
    DeterministicWrenchConfig,
    save_deterministic_wrench,
)
from facet0.training.train_alignment import build_example, load_split, sample_sequence_pool


@dataclasses.dataclass(frozen=True)
class DeterministicTrainConfig:
    checkpoint: Path
    dataset: Path
    split_path: Path
    output_dir: Path
    mode: str = "overfit"
    pool_size: int = 64
    candidate_multiplier: int = 1
    high_change_fraction: float = 0.0
    num_steps: int = 2000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 0.1
    history_len: int = 10
    horizon: int = 50
    width: int = 256
    seed: int = 0
    log_every: int = 50
    checkpoint_every: int = 500
    action_source: str = "demonstration"
    policy_batch_size: int = 1
    policy_num_steps: int = 10

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict | None = None):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.update(overrides or {})
        for key in ("checkpoint", "dataset", "split_path", "output_dir"):
            raw[key] = Path(raw[key])
        unknown = set(raw) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown deterministic wrench config keys: {sorted(unknown)}")
        return cls(**raw)

    def model_config(self) -> DeterministicWrenchConfig:
        return DeterministicWrenchConfig(
            history_len=self.history_len,
            horizon=self.horizon,
            width=self.width,
        )


def create_pool(config: DeterministicTrainConfig):
    dataset = ManuFacetDataset(config.dataset)
    train_episodes = load_split(config.split_path)["train"]
    rng = np.random.default_rng(config.seed)
    candidate_size = config.pool_size * config.candidate_multiplier
    pairs = sample_sequence_pool(dataset, train_episodes, candidate_size, rng)
    stats = normalize.load(config.checkpoint / "assets")
    wrench_normalizer = WrenchNormalizer.from_openpi_norm_stats(stats)
    candidates = build_wrench_control_pool(
        dataset,
        pairs,
        wrench_normalizer=wrench_normalizer,
        action_q01=stats["actions"].q01,
        action_q99=stats["actions"].q99,
        history_len=config.history_len,
        horizon=config.horizon,
    )
    if candidate_size != config.pool_size or config.high_change_fraction:
        pool = stratify_by_change(
            candidates,
            output_size=config.pool_size,
            high_change_fraction=config.high_change_fraction,
            rng=rng,
        )
    else:
        pool = candidates
    if config.action_source == "policy":
        policy_actions = predict_policy_actions(
            config,
            dataset,
            [tuple(map(int, pair)) for pair in pool.pairs],
            wrench_normalizer,
            batch_size=config.policy_batch_size,
            num_steps=config.policy_num_steps,
            seed=config.seed,
        )
        pool = dataclasses.replace(pool, actions=policy_actions)
    elif config.action_source != "demonstration":
        raise ValueError("action_source must be 'demonstration' or 'policy'")
    return pool


def predict_policy_actions(
    config,
    dataset,
    pairs,
    wrench_normalizer,
    *,
    batch_size,
    num_steps,
    seed,
):
    """Generate normalized action chunks from the frozen official FACET policy."""
    pi0_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=config.horizon,
        max_token_len=200,
    )
    params = _model.restore_params(config.checkpoint / "params", dtype=jnp.bfloat16)
    model = pi0_cfg.load(params, remove_extra_params=False)
    stats = normalize.load(config.checkpoint / "assets")
    facet_inputs = FacetInputs()
    delta_actions = openpi_transforms.DeltaActions(EEF_DELTA_MASK)
    normalize_transform = openpi_transforms.Normalize(stats, use_quantiles=True)
    model_input_transforms = list(training_config.ModelTransformFactory()(pi0_cfg).inputs)

    @nnx.jit
    def sample_actions(model, rng, observation, sampling_steps):
        return model.sample_actions(rng, observation, num_steps=sampling_steps)

    outputs = []
    rng = jax.random.key(seed)
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
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
                wrench_target_representation="delta_from_current",
                action_horizon=config.horizon,
                wrench_history_len=config.history_len,
            )
            for episode, frame in batch_pairs
        ]
        batch = jax.tree.map(lambda *leaves: np.stack(leaves), *examples)
        rng, sample_rng = jax.random.split(rng)
        actions = sample_actions(
            model,
            sample_rng,
            _model.Observation.from_dict(batch),
            num_steps,
        )
        outputs.append(np.asarray(actions)[..., :7])
    return np.concatenate(outputs)


def make_train_step(huber_delta: float):
    @nnx.jit
    def train_step(optimizer, history, history_valid, actions, target, target_valid):
        def loss_fn(model):
            prediction = model(history, history_valid, actions)
            element_loss = optax.huber_loss(prediction, target, delta=huber_delta).mean(axis=-1)
            denominator = jnp.maximum(target_valid.sum(), 1)
            loss = (element_loss * target_valid).sum() / denominator
            mae = (jnp.abs(prediction - target).mean(axis=-1) * target_valid).sum() / denominator
            return loss, mae

        (loss, mae), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(gradients)
        return loss, mae

    return train_step


def run_training(config: DeterministicTrainConfig) -> list[dict]:
    if config.mode not in ("overfit", "full"):
        raise ValueError("mode must be 'overfit' or 'full'")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(config).items()
    }
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    pool = create_pool(config)
    (config.output_dir / "sequence_pool.json").write_text(
        json.dumps(
            [
                {
                    "episode_index": int(pair[0]),
                    "frame_index": int(pair[1]),
                    "change_score": float(score),
                }
                for pair, score in zip(pool.pairs, pool.change_score, strict=True)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    model = DeterministicResidualWrenchModel(config.model_config(), rngs=nnx.Rngs(config.seed))
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        wrt=nnx.Param,
    )
    train_step = make_train_step(config.huber_delta)
    rng = np.random.default_rng(config.seed + 1)
    history: list[dict] = []
    for step in range(config.num_steps):
        indices = rng.integers(0, len(pool), size=config.batch_size)
        loss, mae = train_step(
            optimizer,
            jnp.asarray(pool.history[indices]),
            jnp.asarray(pool.history_valid[indices]),
            jnp.asarray(pool.actions[indices]),
            jnp.asarray(pool.target[indices]),
            jnp.asarray(pool.target_valid[indices]),
        )
        if step % config.log_every == 0 or step == config.num_steps - 1:
            record = {"step": step, "huber_loss": float(loss), "normalized_mae": float(mae)}
            history.append(record)
            (config.output_dir / "train_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        if config.checkpoint_every and step and step % config.checkpoint_every == 0:
            save_deterministic_wrench(model, config.output_dir / f"wrench_head_step{step}.npz")
    save_deterministic_wrench(model, config.output_dir / "wrench_head_final.npz")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {
        key: value
        for key, value in {"num_steps": args.num_steps, "output_dir": args.output_dir}.items()
        if value is not None
    }
    config = DeterministicTrainConfig.from_yaml(args.config, overrides)
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
