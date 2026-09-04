#!/usr/bin/env python3
"""Stage 5 alignment training: an independent 6-dim wrench flow-matching head on top of the
frozen, official FACET-0 action policy.

Per FACET0_REPRODUCTION_PLAN.md Stage 5 / HANDOFF.md Sec 10: run in ``overfit`` mode first (32-128
short sequences, memorized across many steps) and confirm both losses collapse before ever
running ``full`` mode against the whole train split. This script never writes into the official
checkpoint directory or `ManuFacet-1K/`; trained wrench-head weights are saved separately under
``--output-dir``. `Facet-0/*` params stay frozen both by gradient (`wrench_head_trainable_filter`)
and by construction (`nnx.Optimizer(..., wrt=...)` never sees a `pi0.*` leaf), so the official
action policy this script loads is never mutated -- `scripts/verify_golden_inference.py` is
unaffected by anything this script does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
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
from facet0.data.wrench_windows import WRENCH_DIM, build_wrench_window
from facet0.models.joint_action_wrench import (
    JointActionWrenchConfig,
    JointActionWrenchModel,
    wrench_head_trainable_filter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("facet0.train_alignment")


@dataclasses.dataclass(frozen=True)
class AlignmentTrainConfig:
    """Stage 5 training hyperparameters, with provenance per HANDOFF.md's tagging convention.

    `paper`: value is stated by FACET0_REPRODUCTION_PLAN.md/the paper. `inferred`: derived from
    the released checkpoint/dataset, not stated outright. `experimental`: not specified anywhere;
    a reproduction choice, expected to be revisited.
    """

    checkpoint: Path
    dataset: Path
    split_path: Path
    output_dir: Path

    mode: str = "overfit"  # experimental: "overfit" (memorize a fixed pool) or "full"
    num_overfit_sequences: int = 64  # paper (range 32-128), this is the chosen default
    num_steps: int = 2000  # experimental
    batch_size: int = 8  # experimental
    learning_rate: float = 1e-4  # experimental
    seed: int = 0  # project

    action_horizon: int = 50  # paper
    max_token_len: int = 200  # paper (pi05 default)
    wrench_history_len: int = 10  # paper (K=10)
    wrench_head_width: int = 512  # experimental
    lambda_pre: float = 0.1  # paper (initial wrench loss weight)

    log_every: int = 10  # project
    checkpoint_every: int = 200  # project
    resume_from: Path | None = None  # project: a previous run's wrench_head_*.npz to warm-start from

    @classmethod
    def from_yaml(cls, path: Path, *, overrides: dict[str, Any] | None = None) -> "AlignmentTrainConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.update(overrides or {})
        for key in ("checkpoint", "dataset", "split_path", "output_dir", "resume_from"):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        field_names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - field_names
        if unknown:
            raise ValueError(f"Unknown alignment config keys: {sorted(unknown)}")
        return cls(**raw)

    def joint_config(self) -> JointActionWrenchConfig:
        return JointActionWrenchConfig(
            wrench_dim=WRENCH_DIM,
            wrench_history_len=self.wrench_history_len,
            wrench_head_width=self.wrench_head_width,
        )


def load_split(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["splits"]


def _episode_length(dataset: ManuFacetDataset, episode_index: int) -> int:
    return int(dataset.episodes[episode_index]["length"])


def sample_sequence_pool(
    dataset: ManuFacetDataset,
    episode_indices: list[int],
    size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Draws `size` random `(episode_index, frame_index)` pairs from the given episodes."""
    chosen_episodes = rng.choice(episode_indices, size=size, replace=size > len(episode_indices))
    pool = []
    for episode_index in chosen_episodes:
        length = _episode_length(dataset, int(episode_index))
        frame_index = int(rng.integers(0, length))
        pool.append((int(episode_index), frame_index))
    return pool


def next_batch_pairs(
    dataset: ManuFacetDataset,
    episode_indices: list[int],
    batch_size: int,
    rng: np.random.Generator,
    *,
    fixed_pool: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """A fresh random batch for "full" mode, or a batch resampled (with replacement) from
    `fixed_pool` for "overfit" mode -- the repetition across steps is what drives memorization.
    """
    if fixed_pool is not None:
        indices = rng.integers(0, len(fixed_pool), size=batch_size)
        return [fixed_pool[i] for i in indices]
    return sample_sequence_pool(dataset, episode_indices, batch_size, rng)


def build_example(
    dataset: ManuFacetDataset,
    episode_index: int,
    frame_index: int,
    *,
    facet_inputs: FacetInputs,
    delta_actions: openpi_transforms.DeltaActions,
    normalize_transform: openpi_transforms.Normalize,
    model_input_transforms: list,
    action_horizon: int,
    wrench_history_len: int,
) -> dict[str, np.ndarray]:
    """Builds one unbatched training example: the same input-transform chain
    `facet0.policies.create_facet_policy` uses for inference, plus the causal wrench
    history/future target from `facet0.data.wrench_windows`.
    """
    raw = dataset.load_frame(episode_index, frame_index, include_images=True)
    action_chunk, _action_valid = dataset.load_action_chunk(episode_index, frame_index, horizon=action_horizon)
    raw["action"] = action_chunk

    data = facet_inputs(raw)
    data = delta_actions(data)
    data = normalize_transform(data)
    for transform in model_input_transforms:
        data = transform(data)

    window = build_wrench_window(
        dataset, episode_index, frame_index, history_len=wrench_history_len, horizon=action_horizon
    )
    data["wrench_history"] = window.history
    data["wrench_history_valid"] = window.history_valid
    data["wrench_target"] = window.future
    data["wrench_target_valid"] = window.future_valid
    return data


def stack_examples(examples: list[dict]) -> dict:
    return jax.tree.map(lambda *leaves: np.stack(leaves, axis=0), *examples)


def build_model_and_transforms(config: AlignmentTrainConfig):
    """Restores the frozen Pi0 backbone and builds the exact same input-transform chain
    `facet0.policies.create_facet_policy` uses for inference (see that module for why each
    transform is there), so training sees data distributed identically to inference.
    """
    pi0_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
    )
    params = _model.restore_params(config.checkpoint / "params", dtype=jnp.bfloat16)
    pi0_model = pi0_cfg.load(params, remove_extra_params=False)
    norm_stats = normalize.load(config.checkpoint / "assets")
    model_input_transforms = list(training_config.ModelTransformFactory()(pi0_cfg).inputs)

    facet_inputs = FacetInputs()
    delta_actions = openpi_transforms.DeltaActions(EEF_DELTA_MASK)
    normalize_transform = openpi_transforms.Normalize(norm_stats, use_quantiles=True)

    model = JointActionWrenchModel(pi0_model, config.joint_config(), rngs=nnx.Rngs(config.seed))
    return model, facet_inputs, delta_actions, normalize_transform, model_input_transforms


def save_wrench_head(model: JointActionWrenchModel, path: Path) -> None:
    """Saves only `model.wrench_head`'s parameters -- never touches `model.pi0` or the checkpoint
    directory it was restored from.
    """
    state = nnx.state(model.wrench_head)
    flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in flat.items()})


def load_wrench_head(model: JointActionWrenchModel, path: Path) -> None:
    with np.load(path) as data:
        flat = {tuple(k.split("/")): jnp.asarray(v) for k, v in data.items()}
    pure = traverse_util.unflatten_dict(flat)
    graphdef, state = nnx.split(model.wrench_head)
    state.replace_by_pure_dict(pure)
    model.wrench_head = nnx.merge(graphdef, state)


def make_train_step(lambda_pre: float):
    trainable_filter = nnx.All(nnx.Param, wrench_head_trainable_filter())

    @nnx.jit
    def train_step(
        optimizer: nnx.Optimizer,
        rng,
        observation: _model.Observation,
        actions,
        wrench_history,
        wrench_history_valid,
        wrench_target,
        wrench_target_valid,
    ):
        def loss_fn(model: JointActionWrenchModel):
            action_loss, wrench_loss = model.compute_losses(
                rng,
                observation,
                actions,
                wrench_history,
                wrench_history_valid,
                wrench_target,
                wrench_target_valid,
                train=True,
            )
            action_loss_mean = jnp.mean(action_loss)
            wrench_denom = jnp.maximum(jnp.sum(wrench_target_valid), 1)
            wrench_loss_mean = jnp.sum(wrench_loss) / wrench_denom
            total = action_loss_mean + lambda_pre * wrench_loss_mean
            return total, (action_loss_mean, wrench_loss_mean)

        grad_fn = nnx.grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter), has_aux=True)
        grads, (action_loss_mean, wrench_loss_mean) = grad_fn(optimizer.model)
        optimizer.update(grads)
        return action_loss_mean, wrench_loss_mean

    return train_step


def run_training(config: AlignmentTrainConfig) -> list[dict[str, Any]]:
    if config.mode not in ("overfit", "full"):
        raise ValueError(f"mode must be 'overfit' or 'full', got {config.mode!r}")

    dataset = ManuFacetDataset(config.dataset)
    splits = load_split(config.split_path)
    train_episodes = splits["train"]

    model, facet_inputs, delta_actions, normalize_transform, model_input_transforms = build_model_and_transforms(
        config
    )
    if config.resume_from is not None:
        logger.info("resuming wrench_head weights from %s", config.resume_from)
        load_wrench_head(model, config.resume_from)
    optimizer = nnx.Optimizer(
        model, optax.adamw(config.learning_rate), wrt=nnx.All(nnx.Param, wrench_head_trainable_filter())
    )
    train_step = make_train_step(config.lambda_pre)

    data_rng = np.random.default_rng(config.seed)
    fixed_pool = None
    if config.mode == "overfit":
        fixed_pool = sample_sequence_pool(dataset, train_episodes, config.num_overfit_sequences, data_rng)
        logger.info("overfit mode: memorizing a fixed pool of %d sequences", len(fixed_pool))

    step_rng = jax.random.key(config.seed)
    history: list[dict[str, Any]] = []
    for step in range(config.num_steps):
        pairs = next_batch_pairs(dataset, train_episodes, config.batch_size, data_rng, fixed_pool=fixed_pool)
        examples = [
            build_example(
                dataset,
                episode_index,
                frame_index,
                facet_inputs=facet_inputs,
                delta_actions=delta_actions,
                normalize_transform=normalize_transform,
                model_input_transforms=model_input_transforms,
                action_horizon=config.action_horizon,
                wrench_history_len=config.wrench_history_len,
            )
            for episode_index, frame_index in pairs
        ]
        batch = stack_examples(examples)
        observation = _model.Observation.from_dict(batch)

        step_rng, iter_rng = jax.random.split(step_rng)
        started = time.perf_counter()
        action_loss, wrench_loss = train_step(
            optimizer,
            iter_rng,
            observation,
            jnp.asarray(batch["actions"]),
            jnp.asarray(batch["wrench_history"]),
            jnp.asarray(batch["wrench_history_valid"]),
            jnp.asarray(batch["wrench_target"]),
            jnp.asarray(batch["wrench_target_valid"]),
        )
        step_seconds = time.perf_counter() - started

        if step % config.log_every == 0 or step == config.num_steps - 1:
            record = {
                "step": step,
                "action_loss": float(action_loss),
                "wrench_loss": float(wrench_loss),
                "step_seconds": round(step_seconds, 4),
            }
            history.append(record)
            logger.info(
                "step=%d action_loss=%.6f wrench_loss=%.6f (%.2fs)",
                step,
                record["action_loss"],
                record["wrench_loss"],
                step_seconds,
            )
            # written every log_every steps (not just at the end) so a killed/interrupted run
            # doesn't lose its history -- `run_inference.py`-style long jobs in this environment
            # have been observed to be killed by something outside this process's control.
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / "train_history.json").write_text(json.dumps(history, indent=2))

        if config.checkpoint_every and step > 0 and step % config.checkpoint_every == 0:
            save_wrench_head(model, config.output_dir / f"wrench_head_step{step}.npz")

    save_wrench_head(model, config.output_dir / "wrench_head_final.npz")
    (config.output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config, e.g. configs/alignment/overfit.yaml")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "Facet-0/facet0-post-training")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "ManuFacet-1K/Facet0-1")
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=["overfit", "full"])
    parser.add_argument("--num-overfit-sequences", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lambda-pre", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", type=Path, help="a previous run's wrench_head_*.npz to warm-start from")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    cli_overrides = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "mode": args.mode,
        "num_overfit_sequences": args.num_overfit_sequences,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_pre": args.lambda_pre,
        "seed": args.seed,
        "split_path": args.split_path,
        "output_dir": args.output_dir,
        "resume_from": args.resume_from,
    }
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    if args.config is not None:
        config = AlignmentTrainConfig.from_yaml(args.config, overrides=cli_overrides)
    else:
        required = {"split_path", "output_dir"} - set(cli_overrides)
        if required:
            raise SystemExit(f"--config not given; missing required flags: {sorted(required)}")
        config = AlignmentTrainConfig(**cli_overrides)

    logger.info("Stage 5 alignment training: mode=%s output_dir=%s", config.mode, config.output_dir)
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
