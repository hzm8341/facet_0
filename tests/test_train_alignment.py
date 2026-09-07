from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config as training_config

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.transforms import EEF_DELTA_MASK, FacetInputs
from facet0.data.wrench_windows import WrenchNormalizer
from facet0.models.joint_action_wrench import JointActionWrenchConfig, JointActionWrenchModel, wrench_head_trainable_filter
from openpi import transforms as openpi_transforms
from facet0.training.train_alignment import (
    AlignmentTrainConfig,
    build_example,
    load_wrench_head,
    make_train_step,
    next_batch_pairs,
    sample_sequence_pool,
    save_wrench_head,
    stack_examples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "ManuFacet-1K" / "Facet0-1"
CHECKPOINT_ASSETS = PROJECT_ROOT / "Facet-0" / "facet0-post-training" / "assets"

BATCH = 2
ACTION_DIM = 8
ACTION_HORIZON = 4
MAX_TOKEN_LEN = 16
WRENCH_HISTORY_LEN = 3


def _tiny_pi0_config() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=MAX_TOKEN_LEN,
    )


def _build_model(seed: int = 0) -> JointActionWrenchModel:
    config = _tiny_pi0_config()
    pi0 = config.create(jax.random.key(seed))
    joint_config = JointActionWrenchConfig(wrench_dim=6, wrench_history_len=WRENCH_HISTORY_LEN, wrench_head_width=32)
    return JointActionWrenchModel(pi0, joint_config, rngs=nnx.Rngs(seed + 1))


def _fake_batch(config: pi0_config.Pi0Config):
    observation = config.fake_obs(batch_size=BATCH)
    actions = config.fake_act(batch_size=BATCH)
    rng = np.random.default_rng(0)
    wrench_history = jnp.asarray(rng.normal(size=(BATCH, WRENCH_HISTORY_LEN, 6)), dtype=jnp.float32)
    wrench_history_valid = jnp.ones((BATCH, WRENCH_HISTORY_LEN), dtype=jnp.bool_)
    wrench_target = jnp.asarray(rng.normal(size=(BATCH, ACTION_HORIZON, 6)), dtype=jnp.float32)
    wrench_target_valid = jnp.ones((BATCH, ACTION_HORIZON), dtype=jnp.bool_)
    return observation, actions, wrench_history, wrench_history_valid, wrench_target, wrench_target_valid


def test_train_step_reduces_loss_on_a_fixed_tiny_batch():
    """Mechanics smoke test standing in for the real overfit run: repeatedly stepping on the same
    fixed batch *and* the same flow-matching noise/time draw (a fixed rng, not re-split every
    step -- `compute_losses` resamples noise/time from whatever rng it is given) must drive the
    combined loss down, exercising the exact same `nnx.jit` + `nnx.grad(...,
    argnums=nnx.DiffState(...))` + `nnx.Optimizer` wiring the real training loop uses, without
    paying for the real 2B-parameter checkpoint. Re-splitting the rng every step (a fresh flow
    target each time) would make single-step loss too noisy to expect monotonic improvement over
    only a handful of steps.
    """
    model = _build_model()
    optimizer = nnx.Optimizer(
        model, optax.adamw(1e-2), wrt=nnx.All(nnx.Param, wrench_head_trainable_filter())
    )
    train_step = make_train_step(lambda_pre=0.1)
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())

    fixed_rng = jax.random.key(0)
    action_losses, wrench_losses = [], []
    for _ in range(30):
        action_loss, wrench_loss = train_step(
            optimizer, fixed_rng, obs, actions, history, history_valid, target, target_valid
        )
        action_losses.append(float(action_loss))
        wrench_losses.append(float(wrench_loss))

    # only the wrench head is trainable, so the (frozen-backbone) action loss must not move...
    assert action_losses[-1] == pytest.approx(action_losses[0], rel=1e-4)
    # ...while the wrench loss, computed from the same fixed noise/time draw every step, must
    # drop substantially as the wrench head fits that one fixed flow-matching target.
    assert wrench_losses[-1] < 0.5 * wrench_losses[0]


def test_save_and_load_wrench_head_round_trip(tmp_path: Path):
    source = _build_model(seed=1)
    optimizer = nnx.Optimizer(
        source, optax.adamw(1e-2), wrt=nnx.All(nnx.Param, wrench_head_trainable_filter())
    )
    train_step = make_train_step(lambda_pre=0.1)
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())
    train_step(optimizer, jax.random.key(0), obs, actions, history, history_valid, target, target_valid)

    checkpoint_path = tmp_path / "wrench_head.npz"
    save_wrench_head(source, checkpoint_path)

    target_model = _build_model(seed=99)  # different init, and a different (fresh) pi0 backbone
    source_pi0_kernel = np.asarray(source.pi0.action_out_proj.kernel.value)
    target_pi0_kernel_before = np.asarray(target_model.pi0.action_out_proj.kernel.value)

    load_wrench_head(target_model, checkpoint_path)

    # wrench head now matches the source model...
    source_head_state = nnx.state(source.wrench_head).to_pure_dict()
    target_head_state = nnx.state(target_model.wrench_head).to_pure_dict()
    jax.tree.map(np.testing.assert_array_equal, source_head_state, target_head_state)

    # ...but the frozen pi0 backbone was never touched by load_wrench_head.
    target_pi0_kernel_after = np.asarray(target_model.pi0.action_out_proj.kernel.value)
    np.testing.assert_array_equal(target_pi0_kernel_before, target_pi0_kernel_after)
    assert not np.allclose(source_pi0_kernel, target_pi0_kernel_after)


def test_sample_sequence_pool_and_next_batch_pairs_use_only_given_episodes():
    class _FakeDataset:
        episodes = {0: {"length": 5}, 1: {"length": 3}, 2: {"length": 10}}

    dataset = _FakeDataset()
    rng = np.random.default_rng(0)
    pool = sample_sequence_pool(dataset, [0, 1], size=20, rng=rng)

    assert len(pool) == 20
    for episode_index, frame_index in pool:
        assert episode_index in (0, 1)
        assert 0 <= frame_index < dataset.episodes[episode_index]["length"]

    fixed_batch = next_batch_pairs(dataset, [0, 1], batch_size=4, rng=rng, fixed_pool=pool)
    assert len(fixed_batch) == 4
    assert all(pair in pool for pair in fixed_batch)

    fresh_batch = next_batch_pairs(dataset, [2], batch_size=4, rng=rng)
    assert all(episode_index == 2 for episode_index, _ in fresh_batch)


def test_stack_examples_batches_leaf_arrays():
    examples = [
        {"state": np.zeros(3), "image": {"cam": np.zeros((2, 2, 3))}},
        {"state": np.ones(3), "image": {"cam": np.ones((2, 2, 3))}},
    ]
    batch = stack_examples(examples)
    assert batch["state"].shape == (2, 3)
    assert batch["image"]["cam"].shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(batch["state"][1], np.ones(3))


def test_alignment_config_from_yaml_rejects_unknown_keys(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"checkpoint": "x", "not_a_real_field": 1}))
    with pytest.raises(ValueError):
        AlignmentTrainConfig.from_yaml(
            bad_yaml, overrides={"dataset": "d", "split_path": "s", "output_dir": "o"}
        )


def test_alignment_config_from_yaml_parses_paths_and_defaults(tmp_path: Path):
    yaml_path = tmp_path / "overfit.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "checkpoint": "Facet-0/facet0-post-training",
                "dataset": "ManuFacet-1K/Facet0-1",
                "split_path": "configs/splits/facet0_1_seed20260903.json",
                "output_dir": "runs/overfit_v1",
                "mode": "overfit",
                "num_overfit_sequences": 32,
            }
        )
    )
    config = AlignmentTrainConfig.from_yaml(yaml_path)
    assert isinstance(config.checkpoint, Path)
    assert config.mode == "overfit"
    assert config.num_overfit_sequences == 32
    assert config.lambda_pre == pytest.approx(0.1)  # default carried through


@pytest.mark.skipif(not DATASET.exists(), reason="ManuFacet-1K/Facet0-1 not present")
@pytest.mark.skipif(not CHECKPOINT_ASSETS.exists(), reason="Facet-0 checkpoint assets not present")
def test_build_example_matches_production_pipeline_shapes():
    """Exercises the real data-transform chain (same one `create_facet_policy` uses for
    inference) at production scale (action_dim=32, action_horizon=50, max_token_len=200) --
    without restoring the 2B-parameter checkpoint, since `ModelTransformFactory`/`normalize.load`
    only need the small config/JSON assets, not the weights.
    """
    pi0_cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
    norm_stats = normalize.load(CHECKPOINT_ASSETS)
    model_input_transforms = list(training_config.ModelTransformFactory()(pi0_cfg).inputs)
    facet_inputs = FacetInputs()
    delta_actions = openpi_transforms.DeltaActions(EEF_DELTA_MASK)
    normalize_transform = openpi_transforms.Normalize(norm_stats, use_quantiles=True)
    wrench_normalizer = WrenchNormalizer.from_openpi_norm_stats(norm_stats)

    dataset = ManuFacetDataset(DATASET)
    example = build_example(
        dataset,
        episode_index=0,
        frame_index=5,
        facet_inputs=facet_inputs,
        delta_actions=delta_actions,
        normalize_transform=normalize_transform,
        model_input_transforms=model_input_transforms,
        wrench_normalizer=wrench_normalizer,
        wrench_target_representation="absolute",
        action_horizon=50,
        wrench_history_len=10,
    )

    assert example["state"].shape == (32,)
    assert example["actions"].shape == (50, 32)
    assert set(example["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for image in example["image"].values():
        assert image.shape == (224, 224, 3)
    assert example["tokenized_prompt"].ndim == 1
    assert example["tokenized_prompt_mask"].shape == example["tokenized_prompt"].shape
    assert example["wrench_history"].shape == (10, 6)
    assert example["wrench_history_valid"].shape == (10,)
    assert example["wrench_target"].shape == (50, 6)
    assert example["wrench_target_valid"].shape == (50,)
    assert np.all(np.isfinite(example["state"]))
    assert np.all(np.isfinite(example["actions"]))
    # The released checkpoint's state quantiles define the same roughly [-1, 1]
    # numerical regime used for action flow matching. Real outliers may exceed it.
    assert np.quantile(np.abs(example["wrench_history"]), 0.9) < 3.0


@pytest.mark.skipif(not (PROJECT_ROOT / "configs" / "splits").exists(), reason="no split file present")
def test_load_split_reads_the_real_split_file():
    from facet0.training.train_alignment import load_split

    split_files = list((PROJECT_ROOT / "configs" / "splits").glob("*.json"))
    assert split_files, "expected at least one split file"
    splits = load_split(split_files[0])
    assert set(splits) == {"train", "validation", "test"}
    assert all(isinstance(i, int) for i in splits["train"][:5])
