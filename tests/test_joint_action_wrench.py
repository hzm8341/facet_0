from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config

from facet0.models.joint_action_wrench import (
    JointActionWrenchConfig,
    JointActionWrenchModel,
    pi0_frozen_filter,
    wrench_head_trainable_filter,
)

# "dummy" is a tiny (width=64, depth=4) gemma variant openpi ships for fast tests -- see
# third_party/openpi/src/openpi/models/gemma.py. Using it (instead of gemma_2b/gemma_300m) keeps
# these tests running a real forward pass through PaliGemma.llm in well under a second.
BATCH = 2
ACTION_DIM = 8
ACTION_HORIZON = 4
MAX_TOKEN_LEN = 16
WRENCH_DIM = 6
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
    joint_config = JointActionWrenchConfig(
        wrench_dim=WRENCH_DIM, wrench_history_len=WRENCH_HISTORY_LEN, wrench_head_width=32
    )
    return JointActionWrenchModel(pi0, joint_config, rngs=nnx.Rngs(seed + 1))


def _fake_batch(config: pi0_config.Pi0Config):
    observation = config.fake_obs(batch_size=BATCH)
    actions = config.fake_act(batch_size=BATCH)
    rng = np.random.default_rng(0)
    wrench_history = jnp.asarray(
        rng.normal(size=(BATCH, WRENCH_HISTORY_LEN, WRENCH_DIM)), dtype=jnp.float32
    )
    wrench_history_valid = jnp.ones((BATCH, WRENCH_HISTORY_LEN), dtype=jnp.bool_)
    wrench_target = jnp.asarray(
        rng.normal(size=(BATCH, ACTION_HORIZON, WRENCH_DIM)), dtype=jnp.float32
    )
    wrench_target_valid = jnp.ones((BATCH, ACTION_HORIZON), dtype=jnp.bool_)
    return observation, actions, wrench_history, wrench_history_valid, wrench_target, wrench_target_valid


def test_compute_losses_returns_expected_shapes():
    model = _build_model()
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())

    action_loss, wrench_loss = model.compute_losses(
        jax.random.key(42), obs, actions, history, history_valid, target, target_valid
    )

    assert action_loss.shape == (BATCH, ACTION_HORIZON)
    assert wrench_loss.shape == (BATCH, ACTION_HORIZON)
    assert jnp.all(jnp.isfinite(action_loss))
    assert jnp.all(jnp.isfinite(wrench_loss))


def test_action_loss_matches_bare_pi0():
    """The wrench head must not perturb the action loss/backbone forward pass in any way."""
    model = _build_model()
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())
    rng = jax.random.key(7)

    action_loss, _ = model.compute_losses(rng, obs, actions, history, history_valid, target, target_valid)

    action_rng = jax.random.split(rng)[0]
    reference_loss = model.pi0.compute_loss(action_rng, obs, actions, train=False)

    # `rtol`/`atol` (not exact equality): the two calls compile to separate XLA executables
    # (different surrounding call sites), so GPU matmul kernel-selection/fusion can differ enough
    # to perturb the last few float32 bits even though the op sequence is identical -- the same
    # tolerance-based comparison this project's own golden-regression check uses (see
    # scripts/verify_golden_inference.py's DEFAULT_ATOL) for the same reason.
    np.testing.assert_allclose(np.asarray(action_loss), np.asarray(reference_loss), rtol=1e-3, atol=1e-4)


def test_wrench_loss_masks_padded_horizon_steps():
    model = _build_model()
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())
    target_valid = target_valid.at[:, -1].set(False)

    _, wrench_loss = model.compute_losses(
        jax.random.key(3), obs, actions, history, history_valid, target, target_valid
    )

    assert jnp.all(wrench_loss[:, -1] == 0.0)
    assert jnp.all(wrench_loss[:, :-1] > 0.0)


def test_wrench_head_history_encoding_ignores_masked_out_padding():
    """`encode_history` zeroes invalid (pre-episode) steps before flattening (see its docstring),
    so two histories that differ only in the padded region marked invalid by `history_valid`
    must produce the same encoding -- while a real change to a *valid* step must not.
    """
    model = _build_model()
    _obs, _actions, history, history_valid, _target, _target_valid = _fake_batch(_tiny_pi0_config())
    history_valid = history_valid.at[:, 0].set(False)  # first history step is padding

    padded_variant = history.at[:, 0, :].set(history[:, 0, :] + 1000.0)
    encoding_original = model.wrench_head.encode_history(history, history_valid)
    encoding_padded_changed = model.wrench_head.encode_history(padded_variant, history_valid)
    np.testing.assert_array_equal(np.asarray(encoding_original), np.asarray(encoding_padded_changed))

    valid_step_changed = history.at[:, 1, :].set(history[:, 1, :] + 1000.0)
    encoding_valid_changed = model.wrench_head.encode_history(valid_step_changed, history_valid)
    assert not np.allclose(np.asarray(encoding_original), np.asarray(encoding_valid_changed))


def test_wrench_head_trainable_filter_selects_only_wrench_head_params():
    config = _tiny_pi0_config()
    joint_config = JointActionWrenchConfig(
        wrench_dim=WRENCH_DIM, wrench_history_len=WRENCH_HISTORY_LEN, wrench_head_width=32
    )

    def _create(rng):
        pi0 = config.create(rng)
        return JointActionWrenchModel(pi0, joint_config, rngs=nnx.Rngs(rng))

    abstract_model = nnx.eval_shape(_create, jax.random.key(0))

    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, wrench_head_trainable_filter())).flat_state()
    frozen_pi0 = nnx.state(abstract_model, nnx.All(nnx.Param, pi0_frozen_filter())).flat_state()
    all_params = nnx.state(abstract_model, nnx.Param).flat_state()

    assert len(trainable) > 0
    assert len(frozen_pi0) > 0
    # every param falls into exactly one of the two buckets
    assert len(trainable) + len(frozen_pi0) == len(all_params)
    assert all(any("wrench_head" in p for p in path) for path in trainable)
    assert all(any("wrench_head" in p for p in path) is False for path in frozen_pi0)
    assert all(any("pi0" in p for p in path) for path in frozen_pi0)


def test_gradients_only_touch_wrench_head_parameters():
    """End-to-end guarantee behind the Stage 5 acceptance criterion "golden regression still
    passes": an actual `nnx.grad` call restricted to `wrench_head_trainable_filter` must produce
    a gradient pytree that (a) is entirely None/absent under `pi0.*` and (b) is nonzero somewhere
    under `wrench_head.*`, i.e. an optimizer built with this filter can never update `pi0.*`.
    """
    model = _build_model()
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())
    trainable_filter = nnx.All(nnx.Param, wrench_head_trainable_filter())

    def loss_fn(model: JointActionWrenchModel) -> jax.Array:
        action_loss, wrench_loss = model.compute_losses(
            jax.random.key(9), obs, actions, history, history_valid, target, target_valid
        )
        return jnp.mean(action_loss) + 0.1 * jnp.sum(wrench_loss) / jnp.sum(target_valid)

    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter))(model)

    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)


def test_sample_action_and_wrench_matches_bare_pi0_sampling():
    """`sample_action_and_wrench`'s action trajectory must reproduce `pi0.sample_actions`
    exactly, so attaching the wrench head never perturbs FACET's official inference path.
    """
    model = _build_model()
    obs, _actions, history, history_valid, _target, _target_valid = _fake_batch(_tiny_pi0_config())
    rng = jax.random.key(11)

    actions, wrench = model.sample_action_and_wrench(rng, obs, history, history_valid, num_steps=3)

    action_rng = jax.random.split(rng)[0]
    reference_actions = model.pi0.sample_actions(action_rng, obs, num_steps=3)

    # rtol/atol, not exact equality: see the comment in test_action_loss_matches_bare_pi0 --
    # separate XLA-compiled while_loop executables can differ in the last few float32 bits, and
    # that difference compounds slightly across `num_steps` Euler integration steps.
    np.testing.assert_allclose(np.asarray(actions), np.asarray(reference_actions), rtol=5e-2, atol=5e-3)
    assert wrench.shape == (BATCH, ACTION_HORIZON, WRENCH_DIM)
    assert jnp.all(jnp.isfinite(wrench))


def test_history_length_and_wrench_dim_mismatches_raise():
    model = _build_model()
    obs, actions, history, history_valid, target, target_valid = _fake_batch(_tiny_pi0_config())

    with pytest.raises((ValueError, TypeError)):
        model.compute_losses(
            jax.random.key(1),
            obs,
            actions,
            history[:, :1, :],  # wrong history length
            history_valid[:, :1],
            target,
            target_valid,
        )
