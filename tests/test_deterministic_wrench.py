from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import optax

from facet0.data.wrench_control import WrenchControlPool, stratify_by_change
from facet0.models.deterministic_wrench import (
    DeterministicResidualWrenchModel,
    DeterministicWrenchConfig,
    load_deterministic_wrench,
    save_deterministic_wrench,
)


def _inputs(config: DeterministicWrenchConfig):
    rng = np.random.default_rng(0)
    history = jnp.asarray(rng.normal(size=(4, config.history_len, config.wrench_dim)))
    history_valid = jnp.ones((4, config.history_len), dtype=jnp.bool_)
    actions = jnp.asarray(rng.normal(size=(4, config.horizon, config.action_dim)))
    return history, history_valid, actions


def test_zero_initialization_exactly_matches_residual_baseline():
    config = DeterministicWrenchConfig(history_len=3, horizon=5, width=16)
    model = DeterministicResidualWrenchModel(config, rngs=nnx.Rngs(0))
    prediction = model(*_inputs(config))
    np.testing.assert_array_equal(prediction, np.zeros((4, 5, 6), dtype=np.float32))


def test_fixed_batch_huber_training_reduces_loss():
    config = DeterministicWrenchConfig(history_len=3, horizon=5, width=32)
    model = DeterministicResidualWrenchModel(config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    history, history_valid, actions = _inputs(config)
    target = jnp.broadcast_to(jnp.linspace(0.0, 0.5, config.horizon)[None, :, None], (4, 5, 6))

    @nnx.jit
    def step(opt):
        def loss_fn(m):
            return optax.huber_loss(m(history, history_valid, actions), target, delta=0.1).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
        opt.update(grads)
        return loss

    first = float(step(optimizer))
    for _ in range(99):
        last = float(step(optimizer))
    assert last < 0.1 * first


def test_checkpoint_round_trip(tmp_path):
    config = DeterministicWrenchConfig(history_len=3, horizon=5, width=16)
    source = DeterministicResidualWrenchModel(config, rngs=nnx.Rngs(0))
    path = tmp_path / "head.npz"
    save_deterministic_wrench(source, path)
    restored = DeterministicResidualWrenchModel(config, rngs=nnx.Rngs(99))
    load_deterministic_wrench(restored, path)
    np.testing.assert_array_equal(restored(*_inputs(config)), source(*_inputs(config)))


def test_change_stratification_selects_high_scores():
    count = 10
    pool = WrenchControlPool(
        pairs=np.stack([np.arange(count), np.zeros(count)], axis=-1),
        history=np.zeros((count, 2, 6)),
        history_valid=np.ones((count, 2), dtype=bool),
        actions=np.zeros((count, 3, 7)),
        target=np.zeros((count, 3, 6)),
        target_valid=np.ones((count, 3), dtype=bool),
        change_score=np.arange(count, dtype=np.float32),
    )
    selected = stratify_by_change(
        pool,
        output_size=4,
        high_change_fraction=0.5,
        rng=np.random.default_rng(0),
    )
    assert len(selected) == 4
    assert {8.0, 9.0}.issubset(set(selected.change_score))
