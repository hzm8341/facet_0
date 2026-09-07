import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from facet0.models.bottleneck import FacetBottleneck
from facet0.models.local_td3 import BoundedActor, LocalTD3Config, TwinCritic, polyak_average
from facet0.training.train_local_adaptation import actor_td3_bc_loss, td3_bootstrap_target


def test_bottleneck_outputs_four_times_256():
    model = FacetBottleneck((4, 5, 6, 7), rngs=nnx.Rngs(0))
    output = model(tuple(jnp.zeros((2, dim)) for dim in (4, 5, 6, 7)))
    assert output.shape == (2, 1024)


def test_actor_bounds_and_twin_critic_shapes():
    config = LocalTD3Config(
        embedding_dim=16,
        width=32,
        action_low=(-0.1,) * 7,
        action_high=(0.2,) * 7,
    )
    actor = BoundedActor(config, rngs=nnx.Rngs(0))
    action = np.asarray(actor(jnp.zeros((3, 16))))
    assert action.shape == (3, 7)
    assert np.all(action >= -0.1) and np.all(action <= 0.2)
    critic = TwinCritic(config, rngs=nnx.Rngs(1))
    q1, q2 = critic(jnp.zeros((3, 16)), jnp.asarray(action))
    assert q1.shape == q2.shape == (3,)


def test_polyak_and_td3_targets():
    averaged = polyak_average({"x": np.asarray([0.0])}, {"x": np.asarray([2.0])}, 0.25)
    np.testing.assert_allclose(averaged["x"], [0.5])
    with pytest.raises(ValueError):
        polyak_average({"x": np.asarray([0.0])}, {"x": np.asarray([1.0])}, 2.0)
    target = td3_bootstrap_target(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([3.0, 3.0]),
        np.asarray([2.0, 2.0]),
        discount=0.9,
    )
    np.testing.assert_allclose(target, [2.8, 1.0])
    assert np.isfinite(
        actor_td3_bc_loss(
            np.asarray([1.0]),
            np.zeros((1, 7)),
            np.ones((1, 7)),
            bc_weight=0.1,
        )
    )
