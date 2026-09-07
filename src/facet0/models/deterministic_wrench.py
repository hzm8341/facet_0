"""Deterministic future-wrench residual control model.

This is an experimental control for the stochastic flow head. A zero prediction means
"hold the current wrench", so its initialization exactly matches the causal constant baseline.
The demonstrated action chunk is an explicit condition during this offline learnability study;
deployment must replace it with the frozen FACET policy's predicted action chunk.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class DeterministicWrenchConfig:
    history_len: int = 10
    horizon: int = 50
    wrench_dim: int = 6
    action_dim: int = 7
    width: int = 256


class DeterministicResidualWrenchModel(nnx.Module):
    """Predict normalized future wrench deltas from causal history and an action chunk."""

    def __init__(self, config: DeterministicWrenchConfig, *, rngs: nnx.Rngs):
        self.config = config
        width = config.width
        self.history_in = nnx.Linear(config.wrench_dim, width, rngs=rngs)
        self.history_fuse = nnx.Linear(config.history_len * width, width, rngs=rngs)
        self.action_in = nnx.Linear(config.action_dim, width, rngs=rngs)
        self.fuse_in = nnx.Linear(2 * width + 1, width, rngs=rngs)
        self.fuse_out = nnx.Linear(width, width, rngs=rngs)
        self.output = nnx.Linear(
            width,
            config.wrench_dim,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        history: jax.Array,
        history_valid: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        config = self.config
        if history.shape[-2:] != (config.history_len, config.wrench_dim):
            raise ValueError(
                f"history must end in {(config.history_len, config.wrench_dim)}, got {history.shape}"
            )
        if actions.shape[-2:] != (config.horizon, config.action_dim):
            raise ValueError(
                f"actions must end in {(config.horizon, config.action_dim)}, got {actions.shape}"
            )
        masked_history = history * history_valid[..., None].astype(history.dtype)
        history_tokens = nnx.swish(self.history_in(masked_history))
        history_condition = nnx.swish(self.history_fuse(history_tokens.reshape(history.shape[0], -1)))
        history_condition = jnp.broadcast_to(
            history_condition[:, None, :],
            (history.shape[0], config.horizon, config.width),
        )
        action_tokens = nnx.swish(self.action_in(actions))
        position = jnp.linspace(0.0, 1.0, config.horizon, dtype=history.dtype)
        position = jnp.broadcast_to(position[None, :, None], (*actions.shape[:-1], 1))
        hidden = jnp.concatenate([history_condition, action_tokens, position], axis=-1)
        hidden = nnx.swish(self.fuse_in(hidden))
        hidden = nnx.swish(self.fuse_out(hidden))
        return self.output(hidden)


def save_deterministic_wrench(model: DeterministicResidualWrenchModel, path) -> None:
    state = nnx.state(model)
    flat = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(value) for key, value in flat.items()})


def load_deterministic_wrench(model: DeterministicResidualWrenchModel, path) -> None:
    with np.load(path) as data:
        flat = {tuple(key.split("/")): jnp.asarray(value) for key, value in data.items()}
    pure = traverse_util.unflatten_dict(flat)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pure)
    nnx.update(model, state)
