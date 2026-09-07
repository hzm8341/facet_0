"""Experimental bounded actor and twin critics for local TD3 adaptation."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class LocalTD3Config:
    embedding_dim: int = 1024
    action_dim: int = 7
    width: int = 256
    action_low: tuple[float, ...] = (-1.0,) * 7
    action_high: tuple[float, ...] = (1.0,) * 7


class BoundedActor(nnx.Module):
    def __init__(self, config: LocalTD3Config, *, rngs: nnx.Rngs):
        self.config = config
        self.in_proj = nnx.Linear(config.embedding_dim, config.width, rngs=rngs)
        self.hidden = nnx.Linear(config.width, config.width, rngs=rngs)
        self.output = nnx.Linear(config.width, config.action_dim, rngs=rngs)

    def __call__(self, embedding: jax.Array) -> jax.Array:
        hidden = nnx.swish(self.in_proj(embedding))
        hidden = nnx.swish(self.hidden(hidden))
        unit = jnp.tanh(self.output(hidden))
        low = jnp.asarray(self.config.action_low, dtype=unit.dtype)
        high = jnp.asarray(self.config.action_high, dtype=unit.dtype)
        return low + (unit + 1.0) * 0.5 * (high - low)


class _Critic(nnx.Module):
    def __init__(self, config: LocalTD3Config, *, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(config.embedding_dim + config.action_dim, config.width, rngs=rngs)
        self.hidden = nnx.Linear(config.width, config.width, rngs=rngs)
        self.output = nnx.Linear(config.width, 1, rngs=rngs)

    def __call__(self, embedding: jax.Array, action: jax.Array) -> jax.Array:
        hidden = nnx.swish(self.in_proj(jnp.concatenate([embedding, action], axis=-1)))
        hidden = nnx.swish(self.hidden(hidden))
        return self.output(hidden)[..., 0]


class TwinCritic(nnx.Module):
    def __init__(self, config: LocalTD3Config, *, rngs: nnx.Rngs):
        self.q1 = _Critic(config, rngs=rngs)
        self.q2 = _Critic(config, rngs=rngs)

    def __call__(self, embedding: jax.Array, action: jax.Array):
        return self.q1(embedding, action), self.q2(embedding, action)


def polyak_average(target, source, tau: float):
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    return jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)
