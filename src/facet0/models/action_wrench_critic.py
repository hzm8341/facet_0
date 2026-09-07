"""Experimental distributional Action-Wrench Critic for offline proxy tests."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    input_dim: int = 1024
    width: int = 256
    num_return_bins: int = 51


class ActionWrenchCritic(nnx.Module):
    """Distributional return head plus success, efficiency, violation and recovery heads."""

    def __init__(self, config: CriticConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.in_proj = nnx.Linear(config.input_dim, config.width, rngs=rngs)
        self.hidden = nnx.Linear(config.width, config.width, rngs=rngs)
        self.return_logits = nnx.Linear(config.width, config.num_return_bins, rngs=rngs)
        self.success_logit = nnx.Linear(config.width, 1, rngs=rngs)
        self.efficiency = nnx.Linear(config.width, 1, rngs=rngs)
        self.violation_logit = nnx.Linear(config.width, 1, rngs=rngs)
        self.recovery_logit = nnx.Linear(config.width, 1, rngs=rngs)

    def __call__(self, embedding: jax.Array) -> dict[str, jax.Array]:
        hidden = nnx.swish(self.in_proj(embedding))
        hidden = nnx.swish(self.hidden(hidden))
        return {
            "return_logits": self.return_logits(hidden),
            "success_logit": self.success_logit(hidden)[..., 0],
            "efficiency": self.efficiency(hidden)[..., 0],
            "violation_logit": self.violation_logit(hidden)[..., 0],
            "recovery_logit": self.recovery_logit(hidden)[..., 0],
        }
