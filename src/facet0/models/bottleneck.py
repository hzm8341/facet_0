"""Experimental four-stream FACET bottleneck used by local adaptation."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class FacetBottleneck(nnx.Module):
    """Project four semantic/contact streams to 256 dims and concatenate to 1024."""

    def __init__(self, input_dims: tuple[int, int, int, int], *, rngs: nnx.Rngs):
        self.input_dims = input_dims
        self.projections = [nnx.Linear(dim, 256, rngs=rngs) for dim in input_dims]

    def __call__(self, streams: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        if len(streams) != 4:
            raise ValueError("FacetBottleneck requires exactly four streams")
        projected = []
        for stream, expected_dim, projection in zip(
            streams, self.input_dims, self.projections, strict=True
        ):
            if stream.shape[-1] != expected_dim:
                raise ValueError(f"expected stream width {expected_dim}, got {stream.shape[-1]}")
            projected.append(nnx.swish(projection(stream)))
        return jnp.concatenate(projected, axis=-1)
