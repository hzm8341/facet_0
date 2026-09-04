"""Independent 6-dim wrench flow-matching head wrapping a frozen Pi0 action policy.

Per HANDOFF.md Sec 10.2 / FACET0_REPRODUCTION_PLAN.md Stage 5: the official 7-dim action
policy (`Pi0.action_in_proj` / `Pi0.action_out_proj` / the shared PaliGemma+action-expert
transformer) must stay byte-for-bit unchanged. This module never edits `third_party/openpi`
source and never mutates a `Pi0` instance's parameters; it only *reads* `Pi0.embed_prefix`,
`Pi0.embed_suffix`, `Pi0.PaliGemma.llm`, and `Pi0.action_out_proj` -- the exact same calls
`Pi0.compute_loss` makes, in the exact same order, on the exact same inputs -- so the action
loss/vector field this module produces is identical to calling `Pi0.compute_loss` directly.
`Pi0.sample_actions` (the golden-regression inference path) never imports this module at all.

The new `WrenchHead` reads the shared backbone's suffix hidden states through
`jax.lax.stop_gradient`, so no gradient can flow from the wrench loss into the backbone even
if a caller forgets to freeze it via `wrench_head_trainable_filter`.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

WRENCH_DIM = 6


@dataclasses.dataclass(frozen=True)
class JointActionWrenchConfig:
    """Wrench head architecture hyperparameters.

    `wrench_dim` and `wrench_history_len` follow FACET0_REPRODUCTION_PLAN.md Stage 5 (`paper`:
    K=10 causal history steps, 6-dim wrench). `wrench_head_width` (and the head's MLP shapes,
    which are derived from it) are not specified by the paper and are `experimental`.
    """

    wrench_dim: int = WRENCH_DIM  # paper
    wrench_history_len: int = 10  # paper (K=10)
    wrench_head_width: int = 512  # experimental


class WrenchHead(nnx.Module):
    """Predicts a `[action_horizon, wrench_dim]` flow-matching vector field.

    Conditions on (a) the frozen backbone's suffix hidden states (stop-gradient) and (b) an
    encoding of the causal wrench history window from `facet0.data.wrench_windows`. Structurally
    mirrors `Pi0`'s non-pi05 action-time MLP (concat + 2-layer MLP) for consistency with the
    rest of the codebase; this fusion shape is `experimental`, not paper-specified.
    """

    def __init__(
        self,
        config: JointActionWrenchConfig,
        *,
        backbone_width: int,
        action_horizon: int,
        rngs: nnx.Rngs,
    ):
        width = config.wrench_head_width
        self.wrench_dim = config.wrench_dim
        self.wrench_history_len = config.wrench_history_len
        self.action_horizon = action_horizon
        self.width = width

        self.backbone_cond_proj = nnx.Linear(backbone_width, width, rngs=rngs)
        self.history_in_proj = nnx.Linear(config.wrench_dim, width, rngs=rngs)
        self.history_encoder = nnx.Linear(config.wrench_history_len * width, width, rngs=rngs)
        self.wrench_in_proj = nnx.Linear(config.wrench_dim, width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(width, width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(width, width, rngs=rngs)
        self.fuse_mlp_in = nnx.Linear(4 * width, width, rngs=rngs)
        self.fuse_mlp_out = nnx.Linear(width, width, rngs=rngs)
        self.wrench_out_proj = nnx.Linear(width, config.wrench_dim, rngs=rngs)

    def encode_history(self, history: jax.Array, history_valid: jax.Array) -> jax.Array:
        """Flatten the causal wrench history into a single conditioning vector.

        ``history``: float ``[b, k, wrench_dim]``. ``history_valid``: bool ``[b, k]``. Returns
        float ``[b, width]``.

        Invalid (pre-episode) steps are zeroed before flattening rather than dropped, so the
        encoder always sees a fixed-length `[k, width]` input regardless of how much of the
        history window fell before frame 0.
        """
        masked = history * history_valid[..., None].astype(history.dtype)
        tokens = self.history_in_proj(masked)
        flat = tokens.reshape(tokens.shape[0], -1)
        return nnx.swish(self.history_encoder(flat))

    def __call__(
        self,
        backbone_hidden: jax.Array,
        history: jax.Array,
        history_valid: jax.Array,
        noisy_wrench: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        """``backbone_hidden``: float ``[b, ah, backbone_width]``. ``history``/``history_valid``:
        as in ``encode_history``. ``noisy_wrench``: float ``[b, ah, wrench_dim]``. ``timestep``:
        float ``[b]``. Returns the predicted flow vector field, float ``[b, ah, wrench_dim]``.
        """
        backbone_cond = self.backbone_cond_proj(jax.lax.stop_gradient(backbone_hidden))
        history_cond = self.encode_history(history, history_valid)

        wrench_tokens = self.wrench_in_proj(noisy_wrench)
        time_emb = posemb_sincos(timestep, self.width, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        time_emb = self.time_mlp_out(time_emb)

        history_tokens = jnp.broadcast_to(history_cond[:, None, :], wrench_tokens.shape)
        time_tokens = jnp.broadcast_to(time_emb[:, None, :], wrench_tokens.shape)

        fused = jnp.concatenate([wrench_tokens, backbone_cond, history_tokens, time_tokens], axis=-1)
        hidden = nnx.swish(self.fuse_mlp_in(fused))
        hidden = self.fuse_mlp_out(hidden)
        return self.wrench_out_proj(hidden)


def wrench_head_trainable_filter() -> nnx.filterlib.Filter:
    """Selects only `WrenchHead` parameters. Pass as `nnx.Optimizer(..., wrt=...)` and
    `nnx.grad(..., argnums=nnx.DiffState(0, ...))` so gradients/updates never reach `Pi0`.
    """
    return nnx_utils.PathRegex(".*wrench_head.*")


def pi0_frozen_filter() -> nnx.filterlib.Filter:
    """Complement of `wrench_head_trainable_filter`; used in tests to assert the entire Pi0
    backbone is excluded from the trainable set.
    """
    return nnx_utils.PathRegex(".*pi0.*")


class JointActionWrenchModel(nnx.Module):
    """Composes a frozen `Pi0` action policy with an independent, trainable `WrenchHead`.

    `self.pi0` is held by reference, never edited. Training code must freeze it (e.g. via
    `nnx.Optimizer(model, tx, wrt=nnx.All(nnx.Param, wrench_head_trainable_filter()))`) so
    only `self.wrench_head`'s parameters ever receive an optimizer update.
    """

    def __init__(self, pi0: Pi0, config: JointActionWrenchConfig, rngs: nnx.Rngs):
        self.pi0 = pi0
        self.config = config
        backbone_width = pi0.action_out_proj.in_features
        self.wrench_head = WrenchHead(
            config,
            backbone_width=backbone_width,
            action_horizon=pi0.action_horizon,
            rngs=rngs,
        )

    def compute_losses(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        wrench_history: jax.Array,
        wrench_history_valid: jax.Array,
        wrench_target: jax.Array,
        wrench_target_valid: jax.Array,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns `(action_loss, wrench_loss)`, both per-horizon-step like `Pi0.compute_loss`.

        ``wrench_history``: float ``[b, wrench_history_len, wrench_dim]``. ``wrench_history_valid``:
        bool ``[b, wrench_history_len]``. ``wrench_target``: float ``[b, action_horizon, wrench_dim]``.
        ``wrench_target_valid``: bool ``[b, action_horizon]``. Returns two float ``[b, action_horizon]``
        arrays.

        `action_loss` reproduces `Pi0.compute_loss(action_rng, ...)` exactly bit-for-bit, where
        `action_rng = jax.random.split(rng)[0]` -- i.e. calling this with `rng` draws the same
        action noise/time and the same backbone forward pass as calling
        `pi0.compute_loss(jax.random.split(rng)[0], observation, actions, train=train)` directly
        would (see `test_joint_action_wrench.py::test_action_loss_matches_bare_pi0`). The
        resulting suffix hidden states are reused as the wrench head's conditioning context
        instead of paying for a second backbone forward pass. `wrench_loss` entries for padded
        (post-episode-end) horizon steps are zeroed via `wrench_target_valid`; callers must
        divide by `wrench_target_valid.sum()` (not `wrench_target_valid.size`) to get a
        correctly masked mean.
        """
        pi0 = self.pi0
        action_rng, wrench_rng = jax.random.split(rng)
        preprocess_rng, action_noise_rng, action_time_rng = jax.random.split(action_rng, 3)
        wrench_noise_rng, wrench_time_rng = jax.random.split(wrench_rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        action_noise = jax.random.normal(action_noise_rng, actions.shape)
        action_time = jax.random.beta(action_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        action_time_expanded = action_time[..., None, None]
        action_x_t = action_time_expanded * action_noise + (1 - action_time_expanded) * actions
        action_u_t = action_noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = pi0.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pi0.embed_suffix(
            observation, action_x_t, action_time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = pi0.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        backbone_hidden = suffix_out[:, -pi0.action_horizon :]
        action_v_t = pi0.action_out_proj(backbone_hidden)
        action_loss = jnp.mean(jnp.square(action_v_t - action_u_t), axis=-1)

        wrench_noise = jax.random.normal(wrench_noise_rng, wrench_target.shape)
        wrench_time = jax.random.beta(wrench_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        wrench_time_expanded = wrench_time[..., None, None]
        wrench_x_t = wrench_time_expanded * wrench_noise + (1 - wrench_time_expanded) * wrench_target
        wrench_u_t = wrench_noise - wrench_target
        wrench_v_t = self.wrench_head(
            backbone_hidden, wrench_history, wrench_history_valid, wrench_x_t, wrench_time
        )
        wrench_loss = jnp.mean(jnp.square(wrench_v_t - wrench_u_t), axis=-1)
        wrench_loss = wrench_loss * wrench_target_valid.astype(wrench_loss.dtype)

        return action_loss, wrench_loss

    def sample_action_and_wrench(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        wrench_history: jax.Array,
        wrench_history_valid: jax.Array,
        *,
        num_steps: int = 10,
    ) -> tuple[jax.Array, jax.Array]:
        """Jointly Euler-integrates the frozen action flow field and the wrench head's flow
        field from noise, sharing one prefix KV-cache and, at every step, the same suffix
        backbone hidden state -- mirroring how `compute_losses` derives both losses from one
        shared forward pass per training step.

        This reproduces `Pi0.sample_actions` for the action trajectory bit-for-bit: calling this
        with `rng` draws the same action noise and the same step-by-step backbone forward passes
        as `pi0.sample_actions(jax.random.split(rng)[0], observation, num_steps=num_steps,
        noise=None)` would (see `test_action_matches_bare_pi0_sampling`). The wrench trajectory
        this produces is `experimental`: the paper does not specify a wrench sampling procedure,
        only the training-time auxiliary loss.

        Returns `(actions, wrench)`, both float `[b, action_horizon, dim]`.
        """
        pi0 = self.pi0
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_rng, wrench_rng = jax.random.split(rng)
        action_noise = jax.random.normal(action_rng, (batch_size, pi0.action_horizon, pi0.action_dim))
        wrench_noise = jax.random.normal(
            wrench_rng, (batch_size, pi0.action_horizon, self.wrench_head.wrench_dim)
        )

        prefix_tokens, prefix_mask, prefix_ar_mask = pi0.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = pi0.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        def step(carry):
            action_x_t, wrench_x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pi0.embed_suffix(
                observation, action_x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_prefix_out, suffix_out), _ = pi0.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            backbone_hidden = suffix_out[:, -pi0.action_horizon :]
            action_v_t = pi0.action_out_proj(backbone_hidden)
            wrench_v_t = self.wrench_head(
                backbone_hidden,
                wrench_history,
                wrench_history_valid,
                wrench_x_t,
                jnp.broadcast_to(time, batch_size),
            )
            return action_x_t + dt * action_v_t, wrench_x_t + dt * wrench_v_t, time + dt

        def cond(carry):
            _action_x_t, _wrench_x_t, time = carry
            return time >= -dt / 2

        action_0, wrench_0, _time = jax.lax.while_loop(cond, step, (action_noise, wrench_noise, 1.0))
        return action_0, wrench_0
