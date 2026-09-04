"""Build an inference-only FACET policy from the released Orbax checkpoint."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.policies import policy as policy_lib
from openpi.shared import normalize
from openpi.training import config as training_config

from facet0.data.transforms import EEF_DELTA_MASK, FacetInputs, FacetOutputs


def create_facet_policy(
    checkpoint: str | Path,
    *,
    seed: int = 0,
    num_steps: int = 10,
    camera_mapping: dict[str, str] | None = None,
) -> policy_lib.Policy:
    """Restore the official checkpoint and attach the FACET input/output transforms."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    checkpoint = Path(checkpoint).resolve()
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )
    params = model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params, remove_extra_params=False)
    norm_stats = normalize.load(checkpoint / "assets")
    model_transforms = training_config.ModelTransformFactory()(config)
    facet_inputs = FacetInputs() if camera_mapping is None else FacetInputs(camera_mapping)

    policy = policy_lib.Policy(
        model,
        transforms=[
            facet_inputs,
            # Public parquet actions are absolute poses, while checkpoint action
            # statistics are centered delta poses for channels 0:6. The gripper
            # remains absolute.
            transforms.DeltaActions(EEF_DELTA_MASK),
            transforms.Normalize(norm_stats, use_quantiles=True),
            *model_transforms.inputs,
        ],
        output_transforms=[
            *model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            transforms.AbsoluteActions(EEF_DELTA_MASK),
            FacetOutputs(),
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata={
            "checkpoint": str(checkpoint),
            "model_type": "pi05",
            "camera_mapping_status": "inferred_from_public_names",
        },
    )
    # OpenPI commit 215abfb uses `rng or default_key` in Policy.__init__, which
    # asks Python to coerce a scalar JAX key to bool. Initialize with its default
    # and then set the requested deterministic key without patching upstream.
    policy._rng = jax.random.key(seed)  # noqa: SLF001
    return policy
