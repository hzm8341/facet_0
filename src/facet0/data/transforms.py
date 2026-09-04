"""Translate FACET observations to and from the OpenPI model schema."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib


STATE_DIM = 13
EXECUTED_ACTION_DIM = 7
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 50
EEF_DELTA_MASK = (True, True, True, True, True, True, False)

DEFAULT_CAMERA_MAPPING = {
    "view1": "base_0_rgb",
    "hand": "left_wrist_0_rgb",
    "view2": "right_wrist_0_rgb",
}


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        # LeRobot commonly exposes decoded images as float32 in [0, 1].
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _validate_last_dim(name: str, value: np.ndarray, expected: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != expected:
        raise ValueError(f"{name} must end in dimension {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def public_to_model_gripper(value: np.ndarray) -> np.ndarray:
    """Convert public continuous opening (larger=open) to model closure (larger=closed)."""
    array = np.asarray(value, dtype=np.float32).copy()
    array[..., 6] = 1.0 - array[..., 6]
    return array


@dataclasses.dataclass(frozen=True)
class FacetInputs(transforms.DataTransformFn):
    """Map public ManuFacet fields to the three-image OpenPI pi0.5 schema.

    The camera mapping is the best current inference from public field names and
    remains configurable until the FACET authors publish the exact mapping.
    """

    camera_mapping: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CAMERA_MAPPING)
    )

    def __post_init__(self) -> None:
        expected_targets = set(model_lib.IMAGE_KEYS)
        if set(self.camera_mapping.values()) != expected_targets:
            raise ValueError(
                "camera_mapping targets must contain each OpenPI image key exactly once: "
                f"{sorted(expected_targets)}"
            )

    def __call__(self, data: dict) -> dict:
        raw_images = data.get("observation.images")
        if not isinstance(raw_images, Mapping):
            raise ValueError("Missing observation.images mapping")

        missing = sorted(set(self.camera_mapping) - set(raw_images))
        if missing:
            raise ValueError(f"Missing FACET camera images: {missing}")

        images = {
            target: _parse_image(raw_images[source])
            for source, target in self.camera_mapping.items()
        }
        state = public_to_model_gripper(
            _validate_last_dim("observation.state", data["observation.state"], STATE_DIM)
        )

        output = {
            "state": state,
            "image": images,
            "image_mask": {key: np.True_ for key in images},
        }
        if "action" in data:
            output["actions"] = public_to_model_gripper(
                _validate_last_dim("action", data["action"], EXECUTED_ACTION_DIM)
            )
        if "prompt" in data:
            prompt = data["prompt"]
            output["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return output


@dataclasses.dataclass(frozen=True)
class FacetOutputs(transforms.DataTransformFn):
    """Return only the seven FACET channels that are executed on the robot."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] < EXECUTED_ACTION_DIM:
            raise ValueError(f"Model action dimension is too small: {actions.shape}")
        return {**data, "actions": actions[..., :EXECUTED_ACTION_DIM]}
