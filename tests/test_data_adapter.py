from __future__ import annotations

from pathlib import Path

import numpy as np

from openpi import transforms

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.transforms import EEF_DELTA_MASK, FacetInputs, FacetOutputs


DATASET = Path(__file__).resolve().parents[1] / "ManuFacet-1K" / "Facet0-1"


def test_facet_inputs_shapes_and_camera_mapping():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    transformed = FacetInputs()(
        {
            "observation.images": {"view1": image, "hand": image, "view2": image},
            "observation.state": np.arange(13, dtype=np.float32),
            "action": np.arange(7, dtype=np.float32),
            "prompt": "install the ram",
        }
    )
    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert transformed["state"].shape == (13,)
    assert transformed["actions"].shape == (7,)
    assert transformed["state"][6] == -5.0
    assert transformed["actions"][6] == -5.0
    assert all(transformed["image_mask"].values())


def test_facet_outputs_keeps_only_executed_channels():
    actions = np.arange(2 * 50 * 32).reshape(2, 50, 32)
    output = FacetOutputs()({"actions": actions})
    np.testing.assert_array_equal(output["actions"], actions[..., :7])


def test_public_episode_signals_and_chunk_padding():
    dataset = ManuFacetDataset(DATASET)
    assert len(dataset) == 2622
    sample = dataset.load_frame(0, 0, include_images=False)
    assert sample["observation.state"].shape == (13,)
    assert sample["action"].shape == (7,)
    assert sample["prompt"].startswith("Overall instruction:")

    last_frame = dataset.episodes[0]["length"] - 1
    actions, valid = dataset.load_action_chunk(0, last_frame, horizon=50)
    assert actions.shape == (50, 7)
    assert valid.shape == (50,)
    assert valid.sum() == 1
    np.testing.assert_array_equal(actions[0], actions[-1])


def test_public_video_path_uses_full_feature_key():
    dataset = ManuFacetDataset(DATASET)
    path = dataset.video_path(0, "view1")
    assert path.parent.name == "observation.images.view1"
    assert path.is_file()


def test_absolute_delta_action_round_trip():
    state = np.array([1, 2, 3, 4, 5, 6, 0.25], dtype=np.float32)
    actions = np.array([[2, 4, 6, 8, 10, 12, 0.75]], dtype=np.float32)
    sample = {"state": state.copy(), "actions": actions.copy()}
    sample = transforms.DeltaActions(EEF_DELTA_MASK)(sample)
    np.testing.assert_array_equal(sample["actions"][0, :6], [1, 2, 3, 4, 5, 6])
    assert sample["actions"][0, 6] == 0.75
    sample = transforms.AbsoluteActions(EEF_DELTA_MASK)(sample)
    np.testing.assert_allclose(sample["actions"], actions)
