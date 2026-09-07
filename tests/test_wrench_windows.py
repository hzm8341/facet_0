from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from facet0.data.manufacet import ManuFacetDataset
from facet0.data.wrench_windows import (
    WRENCH_DIM,
    WrenchNormalizer,
    build_wrench_window,
    causal_wrench_window,
    extract_wrench,
    has_future_leakage,
)


DATASET = Path(__file__).resolve().parents[1] / "ManuFacet-1K" / "Facet0-1"


def _fake_wrench(num_frames: int) -> np.ndarray:
    # Each frame's wrench equals its own frame index broadcast across channels, so any
    # read-back value directly reveals which frame index it came from.
    return np.tile(np.arange(num_frames, dtype=np.float32)[:, None], (1, WRENCH_DIM))


def test_extract_wrench_slices_last_six_dims():
    state = np.arange(13, dtype=np.float32)
    np.testing.assert_array_equal(extract_wrench(state), state[7:13])


def test_extract_wrench_rejects_short_state():
    with pytest.raises(ValueError):
        extract_wrench(np.zeros(12, dtype=np.float32))


def test_wrench_normalizer_round_trip_and_broadcasting():
    normalizer = WrenchNormalizer(
        q01=np.asarray([-10, -20, -30, -1, -2, -3], dtype=np.float32),
        q99=np.asarray([10, 20, 30, 1, 2, 3], dtype=np.float32),
    )
    wrench = np.asarray(
        [[-10, -20, -30, -1, -2, -3], [10, 20, 30, 1, 2, 3]],
        dtype=np.float32,
    )
    normalized = normalizer.normalize(wrench)
    np.testing.assert_allclose(normalized[0], -1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1], 1.0, atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(normalized), wrench, atol=2e-6)
    delta = wrench[1] - wrench[0]
    np.testing.assert_allclose(
        normalizer.unnormalize_delta(normalizer.normalize_delta(delta)), delta, atol=2e-6
    )


def test_causal_wrench_window_mid_episode_has_no_padding():
    wrench = _fake_wrench(20)
    window = causal_wrench_window(wrench, frame_index=15, history_len=10, horizon=5)

    assert window.history.shape == (10, WRENCH_DIM)
    assert window.history_valid.all()
    np.testing.assert_array_equal(window.history_frame_indices, np.arange(6, 16))
    np.testing.assert_array_equal(window.history[:, 0], np.arange(6, 16))

    assert window.future.shape == (5, WRENCH_DIM)
    assert window.future_valid.all()
    np.testing.assert_array_equal(window.future_frame_indices, np.arange(15, 20))
    np.testing.assert_array_equal(window.future[:, 0], np.arange(15, 20))


def test_causal_wrench_window_pads_history_at_episode_start():
    wrench = _fake_wrench(20)
    window = causal_wrench_window(wrench, frame_index=0, history_len=10, horizon=5)

    np.testing.assert_array_equal(window.history_valid, [False] * 9 + [True])
    np.testing.assert_array_equal(window.history_frame_indices, [-1] * 9 + [0])
    # Padded entries repeat the earliest real frame (load_action_chunk's tail-padding
    # convention, mirrored here for the start of the episode).
    assert np.all(window.history[:, 0] == 0.0)


def test_causal_wrench_window_pads_future_at_episode_end():
    wrench = _fake_wrench(20)
    window = causal_wrench_window(wrench, frame_index=19, history_len=10, horizon=5)

    np.testing.assert_array_equal(window.future_valid, [True] + [False] * 4)
    np.testing.assert_array_equal(window.future_frame_indices, [19, -1, -1, -1, -1])
    assert np.all(window.future[:, 0] == 19.0)


def test_causal_wrench_window_history_never_references_future_frames():
    wrench = _fake_wrench(37)
    for frame_index in (0, 1, 9, 10, 20, 36):
        window = causal_wrench_window(wrench, frame_index, history_len=10, horizon=8)
        assert not has_future_leakage(window, frame_index)
        valid_indices = window.history_frame_indices[window.history_valid]
        assert valid_indices.max() <= frame_index


def test_has_future_leakage_detects_a_corrupted_window():
    wrench = _fake_wrench(20)
    window = causal_wrench_window(wrench, frame_index=10, history_len=10, horizon=5)
    # Simulate a bug that lets a future frame slip into the causal history.
    corrupted_indices = np.where(window.history_valid, 11, -1)
    leaking = dataclasses.replace(window, history_frame_indices=corrupted_indices)

    assert has_future_leakage(leaking, frame_index=10)


def test_causal_wrench_window_rejects_out_of_range_frame_index():
    wrench = _fake_wrench(5)
    with pytest.raises(IndexError):
        causal_wrench_window(wrench, frame_index=5, history_len=3, horizon=2)
    with pytest.raises(IndexError):
        causal_wrench_window(wrench, frame_index=-1, history_len=3, horizon=2)


def test_causal_wrench_window_rejects_bad_shapes_and_lengths():
    wrench = _fake_wrench(5)
    with pytest.raises(ValueError):
        causal_wrench_window(wrench, 0, history_len=0, horizon=2)
    with pytest.raises(ValueError):
        causal_wrench_window(wrench, 0, history_len=2, horizon=0)
    with pytest.raises(ValueError):
        causal_wrench_window(wrench[:, :5], 0, history_len=2, horizon=2)


def test_build_wrench_window_matches_public_dataset():
    dataset = ManuFacetDataset(DATASET)
    frame_index = 5
    window = build_wrench_window(dataset, 0, frame_index, history_len=10, horizon=50)

    assert window.history.shape == (10, WRENCH_DIM)
    assert window.future.shape == (50, WRENCH_DIM)

    sample = dataset.load_frame(0, frame_index, include_images=False)
    expected_current = extract_wrench(sample["observation.state"])
    np.testing.assert_allclose(window.history[-1], expected_current)
    np.testing.assert_allclose(window.future[0], expected_current)
    assert not has_future_leakage(window, frame_index)


def test_build_wrench_window_pads_at_episode_boundaries():
    dataset = ManuFacetDataset(DATASET)
    last_frame = dataset.episodes[0]["length"] - 1

    start_window = build_wrench_window(dataset, 0, 0, history_len=10, horizon=50)
    assert start_window.history_valid.sum() == 1
    assert not has_future_leakage(start_window, 0)

    end_window = build_wrench_window(dataset, 0, last_frame, history_len=10, horizon=50)
    assert end_window.future_valid.sum() == 1
    assert not has_future_leakage(end_window, last_frame)
