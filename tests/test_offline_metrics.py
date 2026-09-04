from __future__ import annotations

import numpy as np
import pytest

from facet0.evaluation import offline_metrics as metrics


def test_per_channel_mae_and_rmse_on_hand_constructed_example():
    pred = np.array([[1.0, 0.0], [3.0, 0.0]])
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    # errors: channel0 -> [1, 3], channel1 -> [0, 0]
    np.testing.assert_allclose(metrics.per_channel_mae(pred, target), [2.0, 0.0])
    np.testing.assert_allclose(metrics.per_channel_rmse(pred, target), [np.sqrt((1 + 9) / 2), 0.0])


def test_per_channel_mae_respects_valid_mask():
    pred = np.array([[10.0], [1.0], [1.0]])
    target = np.array([[0.0], [0.0], [0.0]])
    valid = np.array([False, True, True])
    # the huge error at index 0 must be excluded
    np.testing.assert_allclose(metrics.per_channel_mae(pred, target, valid), [1.0])


def test_translation_and_rotation_error_are_euclidean_distance():
    pred = np.array([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = np.zeros((1, 7))
    assert metrics.translation_error(pred, target) == pytest.approx(5.0)
    assert metrics.rotation_error(pred, target) == pytest.approx(0.0)


def test_gripper_accuracy_and_mae():
    pred = np.array([[0, 0, 0, 0, 0, 0, 0.9], [0, 0, 0, 0, 0, 0, 0.1]])
    target = np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, 0.0]])
    assert metrics.gripper_accuracy(pred, target) == pytest.approx(1.0)
    assert metrics.gripper_mae(pred, target) == pytest.approx(0.1)

    # a case that disagrees at the threshold boundary
    pred_disagree = np.array([[0, 0, 0, 0, 0, 0, 0.6], [0, 0, 0, 0, 0, 0, 0.1]])
    target_disagree = np.array([[0, 0, 0, 0, 0, 0, 0.4], [0, 0, 0, 0, 0, 0, 0.0]])
    assert metrics.gripper_accuracy(pred_disagree, target_disagree) == pytest.approx(0.5)


def test_action_chunk_smoothness_zero_for_constant_velocity_trajectory():
    # constant-velocity (linear) motion has zero second difference -> perfectly "smooth"
    t = np.arange(5.0)[:, None]
    actions = np.concatenate([t, t, t, t, t, t], axis=-1)  # [5, 6], all channels ramp linearly
    assert metrics.action_chunk_smoothness(actions) == pytest.approx(0.0, abs=1e-9)


def test_action_chunk_smoothness_positive_for_jittery_trajectory():
    actions = np.zeros((5, 6))
    actions[:, 0] = [0.0, 1.0, 0.0, 1.0, 0.0]  # oscillating -> large second differences
    assert metrics.action_chunk_smoothness(actions) > 0.5


def test_action_chunk_smoothness_requires_horizon_at_least_three():
    with pytest.raises(ValueError):
        metrics.action_chunk_smoothness(np.zeros((2, 6)))


def test_wrench_mae_and_peak_error():
    pred = np.array([[1.0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    target = np.zeros((2, 6))
    np.testing.assert_allclose(metrics.wrench_mae(pred, target), [3.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(metrics.wrench_peak_error(pred, target), [5.0, 0, 0, 0, 0, 0])


def test_wrench_exceedance_recall_hand_constructed():
    # target exceeds threshold=1.0 at steps 0 and 2 (channel 0); prediction only catches step 0.
    target = np.array([[2.0], [0.0], [3.0]])
    pred = np.array([[2.0], [0.0], [0.5]])
    assert metrics.wrench_exceedance_recall(pred, target, threshold=1.0) == pytest.approx(0.5)


def test_wrench_exceedance_recall_raises_when_no_events():
    target = np.zeros((3, 1))
    pred = np.zeros((3, 1))
    with pytest.raises(ValueError):
        metrics.wrench_exceedance_recall(pred, target, threshold=1.0)


def test_wrench_constant_baseline_repeats_last_valid_step():
    history = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    history_valid = np.array([True, True, False])  # last step is padding
    baseline = metrics.wrench_constant_baseline(history, history_valid, horizon=4)
    assert baseline.shape == (4, 2)
    np.testing.assert_allclose(baseline, np.tile([2.0, 2.0], (4, 1)))


def test_wrench_constant_baseline_batched():
    history = np.stack(
        [
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[9.0], [8.0], [7.0]]),
        ]
    )  # [2, 3, 1]
    history_valid = np.array([[True, True, True], [True, False, False]])
    baseline = metrics.wrench_constant_baseline(history, history_valid, horizon=2)
    assert baseline.shape == (2, 2, 1)
    np.testing.assert_allclose(baseline[0], [[3.0], [3.0]])
    np.testing.assert_allclose(baseline[1], [[9.0], [9.0]])


def test_wrench_constant_baseline_beaten_by_a_better_predictor_on_a_ramp():
    """Sanity check mirroring the Stage 5 acceptance criterion: a predictor that tracks a wrench
    ramp should have lower MAE than the "hold constant" baseline.
    """
    horizon = 5
    target = np.arange(1, horizon + 1, dtype=np.float64)[:, None]  # ramps 1..5
    history = np.array([[0.0]])
    history_valid = np.array([True])
    baseline = metrics.wrench_constant_baseline(history, history_valid, horizon)
    perfect_pred = target.copy()

    baseline_mae = metrics.wrench_mae(baseline, target)
    predictor_mae = metrics.wrench_mae(perfect_pred, target)
    assert predictor_mae[0] < baseline_mae[0]


def test_compute_action_metrics_and_wrench_metrics_round_trip_to_dict():
    pred_actions = np.random.default_rng(0).normal(size=(2, 5, 7))
    target_actions = np.random.default_rng(1).normal(size=(2, 5, 7))
    action_report = metrics.compute_action_metrics(pred_actions, target_actions)
    action_dict = action_report.to_dict()
    assert set(action_dict) == {
        "channel_mae",
        "channel_rmse",
        "translation_error_mean",
        "rotation_error_mean",
        "gripper_accuracy",
        "gripper_mae",
        "chunk_smoothness",
    }
    assert len(action_dict["channel_mae"]) == 7

    pred_wrench = np.random.default_rng(2).normal(size=(2, 5, 6))
    target_wrench = np.random.default_rng(3).normal(size=(2, 5, 6))
    wrench_report = metrics.compute_wrench_metrics(pred_wrench, target_wrench, exceedance_threshold=0.1)
    wrench_dict = wrench_report.to_dict()
    assert set(wrench_dict) == {"channel_mae", "channel_peak_error", "exceedance_recall"}
    assert len(wrench_dict["channel_mae"]) == 6
    assert wrench_dict["exceedance_recall"] is not None


def test_metrics_raise_on_mismatched_valid_shape():
    pred = np.zeros((2, 3, 6))
    target = np.zeros((2, 3, 6))
    bad_valid = np.zeros((2, 4), dtype=bool)
    with pytest.raises(ValueError):
        metrics.per_channel_mae(pred, target, bad_valid)
