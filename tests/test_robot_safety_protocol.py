import numpy as np

from facet0.evaluation.protocol import TrialSummary
from facet0.robot.safety import SafetyConfig, filter_command


def _config():
    return SafetyConfig(
        workspace_low=(0.0, -1.0, 0.0),
        workspace_high=(1.0, 1.0, 1.0),
        max_translation_step=0.1,
        max_rotation_step=0.2,
        wrench_abs_limit=(10, 10, 10, 1, 1, 1),
    )


def test_safety_filter_clips_motion_workspace_and_gripper():
    current = np.zeros(7)
    proposed = np.asarray([2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0])
    result = filter_command(current, proposed, np.zeros(6), command_age_seconds=0.0, config=_config())
    assert not result.stopped and result.clipped
    assert np.linalg.norm(result.command[:3]) <= 0.1 + 1e-9
    assert np.linalg.norm(result.command[3:6]) <= 0.2 + 1e-9
    assert result.command[6] == 1.0


def test_wrench_timeout_and_nonfinite_fail_closed():
    current = np.zeros(7)
    assert filter_command(current, current, np.asarray([11, 0, 0, 0, 0, 0]), command_age_seconds=0, config=_config()).stopped
    assert filter_command(current, current, np.zeros(6), command_age_seconds=1, config=_config()).stopped
    bad = current.copy()
    bad[0] = np.nan
    assert filter_command(current, bad, np.zeros(6), command_age_seconds=0, config=_config()).stopped


def test_wilson_interval_contains_observed_rate():
    summary = TrialSummary(successes=16, trials=20)
    low, high = summary.wilson_interval()
    assert low < summary.rate < high
    assert 0 <= low <= high <= 1
