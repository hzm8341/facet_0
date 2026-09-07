import numpy as np
import pytest

from facet0.data.vqa_targets import build_vqa_targets, structured_attention_mask
from facet0.training.train_multitask import AlternatingUpdateSchedule


def test_vqa_targets_are_traceable_to_public_metadata():
    targets = build_vqa_targets(task="install RAM", subtask=2, next_subtask=3)
    assert targets.overall_task == "install RAM"
    assert targets.current_subtask == "subtask 2"
    assert targets.next_instruction == "subtask 3"


def test_structured_mask_prevents_action_label_leakage():
    mask = structured_attention_mask(3, 2, 3)
    assert mask.shape == (8, 8)
    assert not mask[3:5, 5:].any()
    np.testing.assert_array_equal(mask[5:, 5:], np.tril(np.ones((3, 3), dtype=bool)))


def test_four_to_one_schedule():
    schedule = AlternatingUpdateSchedule()
    assert [schedule.branch(i) for i in range(5)] == ["action_wrench"] * 4 + ["vqa"]
    assert schedule.counts(10) == {"action_wrench": 8, "vqa": 2}
    with pytest.raises(ValueError):
        schedule.branch(-1)
