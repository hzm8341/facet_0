"""Candidate ranking utilities for experimental value-guided action selection."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class CandidateScores:
    expected_return: np.ndarray
    violation_probability: np.ndarray
    success_probability: np.ndarray


def rank_candidates(
    scores: CandidateScores,
    *,
    violation_penalty: float = 1.0,
    enabled: bool = True,
) -> np.ndarray:
    count = len(scores.expected_return)
    if not enabled:
        return np.arange(count)
    utility = (
        np.asarray(scores.expected_return)
        + np.asarray(scores.success_probability)
        - violation_penalty * np.asarray(scores.violation_probability)
    )
    return np.argsort(-utility, kind="stable")
