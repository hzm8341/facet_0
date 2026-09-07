"""Paper-level trial bookkeeping without claiming unavailable robot results."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class TrialSummary:
    successes: int
    trials: int
    confidence: float = 0.95

    def __post_init__(self):
        if self.trials <= 0 or not 0 <= self.successes <= self.trials:
            raise ValueError("require 0 <= successes <= trials and trials > 0")
        if self.confidence != 0.95:
            raise ValueError("only the predeclared 95% interval is implemented")

    @property
    def rate(self) -> float:
        return self.successes / self.trials

    def wilson_interval(self) -> tuple[float, float]:
        z = 1.959963984540054
        n = self.trials
        p = self.rate
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        margin = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return center - margin, center + margin
