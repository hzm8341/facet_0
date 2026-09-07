"""Scheduling primitives for the paper's experimental 4:1 action/VQA alternation."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AlternatingUpdateSchedule:
    action_updates: int = 4
    vqa_updates: int = 1

    def __post_init__(self):
        if self.action_updates <= 0 or self.vqa_updates <= 0:
            raise ValueError("update counts must be positive")

    @property
    def cycle_length(self) -> int:
        return self.action_updates + self.vqa_updates

    def branch(self, global_step: int) -> str:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        return "action_wrench" if global_step % self.cycle_length < self.action_updates else "vqa"

    def counts(self, total_steps: int) -> dict[str, int]:
        if total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        return {
            "action_wrench": sum(self.branch(step) == "action_wrench" for step in range(total_steps)),
            "vqa": sum(self.branch(step) == "vqa" for step in range(total_steps)),
        }
