"""Experimental VQA supervision and structured masks derived from public metadata."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class VQATargets:
    current_subtask: str
    overall_task: str
    next_instruction: str


def build_vqa_targets(
    *,
    task: str,
    subtask: int,
    next_subtask: int | None,
) -> VQATargets:
    """Build traceable proxy labels; public data exposes IDs, not authored subtask text."""
    current = f"subtask {subtask}"
    next_instruction = "episode complete" if next_subtask is None else f"subtask {next_subtask}"
    return VQATargets(
        current_subtask=current,
        overall_task=str(task),
        next_instruction=next_instruction,
    )


def structured_attention_mask(
    context_tokens: int,
    action_tokens: int,
    vqa_tokens: int,
) -> np.ndarray:
    """Mask where action tokens cannot read VQA labels and VQA labels are causal."""
    if min(context_tokens, action_tokens, vqa_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    total = context_tokens + action_tokens + vqa_tokens
    mask = np.zeros((total, total), dtype=bool)
    context_end = context_tokens
    action_end = context_end + action_tokens
    mask[:context_end, :context_end] = True
    mask[context_end:action_end, :action_end] = True
    if vqa_tokens:
        mask[action_end:, :action_end] = True
        mask[action_end:, action_end:] = np.tril(np.ones((vqa_tokens, vqa_tokens), dtype=bool))
    return mask
