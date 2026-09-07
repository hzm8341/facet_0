# FACET-0 phases 5-10 completion status

Date: 2026-09-05

## Phase 5: public-data offline gate

Status: completed at the explicitly experimental deterministic-control level.

The stochastic wrench flow head trained stably after quantile normalization but did not beat the
constant-wrench baseline. A zero-initialized deterministic residual head was therefore added as a
controlled alternative. It conditions on causal wrench history and the frozen FACET policy action
chunk. Residual scale was selected only on validation and then frozen for test.

| Evaluation | Model MAE | Baseline MAE | Relative improvement |
|---|---:|---:|---:|
| validation, 128 sequences, policy action, scale 0.2 | 1.0379 | 1.0700 | 3.00% |
| test, 264 episodes, policy action, frozen scale 0.2 | 1.0005 | 1.0170 | 1.62% |

The test force-group MAE is 1.8662 vs 1.8985; torque-group MAE is 0.13482 vs 0.13550.
This is a small offline improvement, not evidence of closed-loop robot success.

## Phase 6: VQA

Implemented and tested:

- metadata-derived current subtask, overall task and next-instruction proxy targets;
- structured attention mask preventing action tokens from reading VQA labels;
- exact 4:1 action-wrench/VQA scheduler.

Blocked: public data provides numeric subtask IDs but not authored semantic subtask or
next-instruction text, so faithful VQA training and scoring require external labels.

## Phase 7: critic and value guidance

Implemented and tested:

- categorical return head and four experimental auxiliary heads;
- scalar-to-two-hot projection and expected-return decoding;
- configurable offline reward proxy;
- risk-aware candidate ranking and strict disabled-mode fallback.

Blocked: no failed/intervention/recovery rollout or reward labels are published. Only numerical
and synthetic ranking tests are valid; critic calibration or RL-performance claims are not.

## Phase 8: local adaptation

Implemented and tested:

- four-stream 256-dim projections concatenated into a 1024-dim bottleneck;
- bounded absolute-action actor, twin critics and Polyak averaging;
- TD3 bootstrap target and TD3+BC actor loss.

Blocked: no task-local reward replay or matching ten-demonstration protocol is available.

## Phase 9: runtime safety

Implemented and tested:

- fail-closed behavior for non-finite inputs, watchdog timeout and wrench-limit violations;
- workspace, translation-step, rotation-step and gripper clipping;
- shadow-only configuration.

Blocked: hardware model, coordinate frames, calibrated limits, controller API and safety approval
are absent. No command has been sent to a robot.

## Phase 10: paper evaluation

Implemented: five-task protocol skeleton, required log fields and predeclared 95% Wilson
intervals.

Blocked: matching robot, fixtures, initialization distribution and physical trials are
unavailable.

## Honest completion boundary

All locally testable software/proxy work requested by the plan is present. Full phases 6-10 cannot
be truthfully completed from the released checkpoint and 50-hour public subset alone. The missing
inputs are external data and hardware, not unexecuted local code.

## Final verification

- Ruff: all checks passed across scripts, src and tests.
- Pytest: 65 passed.
- JAX environment smoke test: GPU backend, finite 1024 x 1024 result, success true.
- Frozen FACET action golden regression: passed within the predeclared channel tolerances.
- No robot command was issued and no official checkpoint or public dataset file was modified.
