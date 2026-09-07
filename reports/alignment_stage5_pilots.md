# Stage 5 action-wrench alignment pilot report

Date: 2026-09-05

## Outcome

The frozen FACET-0 action backbone plus independent wrench flow head is implemented and
mechanically stable. Quantile normalization fixed the original raw-scale instability, and a
residual target improved inference, but neither 1000-step full-data pilot beats the causal
"hold current wrench" validation baseline. The 20,000-step run is therefore not approved yet.

The official checkpoint and public dataset were not modified.

## Changes in this iteration

- Added checkpoint-state q01/q99 normalization for wrench channels 7:13.
- Added absolute and delta-from-current wrench target representations.
- Added resolved-config and fixed overfit-pool provenance artifacts.
- Added scripts/evaluate_alignment.py for physical-unit MAE/peak error and baseline comparison.
- Added experimental multi-sample averaging for flow outputs used as point predictions.
- Added v2 normalized and v3 residual overfit/full configurations.

## Training pilots

All losses below are logged every 10 steps (overfit) or 50 steps (full). Because flow noise and
batches are resampled, the gate uses window means rather than the final stochastic point.

| Run | Steps | Target | First 5 wrench-loss mean | Last 5 mean | Minimum |
|---|---:|---|---:|---:|---:|
| overfit v2 | 300 | normalized absolute | 0.952 | 0.272 | 0.103 |
| full v2 pilot | 1000 | normalized absolute | 0.748 | 0.436 | 0.077 |
| overfit v3 | 300 | normalized residual | 0.933 | 0.245 | 0.080 |
| full v3 pilot | 1000 | normalized residual | 0.717 | 0.427 | 0.106 |

The action loss remains stochastic because the official action policy is frozen; it is a
regression monitor, not an optimization target for the wrench-only optimizer.

## Fixed validation comparison

These exploratory numbers use the same 8 validation sequences, seed 0 and 10 Euler steps.
The sample size is intentionally small and is not a final benchmark.

| Head | Flow samples averaged | Mean of 6 channel MAEs | Constant baseline | Pass |
|---|---:|---:|---:|---|
| overfit v2 absolute | 1 | 1.733 | 0.421 | no |
| full v2 pilot absolute | 1 | 1.983 | 0.421 | no |
| overfit v3 residual | 1 | 1.498 | 0.421 | no |
| overfit v3 residual | 8 | 0.786 | 0.421 | no |
| full v3 pilot residual | 8 | 0.968 | 0.421 | no |

Residual prediction and ensemble averaging help, but the acceptance criterion is still not met.
Do not interpret offline MAE as robot success rate.

## Recommended next experiment

Before spending on 20,000 steps:

1. Add a deterministic residual-regression head (Huber/L1) as a control. Its zero output exactly
   matches the strong causal baseline and avoids generative variance in a point-MAE objective.
2. Evaluate on at least 128 fixed validation sequences and report force and torque groups
   separately; the current mean mixes different physical units.
3. Oversample contact transitions or stratify by future wrench change. Uniform frame sampling is
   dominated by near-constant windows and supplies little signal beyond the baseline.
4. If retaining flow matching, calibrate sample count and residual shrinkage on validation, then
   report once on the untouched test split.
5. Only start configs/alignment/full_v3_residual.yaml at 20,000 steps after a short pilot shows a
   validation improvement over the constant baseline.

## Reproduction commands

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
      third_party/openpi/.venv/bin/python -m facet0.training.train_alignment \
      --config configs/alignment/overfit_v3_residual.yaml

    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
      third_party/openpi/.venv/bin/python scripts/evaluate_alignment.py \
      --config configs/alignment/overfit_v3_residual.yaml \
      --head runs/alignment/overfit_v3_residual/wrench_head_final.npz \
      --split validation --num-sequences 8 --batch-size 1 \
      --num-steps 10 --num-samples 8 \
      --output reports/alignment_overfit_v3_validation_ensemble8.json

## Verification

- Final full suite: 65 passed.
- Ruff on all modified Python files: passed.
- Known warnings: 1676 JAX/Flax deprecations from the pinned OpenPI environment.

## Final deterministic control result

The recommended deterministic residual control was subsequently implemented. It uses a
zero-initialized output (exact constant-baseline fallback), causal wrench history, and an action
chunk condition. A 2,048-sequence pool was selected from 4,096 candidates with 50% high-change
stratification.

| Action condition | Residual scale | MAE | Baseline | Result |
|---|---:|---:|---:|---|
| demonstration oracle | 1.0 | 0.964 | 1.070 | pass, oracle only |
| frozen FACET policy | 1.0 | 1.228 | 1.070 | fail |
| frozen FACET policy | 0.2 selected on validation | 1.038 | 1.070 | pass |

After freezing scale 0.2, the one-time 264-episode test evaluation produced MAE 1.0005 versus
baseline 1.0170 (1.62% relative improvement). Four of six channels improved; grouped force and
torque MAEs both improved slightly. The margin is small and is a public-subset offline result.

Primary artifacts:

- reports/deterministic_full_pilot_validation_128_policy_calibration.json
- reports/deterministic_full_pilot_test_264_policy_scale02.json
- runs/alignment/deterministic_full_pilot/wrench_head_final.npz
