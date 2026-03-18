# Phase 4 Runtime Blocker

## Status
Phase 4 is not cleared. The implementation is currently blocked by runtime before the walk-forward backtest can complete within the required operational budget.

## What Was Verified
- `python3 -m compileall src tests`
- `pytest`
- Phase 4 feature pipeline, TCN model, trainer, and Phase 2 integration all compile and pass tests.

## Runtime Findings
- `python3 -m src.main train-tcn` completed successfully on the corrected mini-batch trainer path.
- Latest measured wall time for `train-tcn`: `577.51` seconds (`9.63` minutes).
- The saved artifacts were:
  - `data/processed/phase4_latest_ensemble.pt`
  - `data/processed/phase4_latest_training_summary.json`
- `python3 -m src.main run-phase4` was started on March 17, 2026 and allowed to run for approximately `2958` seconds (`49.3` minutes).
- The walk-forward run was still active at that point and had not produced final Phase 4 output artifacts:
  - `phase4_predictions.csv`
  - `phase4_daily_results.csv`
  - `phase4_trade_log.csv`
  - `phase4_monthly_results.csv`
  - `phase4_summary.json`

## Changes Made To Reduce Runtime
1. Removed redundant Phase 4 graph-history recomputation:
   - Reused precomputed `graph_matrices` inside `compute_graph_signals`.
   - Reused the same `daily_signals` and `graph_matrices` inside `FeatureBuilder.prepare_graph_engine_state`.
2. Restored the mini-batch TCN training loop:
   - The earlier full-batch trainer path was materially slower in practice.
   - The current trainer uses `DataLoader` batching with the de-duplicated Phase 4 context.

## Conclusion
The current Phase 4 implementation misses the runtime target that the full CPU walk-forward protocol should finish in under 30 minutes. Until that is fixed, Phase 4 remains operationally blocked even before considering Sharpe improvement, calibration, or veto quality.

## Recommended Next Step
Optimize the walk-forward execution itself before rerunning the Phase 4 gate:
- profile per-window runtime to separate graph preparation from model training;
- cache or reuse window-invariant graph features where possible;
- consider a subprocess-based window parallelization approach if the environment allows it;
- only after the full run completes inside the runtime budget should the project resume `run-phase4`, `verify-phase4`, and Phase 4 gate evaluation.
