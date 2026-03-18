# Phase 4 Deferred

## Decision

Phase 4 is deferred. The TCN overlay is net negative versus the cleared Phase 2 baseline and should not be used.

## Runtime

- `train-tcn` now pretrains all walk-forward windows and completed in `1902.85s` (`31.7` minutes).
- `run-phase4` now completes in `93.22s`, so the iteration-critical backtest path is fixed.
- The expensive step was confirmed to be walk-forward retraining inside the old `run-phase4`, which has been removed.

## Final Phase 4 Run

- Run ID: `9cd9f337-6a6c-4168-80ff-142097fc219f`
- Sharpe: `-0.3305`
- Annualized return: `-0.58%`
- Max drawdown: `5.99%`
- Profitable active months: `59.09%`
- Calibration rate: `98.86%`
- Calibration error: `30.86` percentage points
- TCN veto rate: `37.93%`
- TCN veto accuracy: `15.63%`
- Predicted/actual residual correlation: `-0.0012`

## Comparison To Phase 2 Baseline

- Phase 2 baseline Sharpe: `0.7423`
- Phase 4 Sharpe: `-0.3305`
- Phase 2 baseline max drawdown: `5.27%`
- Phase 4 max drawdown: `5.99%`

The TCN both reduced returns and slightly worsened drawdown.

## Why It Failed

- The model did not learn a useful predictive signal:
  - predicted vs actual residual correlation was effectively zero;
  - veto accuracy was only `15.63%`, far below the threshold for a useful filter.
- Uncertainty was badly miscalibrated:
  - `98.86%` of realized residuals fell inside the predicted `1-sigma` interval, which indicates severe overestimation of uncertainty.
- The veto layer was too active and mostly wrong:
  - veto rate was `37.93%`;
  - the system removed many trades, but the removed trades were not predominantly losers.

## Outcome

Because `sharpe_ratio <= 0.74`, the Phase 4 gate logic says the TCN is net negative and should be disabled. The project should proceed to Phase 5 with the Phase 2 cleared system as the active baseline.
