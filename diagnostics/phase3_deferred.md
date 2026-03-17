# Phase 3 Deferred

## Decision

Phase 3 is deferred. Phase 4 should start from [config/phase2_cleared.yaml](/home/djmann/projects/Investment-alpha-engine/config/phase2_cleared.yaml), not from the TDA overlay.

## Final Attempt

The confirmation-based detector plus softer overlay produced run `4f2f843e-f019-4ebe-8d8b-3507c8f12fdc`.

- Sharpe: `0.7632`
- Annualized return: `3.4311%`
- Max drawdown: `5.2696%`
- Profitable active months: `64.44%`
- Regime hit rate: `25.00%`
- False positive rate: `40.00%`
- Flagged regime days: `5`
- `NEW_REGIME` days: `0`

For comparison, the cleared Phase 2 baseline produced:

- Sharpe: `0.7423`
- Annualized return: `3.2423%`
- Max drawdown: `5.2696%`
- Profitable active months: `64.44%`

## Why Phase 3 Is Deferred

The detector no longer satisfies the Phase 3 objective.

- Crisis detection collapsed from `75%` to `25%` after confirmation logic was added.
- The overlay did not improve drawdown in any meaningful way. Drawdown was effectively unchanged versus Phase 2.
- The performance recovery came from making the TDA layer almost inactive, not from better crisis protection.

In other words, the latest Phase 3 result is better interpreted as "Phase 2 with the TDA layer mostly turned off" than as a successful topological regime detector.

## Why Gate Adjustment Was Rejected

A drawdown-only gate adjustment is not sufficient here.

The latest run does clear `Sharpe >= 0.75`, but relaxing only the drawdown criterion would still leave a detector that misses `3` of the `4` named crisis windows and never enters `NEW_REGIME`. That would certify an overlay that does not actually do the job Phase 3 was meant to do.

## Practical Conclusion

The TDA overlay is optional. On the current 8-asset universe, it does not add enough reliable regime awareness to justify carrying it forward.

The correct move is:

1. Defer TDA for now.
2. Use the cleared Phase 2 system as the Phase 4 baseline.
3. Revisit TDA only after the universe is larger or the topology summary is richer, so the persistence diagrams contain enough structure to support stable regime detection.

## Supporting Artifacts

- Current validation report: [regime_detection_validation.md](/home/djmann/projects/Investment-alpha-engine/diagnostics/regime_detection_validation.md)
- Current false-positive analysis: [false_positive_analysis.md](/home/djmann/projects/Investment-alpha-engine/diagnostics/false_positive_analysis.md)
- Current monthly breakdown: [monthly_breakdown.csv](/home/djmann/projects/Investment-alpha-engine/diagnostics/monthly_breakdown.csv)
