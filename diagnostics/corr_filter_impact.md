# Correlation Filter Impact

## Summary
The promoted v3 run `f8c4db2d-853f-4c76-9a0a-e37461454bda` improved Sharpe materially versus the honest baseline, but it did not clear the Phase 2 gate.

- Baseline recompute on the old single-tier setup: Sharpe `0.5510`, profitable active months `58.49%` (`31/53`)
- Promoted v3 run: Sharpe `0.7038`, profitable active months `59.62%` (`31/52`)
- Net effect: the strategy gained risk-adjusted performance, but only moved one month closer to the profitable-month gate

## Monthly Distribution Change
The baseline had `22` active losing months. Under the promoted v3 run:

- `1` previously losing month became profitable: `2026-02-01`
- `1` previously losing month became inactive and flat: `2026-03-01`
- `20` previously losing months remained losing

The profitable active month count did not increase above `31`. The improvement came from removing one active month from the denominator rather than converting a broad set of weak months into winners.

## Interpretation
The correlation filter helped by improving trade quality enough to lift Sharpe above `0.70`, but it did not smooth the month-to-month P&L distribution enough to pass the profitability gate.

- The promoted run still had `21` active losing months
- The monthly diagnostic for the promoted run showed `15` losing months with fewer than `5` entered trades
- Only `3` losing months were cost-flipped, so the remaining miss is not mainly a transaction-cost problem

The key result is that the filter improved the strategy by trading somewhat better, not by sitting out a large set of clearly bad months.
