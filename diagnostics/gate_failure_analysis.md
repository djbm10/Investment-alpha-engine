# Phase 2 Gate Failure Analysis

Date: 2026-03-17

Status: Phase 2 remains blocked. Do not proceed to Phase 3.

## Current Failed Candidate

Run ID: `2536e629-3794-47bd-bb63-27a304c6be9c`

- Sharpe ratio: `-0.2635`
- Profitable out-of-sample months: `45.90%`
- Max drawdown: `9.60%`
- Profit factor: `0.8813`
- Annual turnover: `4105.76%`
- Win rate: `52.81%`

The constrained v2 sweep evaluated 27 candidates and produced 0 gate passes. All 27 candidates had negative Sharpe ratios.

## Universe Correlation Check

Average pairwise correlation for the current 11-sector ETF universe:

- Latest 60-day window: `0.2538`
- Rolling 60-day mean across the full history: `0.5190`
- Rolling 60-day median across the full history: `0.4826`
- Rolling 60-day min / max: `0.2252` / `0.8937`

For reference:

- Latest 40-day window: `0.2686`
- Rolling 40-day mean across the full history: `0.5139`

Interpretation:

- The universe is not structurally uncorrelated across the full history.
- The current regime is low-correlation. The latest 60-day and 40-day average pairwise correlations are both below `0.30`.
- That means the full 11-ETF universe is currently giving the graph engine a weak peer-consensus structure exactly when the gate needs to be passed.

## 40-Day Lookback Check

Using the same promoted v2 candidate parameters except for the graph lookback:

### 60-Day Lookback

- Sharpe ratio: `-0.2635`
- Profitable months: `45.90%`
- Max drawdown: `9.60%`
- Profit factor: `0.8813`
- Annual turnover: `4105.76%`

### 40-Day Lookback

- Sharpe ratio: `-0.3543`
- Profitable months: `42.62%`
- Max drawdown: `9.60%`
- Profit factor: `0.8715`
- Annual turnover: `3499.56%`

Interpretation:

- Shortening the correlation lookback from 60 to 40 days did not help.
- The 40-day variant reduced turnover somewhat, but it worsened Sharpe, profitable-month fraction, and profit factor.
- Based on this check alone, the failure is not explained by a stale 60-day graph window.

## Additional Finding: Tier-2 Entries Regressed Signal Quality

Using the same promoted parameter set but effectively disabling the new moderate-signal tier (`tier2_fraction = 1.0`):

- Sharpe ratio: `0.5510`
- Profitable months: `50.82%`
- Max drawdown: `9.11%`
- Profit factor: `1.2606`
- Annual turnover: `2994.38%`

Interpretation:

- The tiered-entry change materially degraded gross signal quality.
- The added moderate-signal trades increased turnover and weakened profit factor instead of improving month-to-month consistency.
- The current blocker is therefore not just insufficient participation; it is that the additional tier is low quality in this universe.

## Monthly P&L Pattern

For run `2536e629-3794-47bd-bb63-27a304c6be9c`:

- `33` of `61` out-of-sample months were unprofitable.
- Only `3` losing months had fewer than 5 entered trades.
- `5` losing months were cost-flipped from positive gross to non-positive net.

Interpretation:

- The majority of losing months are not explained by sparse trading alone.
- Transaction costs matter, but they are not the dominant failure mode.
- Gross signal quality deteriorated after the adaptive-sizing plus tiered-entry changes, especially the tiered-entry expansion.

## Recommended Next Steps

1. Revert the tier-2 entry system as the active default. Keep the implementation available, but do not use it in the live Phase 2 baseline until it shows positive edge in a narrower universe.
2. Keep the 60-day graph lookback as the default. The 40-day check was worse on every important metric except turnover.
3. Test a more correlated sub-universe before any further Phase 2 parameter work. The latest average pairwise correlation of the full ETF universe is below `0.30`, which is too weak for a stable peer-consensus signal right now.
4. Start with a narrower, more internally correlated ETF cluster. A reasonable next Phase 2 experiment is a tech/communications-heavy subset such as `XLK`, `XLC`, and `XLY`, then compare its rolling pairwise correlation and out-of-sample Sharpe against the full 11-ETF set.
5. Only revisit higher risk-budget utilization after the gross edge is restored. Increasing size on the current v2 candidate would mostly magnify a negative signal.

## Decision

Phase 2 gate failed. The repo should remain in Phase 2. Do not proceed to Phase 3.
