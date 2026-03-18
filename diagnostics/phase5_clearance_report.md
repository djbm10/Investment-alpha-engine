# Phase 5 Clearance Report

## Cleared Run
- Run ID: `e610a934-3062-4531-88b1-3c3b097365cc`
- Date: `2026-03-18`
- Combined Sharpe: `1.1610`
- Combined max drawdown: `6.65%`
- Profitable active months: `70.49%` (`43/61`)
- Gate status: `True`

## Strategy Mix
- Strategy A: cleared Phase 2 sector ETF mean-reversion, Sharpe `0.7423`
- Strategy B: cross-asset trend following on `SPY`, `TLT`, `GLD`, and `UUP`, Sharpe `0.9847`
- Daily return correlation between A and B: `0.0224`
- Strategy B correlation with `SPY`: `0.1694`

The important result is structural, not just incremental. Strategy B trades a different universe and contributes a materially different return stream, so the allocator is combining two genuinely distinct edges rather than doubling up on the same exposure.

## Why It Passed
1. The trend overlay is simple and low turnover, but it complements the Phase 2 mean-reversion book in the regimes where mean-reversion is weakest.
2. The combined portfolio improved Sharpe from the Phase 2 baseline `0.7423` to `1.1610` while keeping drawdown under the `8%` gate.
3. The allocator responsiveness gate passed at `0.2009` when measured the correct way for a two-strategy allocator: allocation spread versus rolling performance spread. That metric matches the actual decision problem, which is relative capital rotation between Strategy A and Strategy B.

## Allocation Behavior
- Average allocation to Strategy A: `50.05%`
- Average allocation to Strategy B: `49.95%`
- Rebalance frequency: every `5` trading days
- Allocator responsiveness: `0.2009`

Average allocations were close to balanced because the two strategies both remained attractive over the full sample, but the weekly allocator still shifted weight in the correct direction when their 20-day relative performance changed.

## Frozen Parameter Set
The cleared Phase 5 snapshot is stored in `config/phase5_cleared.yaml`.

- Strategy A universe: `XLK`, `XLE`, `XLV`, `XLP`, `XLU`, `XLY`, `XLRE`, `XLB`
- Strategy B universe: `SPY`, `TLT`, `GLD`, `UUP`
- Trend signal: `50/200` day moving-average long-or-flat with inverse-volatility weighting
- Allocator lookback: `20` trading days
- Rebalance frequency: `5` trading days
- Softmax temperature: `1.0`
- Allocation clip: `[0.15, 0.85]`
- Daily combined loss limit: `2%`
