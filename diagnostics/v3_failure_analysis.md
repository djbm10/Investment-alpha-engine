# V3 Failure Analysis

## Gate Result
The promoted v3 candidate failed the Phase 2 gate by a narrow margin.

- Run ID: `f8c4db2d-853f-4c76-9a0a-e37461454bda`
- Sharpe: `0.7038`
- Max drawdown: `5.66%`
- Profitable active months: `59.62%`
- Gate status: `False`

The only missed threshold is the profitable-month requirement. The run is exactly one active profitable month short of the `60%` gate.

## Regime Distribution
Daily regime counts from the promoted run:

- `TRADEABLE`: `1446` days
- `REDUCED`: `52` days
- `NO_TRADE`: `0` days

This shows the filter almost never suppressed trading outright. It only reduced sizing on about `3.5%` of trading days and never created a true sit-out regime.

## Tradeable-Day Signal Check
Sharpe computed using only `TRADEABLE` days was `0.5007`.

That means the promoted configuration does not fail because there are too few tradeable days. The strategy remains active almost all the time. The issue is that the current regime filter is too weak to materially reshape the monthly P&L distribution.

## Root Cause
The failure is not primarily insufficient tradeable days.

- There were `0` `NO_TRADE` days
- The graph-density metric was effectively saturated on this universe; the promoted sweep reported mean graph density `1.0`
- The density floor therefore did not bind in practice
- The filter improved Sharpe, but mostly through modest sizing changes, not by avoiding broad sets of bad months

The remaining blocker is month-level consistency. The strategy is profitable enough on average, but not consistent enough month to month on the current sector-ETF universe.

## Recommendation
Do not proceed to Phase 3.

The next Phase 2 step should be one of:

- Replace the current universe with a narrower, more correlated sub-universe so the graph structure has genuine peer-group meaning and the density filter can actually bind
- Redefine the density/sparsification rule so it can produce real `NO_TRADE` periods on this universe before running another sweep

The current sector-ETF universe leaves the graph nearly fully connected, so the correlation regime filter does not have enough leverage to fix the monthly consistency problem by itself.
