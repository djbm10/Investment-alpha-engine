# Phase 2 Clearance Report

## Cleared Run
- Run ID: `b440fc17-3c9c-46a3-840e-d4837b52f9d9`
- Date: `2026-03-17`
- Sharpe: `0.7423`
- Profitable active months: `64.44%` (`29/45`)
- Max drawdown: `5.27%`
- Gate status: `True`

## What Worked
Phase 2 cleared only after combining structural changes rather than continuing to micro-optimize thresholds on the original 11-asset ETF universe.

1. The per-asset audit identified `XLI`, `XLF`, and `XLC` as clear drag assets on the promoted v3 run.
2. Removing those three names raised the local Phase 2 result from a marginal near-pass to a clear pass.
3. The node-level correlation filter (`node_corr_floor = 0.20`) was added so an individual asset can be sidelined even when the universe-level correlation regime still looks healthy.
4. Losing trades were held materially longer than winners, so `max_holding_days` was tightened from `10` to `9`.

## Cleared Parameter Set
The frozen snapshot for Phase 3 handoff is stored in `config/phase2_cleared.yaml`.

- Universe: `XLK`, `XLE`, `XLV`, `XLP`, `XLU`, `XLY`, `XLRE`, `XLB`
- `lookback_window`: `60`
- `diffusion_alpha`: `0.05`
- `diffusion_steps`: `3`
- `sigma_scale`: `1.0`
- `min_weight`: `0.1`
- `zscore_lookback`: `75`
- `signal_threshold`: `2.8`
- `risk_budget_utilization`: `0.3`
- `corr_floor`: `0.30`
- `density_floor`: `0.30`
- `node_corr_floor`: `0.20`
- `max_holding_days`: `9`

## Post-Clear Diagnostics
The cleared run still showed one weak name inside the reduced universe:

- `XLE` was both the only drag asset and the only inert asset on the final run

That does not block Phase 3 because the Phase 2 gate already passed, but it is useful context for future regime-aware portfolio construction.
