# Project State

## Last Updated
2026-04-08

## Current Focus
Paper trading validation (Phase 7). Pipeline execution reliability is the blocking issue — 1/60
trading days executed. Signal logic and parameters are validated and frozen.

## Active Branch
`main`

## What Was Just Done
- Added `DecisionSummary` dataclass to `src/pipeline.py` with z-score diagnostics
- Added rejection tracker (`defaultdict`) in `run_daily()`
- Signal-level and risk-level rejection counting instrumented
- Extended summary to include `max_abs_z`, `mean_abs_z`, `z_std_dev`, `signal_strength_ratio`,
  `low_dispersion_flag`, `top_z_scores`, `closest_to_threshold`
- All fields persisted to `logs/decision_log.jsonl`

## Root Cause Found
XLE (|z|=4.06, only above-threshold signal) is blocked by `node_tradeable_mask()` because its
`node_avg_corr` falls below `node_corr_floor=0.20` during REDUCED regime (avg_pairwise_corr=0.28).
The triple-gate (raised threshold × position scale × node block) is over-conservative in REDUCED regime.

## Completed This Session
- ✅ **RCA-1**: `node_tradeable_mask()` now accepts `regime_state`; applies `reduced_node_corr_multiplier=0.75`
  in REDUCED regime. Call-site in `graph_engine.py` updated to pass regime. Config field added to
  `Phase2Config` and `config/phase7.yaml`. 4 new tests passing.
- ✅ **RCA-2**: `_z_pairs` tuples now carry `node_avg_corr`. `DecisionSummary.top_z_scores` and
  `closest_to_threshold` widened to `(asset, z/abs_z, node_avg_corr)`. Print and JSON updated.
- ✅ **RCA-3**: `graph_density` field added to `DecisionSummary`, populated, printed, persisted.

## Next Steps
1. **Fix pipeline execution reliability** — paper trading shows 1/60 expected trading days executed.
   Root cause is pipeline not running daily, not signal logic. This is the Phase 7 gate blocker.
2. **Monitor next `run_daily()` output** — XLE (node_avg_corr ~0.18) should now pass the
   REDUCED-regime effective floor of 0.15 and generate a short trade if z-score holds above
   effective threshold (~3.50).
3. **Do not change any signal parameters** — frozen at Phase 2 cleared values until 90 days of
   clean paper trading data exists.

## Deliberate Decisions — Do Not Revisit Without New Evidence

### REDUCED regime threshold multiplier stays at 1.25× (decided 2026-04-08)
**The gap:** Gameplan §3.3 specifies TRANSITIONING regime → 50% threshold widening. Code uses
`REDUCED_THRESHOLD_MULTIPLIER = 1.25` (25% widening) in `correlation_filter.py`.

**Why we are NOT fixing it:**
1. The 1.25× value is part of the **Phase 2 cleared parameter set** (Sharpe 0.742, cleared 2026-03-17).
   Changing it mid-paper-trading run would invalidate the paper trading as a live validation of that
   cleared system.
2. The gameplan's 1.50× is a suggested starting point, not a validated optimum. The actual
   optimization already ran and landed at 1.25×.
3. The backtest mistake analysis (57 losing trades) shows **72% are REVERSAL_OVERSHOOT** and
   **16% TREND_REVERSAL** — neither is caused by weak-signal entries. A stricter entry threshold
   does not address the actual loss drivers.
4. Only 3.5% of losses are COST_KILLED, and the corrective action for those is already
   per-asset z-score adjustment, not a universal regime multiplier.

**When to revisit:** Only if a full paper trading period produces a new mistake category specifically
tied to weak-signal entries in REDUCED regime. Do not change speculatively.

**Code location:** `REDUCED_THRESHOLD_MULTIPLIER = 1.25` in `src/correlation_filter.py` line 11.

## Key Config Values
- `node_corr_floor`: 0.20 (in `config/phase7.yaml` and `config/phase6.yaml`)
- `corr_floor`: 0.30
- `density_floor`: 0.30 (phase7) / 0.40 (Phase2Config default)
- `signal_threshold`: ~2.80 (derived from signal_strength_ratio)
- REDUCED regime threshold multiplier: 1.25×

## Key Files
- `src/correlation_filter.py` — `node_tradeable_mask()`, `regime_controls()`
- `src/graph_engine.py` — calls `node_tradeable_mask()` at line 92
- `src/config_loader.py` — `Phase2Config` dataclass
- `src/pipeline.py` — `DecisionSummary`, `_log_decision_summary()`, `run_daily()`
- `config/phase7.yaml` — live config values
- `tests/test_correlation_filter.py` — existing node mask tests
