# Phase 3 Failure Analysis

## Outcome

Phase 3 remains blocked.

The best TDA-overlay candidate was rerun with [config/phase3.yaml](/home/djmann/projects/Investment-alpha-engine/config/phase3.yaml) and produced run `c2f9f60e-b9c5-4068-8aa9-a242e7a81bac`.

## Gate Results

Phase 3 gate criteria vs. best result:

1. Historical crisis hit rate >= 70%: `PASS`
   - Result: `75.00%` (`3/4` crisis windows flagged)
   - Missed window: COVID crash (`2020-02-19` to `2020-03-23`)
2. Max drawdown reduced by at least 20% relative without annualized return dropping by more than 10%: `FAIL`
   - Phase 2 baseline max drawdown: `5.27%`
   - Required max drawdown: `< 4.22%`
   - Phase 3 max drawdown: `5.75%`
   - Relative drawdown change: `-9.11%` reduction, which means drawdown got worse
   - Phase 2 baseline annualized return: `3.24%`
   - Phase 3 annualized return: `3.47%`
   - Annualized return ratio: `107.09%`
3. Combined system Sharpe > 0.80 after costs: `FAIL`
   - Phase 2 baseline Sharpe: `0.7423`
   - Phase 3 best Sharpe: `0.7045`
   - Shortfall to gate: `0.0955`

## What Failed

The main issue is integration, not basic crisis detection.

- The detector flagged `44` `TRANSITIONING` days and `54` `NEW_REGIME` days, for `98` flagged days total.
- False positive rate was `86.73%`, which means most flagged days were outside the known crisis windows even after the allowed proximity buffer.
- The detector did catch the 2022 rate shock, the SVB crisis, and the August 2024 volatility spike.
- The detector missed the COVID crash entirely, which is the most important stress event in the sample.

This means the overlay is active often enough to disrupt the strategy, but not accurate enough to protect capital when it matters most.

## Why The Overlay Hurt Performance

The current overlay logic is directionally sensible but operationally noisy:

- `TRANSITIONING` reduces position size, raises entry thresholds, and shortens the effective lookback.
- `NEW_REGIME` freezes new entries and can trigger emergency recalibration.

That combination suppressed good trades often enough to lower Sharpe from `0.7423` to `0.7045`, while still failing to prevent the worst drawdown expansion from `5.27%` to `5.75%`.

The net result is a regime layer that adds friction without adding enough protection.

## Detection Notes

The first Phase 3 iteration used Wasserstein distance on equal-weight H0/H1 persistence diagrams. That was sufficient to clear the basic crisis-hit gate but not sufficient to produce a usable trading overlay.

If Phase 3 work continues, the next detection-level questions are:

- whether H1 carries most of the useful signal while H0 adds noise
- whether bottleneck distance is more stable than Wasserstein distance here
- whether persistence entropy or landscape norms are better daily summaries than raw diagram distances

Those are valid next experiments, but they are not the immediate bottleneck. The immediate bottleneck is the overlay's false-positive rate and the fact that flagged regimes do not improve drawdown.

## Recommendation

Keep Phase 2 as the live baseline and keep the project blocked in Phase 3.

The next Phase 3 iteration should focus on:

1. Reducing false positives before changing any more trading behavior.
2. Separating detector evaluation from overlay evaluation so the detector can be improved without immediately perturbing portfolio logic.
3. Testing whether a stricter `NEW_REGIME` definition or a persistence feature that better isolates structural breaks can improve drawdown without suppressing too many profitable trades.

Do not proceed to Phase 4 until the Phase 3 overlay can improve drawdown materially while preserving or improving the cleared Phase 2 Sharpe profile.
