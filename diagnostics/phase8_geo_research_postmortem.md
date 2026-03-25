# Phase 8 Geo Research Postmortem

Date: 2026-03-22

Status:
- Research-only
- Frozen
- Not approved for Batch 5
- Not approved for production integration

## Hypothesis 1: Per-Trade Geo Overlay
Result: failed

The direct geo overlay failed the acceptance gate. It reduced net Sharpe after costs, slightly worsened max drawdown, and removed winning trades. The overlay also failed to improve the high-geo windows it was supposed to protect.

Reference:
- [phase8_geo_overlay_postmortem.md](/home/djmann/projects/Investment-alpha-engine/diagnostics/phase8_geo_overlay_postmortem.md)
- [geo_overlay_evaluation.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_overlay_evaluation.json)

## Hypothesis 2: Geo Regime Labeler
Result: behavior differs, but no useful policy improvement

The Stage 1 regime-labeler experiment found that baseline behavior does differ across geo regimes, so there was enough signal to test the idea. But the simple regime-conditioned policies did not help. The best real-label variant was `threshold_only`, and it still had:
- `delta_net_sharpe_after_costs = -0.0589`
- `delta_max_drawdown = +0.0337`
- `placebo_margin = -0.0064`

The Stage 1 artifact therefore ended with:
- `research_signal_present = false`
- `production_gate_pass = false`

Reference:
- [geo_regime_experiment.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_regime_experiment_stage1/geo_regime_experiment.json)

## Decision
No Batch 5 integration. No production integration. Freeze this line of work.

## Reopen Condition
Only reopen if the hypothesis changes materially. Do not reopen for more tuning of the current overlay or current regime-labeler policy variants.
