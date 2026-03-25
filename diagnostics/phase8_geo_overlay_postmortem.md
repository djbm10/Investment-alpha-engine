# Phase 8 Geo Overlay Postmortem

Date: 2026-03-22

Status:
- Research-only
- Failed acceptance gate
- Not approved for Batch 5 or live integration

## Summary
The WorldMonitor-style geo overlay branch is frozen. The evaluation harness remains useful and should be kept, but the current overlay hypothesis did not clear the fixed Phase 8 acceptance gate.

## What Was Tried
- Baseline deterministic geo snapshot experiments `A/B/C` in [data/processed/geo_variant_experiments/geo_overlay_variant_comparison.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_variant_experiments/geo_overlay_variant_comparison.json).
- Mapping-focused refinements `C/D/E` in [data/processed/geo_variant_experiments_mapping/geo_overlay_variant_comparison.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_variant_experiments_mapping/geo_overlay_variant_comparison.json).
- Structural emphasis variants `E/F` in [data/processed/geo_variant_experiments_structural/geo_overlay_variant_comparison.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_variant_experiments_structural/geo_overlay_variant_comparison.json).
- Final expanded structural `F` snapshot in [data/processed/geo_variant_experiments_structural_expanded/geo_overlay_variant_comparison.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_variant_experiments_structural_expanded/geo_overlay_variant_comparison.json) and [data/processed/geo_variant_experiments_structural_expanded/geo_overlay_evaluation_f.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_variant_experiments_structural_expanded/geo_overlay_evaluation_f.json).

## What Failed
- The final evaluation report in [data/processed/geo_overlay_evaluation.json](/home/djmann/projects/Investment-alpha-engine/data/processed/geo_overlay_evaluation.json) did not pass acceptance.
- Overall net Sharpe after costs fell from `0.5273` to `0.4394`.
- Max drawdown worsened from `0.05270` to `0.05325`.
- The overlay blocked 2 trades, and both removed trades were winners with average baseline PnL of `+0.0239`.
- Geo stress coverage stayed below the required floor: median coverage on geo days was `0.4140` versus a `0.50` per-asset coverage floor, so `geo_stress.available` remained `False`.
- In high-geo blocks, the overlay did not deliver the intended protection:
  - Net Sharpe after costs moved from `0.1346` to `-0.7340`.
  - Max drawdown worsened slightly from `0.04378` to `0.04405`.

## What Improved But Still Failed
- Later structural profiles reduced the damage versus earlier variants.
- Structural `E` was the closest to viable on headline metrics:
  - Sharpe ratio changed from `0.7423` to `0.7139`.
  - Max drawdown changed from `0.05270` to `0.05279`.
  - Only 1 trade was removed.
- Even so, `E` still failed acceptance. The overlay still did not produce a robust, repeatable improvement in the stress slices that matter, and it still relied on weak geo coverage.
- Expanded structural `F` improved the hypothesis framing, but it did not improve the final gate outcome enough to justify integration.

## Final Reason For Rejection
This branch is rejected because the current WorldMonitor-style geo overlay does not add reliable value as a trade filter. It weakens portfolio-level performance, lacks sufficient geo coverage to make the stress signal trustworthy, and fails to improve the high-geo windows that were supposed to justify the overlay in the first place.

## What Stays
- Keep the harness in `src/geo/` and the related tests. The framework did its job by rejecting a weak integration candidate before Batch 5 or live deployment.
- Keep the evaluation artifacts and experiment outputs for reference.

## Reopen Rule
Do not reopen this branch for better tuning of the same overlay. Reopen only if the hypothesis changes materially, for example:
- A different strategy class
- A different role for geopolitics, such as regime labeling instead of trade filtering
- A new dataset or mapping concept rather than more tuning of the current setup
